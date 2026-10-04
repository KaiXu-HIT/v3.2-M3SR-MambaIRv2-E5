"""E5: frozen E4 depth interventions with paired images and Gumbel seeds.

All eight conditions reuse one strictly loaded E4 model. Only the already
normalized LR Depth tensor changes. The RGB route is reset to the same seed
before every forward and audited for equality against D0.
"""
import argparse
import csv
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CONDITIONS = ('D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7')
NAMES = dict(D0='Correct Depth', D1='Zero Depth',
             D2='Constant Depth 0.5', D3='Spatial Shuffle',
             D4='Other-image Depth', D5='Gaussian Noise',
             D6='Blurred Depth', D7='Depth Edge Only')
DATASETS = ('Set5', 'Set14', 'B100', 'Urban100', 'Manga109')
MECHANISM = ('confidence_mean', 'local_alpha_mean', 'gate_mean',
             'correction_rms')


def corruption_generator(key, seed):
    # Local CPU RNG makes D3/D5 reproducible and independent of the model's
    # hard-Gumbel stream. Corruptions stay fixed across matched model seeds.
    digest = hashlib.sha256(f'{seed}:{key}'.encode('utf-8')).digest()
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int.from_bytes(digest[:8], 'little') % (2**63))
    return generator


def gaussian_blur(depth, sigma=5.0):
    """Strong LR-space Gaussian blur with replicate borders, no renormalization."""
    radius = int(math.ceil(3 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=depth.dtype)
    kernel = torch.exp(-x.square() / (2 * sigma * sigma))
    kernel /= kernel.sum()
    square = torch.outer(kernel, kernel)[None, None]
    return F.conv2d(F.pad(depth, (radius,) * 4, mode='replicate'), square)


def sobel_magnitude(depth):
    """Use E4's exact fixed Sobel kernel; D7 contains edges and no raw D."""
    k = depth.new_tensor([[-1., 0., 1.], [-2., 0., 2.],
                          [-1., 0., 1.]]) / 8
    kernels = torch.stack((k, k.t()))[:, None]
    grad = F.conv2d(F.pad(depth, (1, 1, 1, 1), mode='replicate'), kernels)
    return torch.linalg.vector_norm(grad, dim=1, keepdim=True)


def depth_variants(depth, other, image_key, corruption_seed):
    """Construct D0-D7 before inference, preserving source Depth preprocessing."""
    if depth.ndim != 4 or depth.shape[:2] != (1, 1):
        raise ValueError('E5 expects one normalized LR Depth map, NCHW=1x1xHxW.')
    if other.shape[:2] != (1, 1):
        raise ValueError('D4 must use a different paired image Depth map.')
    if not torch.isfinite(depth).all() or depth.min() < 0 or depth.max() > 1:
        raise ValueError('Source Depth must be the dataset-normalized [0,1] map.')
    h, w = depth.shape[-2:]
    flat = depth.reshape(-1)
    shuffle = flat[torch.randperm(flat.numel(),
                                  generator=corruption_generator(image_key + ':shuffle', corruption_seed))]
    random = torch.randn(depth.shape,
                         generator=corruption_generator(image_key + ':noise', corruption_seed),
                         dtype=depth.dtype)
    # D5 has exactly the per-image mean and population std of D0. We leave
    # possible values outside [0,1] unclipped to avoid changing these moments.
    target_std = depth.std(unbiased=False)
    random = (random - random.mean()) / random.std(unbiased=False).clamp_min(1e-12)
    noise = random * target_std + depth.mean()
    # A deterministic non-self paired image from this same test set. Resize
    # only for differing LR shapes, and record that provenance in each row.
    other_resized = F.interpolate(other, size=(h, w), mode='bilinear',
                                  align_corners=False) if other.shape[-2:] != (h, w) else other.clone()
    result = dict(D0=depth.clone(), D1=torch.zeros_like(depth),
                  D2=torch.full_like(depth, .5), D3=shuffle.reshape_as(depth),
                  D4=other_resized, D5=noise, D6=gaussian_blur(depth),
                  D7=sobel_magnitude(depth))
    if any(value.shape != depth.shape or not torch.isfinite(value).all()
           for value in result.values()):
        raise RuntimeError('A Depth intervention changed geometry or produced nonfinite values.')
    return result


def seed_model(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def factor_statistics(maps):
    required = ('confidence', 'local_alpha', 'gate',
                'correction_square_mean')
    if any(key not in maps for key in required):
        raise RuntimeError('E4 factor hooks did not capture all four quantities.')
    return dict(confidence_mean=float(np.mean(maps['confidence'])),
                local_alpha_mean=float(np.mean(maps['local_alpha'])),
                gate_mean=float(np.mean(maps['gate'])),
                correction_rms=float(np.sqrt(np.mean(maps['correction_square_mean']))))


def paired_summary(rows, seeds, dataset_names):
    """Average images per seed, then compare the same images and seeds."""
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError('E5 requires at least three distinct matched seeds.')
    index = {}
    for row in rows:
        key = row['dataset'], row['image_id'], row['seed'], row['condition']
        if key in index:
            raise ValueError(f'Duplicate E5 result: {key}')
        index[key] = row
    summary = {}
    for dataset in dataset_names:
        images = sorted({row['image_id'] for row in rows if row['dataset'] == dataset})
        if not images:
            raise ValueError(f'No E5 test images in {dataset}.')
        summary[dataset] = {}
        for condition in CONDITIONS:
            seed_means = {field: [] for field in ('psnr', 'ssim') + MECHANISM}
            paired = []
            for seed in seeds:
                samples = [index[(dataset, image, seed, condition)] for image in images]
                correct = [index[(dataset, image, seed, 'D0')] for image in images]
                for field in seed_means:
                    seed_means[field].append(statistics.mean(x[field] for x in samples))
                paired.append(statistics.mean(c['psnr'] - x['psnr']
                                              for c, x in zip(correct, samples)))
            summary[dataset][condition] = dict(
                images=len(images),
                metrics={field: dict(mean=statistics.mean(values),
                                     std=statistics.stdev(values),
                                     by_seed=values)
                         for field, values in seed_means.items()},
                correct_minus_condition_psnr=dict(mean=statistics.mean(paired),
                                                  std=statistics.stdev(paired),
                                                  by_seed=paired))
    five_set = {}
    for condition in CONDITIONS:
        by_seed = [statistics.mean(summary[dataset][condition]['metrics']['psnr']['by_seed'][i]
                                   for dataset in dataset_names)
                   for i in range(len(seeds))]
        delta = [statistics.mean(summary[dataset][condition]
                                 ['correct_minus_condition_psnr']['by_seed'][i]
                                 for dataset in dataset_names)
                 for i in range(len(seeds))]
        five_set[condition] = dict(psnr_mean=statistics.mean(by_seed),
                                   psnr_std=statistics.stdev(by_seed),
                                   correct_minus_condition_psnr=statistics.mean(delta),
                                   paired_delta_by_seed=delta)
    correct = five_set['D0']['psnr_mean']
    checks = dict(correct_gt_zero=correct > five_set['D1']['psnr_mean'],
                  correct_gt_shuffle=correct > five_set['D3']['psnr_mean'],
                  correct_gt_shuffle_all_seeds=all(
                      delta > 0 for delta in five_set['D3']['paired_delta_by_seed']),
                  minimum_ordering=(correct > five_set['D1']['psnr_mean'] and
                                    correct > five_set['D3']['psnr_mean']),
                  shuffle_no_go=correct <= five_set['D3']['psnr_mean'],
                  shuffle_near_tie_001db=(
                      abs(five_set['D3']['correct_minus_condition_psnr']) <= .01),
                  ideal_ordering=(correct > five_set['D7']['psnr_mean'] >
                                  five_set['D6']['psnr_mean'] > five_set['D1']['psnr_mean']))
    shifts = {condition: {
        field: statistics.mean(summary[dataset][condition]['metrics'][field]['mean'] -
                               summary[dataset]['D0']['metrics'][field]['mean']
                               for dataset in dataset_names)
        for field in MECHANISM} for condition in CONDITIONS[1:]}
    return dict(seeds=seeds, datasets=list(dataset_names), conditions=NAMES,
                per_dataset=summary, five_set=five_set, criteria=checks,
                mechanism_shift_vs_correct=shifts,
                note='All comparisons use one frozen E4 checkpoint, same images and paired Gumbel seeds. Positive paired delta means correct Depth scores higher.')


def write_report(output, rows, report, provenance):
    output.mkdir(parents=True, exist_ok=True)
    with (output/'E5_per_image.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report.update(provenance=provenance, images_per_dataset={
        dataset: report['per_dataset'][dataset]['D0']['images']
        for dataset in report['datasets']})
    (output/'E5_causality.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# E5 Depth causality: frozen E4', '',
             f"Seeds: {report['seeds']}; model SHA256: {provenance['checkpoint_sha256']}",
             'Positive Δ means correct Depth scores higher. PSNR is uint8 Y-channel x4 crop 4.', '',
             '| Condition | 5-set PSNR mean±seed std | Correct − condition (dB) |',
             '|---|---:|---:|']
    for condition in CONDITIONS:
        item = report['five_set'][condition]
        lines.append(f"| {condition} {NAMES[condition]} | {item['psnr_mean']:.4f}±{item['psnr_std']:.4f} | "
                     f"{item['correct_minus_condition_psnr']:+.4f} |")
    lines.extend(['', '| Dataset | ' + ' | '.join(CONDITIONS) + ' |',
                  '|---|' + '|'.join(['---:'] * len(CONDITIONS)) + '|'])
    for dataset in report['datasets']:
        values = [report['per_dataset'][dataset][condition]
                  ['correct_minus_condition_psnr']['mean'] for condition in CONDITIONS]
        lines.append('| ' + dataset + ' | ' + ' | '.join(f'{value:+.4f}' for value in values) + ' |')
    lines.extend(['', '## Mechanism shifts versus D0', '',
                  '| Condition | confidence | local alpha | gate | correction RMS |',
                  '|---|---:|---:|---:|---:|'])
    for condition, shift in report['mechanism_shift_vs_correct'].items():
        lines.append('| ' + condition + ' | ' + ' | '.join(
            f'{shift[field]:+.6f}' for field in MECHANISM) + ' |')
    if not report['complete_five_set']:
        lines += ['', '**Partial smoke run: do not interpret these values as full E5 results.**']
    lines += ['', 'Minimum ordering: ' + str(report['criteria']['minimum_ordering']),
              'Correct versus shuffle no-go: ' + str(report['criteria']['shuffle_no_go']),
              'Correct versus shuffle within ±0.01 dB (review): ' +
              str(report['criteria']['shuffle_near_tie_001db']),
              'Inspect per-image paired deltas, seed spread and factor maps before causal claims.', '']
    (output/'E5_summary.md').write_text('\n'.join(lines), encoding='utf-8')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run(args):
    from basicsr.archs.e4_udrv2_arch import E4UDRMambaIRv2
    from basicsr.data import build_dataset
    from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
    from basicsr.utils.img_util import tensor2img
    from scripts.udr.e0_mechanism_audit import infer_partitioned, load_config, load_weights

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('E5 production inference needs CUDA Mamba extensions.')
    if len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Use at least three distinct matched seeds.')
    e4_config = Path(args.e4_config).resolve()
    checkpoint = Path(args.e4_checkpoint).resolve()
    e4_cfg, rgb_cfg = load_config(e4_config), load_config(args.rgb_config)
    if e4_cfg['network_g']['type'] != 'E4UDRMambaIRv2':
        raise ValueError('E5 must test the selected E4 architecture.')
    if e4_cfg['scale'] != 4 or e4_cfg['val']['metrics'] != rgb_cfg['val']['metrics']:
        raise ValueError('E5 must preserve original x4 Y-channel/crop-4 metrics.')
    names = [e4_cfg['datasets'][key]['name'] for key in sorted(e4_cfg['datasets'])]
    if tuple(names) != DATASETS or set(e4_cfg['datasets']) != set(rgb_cfg['datasets']):
        raise ValueError('E5 needs the exact original five test datasets.')
    for key in e4_cfg['datasets']:
        a, b = e4_cfg['datasets'][key], rgb_cfg['datasets'][key]
        for field in ('name', 'dataroot_gt', 'dataroot_lq', 'filename_tmpl'):
            if a[field] != b[field]:
                raise ValueError(f'E5 test dataset differs from RGB reference: {key}/{field}')
    options = deepcopy(e4_cfg['network_g'])
    options.pop('type')
    model = E4UDRMambaIRv2(**options)
    load_weights(model, checkpoint)  # Full strict E4 weights; never optimized.
    model = model.to(args.device).eval()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    metric_kwargs = dict(crop_border=4, test_y_channel=True)
    rows = []
    for key in sorted(e4_cfg['datasets']):
        dsopt = deepcopy(e4_cfg['datasets'][key])
        dsopt.update(phase='test', scale=4)
        dataset = build_dataset(dsopt)
        dataset.paths.sort(key=lambda item: item['lq_path'])
        if len(dataset) < 2:
            raise ValueError('D4 requires at least two distinct test images per dataset.')
        count = min(len(dataset), args.max_images) if args.max_images else len(dataset)
        for index in range(count):
            sample = dataset[index]
            partner = dataset[(index + 1) % len(dataset)]
            image_id = Path(sample['gt_path']).stem
            partner_id = Path(partner['gt_path']).stem
            if image_id == partner_id:
                raise RuntimeError('D4 accidentally selected the same image.')
            depth = sample['depth'].unsqueeze(0)
            variants = depth_variants(depth, partner['depth'].unsqueeze(0),
                                      f'{dsopt["name"]}/{image_id}', args.corruption_seed)
            rgb = sample['lq'].unsqueeze(0).to(args.device)
            gt_image = tensor2img(sample['gt'].unsqueeze(0))
            for seed in args.seeds:
                route_reference = None
                first_row = len(rows)
                for condition in CONDITIONS:
                    seed_model(seed)  # Match the RGB Gumbel route for all D0-D7.
                    prediction, maps, _ = infer_partitioned(
                        model, rgb, variants[condition].to(args.device), audit=True)
                    routes = tuple(maps[name] for name in
                                   ('route_entropy', 'route_maxprob',
                                    'route_margin', 'ambiguity'))
                    if route_reference is None:
                        route_reference = routes
                    else:
                        for expected, actual in zip(route_reference, routes):
                            np.testing.assert_allclose(expected, actual, rtol=0, atol=1e-6,
                                                       err_msg='Depth intervention changed matched RGB routing')
                    sr_image = tensor2img(prediction)
                    values = variants[condition]
                    rows.append(dict(dataset=dsopt['name'], image_id=image_id,
                                     seed=seed, condition=condition,
                                     condition_name=NAMES[condition],
                                     other_image_id=partner_id if condition == 'D4' else '',
                                     other_resized=(partner['depth'].shape != sample['depth'].shape)
                                     if condition == 'D4' else False,
                                     depth_mean=float(values.mean()),
                                     depth_std=float(values.std(unbiased=False)),
                                     psnr=float(calculate_psnr(sr_image, gt_image, **metric_kwargs)),
                                     ssim=float(calculate_ssim(sr_image, gt_image, **metric_kwargs)),
                                     **factor_statistics(maps)))
                correct = rows[first_row]
                for row in rows[first_row:]:
                    row['paired_delta_psnr'] = correct['psnr'] - row['psnr']
                    row['paired_delta_ssim'] = correct['ssim'] - row['ssim']
            print(f'E5 {dsopt["name"]} {index + 1}/{count}: {image_id}', flush=True)
    report = paired_summary(rows, args.seeds, DATASETS)
    report['complete_five_set'] = args.max_images == 0
    report['corruption'] = dict(seed=args.corruption_seed,
                                spatial_shuffle='full LR pixel permutation',
                                other_image='next sorted paired image in same dataset; bilinear resize only if needed',
                                gaussian_noise='exact D0 per-image mean and population std; unclipped',
                                blur='Gaussian sigma=5 LR pixels, radius=15, replicate border',
                                edge='E4 fixed Sobel magnitude, no original Depth intensity')
    provenance = dict(e4_config=str(e4_config), e4_checkpoint=str(checkpoint),
                      checkpoint_sha256=file_sha256(checkpoint),
                      e4_uncertainty_mode=model.uncertainty_mode)
    write_report(Path(args.output).resolve(), rows, report, provenance)
    print('Saved E5 causal intervention report to', Path(args.output).resolve())


def self_test():
    depth = torch.arange(100, dtype=torch.float32).reshape(1, 1, 10, 10) / 99
    other = torch.flip(depth, dims=(-1,))
    variants = depth_variants(depth, other, 'Set5/image1', 2026)
    assert tuple(variants) == CONDITIONS
    assert torch.equal(variants['D0'], depth)
    assert torch.count_nonzero(variants['D1']) == 0
    assert torch.all(variants['D2'] == .5)
    assert torch.equal(variants['D3'].flatten().sort().values,
                       depth.flatten().sort().values)
    assert not torch.equal(variants['D3'], depth)
    assert torch.equal(variants['D4'], other)
    assert abs(variants['D5'].mean().item() - depth.mean().item()) < 1e-6
    assert abs(variants['D5'].std(unbiased=False).item() - depth.std(unbiased=False).item()) < 1e-6
    assert torch.equal(depth_variants(depth, other, 'Set5/image1', 2026)['D5'], variants['D5'])
    assert variants['D6'].std() < depth.std()
    assert torch.allclose(variants['D7'], sobel_magnitude(depth))
    rows = []
    for dataset in DATASETS:
        for image in ('a', 'b'):
            for seed in (10, 11, 12):
                for idx, condition in enumerate(CONDITIONS):
                    rows.append(dict(dataset=dataset, image_id=image, seed=seed,
                                     condition=condition, psnr=30 - idx * .02,
                                     ssim=.9, confidence_mean=.5 - idx * .01,
                                     local_alpha_mean=.02 - idx * .001,
                                     gate_mean=.3 - idx * .01,
                                     correction_rms=.01 - idx * .001))
    report = paired_summary(rows, [10, 11, 12], DATASETS)
    assert report['criteria']['minimum_ordering']
    assert abs(report['five_set']['D3']['correct_minus_condition_psnr'] - .06) < 1e-10
    report['complete_five_set'] = True
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp)
        write_report(output, rows, report,
                     dict(checkpoint_sha256='synthetic', e4_checkpoint='synthetic'))
        assert (output/'E5_per_image.csv').is_file()
        assert (output/'E5_summary.md').is_file()
        saved = json.loads((output/'E5_causality.json').read_text(encoding='utf-8'))
        assert tuple(saved['five_set']) == CONDITIONS
    try:
        paired_summary(rows[:-1], [10, 11, 12], DATASETS)
    except KeyError:
        pass
    else:
        raise AssertionError('Incomplete paired Depth results were accepted.')
    print('PASS: D0-D7 transforms, independent RNG, exact noise moments and paired five-set report')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--e4-config', help='Generated E4 test YAML from the completed E4 run.')
    parser.add_argument('--e4-checkpoint', help='The single trained E4 Phase B checkpoint.')
    parser.add_argument('--rgb-config', default=str(ROOT/'options/test/mambairv2/test_UDR_RGB_reference_x4.yml'))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--seeds', nargs='+', type=int, default=[10, 11, 12])
    parser.add_argument('--corruption-seed', type=int, default=2026)
    parser.add_argument('--max-images', type=int, default=0,
                        help='0 uses all five datasets; a positive value is only for a partial smoke run.')
    parser.add_argument('--output', default=str(ROOT/'results/E5_depth_causality'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not args.e4_config or not args.e4_checkpoint:
            parser.error('Production E5 requires --e4-config and --e4-checkpoint.')
        run(args)


if __name__ == '__main__':
    main()
