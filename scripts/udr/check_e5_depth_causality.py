"""CPU reference integration check for E5's frozen E4 paired inference."""
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.udr.check_e4_udrv2 import load_architectures
from scripts.udr.e0_mechanism_audit import infer_partitioned
from scripts.udr.e5_depth_causality import (CONDITIONS, depth_variants,
                                            factor_statistics, seed_model)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    _, _, _, _, e4 = load_architectures()
    model = e4.E4UDRMambaIRv2(
        uncertainty_mode='learned_error', img_size=8, embed_dim=12,
        depths=(1,) * 6, num_heads=(3,) * 6, window_size=4,
        d_state=2, inner_rank=4, num_tokens=4, mlp_ratio=1.,
        upscale=4, upsampler='pixelshuffle').eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    rgb = torch.rand(1, 3, 8, 8)
    depth = torch.rand(1, 1, 8, 8)
    other = torch.flip(depth, dims=(-1,))
    variants = depth_variants(depth, other, 'synthetic/image', 2026)
    for seed in (10, 11, 12):
        reference = None
        for condition in CONDITIONS:
            seed_model(seed)
            prediction, maps, _ = infer_partitioned(model, rgb, variants[condition], audit=True)
            assert prediction.shape == (1, 3, 32, 32)
            assert torch.isfinite(prediction).all()
            assert all(np.isfinite(value) for value in factor_statistics(maps).values())
            routing = [maps[key] for key in
                       ('route_entropy', 'route_maxprob', 'route_margin', 'ambiguity')]
            if reference is None:
                reference = routing
            else:
                for expected, actual in zip(reference, routing):
                    np.testing.assert_allclose(expected, actual, rtol=0, atol=1e-6)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    print('PASS: one frozen E4 state, D0-D7 full forward, paired RGB routing and factor hooks')


if __name__ == '__main__':
    main()
