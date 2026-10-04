# E5：固定 E4 的 Depth 因果性验证

本实验**不训练 E5 模型**。加载同一个训练完成的 E4 Phase B checkpoint，固定模型参数、RGB 图像、GT、五数据集、分块推理与 Y 通道 ×4/crop 4 指标，仅在推理时替换已按原数据集规则归一化的 LR Depth。每张图的八个条件都重置相同的 hard-Gumbel 种子，并在前向 hook 中逐像素核对 RGB route probability 摘要和 `U_R`；不一致直接报错。这样对照不受模型参数量、额外 branch 或随机路由差异影响。

| 代号 | Depth 输入 |
|---|---|
| D0 | 正确配对 Depth |
| D1 | 全零 |
| D2 | 常数 0.5 |
| D3 | 对该图所有 LR 像素做确定性随机空间置换 |
| D4 | 同一测试集排序后的下一张图的配对 Depth；尺寸不同时仅双线性缩放，并记录来源/缩放 |
| D5 | 独立高斯随机图；仿射校准到 D0 **逐图精确均值和总体标准差**，不截断以免破坏统计量 |
| D6 | LR 空间 Gaussian blur，σ=5、半径 15、replicate 边界 |
| D7 | 与 E4 Depth 分支相同的固定 Sobel `|∇D|`，不再输入原 Depth 强度 |

D3/D5 用独立的本地 CPU RNG，固定 `corruption_seed=2026`，同一张图跨三个 Gumbel 种子使用**同一幅破坏图**。D4 严格不取本图。源码中的注释标明了每个干预约定，便于根据实验需要审查。

## 权重与命令

先确保 E2/E3 的实测报告已按 [E2](E2_GUIDE.md)、[E3](E3_GUIDE.md) 指南生成，并按 [E4](E4_GUIDE.md) 选出最佳组合且完成 E4 训练。本仓库不会预填不存在的 E4 权重或指标。可在 E5 仓库里复现 E4 的选择/训练；E5 只读取最终 E4 权重。以下命令在已安装项目依赖和 CUDA 扩展的 Linux 服务器执行，原数据路径和评估设置不改：

```bash
git clone https://github.com/KaiXu-HIT/v3.2-M3SR-MambaIRv2-E5.git
cd v3.2-M3SR-MambaIRv2-E5
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
E2ROOT=/home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2-E2
E3ROOT=/home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2-E3
E0STATS="$E2ROOT/results/E0_udr_mechanism_audit/E0_statistics.json"
RGB=/home/BRAIN/xukai/code/v1.0-M3SR-MambaIRv2/experiments/v1.0_RGB_MambaIRv2_x4/models/net_g_490000.pth

# 如果尚未在 E4 项目生成最终权重，在本仓库完成相同的 E4 选择与训练：
python scripts/udr/prepare_e4_integration.py \
  --e2-results "$E2ROOT/results/E2_udrv2_uncertainty" \
  --e3-results "$E3ROOT/results/E3_local_alpha" \
  --e0-statistics "$E0STATS" \
  --rgb-teacher-checkpoint "$RGB"
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
  -opt options/train/mambairv2/train_E4_phaseA_x4.yml
test -f experiments/E4_selected_phaseA_x4/models/net_g_30000.pth
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
  -opt options/train/mambairv2/train_E4_phaseB_x4.yml
test -f experiments/E4_selected_phaseB_x4/models/net_g_100000.pth

# E5 不再训练。先做 CPU 合约检查，再用完整五数据集运行三种匹配种子：
python scripts/udr/e5_depth_causality.py --self-test
python scripts/udr/check_e5_depth_causality.py
CUDA_VISIBLE_DEVICES=0 python scripts/udr/e5_depth_causality.py \
  --e4-config options/test/mambairv2/test_E4_x4.yml \
  --e4-checkpoint experiments/E4_selected_phaseB_x4/models/net_g_100000.pth \
  --seeds 10 11 12 --corruption-seed 2026 \
  --output results/E5_depth_causality
```

若 E4 已在单独的 E4 仓库训练完成，可省略上面的选择及训练，直接传入那里的完整路径：

```bash
E4ROOT=/home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2-E4
CUDA_VISIBLE_DEVICES=0 python scripts/udr/e5_depth_causality.py \
  --e4-config "$E4ROOT/options/test/mambairv2/test_E4_x4.yml" \
  --e4-checkpoint "$E4ROOT/experiments/E4_selected_phaseB_x4/models/net_g_100000.pth" \
  --seeds 10 11 12 --corruption-seed 2026 \
  --output results/E5_depth_causality
```

测试脚本输出逐图/逐 seed 的 `E5_per_image.csv`、包含五集及逐数据集 mean±std、配对差与因子变化的 `E5_causality.json`，以及简表 `E5_summary.md`。每数据集先对图像求平均，再对三个匹配种子报告均值/标准差；五集平均沿用原实验各数据集等权口径。CSV 包含正确 Depth 相对各干预的逐图配对 PSNR/SSIM 差，以及 `confidence_mean`、`local_alpha_mean`、`gate_mean`、`correction_rms`、输入 Depth 均值和标准差。报告保存 E4 checkpoint SHA256 以确认单一权重来源。

主要判据是五集平均 `P_correct > P_zero` 且 `P_correct > P_shuffle`。理想排序还关注 edge、blur 和错误 Depth 的相对表现。若正确 Depth 不优于 shuffle，脚本标记 No-Go；两者差距在 ±0.01 dB 内也会提示复查（这是分析容差，不是方案新增的成功门槛）。错误或噪声 Depth 下四项机制量是否下降只报告真实方向，不强行判为成功。`--max-images N` 仅用于部分烟雾测试，输出会醒目标记不可作为完整 E5 结论。
