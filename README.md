<div align="center">
  <h2>HFP-SAM：用于高效海洋动物分割的分层频域提示 SAM</h2>
  <p>
    <a href="tipcode/train_y.py">训练脚本</a> ·
    <a href="tipcode/download_sam_ckpt.sh">下载 SAM 权重</a>
  </p>
  <p>
    <img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" />
    <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" />
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg" />
  </p>
</div>

## 项目简介
本仓库提供论文 **HFP-SAM: Hierarchical Frequency Prompted SAM for Efficient Marine Animal Segmentation** 的实验代码（GitHub 版本仅保留与论文强相关的代码与说明）。
方法面向 **Marine Animal Segmentation (MAS)**，围绕 SAM 在细粒度细节与频域信息感知不足的问题，引入：

- **FGA（Frequency Guided Adapter）**：利用频域先验 mask，高效注入海洋场景信息到冻结的 SAM 主干
- **FPS（Frequency-aware Point Selection）**：基于频域分析生成高亮区域，并与粗分割结果融合产生点提示
- **FVM（Full-View Mamba）**：以线性复杂度提取空间与通道上下文

> 说明：**数据集与大模型权重默认不随仓库发布**（避免 GitHub 体积/协议问题）。仓库提供下载脚本与数据格式说明。

## 方法概览

```mermaid
flowchart LR
  A[Input image] --> B[FGA: Frequency Guided Adapter]
  B --> C[SAM backbone (frozen)]
  C --> D[Coarse mask]
  D --> E[FPS: Frequency-aware Point Selection]
  E --> F[Point prompts + mask prompts]
  F --> G[SAM prompt encoder + mask decoder]
  G --> H[FVM: Full-View Mamba]
  H --> I[Final mask]
```

## 论文与材料
本仓库**不包含**论文源文件/response/review 等材料（仅发布代码）。如需论文/补充材料，请使用论文官方渠道或联系作者。

## 目录结构

```text
TIP-HFP-SAM/
  tipcode/               # 训练/测试代码（主实现）
  requirements.txt       # Python 依赖（torch/torchvision 建议按官方方式安装）
  LICENSE                # Apache-2.0
  CITATION.cff           # GitHub 引用信息
```

## 环境配置

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

> 建议优先参考 PyTorch 官方安装说明，安装与你的 CUDA/平台匹配的 `torch/torchvision`。

## 可选：FVM/Mamba 加速依赖
`tipcode/segment_anything/modeling/vmamba.py` 的 FVM/Mamba 分支依赖 CUDA 扩展 `selective_scan_cuda_core`。
若你的环境未安装该扩展，代码会**自动降级**为 `DWConv+PWConv` 近似实现以保证可运行（**结果会与论文不同**）。

## 准备 SAM 预训练权重（必需）
SAM 的 `vit_b` checkpoint **体积约 375MB**，已在 `.gitignore` 中排除。请下载到 `tipcode/` 下：

```bash
bash tipcode/download_sam_ckpt.sh
```

默认会生成：`tipcode/sam_vit_b_01ec64.pth`

## 数据准备
`tipcode/dataset_fre.py` 默认的数据组织如下（文件名去掉后缀需一一对应）：

```text
<data_root>/
  Image/          xxx.jpg
  Masks/          xxx.png        # 0/255 的单通道 mask
  Frequency_2/    xxx.jpg        # 单通道频域先验图（0~255）
```

## 训练

```bash
python3 tipcode/train_y.py \
  --data_root <data_root> \
  --sam_ckpt tipcode/sam_vit_b_01ec64.pth \
  --save_dir checkpoint \
  --epochs 48 \
  --batch_size 6
```

训练权重（`.pth`）默认会输出到 `--save_dir`，并被 `.gitignore` 自动排除。

## 测试 / 推理（示例）
仓库内提供了一个示例测试脚本（路径参数可按你的数据集调整）：

```bash
python3 tipcode/test.py \
  --checkpoint checkpoint/xxx.pth \
  --test_image_path <test_root>/image/ \
  --test_gt_path <test_root>/masks/ \
  --test_fre_path <test_root>/Frequency_2/ \
  --save_path outputs/pred_masks
```

## 引用
仓库已提供 `CITATION.cff`（GitHub 会自动识别）。

## 致谢
- 本项目包含并修改了 Meta 的 **Segment Anything** 代码（Apache-2.0）。SAM 权重请从官方渠道下载。

