<div align="center">
  <h2>HFP-SAM: Hierarchical Frequency Prompted SAM for Efficient Marine Animal Segmentation</h2>
  <p>
    <a href="tipcode/train_y.py">Training</a> ·
    <a href="tipcode/download_sam_ckpt.sh">Download SAM checkpoint</a>
  </p>
  <p>
    <img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" />
    <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" />
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg" />
  </p>
</div>

## Overview
This repository provides the code for **HFP-SAM: Hierarchical Frequency Prompted SAM for Efficient Marine Animal Segmentation**.
This GitHub release is **code-only** (no paper sources / response letters / review files).

HFP-SAM targets **Marine Animal Segmentation (MAS)** and introduces:

- **FGA (Frequency Guided Adapter)**: injects marine-scene priors into a frozen SAM backbone using frequency-domain prior masks
- **FPS (Frequency-aware Point Selection)**: selects informative point prompts by combining frequency analysis with coarse SAM masks
- **FVM (Full-View Mamba)**: aggregates spatial + channel context with linear complexity

> Note: **Datasets and large model weights are not included** in this repo. A download script and dataset format are provided.

## Method at a Glance

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

## Repository Layout

```text
TIP-HFP-SAM/
  tipcode/               # training / inference code
  requirements.txt       # python deps (torch/torchvision: install via PyTorch official guide)
  LICENSE                # Apache-2.0
  CITATION.cff           # citation metadata for GitHub
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

> We recommend installing `torch/torchvision` following the official PyTorch guide to match your CUDA / platform.

## Optional: FVM/Mamba CUDA extension
`tipcode/segment_anything/modeling/vmamba.py` relies on the CUDA extension `selective_scan_cuda_core` for the FVM/Mamba path.
If it is not available, the code will **automatically fall back** to a lightweight `DWConv+PWConv` approximation for debuggability (**results will differ from the paper**).

## Download SAM checkpoint (required)
The SAM `vit_b` checkpoint is ~375MB and is ignored by `.gitignore`. Download it into `tipcode/`:

```bash
bash tipcode/download_sam_ckpt.sh
```

This will create: `tipcode/sam_vit_b_01ec64.pth`

## Data Format
`tipcode/dataset_fre.py` expects the following structure (file stems must match):

```text
<data_root>/
  Image/          xxx.jpg
  Masks/          xxx.png        # single-channel mask (0/255)
  Frequency_2/    xxx.jpg        # single-channel frequency prior (0~255)
```

## Training

```bash
python3 tipcode/train_y.py \
  --data_root <data_root> \
  --sam_ckpt tipcode/sam_vit_b_01ec64.pth \
  --save_dir checkpoint \
  --epochs 48 \
  --batch_size 6
```

Training checkpoints (`.pth`) will be written to `--save_dir` and ignored by `.gitignore`.

## Inference / Testing (example)

```bash
python3 tipcode/test.py \
  --checkpoint checkpoint/xxx.pth \
  --test_image_path <test_root>/image/ \
  --test_gt_path <test_root>/masks/ \
  --test_fre_path <test_root>/Frequency_2/ \
  --save_path outputs/pred_masks
```

## Citation
This repository includes `CITATION.cff` (auto-detected by GitHub).

## Acknowledgements
- This project includes and modifies Meta's **Segment Anything** code (Apache-2.0). Please obtain SAM weights from the official source.

