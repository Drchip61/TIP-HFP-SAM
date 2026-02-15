"""
HFP-SAM 训练脚本（频域提示 + SAM）。

原始版本依赖在 tipcode 目录下运行，并使用相对路径读取数据/权重。
这里做了轻量参数化，便于从仓库根目录直接运行与复现实验。
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

import dataset_fre
from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry


def _parse_args() -> argparse.Namespace:
    this_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser("HFP-SAM training (frequency prompts)")
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(this_dir / "train"),
        help="数据根目录（需要包含 Image/ Masks/ Frequency_2/）",
    )
    parser.add_argument(
        "--sam_ckpt",
        type=str,
        default=str(this_dir / "sam_vit_b_01ec64.pth"),
        help="SAM vit_b checkpoint 路径（sam_vit_b_01ec64.pth）",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(this_dir / "checkpoint"),
        help="保存训练权重的目录（默认会被 .gitignore 忽略）",
    )
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--base_lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_period", type=int, default=2950)
    parser.add_argument("--max_iterations", type=int, default=29500)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_every_epochs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    warnings.filterwarnings("ignore")

    device = torch.device(
        args.device if (str(args.device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    data_root = Path(args.data_root)
    sam_ckpt = Path(args.sam_ckpt)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not sam_ckpt.exists():
        raise FileNotFoundError(
            f"未找到 SAM checkpoint：{sam_ckpt}\n"
            f"请先运行：bash {Path(__file__).resolve().parent / 'download_sam_ckpt.sh'}"
        )

    # build model
    sam = sam_model_registry["vit_b"](checkpoint=str(sam_ckpt))
    sam = sam[0]
    model = LoRA_Sam(sam, int(args.lora_rank)).to(device)
    # 用 SAM checkpoint 初始化 prompt encoder / mask decoder（与原脚本一致）
    model.load_lora_parameters(str(sam_ckpt))
    model.train()

    # dataset
    cfg = dataset_fre.Config(
        datapath=str(data_root),
        savepath=str(save_dir),
        mode="train",
        batch=int(args.batch_size),
        lr=float(args.base_lr),
        momen=0.9,
        decay=5e-4,
        epoch=int(args.epochs),
    )
    dataset = dataset_fre.Data(cfg)
    train_loader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=int(args.batch_size),
        pin_memory=(device.type == "cuda"),
        num_workers=int(args.num_workers),
    )

    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(args.base_lr) / max(int(args.warmup_period), 1),
        betas=(0.9, 0.999),
        weight_decay=float(args.weight_decay),
    )

    iter_num = 0
    max_iterations = int(args.max_iterations) if int(args.max_iterations) > 0 else int(args.epochs) * len(train_loader)
    warmup_period = max(int(args.warmup_period), 1)

    for epoch_num in range(int(args.epochs)):
        show_dict = {"epoch": epoch_num}
        for i_batch, (im1, label0, label2, fre) in enumerate(
            tqdm.tqdm(train_loader, ncols=80, postfix=show_dict)
        ):
            im1 = im1.to(device, non_blocking=True).float()
            label0 = label0.to(device, non_blocking=True)
            label2 = label2.to(device, non_blocking=True)
            fre = fre.to(device, non_blocking=True).float()

            outputs = model(im1, fre, 1, 512)

            loss0 = ce_loss(outputs[0], label0.long())
            loss1 = ce_loss(outputs[1], label0.long())
            loss2 = ce_loss(outputs[2], label2.long())
            loss = loss0 + loss1 + loss2

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # warmup + poly decay（沿用原逻辑）
            if iter_num < warmup_period:
                lr_ = float(args.base_lr) * ((iter_num + 1) / warmup_period)
            else:
                shift_iter = iter_num - warmup_period
                lr_ = float(args.base_lr) * (1.0 - shift_iter / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = float(lr_)
            iter_num += 1

            if i_batch % 50 == 0:
                print(
                    f"{i_batch} "
                    f"loss0:{loss0.item():.3f} | loss1:{loss1.item():.3f} | loss2:{loss2.item():.3f} "
                    f"| lr:{optimizer.param_groups[0]['lr']:.6f}"
                )

        if int(args.save_every_epochs) > 0 and (epoch_num + 1) % int(args.save_every_epochs) == 0:
            out_path = save_dir / f"samed_epoch_{epoch_num + 1:03d}.pth"
            torch.save(model.state_dict(), out_path)
            print(f"[Saving Snapshot] {out_path}")


if __name__ == "__main__":
    main()
