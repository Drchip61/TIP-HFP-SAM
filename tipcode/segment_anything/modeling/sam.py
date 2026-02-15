# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from icecream import ic
import copy
from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .frequency_final_point import *
import numpy as np
from torchprofile import profile_macs
import time
import torch

def find_max_k_in_grid(img_tensor, g, k, t = 1):
    #print(img_tensor.size())
    C, H, W = img_tensor.shape  # 图像的通道数、高度和宽度
    grid_h, grid_w = H // g, W // g  # 计算每个小格子的高度和宽度

      # 存储每个格子的最大值及其位置
    final_pos = []
    final_val = []

    # 遍历每个格子
    for c in range(C):
        max_values_positions = []
        max_values_ = []
        min_values_positions = []
        min_values_ = []
        for i in range(g):
            for j in range(g):
            # 定位当前格子
                grid = img_tensor[c, i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
            #print(grid[].size())
            # 在每个通道中找到最大的 k 个值及其索引
            
                #print(grid[c].reshape(-1).size())
                grid_channel = grid.reshape(-1)  # 展平当前通道的格子
                max_vals, max_indices = torch.topk(grid_channel, k)
                min_vals, min_indices = torch.topk(-grid_channel, k)
                
                # 转换索引为二维坐标
                for idx in range(len(max_indices)):
                    row = max_indices[idx] // grid_w + i*grid_h
                    col = max_indices[idx] % grid_w + j*grid_w
                    max_values_positions.append((row.item(), col.item()))
                    #print(max_vals[idx],abs(1-t))
                    if max_vals[idx] > 0.5:
                        max_values_.append(t)
                    else:
                        max_values_.append(abs(1-t))

                    row = min_indices[idx] // grid_w + i*grid_h
                    col = min_indices[idx] % grid_w + j*grid_w
                    min_values_positions.append((row.item(), col.item()))
                    #print(-min_vals[idx],abs(1-t))
                    if -min_vals[idx] < 0.5:
                        min_values_.append(abs(1-t))
                    else:
                        min_values_.append(t)
        temp_pos = torch.cat([torch.tensor(max_values_positions),torch.tensor(min_values_positions)],dim=0)
        temp_val = torch.cat([torch.tensor(max_values_),torch.tensor(min_values_)],dim=0)
        #print(temp_val)
        final_pos.append(temp_pos)
        final_val.append(temp_val)
        #final_val.append(temp_val)
        #final_pos.append()
        #final_val.append()

    
    te = torch.stack(final_pos)
    tw = torch.stack(final_val)
    #print(tw.size())
    return te,tw#final_pos,final_val#max_values_positions,max_values_


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.mask_de128 = copy.deepcopy(self.mask_decoder)
        #self.mask_de256 = copy.deepcopy(self.mask_decoder)
        
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(self,deal_mask:torch.nn.Module,deal_prompt:torch.nn.Module,fre_adapter:torch.nn.Module,batched_input, fre,multimask_output, image_size):
        if isinstance(batched_input, list):
            outputs = self.forward_test(batched_input,fre, multimask_output)
        else:
            outputs = self.forward_train(deal_mask,deal_prompt,fre_adapter,batched_input,fre, multimask_output, image_size)
        return outputs

    def forward_train(self,deal_mask:torch.nn.Module,deal_prompt:torch.nn.Module,fre_adapter:torch.nn.Module,batched_input, fre,multimask_output, image_size):
        
        input_images = batched_input#self.preprocess(batched_input)

        fre_mask = frequency_grid_mask(fre)
        #print(fre.size())
        image_embeddings = self.image_encoder(input_images,fre_mask,fre_adapter)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, boxes=None, masks=None,
        )
        low_res_masks1, iou_predictions,new_mask = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )

        low_res_masks_b = low_res_masks1[:,1,:,:].unsqueeze(1)
        #act = nn.Sigmoid()
        #mask_prompt_deal = torch.sigmoid(low_res_masks1)
        #mask_prompt_deal = (mask_prompt_deal>0.7).int()
        #print(mask_prompt_deal)
        #new1 = low_res_masks1[:,1,:,:].unsqueeze(1)
        mask_prompt_deal = torch.softmax(low_res_masks1,1)#.cpu().numpy()
        #print(mask_prompt_deal[0,:,:10,:10])
        #mask_prompt_deal[:,0,:,:] += 0.2
        #print(mask_prompt_deal[0,:,:10,:10])
        mask_prompt_01 = torch.argmax(mask_prompt_deal, axis=1).unsqueeze(1)#.astype(np.uint16)
        
        masks_prompt_deal = self.postprocess_masks(
            low_res_masks1,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )

        #masks_prompt_deal = new_mask
        #print(masks_prompt_deal.size())

        #masks_prompt_deal = torch.sigmoid(masks_prompt_deal)#.cpu().numpy()
        #print(masks_prompt_deal.size())

        
        #
        c = masks_prompt_deal[:,1,:,:]
        d = masks_prompt_deal[:,0,:,:]
        #masks_prompt_deal = masks_prompt_deal.squeeze(1)

        grid_point,grid_value = point_prompt(fre,c)

        point_yty = [grid_point.cuda(),grid_value.cuda()]
        sparse_embeddings, dense_embeddings = deal_prompt(
            points=point_yty, boxes=None, masks=mask_prompt_01.float(),
        )
        low_res_masks2, iou_predictions,new_mask = deal_mask(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )
        low_res_masks_b2 = low_res_masks2[:,1,:,:].unsqueeze(1)
        masks = self.postprocess_masks(
            low_res_masks2,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )

        outputs = {
            'masks': masks_prompt_deal,
            'iou_predictions': iou_predictions,
            'low_res_logits': [low_res_masks_b,low_res_masks_b2,new_mask]
        }
        return outputs

    @torch.no_grad()
    def forward_test(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

