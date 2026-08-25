import torch
import numpy as np


def mask_iou(masks_a, masks_b):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Safely convert incoming data (even CuPy arrays) to PyTorch tensors
    if not isinstance(masks_a, torch.Tensor):
        if hasattr(masks_a, 'get'): masks_a = masks_a.get()
        masks_a = torch.tensor(masks_a, device=device, dtype=torch.bool)
    if not isinstance(masks_b, torch.Tensor):
        if hasattr(masks_b, 'get'): masks_b = masks_b.get()
        masks_b = torch.tensor(masks_b, device=device, dtype=torch.bool)

    intersection = torch.logical_and(masks_a, masks_b).sum(dim=(1, 2))
    union = torch.logical_or(masks_a, masks_b).sum(dim=(1, 2))

    iou = torch.zeros_like(intersection, dtype=torch.float32)
    valid = union > 0
    iou[valid] = intersection[valid].float() / union[valid].float()

    return iou.unsqueeze(1).cpu().numpy()


def mask_asymmetric_iou(masks_a, masks_b):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not isinstance(masks_a, torch.Tensor):
        if hasattr(masks_a, 'get'): masks_a = masks_a.get()
        masks_a = torch.tensor(masks_a, device=device, dtype=torch.bool)
    if not isinstance(masks_b, torch.Tensor):
        if hasattr(masks_b, 'get'): masks_b = masks_b.get()
        masks_b = torch.tensor(masks_b, device=device, dtype=torch.bool)

    intersection = torch.logical_and(masks_a, masks_b).sum(dim=(1, 2))
    area_a = masks_a.sum(dim=(1, 2))

    iou = torch.zeros_like(intersection, dtype=torch.float32)
    valid = area_a > 0
    iou[valid] = intersection[valid].float() / area_a[valid].float()

    return iou.unsqueeze(1).cpu().numpy()