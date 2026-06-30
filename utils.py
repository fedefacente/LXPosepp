import torch.nn.functional as F
import torch
import kornia.geometry as kg

def square(img: torch.Tensor,height,width)-> torch.Tensor:
    # padding real image
    target_size = max(height,width)
    pad_h = target_size - height
    pad_w = target_size - width
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    value = img.max()
    img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=value)

    return img

def warp_to_canonical_space(img, K_real, K_can):
    B, C, H_img, W_img = img.shape

    def to_img_coords(K):
        K_adj = K.clone()
        K_adj[:, 1, 2] = H_img - K[:, 1, 2]
        return K_adj

    K_real_adj = to_img_coords(K_real)
    K_can_adj  = to_img_coords(K_can)

    H = K_can_adj @ torch.inverse(K_real_adj)

    img_warped = kg.warp_perspective(
        img, H, (H_img, W_img),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    return img_warped