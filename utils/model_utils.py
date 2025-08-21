import torch
import torch.nn.functional as F
from torchvision import transforms

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()

    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    if len(size) == 3:
        feat_std = feat_var.sqrt().view(N, C, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1)
    else:
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def get_img(img, resolution=512):
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    img = transform(img)
    return img.unsqueeze(0)

@torch.no_grad()
def slerp(p0: torch.Tensor, p1: torch.Tensor, t, use_adain=False, eps: float = 1e-7):
    """
    Spherical interpolation between p0 and p1.
    p0, p1: (..., D) tensors
    t: float or 0..1 tensor broadcastable to p0[..., :1]
    """
    device = p0.device
    in_dtype = p0.dtype

    # 工作精度：MPS 下固定 float32；其他设备保持在 16/32/bfloat16 中
    if device.type == "mps":
        work_dtype = torch.float32
    else:
        work_dtype = in_dtype if in_dtype in (torch.float16, torch.float32, torch.bfloat16) else torch.float32

    p0 = p0.to(device=device, dtype=work_dtype)
    p1 = p1.to(device=device, dtype=work_dtype)

    # 归一化避免数值问题
    p0n = F.normalize(p0, dim=-1, eps=eps)
    p1n = F.normalize(p1, dim=-1, eps=eps)

    # t 统一成张量
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=device, dtype=work_dtype)
    t = t.to(device=device, dtype=work_dtype)

    # 余弦与夹角
    dot = (p0n * p1n).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    # 当两向量几乎平行时退化为线性插值
    mask = (sin_omega.abs() > 1e-4).unsqueeze(-1)

    # 为了广播，把 t 变成和最后一维匹配
    t_exp = t.view(*([1] * (p0.dim() - 1)), 1)

    out_slerp = (torch.sin((1 - t_exp) * omega.unsqueeze(-1)) / sin_omega.unsqueeze(-1)) * p0 + \
                (torch.sin(t_exp * omega.unsqueeze(-1)) / sin_omega.unsqueeze(-1)) * p1
    out_lerp = (1 - t_exp) * p0 + t_exp * p1

    out = torch.where(mask, out_slerp, out_lerp)

    # 如果你在别处用 AdaIN，可在这里加分支（保持 fp32/mps 兼容）
    # if use_adain:
    #     out = your_adain(out, ...)

    return out.to(in_dtype)


def do_replace_attn(key: str):
    # return key.startswith('up_blocks.2') or key.startswith('up_blocks.3')
    return key.startswith('up')
