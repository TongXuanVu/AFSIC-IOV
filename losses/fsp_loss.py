import torch
import torch.nn.functional as F

def compute_fsp_loss(features, targets, proto_matrix, T_fsp=0.5, reduction="mean"):
    """Few-Shot Sparse Pairwise Loss.

    reduction="mean"  -> mot so vo huong (hanh vi cu, giu nguyen cho moi caller
                         khac).
    reduction="none"  -> vector [B] loss tung mau, de caller ap TRONG SO LOP.
                         Can thiet vi trung binh tron tren lo 99,6% Benign lam
                         cac lop hiem gan nhu khong dong gop gi vao gradient.
    """
    z_norm = F.normalize(features, p=2, dim=1)
    sims = torch.mm(z_norm, proto_matrix.t())
    batch_size = z_norm.size(0)

    # Positive pairs
    s_pos = sims[torch.arange(batch_size), targets]

    # Negative pairs (exclude the true target)
    mask = torch.ones_like(sims).scatter_(1, targets.unsqueeze(1), 0.0)
    s_neg, _ = torch.max(sims * mask - (1.0 - mask) * 1e9, dim=1)

    # softplus((s_neg - s_pos)/T) == log(1 + exp(.)) nhung on dinh so hoc hon
    per_sample = F.softplus((s_neg - s_pos) / T_fsp)
    if reduction == "none":
        return per_sample
    return per_sample.mean()
