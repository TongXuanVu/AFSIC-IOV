import torch
import torch.nn.functional as F

def compute_proto_loss(features, targets, proto_matrix, normalize=True, reduction="mean"):
    """Prototype Alignment Loss.

    Dac ta (muc 6):  L_proto = sum ||z_i(x) - p_tilde_{i,y}||^2   tren z GOC.

    normalize=True (mac dinh, giu hanh vi cu):
        chuan hoa z truoc khi tinh MSE, nen thuc chat la 2(1 - cos(z, p)) -
        cung huong toi uu nhung THANG GIA TRI khac dac ta, tuc lambda_proto
        trong config mang y nghia khac lambda_proto trong cong thuc cua bai.
        Cach nay dong nhat voi phan con lai cua phuong phap (classifier va
        FSP loss deu dung cosine).

    normalize=False:
        dung cong thuc dac ta - khoang cach Euclid binh phuong tren z goc.
        Neu dung thi phai do lai lambda_proto.

    reduction="none" -> vector [B] loss tung mau, de caller ap TRONG SO LOP
        (xem ghi chu o compute_fsp_loss). "mean" giu nguyen hanh vi cu.
    """
    z = F.normalize(features, p=2, dim=1) if normalize else features
    per_sample = F.mse_loss(z, proto_matrix[targets], reduction="none").mean(dim=1)
    if reduction == "none":
        return per_sample
    return per_sample.mean()
