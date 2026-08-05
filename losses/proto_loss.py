import torch
import torch.nn.functional as F

def compute_proto_loss(features, targets, proto_matrix, normalize=True):
    """Prototype Alignment Loss.

    Đặc tả (mục 6):  L_proto = sum ||z_i(x) - p_tilde_{i,y}||^2   trên z GỐC.

    normalize=True (mặc định, giữ hành vi cũ):
        chuẩn hoá z trước khi tính MSE, nên thực chất là 2(1 - cos(z, p)) —
        cùng hướng tối ưu nhưng THANG GIÁ TRỊ khác đặc tả, tức lambda_proto
        trong config mang ý nghĩa khác lambda_proto trong công thức của bài.
        Cách này đồng nhất với phần còn lại của phương pháp (classifier và
        FSP loss đều dùng cosine).

    normalize=False:
        đúng công thức đặc tả — khoảng cách Euclid bình phương trên z gốc.
        Nếu dùng thì phải dò lại lambda_proto.
    """
    z = F.normalize(features, p=2, dim=1) if normalize else features
    loss_proto = F.mse_loss(z, proto_matrix[targets], reduction="mean")
    return loss_proto
