"""Loader vectorized cho dữ liệu dạng bảng (CAN-bus, numpy 2D).

DataLoader chuẩn gọi __getitem__ TỪNG MẪU rồi collate — với 29M mẫu/epoch,
riêng overhead Python + convert float16→float32 từng dòng đã chiếm hàng chục
phút mỗi lượt quét, GPU ngồi chờ CPU. Loader này cắt mảng numpy theo cả batch
(fancy-indexing vectorized) nên nhanh hơn 20–50 lần.

Dùng make_loader() thay cho torch.utils.data.DataLoader tại các điểm tạo
loader; tự fallback về DataLoader chuẩn với dataset không phải dạng bảng.
"""
import math

import numpy as np
import torch
from torch.utils.data import DataLoader


class FastTensorLoader:
    """Iterate (idx, x_float32, y_int64) theo batch bằng slicing numpy.

    gather(sel) -> (x_batch, y_batch) là hàm lấy mẫu theo mảng chỉ số.
    """

    def __init__(self, n, gather, batch_size, shuffle=False):
        self.n = int(n)
        self.gather = gather
        self.batch_size = int(batch_size)
        self.shuffle = shuffle

    def __len__(self):
        return max(1, math.ceil(self.n / self.batch_size))

    def __iter__(self):
        order = np.random.permutation(self.n) if self.shuffle else np.arange(self.n)
        for s in range(0, self.n, self.batch_size):
            sel = order[s:s + self.batch_size]
            x, y = self.gather(sel)
            yield (
                torch.from_numpy(sel),
                torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64)),
            )


def _is_tabular(arr):
    return isinstance(arr, np.ndarray) and arr.ndim == 2


def make_loader(dataset, batch_size, shuffle=False, num_workers=0):
    """Trả về FastTensorLoader cho dataset dạng bảng, DataLoader chuẩn nếu không."""
    from utils.data_manager import DummyDataset, SubDummyDataset

    if isinstance(dataset, SubDummyDataset) and _is_tabular(dataset.x_source):
        x_src, y_src = dataset.x_source, dataset.y_source
        indices = np.asarray(dataset.indices, dtype=np.int64)
        len_src = dataset.len_source
        app_x, app_y = dataset.append_x, dataset.append_y
        if app_x is not None and not _is_tabular(np.asarray(app_x)):
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        if app_x is not None:
            app_x = np.asarray(app_x)
            app_y = np.asarray(app_y)

        def gather(sel):
            src_mask = sel < len_src
            if app_x is None or src_mask.all():
                rows = indices[sel]
                return x_src[rows], y_src[rows]
            src_sel = sel[src_mask]
            app_sel = sel[~src_mask] - len_src
            x = np.concatenate([x_src[indices[src_sel]], app_x[app_sel]], axis=0)
            y = np.concatenate([y_src[indices[src_sel]], app_y[app_sel]], axis=0)
            return x, y

        return FastTensorLoader(len(dataset), gather, batch_size, shuffle)

    if isinstance(dataset, DummyDataset) and _is_tabular(np.asarray(dataset.images) if not isinstance(dataset.images, np.ndarray) else dataset.images):
        x = np.asarray(dataset.images)
        y = np.asarray(dataset.labels)

        def gather(sel):
            return x[sel], y[sel]

        return FastTensorLoader(len(dataset), gather, batch_size, shuffle)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
