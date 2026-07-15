"""
Dataset adapter cho CAN-bus IoV federated data (AFSIC-IoV).

Data format (label-skew, tham gia động giữa các client):
  - Train: data/federated_data/client_{client_id}_task_{task_id}.pt   (task_id: 1..5)
           mỗi file là dict {"x": Tensor[N, 31] float16, "y": Tensor[N] int64}
           và CHỈ chứa các lớp mới của stage đó. Client không tham gia stage
           nào thì không có file tương ứng.
  - Test:  data/global_test_data.pt (chung cho mọi client, chỉ client 0 load)

13 lớp (xem data/class_mapping.json):
  Task 1: Benign(0), DoS(1), double(2)
  Task 2: force-neutral(3), fuzzing(4), interval(5)
  Task 3: rpm(6), rpm-accessory(7), speed(8)
  Task 4: speed-accessory(9), standstill(10)
  Task 5: systematic(11), triple(12)
=> task_increments = [3, 3, 3, 2, 2]

Lưu ý RAM: dữ liệu được giữ nguyên float16 trong bộ nhớ; DummyDataset sẽ
convert sang float32 theo từng batch. Với client lớn (~29M mẫu) có thể đặt
MAX_SAMPLES_PER_CLASS / TEST_MAX_SAMPLES_PER_CLASS (trainer set từ config
"max_samples_per_class" / "test_max_samples_per_class") để cắt bớt lớp đa số.
"""
import numpy as np
import torch
import os

# --- Đường dẫn tới data ---
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_DATA_DIR = os.path.join(_REPO_ROOT, "data")

_TEST_FILE = os.path.join(_LOCAL_DATA_DIR, "global_test_data.pt")
_FEDERATED_DIR = os.path.join(_LOCAL_DATA_DIR, "federated_data")

# Quét file test và thư mục data trên Kaggle nếu chạy trên Kaggle
if os.path.exists("/kaggle/input"):
    import glob
    print("[iCANIoV] Đang quét /kaggle/input để tìm dữ liệu...")
    test_paths = glob.glob("/kaggle/input/**/global_test_data.pt", recursive=True)
    if test_paths:
        _TEST_FILE = test_paths[0]
        print(f"[iCANIoV] Auto-detected Test File: {_TEST_FILE}")
    train_files = glob.glob("/kaggle/input/**/federated_data/client_*_task_*.pt", recursive=True)
    if train_files:
        _FEDERATED_DIR = os.path.dirname(train_files[0])
        print(f"[iCANIoV] Auto-detected Data Dir: {_FEDERATED_DIR}")

_NUM_TASKS = 5
_NUM_CLASSES = 13
_NUM_FEATURES = 31

# Giới hạn số mẫu mỗi lớp (None = dùng toàn bộ). Trainer set từ config.
MAX_SAMPLES_PER_CLASS = None
TEST_MAX_SAMPLES_PER_CLASS = None

# Thứ tự lớp tự nhiên 0..12 (nhãn trong file .pt đã theo class_mapping.json,
# các lớp đã được xếp sẵn theo thứ tự task tăng dần).
DEFAULT_TASK_CLASS_ORDER = list(range(_NUM_CLASSES))


def _subsample_per_class(x, y, cap, seed):
    """Giữ tối đa `cap` mẫu mỗi lớp (giữ nguyên toàn bộ lớp thiểu số)."""
    if cap is None:
        return x, y
    rng = np.random.default_rng(seed)
    keep_parts = []
    for cls in torch.unique(y).tolist():
        idx = torch.nonzero(y == cls, as_tuple=True)[0]
        if len(idx) > cap:
            sel = rng.choice(len(idx), size=int(cap), replace=False)
            idx = idx[torch.from_numpy(np.sort(sel))]
        keep_parts.append(idx)
    keep = torch.cat(keep_parts)
    return x[keep], y[keep]


class iCANIoV:
    """CAN-bus IoV dataset adapter tương thích với DataManager (Federated Learning)."""
    use_path = False
    train_trsf = []
    test_trsf = []
    common_trsf = []

    def download_data(self, client_id=None):
        assert client_id is not None, "[iCANIoV] Yêu cầu client_id cho Federated Learning."

        xs, ys = [], []
        for task_id in range(1, _NUM_TASKS + 1):
            path = os.path.join(_FEDERATED_DIR, f"client_{client_id}_task_{task_id}.pt")
            if not os.path.exists(path):
                # Client không tham gia stage này (tham gia động) - bỏ qua
                continue
            task_data = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(task_data, dict):
                x, y = task_data["x"], task_data["y"]
            else:
                x, y = task_data
            x, y = _subsample_per_class(
                x, y, MAX_SAMPLES_PER_CLASS, seed=client_id * 100 + task_id
            )
            xs.append(x)
            ys.append(y)

        if xs:
            # Giữ float16 để tiết kiệm RAM; DummyDataset convert float32 theo batch
            self.train_data = torch.cat(xs).numpy()
            self.train_targets = torch.cat(ys).numpy().astype(np.int64)
        else:
            self.train_data = np.empty((0, _NUM_FEATURES), dtype=np.float16)
            self.train_targets = np.empty((0,), dtype=np.int64)
        del xs, ys

        # Test set chung: chỉ load cho client 0 để tiết kiệm RAM
        if client_id == 0:
            assert os.path.exists(_TEST_FILE), f"[iCANIoV] Không tìm thấy file test: {_TEST_FILE}"
            test_dict = torch.load(_TEST_FILE, map_location="cpu", weights_only=False)
            if isinstance(test_dict, dict):
                tx, ty = test_dict["x"], test_dict["y"]
            else:
                tx, ty = test_dict
            tx, ty = _subsample_per_class(tx, ty, TEST_MAX_SAMPLES_PER_CLASS, seed=12345)
            self.test_data = tx.numpy()
            self.test_targets = ty.numpy().astype(np.int64)
            del test_dict, tx, ty
        else:
            self.test_data = np.empty((0, _NUM_FEATURES), dtype=np.float16)
            self.test_targets = np.empty((0,), dtype=np.int64)

        self.class_order = DEFAULT_TASK_CLASS_ORDER
        print(
            f"[iCANIoV - Client {client_id}] Loaded: "
            f"train={self.train_data.shape}, test={self.test_data.shape}, "
            f"classes={sorted(np.unique(self.train_targets).tolist())}"
        )
