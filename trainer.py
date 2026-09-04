import sys
import logging
import copy

# Console Windows mặc định cp1258 không in được log tiếng Việt
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import csv
import os
import glob
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms

from utils import factory
from utils.data_manager import DataManager, DummyDataset
from utils.toolkit import count_parameters
from utils.aggregation import is_aggregated_state_key, compute_aggregation_weights
from utils.fast_loader import make_loader
import seaborn as sns
from sklearn.metrics import confusion_matrix


def _is_afsic(args):
    return str(args["model_name"]).lower() in ("afsic-ids", "afsic-iov")


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        if 'num_batches_tracked' in key:
            w_avg[key] = w_avg[key].true_divide(len(w))
        else:
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def _make_transform(data_manager, mode):
    if mode == "train":
        return transforms.Compose([*data_manager._train_trsf, *data_manager._common_trsf])
    return transforms.Compose([*data_manager._test_trsf, *data_manager._common_trsf])


def _concat_or_none(parts):
    parts = [p for p in parts if p is not None and len(p) > 0]
    if not parts:
        return None
    return np.concatenate(parts, axis=0)


def _take_items(values, indices):
    if isinstance(values, np.ndarray):
        return values[indices]
    return [values[int(i)] for i in indices]


def _resolve_kshot(kshot, n_total):
    """Quy đổi kshot ra số mẫu cụ thể.

        kshot >= 1  -> số mẫu tuyệt đối (10-shot: kshot=10)
        0 < kshot < 1 -> TỈ LỆ trên số mẫu sẵn có (1%: kshot=0.01)
        None / <= 0 -> không giới hạn

    Cho phép biểu diễn kịch bản "1%" mà không cần thêm tham số mới, đồng thời
    giữ nguyên ngữ nghĩa cũ cho mọi config đang dùng kshot nguyên.
    """
    if kshot is None:
        return None
    k = float(kshot)
    if k <= 0:
        return None
    if k < 1.0:
        return max(1, int(round(k * n_total)))
    return int(k)


def _build_fewshot_train_dataset(data_manager, local_model, args, task, client_id):
    """Build few-shot new-class data plus local exemplar replay."""
    kshot = args.get("kshot", None) if args.get("fewshot_enabled", True) else None
    rng = np.random.default_rng(args.get("seed", 0) + task * 1009 + client_id * 9173)
    data_parts, target_parts = [], []
    new_count = 0

    for class_idx in range(local_model._known_classes, local_model._total_classes):
        data, targets, _ = data_manager.get_dataset(
            np.arange(class_idx, class_idx + 1),
            source="train",
            mode="test",
            ret_data=True,
        )
        if len(data) == 0:
            continue
        k_resolved = _resolve_kshot(kshot, len(data))
        if k_resolved is None:
            keep = len(data)
            data_parts.append(data)
            target_parts.append(targets)
        else:
            keep = min(k_resolved, len(data))
            selected = rng.choice(len(data), size=keep, replace=False)
            data_parts.append(_take_items(data, selected))
            target_parts.append(_take_items(targets, selected))
        new_count += keep

    memory = local_model._get_memory()
    if memory is not None and len(memory) != 0:
        mem_data, mem_targets = memory
        data_parts.append(mem_data)
        target_parts.append(mem_targets)

    data = _concat_or_none(data_parts)
    targets = _concat_or_none(target_parts)
    if data is None or targets is None:
        return None

    dataset = DummyDataset(data, targets, _make_transform(data_manager, "train"), data_manager.use_path)
    dataset.len_new_data = new_count
    return dataset


def _build_standard_train_dataset(data_manager, local_model, args):
    dataset = data_manager.get_dataset(
        np.arange(local_model._known_classes, local_model._total_classes),
        source="train",
        mode="train",
        appendent=local_model._get_memory(),
    )
    dataset.len_new_data = getattr(dataset, "len_source", len(dataset))
    return dataset


def _build_local_eval_loader(data_manager, total_classes, batch_size, max_samples=None,
                             seed=0):
    """Tap danh gia CUC BO de tinh acc_i (dau vao cua beta_acc trong Q_i).

    TOI UU: chi client 0 co du lieu test, moi client khac roi xuong nhanh
    source="train" -> forward TOAN BO du lieu huan luyen cua minh, MOI round,
    chi de lay mot con so acc. Voi client 12,6 trieu mau day la mot luot day du.
    `max_samples` (config: quality_eval_max_samples) lay mau ngau nhien co dinh
    theo seed. Day la lay mau khi ĐO, khong cat du lieu huan luyen.
    Mac dinh None = giu nguyen hanh vi cu.
    """
    indices = np.arange(0, total_classes)
    if max_samples is None:
        dataset = data_manager.get_dataset(indices, source="test", mode="test")
        if len(dataset) == 0:
            dataset = data_manager.get_dataset(indices, source="train", mode="test")
        return make_loader(dataset, batch_size=batch_size, shuffle=False) if len(dataset) else None

    # Hai cai bay o day:
    #   1. torch.utils.data.Subset -> make_loader khong nhan dang, roi ve
    #      DataLoader chuan, cham 20-50 lan (utils/fast_loader.py).
    #   2. get_dataset(ret_data=True) -> COPY toan bo mang truoc khi lay mau
    #      (client lon: 12,6 trieu x 31 float16 = 780 MB).
    # Cach dung: lay SubDummyDataset (vốn là VIEW) roi thay mang chi so cua no.
    from utils.data_manager import SubDummyDataset
    dataset = data_manager.get_dataset(indices, source="test", mode="test")
    if len(dataset) == 0:
        dataset = data_manager.get_dataset(indices, source="train", mode="test")
    if len(dataset) == 0:
        return None
    if isinstance(dataset, SubDummyDataset) and dataset.len_source > int(max_samples):
        sel = np.sort(np.random.default_rng(seed).choice(
            dataset.len_source, size=int(max_samples), replace=False))
        dataset = SubDummyDataset(
            dataset.x_source, dataset.y_source,
            np.asarray(dataset.indices)[sel],
            dataset.trsf, dataset.use_path,
            (dataset.append_x, dataset.append_y) if dataset.append_x is not None else None)
    return make_loader(dataset, batch_size=batch_size, shuffle=False)


def _build_global_learned_test_loader(global_data_manager, total_classes, batch_size,
                                      args_eval_cap=None):
    """Evaluate only classes that have already been introduced.

    CIC-IoT23 global_test_data can contain all 34 classes, while task 0 may
    expose only 6 classes. Filtering here prevents penalizing the model for
    classes that are intentionally unseen at the current incremental stage.
    """
    learned_classes = np.arange(0, total_classes)
    dataset = global_data_manager.get_dataset(learned_classes, source="test", mode="test")
    if len(dataset) == 0:
        return None

    # eval_max_per_class: gioi han SO MAU MOI LOP cua tap test.
    #
    # [ĐO] Danh gia ton 76 giay/round (log 15-08, round 1: 04:17:34 -> 04:18:50).
    # Hien la 9% mot round, nhung sau khi bo ba luot do thua no thanh ~30%.
    #
    # Vi sao cat theo LOP chu khong danh gia thua round: tap test task 0 co
    # 99,62% Benign, nen 99,62% chi phi danh gia nam o mot lop duy nhat. Cat
    # Benign xuong 200k va GIU NGUYEN toan bo mau cua lop tan cong thi:
    #   - rec_macro (recall trung binh theo lop) gan nhu khong doi: recall cua
    #     lop hiem tinh tren DU mau nhu cu, recall cua Benign uoc luong tu 200k
    #     mau (sai so chuan < 0,1%).
    #   - f1_macro THI DOI, vi precision phu thuoc ti le lop.
    #
    # ⇒ Dung cho SANG LOC (cham diem bang rec_macro). Lan chay XAC NHAN cuoi
    # cung phai dat lai None de lay dung con so dua vao luan van.
    #
    # Cach nay giu duoc duong cong TUNG ROUND — thu ma "5 round danh gia 1 lan"
    # se pha: AFSIC thoat khoi nghiem hang so o round 9 va 11, luoi 5/10/15/20/
    # 25/30 bo lot ca hai va cho ra mot duong thang 33,27 sai su that.
    cap = args_eval_cap
    if cap:
        from utils.data_manager import SubDummyDataset
        if isinstance(dataset, SubDummyDataset) and dataset.append_x is None:
            idx = np.asarray(dataset.indices)
            lab = np.asarray(dataset.y_source)[idx]
            rng = np.random.default_rng(12345)      # co dinh -> moi round, moi arm cung tap test
            keep = []
            for c in np.unique(lab):
                pos = np.nonzero(lab == c)[0]
                if len(pos) > cap:
                    pos = pos[np.sort(rng.choice(len(pos), size=int(cap), replace=False))]
                keep.append(pos)
            keep = np.sort(np.concatenate(keep))
            if len(keep) < len(idx):
                logging.info(f"eval_max_per_class={cap}: tap test {len(idx):,} -> {len(keep):,} mau "
                             f"(rec_macro gan nhu khong doi; f1_macro KHONG so sanh duoc voi ban day du)")
                dataset = SubDummyDataset(dataset.x_source, dataset.y_source, idx[keep],
                                          dataset.trsf, dataset.use_path, None)
    return make_loader(dataset, batch_size=batch_size, shuffle=False)


def _aggregate_client_prototypes(global_model, client_protos, num_clients, args=None, client_stats=None):
    """Per-class reliability-aware prototype aggregation (AFSIC-IoV).

    Trọng số per-class:
        r_{i,c} = β_n·log(1+n_{i,c}) − β_σ·σ_{i,c} + β_q·q_i − β_drift·Drift_i − β_upd·UpdateNorm_i
        α_{i,c} = softmax_i(r_{i,c} / τ_proto)

    client_stats: {client_id: {"q":..., "drift":..., "update_norm":...}} — nếu None
    (ví dụ lần gọi đầu trước khi có quality score) thì các term đó bằng 0 và
    trọng số chỉ dựa trên số mẫu + độ phân tán per-class.
    Khi client_stats khác None, chỉ các client có trong đó được tham gia.
    """
    args = args or {}
    beta_n = args.get("proto_beta_n", 1.0)
    beta_sigma = args.get("proto_beta_sigma", 1.0)
    beta_q = args.get("proto_beta_q", 1.0)
    beta_drift = args.get("proto_beta_drift", 0.5)
    beta_upd = args.get("proto_beta_update", 0.2)
    tau_proto = args.get("tau_proto_aggregation", 1.0)
    # So hang quy mo: CHUAN HOA min-max ve [0,1] giong het cach utils/aggregation.py
    # xu ly size_term cho Q_i (cung doc khoa size_term_mode).
    #
    # Ban cu cong THANG beta_n*log1p(n_ic) trong khi bon so hang con lai deu
    # trong [0,1]: sigma_ic in [0,1], q_i = alpha ~ 0,01, drift/upd la RMS tren
    # tham so nen rat nho. Ma log1p(29.000.000) = 17,18 con log1p(365) = 5,90.
    # Voi tau_proto = 1 thi softmax cho ti so trong so e^11,28 = 79.235 : 1
    # => "per-class reliability-aware prototype aggregation" thuc chat chi la
    # "lay prototype cua client nhieu mau nhat cho lop do", bon so hang do tin
    # cay con lai nam duoi nguong nhieu. Chinh aggregation.py:56-61 da canh bao
    # dung hien tuong nay cho Q_i va da chuan hoa; nhanh prototype thi quen.
    proto_size_mode = args.get("size_term_mode", "norm")

    for class_id in range(global_model._total_classes):
        active_protos = []
        r_scores = []
        size_raw = []
        counts = []
        dispersions = []
        qualities = []
        for c in range(num_clients):
            if client_stats is not None and c not in client_stats:
                continue
            if class_id not in client_protos[c]:
                continue
            info = client_protos[c][class_id]
            n_ic = max(1, int(info.get("count", 1)))
            sigma_ic = float(info.get("dispersion", 0.0))
            stats = (client_stats or {}).get(c, {})
            q_i = float(stats.get("q", 0.0))
            drift_i = float(stats.get("drift", 0.0))
            upd_i = float(stats.get("update_norm", 0.0))

            r_ic = (
                - beta_sigma * sigma_ic
                + beta_q * q_i
                - beta_drift * drift_i
                - beta_upd * upd_i
            )
            active_protos.append(info["prototype"])
            r_scores.append(r_ic)
            size_raw.append(float(np.log1p(n_ic)))
            counts.append(n_ic)
            dispersions.append(sigma_ic)
            qualities.append(q_i)

        if not active_protos:
            continue

        # Cong so hang quy mo SAU khi chuan hoa (xem ghi chu o dau ham).
        if size_raw:
            if proto_size_mode == "norm" and len(size_raw) > 1:
                _lo, _hi = min(size_raw), max(size_raw)
                _rng = (_hi - _lo) or 1.0
                size_terms = [(v - _lo) / _rng for v in size_raw]
            elif proto_size_mode == "norm":
                size_terms = [1.0 for _ in size_raw]
            else:                      # "raw" — hanh vi cu, chi de tai lap
                size_terms = size_raw
            r_scores = [r + beta_n * st for r, st in zip(r_scores, size_terms)]

        stacked = torch.stack(active_protos).float()
        w_tensor = torch.softmax(
            torch.tensor(r_scores, dtype=torch.float32) / tau_proto, dim=0
        ).unsqueeze(1)
        global_proto = torch.sum(stacked * w_tensor, dim=0)
        global_proto = global_proto / (torch.norm(global_proto, p=2) + 1e-8)

        total_count = int(sum(counts))
        mean_dispersion = sum(dispersions) / len(dispersions)
        mean_quality = sum(qualities) / len(qualities) if client_stats is not None else 1.0
        global_model.global_proto_memory.update_prototype(class_id, global_proto, total_count, mean_dispersion, mean_quality)


def _log_classifier_separability(model, tag=""):
    """Chẩn đoán rẻ: fc là CosineLinear nên logit = sigma*cos(z, w_c).
    Nếu mọi cặp w_c gần trùng nhau thì bộ phân loại KHÔNG THỂ tách lớp, bất kể
    backbone tốt đến đâu -> mô hình sẽ đoán đúng một lớp. Log ma trận cosine
    giữa các w_c dưới dạng min/mean/max để khỏi phải chạy lại mới biết.
    """
    try:
        W = model._network.fc.weight.data.float()
        W = torch.nn.functional.normalize(W, p=2, dim=1)
        C = W @ W.t()
        n = C.size(0)
        if n < 2:
            return
        off = C[~torch.eye(n, dtype=torch.bool, device=C.device)]
        logging.info(
            f"[DIAG]{tag} classifier cos(w_i,w_j) off-diag: "
            f"min={off.min():.4f} mean={off.mean():.4f} max={off.max():.4f} "
            f"(gan 1.0 = khong tach duoc lop)"
        )
    except Exception as e:
        logging.warning(f"[DIAG] khong tinh duoc classifier separability: {e}")


def _calibrate_classifier_from_prototypes(model, is_task_init=False):
    """Ghi trọng số classifier từ prototype.

    LƯU Ý QUAN TRỌNG: fc là CosineLinear -> logit = sigma * cos(z, w_c), KHÔNG
    có bias. init_new_class_weights_from_prototypes ghi đè w_c cho MỌI lớp
    (bất kể tên hàm có chữ "new"). Hàm này chạy sau MỖI lần gộp trong vòng
    round, nên mô hình toàn cục thực chất là bộ phân loại nearest-prototype:
    mọi thứ fc học được trong local training và mọi thứ khâu gộp trọng số làm
    với fc đều bị vứt bỏ mỗi round.

    Hai cờ để kiểm soát, đều mặc định GIỮ NGUYÊN hành vi cũ:

      calibrate_with_prototypes (mặc định true)
          false -> tắt hẳn; classifier toàn cục là fc đã huấn luyện + gộp.
      calibrate_once_per_task (mặc định false)
          true  -> chỉ calibrate ở lúc khởi tạo task (đúng vai trò "khởi tạo
                   lớp mới"), các round sau để fc học và được gộp bình thường.
      calibrate_new_classes_only (mặc định false)
          true  -> chỉ ghi đè các lớp thuộc task hiện tại, giữ nguyên phần đã
                   học của lớp cũ.
    """
    if not hasattr(model, "global_proto_memory"):
        return
    if not model.args.get("calibrate_with_prototypes", True):
        return
    if model.args.get("calibrate_once_per_task", False) and not is_task_init:
        return
    # AFSIC-IoV: dùng personalized prototypes (trộn local/global theo rho);
    # với global model không có local_protos, kết quả trùng global prototypes.
    if hasattr(model, "get_calibration_prototypes"):
        prototypes = model.get_calibration_prototypes()
    else:
        prototypes = model.global_proto_memory.get_all_prototypes()
    # LUU Y: dieu kien cu la `... and not is_task_init`, tuc co nay bi VO HIEU
    # dung o lan goi duy nhat co y nghia (khoi tao task). Bo ve dung y nghia:
    # bat co -> chi ghi de trong so cua cac lop MOI, giu nguyen phan lop cu ma
    # cac task truoc da hoc.
    if model.args.get("calibrate_new_classes_only", False):
        class_ids = range(model._known_classes, model._total_classes)
    else:
        class_ids = range(model._total_classes)
    model._network.init_new_class_weights_from_prototypes(prototypes, class_ids)


_PERSONALIZED_KEY_MARKERS = ("stability_encoder", "plasticity_adapter", "gate")


def _load_global_into_client(local_model, global_state, task, args):
    """Nạp trọng số global vào client.

    Khi bật personalized_adapter (PerFL) và task > 0: chỉ nạp các key chia sẻ
    (fc + convnet gốc đã đông cứng). GIỮ NGUYÊN toàn bộ nhánh cá nhân hóa của
    client: adapter, gate, và stability_encoder (vốn được hợp nhất từ adapter
    cá nhân hóa của các task trước nên cũng khác nhau giữa các client).
    """
    if task > 0 and args.get("personalized_adapter", False):
        own_state = local_model._network.state_dict()
        for k in own_state.keys():
            if k in global_state and not any(m in k for m in _PERSONALIZED_KEY_MARKERS):
                own_state[k] = global_state[k]
        local_model._network.load_state_dict(own_state)
    else:
        local_model._network.load_state_dict(global_state)


def _eval_model_metrics(model, loader):
    """Đánh giá CNN head của model trên loader, trả về dict metrics."""
    model._network.eval()
    y_pred, y_true, loss = model._eval_cnn(loader)
    return model._evaluate(y_pred, y_true, loss=loss)


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train_federated(args)


def _train_federated(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]

    timestamp = datetime.now().strftime("%d-%m-%y_%H-%M")
    run_dir = os.path.join(
        "logs",
        args["model_name"] + "_federated",
        args["dataset"],
        "{}_seed{}_{}_clients{}".format(
            timestamp, args["seed"], args["convnet_type"], args["num_clients"]
        ),
    )
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    logfilename = os.path.join(run_dir, "training.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[logging.FileHandler(filename=logfilename), logging.StreamHandler(sys.stdout)],
    )

    csv_path = os.path.join(run_dir, "metrics_round_by_round.csv")
    csv_file = open(csv_path, "a" if args.get("resume") else "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if not args.get("resume"):
        csv_writer.writerow([
            "task", "round", "global_round", "method",
            "acc", "prec_mic", "prec_mac", "prec_wei",
            "rec_mic", "rec_mac", "rec_wei",
            "f1_mic", "f1_mac", "f1_wei", "loss", "avg_acc",
        ])

    # CSV per-client (PerFL): F1 của model cá nhân hóa từng client + personalization gain
    per_client_csv_path = os.path.join(run_dir, "metrics_per_client.csv")
    per_client_file = open(per_client_csv_path, "a" if args.get("resume") else "w", newline="", encoding="utf-8")
    per_client_writer = csv.writer(per_client_file)
    if not args.get("resume"):
        per_client_writer.writerow([
            "task", "round", "client", "acc", "f1_macro", "f1_weighted",
            "global_acc", "global_f1_macro", "gain_acc", "gain_f1_macro",
        ])

    _set_random()
    _set_device(args)
    print_args(args)
    logging.info("Run directory: {}".format(run_dir))
    args["run_dir"] = run_dir

    # Giới hạn mẫu (tùy chọn) cho dataset CAN-IoV trước khi load
    if args["dataset"] == "can_iov":
        from utils import data_can_iov
        if args.get("max_samples_per_class"):
            data_can_iov.MAX_SAMPLES_PER_CLASS = int(args["max_samples_per_class"])
            logging.info(f"[can_iov] MAX_SAMPLES_PER_CLASS = {data_can_iov.MAX_SAMPLES_PER_CLASS}")
        if args.get("test_max_samples_per_class"):
            data_can_iov.TEST_MAX_SAMPLES_PER_CLASS = int(args["test_max_samples_per_class"])
            logging.info(f"[can_iov] TEST_MAX_SAMPLES_PER_CLASS = {data_can_iov.TEST_MAX_SAMPLES_PER_CLASS}")

    logging.info(f"Initializing DataManagers for {args['num_clients']} clients...")
    client_dms = []
    for c in range(args["num_clients"]):
        dm = DataManager(
            args["dataset"],
            args["shuffle"],
            args["seed"],
            args["init_cls"],
            args["increment"],
            client_id=c,
            class_order=args.get("class_order"),
            task_increments=args.get("task_increments"),
        )
        if args.get("debug"):
            dm._train_data = dm._train_data[:2000]
            dm._train_targets = dm._train_targets[:2000]
        client_dms.append(dm)

    nb_tasks = client_dms[0].nb_tasks
    # max_tasks: dừng sớm sau N task đầu (mặc định None = chạy hết). Dùng để
    # chạy thí nghiệm ablation ngắn trên task 0 mà không đụng tới dữ liệu hay
    # task_increments (vốn bắt buộc phải cộng đủ số lớp).
    _max_tasks = args.get("max_tasks")
    _nb_tasks_full = nb_tasks          # so task THAT SU cua chuong trinh hoc
    if _max_tasks:
        nb_tasks = min(nb_tasks, int(_max_tasks))
        logging.info(f"max_tasks={_max_tasks} -> chi chay {nb_tasks} task dau")
    logging.info(f"Task increments: {client_dms[0]._increments}")

    global_model = factory.get_model(args["model_name"], args)
    local_models = [factory.get_model(args["model_name"], args) for _ in range(args["num_clients"])]

    start_task = 0
    start_round = 0
    results_all = []

    # ── Lịch sử metrics để vẽ biểu đồ ──────────────────────────────────────
    history = {
        "cnn":  {"acc": [], "precision": [], "recall": [], "f1": []},
        "nme":  {"acc": [], "precision": [], "recall": [], "f1": []},
    }
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}

    checkpoint = None
    if args.get("resume"):
        if os.path.isfile(args["resume"]):
            logging.info(f"==> Resuming from checkpoint: {args['resume']}")
            checkpoint = torch.load(args["resume"], map_location='cpu', weights_only=False)
            start_task = checkpoint['task']
            if checkpoint.get('is_memory_phase', False):
                start_round = args["num_rounds"]
            else:
                start_round = checkpoint['round'] + 1
                if start_round >= args["num_rounds"]:
                    start_round = args["num_rounds"]
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {args['resume']}")
            
    for task in range(nb_tasks):
        # Save previous global network BEFORE expansion. AFSIC KD needs this old model.
        prev_global_network = copy.deepcopy(global_model._network) if _is_afsic(args) and task > 0 else None

        # 1. Mở rộng kiến trúc (nhưng không train) để lấy đúng kích thước mô hình
        global_model.incremental_train(client_dms[0], skip_train=True)
        global_model._network.to(args["device"][0])

        for c in range(args["num_clients"]):
            local_models[c].skip_rehearsal = True
            local_models[c].incremental_train(client_dms[c], skip_train=True)
            local_models[c].skip_rehearsal = False

        if checkpoint is not None and task == checkpoint['task']:
            logging.info(f"Phục hồi trạng thái cho Task {task} từ Checkpoint...")
            global_model._network.load_state_dict(checkpoint['model_state_dict'])
            if _is_afsic(args) and checkpoint.get('global_proto_memory') is not None:
                global_model.global_proto_memory = checkpoint['global_proto_memory']
            _restored_net = 0
            for c in range(args["num_clients"]):
                c_state = checkpoint['client_states'][c]
                local_models[c]._data_memory = c_state.get('data_memory')
                local_models[c]._targets_memory = c_state.get('targets_memory')
                if _is_afsic(args) and c_state.get('local_memory') is not None:
                    local_models[c].local_memory = c_state['local_memory']
                # Phục hồi nhánh cá nhân hóa của client (xem chú thích ở chỗ lưu).
                if c_state.get('net') is not None:
                    try:
                        local_models[c]._network.load_state_dict(c_state['net'], strict=False)
                        local_models[c]._network.to(args["device"][0])
                        _restored_net += 1
                    except Exception as e:
                        logging.warning(f"Không nạp được trọng số client {c}: {e}")
            if _restored_net:
                logging.info(f"Đã phục hồi trọng số riêng cho {_restored_net}/{args['num_clients']} client.")
            else:
                logging.warning(
                    "CHECKPOINT CŨ — không có trọng số client. Nhánh cá nhân hóa "
                    "(stability_encoder/adapter/gate) sẽ là khởi tạo ngẫu nhiên. "
                    "Chỉ nên resume từ checkpoint task 0 (ckpt_round0030)."
                )

        for c in range(args["num_clients"]):
            if _is_afsic(args) and task > 0:
                local_models[c].train_loader = None
                continue
            else:
                train_dataset = None
                has_file = True
                _fed_dir = None
                if args["dataset"] == "can_iov":
                    from utils.data_can_iov import _FEDERATED_DIR as _fed_dir
                if _fed_dir is not None:
                    task_file = os.path.join(_fed_dir, f"client_{c}_task_{task+1}.pt")
                    if not os.path.exists(task_file):
                        has_file = False
                        logging.info(f"Client {c} không có file {os.path.basename(task_file)}. Tự động bỏ qua.")
                
                if has_file:
                    train_dataset = _build_standard_train_dataset(client_dms[c], local_models[c], args)
            # Ngăn chặn Model Collapse: Nếu client chỉ có Rehearsal Memory mà không có data mới, thì BỎ QUA không train
            has_new_data = (
                train_dataset is not None
                and getattr(train_dataset, 'len_new_data', getattr(train_dataset, 'len_source', len(train_dataset))) > 0
            )
            
            if train_dataset is not None and len(train_dataset) > 0 and has_new_data:
                local_models[c].train_loader = make_loader(
                    train_dataset, batch_size=args["batch_size"], shuffle=True
                )
            else:
                local_models[c].train_loader = None

        if task < start_task:
            for c in range(args["num_clients"]):
                local_models[c].after_task()
            global_model.after_task()
            continue

        logging.info(f"========== Bắt đầu Task {task} ==========")
        current_start_round = start_round if task == start_task else 0
        # Stage t >= 1: Supervised task-incremental registration and prototype initialization
        if _is_afsic(args) and task > 0:
            logging.info("Supervised task-incremental registration: computing prototypes from labeled task data.")
            client_protos = []
            for c in range(args["num_clients"]):
                _load_global_into_client(local_models[c], global_model._network.state_dict(), task, args)
                local_models[c]._network.to(args["device"][0])
                old_protos = local_models[c].compute_local_prototypes(
                    client_dms[c],
                    class_ids=range(local_models[c]._known_classes),
                    max_samples_per_class=args.get("proto_max_samples"),
                    seed=args.get("seed", 0) + c,
                    report_full_count=True,   # cat mau chi de ĐO
                )
                max_samples = (args.get("kshot", 10) if args.get("fewshot_enabled", True)
                               else args.get("proto_max_samples"))
                new_protos = local_models[c].compute_local_prototypes(
                    client_dms[c],
                    class_ids=range(local_models[c]._known_classes, local_models[c]._total_classes),
                    max_samples_per_class=max_samples,
                    seed=args.get("seed", 0) + task * 1009 + c,
                    report_full_count=(max_samples is not None
                                       and max_samples == args.get("proto_max_samples")),
                )
                protos = {}
                protos.update(old_protos)
                protos.update(new_protos)
                client_protos.append(protos)

            _aggregate_client_prototypes(global_model, client_protos, args["num_clients"], args=args)
            
            for c in range(args["num_clients"]):
                local_models[c].global_proto_memory = copy.deepcopy(global_model.global_proto_memory)
                _calibrate_classifier_from_prototypes(local_models[c], is_task_init=True)
            
            _calibrate_classifier_from_prototypes(global_model, is_task_init=True)
            logging.info("Class classifier weights successfully initialized from prototypes.")

            for c in range(args["num_clients"]):
                train_dataset = _build_fewshot_train_dataset(
                    client_dms[c],
                    local_models[c],
                    args,
                    task,
                    c
                )
                has_new_data = (
                    train_dataset is not None
                    and getattr(train_dataset, 'len_new_data', getattr(train_dataset, 'len_source', len(train_dataset))) > 0
                )
                if train_dataset is not None and len(train_dataset) > 0 and has_new_data:
                    local_models[c].train_loader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=0
                    )
                else:
                    local_models[c].train_loader = None

        # Keep track of best accuracies for forgetting calculations
        if 'best_accs' not in locals():
            best_accs = {}

        for round_idx in range(current_start_round, args["num_rounds"]):
            global_round = task * args["num_rounds"] + round_idx
            # LR decay theo round: client đọc current_round trong _train
            args["current_round"] = round_idx
            _lr_eff = args.get("lr", 0.001) * (
                args.get("gamma", 0.1) ** sum(1 for m in args.get("milestones", []) if round_idx >= m)
            )
            logging.info(
                f"--- Task {task} | Round {round_idx+1}/{args['num_rounds']} "
                f"(Global {global_round+1}) | lr={_lr_eff:g} ---"
            )
            client_weights = []
            client_accs = []
            client_protos = []
            
            global_state_round_start = copy.deepcopy(global_model._network.state_dict())
            
            for c in range(args["num_clients"]):
                if local_models[c].train_loader is None: 
                    client_accs.append(0.0)
                    client_protos.append({})
                    continue
                
                _load_global_into_client(local_models[c], global_model._network.state_dict(), task, args)
                local_models[c]._network.to(args["device"][0])

                if _is_afsic(args):
                    local_models[c].global_proto_memory = copy.deepcopy(global_model.global_proto_memory)
                    if task > 0 and prev_global_network is not None:
                        local_models[c]._old_network = copy.deepcopy(prev_global_network).to(args["device"][0])
                        local_models[c]._old_network.eval()
                        for p in local_models[c]._old_network.parameters():
                            p.requires_grad = False
                
                local_models[c].args["epochs"] = args["local_epochs"]
                local_models[c].args["start_round"] = 0
                local_models[c]._train(local_models[c].train_loader, None)
                
                client_weights.append(copy.deepcopy(local_models[c]._network.state_dict()))

                if _is_afsic(args):
                    # Fast eval for client quality validation Acc
                    local_models[c]._network.eval()
                    quality_loader = _build_local_eval_loader(
                        client_dms[c], local_models[c]._total_classes, args["batch_size"],
                        max_samples=args.get("quality_eval_max_samples"),
                        seed=args.get("seed", 0) + c)
                    if quality_loader is not None:
                        test_acc_dict = local_models[c]._compute_accuracy(local_models[c]._network, quality_loader)
                        client_accs.append(test_acc_dict["total"] / 100.0)
                    else:
                        client_accs.append(0.0)
                    
                    # Compute local prototypes
                    # proto_max_samples: giới hạn số mẫu mỗi lớp CŨ khi tính
                    # prototype. Prototype là trung bình của feature đã chuẩn
                    # hóa L2 nên n=20.000 đã cho sai số lấy mẫu ~0,7%; trong
                    # khi để None thì mỗi round phải duyệt toàn bộ dữ liệu lớp
                    # cũ (client 0 có 29,2 triệu mẫu Benign = 3.565 batch),
                    # chiếm ~83% thời gian một round. Mặc định None = giữ
                    # nguyên hành vi cũ.
                    old_protos = local_models[c].compute_local_prototypes(
                        client_dms[c],
                        class_ids=range(local_models[c]._known_classes),
                        max_samples_per_class=args.get("proto_max_samples"),
                        seed=args.get("seed", 0) + c,
                        report_full_count=True,   # cat mau chi de ĐO
                    )
                    # TOI UU: o task 0 (va moi khi khong bat few-shot) ban cu
                    # truyen None -> duyet TOAN BO du lieu cua moi lop moi, cua
                    # MOI client, MOI round, chi de tinh mot vector trung binh.
                    # Prototype la mean cua feature da chuan hoa L2 nen n=20.000
                    # da cho sai so lay mau ~0,7% (xem ghi chu proto_max_samples
                    # o nhanh old_protos). Day la lay mau khi ĐO, khong phai cat
                    # du lieu huan luyen — client van train tren toan bo du lieu.
                    # Mac dinh None = giu nguyen hanh vi cu.
                    _new_cap = (args.get("kshot", 10)
                                if task > 0 and args.get("fewshot_enabled", True)
                                else args.get("proto_max_samples"))
                    new_protos = local_models[c].compute_local_prototypes(
                        client_dms[c],
                        class_ids=range(local_models[c]._known_classes, local_models[c]._total_classes),
                        max_samples_per_class=_new_cap,
                        seed=args.get("seed", 0) + task * 1009 + c,
                        # True chi khi cap den TU proto_max_samples (do luong).
                        # Voi kshot/1% thi cat mau la THAT -> giu False.
                        report_full_count=(_new_cap is not None
                                           and _new_cap == args.get("proto_max_samples")),
                    )
                    protos = {}
                    protos.update(old_protos)
                    protos.update(new_protos)
                    client_protos.append(protos)
                else:
                    client_accs.append(0.0)
                    client_protos.append({})
            
            if client_weights:
                if _is_afsic(args):
                    # a. Update global prototypes on the server
                    _aggregate_client_prototypes(global_model, client_protos, args["num_clients"], args=args)

                    for c in range(args["num_clients"]):
                        local_models[c].global_proto_memory = copy.deepcopy(global_model.global_proto_memory)

                    # b. Compute quality scores and aggregation weights
                    active_client_indices = [c for c in range(args["num_clients"]) if local_models[c].train_loader is not None]

                    alpha, accepted_positions, Q_list, agg_stats = compute_aggregation_weights(
                        args=args,
                        global_model=global_model,
                        client_accs=client_accs,
                        client_protos=client_protos,
                        client_weights=client_weights,
                        global_state_round_start=global_state_round_start,
                        active_client_indices=active_client_indices,
                        task=task
                    )

                    active_client_indices = [active_client_indices[pos] for pos in accepted_positions]
                    client_weights_accepted = [client_weights[pos] for pos in accepted_positions]

                    for idx, c in enumerate(active_client_indices):
                        logging.info(f"Client {c} Aggregation Weight alpha_i: {alpha[idx]:.4f}")

                    # Per-class reliability-aware prototype aggregation với đầy đủ
                    # thống kê per-client: q (alpha), drift, update_norm
                    client_stats = {
                        c: {
                            "q": alpha[idx],
                            "drift": agg_stats["drift"][accepted_positions[idx]],
                            "update_norm": agg_stats["update_norm"][accepted_positions[idx]],
                        }
                        for idx, c in enumerate(active_client_indices)
                    }
                    _aggregate_client_prototypes(global_model, client_protos, args["num_clients"], args=args, client_stats=client_stats)
                    for c in range(args["num_clients"]):
                        local_models[c].global_proto_memory = copy.deepcopy(global_model.global_proto_memory)

                    # Perform quality-aware weighted aggregation
                    global_dict = copy.deepcopy(global_model._network.state_dict())
                    aggregate_backbone = args.get("aggregate_backbone", False)

                    # Chuẩn hoá theo SỐ BƯỚC cục bộ (FedNova, Wang et al. 2020).
                    #
                    # Gộp thường:  θ ← Σ αᵢ·θᵢ            = θ₀ + Σ αᵢ·Δθᵢ
                    # Chuẩn hoá :  θ ← θ₀ + τ̄ · Σ αᵢ·Δθᵢ/τᵢ ,  τ̄ = Σ αᵢ·τᵢ
                    #
                    # Chia cho τᵢ bỏ đi phần "client lớn đi xa hơn chỉ vì chạy
                    # nhiều bước hơn"; nhân lại τ̄ giữ nguyên độ lớn bước tổng
                    # nên tốc độ học không bị bóp. KHÔNG bỏ mẫu nào — mọi client
                    # vẫn train trên toàn bộ dữ liệu của mình.
                    #
                    # Mặc định TẮT để mọi config cũ giữ nguyên hành vi.
                    normalize_steps = args.get("normalize_local_steps", False)
                    tau_l = None
                    if normalize_steps and "tau_local" in agg_stats:
                        tau_l = [max(1, agg_stats["tau_local"][accepted_positions[i]])
                                 for i in range(len(alpha))]
                        tau_bar = sum(alpha[i] * tau_l[i] for i in range(len(alpha)))
                        logging.info(
                            f"normalize_local_steps: tau_i {min(tau_l)}..{max(tau_l)} "
                            f"| tau_bar={tau_bar:.1f}"
                        )

                    # ── P2: gộp THỐNG KÊ BatchNorm theo SỐ MẪU ──────────
                    # running_mean/running_var là ước lượng của phân phối dữ
                    # liệu, không phải biến tối ưu. Gộp chúng bằng alpha (gần
                    # đều trên 50 client) cho ra thống kê không phản ánh phân
                    # phối thật: client 1 batch chỉ dịch 10% về dữ liệu riêng
                    # (momentum 0.1) còn client 1.543 batch hội tụ hẳn, rồi hai
                    # loại đó bị trung bình ngang nhau.
                    #
                    # Cân theo n_i cho ước lượng không chệch. Trọng số mô hình
                    # vẫn gộp theo alpha (chất lượng) như cũ.
                    # Tham chiếu cho biến thể mạnh hơn: FedBN, Li et al. ICLR 2021.
                    #
                    # Mặc định TẮT để mọi config cũ giữ nguyên hành vi.
                    bn_by_count = args.get("bn_stats_by_count", False)
                    w_bn = None
                    if bn_by_count and "n_samples" in agg_stats:
                        ns = [max(1, agg_stats["n_samples"][accepted_positions[i]])
                              for i in range(len(alpha))]
                        tot = float(sum(ns))
                        w_bn = [n / tot for n in ns]
                        logging.info(
                            f"bn_stats_by_count: n_i {min(ns)}..{max(ns)} "
                            f"| w_bn {min(w_bn):.2e}..{max(w_bn):.4f}"
                        )

                    for k in global_dict.keys():
                        if not is_aggregated_state_key(k, task, aggregate_backbone,
                                       args.get("plastic_source_trainable", False)):
                            continue
                        # num_batches_tracked: trung bình có trọng số của một bộ
                        # đếm là vô nghĩa; lấy max cho đúng ngữ nghĩa "đã thấy
                        # bao nhiêu batch".
                        if not global_dict[k].is_floating_point():
                            if bn_by_count and client_weights_accepted:
                                global_dict[k] = torch.stack(
                                    [w[k] for w in client_weights_accepted]
                                ).max(dim=0).values.to(global_dict[k].dtype)
                            continue
                        if w_bn is not None and ("running_mean" in k or "running_var" in k):
                            val = client_weights_accepted[0][k].float() * w_bn[0]
                            for c_idx in range(1, len(client_weights_accepted)):
                                val += client_weights_accepted[c_idx][k].float() * w_bn[c_idx]
                            global_dict[k] = val.to(global_dict[k].dtype)
                            continue
                        if tau_l is not None and k in global_state_round_start:
                            base = global_state_round_start[k].float()
                            acc = torch.zeros_like(base)
                            for c_idx in range(len(client_weights_accepted)):
                                acc += (client_weights_accepted[c_idx][k].float() - base) \
                                       * (alpha[c_idx] / tau_l[c_idx])
                            val = base + tau_bar * acc
                        else:
                            val = client_weights_accepted[0][k].float() * alpha[0]
                            for c_idx in range(1, len(client_weights_accepted)):
                                val += client_weights_accepted[c_idx][k].float() * alpha[c_idx]
                        global_dict[k] = val.to(global_dict[k].dtype)
                    global_model._network.load_state_dict(global_dict)
                    _calibrate_classifier_from_prototypes(global_model)
                    _log_classifier_separability(global_model)
                else:
                    global_weights = average_weights(client_weights)
                    global_model._network.load_state_dict(global_weights)

            # ── Đánh giá Global Model cuối MỖI ROUND ──
            global_model.test_loader = _build_global_learned_test_loader(
                client_dms[0],
                global_model._total_classes,
                args["batch_size"],
                args_eval_cap=args.get("eval_max_per_class"),
            )
            if global_model.test_loader is None:
                logging.warning("Global learned-class test set is empty; skipping evaluation for this round.")
                continue
            logging.info(
                f"Evaluating on global_test_data filtered to learned classes: "
                f"0-{global_model._total_classes - 1}"
            )
            
            cnn_accy, nme_accy, y_pred, y_true = global_model.eval_task()
            
            results_all.append(cnn_accy)
            avg_acc = sum(r['top1'] for r in results_all) / len(results_all)
            
            # Calculate parameter growth and communication cost
            total_params = sum(p.numel() for p in global_model._network.parameters())
            trainable_params = sum(p.numel() for p in global_model._network.parameters() if p.requires_grad)
            comm_cost = trainable_params * 4 / (1024 * 1024) # MB
            
            # Forgetting theo đúng ranh giới task (dùng task_increments thay vì
            # group keys 10 lớp cố định vốn không khớp với increments lệch nhau)
            y_top1 = y_pred[:, 0] if y_pred.ndim > 1 else y_pred.flatten()
            task_bounds = np.cumsum([0] + list(client_dms[0]._increments))
            forgetting_vals = []
            for t in range(task):
                lo, hi = task_bounds[t], task_bounds[t + 1]
                idxes = np.where((y_true >= lo) & (y_true < hi))[0]
                if len(idxes) == 0:
                    continue
                curr_acc = float((y_top1[idxes] == y_true[idxes]).mean() * 100)
                best_key = f"best_task_{t}"
                best_accs[best_key] = max(best_accs.get(best_key, curr_acc), curr_acc)
                forgetting_vals.append(max(0.0, best_accs[best_key] - curr_acc))
            mean_forgetting = sum(forgetting_vals) / len(forgetting_vals) if forgetting_vals else 0.0

            logging.info(
                f"[Task {task} | Round {round_idx+1}] "
                f"Acc: {cnn_accy['top1']:.2f}% | "
                f"Old Acc: {cnn_accy.get('grouped', {}).get('old_acc', 0):.2f}% | "
                f"New Acc: {cnn_accy.get('grouped', {}).get('new_acc', 0):.2f}% | "
                f"Forgetting: {mean_forgetting:.2f}% | "
                f"Params: {total_params:,} | Comm Cost: {comm_cost:.2f} MB"
            )

            # Ghi file CSV
            csv_writer.writerow([
                task, round_idx + 1, global_round + 1, str(args["model_name"]).upper(),
                round(cnn_accy["top1"], 4),
                round(cnn_accy.get("precision_micro", 0), 4),
                round(cnn_accy.get("precision_macro", 0), 4),
                round(cnn_accy.get("precision_weighted", 0), 4),
                round(cnn_accy.get("recall_micro", 0), 4),
                round(cnn_accy.get("recall_macro", 0), 4),
                round(cnn_accy.get("recall_weighted", 0), 4),
                round(cnn_accy.get("f1_micro", 0), 4),
                round(cnn_accy.get("f1_macro", 0), 4),
                round(cnn_accy.get("f1_weighted", 0), 4),
                round(cnn_accy.get("loss", 0), 6),
                round(avg_acc, 4),
            ])
            csv_file.flush()

            # ── Đánh giá per-client (PerFL): personalized model vs global model ──
            eval_every = int(args.get("per_client_eval_every", 0))  # 0 = chỉ round cuối task
            do_pc_eval = _is_afsic(args) and args.get("per_client_eval", True) and (
                round_idx == args["num_rounds"] - 1
                or (eval_every > 0 and (round_idx + 1) % eval_every == 0)
            )
            if do_pc_eval:
                for c in range(args["num_clients"]):
                    if local_models[c].train_loader is None:
                        continue
                    # Model cá nhân hóa: phần chia sẻ lấy từ global sau aggregation,
                    # adapter/gate giữ bản cục bộ, classifier calibrate theo
                    # personalized prototypes của chính client.
                    _load_global_into_client(local_models[c], global_model._network.state_dict(), task, args)
                    _calibrate_classifier_from_prototypes(local_models[c])
                    pc_metrics = _eval_model_metrics(local_models[c], global_model.test_loader)
                    gain_acc = float(pc_metrics["top1"]) - float(cnn_accy["top1"])
                    gain_f1 = float(pc_metrics.get("f1_macro", 0)) - float(cnn_accy.get("f1_macro", 0))
                    logging.info(
                        f"[PerFL] Client {c}: Acc {pc_metrics['top1']:.2f}% (gain {gain_acc:+.2f}) | "
                        f"F1-mac {pc_metrics.get('f1_macro', 0):.2f}% (gain {gain_f1:+.2f})"
                    )
                    per_client_writer.writerow([
                        task, round_idx + 1, c,
                        round(float(pc_metrics["top1"]), 4),
                        round(float(pc_metrics.get("f1_macro", 0)), 4),
                        round(float(pc_metrics.get("f1_weighted", 0)), 4),
                        round(float(cnn_accy["top1"]), 4),
                        round(float(cnn_accy.get("f1_macro", 0)), 4),
                        round(gain_acc, 4),
                        round(gain_f1, 4),
                    ])
                per_client_file.flush()

            if round_idx == args["num_rounds"] - 1:
                cnn_curve["top1"].append(cnn_accy["top1"])
                cnn_curve["top5"].append(cnn_accy["top5"])
                history["cnn"]["acc"].append(cnn_accy["top1"])
                history["cnn"]["precision"].append(cnn_accy.get("precision_macro", 0))
                history["cnn"]["recall"].append(cnn_accy.get("recall_macro", 0))
                history["cnn"]["f1"].append(cnn_accy.get("f1_macro", 0))
                
                if nme_accy is not None:
                    nme_curve["top1"].append(nme_accy["top1"])
                    nme_curve["top5"].append(nme_accy["top5"])
                    history["nme"]["acc"].append(nme_accy["top1"])
                    history["nme"]["precision"].append(nme_accy.get("precision_macro", 0))
                    history["nme"]["recall"].append(nme_accy.get("recall_macro", 0))
                    history["nme"]["f1"].append(nme_accy.get("f1_macro", 0))
                
                plot_confusion_matrix(y_true, y_pred, task, run_dir)

            # Lưu Checkpoint mỗi Round
            #
            # ckpt_every_n_rounds: 0 = KHONG luu. Moi checkpoint chua state cua
            # CA 100 client (~52 MB), 30 round = ~1,5 GB. Khi sang loc ablation
            # task 0 thi khong ai resume tu day ca.
            _cke = args.get("ckpt_every_n_rounds", 1)
            if _cke and (round_idx + 1) % int(_cke) == 0:
              client_states = []
              for c in range(args["num_clients"]):
                client_states.append({
                    'data_memory': getattr(local_models[c], '_data_memory', None),
                    'targets_memory': getattr(local_models[c], '_targets_memory', None),
                    'local_memory': getattr(local_models[c], 'local_memory', None),
                    # Trọng số mạng riêng của client. Bắt buộc phải lưu: nhánh
                    # cá nhân hóa (stability_encoder / plasticity_adapter / gate)
                    # nằm trong _PERSONALIZED_KEY_MARKERS nên KHÔNG bao giờ được
                    # nạp đè từ global ở task >= 1. Thiếu nó, resume giữa chừng
                    # sẽ để client chạy tiếp với nhánh cá nhân hóa khởi tạo
                    # ngẫu nhiên (loss round 1 task 1 nhảy 1.6 -> 3.3).
                    'net': {k: v.cpu() for k, v in local_models[c]._network.state_dict().items()},
                })
              ckpt_name = f'ckpt_round{global_round+1:04d}_task{task:02d}_r{round_idx+1:03d}_acc{cnn_accy["top1"]:.1f}.pth'
              torch.save({
                'task': task,
                'round': round_idx,
                'global_round': global_round,
                'model_state_dict': global_model._network.state_dict(),
                'known_classes': global_model._known_classes,
                'client_states': client_states,
                'global_proto_memory': getattr(global_model, 'global_proto_memory', None),
                'metrics': cnn_accy
              }, os.path.join(ckpt_dir, ckpt_name))

        # Cuối Task, xây dựng lại bộ nhớ Rehearsal
        #
        # [ĐO] Pha nay ton 2 GIO 6 PHUT va 6,6 GB cho MOT lan chay task 0
        # (log 29-08: round 30 xong luc 15:17, "Training Finished" luc 17:23).
        # Ly do: voi MOI client c no luu mot checkpoint chua state cua CA 100
        # client -> O(n^2). Cong them herding tren lop Benign 29 trieu mau.
        #
        # Bo nho nay chi duoc DUNG o task SAU. O task cuoi cung duoc chay
        # (max_tasks, hoac task cuoi that su) no khong bao gio duoc doc.
        # Voi ablation task 0 thi day la 40% thoi gian va toan bo dung luong,
        # tieu vao thu khong ai dung.
        # CHI bo qua pha memory khi day la task CUOI CUNG CUA CHUONG TRINH HOC
        # (task 4) — luc do khong con task sau nao doc bo nho nua.
        #
        # Ban cu dung (task >= nb_tasks - 1) voi nb_tasks DA bi max_tasks thu
        # nho, nen mot config kieu "chi chay task 0 roi ban giao checkpoint cho
        # session sau" (afsic-iov-task0.json, max_tasks=1) se bi bo qua pha
        # memory va KHONG sinh ra ckpt_task00_memory_client99.pth — dung file ma
        # session task 1..4 can de resume.
        _het_chuong_trinh = (task >= _nb_tasks_full - 1)
        if args.get("skip_memory_phase", False) or _het_chuong_trinh:
            logging.info(
                f"BO QUA pha Rehearsal Memory o task {task} (task cuoi duoc chay) "
                f"— tiet kiem ~2 gio va ~6,6 GB. Dat skip_memory_phase=false neu "
                f"can checkpoint de chay tiep task sau.")
            for c in range(args["num_clients"]):
                local_models[c].after_task()
            global_model.after_task()
            break

        logging.info(f"Xây dựng Rehearsal Memory cho các Clients tại cuối Task {task}...")
        current_client_start = checkpoint.get('last_client_done', -1) + 1 if (checkpoint is not None and task == checkpoint['task']) else 0
        # force_rebuild_memory: dung lai bo nho exemplar tu dau du checkpoint bao
        # da xong (last_client_done=99). Can khi DOI CHINH SACH replay
        # (memory_ratio -> memory_size chia deu) ma khong muon chay lai ca task 0.
        if args.get("force_rebuild_memory", False):
            logging.info("force_rebuild_memory=True -> dung lai exemplar cho ca 100 client "
                         "theo chinh sach replay MOI (bo qua last_client_done cua checkpoint).")
            current_client_start = 0
        
        for c in range(args["num_clients"]):
            if c >= current_client_start:
                if local_models[c].train_loader is not None:
                    _load_global_into_client(local_models[c], global_model._network.state_dict(), task, args)
                    local_models[c]._network.to(args["device"][0])
                    try:
                        local_models[c].build_rehearsal_memory(client_dms[c], local_models[c].samples_per_class)
                    except Exception as e:
                        logging.warning(f"Lỗi khi build memory cho client {c}: {e}")
                
                # ===== LƯU CHECKPOINT SAU KHI CLIENT C TẠO XONG MEMORY =====
                #
                # memory_ckpt_per_client (mac dinh True = hanh vi cu):
                #   True  — luu sau MOI client. An toan tuyet doi neu bi ngat giua
                #           pha memory, nhung la O(n^2): moi lan luu chua state cua
                #           CA 100 client. [DO] 2 gio va 6,6 GB cho mot task.
                #   False — chi luu MOT LAN sau khi ca 100 client xong. Resume van
                #           duoc o muc TASK (du cho lich chay nhieu session), chi
                #           mat kha nang resume GIUA pha memory.
                # Voi lan chay xac nhan 5 task: tiet kiem ~4 gio, gan mot session.
                _mem_ck = args.get("memory_ckpt_per_client", True)
                if _mem_ck or c == args["num_clients"] - 1:
                  client_states = []
                  for state_c in range(args["num_clients"]):
                    client_states.append({
                        'data_memory': getattr(local_models[state_c], '_data_memory', None),
                        'targets_memory': getattr(local_models[state_c], '_targets_memory', None),
                        'local_memory': getattr(local_models[state_c], 'local_memory', None),
                        'net': {k: v.cpu() for k, v in local_models[state_c]._network.state_dict().items()},
                    })
                    
                  ckpt_name = f'ckpt_task{task:02d}_memory_client{c:02d}.pth'
                  torch.save({
                    'task': task,
                    'round': args["num_rounds"] - 1,
                    'global_round': checkpoint['global_round'] if (checkpoint is not None and 'global_round' in checkpoint) else (task * args["num_rounds"] + args["num_rounds"] - 1),
                    'is_memory_phase': True,
                    'last_client_done': c,
                    'model_state_dict': global_model._network.state_dict(),
                    'client_states': client_states,
                    'known_classes': global_model._known_classes,
                    'global_proto_memory': getattr(global_model, 'global_proto_memory', None)
                  }, os.path.join(ckpt_dir, ckpt_name))
                  logging.info(f"Da luu Checkpoint Memory (client {c})")
                
            local_models[c].after_task()

        global_model.after_task()

    csv_file.close()
    per_client_file.close()

    # ── Vẽ biểu đồ ──────────────────────────────────────────────────────────
    _plot_metrics(history, run_dir, args)
    logging.info("Plots saved in: {}".format(run_dir))
    
    logging.info("Training Finished.")


def run_test(args):
    """
    Chế độ TEST: Tải các checkpoint và đánh giá toàn bộ.
    """
    _set_random()
    _set_device(args)
    
    test_ckpt_root = args.get("test_checkpoint_dir", "")
    if not test_ckpt_root or not os.path.exists(test_ckpt_root):
        logging.error(f"[TEST] Thư mục checkpoint không hợp lệ: {test_ckpt_root}")
        return

    ckpt_files = sorted(glob.glob(os.path.join(test_ckpt_root, "checkpoints", "ckpt_round*.pth")))
    if not ckpt_files:
        logging.error(f"[TEST] Không tìm thấy checkpoint nào trong {test_ckpt_root}/checkpoints/")
        return
        
    logging.info(f"[TEST] Tìm thấy {len(ckpt_files)} checkpoint. Bắt đầu đánh giá...")
    
    # Init DataManager cho Client 0 để lấy Test Set chung
    dm = DataManager(
        args["dataset"],
        False,
        args["seed"],
        args["init_cls"],
        args["increment"],
        client_id=0,
        class_order=args.get("class_order"),
        task_increments=args.get("task_increments"),
    )
    
    csv_path = os.path.join(test_ckpt_root, "test_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "checkpoint", "task", "round", "global_round", "acc", 
            "prec_mic", "prec_mac", "prec_wei", 
            "rec_mic", "rec_mac", "rec_wei", 
            "f1_mic", "f1_mac", "f1_wei"
        ])

        global_model = factory.get_model(args["model_name"], args)
        
        for idx, cp in enumerate(ckpt_files):
            state = torch.load(cp, map_location='cpu', weights_only=False)
            task = state['task']
            
            # Cập nhật kiến trúc Model theo số task
            global_model = factory.get_model(args["model_name"], args)
            for _ in range(task + 1):
                global_model.incremental_train(dm, skip_train=True)
            
            global_model._network.load_state_dict(state['model_state_dict'])
            global_model._network.to(args["device"][0])
            global_model._network.eval()
            
            # --mode test la duong lay CON SO BAO CAO -> khong bao gio cat tap test.
            global_model.test_loader = _build_global_learned_test_loader(
                dm,
                global_model._total_classes,
                args["batch_size"],
            )
            if global_model.test_loader is None:
                logging.warning(f"[TEST] Empty learned-class global test set for checkpoint: {os.path.basename(cp)}")
                continue
            
            cnn_accy, _, y_pred, y_true = global_model.eval_task()
            
            logging.info(f"[TEST] {os.path.basename(cp)} | Task {task} | Acc: {cnn_accy['top1']:.2f}% | F1-Mac: {cnn_accy.get('f1_macro', 0):.2f}%")
            
            writer.writerow([
                os.path.basename(cp), task, state['round'], state['global_round'],
                round(cnn_accy["top1"], 4),
                round(cnn_accy.get("precision_micro", 0), 4),
                round(cnn_accy.get("precision_macro", 0), 4),
                round(cnn_accy.get("precision_weighted", 0), 4),
                round(cnn_accy.get("recall_micro", 0), 4),
                round(cnn_accy.get("recall_macro", 0), 4),
                round(cnn_accy.get("recall_weighted", 0), 4),
                round(cnn_accy.get("f1_micro", 0), 4),
                round(cnn_accy.get("f1_macro", 0), 4),
                round(cnn_accy.get("f1_weighted", 0), 4)
            ])
            
            # Vẽ Confusion Matrix cho checkpoint cuối
            if idx == len(ckpt_files) - 1:
                try:
                    plot_confusion_matrix(y_true, y_pred, task, test_ckpt_root)
                except Exception as e:
                    logging.error(f"[TEST] Lỗi khi vẽ Confusion Matrix: {e}")

    logging.info(f"[TEST] Hoàn thành. Kết quả được lưu tại: {csv_path}")


def _set_device(args):
    device_type = args["device"]
    gpus = []
    for device in device_type:
        if str(device) == "-1" or str(device).lower() == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))
        gpus.append(device)
    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))


def _plot_metrics(history, run_dir, args):
    tasks = list(range(1, len(history["cnn"]["acc"]) + 1))
    has_nme = len(history["nme"]["acc"]) > 0

    metrics = ["acc", "precision", "recall", "f1"]
    labels  = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "{} + {} on {} — {}\nSeed: {}  |  Init: {}  Inc: {}".format(
            str(args["model_name"]).upper(), args["convnet_type"], args["dataset"],
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            args["seed"], args["init_cls"], args["increment"],
        ),
        fontsize=13, fontweight="bold",
    )

    for idx, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        ax = axes[idx // 2][idx % 2]
        cnn_vals = history["cnn"][metric]

        ax.plot(tasks, cnn_vals, "o-", color=color, linewidth=2,
                markersize=6, label="CNN")
        if has_nme and len(history["nme"][metric]) == len(tasks):
            nme_vals = history["nme"][metric]
            ax.plot(tasks, nme_vals, "s--", color=color, linewidth=2,
                    markersize=6, alpha=0.6, label="NME")

        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Task", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_xticks(tasks)
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)

        # Annotate each point
        for t, v in zip(tasks, cnn_vals):
            ax.annotate(f"{v:.1f}", (t, v),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=8, color=color)

    plt.tight_layout()
    plot_path = os.path.join(run_dir, "metrics_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ── Biểu đồ tổng hợp 4 metrics trên 1 axes ──────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for metric, label, color in zip(metrics, labels, colors):
        ax2.plot(tasks, history["cnn"][metric], "o-", color=color,
                 linewidth=2, markersize=6, label=label)
    ax2.set_title("CNN — All Metrics per Task", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Task")
    ax2.set_ylabel("Score (%)")
    ax2.set_xticks(tasks)
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10)
    plt.tight_layout()
    combined_path = os.path.join(run_dir, "all_metrics_combined.png")
    plt.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_test_plot(x_vals, y_vals, metric_name, color, marker, args):
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, f'{color}-{marker}', linewidth=2, markersize=4)
    plt.xlabel('Task')
    plt.ylabel(f'{metric_name} (%)' if metric_name != 'Loss' else 'Loss')
    plt.title(f'[TEST - SPCIL] {metric_name} over Tasks')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    safe_name = metric_name.lower().replace("-", "_")
    plt.savefig(os.path.join(args.get('run_dir', '.'), f'test_spcil_{safe_name}.png'), dpi=150)
    plt.close()


def save_combined_plot(x_vals, y_mic, y_mac, y_wei, category_name, args):
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_mic, 'b-o', label=f'Micro-{category_name}', linewidth=1.5, markersize=3)
    plt.plot(x_vals, y_mac, 'g-s', label=f'Macro-{category_name}', linewidth=1.5, markersize=3)
    plt.plot(x_vals, y_wei, 'r-^', label=f'Weighted-{category_name}', linewidth=1.5, markersize=3)
    plt.xlabel('Task')
    plt.ylabel(f'{category_name} (%)')
    plt.title(f'[TEST - SPCIL] {category_name} (Micro vs Macro vs Weighted)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.get('run_dir', '.'), f'test_spcil_{category_name.lower()}_combined.png'), dpi=150)
    plt.close()


TEN_LOP_CAN_IOV = [
    "Benign", "DoS", "double", "force-neutral", "fuzzing", "interval",
    "rpm", "rpm-accessory", "speed", "speed-accessory", "standstill",
    "systematic", "triple",
]


def _luu_confusion_csv(cm, ten_lop, csv_path):
    """Ghi ma tran nham lan ra CSV: SO MAU tuyet doi + ti le theo hang.

    Vi sao can CSV: tap test co 99,62% Benign nen o PNG, thang mau bi mot o
    41,7 trieu mau chiem het — moi o lop tan cong (co 10^2..10^5 mau) deu ve ra
    trang, khong doc duoc gi. So tuyet doi thi khong bi vay.

    File co hai khoi:
      [1] SO MAU        cm[i][j] = so mau lop THAT i duoc doan thanh j
                        + cot Tong_that, Dung, Recall_%
      [2] TI LE % THEO HANG  = cm[i][j] / tong hang i * 100
    Va mot khoi [3] tom tat theo lop: Precision / Recall / F1 / support.
    """
    import csv as _csv
    K = len(ten_lop)
    tong_that = cm.sum(axis=1)
    tong_doan = cm.sum(axis=0)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.writer(f)

        w.writerow(["[1] SO MAU — hang = nhan THAT, cot = nhan DOAN"])
        w.writerow(["That \\ Doan"] + ten_lop + ["Tong_that", "Dung", "Recall_%"])
        for i in range(K):
            rec = 100.0 * cm[i][i] / tong_that[i] if tong_that[i] else 0.0
            w.writerow([ten_lop[i]] + [int(cm[i][j]) for j in range(K)]
                       + [int(tong_that[i]), int(cm[i][i]), f"{rec:.4f}"])
        w.writerow(["Tong_doan"] + [int(tong_doan[j]) for j in range(K)]
                   + [int(cm.sum()), int(np.trace(cm)),
                      f"{100.0 * np.trace(cm) / cm.sum():.4f}" if cm.sum() else "0"])

        w.writerow([])
        w.writerow(["[2] TI LE % THEO HANG (moi hang cong lai = 100)"])
        w.writerow(["That \\ Doan"] + ten_lop)
        for i in range(K):
            t = tong_that[i]
            w.writerow([ten_lop[i]] + [f"{100.0 * cm[i][j] / t:.4f}" if t else "0"
                                       for j in range(K)])

        w.writerow([])
        w.writerow(["[3] TUNG LOP"])
        w.writerow(["Lop", "So_mau_that", "So_lan_duoc_doan", "TP",
                    "Precision_%", "Recall_%", "F1_%"])
        for i in range(K):
            tp = int(cm[i][i])
            p = 100.0 * tp / tong_doan[i] if tong_doan[i] else 0.0
            r = 100.0 * tp / tong_that[i] if tong_that[i] else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            w.writerow([ten_lop[i], int(tong_that[i]), int(tong_doan[i]), tp,
                        f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}"])


def plot_confusion_matrix(y_true, y_pred, task_id, run_dir):
    """Ve Confusion Matrix ra PNG va CSV."""
    # y_pred thuong co dang [N, topk], lay top1
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_top1 = y_pred[:, 0]
    else:
        y_pred_top1 = y_pred.flatten()

    y_true = np.asarray(y_true).ravel()
    y_pred_top1 = np.asarray(y_pred_top1).ravel()

    # labels= TUONG MINH: khong co no thi sklearn suy tap nhan tu
    # union(y_true, y_pred), nen mot lop da hoc ma vang mat o CA hai phia se
    # lam ma tran CO LAI va moi nhan truc sau no bi dich di 1 — sai am tham.
    K = int(max(int(y_true.max()) if y_true.size else 0,
                int(y_pred_top1.max()) if y_pred_top1.size else 0) + 1)
    labels = list(range(K))
    ten_lop = [TEN_LOP_CAN_IOV[i] if i < len(TEN_LOP_CAN_IOV) else str(i)
               for i in labels]
    cm = confusion_matrix(y_true, y_pred_top1, labels=labels)

    csv_path = os.path.join(run_dir, f'confusion_matrix_task_{task_id:02d}.csv')
    _luu_confusion_csv(cm, ten_lop, csv_path)

    # PNG: thang mau LOG + in so vao o. Voi 41,7 trieu Benign va vai tram mau
    # lop hiem, thang tuyen tinh lam moi o tan cong deu ve ra trang.
    from matplotlib.colors import LogNorm
    plt.figure(figsize=(1.15 * K + 5, 1.0 * K + 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', linewidths=.5, linecolor='#dddddd',
        norm=LogNorm(vmin=max(1, cm[cm > 0].min()) if (cm > 0).any() else 1,
                     vmax=max(1, cm.max())),
        xticklabels=ten_lop, yticklabels=ten_lop,
        annot_kws={"size": 7},
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - Task {task_id}  (thang mau LOG)')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)

    save_path = os.path.join(run_dir, f'confusion_matrix_task_{task_id:02d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f'[TEST] Da luu Confusion Matrix: {save_path}')
    logging.info(f'[TEST] Da luu Confusion Matrix CSV: {csv_path}')
