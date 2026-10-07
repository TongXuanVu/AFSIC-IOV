"""Do MO HINH CA NHAN HOA cua tung client tu mot checkpoint round da luu.

Khong huan luyen lai. Dung lai dung cac ham cua trainer.py de dung mo hinh:
  1. Mo rong kien truc qua cac task 0..t nhu luc huan luyen, nap checkpoint
     (mo hinh toan cuc + trong so rieng cua 100 client + prototype toan cuc)
     va bo nho exemplar (file _MEM cua task t-1, chinh la bo nho dung trong task t).
  2. Danh gia lai mo hinh toan cuc tren tap test -> phai TRUNG voi so trong log
     (kiem tra viec dung lai mo hinh la dung).
  3. Moi client co du lieu lop moi o task t: tinh lai prototype cuc bo tu trong
     so cua chinh no (giong buoc cuoi round), nap phan dung chung tu server, ghi
     hang fc lop moi bang prototype ca nhan hoa p~, roi danh gia tren tap test.

Ghi ra per_client_from_ckpt.csv: F1/acc tren moi lop, va F1/recall tren CAC LOP
CLIENT DO CO (lop moi cua task + lop cu co exemplar), so voi mo hinh toan cuc.

Vi du:
  python tools/eval_ca_nhan_hoa.py --config cfg.json --ckpt ckpt_round0150_task04_r030_acc98.3.pth \
      --memory ckpt_task03_memory_client99_MEM.pth --out ketqua_ca_nhan_hoa --verify_f1 10.63
"""
import argparse
import copy
import csv
import json
import logging
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trainer as T  # noqa: E402
from utils import factory  # noqa: E402
from utils.data_manager import DataManager  # noqa: E402


def _top1(y_pred):
    y_pred = np.asarray(y_pred)
    return y_pred[:, 0] if y_pred.ndim > 1 else y_pred.flatten()


def _own_metrics(y_true, y_top1, own):
    from sklearn.metrics import f1_score, recall_score
    if not own:
        return float("nan"), float("nan")
    f1 = 100.0 * f1_score(y_true, y_top1, labels=own, average="macro", zero_division=0)
    rec = 100.0 * recall_score(y_true, y_top1, labels=own, average="macro", zero_division=0)
    return f1, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="checkpoint ROUND (khong phai checkpoint memory)")
    ap.add_argument("--memory", required=True, help="file _MEM cua task truoc (bo nho dung trong task cua ckpt)")
    ap.add_argument("--out", default="ketqua_ca_nhan_hoa")
    ap.add_argument("--verify_f1", type=float, default=None, help="f1_macro toan cuc trong log de doi chieu")
    ap.add_argument("--max_clients", type=int, default=None, help="chi do N client dau (chay thu)")
    cli = ap.parse_args()

    os.makedirs(cli.out, exist_ok=True)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s => %(message)s",
                        handlers=[logging.FileHandler(os.path.join(cli.out, "eval.log")),
                                  logging.StreamHandler(sys.stdout)])

    args = json.load(open(cli.config, encoding="utf-8"))
    if isinstance(args.get("seed"), list):
        args["seed"] = args["seed"][0]
    T._dat_ten_lop(args.get("dataset"))
    T._set_random()
    T._set_device(args)
    dev = args["device"][0]

    if args["dataset"] == "can_iov":
        from utils import data_can_iov
        if args.get("max_samples_per_class"):
            data_can_iov.MAX_SAMPLES_PER_CLASS = int(args["max_samples_per_class"])
        if args.get("test_max_samples_per_class"):
            data_can_iov.TEST_MAX_SAMPLES_PER_CLASS = int(args["test_max_samples_per_class"])

    ck = torch.load(cli.ckpt, map_location="cpu", weights_only=False)
    mem = torch.load(cli.memory, map_location="cpu", weights_only=False)
    task = int(ck["task"])
    assert not ck.get("is_memory_phase", False), "Can checkpoint ROUND, khong phai checkpoint memory"
    assert int(mem["task"]) == task - 1, f"File MEM la task {mem['task']}, can task {task - 1}"
    logging.info(f"ckpt task {task} round {ck['round'] + 1} | MEM task {mem['task']} | {cli.ckpt}")

    n = args["num_clients"]
    client_dms = [DataManager(args["dataset"], args["shuffle"], args["seed"], args["init_cls"], args["increment"],
                              client_id=c, class_order=args.get("class_order"),
                              task_increments=args.get("task_increments")) for c in range(n)]
    global_model = factory.get_model(args["model_name"], args)
    local_models = [factory.get_model(args["model_name"], args) for _ in range(n)]

    # Mo rong kien truc giong het vong lap task cua trainer
    for t in range(task + 1):
        global_model.incremental_train(client_dms[0], skip_train=True)
        global_model._network.to(dev)
        for c in range(n):
            local_models[c].skip_rehearsal = True
            local_models[c].incremental_train(client_dms[c], skip_train=True)
            local_models[c].skip_rehearsal = False
        if t < task:
            for c in range(n):
                local_models[c].after_task()
            global_model.after_task()

    global_model._network.load_state_dict(ck["model_state_dict"])
    if ck.get("global_proto_memory") is not None:
        global_model.global_proto_memory = ck["global_proto_memory"]
    if ck.get("class_prior_counts"):
        global_model.class_prior_counts = {int(k): int(v) for k, v in ck["class_prior_counts"].items()}
    for c in range(n):
        local_models[c]._network.load_state_dict(ck["client_states"][c]["net"], strict=False)
        local_models[c]._network.to(dev)
        ms = mem["client_states"][c]
        local_models[c]._data_memory = ms.get("data_memory")
        local_models[c]._targets_memory = ms.get("targets_memory")
        if ms.get("local_memory") is not None:
            local_models[c].local_memory = ms["local_memory"]
    del mem

    # 1) Mo hinh toan cuc: doi chieu voi log
    global_model.test_loader = T._build_global_learned_test_loader(
        client_dms[0], global_model._total_classes, args["batch_size"],
        args_eval_cap=args.get("eval_max_per_class"))
    t0 = time.time()
    g_accy, _, g_pred, y_true = global_model.eval_task()
    g_top1 = _top1(g_pred)
    del g_pred
    g_f1 = float(g_accy.get("f1_macro", 0))
    logging.info(f"[GLOBAL] acc {g_accy['top1']:.2f} | f1_macro {g_f1:.2f} | {time.time() - t0:.0f}s")
    if cli.verify_f1 is not None:
        ok = abs(g_f1 - cli.verify_f1) < 0.05
        logging.info(f"[KIEM TRA] f1_macro dung lai {g_f1:.2f} vs log {cli.verify_f1:.2f} -> {'KHOP' if ok else 'KHONG KHOP'}")
        assert ok, "Mo hinh dung lai khong khop log - dung lai, khong tin so ca nhan hoa"

    # 2) Tung client
    known, total = global_model._known_classes, global_model._total_classes
    global_state = global_model._network.state_dict()
    out_csv = os.path.join(cli.out, "per_client_from_ckpt.csv")
    rows = []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "client", "own_classes", "new_classes", "acc", "f1_macro", "f1_own", "rec_own",
                    "global_acc", "global_f1_macro", "global_f1_own", "global_rec_own", "gain_f1_own"])
        done = 0
        for c in range(n):
            if cli.max_clients is not None and done >= cli.max_clients:
                break
            lm = local_models[c]
            new_cls = [k for k in range(known, total)
                       if len(client_dms[c].get_dataset(np.arange(k, k + 1), source="train", mode="test")) > 0]
            if not new_cls:
                continue   # trainer: client khong co du lieu lop moi -> khong tham gia task nay
            mem_t = lm._targets_memory
            old_cls = sorted(int(x) for x in np.unique(np.asarray(mem_t))) if mem_t is not None and len(mem_t) else []
            own = sorted(set(old_cls) | set(new_cls))
            t1 = time.time()
            lm.global_proto_memory = copy.deepcopy(global_model.global_proto_memory)
            lm._network.eval()
            # Prototype cuc bo: giong het buoc cuoi round trong trainer
            lm.compute_local_prototypes(client_dms[c], class_ids=range(known),
                                        max_samples_per_class=args.get("proto_max_samples"),
                                        seed=args.get("seed", 0) + c, report_full_count=True)
            _cap = (args.get("kshot", 10) if task > 0 and args.get("fewshot_enabled", True)
                    else args.get("proto_max_samples"))
            lm.compute_local_prototypes(client_dms[c], class_ids=range(known, total), max_samples_per_class=_cap,
                                        seed=args.get("seed", 0) + task * 1009 + c,
                                        report_full_count=(_cap is not None and _cap == args.get("proto_max_samples")))
            T._load_global_into_client(lm, global_state, task, args)
            lm._network.to(dev)
            T._calibrate_classifier_from_prototypes(lm)
            lm._network.eval()
            p_pred, p_true, p_loss = lm._eval_cnn(global_model.test_loader)
            pm = lm._evaluate(p_pred, p_true, loss=p_loss)
            p_top1 = _top1(p_pred)
            del p_pred
            f1_own, rec_own = _own_metrics(p_true, p_top1, own)
            gf1_own, grec_own = _own_metrics(y_true, g_top1, own)
            row = [task, c, " ".join(map(str, own)), " ".join(map(str, new_cls)),
                   round(float(pm["top1"]), 4), round(float(pm.get("f1_macro", 0)), 4),
                   round(f1_own, 4), round(rec_own, 4),
                   round(float(g_accy["top1"]), 4), round(g_f1, 4), round(gf1_own, 4), round(grec_own, 4),
                   round(f1_own - gf1_own, 4)]
            w.writerow(row)
            f.flush()
            rows.append(row)
            done += 1
            logging.info(f"[PerFL] client {c} lop {own} (moi {new_cls}): F1 lop rieng {f1_own:.2f} rec {rec_own:.2f} "
                         f"| toan cuc tren cung lop: F1 {gf1_own:.2f} rec {grec_own:.2f} | F1-mac moi lop {pm.get('f1_macro', 0):.2f} "
                         f"| {time.time() - t1:.0f}s")

    if rows:
        m = lambda i: float(np.nanmean([r[i] for r in rows]))
        logging.info(f"[TONG] {len(rows)} client | F1 lop rieng TB {m(6):.2f} (toan cuc {m(10):.2f}, gain {m(12):+.2f}) "
                     f"| recall lop rieng TB {m(7):.2f} (toan cuc {m(11):.2f}) | F1-mac moi lop TB {m(5):.2f} (toan cuc {g_f1:.2f})")
    logging.info(f"Ghi {out_csv}")


if __name__ == "__main__":
    main()
