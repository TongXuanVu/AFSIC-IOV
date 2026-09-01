"""In ma tran nham lan + bao cao TUNG LOP cua MOT checkpoint, tren tap test DAY DU.

Vi sao can: CSV round-by-round chi co so gop (acc, f1_macro). No khong cho biet
lop nao that su duoc phat hien va Benign di dau. Ma tran nham lan PNG luu trong
run_dir bi ghi de moi round nen chi con round CUOI.

Dung dung duong code cua run_test (trainer.py) — khong cat tap test.

  python tools/xem_confusion.py \
      --ckpt logs/.../checkpoints/ckpt_round0011_task00_r011_acc97.1.pth \
      --config configs/exps/afsic-iov-task0.json

Chi phi ~1-2 phut GPU cho mot checkpoint (41,8 trieu mau).
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trainer as T
from utils import factory
from utils.data_manager import DataManager

# Thu tu lop CAN-bus theo task_increments [3,3,3,2,2]
TEN_LOP = ["Benign", "DoS", "double", "force-neutral", "fuzzing", "interval",
           "rpm", "rpm-accessory", "speed", "speed-accessory", "standstill",
           "systematic", "triple"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--csv", default="", help="Ghi ma tran ra file CSV (tuy chon)")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    args = json.load(open(a.config, encoding="utf-8"))
    if isinstance(args.get("seed"), list):
        args["seed"] = args["seed"][0]
    T._set_random()
    T._set_device(args)

    state = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    task = state["task"]
    print(f"\ncheckpoint : {os.path.basename(a.ckpt)}")
    print(f"task {task} | round {state.get('round')} | "
          f"metrics luc luu: {state.get('metrics', {}).get('top1', '?')}")

    dm = DataManager(args["dataset"], False, args["seed"], args["init_cls"],
                     args["increment"], client_id=0,
                     class_order=args.get("class_order"),
                     task_increments=args.get("task_increments"))

    model = factory.get_model(args["model_name"], args)
    for _ in range(task + 1):
        model.incremental_train(dm, skip_train=True)
    try:
        model._network.load_state_dict(state["model_state_dict"])
    except RuntimeError as e:
        # Thuong gap: config dung de xem KHAC config luc chay (vd adapter_bottleneck).
        print("\n[!] Kien truc khong khop checkpoint. Dung DUNG config da chay.")
        print("    Neu thieu/thua khoa 'adapter', them lai \"adapter_bottleneck\": 64 "
              "vao config roi chay lai.\n")
        raise
    model._network.to(args["device"][0])
    model._network.eval()

    # KHONG truyen args_eval_cap => tap test DAY DU, giong duong --mode test
    model.test_loader = T._build_global_learned_test_loader(
        dm, model._total_classes, args["batch_size"])
    if model.test_loader is None:
        print("[!] Tap test rong."); return

    # Goi thang _eval_cnn thay vi eval_task vi hai ly do:
    #   1. y_pred cua no la [N, topk] -> phai lay cot 0 (top-1).
    #   2. eval_task GHI DE y_pred bang du doan NME neu model co _class_means
    #      (base.py:103-104), trong khi acc/f1 bao cao lai la cua nhanh CNN.
    #      Goi thang _eval_cnn thi chac chan dang xem dung nhanh CNN.
    y_pred, y_true, loss = model._eval_cnn(model.test_loader)
    accy = model._evaluate(y_pred, y_true, loss=loss)
    y_pred = np.asarray(y_pred)
    y_pred_top1 = y_pred[:, 0] if y_pred.ndim > 1 else y_pred.ravel()
    y_true = np.asarray(y_true).ravel()

    K = model._total_classes
    ten = TEN_LOP[:K]

    cm = confusion_matrix(y_true, y_pred_top1, labels=list(range(K)))
    tong = cm.sum()

    print(f"\nTong mau test: {tong:,}   ({K} lop da hoc)")
    print(f"acc {accy['top1']:.2f} | f1_macro {accy.get('f1_macro',0):.2f} | "
          f"rec_macro {accy.get('recall_macro',0):.2f} | "
          f"prec_macro {accy.get('precision_macro',0):.2f}")

    print("\n=== MA TRAN NHAM LAN (hang = that, cot = doan) ===")
    w = max(14, max(len(t) for t in ten) + 1)
    print(" " * w + "".join(f"{t[:11]:>13}" for t in ten) + f"{'TONG':>14}")
    for i, t in enumerate(ten):
        print(f"{t:<{w}}" + "".join(f"{cm[i][j]:>13,}" for j in range(K)) +
              f"{cm[i].sum():>14,}")
    print(f"{'TONG doan':<{w}}" + "".join(f"{cm[:,j].sum():>13,}" for j in range(K)))

    print("\n=== TI LE THEO HANG (recall tung lop, %) ===")
    for i, t in enumerate(ten):
        s = cm[i].sum()
        if s == 0:
            print(f"{t:<{w}}(khong co mau test)"); continue
        print(f"{t:<{w}}" + "".join(f"{100.0*cm[i][j]/s:>12.2f}%" for j in range(K)) +
              f"   n={s:,}")

    print("\n=== BAO CAO TUNG LOP ===")
    print(classification_report(y_true, y_pred_top1, labels=list(range(K)),
                                target_names=ten, digits=4, zero_division=0))

    if a.csv:
        import csv as _csv
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            wr = _csv.writer(f)
            wr.writerow(["that\\doan"] + ten)
            for i, t in enumerate(ten):
                wr.writerow([t] + [int(x) for x in cm[i]])
        print(f"\nDa ghi {a.csv}")


if __name__ == "__main__":
    main()
