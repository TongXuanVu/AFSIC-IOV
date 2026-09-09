"""
Dem SO MAU HUAN LUYEN that su cua tung lop tren toan lien doan.

Dung de lam prior cho hieu chinh logit (logit_prior_adjust). Vi sao can file
nay thay vi doc thang global_proto_memory trong checkpoint:

  Tu task 1 tro di, lop cu chi con ton tai trong tap huan luyen duoi dang
  replay 1%, con lop MOI cua task do giu du lieu day du. Nen so dem trong
  global_proto_memory o cuoi task 4 la "1% cua 11 lop cu" tron voi "100% cua
  2 lop moi" — lop moi bi de cao gap ~100 lan. Ma dung hai lop moi ay moi la
  hai lop dang nuot het du doan, nen prior lech theo huong lam nang them.

Script nay chi doc TENSOR NHAN cua tap huan luyen. Khong dung mot con so nao
cua tap test.

Chay:
    python tools/dem_so_mau_moi_lop.py --out prior_counts.json
    python tools/dem_so_mau_moi_lop.py --data_dir /duong/dan/federated_data
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

NUM_CLASSES = 13


def tim_thu_muc_data(explicit=None):
    if explicit:
        return explicit
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local = os.path.join(here, "data", "federated_data")
    if os.path.isdir(local) and glob.glob(os.path.join(local, "client_*_task_*.pt")):
        return local
    hits = glob.glob("/kaggle/input/**/federated_data/client_*_task_*.pt", recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None, help="Thu muc federated_data")
    ap.add_argument("--out", default="prior_counts.json")
    ap.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    args = ap.parse_args()

    data_dir = tim_thu_muc_data(args.data_dir)
    if not data_dir:
        print("[LOI] Khong tim thay thu muc federated_data.")
        return 1
    files = sorted(glob.glob(os.path.join(data_dir, "client_*_task_*.pt")))
    if not files:
        print(f"[LOI] Khong co file client_*_task_*.pt trong {data_dir}")
        return 1
    print(f"[DEM] Thu muc : {data_dir}")
    print(f"[DEM] So file : {len(files)}")

    tong = np.zeros(args.num_classes, dtype=np.int64)
    theo_task = {}
    for i, p in enumerate(files, 1):
        d = torch.load(p, map_location="cpu", weights_only=False)
        y = d["y"] if isinstance(d, dict) else d[1]
        y = y.numpy().astype(np.int64).ravel()
        bc = np.bincount(y, minlength=args.num_classes)[: args.num_classes]
        tong += bc
        # task_id trong ten file la 1..5
        base = os.path.basename(p)
        try:
            tid = int(base.rsplit("_task_", 1)[1].split(".")[0])
        except Exception:
            tid = -1
        theo_task[tid] = theo_task.get(tid, np.zeros(args.num_classes, dtype=np.int64)) + bc
        del d, y
        if i % 50 == 0 or i == len(files):
            print(f"[DEM]   {i}/{len(files)} file | tong hien tai = {int(tong.sum()):,}")

    print("\n[DEM] SO MAU HUAN LUYEN MOI LOP (toan lien doan):")
    for c in range(args.num_classes):
        pi = tong[c] / max(1, tong.sum())
        print(f"   lop {c:>2} : {int(tong[c]):>12,}   pi = {pi:.8f}   log pi = {np.log(pi + 1e-12):8.4f}")
    print(f"   {'TONG':>6} : {int(tong.sum()):>12,}")

    if any(t < 0 for t in theo_task):
        pass
    else:
        print("\n[DEM] Kiem tra: lop nao xuat hien o task nao (task_id 1..5)")
        for tid in sorted(theo_task):
            co = [c for c in range(args.num_classes) if theo_task[tid][c] > 0]
            print(f"   task {tid}: lop {co}")

    if int((tong == 0).sum()):
        thieu = [c for c in range(args.num_classes) if tong[c] == 0]
        print(f"\n[CANH BAO] Cac lop sau co 0 mau: {thieu}. "
              f"Hieu chinh prior se bi bo qua neu con lop bang 0.")

    out = {
        "_ghi_chu": "So mau HUAN LUYEN moi lop tren toan lien doan. "
                    "Sinh boi tools/dem_so_mau_moi_lop.py. Khong dung du lieu tap test.",
        "_data_dir": data_dir,
        "_so_file": len(files),
        "_tong": int(tong.sum()),
        "counts": [int(v) for v in tong],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[DEM] Da ghi: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
