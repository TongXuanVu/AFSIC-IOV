"""Cell Kaggle — sàng lọc ablation TASK 0 cho AFSIC-IoV 100 client.

MỤC ĐÍCH: tìm bước nào gỡ được hiện tượng "đoán đúng một lớp" ở task 0.

RÀNG BUỘC ĐÃ GIỮ: không đụng dữ liệu huấn luyện.
    max_samples_per_class = null, test_max_samples_per_class = null.
    Việc lấy mẫu (proto_max_samples / quality_eval_max_samples) chỉ áp cho khâu
    ĐO — client vẫn train trên toàn bộ dữ liệu của mình.

BA QUY TẮC (xem GIAO_THUC_CHAY.md):
  a) Mọi arm dùng CÙNG chế độ nhanh, kể cả arm đối chứng A_goc.
  b) Đủ 30 round. HFIN nằm im ở 33,27 suốt 14 round mới thoát.
  c) Chấm bằng cả đường cong (N_thoat, R_max), không bằng round cuối.

TRƯỚC KHI CHẠY: git add -A && git commit && git push
"""
import glob
import json
import os
import re
import subprocess
import time

# ──────────────────────────────────────────────────────────────────────────
REPO = "https://github.com/TongXuanVu/AFSIC-IOV.git"
ROOT = "/kaggle/input/iov100client"      # SỬA theo tên dataset thật
NUM_ROUNDS = 30

# VÒNG 1 — tài khoản A chạy 2 arm này:
DANH_SACH = ["LR1_thap_hang", "A_goc"]
# VÒNG 1 — tài khoản B chạy 2 arm này (đổi dòng trên thành):
#   DANH_SACH = ["LR2_khong_giam", "E0_tat_calibrate"]
#
# VÒNG 2 (chỉ nếu vòng 1 có arm thắng):
#   ["LR1_cong_E0", "LR3_giam_som", "FULL_cong_don"]
# Xếp sau, cho bảng ablation của luận văn:
#   ["E1_proto_khong_drift", "E2_beta_acc_n_0", "E3_do_dung_theta",
#    "E4_fednova", "E5_bn_theo_n", "FEDAVG", "FEDAVG_giu_proto",
#    "E0b_calibrate_1lan"]

NGUONG_REC = 33.40   # rec_macro của bộ đoán một lớp là 33,33
# ──────────────────────────────────────────────────────────────────────────

os.system("rm -rf /kaggle/working/AFSIC-IOV")
os.system(f"git clone -q {REPO} /kaggle/working/AFSIC-IOV")
os.chdir("/kaggle/working/AFSIC-IOV")

shards = glob.glob(f"{ROOT}/**/federated_data/client_*_task_*.pt", recursive=True)
tests = glob.glob(f"{ROOT}/**/global_test_data.pt", recursive=True)
assert shards, f"Khong thay shard nao duoi {ROOT} — sua bien ROOT"
assert tests, f"Khong thay global_test_data.pt duoi {ROOT}"
print(f"OK — {len(shards)} shard, test: {tests[0]}")

tong_ket = {}
for ten in DANH_SACH:
    cfg_path = f"configs/exps/abl_{ten}.json"
    assert os.path.exists(cfg_path), f"Thieu {cfg_path} — da push code chua?"
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    cfg["num_rounds"] = NUM_ROUNDS
    cfg["max_tasks"] = 1
    cfg["max_samples_per_class"] = None        # nhac lai: DU LIEU NGUYEN VEN
    cfg["test_max_samples_per_class"] = None
    json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"\n{'='*72}\n>>> {ten}\n{'='*72}")
    t0 = time.time()
    subprocess.run(["python", "main.py", "--config", cfg_path], check=False)
    phut = (time.time() - t0) / 60
    print(f"\n[{ten}] tong {phut:.0f} phut  =>  {phut/NUM_ROUNDS:.1f} phut/round")
    if phut / NUM_ROUNDS > 8:
        print("[!] CANH BAO: >8 phut/round. Toi uu chua an — DUNG LAI, dung dot "
              "tiep ngan sach. Cho nghen nam cho khac.")

    runs = sorted(glob.glob("logs/afsic-iov_federated/can_iov/*_clients100"),
                  key=os.path.getmtime)
    if not runs:
        print(f"[!] {ten}: khong thay thu muc log")
        continue
    run = runs[-1] + f"__{ten}"
    os.rename(runs[-1], run)

    # ── Chấm điểm bằng cả đường cong ──
    csv = os.path.join(run, "metrics_round_by_round.csv")
    if os.path.exists(csv):
        import csv as _csv
        rows = [r for r in _csv.DictReader(open(csv))
                if r.get("task", "").strip() not in ("", "task")]
        rec = []
        for r in rows:
            try:
                rec.append((int(float(r["round"])), float(r["rec_mac"]),
                            float(r["f1_mac"]), float(r["acc"])))
            except (ValueError, KeyError):
                pass
        if rec:
            n_thoat = sum(1 for _, rm, _, _ in rec if rm > NGUONG_REC)
            r_max = max(rm for _, rm, _, _ in rec)
            tong_ket[ten] = (n_thoat, r_max, rec[-1][1])
            print(f"\n--- {ten}: {'round':>5}{'acc':>8}{'rec_mac':>9}{'f1_mac':>8}")
            for rd, rm, fm, ac in rec:
                dau = "  <<<" if rm > NGUONG_REC else ""
                print(f"{'':13}{rd:>5}{ac:>8.2f}{rm:>9.2f}{fm:>8.2f}{dau}")
            print(f"    N_thoat = {n_thoat}/{len(rec)} | R_max = {r_max:.2f}")

    log = os.path.join(run, "training.log")
    if os.path.exists(log):
        diag = [l.strip() for l in open(log, encoding="utf-8", errors="replace")
                if "[DIAG]" in l]
        for l in diag[-2:]:
            print("    " + re.sub(r"^.*\[DIAG\]", "[DIAG]", l))
        print(f"\n    --- UpdateNorm con bam so batch khong? ---")
        subprocess.run(["python", "tools/kiem_update_norm.py", log, "8192"])

print(f"\n\n{'#'*72}\n# TONG KET\n{'#'*72}")
print(f"{'arm':<24}{'N_thoat':>9}{'R_max':>9}{'rec cuoi':>10}")
for ten, (n, rmax, rcuoi) in tong_ket.items():
    print(f"{ten:<24}{n:>9}{rmax:>9.2f}{rcuoi:>10.2f}")
print(f"""
{'-'*72}
DOI CHIEU tren CUNG bo du lieu (30 round task 0):
  HFIN        N_thoat = 16   R_max = 49.84   -> hoc duoc
  SPCIL-FL    N_thoat =  3   R_max = 39.27   -> chi chop, khong hoi tu
  AFSIC goc   N_thoat =  2   R_max = 43.93   -> thoat roi tut lai

DOC KET QUA
  N_thoat >= 10  va R_max > 45   -> arm NAY AN, sang vong 2
  N_thoat 3-9                    -> co dau hieu, can them seed truoc khi tin
  N_thoat <= 2                   -> khong khac gi baseline

Nhin rec_macro, KHONG nhin f1_macro: tap test 99,62% Benign nen f1_macro che
mat viec mo hinh doi recall Benign lay recall lop tan cong (round 9 cua AFSIC:
rec_mac 43,93 nhung f1_mac chi 33,01).

Neu CA BON arm deu co N_thoat <= 2: dung chay vong 2. Bon arm that bai van loai
duoc LR, calibration va cach gop ra khoi danh sach nghi pham.
{'-'*72}""")
