import csv, glob, json, os, re, shutil, signal, subprocess, sys, time, zipfile

# CELL KAGGLE CHAY AFSIC-IoV (nhanh main, config exp_logit_prior.json).
#   LA_TAU = 1.0 cho giai doan 1, 0.5 cho cac giai doan tiem tien (xem README, muc tai lap ket qua).
#   QUALITY_GATE = True -> cap nhat toan cuc co kiem dinh (quality_gated_update).
#   RATIO_SCALE = True  -> kich ban 1%: nhan 1/kshot so dem lop tiem tien khi chon tau*.
#   RESUME = None    -> chay TU DAU (task 1 tro di), CUNG mot cau hinh cho moi task.
#   RESUME = "auto"  -> tu tim ckpt_taskXX_memory_client99.pth trong /kaggle/input
#   RESUME = "ten_thu_muc" -> chi tim trong thu muc input do
#   RESUME_TASK = None -> lay checkpoint task cao nhat tim thay; so nguyen -> dung task do
#                         (vd 0 = sau task 1, dung cho 1percent/10shot vi task 1 giong full)
#   EVAL_EVERY = 1   -> danh gia + hieu chinh MOI round (de ve do thi theo round)
# Cell tu dung truoc gioi han 12 gio cua Kaggle (TIME_BUDGET_H), dong goi ket qua cac
# task da xong va in ra file checkpoint de chay tiep.
BRANCH = "main"
SCENARIO = "full"          # "full" | "1percent" | "10shot"
LA_TAU = 1.0               # 1.0 giai doan 1, 0.5 cac giai doan sau
QUALITY_GATE = False
RATIO_SCALE = False
RESUME = None              # None | "auto" | ten thu muc trong /kaggle/input
RESUME_TASK = None         # None = task cao nhat; 0 = checkpoint sau task 1
EVAL_EVERY = 1             # 1 = moi round (do thi theo round); 5 = nhanh hon
TIME_BUDGET_H = 11.3       # dung main.py sau ngan nay gio de con thoi gian dong goi

BASE_NAME = "exp_logit_prior"
REPO = "https://github.com/TongXuanVu/AFSIC-IOV.git"
WORK = "/kaggle/working"
CODE = os.path.join(WORK, "AFSIC-IOV")
T_START = time.time()
assert SCENARIO in ("full", "1percent", "10shot")
CONFIG_NAME = (f"{BASE_NAME}_{SCENARIO}_la{str(LA_TAU).replace('.', '')}" + ("_gate" if QUALITY_GATE else "")
               + ("_ratio" if RATIO_SCALE else ""))
CONFIG = f"configs/exps/{CONFIG_NAME}.json"

assert glob.glob("/kaggle/input/**/federated_data/client_*_task_*.pt", recursive=True) and \
       glob.glob("/kaggle/input/**/global_test_data.pt", recursive=True), "Thieu dataset IoV 100 client"
RESUME_ARGS = []
if RESUME:
    if RESUME == "auto":
        cands = glob.glob("/kaggle/input/**/ckpt_task*_memory_client99.pth", recursive=True)
    else:
        cands = glob.glob(f"/kaggle/input/**/{RESUME}/**/ckpt_task*_memory_client99.pth", recursive=True) + \
                glob.glob(f"/kaggle/input/**/{RESUME}/ckpt_task*_memory_client99.pth", recursive=True)
    if RESUME_TASK is not None:
        cands = [p for p in cands if int(re.search(r"ckpt_task(\d+)_memory", p).group(1)) == int(RESUME_TASK)]
    cands = sorted(set(cands), key=lambda p: int(re.search(r"ckpt_task(\d+)_memory", p).group(1)))
    assert cands, f"Khong thay ckpt_taskXX_memory_client99.pth (RESUME={RESUME}, RESUME_TASK={RESUME_TASK})"
    MDL = cands[-1]
    MEM = MDL.replace(".pth", "_MEM.pth")
    assert os.path.exists(MEM), f"Thieu {MEM}"
    RESUME_ARGS = ["--resume", MDL, "--resume_memory", MEM]
    _t_done = int(re.search(r"ckpt_task(\d+)_memory", MDL).group(1))
    print(f"[resume] da xong task {_t_done} (dem tu 0) -> chay tiep tu task {_t_done + 1} | file {MDL}")
else:
    print("[tu dau] chay task 1 tro di voi cung cau hinh cua nhanh")

shutil.rmtree(CODE, ignore_errors=True)
subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", BRANCH, REPO, CODE], check=True)
os.chdir(CODE)
COMMIT = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
print("[code]", BRANCH, "@", COMMIT, "|", subprocess.check_output(["git", "log", "--oneline", "-1"]).decode().strip())

cfg = json.load(open(f"configs/exps/{BASE_NAME}.json", encoding="utf-8"))
cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
if SCENARIO == "1percent":
    cfg.update(fewshot_enabled=True, kshot=0.01)
elif SCENARIO == "10shot":
    cfg.update(fewshot_enabled=True, kshot=10)
cfg["eval_every_n_rounds"] = int(EVAL_EVERY)
cfg["la_tau"] = float(LA_TAU)
cfg["quality_gated_update"] = bool(QUALITY_GATE)
cfg["calib_fewshot_ratio_scale"] = bool(RATIO_SCALE)
cfg.pop("max_tasks", None)
json.dump(cfg, open(CONFIG, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

for k, v in {"shared_encoder_trainable": True, "single_shared_encoder": True, "aggregate_encoder_only": True,
             "adapter_input": "raw", "personalized_adapter": True, "prototype_classifier": True,
             "proto_q_mode": "class_recall", "decision_calibration": True, "log_client_recall": True,
             "memory_ratio": 0.01, "lr": 3e-05, "num_clients": 100, "num_rounds": 30}.items():
    assert cfg.get(k) == v, f"{k}={cfg.get(k)!r}, can {v!r}"
_src = "".join(open(f, encoding="utf-8").read() for f in
               ("trainer.py", "models/afsic_ids.py", "utils/inc_net.py", "utils/aggregation.py"))
assert int(cfg.get("calib_every_n_rounds", 0) or 0) == 1, "Code tren GitHub chua phai ban HC moi round - chua push?"
for _f in ("_decision_calibration", "_calib_loader_fewshot", "log_client_recall", "single_shared_encoder",
           "aggregate_encoder_only", "quality_gated_update", "calib_fewshot_ratio_scale"):
    assert _f in _src, f"Code chua co {_f} - sai nhanh?"
print(f"[config] {CONFIG_NAME} | quality_mode={cfg.get('quality_mode', 'acc')} | "
      f"local_loss_mode={cfg.get('local_loss_mode', 'ce')} | class_weight_power={cfg.get('class_weight_power')} | "
      f"fewshot={cfg['fewshot_enabled']} kshot={cfg['kshot']} | eval_every={cfg['eval_every_n_rounds']}")

# Chay main.py, tu dung truoc gioi han phien. Log van in thang ra o cell.
budget_s = TIME_BUDGET_H * 3600 - (time.time() - T_START)
proc = subprocess.Popen([sys.executable, "main.py", "--config", CONFIG] + RESUME_ARGS)
stopped = False
try:
    proc.wait(timeout=max(60, budget_s))
except subprocess.TimeoutExpired:
    stopped = True
    print(f"\n[het gio] dung main.py sau {TIME_BUDGET_H} gio de dong goi ket qua")
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
print(f"\n[xong] {(time.time() - T_START) / 3600:.2f} gio | {'DUNG GIUA CHUNG' if stopped else 'chay het'}")

runs = sorted(glob.glob("logs/afsic-iov_federated/can_iov/*_clients100"), key=os.path.getmtime)
assert runs, "Khong thay thu muc log"
run = runs[-1]
open(os.path.join(run, "CODE_COMMIT.txt"), "w").write(
    f"{BRANCH} @ {COMMIT}\nconfig {CONFIG_NAME} (la_tau={LA_TAU}, gate={QUALITY_GATE}, ratio_scale={RATIO_SCALE})\nresume {RESUME_ARGS[1] if RESUME_ARGS else 'tu dau'}\n")
rows = [r for r in csv.reader(open(os.path.join(run, "metrics_round_by_round.csv"), encoding="utf-8-sig"))
        if r and r[0] != "task" and len(r) >= 15]
print("\nKET QUA CUOI CUNG (metrics_round_by_round.csv, sau hieu chinh tau*)\nTASK (dem tu 0) | f1_mac round cuoi da chay | max | so round")
for t in sorted({int(r[0]) for r in rows}):
    f1 = [float(r[12]) for r in rows if int(r[0]) == t]
    print(f"task {t}: f1_mac cuoi {f1[-1]:.2f} | max {max(f1):.2f} | {len(f1)} round")
cal = os.path.join(run, "calibration.csv")
if os.path.exists(cal):
    print("\nHIEU CHINH QUYET DINH cuoi moi task (tau chon tren du lieu client, khong dung tap test):")
    print(f"{'task':>4}{'tau*':>7}{'f1 goc':>9}{'f1 hieu chinh':>15}{'acc goc':>9}{'acc hc':>8}")
    crow = list(csv.DictReader(open(cal, encoding="utf-8")))
    for t in sorted({int(r["task"]) for r in crow}):
        rt = [r for r in crow if int(r["task"]) == t]
        raw = next(r for r in rt if float(r["tau"]) == 0.0)
        sel = next(r for r in rt if r["selected"] == "1")
        print(f"{t:>4}{float(sel['tau']):>7.2f}{float(raw['test_f1_macro']):>9.2f}{float(sel['test_f1_macro']):>15.2f}"
              f"{float(raw['test_acc']):>9.2f}{float(sel['test_acc']):>8.2f}")

done = sorted(glob.glob(os.path.join(run, "checkpoints", "ckpt_task*_memory_client99.pth")))
if done:
    print(f"\n[chay tiep] checkpoint cuoi task moi nhat: {os.path.basename(done[-1])} (+ _MEM.pth).")
    print("  Save Version notebook nay, o phien sau Add Input output cua no, RESUME='auto', RESUME_TASK=None.")
else:
    print("\n[chay tiep] chua co checkpoint cuoi task nao (chua xong task 1 + pha bo nho).")

# Zip nhe: log, CSV, PNG va checkpoint CUOI TASK (de chay tiep). Checkpoint tung round van nam
# trong /kaggle/working cua output nhung khong dua vao zip cho gon.
zpath = os.path.join(WORK, f"ketqua_{CONFIG_NAME}{'_resume' if RESUME else ''}.zip")
with zipfile.ZipFile(zpath, "w", allowZip64=True) as z:
    for root, _, files in os.walk(run):
        for fn in sorted(files):
            if fn.endswith(".pth") and not re.match(r"ckpt_task\d+_memory_client99(_MEM)?\.pth$", fn):
                continue
            full = os.path.join(root, fn)
            arc = os.path.join(os.path.basename(run), os.path.relpath(full, run))
            z.write(full, arc, compress_type=zipfile.ZIP_STORED if fn.endswith(".pth") else zipfile.ZIP_DEFLATED)
print(f"[zip] {zpath} ({os.path.getsize(zpath) / 1e9:.2f} GB)")
