import csv, glob, json, os, re, shutil, subprocess, sys, time, zipfile

# CELL CHAY MOT PHUONG AN (moi nhanh git = mot phuong an, mot phien Kaggle chay du task 2-5).
# Chi can doi BRANCH (va SCENARIO neu chay few-shot). Moi thu khac tu suy ra.
BRANCH = "exp/doi-chung"   # exp/doi-chung | exp/q-cai-thien | exp/fedrs | exp/fedlc | exp/logit-prior
SCENARIO = "full"          # "full" | "1percent" | "10shot"

CONFIGS = {"exp/doi-chung": "exp_doi_chung", "exp/q-cai-thien": "exp_q_cai_thien",
           "exp/fedrs": "exp_fedrs", "exp/fedlc": "exp_fedlc", "exp/logit-prior": "exp_logit_prior"}
REPO = "https://github.com/TongXuanVu/AFSIC-IOV.git"
WORK = "/kaggle/working"
CODE = os.path.join(WORK, "AFSIC-IOV")
assert BRANCH in CONFIGS and SCENARIO in ("full", "1percent", "10shot")
BASE_NAME = CONFIGS[BRANCH]
CONFIG_NAME = f"{BASE_NAME}_{SCENARIO}"
CONFIG = f"configs/exps/{CONFIG_NAME}.json"

mdl = glob.glob("/kaggle/input/**/ckpt_task00_memory_client99.pth", recursive=True)
mem = glob.glob("/kaggle/input/**/ckpt_task00_memory_client99_MEM.pth", recursive=True)
assert mdl and mem, "Chua Add Input cap file ckpt_task00_memory_client99.pth + ..._MEM.pth"
MDL, MEM = mdl[0], mem[0]
assert os.path.getsize(MEM) > 50e6, f"File MEM qua nho ({os.path.getsize(MEM)/1e6:.0f} MB) - sai file?"
assert glob.glob("/kaggle/input/**/federated_data/client_*_task_*.pt", recursive=True) and \
       glob.glob("/kaggle/input/**/global_test_data.pt", recursive=True), "Thieu dataset IoV 100 client"
print("[input]", MDL, "|", MEM)

shutil.rmtree(CODE, ignore_errors=True)
subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", BRANCH, REPO, CODE], check=True)
os.chdir(CODE)
COMMIT = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
print("[code]", BRANCH, "@", COMMIT, "|", subprocess.check_output(["git", "log", "--oneline", "-1"]).decode().strip())

cfg = json.load(open(f"configs/exps/{BASE_NAME}.json", encoding="utf-8"))
cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
if SCENARIO == "1percent":
    cfg.update(fewshot_enabled=True, kshot=0.01, eval_every_n_rounds=5)
elif SCENARIO == "10shot":
    cfg.update(fewshot_enabled=True, kshot=10, eval_every_n_rounds=5)
cfg.pop("max_tasks", None)                      # chay du task 1..4 trong MOT phien
json.dump(cfg, open(CONFIG, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# Phan chung cua moi phuong an + nen bam bai
for k, v in {"shared_encoder_trainable": True, "single_shared_encoder": True, "aggregate_encoder_only": True,
             "adapter_input": "raw", "personalized_adapter": True, "prototype_classifier": True,
             "proto_q_mode": "class_recall", "decision_calibration": True, "log_client_recall": True,
             "memory_ratio": 0.01, "lr": 3e-05, "num_clients": 100, "num_rounds": 30}.items():
    assert cfg.get(k) == v, f"{k}={cfg.get(k)!r}, can {v!r}"
_src = "".join(open(f, encoding="utf-8").read() for f in
               ("trainer.py", "models/afsic_ids.py", "utils/inc_net.py", "utils/aggregation.py"))
for _f in ("_decision_calibration", "log_client_recall", "single_shared_encoder", "aggregate_encoder_only"):
    assert _f in _src, f"Code chua co {_f} - sai nhanh?"
print(f"[config] {CONFIG_NAME} | quality_mode={cfg.get('quality_mode', 'acc')} | "
      f"local_loss_mode={cfg.get('local_loss_mode', 'ce')} | class_weight_power={cfg.get('class_weight_power')} | "
      f"fewshot={cfg['fewshot_enabled']} kshot={cfg['kshot']}")

t0 = time.time()
subprocess.run([sys.executable, "main.py", "--config", CONFIG, "--resume", MDL, "--resume_memory", MEM], check=False)
print(f"\n[xong] {(time.time()-t0)/3600:.2f} gio")

runs = sorted(glob.glob("logs/afsic-iov_federated/can_iov/*_clients100"), key=os.path.getmtime)
assert runs, "Khong thay thu muc log"
run = runs[-1]
open(os.path.join(run, "CODE_COMMIT.txt"), "w").write(f"{BRANCH} @ {COMMIT}\nconfig {CONFIG_NAME}\n")
rows = [r for r in csv.reader(open(os.path.join(run, "metrics_round_by_round.csv"), encoding="utf-8-sig"))
        if r and r[0] != "task" and len(r) >= 15]
for t in sorted({int(r[0]) for r in rows}):
    f1 = [float(r[12]) for r in rows if int(r[0]) == t]
    print(f"task {t}: f1_mac cuoi {f1[-1]:.2f} | max {max(f1):.2f}")
cal = os.path.join(run, "calibration.csv")
if os.path.exists(cal):
    print("\nHIEU CHINH QUYET DINH (tau chon tren du lieu client, khong dung tap test):")
    print(f"{'task':>4}{'tau*':>7}{'f1 goc':>9}{'f1 hieu chinh':>15}{'acc goc':>9}{'acc hc':>8}")
    crow = list(csv.DictReader(open(cal, encoding="utf-8")))
    for t in sorted({int(r["task"]) for r in crow}):
        rt = [r for r in crow if int(r["task"]) == t]
        raw = next(r for r in rt if float(r["tau"]) == 0.0)
        sel = next(r for r in rt if r["selected"] == "1")
        print(f"{t:>4}{float(sel['tau']):>7.2f}{float(raw['test_f1_macro']):>9.2f}{float(sel['test_f1_macro']):>15.2f}"
              f"{float(raw['test_acc']):>9.2f}{float(sel['test_acc']):>8.2f}")

zpath = os.path.join(WORK, f"ketqua_{CONFIG_NAME}.zip")
with zipfile.ZipFile(zpath, "w", allowZip64=True) as z:
    for root, _, files in os.walk(run):
        for fn in sorted(files):
            if re.match(r"ckpt_task\d+_memory_client(?!99)\d+", fn):
                continue
            full = os.path.join(root, fn)
            arc = os.path.join(os.path.basename(run), os.path.relpath(full, run))
            z.write(full, arc, compress_type=zipfile.ZIP_STORED if fn.endswith(".pth") else zipfile.ZIP_DEFLATED)
print(f"[zip] {zpath} ({os.path.getsize(zpath)/1e9:.2f} GB)")
