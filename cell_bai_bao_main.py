import csv, glob, json, os, shutil, subprocess, sys, time

# CELL CHAY LAI THEO DUNG BAI (quyet_dinh_viet_bai_AFSIC-IoV.md muc 1): cau hinh configs/exps/afsic_bai_bao.json
SCENARIO = "full"   # "full" | "1percent" | "10shot"
PHASE = 1           # 1 = task 1-2 (nap checkpoint task 0), 2 = task 3-4 (nap checkpoint task 2 do phien 1 tao ra)
REPO, COMMIT = "https://github.com/TongXuanVu/AFSIC-IOV.git", "0392133"
BASE_NAME = "afsic_bai_bao"
CONFIG_NAME = f"{BASE_NAME}_{SCENARIO}"
CONFIG = f"configs/exps/{CONFIG_NAME}.json"
WORK = "/kaggle/working"
CODE = os.path.join(WORK, "AFSIC-IOV")
HANDOFF = f"handoff_{CONFIG_NAME}"                # thu muc ban giao checkpoint task 2 giua hai phien

assert SCENARIO in ("full", "1percent", "10shot") and PHASE in (1, 2)
if PHASE == 1:
    mdl = glob.glob("/kaggle/input/**/ckpt_task00_memory_client99.pth", recursive=True)
    mem = glob.glob("/kaggle/input/**/ckpt_task00_memory_client99_MEM.pth", recursive=True)
    assert mdl and mem, "Chua Add Input cap file ckpt_task00_memory_client99.pth + ..._MEM.pth"
else:
    mdl = glob.glob(f"/kaggle/input/**/{HANDOFF}/ckpt_task02_memory_client99.pth", recursive=True)
    mem = glob.glob(f"/kaggle/input/**/{HANDOFF}/ckpt_task02_memory_client99_MEM.pth", recursive=True)
    assert mdl and mem, (f"Chua Add Input thu muc {HANDOFF}/ (ckpt_task02_memory_client99.pth + _MEM.pth) "
                         f"do PHASE 1 cua CUNG SCENARIO {SCENARIO} tao ra")
MDL, MEM = mdl[0], mem[0]
assert os.path.getsize(MEM) > 50e6, f"File MEM qua nho ({os.path.getsize(MEM)/1e6:.0f} MB) - sai file?"
assert glob.glob("/kaggle/input/**/federated_data/client_*_task_*.pt", recursive=True) and \
       glob.glob("/kaggle/input/**/global_test_data.pt", recursive=True), "Thieu dataset IoV 100 client"
print("[input]", MDL, "|", MEM)

shutil.rmtree(CODE, ignore_errors=True)
subprocess.run(["git", "clone", "-q", REPO, CODE], check=True)
subprocess.run(["git", "-C", CODE, "checkout", "-q", COMMIT], check=True)
os.chdir(CODE)
print("[code]", subprocess.check_output(["git", "-C", CODE, "log", "--oneline", "-1"]).decode().strip())

cfg = json.load(open(f"configs/exps/{BASE_NAME}.json", encoding="utf-8"))
cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
if SCENARIO == "1percent":
    cfg.update(fewshot_enabled=True, kshot=0.01, eval_every_n_rounds=5)
elif SCENARIO == "10shot":
    cfg.update(fewshot_enabled=True, kshot=10, eval_every_n_rounds=5)
if PHASE == 1:
    cfg["max_tasks"] = 3                # task 0 (nap lai) + task 1, 2; van dung bo nho cuoi task 2
json.dump(cfg, open(CONFIG, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

for k, v in {"shared_encoder_trainable": True, "single_shared_encoder": True, "aggregate_encoder_only": True,
             "adapter_input": "raw", "adapter_bottleneck": 64, "personalized_adapter": True,
             "prototype_classifier": True, "client_personalized_classifier": True,
             "proto_max_samples": None, "quality_eval_max_samples": None,
             "ce_reduction": "sum", "proto_q_mode": "class_recall", "ckpt_every_n_rounds": 1,
             "memory_ratio": 0.01, "lr": 3e-05}.items():
    assert cfg.get(k) == v, f"{k}={cfg.get(k)!r}, can {v!r}"
assert "replay_old_ratio" not in cfg
_src = open("trainer.py", encoding="utf-8").read() + open("models/afsic_ids.py", encoding="utf-8").read() + open("utils/inc_net.py", encoding="utf-8").read() + open("utils/aggregation.py", encoding="utf-8").read()
for _f in ("single_shared_encoder", "aggregate_encoder_only", "ce_reduction", "proto_q_mode", "_quality_eval_per_class", "client_personalized_classifier"):
    assert _f in _src, f"Code chua co {_f} - kiem tra COMMIT"
print(f"[config] {CONFIG_NAME} PHASE {PHASE} max_tasks={cfg.get('max_tasks')} fewshot={cfg['fewshot_enabled']} kshot={cfg['kshot']}")

t0 = time.time()
subprocess.run([sys.executable, "main.py", "--config", CONFIG, "--resume", MDL, "--resume_memory", MEM], check=False)
print(f"\n[xong] {(time.time()-t0)/3600:.2f} gio")

runs = sorted(glob.glob("logs/afsic-iov_federated/can_iov/*_clients100"), key=os.path.getmtime)
assert runs, "Khong thay thu muc log"
run = runs[-1]
rows = [r for r in csv.reader(open(os.path.join(run, "metrics_round_by_round.csv"), encoding="utf-8-sig"))
        if r and r[0] != "task" and len(r) >= 15]
print(f"\n{'task':>4}{'round':>6}{'acc':>8}{'f1_mac':>8}{'f1_wei':>8}")
for r in rows:
    print(f"{int(r[0]):>4}{int(r[1]):>6}{float(r[4]):>8.2f}{float(r[12]):>8.2f}{float(r[13]):>8.2f}")
for t in sorted({int(r[0]) for r in rows}):
    f1 = [float(r[12]) for r in rows if int(r[0]) == t]
    print(f"task {t}: max f1_mac {max(f1):.2f} | cuoi {f1[-1]:.2f}")
for cm in sorted(glob.glob(os.path.join(run, "confusion_matrix_task_*.csv"))):   # F1/recall tung lop cuoi moi task
    rows3 = open(cm, encoding="utf-8-sig").read().split("[3] TUNG LOP")[1].strip().splitlines()[1:]
    print(os.path.basename(cm)[:-4], " | ".join(f"{l.split(',')[0]} {float(l.split(',')[6]):.1f}/{float(l.split(',')[5]):.1f}" for l in rows3))

if PHASE == 1:                               # ban giao checkpoint task 2 cho phien 2
    hd = os.path.join(WORK, HANDOFF); os.makedirs(hd, exist_ok=True)
    for f in ("ckpt_task02_memory_client99.pth", "ckpt_task02_memory_client99_MEM.pth"):
        src = os.path.join(run, "checkpoints", f)
        assert os.path.exists(src), f"Khong thay {src} - phien 1 chua chay xong pha memory task 2"
        shutil.copy2(src, hd)
    print(f"[ban giao] {hd}/ -> Save Version notebook nay, roi o phien 2 Add Input output cua no (giu ten thu muc {HANDOFF})")

# Nen log + checkpoint. File .pth ghi kieu STORED (da nen san, deflate chi ton thoi gian).
# Bo cac file memory trung gian (client09..89): ban client99 da chua du ca 100 client.
import re, zipfile
zpath = os.path.join(WORK, f"ketqua_{CONFIG_NAME}_phase{PHASE}.zip")
with zipfile.ZipFile(zpath, "w", allowZip64=True) as z:
    for root, _, files in os.walk(run):
        for fn in sorted(files):
            if re.match(r"ckpt_task\d+_memory_client(?!99)\d+", fn):
                continue
            full = os.path.join(root, fn)
            arc = os.path.join(os.path.basename(run), os.path.relpath(full, run))
            z.write(full, arc, compress_type=zipfile.ZIP_STORED if fn.endswith(".pth") else zipfile.ZIP_DEFLATED)
print(f"[zip] {zpath} ({os.path.getsize(zpath)/1e9:.2f} GB, gom checkpoint)")
