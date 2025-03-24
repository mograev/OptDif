from pathlib import Path

# Try to locate the DNGO training script
try:
    DNGO_TRAIN_FILE = str(Path(__path__[0]) / "dngo/dngo_train.py")
    DNGO_OPT_FILE = str(Path(__path__[0]) / "dngo/dngo_opt.py")
except:
    pass