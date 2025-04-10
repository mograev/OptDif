from pathlib import Path

# Try to locate the DNGO training script
try:
    DNGO_TRAIN_FILE = str(Path(__path__[0]) / "bo/dngo_train.py")
    GP_TRAIN_FILE = str(Path(__path__[0]) / "bo/gp_train.py")
    OPT_FILE = str(Path(__path__[0]) / "bo/opt.py")
except:
    pass