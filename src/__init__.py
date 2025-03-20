from pathlib import Path

# Try to locate the DNGO training script
try:
    GP_OPT_FILE = str(Path(__path__[0]) / "gp_opt.py")
    GP_OPT_SAMPLING_FILE = str(Path(__path__[0]) / "gp_opt_sampling.py")
    DNGO_TRAIN_FILE = str(Path(__path__[0]) / "dngo/dngo_train.py")
except:
    pass