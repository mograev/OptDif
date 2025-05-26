from pathlib import Path

# Try to locate the DNGO training script
try:
    DNGO_TRAIN_FILE = str(Path(__path__[0]) / "bo/dngo_train.py")
    GP_TRAIN_FILE = str(Path(__path__[0]) / "bo/gp_train.py")
    BO_OPT_FILE = str(Path(__path__[0]) / "bo/opt.py")
    GBO_TRAIN_FILE = str(Path(__path__[0]) / "gbo/gbo_train.py")
    GBO_OPT_FILE = str(Path(__path__[0]) / "gbo/gbo_opt.py")
    ENTMOOT_TRAIN_OPT_FILE = str(Path(__path__[0]) / "entmoot/entmoot_train_opt.py")
except:
    pass