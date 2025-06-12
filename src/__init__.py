from pathlib import Path

# Try to locate the DNGO training script
try:
    DNGO_TRAIN_FILE = str(Path(__path__[0]) / "bo/dngo_train.py")
    DNGO_PCA_TRAIN_FILE = str(Path(__path__[0]) / "bo/dngo_train_pca.py")
    GP_TRAIN_FILE = str(Path(__path__[0]) / "bo/gp_train.py")
    BO_OPT_FILE = str(Path(__path__[0]) / "bo/opt.py")
    BO_PCA_OPT_FILE = str(Path(__path__[0]) / "bo/opt_pca.py")
    GBO_TRAIN_FILE = str(Path(__path__[0]) / "gbo/gbo_train.py")
    GBO_OPT_FILE = str(Path(__path__[0]) / "gbo/gbo_opt.py")
    GBO_PCA_TRAIN_FILE = str(Path(__path__[0]) / "gbo/gbo_train_pca.py")
    GBO_PCA_OPT_FILE = str(Path(__path__[0]) / "gbo/gbo_opt_pca.py")
    GBO_FI_TRAIN_FILE = str(Path(__path__[0]) / "gbo/gbo_train_fi.py")
    GBO_FI_OPT_FILE = str(Path(__path__[0]) / "gbo/gbo_opt_fi.py")
    ENTMOOT_TRAIN_OPT_FILE = str(Path(__path__[0]) / "entmoot/entmoot_train_opt.py")
except:
    pass