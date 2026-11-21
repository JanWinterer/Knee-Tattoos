"""
Load the SKEL model from models/skel/ (or from the SKEL_MODEL_PATH environment variable).
"""
import os
from pathlib import Path
from skel.skel_model import SKEL

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get('SKEL_MODEL_PATH', REPO_ROOT / 'models' / 'skel'))


def load_skel(gender='male', model_path=None):
    model_dir = Path(model_path) if model_path else MODEL_DIR
    pkl = model_dir / f'skel_{gender}.pkl'
    if not pkl.exists():
        raise FileNotFoundError(
            f'{pkl} not found. Download skel_models_v1.1.zip from https://skel.is.tue.mpg.de/download.php '
            f'(free account, non-commercial licence) and put skel_female.pkl and skel_male.pkl in {model_dir}/')
    return SKEL(gender=gender, model_path=str(model_dir))
