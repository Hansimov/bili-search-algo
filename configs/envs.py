from pathlib import Path

from tclogger import OSEnver

REPO_ROOT = Path(__file__).parents[1]
CONFIGS_ROOT = REPO_ROOT / "configs"
ENVS_PATH = CONFIGS_ROOT / "envs.json"
LOG_ROOT = REPO_ROOT / "logs"

ENVS_ENVER = OSEnver(ENVS_PATH)
LOG_ENVS = ENVS_ENVER["logs"]

SECRETS_PATH = CONFIGS_ROOT / "secrets.json"
SECRETS = OSEnver(SECRETS_PATH)
MONGO_ENVS = SECRETS["mongo"]
ELASTIC_ENVS = SECRETS["elastic"]


FASTTEXT_CKPT_ROOT = REPO_ROOT / "models" / "fasttext" / "checkpoints"
