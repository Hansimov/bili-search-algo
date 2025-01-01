from pathlib import Path

from tclogger import OSEnver

REPO_ROOT = Path(__file__).parents[1]
CONFIGS_ROOT = REPO_ROOT / "configs"
ENVS_PATH = CONFIGS_ROOT / "envs.json"
LOG_ROOT = REPO_ROOT / "logs"

ENVS_ENVER = OSEnver(ENVS_PATH)
LOGS_ENVS = ENVS_ENVER["logs"]
PYRO_ENVS = ENVS_ENVER["pyro"]

SECRETS_PATH = CONFIGS_ROOT / "secrets.json"
SECRETS = OSEnver(SECRETS_PATH)
MONGO_ENVS = SECRETS["mongo"]
ELASTIC_ENVS = SECRETS["elastic"]

FASTTEXT_CKPT_ROOT = REPO_ROOT / "models" / "fasttext" / "checkpoints"
SENTENCEPIECE_CKPT_ROOT = REPO_ROOT / "models" / "sentencepiece" / "checkpoints"
TOKEN_FREQ_ROOT = REPO_ROOT

SP_MERGED_MODEL_PREFIX = "sp_merged"
SP_MERGED_MODEL_PATH = SENTENCEPIECE_CKPT_ROOT / f"{SP_MERGED_MODEL_PREFIX}.model"
