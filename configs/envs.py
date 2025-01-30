from pathlib import Path

from tclogger import OSEnver

REPO_ROOT = Path(__file__).parents[1]
CONFIGS_ROOT = REPO_ROOT / "configs"
ENVS_PATH = CONFIGS_ROOT / "envs.json"
DATA_ROOT = REPO_ROOT / "data"
LOG_ROOT = REPO_ROOT / "logs"

ENVS_ENVER = OSEnver(ENVS_PATH)
LOGS_ENVS = ENVS_ENVER["logs"]
PYRO_ENVS = ENVS_ENVER["pyro"]

SECRETS_PATH = CONFIGS_ROOT / "secrets.json"
SECRETS = OSEnver(SECRETS_PATH)
MONGO_ENVS = SECRETS["mongo"]
ELASTIC_ENVS = SECRETS["elastic"]

SENTENCEPIECE_CKPT_ROOT = REPO_ROOT / "models" / "sentencepiece" / "checkpoints"
SP_MERGED_MODEL_PREFIX = "sp_merged"
SP_MERGED_MODEL_PATH = SENTENCEPIECE_CKPT_ROOT / f"{SP_MERGED_MODEL_PREFIX}.model"

TOKEN_FREQS_ROOT = DATA_ROOT / "token_freqs"
TOKEN_FREQ_PREFIX = "merged_video_texts"

FASTTEXT_CKPT_ROOT = REPO_ROOT / "models" / "fasttext" / "checkpoints"
FASTTEXT_MERGED_MODEL_PREFIX = "fasttext_merged"
FASTTEXT_MERGED_MODEL_DIMENSION = 320
