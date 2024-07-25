import platform

DATA_DIR = "D:/data"
if platform.system() != "Windows":
    DATA_DIR = "/home/apps/data"

BASE_DIR = f"{DATA_DIR}/huggingface"
# BASE_DIR = f"{DATA_DIR}/modelscope"
CACHE_DIR = f"{BASE_DIR}/cache"
DS_DIR = f"{BASE_DIR}/datasets"
MODEL_DIR = f"{BASE_DIR}/models"
RESULT_DIR = f"{BASE_DIR}/results"
LOG_DIR = f"{BASE_DIR}/logs"
