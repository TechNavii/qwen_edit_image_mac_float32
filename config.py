import os
from pathlib import Path

class Config:
    MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen-VL-Chat")
    
    DEVICE = os.getenv("DEVICE", "auto")
    
    CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "~/.cache/huggingface")).expanduser()
    
    SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))
    
    SHARE_GRADIO = os.getenv("SHARE_GRADIO", "false").lower() == "true"
    
    MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "2048"))
    
    TEMP_DIR = Path(os.getenv("TEMP_DIR", "/tmp"))
    
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "512"))
    
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
    
    USE_HALF_PRECISION = os.getenv("USE_HALF_PRECISION", "true").lower() == "true"
    
    ENABLE_QUEUE = os.getenv("ENABLE_QUEUE", "true").lower() == "true"
    
    MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "10"))