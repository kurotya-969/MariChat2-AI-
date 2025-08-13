"""
設定管理モジュール
Configuration management module
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()

class Config:
    """アプリケーション設定クラス"""
    
    # API設定
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    
    # デバッグモード
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
    
    # バッチ処理設定
    BATCH_SCHEDULE_HOURS: list = [
        int(h.strip()) for h in os.getenv("BATCH_SCHEDULE_HOURS", "2,3,4").split(",")
    ]
    
    # レート制限設定
    MAX_DAILY_REQUESTS: int = int(os.getenv("MAX_DAILY_REQUESTS", "1"))
    
    # ストレージ設定
    STORAGE_PATH: str = os.getenv("STORAGE_PATH", "/mnt/data/letters.json")
    BACKUP_PATH: str = os.getenv("BACKUP_PATH", "/mnt/data/backup")
    
    # ログ設定
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Streamlit設定
    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    
    # セキュリティ設定
    SESSION_TIMEOUT: int = int(os.getenv("SESSION_TIMEOUT", "3600"))  # 1時間
    
    @classmethod
    def validate_config(cls) -> bool:
        """設定の妥当性をチェック"""
        errors = []
        
        if not cls.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is required")
        
        if not cls.GEMINI_API_KEY:
            errors.append("GEMINI_API_KEY is required")
        
        if not all(h in [2, 3, 4] for h in cls.BATCH_SCHEDULE_HOURS):
            errors.append("BATCH_SCHEDULE_HOURS must contain only 2, 3, or 4")
        
        if errors:
            for error in errors:
                logging.error(f"Configuration error: {error}")
            return False
        
        return True
    
    @classmethod
    def get_log_level(cls) -> int:
        """ログレベルを取得"""
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        return level_map.get(cls.LOG_LEVEL.upper(), logging.INFO)