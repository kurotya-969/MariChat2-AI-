"""
ログ設定ユーティリティ
Logging configuration utility
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from letter_config import Config

def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: Optional[int] = None
) -> logging.Logger:
    """
    ロガーを設定する
    
    Args:
        name: ロガー名
        log_file: ログファイルパス（オプション）
        level: ログレベル（オプション）
    
    Returns:
        設定されたロガー
    """
    logger = logging.getLogger(name)
    
    # 既存のハンドラーをクリア
    logger.handlers.clear()
    
    # ログレベル設定
    if level is None:
        level = Config.get_log_level()
    logger.setLevel(level)
    
    # フォーマッター作成
    formatter = logging.Formatter(Config.LOG_FORMAT)
    
    # コンソールハンドラー
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ファイルハンドラー（指定された場合）
    if log_file:
        # ログディレクトリを作成
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_app_logger() -> logging.Logger:
    """アプリケーション用のロガーを取得"""
    return setup_logger("async_letter_app")

def get_batch_logger() -> logging.Logger:
    """バッチ処理用のロガーを取得"""
    log_file = "/tmp/batch.log" if not Config.DEBUG_MODE else None
    return setup_logger("batch_processor", log_file)

def get_api_logger() -> logging.Logger:
    """API呼び出し用のロガーを取得"""
    log_file = "/tmp/api.log" if not Config.DEBUG_MODE else None
    return setup_logger("api_client", log_file)

def get_storage_logger() -> logging.Logger:
    """ストレージ操作用のロガーを取得"""
    log_file = "/tmp/storage.log" if not Config.DEBUG_MODE else None
    return setup_logger("storage", log_file)