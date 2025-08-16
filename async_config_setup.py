"""
非同期手紙生成システムの設定初期化
Async Letter Generation System Configuration Setup
"""

import os
import sys
from pathlib import Path
from letter_config import Config
from letter_logger import get_app_logger

logger = get_app_logger()

def initialize_config() -> bool:
    """
    設定を初期化し、必要なディレクトリを作成する
    
    Returns:
        初期化が成功したかどうか
    """
    try:
        # 設定の妥当性をチェック
        if not Config.validate_config():
            logger.error("設定の検証に失敗しました")
            return False
        
        # 必要なディレクトリを作成
        storage_path = Path(Config.STORAGE_PATH)
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        backup_path = Path(Config.BACKUP_PATH)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # ログディレクトリを作成（本番環境用）
        if not Config.DEBUG_MODE:
            log_dir = Path("/tmp")
            log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("設定初期化が完了しました")
        logger.info(f"ストレージパス: {Config.STORAGE_PATH}")
        logger.info(f"バックアップパス: {Config.BACKUP_PATH}")
        logger.info(f"デバッグモード: {Config.DEBUG_MODE}")
        logger.info(f"バッチ処理時刻: {Config.BATCH_SCHEDULE_HOURS}")
        
        return True
        
    except Exception as e:
        logger.error(f"設定初期化エラー: {e}")
        return False

def check_api_keys() -> bool:
    """
    API キーの存在をチェックする
    
    Returns:
        API キーが設定されているかどうか
    """
    try:
        missing_keys = []
        
        if not Config.GROQ_API_KEY:
            missing_keys.append("GROQ_API_KEY")
        
        if not Config.GEMINI_API_KEY:
            missing_keys.append("GEMINI_API_KEY")
        
        if missing_keys:
            logger.error(f"必要なAPI キーが設定されていません: {', '.join(missing_keys)}")
            return False
        
        logger.info("API キーの確認が完了しました")
        return True
        
    except Exception as e:
        logger.error(f"API キー確認エラー: {e}")
        return False

def setup_environment() -> bool:
    """
    環境をセットアップする
    
    Returns:
        セットアップが成功したかどうか
    """
    try:
        # 設定を初期化
        if not initialize_config():
            return False
        
        # API キーをチェック
        if not check_api_keys():
            return False
        
        # Hugging Face Spaces 固有の設定
        if os.getenv("SPACE_ID"):
            logger.info(f"Hugging Face Spaces環境で実行中: {os.getenv('SPACE_ID')}")
            
            # Spaces用の追加設定があればここに記述
            
        logger.info("環境セットアップが完了しました")
        return True
        
    except Exception as e:
        logger.error(f"環境セットアップエラー: {e}")
        return False

def get_system_info() -> dict:
    """
    システム情報を取得する
    
    Returns:
        システム情報の辞書
    """
    return {
        "python_version": sys.version,
        "storage_path": Config.STORAGE_PATH,
        "backup_path": Config.BACKUP_PATH,
        "debug_mode": Config.DEBUG_MODE,
        "batch_hours": Config.BATCH_SCHEDULE_HOURS,
        "max_daily_requests": Config.MAX_DAILY_REQUESTS,
        "generation_timeout": Config.GENERATION_TIMEOUT,
        "max_concurrent_generations": Config.MAX_CONCURRENT_GENERATIONS,
        "groq_model": Config.GROQ_MODEL,
        "Together_model": Config.TOGETHER_API_MODEL,
        "available_themes": Config.AVAILABLE_THEMES,
        "space_id": os.getenv("SPACE_ID", "local"),
        "streamlit_port": Config.STREAMLIT_PORT
    }

if __name__ == "__main__":
    # スクリプトとして実行された場合の設定確認
    print("非同期手紙生成システム設定確認")
    print("=" * 50)
    
    if setup_environment():
        print("✅ 環境セットアップ成功")
        
        system_info = get_system_info()
        for key, value in system_info.items():
            print(f"{key}: {value}")
    else:
        print("❌ 環境セットアップ失敗")
        sys.exit(1)