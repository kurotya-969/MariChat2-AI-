"""
設定管理モジュール (Together AI API対応版)
Configuration management module for Together AI API
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()

class Config:
    """アプリケーション設定クラス"""

    # --- API設定 ---
    # Groq APIキー
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    # Together AI APIキー
    TOGETHER_API_KEY: Optional[str] = os.getenv("TOGETHER_API_KEY")

    # --- モード設定 ---
    # デバッグモード (trueにすると一部ログの出力先がコンソールのみになります)
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"

    # --- バッチ処理設定 ---
    # 手紙を生成する時刻のリスト（深夜2時、3時、4時）
    BATCH_SCHEDULE_HOURS: list = [
        int(h.strip()) for h in os.getenv("BATCH_SCHEDULE_HOURS", "2,3,4").split(",")
    ]

    # --- 制限設定 ---
    # ユーザーごとの1日の最大リクエスト数
    MAX_DAILY_REQUESTS: int = int(os.getenv("MAX_DAILY_REQUESTS", "1"))

    # --- ストレージ設定 ---
    # ユーザーデータや手紙を保存するメインのファイルパス
    STORAGE_PATH: str = os.getenv("STORAGE_PATH", "tmp/letters.json")
    # バックアップデータの保存先ディレクトリ
    BACKUP_PATH: str = os.getenv("BACKUP_PATH", "tmp/backup")

    # --- ログ設定 ---
    # アプリケーションのログレベル (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    # ログの出力フォーマット
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # --- UI設定 ---
    # Streamlitアプリケーションが使用するポート番号（Render対応でPORT環境変数を優先）
    STREAMLIT_PORT: int = int(os.getenv("PORT", os.getenv("STREAMLIT_PORT", "7860")))

    # --- セキュリティ設定 ---
    # ユーザーセッションのタイムアウト時間（秒単位）
    SESSION_TIMEOUT: int = int(os.getenv("SESSION_TIMEOUT", "3600"))  # デフォルト: 1時間

    # --- 非同期処理設定 ---
    # 非同期での手紙生成を有効にするか
    ASYNC_LETTER_ENABLED: bool = os.getenv("ASYNC_LETTER_ENABLED", "true").lower() == "true"
    # 手紙生成プロセスのタイムアウト時間（秒単位）
    GENERATION_TIMEOUT: int = int(os.getenv("GENERATION_TIMEOUT", "300"))  # デフォルト: 5分
    # 同時に実行可能な最大手紙生成数
    MAX_CONCURRENT_GENERATIONS: int = int(os.getenv("MAX_CONCURRENT_GENERATIONS", "3"))

    # --- AIモデル設定 ---
    # 手紙の論理構造を生成するためのGroqモデル
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "compound-beta")
    # 手紙の感情表現を生成するためのTogether AIモデル
    TOGETHER_API_MODEL: str = os.getenv("TOGETHER_API_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507-tput")

    # --- コンテンツ設定 ---
    # ユーザーに提示する選択可能なテーマのリスト
    AVAILABLE_THEMES: list = [
        "春の思い出", "夏の夜空", "秋の風景", "冬の静寂",
        "友情について", "家族への感謝", "秘めた恋心", "仕事のやりがい",
        "最近ハマっている趣味", "忘れられない旅行"
    ]

    @classmethod
    def validate_config(cls) -> bool:
        """
        設定値の妥当性をチェックし、問題があればエラーログを出力するクラスメソッド
        
        Returns:
            bool: 設定がすべて有効な場合はTrue、そうでなければFalse
        """
        errors = []

        if not cls.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is not set. Please add it to your .env file.")

        if not cls.TOGETHER_API_KEY:
            errors.append("TOGETHER_API_KEY is not set. Please add it to your .env file.")

        if not all(isinstance(h, int) and h in range(24) for h in cls.BATCH_SCHEDULE_HOURS):
            errors.append(f"BATCH_SCHEDULE_HOURS contains invalid values: {cls.BATCH_SCHEDULE_HOURS}")

        if cls.MAX_CONCURRENT_GENERATIONS < 1:
            errors.append("MAX_CONCURRENT_GENERATIONS must be at least 1.")

        if cls.GENERATION_TIMEOUT < 60:
            errors.append("GENERATION_TIMEOUT must be at least 60 seconds.")
        
        if cls.MAX_DAILY_REQUESTS < 1:
            errors.append("MAX_DAILY_REQUESTS must be at least 1.")

        if errors:
            logging.basicConfig(level=logging.ERROR, format=cls.LOG_FORMAT)
            for error in errors:
                logging.error(f"Configuration validation error: {error}")
            return False

        return True

    @classmethod
    def get_log_level(cls) -> int:
        """
        ログレベルの文字列をloggingモジュールの定数に変換するクラスメソッド
        
        Returns:
            int: loggingモジュールで定義されているログレベル定数
        """
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        # 指定されたログレベルが存在しない場合はINFOをデフォルトとする
        return level_map.get(cls.LOG_LEVEL.upper(), logging.INFO)