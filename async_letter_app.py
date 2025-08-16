"""
メインアプリケーションモジュール
Main application module
"""

import asyncio
import sys
from pathlib import Path

# このファイルが依存している他のモジュールをインポート
from letter_config import Config
from letter_logger import get_app_logger

# このモジュール用のロガーを取得
logger = get_app_logger()

class LetterApp:
    """非同期手紙生成アプリケーションのメインクラス"""
    
    def __init__(self):
        """アプリケーションを初期化"""
        self.config = Config()
        self.logger = logger
        self._initialized = False
    
    async def initialize(self) -> bool:
        """
        アプリケーションを初期化する
        
        Returns:
            初期化が成功したかどうか
        """
        try:
            self.logger.info("アプリケーションを初期化中...")
            
            # 設定の妥当性をチェック
            if not self.config.validate_config():
                self.logger.error("設定の検証に失敗しました")
                return False
            
            # ストレージディレクトリを作成
            await self._setup_storage_directories()
            
            # ログディレクトリを作成
            await self._setup_log_directories()
            
            self._initialized = True
            self.logger.info("アプリケーションの初期化が完了しました")
            return True
            
        except Exception as e:
            self.logger.error(f"アプリケーションの初期化中にエラーが発生しました: {e}")
            return False
    
    async def _setup_storage_directories(self):
        """ストレージディレクトリを作成"""
        storage_dir = Path(self.config.STORAGE_PATH).parent
        backup_dir = Path(self.config.BACKUP_PATH)
        
        storage_dir.mkdir(parents=True, exist_ok=True)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"ストレージディレクトリを作成: {storage_dir}")
        self.logger.info(f"バックアップディレクトリを作成: {backup_dir}")
    
    async def _setup_log_directories(self):
        """ログディレクトリを作成"""
        if not self.config.DEBUG_MODE:
            # ログファイルは個別のロガーで設定されるため、
            # ここでは共通の親ディレクトリを作成する例
            log_dir = Path("/tmp/logs") # 例: /tmp/batch.log などの親
            log_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"ログディレクトリを作成: {log_dir}")
    
    def is_initialized(self) -> bool:
        """アプリケーションが初期化されているかチェック"""
        return self._initialized
    
    def get_config(self) -> Config:
        """設定オブジェクトを取得"""
        return self.config

# グローバルアプリケーションインスタンス（シングルトンパターン）
app_instance = None

async def get_app() -> LetterApp:
    """アプリケーションインスタンスを取得（シングルトン）"""
    global app_instance
    
    if app_instance is None:
        app_instance = LetterApp()
        await app_instance.initialize()
    
    return app_instance

def run_app():
    """アプリケーションを実行（テスト用）"""
    async def main():
        app = await get_app()
        if app.is_initialized():
            logger.info("アプリケーションが正常に起動しました")
        else:
            logger.error("アプリケーションの起動に失敗しました")
            sys.exit(1)
    
    asyncio.run(main())

# このファイルが直接実行された場合にテスト用の起動処理を行う
if __name__ == "__main__":
    run_app()