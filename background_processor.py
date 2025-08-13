"""
バックグラウンド処理統合クラス
Streamlitアプリと独立したバッチ処理の実行機能と
古いデータの自動削除機能を提供します。
"""

import asyncio
import threading
import time
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
import logging
import traceback
import signal
import sys

from batch_scheduler import BatchScheduler
from async_storage_manager import AsyncStorageManager

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BackgroundProcessorError(Exception):
    """バックグラウンドプロセッサー関連のエラー"""
    pass


class BackgroundProcessor:
    """Streamlitアプリと独立したバックグラウンド処理管理クラス"""
    
    def __init__(self, storage_manager: Optional[AsyncStorageManager] = None):
        """
        バックグラウンドプロセッサーを初期化
        
        Args:
            storage_manager: ストレージマネージャー（指定しない場合は新規作成）
        """
        # ストレージマネージャーの初期化
        if storage_manager is None:
            storage_path = os.getenv("STORAGE_PATH", "/mnt/data/letters.json")
            self.storage_manager = AsyncStorageManager(storage_path)
        else:
            self.storage_manager = storage_manager
        
        # バッチスケジューラーの初期化
        self.batch_scheduler = BatchScheduler(self.storage_manager)
        
        # 設定値
        self.target_hours = [2, 3, 4]  # 2時、3時、4時
        self.check_interval = int(os.getenv("BATCH_CHECK_INTERVAL", "60"))  # 1分間隔
        self.cleanup_hour = int(os.getenv("CLEANUP_HOUR", "1"))  # 1時にクリーンアップ
        self.cleanup_retention_days = int(os.getenv("CLEANUP_RETENTION_DAYS", "90"))
        self.enable_background_processing = os.getenv("ENABLE_BACKGROUND_PROCESSING", "true").lower() == "true"
        
        # 実行状態管理
        self.is_running = False
        self.background_thread = None
        self.stop_event = threading.Event()
        self.last_execution_times = {hour: None for hour in self.target_hours}
        self.last_cleanup_date = None
        
        # コールバック関数
        self.on_batch_complete: Optional[Callable] = None
        self.on_cleanup_complete: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        logger.info(f"BackgroundProcessor初期化完了 - 対象時刻: {self.target_hours}, チェック間隔: {self.check_interval}秒")
    
    def start_background_processing(self) -> bool:
        """
        バックグラウンド処理を開始
        
        Returns:
            bool: 開始成功フラグ
        """
        if not self.enable_background_processing:
            logger.info("バックグラウンド処理は無効化されています")
            return False
        
        if self.is_running:
            logger.warning("バックグラウンド処理は既に実行中です")
            return False
        
        try:
            self.is_running = True
            self.stop_event.clear()
            
            # バックグラウンドスレッドを開始
            self.background_thread = threading.Thread(
                target=self._background_loop,
                name="BackgroundProcessor",
                daemon=True
            )
            self.background_thread.start()
            
            # シグナルハンドラーを設定
            self._setup_signal_handlers()
            
            logger.info("バックグラウンド処理を開始しました")
            return True
            
        except Exception as e:
            self.is_running = False
            logger.error(f"バックグラウンド処理の開始に失敗: {str(e)}")
            return False
    
    def stop_background_processing(self) -> bool:
        """
        バックグラウンド処理を停止
        
        Returns:
            bool: 停止成功フラグ
        """
        if not self.is_running:
            logger.info("バックグラウンド処理は実行されていません")
            return True
        
        try:
            logger.info("バックグラウンド処理の停止を開始します...")
            
            # 停止フラグを設定
            self.stop_event.set()
            self.is_running = False
            
            # スレッドの終了を待機
            if self.background_thread and self.background_thread.is_alive():
                self.background_thread.join(timeout=30)  # 30秒でタイムアウト
                
                if self.background_thread.is_alive():
                    logger.warning("バックグラウンドスレッドの停止がタイムアウトしました")
                    return False
            
            logger.info("バックグラウンド処理を停止しました")
            return True
            
        except Exception as e:
            logger.error(f"バックグラウンド処理の停止に失敗: {str(e)}")
            return False
    
    def start_background_processing(self) -> bool:
        if not self.enable_background_processing:
            logger.info("バックグラウンド処理は無効化されています")
            return False
        if self.is_running:
            logger.warning("バックグラウンド処理は既に実行中です")
            return False
        
        try:
            self.is_running = True
            self.stop_event.clear()
            
            # 変更点 1: スレッドのターゲットを新しいラッパー関数に変更
            self.background_thread = threading.Thread(
                target=self._thread_entry_point,
                name="BackgroundProcessor",
                daemon=True
            )
            self.background_thread.start()
            
            self._setup_signal_handlers()
            logger.info("バックグラウンド処理を開始しました")
            return True
        except Exception as e:
            self.is_running = False
            logger.error(f"バックグラウンド処理の開始に失敗: {e}")
            return False

    def stop_background_processing(self) -> bool:
        if not self.is_running:
            logger.info("バックグラウンド処理は実行されていません")
            return True
        
        try:
            logger.info("バックグラウンド処理の停止を開始します...")
            self.stop_event.set()
            if self.background_thread and self.background_thread.is_alive():
                self.background_thread.join(timeout=10)
                if self.background_thread.is_alive():
                    logger.warning("バックグラウンドスレッドの停止がタイムアウトしました")
                    return False
            
            self.is_running = False
            logger.info("バックグラウンド処理を停止しました")
            return True
        except Exception as e:
            logger.error(f"バックグラウンド処理の停止に失敗: {e}")
            return False

    # 変更点 2: スレッドのエントリーポイントとなる同期ラッパー関数を追加
    def _thread_entry_point(self) -> None:
        """バックグラウンドスレッド内でイベントループを実行するためのラッパー"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            logger.info("バックグラウンドスレッドのイベントループを開始します。")
            loop.run_until_complete(self._background_loop())
        except Exception as e:
            logger.error(f"バックグラウンドイベントループで致命的なエラー: {e}\n{traceback.format_exc()}")
        finally:
            logger.info("バックグラウンドスレッドのイベントループを終了します。")
            loop.close()

    # 変更点 3: メインループを async def に変更
    async def _background_loop(self) -> None:
        """バックグラウンド処理の非同期メインループ"""
        logger.info("非同期バックグラウンド処理ループを開始します")
        while not self.stop_event.is_set():
            try:
                current_time = datetime.now()
                current_hour = current_time.hour
                current_date = current_time.strftime("%Y-%m-%d")

                if current_hour in self.target_hours:
                    await self._check_and_run_batch(current_hour, current_date)
                
                if current_hour == self.cleanup_hour:
                    await self._check_and_run_cleanup(current_date)
                
                # 変更点 4: 待機処理を asyncio.sleep に変更
                # stop_eventをチェックしながら1秒ずつ待機することで、素早い停止を可能にする
                for _ in range(self.check_interval):
                    if self.stop_event.is_set():
                        break
                    await asyncio.sleep(1)

            except Exception as e:
                error_msg = f"バックグラウンド処理ループでエラーが発生: {e}"
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                if self.on_error:
                    try:
                        self.on_error(error_msg)
                    except Exception as callback_error:
                        logger.error(f"エラーコールバック実行エラー: {callback_error}")
                await asyncio.sleep(min(self.check_interval * 2, 300))
        
        logger.info("非同期バックグラウンド処理ループを終了しました")
    
    async def _check_and_run_batch(self, hour: int, date: str) -> None:
        """
        バッチ処理の実行チェックと実行
        
        Args:
            hour: 現在の時刻
            date: 現在の日付
        """
        try:
            # 既に今日実行済みかチェック
            last_execution = self.last_execution_times.get(hour)
            if last_execution and last_execution == date:
                return  # 既に実行済み
            
            logger.info(f"{hour}時のバッチ処理を実行します")
            
            # バッチ処理を実行
            result = await self.batch_scheduler.run_hourly_batch(hour)
            
            # 実行時刻を記録
            self.last_execution_times[hour] = date
            
            # 完了コールバックを呼び出し
            if self.on_batch_complete:
                try:
                    self.on_batch_complete(hour, result)
                except Exception as callback_error:
                    logger.error(f"バッチ完了コールバック実行エラー: {str(callback_error)}")
            
            if result.get("success", False):
                logger.info(f"{hour}時のバッチ処理が完了しました - 処理数: {result.get('processed_count', 0)}")
            else:
                logger.error(f"{hour}時のバッチ処理が失敗しました: {result.get('error', '不明なエラー')}")
            
        except Exception as e:
            error_msg = f"{hour}時のバッチ処理チェック中にエラーが発生: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            if self.on_error:
                try:
                    self.on_error(error_msg)
                except Exception as callback_error:
                    logger.error(f"エラーコールバック実行エラー: {str(callback_error)}")
    
    async def _check_and_run_cleanup(self, date: str) -> None:
        """
        クリーンアップ処理の実行チェックと実行
        
        Args:
            date: 現在の日付
        """
        try:
            # 既に今日実行済みかチェック
            if self.last_cleanup_date == date:
                return  # 既に実行済み
            
            logger.info("古いデータのクリーンアップを実行します")
            
            # クリーンアップ処理を実行
            result = await self.batch_scheduler.cleanup_old_data(self.cleanup_retention_days)
            
            # 実行日を記録
            self.last_cleanup_date = date
            
            # 完了コールバックを呼び出し
            if self.on_cleanup_complete:
                try:
                    self.on_cleanup_complete(result)
                except Exception as callback_error:
                    logger.error(f"クリーンアップ完了コールバック実行エラー: {str(callback_error)}")
            
            if result.get("success", False):
                logger.info(f"クリーンアップが完了しました - 削除数: {result.get('deleted_letters', 0)}")
            else:
                logger.error(f"クリーンアップが失敗しました: {result.get('error', '不明なエラー')}")
            
        except Exception as e:
            error_msg = f"クリーンアップ処理チェック中にエラーが発生: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            if self.on_error:
                try:
                    self.on_error(error_msg)
                except Exception as callback_error:
                    logger.error(f"エラーコールバック実行エラー: {str(callback_error)}")
    
    def _setup_signal_handlers(self) -> None:
        """シグナルハンドラーを設定"""
        def signal_handler(signum, frame):
            logger.info(f"シグナル {signum} を受信しました。バックグラウンド処理を停止します...")
            self.stop_background_processing()
            sys.exit(0)
        
        # SIGTERM と SIGINT のハンドラーを設定
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def get_status(self) -> Dict[str, Any]:
        """
        バックグラウンド処理の状態を取得
        
        Returns:
            Dict: 状態情報
        """
        return {
            "is_running": self.is_running,
            "enable_background_processing": self.enable_background_processing,
            "target_hours": self.target_hours,
            "check_interval": self.check_interval,
            "cleanup_hour": self.cleanup_hour,
            "cleanup_retention_days": self.cleanup_retention_days,
            "last_execution_times": self.last_execution_times.copy(),
            "last_cleanup_date": self.last_cleanup_date,
            "thread_alive": self.background_thread.is_alive() if self.background_thread else False,
            "current_time": datetime.now().isoformat()
        }
    
    async def force_run_batch(self, hour: int) -> Dict[str, Any]:
        """
        指定時刻のバッチ処理を強制実行
        
        Args:
            hour: 実行対象の時刻
            
        Returns:
            Dict: 実行結果
        """
        try:
            if hour not in self.target_hours:
                return {
                    "success": False,
                    "error": f"無効な時刻が指定されました: {hour} (有効: {self.target_hours})"
                }
            
            logger.info(f"{hour}時のバッチ処理を強制実行します")
            
            result = await self.batch_scheduler.run_hourly_batch(hour)
            
            # 実行時刻を記録
            current_date = datetime.now().strftime("%Y-%m-%d")
            self.last_execution_times[hour] = current_date
            
            return result
            
        except Exception as e:
            error_msg = f"バッチ処理の強制実行中にエラーが発生: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {
                "success": False,
                "error": error_msg
            }
    
    async def force_run_cleanup(self) -> Dict[str, Any]:
        """
        クリーンアップ処理を強制実行
        
        Returns:
            Dict: 実行結果
        """
        try:
            logger.info("クリーンアップ処理を強制実行します")
            
            result = await self.batch_scheduler.cleanup_old_data(self.cleanup_retention_days)
            
            # 実行日を記録
            current_date = datetime.now().strftime("%Y-%m-%d")
            self.last_cleanup_date = current_date
            
            return result
            
        except Exception as e:
            error_msg = f"クリーンアップ処理の強制実行中にエラーが発生: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {
                "success": False,
                "error": error_msg
            }
    
    def set_callbacks(self, 
                     on_batch_complete: Optional[Callable] = None,
                     on_cleanup_complete: Optional[Callable] = None,
                     on_error: Optional[Callable] = None) -> None:
        """
        コールバック関数を設定
        
        Args:
            on_batch_complete: バッチ処理完了時のコールバック
            on_cleanup_complete: クリーンアップ完了時のコールバック
            on_error: エラー発生時のコールバック
        """
        self.on_batch_complete = on_batch_complete
        self.on_cleanup_complete = on_cleanup_complete
        self.on_error = on_error
        
        logger.info("コールバック関数を設定しました")
    
    async def get_processing_statistics(self, days: int = 7) -> Dict[str, Any]:
        """
        処理統計情報を取得
        
        Args:
            days: 統計対象日数
            
        Returns:
            Dict: 統計情報
        """
        try:
            # バッチスケジューラーから統計を取得
            batch_stats = await self.batch_scheduler.get_batch_statistics(days)
            
            # ストレージ統計を取得
            storage_stats = await self.storage_manager.get_storage_stats()
            
            # バックグラウンド処理の状態を追加
            status = self.get_status()
            
            return {
                "background_processor": status,
                "batch_statistics": batch_stats,
                "storage_statistics": storage_stats,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"統計情報取得エラー: {str(e)}")
            return {"error": str(e)}
    
    def __enter__(self):
        """コンテキストマネージャーのエントリー"""
        self.start_background_processing()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了"""
        self.stop_background_processing()


class StreamlitBackgroundIntegration:
    """Streamlitアプリとバックグラウンド処理の統合クラス"""
    
    def __init__(self):
        self.background_processor = None
        self.is_initialized = False
    
    def initialize(self, storage_manager: Optional[AsyncStorageManager] = None) -> bool:
        """
        バックグラウンド処理を初期化
        
        Args:
            storage_manager: ストレージマネージャー
            
        Returns:
            bool: 初期化成功フラグ
        """
        try:
            if self.is_initialized:
                return True
            
            self.background_processor = BackgroundProcessor(storage_manager)
            
            # コールバック関数を設定
            self.background_processor.set_callbacks(
                on_batch_complete=self._on_batch_complete,
                on_cleanup_complete=self._on_cleanup_complete,
                on_error=self._on_error
            )
            
            # バックグラウンド処理を開始
            success = self.background_processor.start_background_processing()
            
            if success:
                self.is_initialized = True
                logger.info("Streamlitバックグラウンド統合を初期化しました")
            
            return success
            
        except Exception as e:
            logger.error(f"バックグラウンド統合の初期化に失敗: {str(e)}")
            return False
    
    def shutdown(self) -> bool:
        """
        バックグラウンド処理を終了
        
        Returns:
            bool: 終了成功フラグ
        """
        try:
            if not self.is_initialized or not self.background_processor:
                return True
            
            success = self.background_processor.stop_background_processing()
            
            if success:
                self.is_initialized = False
                logger.info("Streamlitバックグラウンド統合を終了しました")
            
            return success
            
        except Exception as e:
            logger.error(f"バックグラウンド統合の終了に失敗: {str(e)}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """統合状態を取得"""
        if not self.is_initialized or not self.background_processor:
            return {"initialized": False, "running": False}
        
        status = self.background_processor.get_status()
        status["initialized"] = self.is_initialized
        
        return status
    
    async def get_statistics(self, days: int = 7) -> Dict[str, Any]:
        """統計情報を取得"""
        if not self.is_initialized or not self.background_processor:
            return {"error": "バックグラウンド処理が初期化されていません"}
        
        return await self.background_processor.get_processing_statistics(days)
    
    def _on_batch_complete(self, hour: int, result: Dict[str, Any]) -> None:
        """バッチ処理完了時のコールバック"""
        logger.info(f"バッチ処理完了通知 - {hour}時: {result.get('success', False)}")
        # Streamlitの状態更新やキャッシュクリアなどを実装可能
    
    def _on_cleanup_complete(self, result: Dict[str, Any]) -> None:
        """クリーンアップ完了時のコールバック"""
        logger.info(f"クリーンアップ完了通知: {result.get('success', False)}")
        # Streamlitの状態更新やキャッシュクリアなどを実装可能
    
    def _on_error(self, error_message: str) -> None:
        """エラー発生時のコールバック"""
        logger.error(f"バックグラウンド処理エラー通知: {error_message}")
        # Streamlitのエラー表示やアラート機能を実装可能


# グローバルインスタンス（Streamlitアプリで使用）
streamlit_background = StreamlitBackgroundIntegration()


# テスト用の関数
async def test_background_processor():
    """BackgroundProcessorのテスト"""
    import tempfile
    
    # 一時ディレクトリでテスト
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test_letters.json")
        storage = AsyncStorageManager(test_file)
        
        print("=== BackgroundProcessorテスト開始 ===")
        
        # バックグラウンドプロセッサーのテスト
        processor = BackgroundProcessor(storage)
        
        # 状態確認テスト
        status = processor.get_status()
        print(f"✓ 状態確認テスト: {status['is_running']}")
        
        # 強制バッチ実行テスト
        batch_result = await processor.force_run_batch(2)
        print(f"✓ 強制バッチ実行テスト: {batch_result['success']}")
        
        # 強制クリーンアップテスト
        cleanup_result = await processor.force_run_cleanup()
        print(f"✓ 強制クリーンアップテスト: {cleanup_result['success']}")
        
        # 統計情報テスト
        stats = await processor.get_processing_statistics()
        print(f"✓ 統計情報取得テスト: {'error' not in stats}")
        
        print("=== 全てのテストが完了しました！ ===")


if __name__ == "__main__":
    asyncio.run(test_background_processor())