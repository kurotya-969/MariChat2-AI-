"""
バッチスケジューラークラス
2時、3時、4時の時刻別バッチ処理機能を実装し、
指定時刻の未処理リクエストを処理する機能を提供します。
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
import traceback

from letter_request_manager import RequestManager
from letter_generator import LetterGenerator
from async_storage_manager import AsyncStorageManager
from async_rate_limiter import AsyncRateLimitManager
from letter_user_manager import UserManager

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchSchedulerError(Exception):
    """バッチスケジューラー関連のエラー"""
    pass


class BatchScheduler:
    """非同期バッチ処理スケジューラー"""
    
    def __init__(self, storage_manager: Optional[AsyncStorageManager] = None):
        """
        バッチスケジューラーを初期化
        
        Args:
            storage_manager: ストレージマネージャー（指定しない場合は新規作成）
        """
        # ストレージマネージャーの初期化
        if storage_manager is None:
            storage_path = os.getenv("STORAGE_PATH", "/mnt/data/letters.json")
            self.storage_manager = AsyncStorageManager(storage_path)
        else:
            self.storage_manager = storage_manager
        
        # 依存コンポーネントの初期化
        self.rate_limiter = AsyncRateLimitManager(self.storage_manager)
        self.request_manager = RequestManager(self.storage_manager, self.rate_limiter)
        self.letter_generator = LetterGenerator()
        self.user_manager = UserManager(self.storage_manager)
        
        # 設定値
        self.available_hours = [2, 3, 4]  # 2時、3時、4時
        self.max_concurrent_generations = int(os.getenv("MAX_CONCURRENT_GENERATIONS", "3"))
        self.generation_timeout = int(os.getenv("GENERATION_TIMEOUT", "300"))  # 5分
        self.retry_failed_requests = os.getenv("RETRY_FAILED_REQUESTS", "true").lower() == "true"
        
        logger.info(f"BatchScheduler初期化完了 - 対象時刻: {self.available_hours}")
    
    async def run_hourly_batch(self, hour: int) -> Dict[str, Any]:
        """
        指定時刻のバッチ処理を実行
        
        Args:
            hour: 実行時刻（2, 3, 4のいずれか）
            
        Returns:
            Dict: バッチ処理の結果
        """
        start_time = datetime.now()
        batch_id = f"{start_time.strftime('%Y%m%d_%H%M%S')}_{hour}"
        
        logger.info(f"=== {hour}時のバッチ処理開始 (ID: {batch_id}) ===")
        
        # 時刻の検証
        if hour not in self.available_hours:
            error_msg = f"無効な時刻が指定されました: {hour}時 (有効: {self.available_hours})"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "batch_id": batch_id,
                "hour": hour,
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        try:
            # バッチ実行の記録開始
            await self._record_batch_start(batch_id, hour, start_time)
            
            # 指定時刻の未処理リクエストを処理
            result = await self.process_pending_requests_for_hour(hour)
            
            # バッチ実行の記録完了
            end_time = datetime.now()
            await self._record_batch_completion(batch_id, hour, start_time, end_time, result)
            
            execution_time = (end_time - start_time).total_seconds()
            
            logger.info(f"=== {hour}時のバッチ処理完了 (所要時間: {execution_time:.2f}秒) ===")
            
            return {
                "success": True,
                "batch_id": batch_id,
                "hour": hour,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "execution_time": execution_time,
                "processed_count": result.get("processed_count", 0),
                "success_count": result.get("success_count", 0),
                "failed_count": result.get("failed_count", 0),
                "errors": result.get("errors", [])
            }
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            error_msg = f"バッチ処理中にエラーが発生しました: {str(e)}"
            
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            # エラーの記録
            await self._record_batch_error(batch_id, hour, start_time, end_time, error_msg)
            
            return {
                "success": False,
                "error": error_msg,
                "batch_id": batch_id,
                "hour": hour,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "execution_time": execution_time
            }
    
    async def process_pending_requests_for_hour(self, hour: int) -> Dict[str, Any]:
        """
        指定時刻の未処理リクエストを処理
        
        Args:
            hour: 処理対象の時刻
            
        Returns:
            Dict: 処理結果の詳細
        """
        try:
            # 未処理リクエストを取得
            pending_requests = await self.request_manager.get_pending_requests_by_hour(hour)
            
            if not pending_requests:
                logger.info(f"{hour}時の未処理リクエストはありません")
                return {
                    "processed_count": 0,
                    "success_count": 0,
                    "failed_count": 0,
                    "errors": []
                }
            
            logger.info(f"{hour}時の未処理リクエスト数: {len(pending_requests)}")
            
            # 並行処理用のセマフォ
            semaphore = asyncio.Semaphore(self.max_concurrent_generations)
            
            # 各リクエストを並行処理
            tasks = []
            for request in pending_requests:
                task = self._process_single_request(semaphore, request)
                tasks.append(task)
            
            # 全てのタスクを実行
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 結果を集計
            success_count = 0
            failed_count = 0
            errors = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_count += 1
                    error_msg = f"リクエスト処理例外: {str(result)}"
                    errors.append({
                        "request_index": i,
                        "user_id": pending_requests[i].get("user_id", "unknown"),
                        "error": error_msg
                    })
                    logger.error(error_msg)
                elif result.get("success", False):
                    success_count += 1
                else:
                    failed_count += 1
                    errors.append({
                        "request_index": i,
                        "user_id": pending_requests[i].get("user_id", "unknown"),
                        "error": result.get("error", "不明なエラー")
                    })
            
            logger.info(f"処理完了 - 成功: {success_count}, 失敗: {failed_count}")
            
            return {
                "processed_count": len(pending_requests),
                "success_count": success_count,
                "failed_count": failed_count,
                "errors": errors
            }
            
        except Exception as e:
            error_msg = f"未処理リクエスト処理中にエラーが発生: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {
                "processed_count": 0,
                "success_count": 0,
                "failed_count": 0,
                "errors": [{"error": error_msg}]
            }
    
    async def _process_single_request(self, semaphore: asyncio.Semaphore, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        単一のリクエストを処理
        
        Args:
            semaphore: 並行処理制御用のセマフォ
            request: 処理対象のリクエスト
            
        Returns:
            Dict: 処理結果
        """
        async with semaphore:
            user_id = request["user_id"]
            theme = request["theme"]
            date = request["date"]
            
            try:
                logger.info(f"手紙生成開始 - ユーザー: {user_id}, テーマ: {theme[:50]}...")
                
                # タイムアウト付きで手紙生成を実行
                user_history = await self.user_manager.get_user_profile(user_id)
                
                generation_task = self.letter_generator.generate_letter(user_id, theme, user_history)
                letter_result = await asyncio.wait_for(generation_task, timeout=self.generation_timeout)
                
                # 生成された手紙をストレージに保存
                await self._save_generated_letter(user_id, date, theme, letter_result)
                
                # リクエストを完了としてマーク
                await self.request_manager.mark_request_processed(user_id, date, "completed")
                
                # ユーザー履歴を更新
                await self.user_manager.update_user_history(user_id, {
                    "date": date,
                    "theme": theme,
                    "status": "completed",
                    "generated_at": datetime.now().isoformat()
                })
                
                logger.info(f"手紙生成完了 - ユーザー: {user_id}")
                
                return {
                    "success": True,
                    "user_id": user_id,
                    "theme": theme,
                    "generation_time": letter_result["metadata"].get("generation_time", 0)
                }
                
            except asyncio.TimeoutError:
                error_msg = f"手紙生成がタイムアウトしました（{self.generation_timeout}秒）"
                logger.error(f"{error_msg} - ユーザー: {user_id}")
                
                await self.request_manager.mark_request_failed(user_id, date, error_msg)
                
                return {
                    "success": False,
                    "user_id": user_id,
                    "error": error_msg
                }
                
            except Exception as e:
                error_msg = f"手紙生成中にエラーが発生: {str(e)}"
                logger.error(f"{error_msg} - ユーザー: {user_id}\n{traceback.format_exc()}")
                
                await self.request_manager.mark_request_failed(user_id, date, error_msg)
                
                return {
                    "success": False,
                    "user_id": user_id,
                    "error": error_msg
                }
    
    async def _save_generated_letter(self, user_id: str, date: str, theme: str, letter_result: Dict[str, Any]) -> None:
        """
        生成された手紙をストレージに保存
        
        Args:
            user_id: ユーザーID
            date: 日付
            theme: テーマ
            letter_result: 生成結果
        """
        try:
            user_data = await self.storage_manager.get_user_data(user_id)
            
            # 手紙データを作成
            letter_data = {
                "theme": theme,
                "content": letter_result["content"],
                "status": "completed",
                "generated_at": datetime.now().isoformat(),
                "metadata": letter_result["metadata"]
            }
            
            # ユーザーデータに追加
            user_data["letters"][date] = letter_data
            user_data["profile"]["total_letters"] = user_data["profile"].get("total_letters", 0) + 1
            user_data["profile"]["last_request"] = date
            
            # ストレージに保存
            await self.storage_manager.update_user_data(user_id, user_data)
            
            logger.info(f"手紙をストレージに保存しました - ユーザー: {user_id}, 日付: {date}")
            
        except Exception as e:
            logger.error(f"手紙保存エラー - ユーザー: {user_id}, 日付: {date}: {str(e)}")
            raise
    
    async def _record_batch_start(self, batch_id: str, hour: int, start_time: datetime) -> None:
        """バッチ実行開始を記録"""
        try:
            system_info = await self.storage_manager.get_system_info()
            
            if "batch_runs" not in system_info:
                system_info["batch_runs"] = {}
            
            system_info["batch_runs"][batch_id] = {
                "hour": hour,
                "start_time": start_time.isoformat(),
                "status": "running"
            }
            
            await self.storage_manager.update_system_info(system_info)
            
        except Exception as e:
            logger.error(f"バッチ開始記録エラー: {str(e)}")
    
    async def _record_batch_completion(self, batch_id: str, hour: int, start_time: datetime, 
                                     end_time: datetime, result: Dict[str, Any]) -> None:
        """バッチ実行完了を記録"""
        try:
            system_info = await self.storage_manager.get_system_info()
            
            if batch_id in system_info.get("batch_runs", {}):
                system_info["batch_runs"][batch_id].update({
                    "end_time": end_time.isoformat(),
                    "status": "completed",
                    "execution_time": (end_time - start_time).total_seconds(),
                    "processed_count": result.get("processed_count", 0),
                    "success_count": result.get("success_count", 0),
                    "failed_count": result.get("failed_count", 0),
                    "error_count": len(result.get("errors", []))
                })
                
                await self.storage_manager.update_system_info(system_info)
            
        except Exception as e:
            logger.error(f"バッチ完了記録エラー: {str(e)}")
    
    async def _record_batch_error(self, batch_id: str, hour: int, start_time: datetime, 
                                end_time: datetime, error_msg: str) -> None:
        """バッチ実行エラーを記録"""
        try:
            system_info = await self.storage_manager.get_system_info()
            
            if batch_id in system_info.get("batch_runs", {}):
                system_info["batch_runs"][batch_id].update({
                    "end_time": end_time.isoformat(),
                    "status": "failed",
                    "execution_time": (end_time - start_time).total_seconds(),
                    "error": error_msg
                })
                
                await self.storage_manager.update_system_info(system_info)
            
        except Exception as e:
            logger.error(f"バッチエラー記録エラー: {str(e)}")
    
    def schedule_all_hours(self) -> None:
        """
        全ての対象時刻でのスケジュール設定
        注意: この関数は実際のスケジューリングライブラリと組み合わせて使用する
        """
        logger.info("バッチスケジュールを設定します")
        
        for hour in self.available_hours:
            logger.info(f"  - {hour}:00 に手紙生成バッチを設定")
        
        # 実際のスケジューリングは外部ライブラリ（APScheduler等）で実装
        # ここでは設定情報のログ出力のみ
        logger.info("スケジュール設定完了（実際の実行は外部スケジューラーが必要）")
    
    async def cleanup_old_data(self, days: int = 90) -> Dict[str, Any]:
        """
        古いデータの自動削除
        
        Args:
            days: 保持日数
            
        Returns:
            Dict: 削除結果
        """
        try:
            logger.info(f"{days}日以前の古いデータを削除します")
            
            # ストレージマネージャーの削除機能を使用
            deleted_count = await self.storage_manager.cleanup_old_data(days)
            
            # リクエストマネージャーの削除機能も使用
            deleted_requests = await self.request_manager.cleanup_old_requests(days)
            
            # バックアップの作成
            backup_path = await self.storage_manager.backup_data()
            
            result = {
                "success": True,
                "deleted_letters": deleted_count,
                "deleted_requests": deleted_requests,
                "backup_created": backup_path,
                "cleanup_date": datetime.now().isoformat()
            }
            
            logger.info(f"古いデータ削除完了: {result}")
            return result
            
        except Exception as e:
            error_msg = f"古いデータ削除中にエラーが発生: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            return {
                "success": False,
                "error": error_msg,
                "cleanup_date": datetime.now().isoformat()
            }
    
    async def get_batch_statistics(self, days: int = 7) -> Dict[str, Any]:
        """
        バッチ処理の統計情報を取得
        
        Args:
            days: 統計対象日数
            
        Returns:
            Dict: 統計情報
        """
        try:
            system_info = await self.storage_manager.get_system_info()
            batch_runs = system_info.get("batch_runs", {})
            
            # 指定日数以内のバッチ実行を抽出
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_batches = []
            
            for batch_id, batch_info in batch_runs.items():
                try:
                    start_time = datetime.fromisoformat(batch_info["start_time"])
                    if start_time >= cutoff_date:
                        recent_batches.append(batch_info)
                except (ValueError, KeyError):
                    continue
            
            # 統計を計算
            total_runs = len(recent_batches)
            successful_runs = len([b for b in recent_batches if b.get("status") == "completed"])
            failed_runs = len([b for b in recent_batches if b.get("status") == "failed"])
            
            total_processed = sum(b.get("processed_count", 0) for b in recent_batches)
            total_success = sum(b.get("success_count", 0) for b in recent_batches)
            total_failed = sum(b.get("failed_count", 0) for b in recent_batches)
            
            avg_execution_time = 0
            if recent_batches:
                execution_times = [b.get("execution_time", 0) for b in recent_batches if b.get("execution_time")]
                if execution_times:
                    avg_execution_time = sum(execution_times) / len(execution_times)
            
            # 時刻別統計
            hourly_stats = {hour: {"runs": 0, "processed": 0, "success": 0} for hour in self.available_hours}
            
            for batch in recent_batches:
                hour = batch.get("hour")
                if hour in hourly_stats:
                    hourly_stats[hour]["runs"] += 1
                    hourly_stats[hour]["processed"] += batch.get("processed_count", 0)
                    hourly_stats[hour]["success"] += batch.get("success_count", 0)
            
            return {
                "period_days": days,
                "total_runs": total_runs,
                "successful_runs": successful_runs,
                "failed_runs": failed_runs,
                "success_rate": (successful_runs / total_runs * 100) if total_runs > 0 else 0,
                "total_processed": total_processed,
                "total_success": total_success,
                "total_failed": total_failed,
                "processing_success_rate": (total_success / total_processed * 100) if total_processed > 0 else 0,
                "avg_execution_time": avg_execution_time,
                "hourly_stats": hourly_stats,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"統計情報取得エラー: {str(e)}")
            return {"error": str(e)}
    
    async def test_batch_processing(self, test_hour: int = 2) -> Dict[str, Any]:
        """
        バッチ処理のテスト実行
        
        Args:
            test_hour: テスト対象の時刻
            
        Returns:
            Dict: テスト結果
        """
        try:
            logger.info(f"バッチ処理テストを開始します（{test_hour}時）")
            
            # API接続テスト
            api_status = await self.letter_generator.check_api_connections()
            if not api_status.get("overall", False):
                return {
                    "success": False,
                    "error": "API接続テストに失敗しました",
                    "api_status": api_status
                }
            
            # テスト用のバッチ処理を実行
            result = await self.run_hourly_batch(test_hour)
            
            return {
                "success": True,
                "message": "バッチ処理テスト完了",
                "api_status": api_status,
                "batch_result": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"バッチ処理テスト中にエラーが発生: {str(e)}"
            }


# テスト用の関数
async def test_batch_scheduler():
    """BatchSchedulerのテスト"""
    import tempfile
    import uuid
    
    # 一時ディレクトリでテスト
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test_letters.json")
        storage = AsyncStorageManager(test_file)
        scheduler = BatchScheduler(storage)
        
        print("=== BatchSchedulerテスト開始 ===")
        
        # テスト用のリクエストを作成
        user_id = str(uuid.uuid4())
        success, message = await scheduler.request_manager.submit_request(user_id, "テストテーマ", 2)
        print(f"✓ テストリクエスト作成: {success} - {message}")
        
        # バッチ処理テスト
        result = await scheduler.run_hourly_batch(2)
        print(f"✓ バッチ処理テスト: {result['success']}")
        
        # 統計情報テスト
        stats = await scheduler.get_batch_statistics()
        print(f"✓ 統計情報取得テスト: {stats}")
        
        # 古いデータ削除テスト
        cleanup_result = await scheduler.cleanup_old_data(0)  # 全て削除
        print(f"✓ データ削除テスト: {cleanup_result['success']}")
        
        print("=== 全てのテストが完了しました！ ===")


if __name__ == "__main__":
    asyncio.run(test_batch_scheduler())