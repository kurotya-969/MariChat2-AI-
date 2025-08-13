"""
リクエスト管理クラス
テーマと生成時刻を含むリクエスト送信機能と
時刻別の未処理リクエスト取得機能を提供します。
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
import uuid

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RequestError(Exception):
    """リクエスト関連のエラー"""
    pass


class RequestManager:
    """リクエスト管理クラス"""
    
    def __init__(self, storage_manager, rate_limiter):
        self.storage = storage_manager
        self.rate_limiter = rate_limiter
        
        # 設定値
        self.valid_generation_hours = [2, 3, 4]  # 2時、3時、4時
        self.max_theme_length = int(os.getenv("MAX_THEME_LENGTH", "200"))
        self.min_theme_length = int(os.getenv("MIN_THEME_LENGTH", "1"))
        
        logger.info(f"RequestManager初期化完了 - 有効な生成時刻: {self.valid_generation_hours}")
    
    async def submit_request(self, user_id: str, theme: str, generation_hour: int, affection: int = None) -> Tuple[bool, str]:
        """
        リクエストを送信する
        
        Args:
            user_id: ユーザーID
            theme: 手紙のテーマ
            generation_hour: 生成時刻（2, 3, 4のいずれか）
            affection: 現在の好感度（オプション）
            
        Returns:
            Tuple[bool, str]: (成功フラグ, メッセージ)
        """
        try:
            # 入力バリデーション
            if not self.validate_theme(theme):
                return False, f"テーマは{self.min_theme_length}文字以上{self.max_theme_length}文字以下で入力してください"
            
            if not self.validate_generation_hour(generation_hour):
                return False, f"生成時刻は{self.valid_generation_hours}のいずれかを選択してください"
            
            # レート制限チェック
            allowed, limit_message = await self.rate_limiter.is_request_allowed(user_id)
            if not allowed:
                return False, limit_message
            
            # 既存のリクエストチェック（同日の重複防止）
            today = datetime.now().strftime("%Y-%m-%d")
            existing_request = await self._get_user_request_for_date(user_id, today)
            
            if existing_request:
                return False, "本日は既にリクエストを送信済みです。1日1回までリクエスト可能です。"
            
            # リクエストデータの作成
            request_data = {
                "theme": theme.strip(),
                "status": "pending",
                "requested_at": datetime.now().isoformat(),
                "generation_hour": generation_hour,
                "request_id": str(uuid.uuid4()),
                "affection": affection  # 好感度情報を追加
            }
            
            # ユーザーデータの取得と更新
            user_data = await self.storage.get_user_data(user_id)
            user_data["requests"][today] = request_data
            
            # ストレージに保存
            await self.storage.update_user_data(user_id, user_data)
            
            # レート制限の記録
            await self.rate_limiter.record_request(user_id)
            
            logger.info(f"リクエスト送信成功 - ユーザー: {user_id}, テーマ: {theme[:50]}..., 生成時刻: {generation_hour}時")
            
            return True, f"リクエストを受け付けました。{generation_hour}時頃に手紙を生成します。"
            
        except Exception as e:
            logger.error(f"リクエスト送信エラー: {e}")
            return False, f"リクエストの送信中にエラーが発生しました: {str(e)}"
    
    async def get_pending_requests_by_hour(self, hour: int) -> List[Dict[str, Any]]:
        """
        指定時刻の未処理リクエストを取得する
        
        Args:
            hour: 生成時刻（2, 3, 4のいずれか）
            
        Returns:
            List[Dict]: 未処理リクエストのリスト
        """
        try:
            if hour not in self.valid_generation_hours:
                logger.warning(f"無効な生成時刻が指定されました: {hour}")
                return []
            
            all_users = await self.storage.get_all_users()
            pending_requests = []
            today = datetime.now().strftime("%Y-%m-%d")
            
            for user_id in all_users:
                user_data = await self.storage.get_user_data(user_id)
                
                # 今日のリクエストをチェック
                if today in user_data["requests"]:
                    request = user_data["requests"][today]
                    
                    # 指定時刻かつ未処理のリクエストを抽出
                    if (request.get("generation_hour") == hour and 
                        request.get("status") == "pending"):
                        
                        pending_requests.append({
                            "user_id": user_id,
                            "theme": request["theme"],
                            "requested_at": request["requested_at"],
                            "generation_hour": request["generation_hour"],
                            "request_id": request.get("request_id", ""),
                            "date": today
                        })
            
            logger.info(f"{hour}時の未処理リクエスト数: {len(pending_requests)}")
            return pending_requests
            
        except Exception as e:
            logger.error(f"未処理リクエスト取得エラー: {e}")
            return []
    
    async def mark_request_processed(self, user_id: str, date: str, status: str = "completed") -> bool:
        """
        リクエストを処理済みにマークする
        
        Args:
            user_id: ユーザーID
            date: 日付（YYYY-MM-DD形式）
            status: 新しいステータス（completed, failed等）
            
        Returns:
            bool: 成功フラグ
        """
        try:
            user_data = await self.storage.get_user_data(user_id)
            
            if date not in user_data["requests"]:
                logger.warning(f"指定された日付のリクエストが見つかりません - ユーザー: {user_id}, 日付: {date}")
                return False
            
            # ステータスを更新
            user_data["requests"][date]["status"] = status
            user_data["requests"][date]["processed_at"] = datetime.now().isoformat()
            
            await self.storage.update_user_data(user_id, user_data)
            
            logger.info(f"リクエストを{status}にマークしました - ユーザー: {user_id}, 日付: {date}")
            return True
            
        except Exception as e:
            logger.error(f"リクエスト処理マークエラー: {e}")
            return False
    
    async def mark_request_failed(self, user_id: str, date: str, error_message: str) -> bool:
        """
        リクエストを失敗にマークする
        
        Args:
            user_id: ユーザーID
            date: 日付（YYYY-MM-DD形式）
            error_message: エラーメッセージ
            
        Returns:
            bool: 成功フラグ
        """
        try:
            user_data = await self.storage.get_user_data(user_id)
            
            if date not in user_data["requests"]:
                logger.warning(f"指定された日付のリクエストが見つかりません - ユーザー: {user_id}, 日付: {date}")
                return False
            
            # ステータスとエラー情報を更新
            user_data["requests"][date]["status"] = "failed"
            user_data["requests"][date]["processed_at"] = datetime.now().isoformat()
            user_data["requests"][date]["error_message"] = error_message
            
            await self.storage.update_user_data(user_id, user_data)
            
            logger.error(f"リクエストを失敗にマークしました - ユーザー: {user_id}, 日付: {date}, エラー: {error_message}")
            return True
            
        except Exception as e:
            logger.error(f"リクエスト失敗マークエラー: {e}")
            return False
    
    def validate_theme(self, theme: str) -> bool:
        """
        テーマのバリデーション
        
        Args:
            theme: テーマ文字列
            
        Returns:
            bool: バリデーション結果
        """
        if not theme or not isinstance(theme, str):
            return False
        
        theme = theme.strip()
        
        # 長さチェック
        if len(theme) < self.min_theme_length or len(theme) > self.max_theme_length:
            return False
        
        # 不正な文字のチェック（基本的な制御文字のみ）
        if any(ord(char) < 32 and char not in ['\n', '\r', '\t'] for char in theme):
            return False
        
        return True
    
    def validate_generation_hour(self, hour: int) -> bool:
        """
        生成時刻のバリデーション
        
        Args:
            hour: 生成時刻
            
        Returns:
            bool: バリデーション結果
        """
        return isinstance(hour, int) and hour in self.valid_generation_hours
    
    async def get_user_request_status(self, user_id: str, date: Optional[str] = None) -> Dict[str, Any]:
        """
        ユーザーのリクエスト状況を取得する
        
        Args:
            user_id: ユーザーID
            date: 日付（指定しない場合は今日）
            
        Returns:
            Dict: リクエスト状況
        """
        try:
            if date is None:
                date = datetime.now().strftime("%Y-%m-%d")
            
            user_data = await self.storage.get_user_data(user_id)
            
            if date not in user_data["requests"]:
                return {
                    "has_request": False,
                    "date": date,
                    "can_request": True
                }
            
            request = user_data["requests"][date]
            
            return {
                "has_request": True,
                "date": date,
                "theme": request["theme"],
                "status": request["status"],
                "generation_hour": request["generation_hour"],
                "requested_at": request["requested_at"],
                "processed_at": request.get("processed_at"),
                "error_message": request.get("error_message"),
                "can_request": False  # 既にリクエスト済み
            }
            
        except Exception as e:
            logger.error(f"リクエスト状況取得エラー: {e}")
            return {"error": str(e)}
    
    async def _get_user_request_for_date(self, user_id: str, date: str) -> Optional[Dict[str, Any]]:
        """
        指定日のユーザーリクエストを取得する（内部使用）
        
        Args:
            user_id: ユーザーID
            date: 日付（YYYY-MM-DD形式）
            
        Returns:
            Optional[Dict]: リクエストデータ（存在しない場合はNone）
        """
        try:
            user_data = await self.storage.get_user_data(user_id)
            return user_data["requests"].get(date)
        except Exception as e:
            logger.error(f"ユーザーリクエスト取得エラー: {e}")
            return None
    
    async def get_all_pending_requests(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        全ての未処理リクエストを時刻別に取得する
        
        Returns:
            Dict[int, List]: 時刻別の未処理リクエスト
        """
        try:
            all_pending = {}
            
            for hour in self.valid_generation_hours:
                pending_requests = await self.get_pending_requests_by_hour(hour)
                all_pending[hour] = pending_requests
            
            return all_pending
            
        except Exception as e:
            logger.error(f"全未処理リクエスト取得エラー: {e}")
            return {}
    
    async def cleanup_old_requests(self, days: int = 30) -> int:
        """
        古いリクエストデータを削除する
        
        Args:
            days: 保持日数
            
        Returns:
            int: 削除されたリクエスト数
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d")
            
            all_users = await self.storage.get_all_users()
            deleted_count = 0
            
            for user_id in all_users:
                user_data = await self.storage.get_user_data(user_id)
                
                # 古いリクエストを削除
                requests_to_delete = []
                for date_str in user_data["requests"]:
                    if date_str < cutoff_str:
                        requests_to_delete.append(date_str)
                
                for date_str in requests_to_delete:
                    del user_data["requests"][date_str]
                    deleted_count += 1
                
                if requests_to_delete:
                    await self.storage.update_user_data(user_id, user_data)
            
            if deleted_count > 0:
                logger.info(f"{deleted_count}件の古いリクエストを削除しました")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"古いリクエスト削除エラー: {e}")
            return 0
    
    async def get_request_statistics(self) -> Dict[str, Any]:
        """
        リクエストの統計情報を取得する
        
        Returns:
            Dict: 統計情報
        """
        try:
            all_users = await self.storage.get_all_users()
            today = datetime.now().strftime("%Y-%m-%d")
            
            stats = {
                "total_users": len(all_users),
                "today_requests": 0,
                "pending_requests": 0,
                "completed_requests": 0,
                "failed_requests": 0,
                "requests_by_hour": {hour: 0 for hour in self.valid_generation_hours},
                "date": today
            }
            
            for user_id in all_users:
                user_data = await self.storage.get_user_data(user_id)
                
                # 今日のリクエストをカウント
                if today in user_data["requests"]:
                    request = user_data["requests"][today]
                    stats["today_requests"] += 1
                    
                    # ステータス別カウント
                    status = request.get("status", "unknown")
                    if status == "pending":
                        stats["pending_requests"] += 1
                    elif status == "completed":
                        stats["completed_requests"] += 1
                    elif status == "failed":
                        stats["failed_requests"] += 1
                    
                    # 時刻別カウント
                    hour = request.get("generation_hour")
                    if hour in stats["requests_by_hour"]:
                        stats["requests_by_hour"][hour] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"統計情報取得エラー: {e}")
            return {"error": str(e)}


# テスト用の関数
async def test_request_manager():
    """RequestManagerのテスト"""
    import tempfile
    from async_storage_manager import AsyncStorageManager
    from async_rate_limiter import AsyncRateLimitManager
    
    # 一時ディレクトリでテスト
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test_letters.json")
        storage = AsyncStorageManager(test_file)
        rate_limiter = AsyncRateLimitManager(storage)
        request_manager = RequestManager(storage, rate_limiter)
        
        print("=== RequestManagerテスト開始 ===")
        
        user_id = str(uuid.uuid4())
        
        # リクエスト送信テスト
        success, message = await request_manager.submit_request(user_id, "春の思い出", 2)
        print(f"✓ リクエスト送信テスト: {success} - {message}")
        
        # 未処理リクエスト取得テスト
        pending = await request_manager.get_pending_requests_by_hour(2)
        print(f"✓ 未処理リクエスト取得テスト: {len(pending)}件")
        
        # リクエスト状況確認テスト
        status = await request_manager.get_user_request_status(user_id)
        print(f"✓ リクエスト状況確認テスト: {status}")
        
        # リクエスト処理マークテスト
        today = datetime.now().strftime("%Y-%m-%d")
        marked = await request_manager.mark_request_processed(user_id, today)
        print(f"✓ リクエスト処理マークテスト: {marked}")
        
        # 統計情報取得テスト
        stats = await request_manager.get_request_statistics()
        print(f"✓ 統計情報取得テスト: {stats}")
        
        print("=== 全てのテストが完了しました！ ===")


if __name__ == "__main__":
    asyncio.run(test_request_manager())