"""
非同期レート制限管理クラス
1日1回制限とAPI呼び出し制限を管理し、
デバッグモード時の制限緩和機能を提供します。
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """レート制限エラー"""
    pass


class AsyncRateLimitManager:
    """非同期レート制限管理クラス"""
    
    def __init__(self, storage_manager, max_requests: int = 1):
        self.storage = storage_manager
        
        # 設定値（環境変数から取得、デフォルト値あり）
        self.max_daily_requests = int(os.getenv("MAX_DAILY_REQUESTS", "1"))
        self.max_api_calls_per_day = int(os.getenv("MAX_API_CALLS_PER_DAY", "10"))
        self.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
        
        # デバッグモード時の制限緩和
        if self.debug_mode:
            self.max_daily_requests = int(os.getenv("DEBUG_MAX_DAILY_REQUESTS", "10"))
            self.max_api_calls_per_day = int(os.getenv("DEBUG_MAX_API_CALLS", "100"))
            logger.info("デバッグモードが有効です。制限が緩和されています")
        
        logger.info(f"レート制限設定 - 1日のリクエスト上限: {self.max_daily_requests}, API呼び出し上限: {self.max_api_calls_per_day}")
    
    async def check_daily_request_limit(self, user_id: str) -> Tuple[bool, Dict[str, Any]]:
        """1日のリクエスト制限をチェック"""
        try:
            user_data = await self.storage.get_user_data(user_id)
            today = datetime.now().strftime("%Y-%m-%d")
            
            # 今日のリクエスト数を取得
            daily_requests = user_data["rate_limits"]["daily_requests"]
            today_requests = daily_requests.get(today, 0)
            
            # 制限チェック
            is_allowed = today_requests < self.max_daily_requests
            
            limit_info = {
                "today_requests": today_requests,
                "max_requests": self.max_daily_requests,
                "remaining": max(0, self.max_daily_requests - today_requests),
                "reset_time": self._get_next_reset_time(),
                "debug_mode": self.debug_mode
            }
            
            if not is_allowed:
                logger.warning(f"ユーザー {user_id} の1日のリクエスト制限に達しました ({today_requests}/{self.max_daily_requests})")
            
            return is_allowed, limit_info
            
        except Exception as e:
            logger.error(f"1日のリクエスト制限チェックエラー: {e}")
            # エラー時は制限を適用
            return False, {"error": str(e)}
    
    async def check_api_call_limit(self, user_id: str) -> Tuple[bool, Dict[str, Any]]:
        """API呼び出し制限をチェック"""
        try:
            user_data = await self.storage.get_user_data(user_id)
            today = datetime.now().strftime("%Y-%m-%d")
            
            # 今日のAPI呼び出し数を取得
            api_calls = user_data["rate_limits"]["api_calls"]
            today_calls = api_calls.get(today, 0)
            
            # 制限チェック
            is_allowed = today_calls < self.max_api_calls_per_day
            
            limit_info = {
                "today_calls": today_calls,
                "max_calls": self.max_api_calls_per_day,
                "remaining": max(0, self.max_api_calls_per_day - today_calls),
                "reset_time": self._get_next_reset_time(),
                "debug_mode": self.debug_mode
            }
            
            if not is_allowed:
                logger.warning(f"ユーザー {user_id} のAPI呼び出し制限に達しました ({today_calls}/{self.max_api_calls_per_day})")
            
            return is_allowed, limit_info
            
        except Exception as e:
            logger.error(f"API呼び出し制限チェックエラー: {e}")
            # エラー時は制限を適用
            return False, {"error": str(e)}
    
    async def record_request(self, user_id: str) -> None:
        """リクエストを記録"""
        try:
            user_data = await self.storage.get_user_data(user_id)
            today = datetime.now().strftime("%Y-%m-%d")
            
            # 今日のリクエスト数を増加
            if "daily_requests" not in user_data["rate_limits"]:
                user_data["rate_limits"]["daily_requests"] = {}
            
            user_data["rate_limits"]["daily_requests"][today] = \
                user_data["rate_limits"]["daily_requests"].get(today, 0) + 1
            
            # プロファイルの最終リクエスト日を更新
            user_data["profile"]["last_request"] = today
            
            await self.storage.update_user_data(user_id, user_data)
            
            logger.info(f"ユーザー {user_id} のリクエストを記録しました")
            
        except Exception as e:
            logger.error(f"リクエスト記録エラー: {e}")
            raise RateLimitError(f"リクエストの記録に失敗しました: {e}")
    
    async def record_api_call(self, user_id: str, api_type: str = "general") -> None:
        """API呼び出しを記録"""
        try:
            user_data = await self.storage.get_user_data(user_id)
            today = datetime.now().strftime("%Y-%m-%d")
            
            # 今日のAPI呼び出し数を増加
            if "api_calls" not in user_data["rate_limits"]:
                user_data["rate_limits"]["api_calls"] = {}
            
            user_data["rate_limits"]["api_calls"][today] = \
                user_data["rate_limits"]["api_calls"].get(today, 0) + 1
            
            await self.storage.update_user_data(user_id, user_data)
            
            logger.info(f"ユーザー {user_id} のAPI呼び出し ({api_type}) を記録しました")
            
        except Exception as e:
            logger.error(f"API呼び出し記録エラー: {e}")
            raise RateLimitError(f"API呼び出しの記録に失敗しました: {e}")
    
    async def get_user_limits_status(self, user_id: str) -> Dict[str, Any]:
        """ユーザーの制限状況を取得"""
        try:
            # リクエスト制限の確認
            request_allowed, request_info = await self.check_daily_request_limit(user_id)
            
            # API呼び出し制限の確認
            api_allowed, api_info = await self.check_api_call_limit(user_id)
            
            # 次回リクエスト可能時刻の計算
            next_request_time = None
            if not request_allowed:
                next_request_time = self._get_next_reset_time()
            
            return {
                "request_limit": {
                    "allowed": request_allowed,
                    "info": request_info
                },
                "api_limit": {
                    "allowed": api_allowed,
                    "info": api_info
                },
                "next_request_time": next_request_time,
                "debug_mode": self.debug_mode
            }
            
        except Exception as e:
            logger.error(f"制限状況取得エラー: {e}")
            return {"error": str(e)}
    
    async def reset_daily_counters(self) -> int:
        """1日のカウンターをリセット（古いデータを削除）"""
        try:
            # 7日以上前のデータを削除
            cutoff_date = datetime.now() - timedelta(days=7)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d")
            
            all_users = await self.storage.get_all_users()
            reset_count = 0
            
            for user_id in all_users:
                user_data = await self.storage.get_user_data(user_id)
                
                # 古い1日のリクエストデータを削除
                daily_requests = user_data["rate_limits"]["daily_requests"]
                dates_to_delete = [date for date in daily_requests.keys() if date < cutoff_str]
                
                for date in dates_to_delete:
                    del daily_requests[date]
                    reset_count += 1
                
                # 古いAPI呼び出しデータを削除
                api_calls = user_data["rate_limits"]["api_calls"]
                dates_to_delete = [date for date in api_calls.keys() if date < cutoff_str]
                
                for date in dates_to_delete:
                    del api_calls[date]
                    reset_count += 1
                
                if reset_count > 0:
                    await self.storage.update_user_data(user_id, user_data)
            
            if reset_count > 0:
                logger.info(f"{reset_count}件の古い制限データをリセットしました")
            
            return reset_count
            
        except Exception as e:
            logger.error(f"カウンターリセットエラー: {e}")
            return 0
    
    def _get_next_reset_time(self) -> str:
        """次のリセット時刻を取得（翌日の0時）"""
        tomorrow = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return tomorrow.isoformat()
    
    async def is_request_allowed(self, user_id: str) -> Tuple[bool, str]:
        """リクエストが許可されているかチェック（統合チェック）"""
        try:
            # 1日のリクエスト制限チェック
            request_allowed, request_info = await self.check_daily_request_limit(user_id)
            
            if not request_allowed:
                remaining_time = self._calculate_remaining_time()
                return False, f"1日のリクエスト制限に達しています。次回リクエスト可能時刻: {remaining_time}"
            
            # API呼び出し制限チェック
            api_allowed, api_info = await self.check_api_call_limit(user_id)
            
            if not api_allowed:
                remaining_time = self._calculate_remaining_time()
                return False, f"API呼び出し制限に達しています。次回リクエスト可能時刻: {remaining_time}"
            
            return True, "リクエスト可能です"
            
        except Exception as e:
            logger.error(f"リクエスト許可チェックエラー: {e}")
            return False, f"制限チェック中にエラーが発生しました: {e}"
    
    def _calculate_remaining_time(self) -> str:
        """次回リクエスト可能までの残り時間を計算"""
        now = datetime.now()
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        remaining = tomorrow - now
        
        hours = remaining.seconds // 3600
        minutes = (remaining.seconds % 3600) // 60
        
        return f"{hours}時間{minutes}分後"
    
    async def get_rate_limit_stats(self) -> Dict[str, Any]:
        """レート制限の統計情報を取得"""
        try:
            all_users = await self.storage.get_all_users()
            today = datetime.now().strftime("%Y-%m-%d")
            
            total_requests_today = 0
            total_api_calls_today = 0
            active_users_today = 0
            
            for user_id in all_users:
                user_data = await self.storage.get_user_data(user_id)
                
                # 今日のリクエスト数
                daily_requests = user_data["rate_limits"]["daily_requests"]
                user_requests_today = daily_requests.get(today, 0)
                total_requests_today += user_requests_today
                
                # 今日のAPI呼び出し数
                api_calls = user_data["rate_limits"]["api_calls"]
                user_api_calls_today = api_calls.get(today, 0)
                total_api_calls_today += user_api_calls_today
                
                # アクティブユーザー数
                if user_requests_today > 0 or user_api_calls_today > 0:
                    active_users_today += 1
            
            return {
                "total_users": len(all_users),
                "active_users_today": active_users_today,
                "total_requests_today": total_requests_today,
                "total_api_calls_today": total_api_calls_today,
                "max_daily_requests": self.max_daily_requests,
                "max_api_calls_per_day": self.max_api_calls_per_day,
                "debug_mode": self.debug_mode,
                "date": today
            }
            
        except Exception as e:
            logger.error(f"統計情報取得エラー: {e}")
            return {"error": str(e)}
    
    def is_debug_mode(self) -> bool:
        """デバッグモードかどうかを確認"""
        return self.debug_mode
    
    async def set_debug_mode(self, enabled: bool) -> None:
        """デバッグモードの設定（動的変更）"""
        self.debug_mode = enabled
        
        if enabled:
            self.max_daily_requests = int(os.getenv("DEBUG_MAX_DAILY_REQUESTS", "10"))
            self.max_api_calls_per_day = int(os.getenv("DEBUG_MAX_API_CALLS", "100"))
            logger.info("デバッグモードを有効にしました")
        else:
            self.max_daily_requests = int(os.getenv("MAX_DAILY_REQUESTS", "1"))
            self.max_api_calls_per_day = int(os.getenv("MAX_API_CALLS_PER_DAY", "10"))
            logger.info("デバッグモードを無効にしました")
    
    async def force_reset_user_limits(self, user_id: str) -> None:
        """特定ユーザーの制限を強制リセット（デバッグ用）"""
        if not self.debug_mode:
            raise RateLimitError("デバッグモードでのみ利用可能です")
        
        try:
            user_data = await self.storage.get_user_data(user_id)
            today = datetime.now().strftime("%Y-%m-%d")
            
            # 今日の制限をリセット
            user_data["rate_limits"]["daily_requests"][today] = 0
            user_data["rate_limits"]["api_calls"][today] = 0
            
            await self.storage.update_user_data(user_id, user_data)
            
            logger.info(f"ユーザー {user_id} の制限を強制リセットしました")
            
        except Exception as e:
            logger.error(f"強制リセットエラー: {e}")
            raise RateLimitError(f"制限のリセットに失敗しました: {e}")


# テスト用の関数
async def test_rate_limit_manager():
    """RateLimitManagerのテスト"""
    import tempfile
    import uuid
    from async_storage_manager import AsyncStorageManager
    
    # 一時ディレクトリでテスト
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test_letters.json")
        storage = AsyncStorageManager(test_file)
        rate_limiter = AsyncRateLimitManager(storage)
        
        print("=== RateLimitManagerテスト開始 ===")
        
        user_id = str(uuid.uuid4())
        
        # 初回リクエストチェック
        allowed, message = await rate_limiter.is_request_allowed(user_id)
        print(f"✓ 初回リクエストチェック: {allowed} - {message}")
        
        # リクエスト記録
        await rate_limiter.record_request(user_id)
        print("✓ リクエスト記録成功")
        
        # API呼び出し記録
        await rate_limiter.record_api_call(user_id, "groq")
        print("✓ API呼び出し記録成功")
        
        # 制限状況確認
        status = await rate_limiter.get_user_limits_status(user_id)
        print(f"✓ 制限状況確認: {status}")
        
        # 統計情報取得
        stats = await rate_limiter.get_rate_limit_stats()
        print(f"✓ 統計情報取得: {stats}")
        
        # デバッグモードテスト
        await rate_limiter.set_debug_mode(True)
        print("✓ デバッグモード有効化成功")
        
        # 強制リセットテスト（デバッグモード時のみ）
        await rate_limiter.force_reset_user_limits(user_id)
        print("✓ 強制リセット成功")
        
        print("=== 全てのテストが完了しました！ ===")


if __name__ == "__main__":
    asyncio.run(test_rate_limit_manager())