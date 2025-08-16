"""
ユーザー管理クラス
ユーザープロファイル管理機能と
ユーザー履歴の更新・取得機能を提供します。
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
import uuid
import hashlib
import secrets

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserError(Exception):
    """ユーザー関連のエラー"""
    pass


class UserManager:
    """ユーザー管理クラス"""
    
    def __init__(self, storage_manager):
        self.storage = storage_manager
        
        # 設定値
        self.session_timeout_hours = int(os.getenv("SESSION_TIMEOUT_HOURS", "24"))
        self.max_history_entries = int(os.getenv("MAX_HISTORY_ENTRIES", "100"))
        self.user_data_retention_days = int(os.getenv("USER_DATA_RETENTION_DAYS", "90"))
        
        logger.info(f"UserManager初期化完了 - セッションタイムアウト: {self.session_timeout_hours}時間")
    
    def generate_user_id(self) -> str:
        """
        新しいユーザーIDを生成する
        
        Returns:
            str: 生成されたユーザーID（UUID4形式）
        """
        return str(uuid.uuid4())
    
    def generate_session_id(self) -> str:
        """
        新しいセッションIDを生成する
        
        Returns:
            str: 生成されたセッションID
        """
        return secrets.token_urlsafe(32)
    
    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        ユーザープロファイルを取得する
        
        Args:
            user_id: ユーザーID
            
        Returns:
            Dict: ユーザープロファイル
        """
        try:
            user_data = await self.storage.get_user_data(user_id)
            profile = user_data["profile"].copy()
            
            # 追加の統計情報を計算
            profile["total_requests"] = len(user_data["requests"])
            profile["completed_letters"] = len([
                letter for letter in user_data["letters"].values()
                if letter.get("status") == "completed"
            ])
            profile["pending_requests"] = len([
                request for request in user_data["requests"].values()
                if request.get("status") == "pending"
            ])
            
            # 最後のアクティビティ時刻を計算
            last_activity = self._calculate_last_activity(user_data)
            if last_activity:
                profile["last_activity"] = last_activity
            
            return profile
            
        except Exception as e:
            logger.error(f"ユーザープロファイル取得エラー: {e}")
            raise UserError(f"ユーザープロファイルの取得に失敗しました: {e}")
    
    async def update_user_profile(self, user_id: str, profile_updates: Dict[str, Any]) -> bool:
        """
        ユーザープロファイルを更新する
        
        Args:
            user_id: ユーザーID
            profile_updates: 更新するプロファイル情報
            
        Returns:
            bool: 更新成功フラグ
        """
        try:
            user_data = await self.storage.get_user_data(user_id)
            
            # 更新可能なフィールドのみを許可
            allowed_fields = {
                "display_name", "preferences", "timezone", "language",
                "notification_settings", "theme_preferences"
            }
            
            for key, value in profile_updates.items():
                if key in allowed_fields:
                    user_data["profile"][key] = value
            
            # 更新時刻を記録
            user_data["profile"]["updated_at"] = datetime.now().isoformat()
            
            await self.storage.update_user_data(user_id, user_data)
            
            logger.info(f"ユーザープロファイルを更新しました: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"ユーザープロファイル更新エラー: {e}")
            return False
    
    async def update_user_history(self, user_id: str, interaction: Dict[str, Any]) -> bool:
        """
        ユーザー履歴を更新する
        
        Args:
            user_id: ユーザーID
            interaction: インタラクション情報
            
        Returns:
            bool: 更新成功フラグ
        """
        try:
            user_data = await self.storage.get_user_data(user_id)
            
            # 履歴エントリの作成
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": interaction.get("type", "unknown"),
                "action": interaction.get("action", ""),
                "details": interaction.get("details", {}),
                "session_id": interaction.get("session_id", ""),
                "entry_id": str(uuid.uuid4())
            }
            
            # 履歴配列の初期化（存在しない場合）
            if "history" not in user_data:
                user_data["history"] = []
            
            # 履歴エントリを追加
            user_data["history"].append(history_entry)
            
            # 履歴の上限チェックと古いエントリの削除
            if len(user_data["history"]) > self.max_history_entries:
                # 古いエントリから削除
                user_data["history"] = user_data["history"][-self.max_history_entries:]
            
            # プロファイルの統計情報を更新
            await self._update_profile_stats(user_data, interaction)
            
            await self.storage.update_user_data(user_id, user_data)
            
            logger.info(f"ユーザー履歴を更新しました: {user_id} - {interaction.get('type', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"ユーザー履歴更新エラー: {e}")
            return False
    
    async def get_user_letter_history(self, user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        ユーザーの手紙履歴を取得する
        
        Args:
            user_id: ユーザーID
            limit: 取得件数の上限
            
        Returns:
            List[Dict]: 手紙履歴のリスト
        """
        try:
            user_data = await self.storage.get_user_data(user_id)
            letters = user_data["letters"]
            
            # 手紙データを日付順（新しい順）でソート
            sorted_letters = []
            for date, letter_data in letters.items():
                letter_info = {
                    "date": date,
                    "theme": letter_data.get("theme", ""),
                    "status": letter_data.get("status", "unknown"),
                    "generated_at": letter_data.get("generated_at"),
                    "content_length": len(letter_data.get("content", "")),
                    "metadata": letter_data.get("metadata", {})
                }
                sorted_letters.append(letter_info)
            
            # 日付順でソート（新しい順）
            sorted_letters.sort(key=lambda x: x["date"], reverse=True)
            
            # 上限が指定されている場合は制限
            if limit:
                sorted_letters = sorted_letters[:limit]
            
            return sorted_letters
            
        except Exception as e:
            logger.error(f"手紙履歴取得エラー: {e}")
            return []
    
    async def get_user_interaction_history(self, user_id: str, 
                                         interaction_type: Optional[str] = None,
                                         limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        ユーザーのインタラクション履歴を取得する
        
        Args:
            user_id: ユーザーID
            interaction_type: フィルタするインタラクションタイプ
            limit: 取得件数の上限
            
        Returns:
            List[Dict]: インタラクション履歴のリスト
        """
        try:
            user_data = await self.storage.get_user_data(user_id)
            history = user_data.get("history", [])
            
            # タイプでフィルタ
            if interaction_type:
                history = [entry for entry in history if entry.get("type") == interaction_type]
            
            # 時刻順でソート（新しい順）
            history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            # 上限が指定されている場合は制限
            if limit:
                history = history[:limit]
            
            return history
            
        except Exception as e:
            logger.error(f"インタラクション履歴取得エラー: {e}")
            return []
    
    async def create_user_session(self, user_id: str, session_info: Dict[str, Any]) -> str:
        """
        ユーザーセッションを作成する
        
        Args:
            user_id: ユーザーID
            session_info: セッション情報
            
        Returns:
            str: セッションID
        """
        try:
            session_id = self.generate_session_id()
            user_data = await self.storage.get_user_data(user_id)
            
            # セッション情報の作成
            session_data = {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(hours=self.session_timeout_hours)).isoformat(),
                "ip_address": session_info.get("ip_address", ""),
                "user_agent": session_info.get("user_agent", ""),
                "is_active": True
            }
            
            # セッション配列の初期化（存在しない場合）
            if "sessions" not in user_data:
                user_data["sessions"] = []
            
            # 古いセッションを無効化
            await self._cleanup_expired_sessions(user_data)
            
            # 新しいセッションを追加
            user_data["sessions"].append(session_data)
            
            await self.storage.update_user_data(user_id, user_data)
            
            logger.info(f"ユーザーセッションを作成しました: {user_id} - {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"セッション作成エラー: {e}")
            raise UserError(f"セッションの作成に失敗しました: {e}")
    
    async def validate_user_session(self, user_id: str, session_id: str) -> bool:
        """
        ユーザーセッションを検証する
        
        Args:
            user_id: ユーザーID
            session_id: セッションID
            
        Returns:
            bool: セッション有効フラグ
        """
        try:
            user_data = await self.storage.get_user_data(user_id)
            sessions = user_data.get("sessions", [])
            
            for session in sessions:
                if (session.get("session_id") == session_id and 
                    session.get("is_active", False)):
                    
                    # 有効期限チェック
                    expires_at = datetime.fromisoformat(session["expires_at"])
                    if datetime.now() < expires_at:
                        return True
                    else:
                        # 期限切れセッションを無効化
                        session["is_active"] = False
                        await self.storage.update_user_data(user_id, user_data)
            
            return False
            
        except Exception as e:
            logger.error(f"セッション検証エラー: {e}")
            return False
    
    async def invalidate_user_session(self, user_id: str, session_id: str) -> bool:
        """
        ユーザーセッションを無効化する
        
        Args:
            user_id: ユーザーID
            session_id: セッションID
            
        Returns:
            bool: 無効化成功フラグ
        """
        try:
            user_data = await self.storage.get_user_data(user_id)
            sessions = user_data.get("sessions", [])
            
            for session in sessions:
                if session.get("session_id") == session_id:
                    session["is_active"] = False
                    session["invalidated_at"] = datetime.now().isoformat()
                    
                    await self.storage.update_user_data(user_id, user_data)
                    
                    logger.info(f"セッションを無効化しました: {user_id} - {session_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"セッション無効化エラー: {e}")
            return False
    
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        ユーザーの設定を取得する
        
        Args:
            user_id: ユーザーID
            
        Returns:
            Dict: ユーザー設定
        """
        try:
            profile = await self.get_user_profile(user_id)
            
            # デフォルト設定
            default_preferences = {
                "theme": "light",
                "language": "ja",
                "timezone": "Asia/Tokyo",
                "notification_enabled": True,
                "generation_time_preference": 2,  # デフォルトは2時
                "theme_suggestions": True,
                "history_retention": True
            }
            
            # ユーザー設定をマージ
            preferences = default_preferences.copy()
            if "preferences" in profile:
                preferences.update(profile["preferences"])
            
            return preferences
            
        except Exception as e:
            logger.error(f"ユーザー設定取得エラー: {e}")
            return {}
    
    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """
        ユーザーの設定を更新する
        
        Args:
            user_id: ユーザーID
            preferences: 更新する設定
            
        Returns:
            bool: 更新成功フラグ
        """
        try:
            current_preferences = await self.get_user_preferences(user_id)
            current_preferences.update(preferences)
            
            return await self.update_user_profile(user_id, {"preferences": current_preferences})
            
        except Exception as e:
            logger.error(f"ユーザー設定更新エラー: {e}")
            return False
    
    async def _update_profile_stats(self, user_data: Dict[str, Any], interaction: Dict[str, Any]) -> None:
        """
        プロファイルの統計情報を更新する（内部使用）
        
        Args:
            user_data: ユーザーデータ
            interaction: インタラクション情報
        """
        try:
            profile = user_data["profile"]
            interaction_type = interaction.get("type", "")
            
            # インタラクションタイプ別の統計更新
            if interaction_type == "letter_request":
                profile["total_letters"] = profile.get("total_letters", 0) + 1
            elif interaction_type == "letter_generated":
                # 生成完了時の統計更新は別途処理
                pass
            elif interaction_type == "app_access":
                profile["last_access"] = datetime.now().isoformat()
            
            # 最終アクティビティ時刻を更新
            profile["last_activity"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"プロファイル統計更新エラー: {e}")
    
    def _calculate_last_activity(self, user_data: Dict[str, Any]) -> Optional[str]:
        """
        最後のアクティビティ時刻を計算する（内部使用）
        
        Args:
            user_data: ユーザーデータ
            
        Returns:
            Optional[str]: 最後のアクティビティ時刻
        """
        try:
            timestamps = []
            
            # プロファイルの最終アクセス時刻
            if "last_access" in user_data["profile"]:
                timestamps.append(user_data["profile"]["last_access"])
            
            # 履歴の最新エントリ
            history = user_data.get("history", [])
            if history:
                latest_history = max(history, key=lambda x: x.get("timestamp", ""))
                timestamps.append(latest_history["timestamp"])
            
            # リクエストの最新エントリ
            requests = user_data.get("requests", {})
            if requests:
                latest_request = max(requests.values(), key=lambda x: x.get("requested_at", ""))
                timestamps.append(latest_request["requested_at"])
            
            if timestamps:
                return max(timestamps)
            
            return None
            
        except Exception as e:
            logger.error(f"最終アクティビティ計算エラー: {e}")
            return None
    
    async def _cleanup_expired_sessions(self, user_data: Dict[str, Any]) -> None:
        """
        期限切れセッションをクリーンアップする（内部使用）
        
        Args:
            user_data: ユーザーデータ
        """
        try:
            sessions = user_data.get("sessions", [])
            current_time = datetime.now()
            
            for session in sessions:
                if session.get("is_active", False):
                    expires_at = datetime.fromisoformat(session["expires_at"])
                    if current_time >= expires_at:
                        session["is_active"] = False
                        session["expired_at"] = current_time.isoformat()
            
        except Exception as e:
            logger.error(f"セッションクリーンアップエラー: {e}")
    
    async def cleanup_old_user_data(self, days: int = None) -> int:
        """
        古いユーザーデータを削除する
        
        Args:
            days: 保持日数（指定しない場合は設定値を使用）
            
        Returns:
            int: 削除されたエントリ数
        """
        try:
            if days is None:
                days = self.user_data_retention_days
            
            cutoff_date = datetime.now() - timedelta(days=days)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d")
            
            all_users = await self.storage.get_all_users()
            deleted_count = 0
            
            for user_id in all_users:
                user_data = await self.storage.get_user_data(user_id)
                
                # 古い履歴エントリを削除
                history = user_data.get("history", [])
                original_count = len(history)
                
                user_data["history"] = [
                    entry for entry in history
                    if entry.get("timestamp", "").split("T")[0] >= cutoff_str
                ]
                
                deleted_count += original_count - len(user_data["history"])
                
                # 古いセッションを削除
                sessions = user_data.get("sessions", [])
                original_session_count = len(sessions)
                
                user_data["sessions"] = [
                    session for session in sessions
                    if session.get("created_at", "").split("T")[0] >= cutoff_str
                ]
                
                deleted_count += original_session_count - len(user_data["sessions"])
                
                if (original_count != len(user_data["history"]) or 
                    original_session_count != len(user_data["sessions"])):
                    await self.storage.update_user_data(user_id, user_data)
            
            if deleted_count > 0:
                logger.info(f"{deleted_count}件の古いユーザーデータを削除しました")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"古いユーザーデータ削除エラー: {e}")
            return 0
    
    async def get_user_statistics(self) -> Dict[str, Any]:
        """
        ユーザーの統計情報を取得する
        
        Returns:
            Dict: 統計情報
        """
        try:
            all_users = await self.storage.get_all_users()
            today = datetime.now().strftime("%Y-%m-%d")
            week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            
            stats = {
                "total_users": len(all_users),
                "active_users_today": 0,
                "active_users_week": 0,
                "total_letters": 0,
                "total_requests": 0,
                "total_sessions": 0,
                "active_sessions": 0,
                "date": today
            }
            
            for user_id in all_users:
                user_data = await self.storage.get_user_data(user_id)
                
                # 手紙とリクエストの総数
                stats["total_letters"] += len(user_data.get("letters", {}))
                stats["total_requests"] += len(user_data.get("requests", {}))
                
                # セッション統計
                sessions = user_data.get("sessions", [])
                stats["total_sessions"] += len(sessions)
                
                for session in sessions:
                    if session.get("is_active", False):
                        expires_at = datetime.fromisoformat(session["expires_at"])
                        if datetime.now() < expires_at:
                            stats["active_sessions"] += 1
                
                # アクティブユーザー統計
                last_activity = self._calculate_last_activity(user_data)
                if last_activity:
                    activity_date = last_activity.split("T")[0]
                    if activity_date >= today:
                        stats["active_users_today"] += 1
                    if activity_date >= week_ago:
                        stats["active_users_week"] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"ユーザー統計取得エラー: {e}")
            return {"error": str(e)}


# テスト用の関数
async def test_user_manager():
    """UserManagerのテスト"""
    import tempfile
    from async_storage_manager import AsyncStorageManager
    
    # 一時ディレクトリでテスト
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test_letters.json")
        storage = AsyncStorageManager(test_file)
        user_manager = UserManager(storage)
        
        print("=== UserManagerテスト開始 ===")
        
        # ユーザーID生成テスト
        user_id = user_manager.generate_user_id()
        print(f"✓ ユーザーID生成テスト: {user_id}")
        
        # プロファイル取得テスト
        profile = await user_manager.get_user_profile(user_id)
        print(f"✓ プロファイル取得テスト: {profile}")
        
        # 履歴更新テスト
        interaction = {
            "type": "letter_request",
            "action": "submit_request",
            "details": {"theme": "テストテーマ"}
        }
        updated = await user_manager.update_user_history(user_id, interaction)
        print(f"✓ 履歴更新テスト: {updated}")
        
        # セッション作成テスト
        session_info = {"ip_address": "127.0.0.1", "user_agent": "test"}
        session_id = await user_manager.create_user_session(user_id, session_info)
        print(f"✓ セッション作成テスト: {session_id}")
        
        # セッション検証テスト
        valid = await user_manager.validate_user_session(user_id, session_id)
        print(f"✓ セッション検証テスト: {valid}")
        
        # 統計情報取得テスト
        stats = await user_manager.get_user_statistics()
        print(f"✓ 統計情報取得テスト: {stats}")
        
        print("=== 全てのテストが完了しました！ ===")


if __name__ == "__main__":
    asyncio.run(test_user_manager())