"""
非同期ストレージ管理クラス
JSONファイルベースの永続ストレージを提供し、
非同期ファイル操作とロック機能を実装します。
"""

import asyncio
import json
import os
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StorageError(Exception):
    """ストレージ関連のエラー"""
    pass


class AsyncStorageManager:
    """非同期ストレージ管理クラス"""
    
    def __init__(self, user_id: str,base_path: str = "mnt/data/users"):
        self.user_id = user_id
        self.base_path = Path(base_path)
        self.user_dir = self.base_path / user_id
        self.file_path = self.user_dir / "letters.json"
        self.backup_dir = self.file_path.parent / "backup"
        self.lock = asyncio.Lock()
        
        # ディレクトリの作成
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # データキャッシュ（重複読み込み防止）
        self._cached_data = None
        self._data_loaded = False
        
        # 初期データ構造
        self.default_data = {
            "users": {},
            "system": {
                "last_backup": None,
                "batch_runs": {},
                "created_at": datetime.now().isoformat()
            }
        }
    
    async def load_data(self, force_reload: bool = False) -> Dict[str, Any]:
        """データファイルを読み込み（キャッシュ機能付き）"""
        async with self.lock:
            # キャッシュされたデータがあり、強制リロードでない場合はキャッシュを返す
            if self._data_loaded and self._cached_data is not None and not force_reload:
                return self._cached_data.copy()
            
            try:
                if not self.file_path.exists():
                    if not self._data_loaded:  # 初回のみログ出力
                        logger.debug("データファイルが存在しないため、初期データを作成します")
                    await self._save_data_unsafe(self.default_data)
                    self._cached_data = self.default_data.copy()
                    self._data_loaded = True
                    return self._cached_data.copy()
                
                # ファイルサイズチェック
                if self.file_path.stat().st_size == 0:
                    if not self._data_loaded:  # 初回のみログ出力
                        logger.warning("データファイルが空のため、初期データを作成します")
                    await self._save_data_unsafe(self.default_data)
                    self._cached_data = self.default_data.copy()
                    self._data_loaded = True
                    return self._cached_data.copy()
                
                # JSONファイルの読み込み
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # データ構造の検証と修復
                data = self._validate_and_repair_data(data)
                
                # キャッシュを更新
                self._cached_data = data.copy()
                
                # 初回のみログ出力
                if not self._data_loaded:
                    logger.debug(f"データファイルを正常に読み込みました: {self.file_path}")
                    self._data_loaded = True
                
                return data
                
            except json.JSONDecodeError as e:
                logger.error(f"JSONファイルの形式が不正です: {e}")
                # バックアップからの復旧を試行
                return await self._restore_from_backup()
                
            except Exception as e:
                logger.error(f"データ読み込みエラー: {e}")
                raise StorageError(f"データの読み込みに失敗しました: {e}")
    
    async def save_data(self, data: Dict[str, Any]) -> None:
        """データファイルに保存"""
        async with self.lock:
            await self._save_data_unsafe(data)
            # キャッシュを更新
            self._cached_data = data.copy()
            self._data_loaded = True
    
    async def _save_data_unsafe(self, data: Dict[str, Any]) -> None:
        """ロックなしでデータを保存（内部使用）"""
        try:
            # データの検証
            validated_data = self._validate_and_repair_data(data)
            
            # 一時ファイルに書き込み
            temp_path = self.file_path.with_suffix('.tmp')
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(validated_data, f, ensure_ascii=False, indent=2)
            
            # アトミックな移動
            shutil.move(str(temp_path), str(self.file_path))
            
            # キャッシュを更新
            self._cached_data = validated_data.copy()
            self._data_loaded = True
            
            logger.debug(f"データを正常に保存しました: {self.file_path}")
            
        except Exception as e:
            logger.error(f"データ保存エラー: {e}")
            # 一時ファイルのクリーンアップ
            temp_path = self.file_path.with_suffix('.tmp')
            if temp_path.exists():
                temp_path.unlink()
            raise StorageError(f"データの保存に失敗しました: {e}")
    
    async def get_user_data(self, user_id: str) -> Dict[str, Any]:
        """特定ユーザーのデータを取得"""
        data = await self.load_data()
        
        if user_id not in data["users"]:
            # 新規ユーザーの初期データを作成
            user_data = {
                "profile": {
                    "created_at": datetime.now().isoformat(),
                    "last_request": None,
                    "total_letters": 0
                },
                "letters": {},
                "requests": {},
                "rate_limits": {
                    "daily_requests": {},
                    "api_calls": {}
                }
            }
            data["users"][user_id] = user_data
            await self.save_data(data)
            logger.info(f"新規ユーザーデータを作成しました: {user_id}")
        
        return data["users"][user_id]
    
    async def update_user_data(self, user_id: str, user_data: Dict[str, Any]) -> None:
        """特定ユーザーのデータを更新"""
        data = await self.load_data()
        data["users"][user_id] = user_data
        await self.save_data(data)
        logger.info(f"ユーザーデータを更新しました: {user_id}")
    
    async def get_all_users(self) -> List[str]:
        """全ユーザーIDのリストを取得"""
        data = await self.load_data()
        return list(data["users"].keys())
    
    async def backup_data(self) -> str:
        """データのバックアップを作成"""
        try:
            if not self.file_path.exists():
                logger.warning("バックアップ対象のファイルが存在しません")
                return ""
            
            # バックアップファイル名（タイムスタンプ付き）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"letters_backup_{timestamp}.json"
            backup_path = self.backup_dir / backup_filename
            
            # ファイルをコピー
            shutil.copy2(str(self.file_path), str(backup_path))
            
            # システム情報を更新
            data = await self.load_data()
            data["system"]["last_backup"] = datetime.now().isoformat()
            await self.save_data(data)
            
            logger.info(f"バックアップを作成しました: {backup_path}")
            
            # 古いバックアップファイルを削除（7日以上前）
            await self._cleanup_old_backups()
            
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"バックアップ作成エラー: {e}")
            raise StorageError(f"バックアップの作成に失敗しました: {e}")
    
    async def _restore_from_backup(self) -> Dict[str, Any]:
        """最新のバックアップから復旧"""
        try:
            # バックアップファイルを検索
            backup_files = list(self.backup_dir.glob("letters_backup_*.json"))
            
            if not backup_files:
                logger.warning("バックアップファイルが見つかりません。初期データを使用します")
                await self._save_data_unsafe(self.default_data)
                return self.default_data.copy()
            
            # 最新のバックアップファイルを選択
            latest_backup = max(backup_files, key=lambda p: p.stat().st_mtime)
            
            logger.info(f"バックアップから復旧します: {latest_backup}")
            
            with open(latest_backup, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 復旧したデータを保存
            await self._save_data_unsafe(data)
            
            # キャッシュを更新
            self._cached_data = data.copy()
            self._data_loaded = True
            
            return data
            
        except Exception as e:
            logger.error(f"バックアップからの復旧に失敗: {e}")
            logger.info("初期データを使用します")
            await self._save_data_unsafe(self.default_data)
            # キャッシュを更新
            self._cached_data = self.default_data.copy()
            self._data_loaded = True
            return self._cached_data.copy()
    
    async def _cleanup_old_backups(self, days: int = 7) -> None:
        """古いバックアップファイルを削除"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            backup_files = list(self.backup_dir.glob("letters_backup_*.json"))
            
            deleted_count = 0
            for backup_file in backup_files:
                file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                if file_time < cutoff_date:
                    backup_file.unlink()
                    deleted_count += 1
            
            if deleted_count > 0:
                logger.info(f"{deleted_count}個の古いバックアップファイルを削除しました")
                
        except Exception as e:
            logger.error(f"バックアップファイルの削除エラー: {e}")
    
    def _validate_and_repair_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """データ構造の検証と修復"""
        if not isinstance(data, dict):
            logger.warning("データが辞書形式ではありません。初期データを使用します")
            return self.default_data.copy()
        
        # 必要なキーの確認と修復
        if "users" not in data:
            data["users"] = {}
        
        if "system" not in data:
            data["system"] = self.default_data["system"].copy()
        
        # システム情報の修復
        system_defaults = {
            "last_backup": None,
            "batch_runs": {},
            "created_at": datetime.now().isoformat()
        }
        
        for key, default_value in system_defaults.items():
            if key not in data["system"]:
                data["system"][key] = default_value
        
        # ユーザーデータの修復
        for user_id, user_data in data["users"].items():
            if not isinstance(user_data, dict):
                continue
            
            # 必要なキーの確認
            user_defaults = {
                "profile": {
                    "created_at": datetime.now().isoformat(),
                    "last_request": None,
                    "total_letters": 0
                },
                "letters": {},
                "requests": {},
                "rate_limits": {
                    "daily_requests": {},
                    "api_calls": {}
                }
            }
            
            for key, default_value in user_defaults.items():
                if key not in user_data:
                    user_data[key] = default_value
        
        return data
    
    async def get_system_info(self) -> Dict[str, Any]:
        """システム情報を取得"""
        data = await self.load_data()
        return data["system"]
    
    async def update_system_info(self, system_info: Dict[str, Any]) -> None:
        """システム情報を更新"""
        data = await self.load_data()
        data["system"].update(system_info)
        await self.save_data(data)
    
    async def cleanup_old_data(self, days: int = 90) -> int:
        """古いデータを削除"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d")
            
            data = await self.load_data()
            deleted_count = 0
            
            for user_id, user_data in data["users"].items():
                # 古い手紙を削除
                letters_to_delete = []
                for date_str in user_data["letters"]:
                    if date_str < cutoff_str:
                        letters_to_delete.append(date_str)
                
                for date_str in letters_to_delete:
                    del user_data["letters"][date_str]
                    deleted_count += 1
                
                # 古いリクエストを削除
                requests_to_delete = []
                for date_str in user_data["requests"]:
                    if date_str < cutoff_str:
                        requests_to_delete.append(date_str)
                
                for date_str in requests_to_delete:
                    del user_data["requests"][date_str]
                
                # 古いレート制限データを削除
                for limit_type in ["daily_requests", "api_calls"]:
                    if limit_type in user_data["rate_limits"]:
                        dates_to_delete = []
                        for date_str in user_data["rate_limits"][limit_type]:
                            if date_str < cutoff_str:
                                dates_to_delete.append(date_str)
                        
                        for date_str in dates_to_delete:
                            del user_data["rate_limits"][limit_type][date_str]
            
            if deleted_count > 0:
                await self.save_data(data)
                logger.info(f"{deleted_count}件の古いデータを削除しました")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"古いデータの削除エラー: {e}")
            return 0
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """ストレージの統計情報を取得"""
        try:
            data = await self.load_data()
            
            total_users = len(data["users"])
            total_letters = sum(len(user_data["letters"]) for user_data in data["users"].values())
            total_requests = sum(len(user_data["requests"]) for user_data in data["users"].values())
            
            file_size = self.file_path.stat().st_size if self.file_path.exists() else 0
            backup_count = len(list(self.backup_dir.glob("letters_backup_*.json")))
            
            return {
                "total_users": total_users,
                "total_letters": total_letters,
                "total_requests": total_requests,
                "file_size_bytes": file_size,
                "backup_count": backup_count,
                "last_backup": data["system"].get("last_backup"),
                "created_at": data["system"].get("created_at")
            }
            
        except Exception as e:
            logger.error(f"統計情報の取得エラー: {e}")
            return {}
    
    def invalidate_cache(self) -> None:
        """キャッシュを無効化（外部からファイルが変更された場合など）"""
        self._cached_data = None
        self._data_loaded = False
        logger.debug("データキャッシュを無効化しました")
    
    async def reload_data(self) -> Dict[str, Any]:
        """データを強制的に再読み込み"""
        return await self.load_data(force_reload=True)


# テスト用の関数
async def test_storage_manager():
    """StorageManagerのテスト"""
    import tempfile
    import uuid
    
    # 一時ディレクトリでテスト
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test_letters.json")
        storage = AsyncStorageManager(test_file)
        
        print("=== StorageManagerテスト開始 ===")
        
        # 初期データの読み込みテスト
        data = await storage.load_data()
        print("✓ 初期データの読み込み成功")
        
        # ユーザーデータの作成テスト
        user_id = str(uuid.uuid4())
        user_data = await storage.get_user_data(user_id)
        print("✓ ユーザーデータの作成成功")
        
        # ユーザーデータの更新テスト
        user_data["profile"]["total_letters"] = 1
        user_data["letters"]["2024-01-20"] = {
            "theme": "テストテーマ",
            "content": "テスト手紙の内容",
            "status": "completed",
            "generated_at": datetime.now().isoformat()
        }
        await storage.update_user_data(user_id, user_data)
        print("✓ ユーザーデータの更新成功")
        
        # データの再読み込みテスト
        updated_data = await storage.get_user_data(user_id)
        assert updated_data["profile"]["total_letters"] == 1
        print("✓ データの永続化確認成功")
        
        # バックアップテスト
        backup_path = await storage.backup_data()
        print(f"✓ バックアップ作成成功: {backup_path}")
        
        # 統計情報テスト
        stats = await storage.get_storage_stats()
        print(f"✓ 統計情報取得成功: {stats}")
        
        print("=== 全てのテストが完了しました！ ===")


if __name__ == "__main__":
    asyncio.run(test_storage_manager())