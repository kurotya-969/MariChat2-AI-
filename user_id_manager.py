"""
ユーザーID永続化管理モジュール
ローカル環境でユーザーIDをファイルに保存し、仮想環境を閉じても継続してプレイできるようにする
"""
import os
import json
import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class UserIDManager:
    """ユーザーIDの永続化を管理するクラス"""
    
    def __init__(self, storage_dir: str = "user_data"):
        """
        Args:
            storage_dir: ユーザーデータを保存するディレクトリ
        """
        self.storage_dir = storage_dir
        self.user_id_file = os.path.join(storage_dir, "user_id.json")
        self._ensure_storage_dir()
    
    def _ensure_storage_dir(self):
        """ストレージディレクトリが存在することを確認"""
        try:
            if not os.path.exists(self.storage_dir):
                os.makedirs(self.storage_dir)
                logger.info(f"ユーザーデータディレクトリを作成: {self.storage_dir}")
        except Exception as e:
            logger.error(f"ストレージディレクトリ作成エラー: {e}")
    
    def get_or_create_user_id(self) -> str:
        """
        保存されたユーザーIDを取得するか、新規作成する
        
        Returns:
            ユーザーID
        """
        try:
            # 既存のユーザーIDファイルをチェック
            if os.path.exists(self.user_id_file):
                user_data = self._load_user_data()
                if user_data and "user_id" in user_data:
                    user_id = user_data["user_id"]
                    logger.info(f"既存のユーザーIDを読み込み: {user_id[:8]}...")
                    
                    # 最終アクセス時刻を更新
                    self._update_last_access(user_id)
                    return user_id
            
            # 新規ユーザーIDを作成
            user_id = self._generate_new_user_id()
            self._save_user_data(user_id)
            logger.info(f"新規ユーザーIDを作成: {user_id[:8]}...")
            return user_id
            
        except Exception as e:
            logger.error(f"ユーザーID取得エラー: {e}")
            # フォールバック: 一時的なIDを生成
            return str(uuid.uuid4())
    
    def _generate_new_user_id(self) -> str:
        """新しいユーザーIDを生成"""
        return str(uuid.uuid4())
    
    def _load_user_data(self) -> Optional[Dict[str, Any]]:
        """ユーザーデータファイルを読み込み"""
        try:
            with open(self.user_id_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        except Exception as e:
            logger.error(f"ユーザーデータ読み込みエラー: {e}")
            return None
    
    def _save_user_data(self, user_id: str, game_data: Optional[Dict[str, Any]] = None):
        """ユーザーデータをファイルに保存"""
        try:
            user_data = {
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "last_access": datetime.now().isoformat(),
                "version": "1.0",
                "game_data": game_data or {}
            }
            
            with open(self.user_id_file, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"ユーザーデータを保存: {self.user_id_file}")
            
        except Exception as e:
            logger.error(f"ユーザーデータ保存エラー: {e}")
    
    def _update_last_access(self, user_id: str):
        """最終アクセス時刻を更新"""
        try:
            user_data = self._load_user_data()
            if user_data:
                user_data["last_access"] = datetime.now().isoformat()
                
                with open(self.user_id_file, 'w', encoding='utf-8') as f:
                    json.dump(user_data, f, ensure_ascii=False, indent=2)
                
                logger.debug(f"最終アクセス時刻を更新: {user_id[:8]}...")
                
        except Exception as e:
            logger.error(f"最終アクセス時刻更新エラー: {e}")
    
    def get_user_info(self) -> Optional[Dict[str, Any]]:
        """ユーザー情報を取得"""
        return self._load_user_data()
    
    def delete_user_data(self) -> bool:
        """
        ユーザーデータを削除（フルリセット用）
        
        Returns:
            削除成功かどうか
        """
        try:
            if os.path.exists(self.user_id_file):
                os.remove(self.user_id_file)
                logger.info(f"ユーザーデータファイルを削除: {self.user_id_file}")
                return True
            else:
                logger.info("削除対象のユーザーデータファイルが存在しません")
                return True
                
        except Exception as e:
            logger.error(f"ユーザーデータ削除エラー: {e}")
            return False
    
    def reset_user_id(self) -> str:
        """
        ユーザーIDをリセットして新規作成
        
        Returns:
            新しいユーザーID
        """
        try:
            # 既存データを削除
            self.delete_user_data()
            
            # 新規IDを作成
            new_user_id = self._generate_new_user_id()
            self._save_user_data(new_user_id)
            
            logger.info(f"ユーザーIDをリセット: {new_user_id[:8]}...")
            return new_user_id
            
        except Exception as e:
            logger.error(f"ユーザーIDリセットエラー: {e}")
            return str(uuid.uuid4())
    
    def is_user_data_exists(self) -> bool:
        """ユーザーデータファイルが存在するかチェック"""
        return os.path.exists(self.user_id_file)
    
    def get_storage_info(self) -> Dict[str, Any]:
        """ストレージ情報を取得（デバッグ用）"""
        try:
            info = {
                "storage_dir": self.storage_dir,
                "user_id_file": self.user_id_file,
                "file_exists": os.path.exists(self.user_id_file),
                "dir_exists": os.path.exists(self.storage_dir)
            }
            
            if info["file_exists"]:
                stat = os.stat(self.user_id_file)
                info["file_size"] = stat.st_size
                info["modified_time"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
            
            return info
            
        except Exception as e:
            logger.error(f"ストレージ情報取得エラー: {e}")
            return {"error": str(e)}
    
    def save_game_data(self, user_id: str, game_data: Dict[str, Any]) -> bool:
        """
        ゲームデータを保存
        
        Args:
            user_id: ユーザーID
            game_data: 保存するゲームデータ
            
        Returns:
            保存成功かどうか
        """
        try:
            user_data = self._load_user_data()
            if not user_data:
                # ユーザーデータが存在しない場合は新規作成
                user_data = {
                    "user_id": user_id,
                    "created_at": datetime.now().isoformat(),
                    "version": "1.0"
                }
            
            # ゲームデータを更新
            user_data["game_data"] = game_data
            user_data["last_access"] = datetime.now().isoformat()
            
            with open(self.user_id_file, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"ゲームデータを保存: {user_id[:8]}...")
            return True
            
        except Exception as e:
            logger.error(f"ゲームデータ保存エラー: {e}")
            return False
    
    def load_game_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        ゲームデータを読み込み
        
        Args:
            user_id: ユーザーID
            
        Returns:
            ゲームデータ（存在しない場合はNone）
        """
        try:
            user_data = self._load_user_data()
            if user_data and user_data.get("user_id") == user_id:
                game_data = user_data.get("game_data", {})
                logger.info(f"ゲームデータを読み込み: {user_id[:8]}... (データサイズ: {len(str(game_data))}文字)")
                return game_data
            else:
                logger.info(f"ゲームデータが見つかりません: {user_id[:8]}...")
                return None
                
        except Exception as e:
            logger.error(f"ゲームデータ読み込みエラー: {e}")
            return None