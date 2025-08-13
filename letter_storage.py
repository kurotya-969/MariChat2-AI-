"""
ストレージ管理モジュール
Storage management module
"""

import json
import asyncio
import aiofiles
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from letter_config import Config
from letter_logger import get_storage_logger

logger = get_storage_logger()

class LetterStorage:
    """手紙データのストレージ管理クラス"""
    
    def __init__(self):
        self.config = Config()
        self.storage_path = Path(self.config.STORAGE_PATH)
        self.backup_path = Path(self.config.BACKUP_PATH)
        self.logger = logger
    
    async def save_letter(self, letter_data: Dict[str, Any]) -> bool:
        """
        手紙データを保存する
        
        Args:
            letter_data: 保存する手紙データ
            
        Returns:
            保存が成功したかどうか
        """
        try:
            # タイムスタンプを追加
            letter_data['saved_at'] = datetime.now().isoformat()
            
            # 既存データを読み込み
            existing_data = await self._load_data()
            
            # 新しいデータを追加
            if 'letters' not in existing_data:
                existing_data['letters'] = []
            
            existing_data['letters'].append(letter_data)
            
            # データを保存
            await self._save_data(existing_data)
            
            self.logger.info(f"手紙データを保存しました: {letter_data.get('id', 'unknown')}")
            return True
            
        except Exception as e:
            self.logger.error(f"手紙データの保存中にエラーが発生しました: {e}")
            return False
    
    async def load_letters(self) -> List[Dict[str, Any]]:
        """
        保存された手紙データを読み込む
        
        Returns:
            手紙データのリスト
        """
        try:
            data = await self._load_data()
            return data.get('letters', [])
            
        except Exception as e:
            self.logger.error(f"手紙データの読み込み中にエラーが発生しました: {e}")
            return []
    
    async def get_letter_by_id(self, letter_id: str) -> Optional[Dict[str, Any]]:
        """
        IDで手紙データを取得する
        
        Args:
            letter_id: 手紙のID
            
        Returns:
            手紙データ（見つからない場合はNone）
        """
        try:
            letters = await self.load_letters()
            for letter in letters:
                if letter.get('id') == letter_id:
                    return letter
            return None
            
        except Exception as e:
            self.logger.error(f"手紙データの取得中にエラーが発生しました: {e}")
            return None
    
    async def delete_letter(self, letter_id: str) -> bool:
        """
        手紙データを削除する
        
        Args:
            letter_id: 削除する手紙のID
            
        Returns:
            削除が成功したかどうか
        """
        try:
            data = await self._load_data()
            letters = data.get('letters', [])
            
            # 指定されたIDの手紙を削除
            original_count = len(letters)
            letters = [letter for letter in letters if letter.get('id') != letter_id]
            
            if len(letters) < original_count:
                data['letters'] = letters
                await self._save_data(data)
                self.logger.info(f"手紙データを削除しました: {letter_id}")
                return True
            else:
                self.logger.warning(f"削除対象の手紙が見つかりませんでした: {letter_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"手紙データの削除中にエラーが発生しました: {e}")
            return False
    
    async def backup_data(self) -> bool:
        """
        データをバックアップする
        
        Returns:
            バックアップが成功したかどうか
        """
        try:
            if not self.storage_path.exists():
                self.logger.warning("バックアップ対象のファイルが存在しません")
                return False
            
            # バックアップディレクトリを作成
            self.backup_path.mkdir(parents=True, exist_ok=True)
            
            # タイムスタンプ付きのバックアップファイル名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_path / f"letters_backup_{timestamp}.json"
            
            # ファイルをコピー
            async with aiofiles.open(self.storage_path, 'r', encoding='utf-8') as src:
                content = await src.read()
                async with aiofiles.open(backup_file, 'w', encoding='utf-8') as dst:
                    await dst.write(content)
            
            self.logger.info(f"データをバックアップしました: {backup_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"データのバックアップ中にエラーが発生しました: {e}")
            return False
    
    async def _load_data(self) -> Dict[str, Any]:
        """内部用：データファイルを読み込む"""
        try:
            if not self.storage_path.exists():
                return {}
            
            async with aiofiles.open(self.storage_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content) if content.strip() else {}
                
        except Exception as e:
            self.logger.error(f"データファイルの読み込み中にエラーが発生しました: {e}")
            return {}
    
    async def _save_data(self, data: Dict[str, Any]) -> None:
        """内部用：データファイルに保存する"""
        # ディレクトリを作成
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(self.storage_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(data, ensure_ascii=False, indent=2))

# グローバルストレージインスタンス
storage_instance = None

def get_storage() -> LetterStorage:
    """ストレージインスタンスを取得（シングルトン）"""
    global storage_instance
    
    if storage_instance is None:
        storage_instance = LetterStorage()
    
    return storage_instance