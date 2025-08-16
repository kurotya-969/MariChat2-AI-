"""
手紙管理マネージャー
Letter management manager
"""

import uuid
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
from letter_models import Letter, LetterRequest, LetterContent, LetterStatus, UserPreferences
from letter_storage import get_storage
from letter_logger import get_app_logger

logger = get_app_logger()

class LetterManager:
    """手紙の生成と管理を行うマネージャークラス"""
    
    def __init__(self):
        self.storage = get_storage()
        self.logger = logger
    
    async def create_letter_request(
        self, 
        user_id: str, 
        message: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        手紙生成リクエストを作成する
        
        Args:
            user_id: ユーザーID
            message: ユーザーからのメッセージ
            preferences: ユーザー設定
            
        Returns:
            作成された手紙のID
        """
        try:
            # 一意のIDを生成
            letter_id = str(uuid.uuid4())
            
            # リクエストオブジェクトを作成
            request = LetterRequest(
                user_id=user_id,
                message=message,
                preferences=preferences or {}
            )
            
            # 手紙オブジェクトを作成
            letter = Letter(
                id=letter_id,
                request=request,
                status=LetterStatus.PENDING
            )
            
            # ストレージに保存
            await self.storage.save_letter(letter.dict())
            
            self.logger.info(f"手紙リクエストを作成しました: {letter_id}")
            return letter_id
            
        except Exception as e:
            self.logger.error(f"手紙リクエストの作成中にエラーが発生しました: {e}")
            raise
    
    async def get_letter(self, letter_id: str) -> Optional[Letter]:
        """
        手紙データを取得する
        
        Args:
            letter_id: 手紙のID
            
        Returns:
            手紙データ（見つからない場合はNone）
        """
        try:
            letter_data = await self.storage.get_letter_by_id(letter_id)
            if letter_data:
                return Letter(**letter_data)
            return None
            
        except Exception as e:
            self.logger.error(f"手紙データの取得中にエラーが発生しました: {e}")
            return None
    
    async def get_user_letters(self, user_id: str) -> List[Letter]:
        """
        ユーザーの手紙一覧を取得する
        
        Args:
            user_id: ユーザーID
            
        Returns:
            ユーザーの手紙リスト
        """
        try:
            all_letters = await self.storage.load_letters()
            user_letters = []
            
            for letter_data in all_letters:
                letter = Letter(**letter_data)
                if letter.request.user_id == user_id:
                    user_letters.append(letter)
            
            # 作成日時でソート（新しい順）
            user_letters.sort(key=lambda x: x.created_at, reverse=True)
            
            return user_letters
            
        except Exception as e:
            self.logger.error(f"ユーザー手紙一覧の取得中にエラーが発生しました: {e}")
            return []
    
    async def update_letter_status(
        self, 
        letter_id: str, 
        status: LetterStatus,
        error_message: Optional[str] = None
    ) -> bool:
        """
        手紙のステータスを更新する
        
        Args:
            letter_id: 手紙のID
            status: 新しいステータス
            error_message: エラーメッセージ（エラー時）
            
        Returns:
            更新が成功したかどうか
        """
        try:
            letter = await self.get_letter(letter_id)
            if not letter:
                self.logger.warning(f"更新対象の手紙が見つかりませんでした: {letter_id}")
                return False
            
            letter.update_status(status, error_message)
            
            # ストレージを更新
            await self._update_letter_in_storage(letter)
            
            self.logger.info(f"手紙ステータスを更新しました: {letter_id} -> {status}")
            return True
            
        except Exception as e:
            self.logger.error(f"手紙ステータスの更新中にエラーが発生しました: {e}")
            return False
    
    async def set_letter_content(
        self, 
        letter_id: str, 
        content: LetterContent
    ) -> bool:
        """
        手紙の内容を設定する
        
        Args:
            letter_id: 手紙のID
            content: 手紙の内容
            
        Returns:
            設定が成功したかどうか
        """
        try:
            letter = await self.get_letter(letter_id)
            if not letter:
                self.logger.warning(f"対象の手紙が見つかりませんでした: {letter_id}")
                return False
            
            letter.set_content(content)
            
            # ストレージを更新
            await self._update_letter_in_storage(letter)
            
            self.logger.info(f"手紙の内容を設定しました: {letter_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"手紙内容の設定中にエラーが発生しました: {e}")
            return False
    
    async def delete_letter(self, letter_id: str) -> bool:
        """
        手紙を削除する
        
        Args:
            letter_id: 削除する手紙のID
            
        Returns:
            削除が成功したかどうか
        """
        try:
            result = await self.storage.delete_letter(letter_id)
            if result:
                self.logger.info(f"手紙を削除しました: {letter_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"手紙の削除中にエラーが発生しました: {e}")
            return False
    
    async def get_pending_letters(self) -> List[Letter]:
        """
        処理待ちの手紙一覧を取得する
        
        Returns:
            処理待ちの手紙リスト
        """
        try:
            all_letters = await self.storage.load_letters()
            pending_letters = []
            
            for letter_data in all_letters:
                letter = Letter(**letter_data)
                if letter.status == LetterStatus.PENDING:
                    pending_letters.append(letter)
            
            # 作成日時でソート（古い順）
            pending_letters.sort(key=lambda x: x.created_at)
            
            return pending_letters
            
        except Exception as e:
            self.logger.error(f"処理待ち手紙一覧の取得中にエラーが発生しました: {e}")
            return []
    
    async def _update_letter_in_storage(self, letter: Letter) -> None:
        """内部用：ストレージ内の手紙データを更新する"""
        # 既存データを削除
        await self.storage.delete_letter(letter.id)
        
        # 新しいデータを保存
        await self.storage.save_letter(letter.dict())

# グローバルマネージャーインスタンス
manager_instance = None

def get_letter_manager() -> LetterManager:
    """手紙マネージャーインスタンスを取得（シングルトン）"""
    global manager_instance
    
    if manager_instance is None:
        manager_instance = LetterManager()
    
    return manager_instance