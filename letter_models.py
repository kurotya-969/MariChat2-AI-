"""
手紙生成アプリのデータモデル定義
LetterRequest、GeneratedLetter、UserProfileのデータクラスと
バリデーション機能を提供します。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
import re
import json


@dataclass
class LetterRequest:
    """手紙リクエストのデータクラス"""
    user_id: str
    theme: str
    requested_at: datetime
    generation_hour: int  # 2, 3, 4のいずれか
    status: str = "pending"
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "user_id": self.user_id,
            "theme": self.theme,
            "requested_at": self.requested_at.isoformat(),
            "generation_hour": self.generation_hour,
            "status": self.status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LetterRequest':
        """辞書からインスタンスを作成"""
        return cls(
            user_id=data["user_id"],
            theme=data["theme"],
            requested_at=datetime.fromisoformat(data["requested_at"]),
            generation_hour=data["generation_hour"],
            status=data.get("status", "pending")
        )


@dataclass
class GeneratedLetter:
    """生成された手紙のデータクラス"""
    user_id: str
    theme: str
    content: str
    generated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "user_id": self.user_id,
            "theme": self.theme,
            "content": self.content,
            "generated_at": self.generated_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GeneratedLetter':
        """辞書からインスタンスを作成"""
        return cls(
            user_id=data["user_id"],
            theme=data["theme"],
            content=data["content"],
            generated_at=datetime.fromisoformat(data["generated_at"]),
            metadata=data.get("metadata", {})
        )


@dataclass
class UserProfile:
    """ユーザープロファイルのデータクラス"""
    user_id: str
    created_at: datetime
    last_request: Optional[str] = None
    total_letters: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_request": self.last_request,
            "total_letters": self.total_letters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """辞書からインスタンスを作成"""
        return cls(
            user_id=data["user_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_request=data.get("last_request"),
            total_letters=data.get("total_letters", 0)
        )


class ValidationError(Exception):
    """バリデーションエラー"""
    pass


class ThemeValidator:
    """テーマのバリデーション機能"""
    
    MIN_LENGTH = 1
    MAX_LENGTH = 100
    
    # 禁止されている文字パターン
    FORBIDDEN_PATTERNS = [
        r'<[^>]*>',  # HTMLタグ
        r'javascript:',  # JavaScript
        r'data:',  # データURL
    ]
    
    @classmethod
    def validate(cls, theme: str) -> bool:
        """テーマの妥当性を検証"""
        if not theme or not isinstance(theme, str):
            raise ValidationError("テーマは文字列である必要があります")
        
        # 長さチェック
        theme = theme.strip()
        if len(theme) < cls.MIN_LENGTH:
            raise ValidationError("テーマは1文字以上入力してください")
        
        if len(theme) > cls.MAX_LENGTH:
            raise ValidationError(f"テーマは{cls.MAX_LENGTH}文字以内で入力してください")
        
        # 禁止パターンチェック
        for pattern in cls.FORBIDDEN_PATTERNS:
            if re.search(pattern, theme, re.IGNORECASE):
                raise ValidationError("不正な文字が含まれています")
        
        return True
    
    @classmethod
    def sanitize(cls, theme: str) -> str:
        """テーマをサニタイズ"""
        if not theme:
            return ""
        
        # 前後の空白を削除
        theme = theme.strip()
        
        # 改行文字を空白に変換
        theme = re.sub(r'\s+', ' ', theme)
        
        return theme


class GenerationTimeValidator:
    """生成時刻のバリデーション機能"""
    
    VALID_HOURS = [2, 3, 4]  # 2時、3時、4時のみ有効
    
    @classmethod
    def validate(cls, hour: int) -> bool:
        """生成時刻の妥当性を検証"""
        if not isinstance(hour, int):
            raise ValidationError("生成時刻は整数である必要があります")
        
        if hour not in cls.VALID_HOURS:
            valid_hours_str = "、".join(map(str, cls.VALID_HOURS))
            raise ValidationError(f"生成時刻は{valid_hours_str}時のいずれかを選択してください")
        
        return True


class DataValidator:
    """データ全般のバリデーション機能"""
    
    @staticmethod
    def validate_user_id(user_id: str) -> bool:
        """ユーザーIDの妥当性を検証"""
        if not user_id or not isinstance(user_id, str):
            raise ValidationError("ユーザーIDは文字列である必要があります")
        
        # UUIDv4形式のチェック（簡易版）
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
        if not re.match(uuid_pattern, user_id, re.IGNORECASE):
            raise ValidationError("ユーザーIDの形式が正しくありません")
        
        return True
    
    @staticmethod
    def validate_letter_request(request: LetterRequest) -> bool:
        """手紙リクエストの妥当性を検証"""
        DataValidator.validate_user_id(request.user_id)
        ThemeValidator.validate(request.theme)
        GenerationTimeValidator.validate(request.generation_hour)
        
        # ステータスの検証
        valid_statuses = ["pending", "processing", "completed", "failed"]
        if request.status not in valid_statuses:
            raise ValidationError(f"ステータスは{valid_statuses}のいずれかである必要があります")
        
        return True
    
    @staticmethod
    def validate_generated_letter(letter: GeneratedLetter) -> bool:
        """生成された手紙の妥当性を検証"""
        DataValidator.validate_user_id(letter.user_id)
        ThemeValidator.validate(letter.theme)
        
        # 手紙内容の検証
        if not letter.content or not isinstance(letter.content, str):
            raise ValidationError("手紙の内容は文字列である必要があります")
        
        if len(letter.content.strip()) < 10:
            raise ValidationError("手紙の内容が短すぎます")
        
        return True
    
    @staticmethod
    def validate_user_profile(profile: UserProfile) -> bool:
        """ユーザープロファイルの妥当性を検証"""
        DataValidator.validate_user_id(profile.user_id)
        
        # 手紙数の検証
        if not isinstance(profile.total_letters, int) or profile.total_letters < 0:
            raise ValidationError("手紙数は0以上の整数である必要があります")
        
        return True


# テスト用のサンプルデータ作成関数
def create_sample_data():
    """テスト用のサンプルデータを作成"""
    import uuid
    
    user_id = str(uuid.uuid4())
    now = datetime.now()
    
    # サンプルリクエスト
    request = LetterRequest(
        user_id=user_id,
        theme="春の思い出",
        requested_at=now,
        generation_hour=2
    )
    
    # サンプル手紙
    letter = GeneratedLetter(
        user_id=user_id,
        theme="春の思い出",
        content="桜の花びらが舞い散る季節になりました。あなたとの思い出が蘇ります...",
        generated_at=now,
        metadata={
            "groq_model": "compound-beta",
            "Together_model": "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
            "generation_time": 12.5
        }
    )
    
    # サンプルプロファイル
    profile = UserProfile(
        user_id=user_id,
        created_at=now,
        last_request="2024-01-20",
        total_letters=1
    )
    
    return request, letter, profile


if __name__ == "__main__":
    # テスト実行
    try:
        request, letter, profile = create_sample_data()
        
        print("=== バリデーションテスト ===")
        
        # リクエストのバリデーション
        DataValidator.validate_letter_request(request)
        print("✓ LetterRequestのバリデーション成功")
        
        # 手紙のバリデーション
        DataValidator.validate_generated_letter(letter)
        print("✓ GeneratedLetterのバリデーション成功")
        
        # プロファイルのバリデーション
        DataValidator.validate_user_profile(profile)
        print("✓ UserProfileのバリデーション成功")
        
        print("\n=== シリアライゼーションテスト ===")
        
        # 辞書変換テスト
        request_dict = request.to_dict()
        request_restored = LetterRequest.from_dict(request_dict)
        print("✓ LetterRequestのシリアライゼーション成功")
        
        letter_dict = letter.to_dict()
        letter_restored = GeneratedLetter.from_dict(letter_dict)
        print("✓ GeneratedLetterのシリアライゼーション成功")
        
        profile_dict = profile.to_dict()
        profile_restored = UserProfile.from_dict(profile_dict)
        print("✓ UserProfileのシリアライゼーション成功")
        
        print("\n=== エラーケーステスト ===")
        
        # 不正なテーマのテスト
        try:
            ThemeValidator.validate("")
        except ValidationError as e:
            print(f"✓ 空のテーマエラー: {e}")
        
        # 不正な生成時刻のテスト
        try:
            GenerationTimeValidator.validate(5)
        except ValidationError as e:
            print(f"✓ 不正な生成時刻エラー: {e}")
        
        print("\n全てのテストが完了しました！")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")