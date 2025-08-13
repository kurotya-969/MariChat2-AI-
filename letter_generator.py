"""
Letter generator that combines Groq and Gemini APIs for high-quality letter generation.
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

from groq_client import GroqClient
from together_client import TogetherClient

logger = logging.getLogger(__name__)

class LetterGenerator:
    """Groq + Together AIの組み合わせによる高品質な手紙生成クラス"""
    
    def __init__(self):
        """GroqとTogether AIクライアントを初期化"""
        self.groq_client = GroqClient()
        self.together_client = TogetherClient()
        
    async def generate_letter(self, user_id: str, theme: str, user_history: Dict[str, Any]) -> Dict[str, Any]:
        """
        テーマとユーザー履歴を基に完成した手紙を生成
        
        Args:
            user_id: ユーザーID
            theme: 手紙のテーマ
            user_history: ユーザーの履歴情報
            
        Returns:
            生成された手紙の情報を含む辞書
            {
                'content': '手紙の内容',
                'metadata': {
                    'theme': 'テーマ',
                    'generated_at': '生成日時',
                    'groq_model': 'モデル名',
                    'together_model': 'モデル名',
                    'generation_time': 生成時間（秒）,
                    'user_id': 'ユーザーID'
                }
            }
            
        Raises:
            Exception: 手紙生成に失敗した場合
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"ユーザー {user_id} のテーマ '{theme}' で手紙生成開始")
            
            # ユーザーコンテキストを構築
            context = self._build_context(theme, user_history)
            
            # ステップ1: Groqで論理構造を生成
            logger.info("Groqで論理構造を生成中...")
            structure = await self.groq_client.generate_structure(theme, context)
            
            # ステップ2: Together AIで感情表現を補完
            logger.info("Together AIで感情表現を補完中...")
            enhanced_context = {**context, 'theme': theme}
            final_letter = await self.together_client.enhance_emotion(structure, enhanced_context)
            
            # 生成時間を計算
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # メタデータを構築
            metadata = {
                'theme': theme,
                'generated_at': datetime.now().isoformat(),
                'groq_model': self.groq_client.model,
                'together_model': self.together_client.model,
                'generation_time': generation_time,
                'user_id': user_id,
                'structure_length': len(structure),
                'final_length': len(final_letter)
            }
            
            logger.info(f"手紙生成完了 (所要時間: {generation_time:.2f}秒)")
            
            return {
                'content': final_letter,
                'metadata': metadata
            }
            
        except Exception as e:
            generation_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"手紙生成失敗 (所要時間: {generation_time:.2f}秒): {str(e)}")
            raise Exception(f"手紙生成に失敗しました: {str(e)}")
    
    def _build_context(self, theme: str, user_history: Dict[str, Any]) -> Dict[str, Any]:
        """
        ユーザー履歴を考慮したコンテキストを生成
        
        Args:
            theme: 手紙のテーマ
            user_history: ユーザーの履歴情報
            
        Returns:
            生成用のコンテキスト辞書
        """
        context = {
            'theme': theme,
            'user_history': user_history,
            'previous_letters': [],
            'interaction_count': 0
        }
        
        # 過去の手紙情報を抽出
        if 'letters' in user_history:
            letters = user_history['letters']
            previous_letters = []
            
            for date, letter_data in letters.items():
                if isinstance(letter_data, dict) and letter_data.get('status') == 'completed':
                    previous_letters.append({
                        'date': date,
                        'theme': letter_data.get('theme', ''),
                        'content_preview': letter_data.get('content', '')[:100] + '...' if letter_data.get('content') else ''
                    })
            
            # 日付順にソート（新しい順）
            previous_letters.sort(key=lambda x: x['date'], reverse=True)
            context['previous_letters'] = previous_letters[:5]  # 最新5通まで
        
        # ユーザープロファイル情報を抽出
        if 'profile' in user_history:
            profile = user_history['profile']
            context['interaction_count'] = profile.get('total_letters', 0)
        
        # 季節情報を追加
        current_month = datetime.now().month
        if current_month in [12, 1, 2]:
            context['season'] = '冬'
        elif current_month in [3, 4, 5]:
            context['season'] = '春'
        elif current_month in [6, 7, 8]:
            context['season'] = '夏'
        else:
            context['season'] = '秋'
        
        # 時間帯情報を追加
        current_hour = datetime.now().hour
        if 5 <= current_hour < 12:
            context['time_of_day'] = '朝'
        elif 12 <= current_hour < 17:
            context['time_of_day'] = '昼'
        elif 17 <= current_hour < 21:
            context['time_of_day'] = '夕方'
        else:
            context['time_of_day'] = '夜'
        
        return context
    
    async def test_generation_pipeline(self, test_theme: str = "テスト") -> Dict[str, Any]:
        """
        手紙生成パイプラインのテスト
        
        Args:
            test_theme: テスト用のテーマ
            
        Returns:
            テスト結果の辞書
        """
        try:
            # テスト用のユーザー履歴
            test_user_history = {
                'profile': {
                    'created_at': datetime.now().isoformat(),
                    'total_letters': 0
                },
                'letters': {},
                'requests': {}
            }
            
            # テスト生成を実行
            result = await self.generate_letter("test_user", test_theme, test_user_history)
            
            return {
                'success': True,
                'result': result,
                'message': 'テスト生成成功'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'テスト生成失敗'
            }
    
    async def check_api_connections(self) -> Dict[str, bool]:
        """
        両方のAPIクライアントの接続状態をチェック
        
        Returns:
            各APIの接続状態を示す辞書
        """
        try:
            groq_status = await self.groq_client.test_connection()
            together_status = await self.together_client.test_connection()
            
            return {
                'groq': groq_status,
                'together': together_status,
                'overall': groq_status and together_status
            }
            
        except Exception as e:
            logger.error(f"API接続チェック失敗: {str(e)}")
            return {
                'groq': False,
                'together': False,
                'overall': False,
                'error': str(e)
            }
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """
        生成統計情報を取得（将来の拡張用）
        
        Returns:
            統計情報の辞書
        """
        return {
            'groq_model': self.groq_client.model,
            'together_model': self.together_client.model,
            'max_retries': max(self.groq_client.max_retries, self.together_client.max_retries),
            'available': True
        }