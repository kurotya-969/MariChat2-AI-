"""
感情分析モジュール
ユーザーのメッセージから感情を分析し、好感度を更新する
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """感情分析を担当するクラス"""
    
    def __init__(self):
        self.analyzer = None
        self._initialize_analyzer()
    
    def _initialize_analyzer(self):
        """感情分析モデルの初期化"""
        try:
            # PyTorchのクラス登録問題を回避するため、環境変数でTransformers使用を制御
            import os
            use_transformers = os.getenv("USE_TRANSFORMERS", "false").lower() == "true"
            
            if not use_transformers:
                logger.info("Transformers使用が無効化されています。ルールベース分析を使用します。")
                self.analyzer = None
                return
            
            # transformersが利用可能な場合は使用
            import torch
            # PyTorchのクラス登録問題を回避するための設定
            torch.set_num_threads(1)  # スレッド数を制限
            
            from transformers import pipeline
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning)
            
            self.analyzer = pipeline(
                "sentiment-analysis", 
                model="koheiduck/bert-japanese-finetuned-sentiment",
                device=-1  # CPUを強制使用してGPU関連の問題を回避
            )
            logger.info("感情分析モデルのロード完了。")
        except Exception as e:
            logger.warning(f"感情分析モデルのロードに失敗、ルールベース分析を使用: {e}")
            self.analyzer = None
    
    def analyze_sentiment(self, message: str) -> Optional[str]:
        """メッセージの感情を分析する"""
        if not isinstance(message, str) or len(message.strip()) == 0:
            return None
        
        # transformersが利用可能な場合
        if self.analyzer:
            try:
                result = self.analyzer(message)[0]
                label = result.get('label', '').upper()
                # ラベルを統一形式に変換
                if 'POSITIVE' in label:
                    return 'positive'
                elif 'NEGATIVE' in label:
                    return 'negative'
                else:
                    return 'neutral'
            except Exception as e:
                logger.error(f"感情分析エラー: {e}")
        
        # フォールバック：ルールベース感情分析
        return self._rule_based_sentiment(message)
    
    def _rule_based_sentiment(self, message: str) -> str:
        """ルールベースの感情分析"""
        positive_words = [
            'ありがとう', 'うれしい', '嬉しい', '楽しい', '好き', '愛してる',
            '素晴らしい', 'いい', '良い', 'すごい', '最高', '幸せ', '感謝',
            'かわいい', '可愛い', '美しい', '優しい', '親切', '素敵'
        ]
        
        negative_words = [
            '嫌い', '悲しい', 'つらい', '辛い', '苦しい', '痛い', '怒り',
            'むかつく', 'うざい', 'きらい', '最悪', 'だめ', 'ダメ',
            '死ね', 'バカ', 'ばか', 'アホ', 'あほ', 'クソ', 'くそ'
        ]
        
        message_lower = message.lower()
        
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def update_affection(self, message: str, current_affection: int, 
                        conversation_context: list = None) -> tuple:
        """
        メッセージに基づいて好感度を更新する
        
        Args:
            message: ユーザーのメッセージ
            current_affection: 現在の好感度
            conversation_context: 会話の文脈（最近のメッセージ）
            
        Returns:
            (新しい好感度, 変化量, 変化理由)
        """
        if not isinstance(current_affection, (int, float)):
            current_affection = 30  # デフォルト値
        
        sentiment = self.analyze_sentiment(message)
        if not sentiment:
            return current_affection, 0, "感情分析失敗"
        
        # 基本的な感情に基づく変化量
        base_change = 0
        if sentiment == 'positive':
            base_change = 3
        elif sentiment == 'negative':
            base_change = -3
        else:  # neutral
            base_change = 0
        
        # メッセージの特徴による調整
        change_modifiers = []
        
        # メッセージの長さによる調整
        if len(message) > 100:
            base_change = int(base_change * 1.3)
            change_modifiers.append("長文")
        elif len(message) > 50:
            base_change = int(base_change * 1.1)
            change_modifiers.append("中文")
        
        # 特定のキーワードによる追加調整
        positive_keywords = ['ありがとう', '感謝', '好き', '愛してる', '素晴らしい', 'かわいい', '美しい']
        negative_keywords = ['嫌い', '死ね', 'バカ', 'アホ', 'クソ', 'うざい', 'きらい']
        
        message_lower = message.lower()
        
        # ポジティブキーワードのチェック
        positive_count = sum(1 for word in positive_keywords if word in message_lower)
        if positive_count > 0:
            base_change += positive_count * 2
            change_modifiers.append(f"ポジティブ語({positive_count})")
        
        # ネガティブキーワードのチェック
        negative_count = sum(1 for word in negative_keywords if word in message_lower)
        if negative_count > 0:
            base_change -= negative_count * 3
            change_modifiers.append(f"ネガティブ語({negative_count})")
        
        # 現在の好感度レベルによる調整
        if current_affection < 20:  # 敵対状態
            if base_change > 0:
                base_change = int(base_change * 0.5)  # ポジティブな変化を抑制
                change_modifiers.append("敵対状態")
        elif current_affection > 80:  # 親密状態
            if base_change < 0:
                base_change = int(base_change * 0.7)  # ネガティブな変化を抑制
                change_modifiers.append("親密状態")
        
        # 会話の文脈による調整
        if conversation_context and len(conversation_context) > 0:
            recent_messages = conversation_context[-3:]  # 最近の3メッセージ
            context_sentiment_count = {'positive': 0, 'negative': 0, 'neutral': 0}
            
            for ctx_msg in recent_messages:
                if isinstance(ctx_msg, dict) and 'content' in ctx_msg:
                    ctx_sentiment = self.analyze_sentiment(ctx_msg['content'])
                    if ctx_sentiment:
                        context_sentiment_count[ctx_sentiment] += 1
            
            # 連続したポジティブ/ネガティブメッセージの場合は効果を減衰
            if sentiment == 'positive' and context_sentiment_count['positive'] >= 2:
                base_change = int(base_change * 0.8)
                change_modifiers.append("連続ポジティブ")
            elif sentiment == 'negative' and context_sentiment_count['negative'] >= 2:
                base_change = int(base_change * 0.8)
                change_modifiers.append("連続ネガティブ")
        
        # 最終的な好感度を計算
        new_affection = current_affection + base_change
        new_affection = max(0, min(100, new_affection))  # 0-100の範囲に制限
        
        # 変化理由を生成
        if base_change == 0:
            reason = "中立的なメッセージ"
        else:
            reason = f"{sentiment}({base_change:+d})"
            if change_modifiers:
                reason += f" [{', '.join(change_modifiers)}]"
        
        return new_affection, base_change, reason
    
    def get_relationship_stage(self, affection: int) -> str:
        """好感度から関係性のステージを取得する"""
        if not isinstance(affection, (int, float)):
            affection = 30  # デフォルト値
        
        if affection < 20:
            return "敵対"
        elif affection < 40:
            return "中立"
        elif affection < 60:
            return "好意"
        elif affection < 80:
            return "親密"
        else:
            return "最接近"