"""
メモリ管理モジュール
会話履歴から重要単語を抽出し、トークン使用量を最適化する
"""
import logging
import re
from typing import List, Dict, Tuple, Any
from collections import Counter
import json

logger = logging.getLogger(__name__)

class MemoryManager:
    """会話履歴のメモリ管理を行うクラス"""
    
    def __init__(self, history_threshold: int = 10):
        """
        Args:
            history_threshold: 履歴圧縮を実行する会話数の閾値
        """
        self.history_threshold = history_threshold
        self.important_words_cache = []
        self.special_memories = {}  # 手紙などの特別な記憶を保存
        
    def extract_important_words(self, messages: List[Dict[str, str]], 
                              dialogue_generator=None) -> List[str]:
        """
        会話履歴から重要単語を抽出する（ルールベースのみ）
        
        Args:
            messages: チャットメッセージのリスト
            dialogue_generator: 対話生成器（使用しない）
            
        Returns:
            重要単語のリスト
        """
        try:
            # メッセージからテキストを結合
            text_content = []
            for msg in messages:
                if msg.get("content"):
                    text_content.append(msg["content"])
            
            combined_text = " ".join(text_content)
            
            # ルールベースの抽出のみ使用
            return self._extract_with_rules(combined_text)
            
        except Exception as e:
            logger.error(f"重要単語抽出エラー: {e}")
            return self._extract_with_rules(" ".join([msg.get("content", "") for msg in messages]))
    

    
    def _extract_with_rules(self, text: str) -> List[str]:
        """
        ルールベースで重要単語を抽出する（強化版）
        
        Args:
            text: 抽出対象のテキスト
            
        Returns:
            重要単語のリスト
        """
        try:
            # 基本的なクリーニング
            text = re.sub(r'[^\w\s]', ' ', text)
            words = text.split()
            
            # ストップワードを除外
            stop_words = {
                'の', 'に', 'は', 'を', 'が', 'で', 'と', 'から', 'まで', 'より',
                'だ', 'である', 'です', 'ます', 'した', 'する', 'される',
                'これ', 'それ', 'あれ', 'この', 'その', 'あの',
                'ここ', 'そこ', 'あそこ', 'どこ', 'いつ', 'なに', 'なぜ',
                'ちょっと', 'とても', 'すごく', 'かなり', 'もう', 'まだ',
                'でも', 'しかし', 'だから', 'そして', 'また', 'さらに',
                'あたし', 'お前', 'ユーザー', 'システム', 'アプリ'
            }
            
            # 重要カテゴリのキーワード
            important_categories = {
                'food': ['コーヒー', 'お茶', '紅茶', 'ケーキ', 'パン', '料理', '食べ物', '飲み物'],
                'hobby': ['読書', '映画', '音楽', 'ゲーム', 'スポーツ', '散歩', '旅行'],
                'emotion': ['嬉しい', '悲しい', '楽しい', '怒り', '不安', '安心', '幸せ'],
                'place': ['家', '学校', '会社', '公園', 'カフェ', '図書館', '駅', '街'],
                'time': ['朝', '昼', '夜', '今日', '明日', '昨日', '週末', '平日'],
                'color': ['赤', '青', '緑', '黄色', '白', '黒', 'ピンク', '紫'],
                'weather': ['晴れ', '雨', '曇り', '雪', '暑い', '寒い', '暖かい', '涼しい']
            }
            
            # 重要そうなパターンを優先
            important_patterns = [
                r'[A-Za-z]{3,}',  # 英単語（3文字以上）
                r'[ァ-ヶー]{2,}',  # カタカナ（2文字以上）
                r'[一-龯]{2,}',   # 漢字（2文字以上）
            ]
            
            important_words = []
            
            # パターンマッチング
            for pattern in important_patterns:
                matches = re.findall(pattern, text)
                important_words.extend(matches)
            
            # カテゴリ別重要語句の検出
            for category, keywords in important_categories.items():
                for keyword in keywords:
                    if keyword in text:
                        important_words.append(keyword)
            
            # 頻度でフィルタリング
            word_counts = Counter(important_words)
            filtered_words = []
            
            for word, count in word_counts.items():
                if (len(word) >= 2 and 
                    word not in stop_words and 
                    not word.isdigit() and  # 数字のみは除外
                    count >= 1):  # 最低1回は出現
                    filtered_words.append(word)
            
            # 重要度でソート（頻度 + カテゴリ重要度）
            def get_importance_score(word):
                base_score = word_counts[word]
                # カテゴリに含まれる語句は重要度アップ
                for keywords in important_categories.values():
                    if word in keywords:
                        base_score += 2
                # 長い語句は重要度アップ
                if len(word) >= 4:
                    base_score += 1
                return base_score
            
            # 重要度順でソートして上位15個を返す
            sorted_words = sorted(filtered_words, key=get_importance_score, reverse=True)
            return sorted_words[:15]
            
        except Exception as e:
            logger.error(f"ルールベース抽出エラー: {e}")
            return []
    
    def should_compress_history(self, messages: List[Dict[str, str]]) -> bool:
        """
        履歴を圧縮すべきかどうかを判定する
        
        Args:
            messages: チャットメッセージのリスト
            
        Returns:
            圧縮が必要かどうか
        """
        # ユーザーとアシスタントのペア数をカウント
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        return len(user_messages) >= self.history_threshold
    
    def compress_history(self, messages: List[Dict[str, str]], 
                        dialogue_generator=None) -> Tuple[List[Dict[str, str]], List[str]]:
        """
        履歴を圧縮し、重要単語を抽出する
        
        Args:
            messages: チャットメッセージのリスト
            dialogue_generator: 対話生成器
            
        Returns:
            (圧縮後のメッセージリスト, 抽出された重要単語のリスト)
        """
        try:
            if not self.should_compress_history(messages):
                return messages, self.important_words_cache
            
            # 最新の数ターンを保持
            keep_recent = 4  # 最新4ターン（ユーザー2回、アシスタント2回）を保持
            
            # 古い履歴から重要単語を抽出
            old_messages = messages[:-keep_recent] if len(messages) > keep_recent else []
            recent_messages = messages[-keep_recent:] if len(messages) > keep_recent else messages
            
            if old_messages:
                # 重要単語を抽出
                new_keywords = self.extract_important_words(old_messages, dialogue_generator)
                
                # 既存のキーワードと統合（重複除去）
                all_keywords = list(set(self.important_words_cache + new_keywords))
                self.important_words_cache = all_keywords[:20]  # 最大20個のキーワードを保持
                
                logger.info(f"履歴を圧縮しました。抽出されたキーワード: {new_keywords}")
            
            return recent_messages, self.important_words_cache
            
        except Exception as e:
            logger.error(f"履歴圧縮エラー: {e}")
            return messages, self.important_words_cache
    
    def get_memory_summary(self) -> str:
        """
        保存されている重要単語から記憶の要約を生成する
        
        Returns:
            記憶の要約文字列
        """
        summary_parts = []
        
        # 通常の重要単語
        if self.important_words_cache:
            keywords_text = "、".join(self.important_words_cache)
            summary_parts.append(f"過去の会話で言及された重要な要素: {keywords_text}")
        
        # 特別な記憶（手紙など）
        if self.special_memories:
            for memory_type, memories in self.special_memories.items():
                if memories:
                    latest_memory = memories[-1]["content"]
                    if memory_type == "letter_content":
                        summary_parts.append(f"最近の手紙の記憶: {latest_memory}")
                    else:
                        summary_parts.append(f"{memory_type}: {latest_memory}")
        
        return "\n".join(summary_parts) if summary_parts else ""
    
    def add_important_memory(self, memory_type: str, content: str) -> str:
        """
        重要な記憶を追加する（手紙の内容など）
        
        Args:
            memory_type: 記憶の種類（例: "letter_content"）
            content: 記憶する内容
            
        Returns:
            ユーザーに表示する通知メッセージ
        """
        if memory_type not in self.special_memories:
            self.special_memories[memory_type] = []
        
        self.special_memories[memory_type].append({
            "content": content,
            "timestamp": logging.Formatter().formatTime(logging.LogRecord("", 0, "", 0, "", (), None))
        })
        
        # 最大5件まで保持
        if len(self.special_memories[memory_type]) > 5:
            self.special_memories[memory_type] = self.special_memories[memory_type][-5:]
        
        logger.info(f"特別な記憶を追加しました: {memory_type}")
        
        # 記憶の種類に応じた通知メッセージを生成
        if memory_type == "letter_content":
            return "🧠✨ 麻理の記憶に新しい手紙の内容が刻まれました。今後の会話でこの記憶を参照することがあります。"
        else:
            return f"🧠✨ 麻理の記憶に新しい{memory_type}が追加されました。"
    
    def get_special_memories(self, memory_type: str = None) -> Dict[str, Any]:
        """
        特別な記憶を取得する
        
        Args:
            memory_type: 取得する記憶の種類（Noneの場合は全て）
            
        Returns:
            記憶の辞書
        """
        if memory_type:
            return self.special_memories.get(memory_type, [])
        return self.special_memories
    
    def clear_memory(self):
        """メモリをクリアする"""
        self.important_words_cache = []
        self.special_memories = {}
        logger.info("メモリをクリアしました")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        メモリの統計情報を取得する
        
        Returns:
            統計情報の辞書
        """
        return {
            "cached_keywords_count": len(self.important_words_cache),
            "cached_keywords": self.important_words_cache,
            "special_memories_count": sum(len(memories) for memories in self.special_memories.values()),
            "special_memories": self.special_memories,
            "history_threshold": self.history_threshold
        }