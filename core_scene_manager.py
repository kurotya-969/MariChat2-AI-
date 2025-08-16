"""
シーン管理モジュール
背景テーマの管理とシーン変更の検出（Groq API使用）
"""
import json
import logging
import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from groq import Groq

logger = logging.getLogger(__name__)

class SceneManager:
    """シーン管理を担当するクラス（Groq API使用）"""
    
    def __init__(self):
        self.theme_urls = {
            "default": "static/ribinngu-hiru.jpg",
            "room_night": "static/ribinngu-yoru-on.jpg",
            "beach_sunset": "static/sunahama-hiru.jpg",
            "festival_night": "static/maturi-yoru.jpg",
            "shrine_day": "static/jinnjya-hiru.jpg",
            "cafe_afternoon": "static/kissa-hiru.jpg",
            "art_museum_night": "static/bijyutukann-yoru.jpg"
        }
        self.groq_client = self._initialize_groq_client()
    
    def _initialize_groq_client(self):
        """Groq APIクライアントの初期化"""
        try:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                logger.warning("環境変数 GROQ_API_KEY が設定されていません。シーン検出機能が制限されます。")
                return None
            
            client = Groq(api_key=api_key)
            logger.info("Groq APIクライアントの初期化が完了しました。")
            return client
        except Exception as e:
            logger.error(f"Groq APIクライアントの初期化に失敗しました: {e}")
            return None
    
    def get_theme_url(self, theme: str) -> str:
        """テーマに対応するURLを取得する"""
        return self.theme_urls.get(theme, self.theme_urls["default"])
    
    def get_available_themes(self) -> List[str]:
        """利用可能なテーマのリストを取得する"""
        return list(self.theme_urls.keys())
    
    def detect_scene_change(self, history: List[Tuple[str, str]], 
                           dialogue_generator=None, current_theme: str = "default") -> Optional[str]:
        """
        会話履歴からシーン変更を検出する（Groq API使用）
        
        Args:
            history: 会話履歴のリスト
            dialogue_generator: 対話生成器（使用しない）
            current_theme: 現在のテーマ
            
        Returns:
            新しいシーン名（変更がない場合はNone）
        """
        if not history:
            logger.info("履歴が空のためシーン検出をスキップ")
            return None
            
        if not self.groq_client:
            logger.warning("Groq APIクライアントが初期化されていません")
            return None
        
        # 最新5件の会話履歴を使用（より多くの文脈を提供）
        recent_history = history[-5:] if len(history) > 5 else history
        history_text = "\n".join([
            f"ユーザー: {u}\n麻理: {m}" for u, m in recent_history
        ])
        
        # まずGroq APIでシーン検出を試行
        logger.info("Groq APIを使用してシーン検出を実行します")
        
        if self.groq_client:
            # Groq APIが利用可能な場合は優先的に使用
            result = self._detect_scene_with_groq(history_text, current_theme)
            if result is not None:
                return result
            logger.info("Groq APIでシーン変更が検出されませんでした")
        else:
            logger.warning("Groq APIクライアントが利用できません")
        
        # Groq APIが失敗またはシーン変更なしの場合、フォールバックとしてキーワード検出
        logger.info("フォールバック: キーワードベースのシーン検出を実行")
        return self._fallback_keyword_detection(history_text, current_theme)
    
    def _has_location_keywords(self, text: str) -> bool:
        """
        テキストに場所関連のキーワードが含まれているかチェック
        
        Args:
            text: チェック対象のテキスト
            
        Returns:
            場所関連キーワードが含まれているかどうか
        """
        location_keywords = [
            # 場所名
            "ビーチ", "海", "砂浜", "海岸", "海辺", "浜辺", "海沿い",
            "神社", "お寺", "寺院", "鳥居", "境内", "参道",
            "カフェ", "喫茶店", "店", "レストラン", "コーヒーショップ",
            "祭り", "花火", "屋台", "縁日", "フェスティバル",
            "部屋", "家", "室内", "寝室", "リビング", "自宅",
            "美術館", "アート", "ギャラリー", "絵画", "彫刻",
            # 移動動詞・状態
            "行く", "行こう", "向かう", "着いた", "到着", "移動", "出かける", "来た", "いる", "にいる", "来ている",
            # 場所の特徴
            "夕日", "夕焼け", "サンセット", "波", "潮風", "海風",
            "お参り", "参拝", "祈り", "おみくじ", "お守り",
            "コーヒー", "お茶", "ラテ", "エスプレッソ", "カフェオレ",
            "浴衣", "夜店", "お祭り", "フェスティバル", "花火大会",
            "ベッド", "夜", "屋内", "家の中", "寝室",
            "アート作品", "絵画", "彫刻", "美術品", "芸術作品",
            # 時間帯
            "夜", "夕方", "朝", "昼間", "午後", "深夜",
            # 天候・雰囲気
            "夕暮れ", "夜明け", "静寂", "賑やか", "幻想的"
        ]
        
        for keyword in location_keywords:
            if keyword in text:
                logger.info(f"場所関連キーワードを検出: {keyword}")
                return True
        
        return False
    
    def _fallback_keyword_detection(self, history_text: str, current_theme: str) -> Optional[str]:
        """
        フォールバック: キーワードベースのシーン検出
        
        Args:
            history_text: 会話履歴のテキスト
            current_theme: 現在のテーマ
            
        Returns:
            新しいシーン名（変更がない場合はNone）
        """
        logger.info("フォールバック: キーワードベースのシーン検出を実行")
        
        # キーワードとシーンのマッピング（拡張版）
        keyword_scene_map = {
            # 美術館関連
            "美術館": "art_museum_night",
            "アート": "art_museum_night", 
            "ギャラリー": "art_museum_night",
            "絵画": "art_museum_night",
            "彫刻": "art_museum_night",
            "芸術": "art_museum_night",
            "展示": "art_museum_night",
            "作品": "art_museum_night",
            
            # カフェ関連
            "カフェ": "cafe_afternoon",
            "喫茶店": "cafe_afternoon",
            "コーヒー": "cafe_afternoon",
            "お茶": "cafe_afternoon",
            "ラテ": "cafe_afternoon",
            "店": "cafe_afternoon",
            
            # 神社関連
            "神社": "shrine_day",
            "お寺": "shrine_day",
            "寺院": "shrine_day",
            "参拝": "shrine_day",
            "お参り": "shrine_day",
            "鳥居": "shrine_day",
            "境内": "shrine_day",
            
            # 海・ビーチ関連
            "海": "beach_sunset",
            "ビーチ": "beach_sunset",
            "砂浜": "beach_sunset",
            "夕日": "beach_sunset",
            "夕焼け": "beach_sunset",
            "海岸": "beach_sunset",
            "波": "beach_sunset",
            
            # 祭り関連
            "祭り": "festival_night",
            "花火": "festival_night",
            "屋台": "festival_night",
            "縁日": "festival_night",
            "お祭り": "festival_night",
            "花火大会": "festival_night",
            
            # 夜・部屋関連
            "夜": "room_night",
            "寝室": "room_night",
            "ベッド": "room_night",
            "部屋": "room_night",
            "家": "room_night",
            "室内": "room_night",
            "深夜": "room_night",
            "夜中": "room_night"
        }
        
        # 優先度順でキーワードをチェック（より具体的なキーワードを優先）
        for keyword, scene in keyword_scene_map.items():
            if keyword in history_text and scene != current_theme:
                logger.info(f"フォールバック検出: キーワード '{keyword}' → シーン '{scene}'")
                return scene
        
        logger.info("フォールバック検出: シーン変更なし")
        return None
    
    def _detect_scene_with_groq(self, history_text: str, current_theme: str) -> Optional[str]:
        """
        Groq APIを使用してシーン変更を検出する
        
        Args:
            history_text: 会話履歴のテキスト
            current_theme: 現在のテーマ
            
        Returns:
            新しいシーン名（変更がない場合はNone）
        """
        try:
            # デバッグログ
            logger.info(f"シーン検出開始 - 現在のテーマ: {current_theme}")
            logger.info(f"会話履歴: {history_text}")
            
            # 利用可能なシーンのリスト
            available_scenes = list(self.theme_urls.keys())
            scenes_description = {
                "default": "デフォルトの部屋",
                "room_night": "夜の部屋・寝室",
                "beach_sunset": "夕日のビーチ・海岸",
                "festival_night": "夜祭り・花火大会",
                "shrine_day": "昼間の神社・寺院",
                "cafe_afternoon": "午後のカフェ・喫茶店",
                "art_museum_night": "夜の美術館"
            }
            
            # より積極的なシーン検出のためのプロンプト
            system_prompt = """あなたは会話の内容から、キャラクターとユーザーの現在位置（シーン）を判定する専門システムです。

会話履歴を分析し、場所の移動や新しい場所への言及があったかを判断してください。

判定基準（積極的に検出）:
1. 場所の名前が明確に言及されている → シーン変更の可能性あり
2. 「〜に行く」「〜に向かう」「〜に着いた」「〜にいる」「〜に来た」 → シーン変更
3. 場所に関連する活動や物の言及 → シーン変更の可能性あり
4. 現在のシーンと異なる場所の特徴的な要素の言及 → シーン変更
5. 時間帯の変化（夜、夕方、朝など）→ シーン変更の可能性あり

利用可能なシーン:
- default: デフォルトの部屋（室内）
- room_night: 夜の部屋・寝室
- beach_sunset: 夕日のビーチ・海岸
- festival_night: 夜祭り・花火大会
- shrine_day: 昼間の神社・寺院
- cafe_afternoon: 午後のカフェ・喫茶店
- art_museum_night: 夜の美術館

出力形式: 必ずJSON形式で回答してください
{"scene": "シーン名", "confidence": "high/medium/low", "reason": "判定理由"} または {"scene": "none", "confidence": "high", "reason": "判定理由"}

重要: JSON以外の文字は一切出力しないでください。"""
            
            user_prompt = f"""現在のシーン: {current_theme} ({scenes_description.get(current_theme, current_theme)})

利用可能なシーン:
{chr(10).join([f"- {scene}: {desc}" for scene, desc in scenes_description.items()])}

会話履歴:
{history_text}

この会話で場所の移動や新しい場所への言及があった場合は、最も適切なシーン名を返してください。
判定の理由も含めて回答してください。"""
            
            # Groq APIを呼び出し
            response = self.groq_client.chat.completions.create(
                model="compound-beta",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,  # 少し創造性を上げる
                max_tokens=150,   # トークン数を増やす
                response_format={"type": "json_object"}
            )
            
            if not response.choices or not response.choices[0].message.content:
                logger.warning("Groq APIからの応答が空です")
                return None
            
            # デバッグ: API応答をログ出力
            api_response = response.choices[0].message.content
            logger.info(f"Groq API応答: {api_response}")
            
            # JSONをパース
            result = json.loads(api_response)
            scene_value = result.get("scene", "none")
            confidence = result.get("confidence", "unknown")
            reason = result.get("reason", "理由不明")
            
            logger.info(f"シーン検出結果: {scene_value}, 信頼度: {confidence}, 理由: {reason}")
            
            # 結果を検証
            if (isinstance(scene_value, str) and 
                scene_value != "none" and 
                scene_value in available_scenes and 
                scene_value != current_theme):
                logger.info(f"Groqでシーン変更を検出: {current_theme} → {scene_value} (理由: {reason})")
                return scene_value
            
            logger.info(f"シーン変更なし: {reason}")
            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"Groq APIのJSON応答パースエラー: {e}")
            logger.error(f"応答内容: {response.choices[0].message.content if response.choices else 'None'}")
            return None
        except Exception as e:
            logger.error(f"Groq APIシーン検出エラー: {e}")
            return None
    
    def create_scene_params(self, theme: str = "default") -> Dict[str, Any]:
        """シーンパラメータを作成する"""
        return {"theme": theme}
    
    def update_scene_params(self, scene_params: Dict[str, Any], 
                           new_theme: str) -> Dict[str, Any]:
        """シーンパラメータを更新する"""
        if not isinstance(scene_params, dict):
            scene_params = self.create_scene_params()
        
        updated_params = scene_params.copy()
        updated_params["theme"] = new_theme
        updated_params["last_updated"] = json.dumps(datetime.now().isoformat())
        return updated_params
    
    def should_update_background(self, scene_params: Dict[str, Any], 
                                current_display_theme: str) -> bool:
        """
        背景を更新すべきかどうかを判定する
        
        Args:
            scene_params: 現在のシーンパラメータ
            current_display_theme: 現在表示されているテーマ
            
        Returns:
            背景更新が必要かどうか
        """
        if not isinstance(scene_params, dict):
            return True
        
        stored_theme = scene_params.get("theme", "default")
        return stored_theme != current_display_theme
    
    def get_scene_transition_message(self, old_theme: str, new_theme: str) -> str:
        """
        シーン変更時のメッセージを生成する
        
        Args:
            old_theme: 変更前のテーマ
            new_theme: 変更後のテーマ
            
        Returns:
            シーン変更メッセージ
        """
        theme_names = {
            "default": "デフォルトの部屋",
            "room_night": "夜の部屋",
            "beach_sunset": "夕日のビーチ",
            "festival_night": "夜祭り",
            "shrine_day": "昼間の神社",
            "cafe_afternoon": "午後のカフェ",
            "art_museum_night": "夜の美術館"
        }
        
        old_name = theme_names.get(old_theme, old_theme)
        new_name = theme_names.get(new_theme, new_theme)
        
        return f"シーンが「{old_name}」から「{new_name}」に変更されました"
    
    def test_scene_detection(self, test_message: str, current_theme: str = "default") -> Optional[str]:
        """
        シーン検出のテスト用メソッド
        
        Args:
            test_message: テスト用メッセージ
            current_theme: 現在のテーマ
            
        Returns:
            検出されたシーン名
        """
        # テスト用の履歴を作成
        test_history = [("ユーザー", test_message), ("麻理", "了解")]
        history_text = f"ユーザー: {test_message}\n麻理: 了解"
        
        logger.info(f"シーン検出テスト - メッセージ: {test_message}")
        
        if not self._has_location_keywords(history_text):
            logger.info("場所関連キーワードなし")
            return None
        
        return self._detect_scene_with_groq(history_text, current_theme)
    
    def get_debug_info(self) -> Dict[str, Any]:
        """
        デバッグ情報を取得
        
        Returns:
            デバッグ情報の辞書
        """
        return {
            "groq_client_initialized": self.groq_client is not None,
            "available_themes": list(self.theme_urls.keys()),
            "theme_count": len(self.theme_urls),
            "groq_api_key_set": bool(os.getenv("GROQ_API_KEY")),
            "current_theme_urls": {theme: self.get_theme_url(theme) for theme in self.get_available_themes()}
        }