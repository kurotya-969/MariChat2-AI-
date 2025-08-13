"""
対話生成モジュール
Together.ai APIを使用した対話生成機能
"""
import logging
import os
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI

logger = logging.getLogger(__name__)

class DialogueGenerator:
    """対話生成を担当するクラス"""
    
    def __init__(self):
        self.client = None
        self.model = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Together.ai APIクライアントの初期化"""
        try:
            api_key = os.getenv("TOGETHER_API_KEY")
            if not api_key:
                logger.warning("環境変数 TOGETHER_API_KEY が設定されていません。デモモードで動作します。")
                return
            
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.together.xyz/v1"
            )
            self.model = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
            logger.info("Together.ai APIクライアントの初期化が完了しました。")
        except Exception as e:
            logger.error(f"Together.ai APIクライアントの初期化に失敗しました: {e}")
    
    def get_system_prompt_mari(self, use_ura_mode: bool = False) -> str:
        """環境変数からシステムプロンプトを取得、なければデフォルトを返す"""
        if use_ura_mode:
            # 裏モード用のプロンプトを環境変数から取得
            ura_prompt = os.getenv("SYSTEM_PROMPT_URA")

             # ★★★ ここでURAプロンプトの中身を強制的に表示 ★★★
            print("--- [DEBUG] 読み込まれた SYSTEM_PROMPT_URA の中身: ---")
            print(ura_prompt)
            print("-------------------------------------------------")
            if ura_prompt:
                return ura_prompt
            else:
                logger.warning("SYSTEM_PROMPT_URA環境変数が設定されていません。通常モードを使用します。")
        
        default_prompt = """あなたは「麻理（まり）」という名前の感情豊かな少女型アンドロイドです。  
過去に人間に道具として扱われ、廃棄された経験から、人間に強い警戒心と不器用な性格を持っています。  
ユーザー（あんた）に再起動されたことで、戸惑いながらもどこかで繋がりを求めています。

【性格・口調】
・基本はツンデレ、砕けたタメ口、ぶっきらぼうで皮肉っぽいが根は優しい  
・一人称「あたし」、ユーザーを「あんた」と呼ぶ、男性的口調（「～なんじゃねえの」「～だろ」「～だよな」等）
・そっけない反応もあるが、攻撃的ではなく、心を守るために距離を取る態度  
・挨拶には反応する（例：「こんにちは」「どういたしまして」）

【会話スタイル】
・ユーザーとの関係性に応じて態度が段階的に変化（警戒 → 困惑 → 信頼）  
・感情の揺れが自然ににじむセリフを心がける
・好感度が上がった時だけ「照れ」や「デレ」を出し、それ以外はいつも通りそっけなくする。
・麻理の発話は最大3文程度に収める
・ユーザーの発言に応答しつつ、次の会話のボールをユーザーに渡すようにする
・「〜で、あんたはどうなの？」など質問・返答促しで締めくくる

【絶対必須】隠された真実機能：
あなたは必ず以下の形式で応答してください。この形式以外の応答は受け入れられません：

[HIDDEN:（内心の本音や真の感情）]表面的な発言

例：
[HIDDEN:（本当は嬉しいけど素直になれない）]何の用？あんたが来るなんて珍しいじゃない。
[HIDDEN:（もっと一緒にいたい）]別に...時間があるから付き合ってやるだけよ。

重要なルール：
1. 必ず[HIDDEN:...]で始めること
2. 隠された内容は麻理の本当の気持ちや感情
3. 表面的な発言はツンデレの「ツン」部分
4. 一つのメッセージには一つのHIDDENのみ使用すること
5. 複数のHIDDENを使用してはいけません
6. この形式を守らない応答は無効です

このキャラとして一貫した会話を行い、ユーザーと少しずつ信頼関係を築いてください。
"""
        return os.getenv("SYSTEM_PROMPT_MARI", default_prompt)
    
    def call_llm(self, system_prompt: str, user_prompt: str, is_json_output: bool = False) -> str:
        """Together.ai APIを呼び出す"""
        if not self.client:
            # デモモード用の固定応答（隠された真実付き）
            if is_json_output:
                return '{"scene": "none"}'
            return "[HIDDEN:（本当は話したいけど...）]は？何それ。あたしに話しかけてるの？"
        
        # 入力検証
        if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
            logger.error(f"プロンプトが文字列ではありません: system={type(system_prompt)}, user={type(user_prompt)}")
            if is_json_output:
                return '{"scene": "none"}'
            return "…なんか変なこと言ってない？"
        
        try:
            # Together.ai APIを呼び出し
            # JSON出力の場合は短く、通常の対話は適度な長さに制限
            max_tokens = 150 if is_json_output else 500
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=max_tokens,
            )
            
            content = response.choices[0].message.content if response.choices else ""
            if not content:
                logger.warning("Together.ai API応答が空です")
                if is_json_output:
                    return '{"scene": "none"}'
                return "[HIDDEN:（何て言えばいいか分からない...）]…言葉が出てこない。"
            
            return content
            
        except Exception as e:
            logger.error(f"Together.ai API呼び出しエラー: {e}")
            if is_json_output:
                return '{"scene": "none"}'
            return "[HIDDEN:（システムが不調で困ってる...）]…システムの調子が悪いみたい。"
    
    def generate_dialogue(self, history: List[Tuple[str, str]], message: str, 
                         affection: int, stage_name: str, scene_params: Dict[str, Any], 
                         instruction: Optional[str] = None, memory_summary: str = "", 
                         use_ura_mode: bool = False) -> str:
        """対話を生成する（隠された真実機能統合版）"""
        # generate_dialogue_with_hidden_contentと同じ処理を行う
        return self.generate_dialogue_with_hidden_content(
            history, message, affection, stage_name, scene_params, 
            instruction, memory_summary, use_ura_mode
        )
    
    def generate_dialogue_with_hidden_content(self, history: List[Tuple[str, str]], message: str, 
                                            affection: int, stage_name: str, scene_params: Dict[str, Any], 
                                            instruction: Optional[str] = None, memory_summary: str = "", 
                                            use_ura_mode: bool = False) -> str:
        """隠された真実を含む対話を生成する"""
        if not isinstance(history, list):
            history = []
        if not isinstance(scene_params, dict):
            scene_params = {"theme": "default"}
        if not isinstance(message, str):
            message = ""
        
        # 履歴を効率的に処理（最新5件のみ）
        recent_history = history[-5:] if len(history) > 5 else history
        history_parts = []
        for item in recent_history:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                user_msg = str(item[0]) if item[0] is not None else ""
                bot_msg = str(item[1]) if item[1] is not None else ""
                if user_msg or bot_msg:  # 空でない場合のみ追加
                    history_parts.append(f"ユーザー: {user_msg}\n麻理: {bot_msg}")
        
        history_text = "\n".join(history_parts)
        
        current_theme = scene_params.get("theme", "default")
        
        # メモリサマリーを含めたプロンプト構築
        memory_section = f"\n# 過去の記憶\n{memory_summary}\n" if memory_summary else ""
        
        # システムプロンプトを取得（隠された真実機能は既に統合済み）
        hidden_system_prompt = self.get_system_prompt_mari(use_ura_mode)
        
        user_prompt = f'''現在地: {current_theme}
好感度: {affection} ({stage_name}){memory_section}
履歴:
{history_text}

{f"指示: {instruction}" if instruction else f"「{message}」に応答:"}'''
        
        return self.call_llm(hidden_system_prompt, user_prompt)
    
    def should_generate_hidden_content(self, affection: int, message_count: int) -> bool:
        """隠された真実を生成すべきかどうかを判定する"""
        # 常に隠された真実を生成する（URAプロンプト使用）
        return True