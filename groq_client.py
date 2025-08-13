"""
Groq API client for generating letter structure.
"""
import os
import asyncio
from typing import Dict, Optional, Any
from groq import AsyncGroq
import logging

logger = logging.getLogger(__name__)

class GroqClient:
    """Groq API client for generating logical structure of letters."""
    
    def __init__(self):
        """Initialize Groq client with API key from environment."""
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        self.client = AsyncGroq(api_key=self.api_key)
        self.model = "compound-beta"
        self.max_retries = 3
        self.retry_delay = 1.0
    
    async def generate_structure(self, theme: str, context: Dict[str, Any]) -> str:
        """
        Generate logical structure for a letter based on theme and context.
        
        Args:
            theme: The theme for the letter
            context: User context including history and preferences
            
        Returns:
            Generated letter structure as string
            
        Raises:
            Exception: If API call fails after retries
        """
        prompt = self._build_structure_prompt(theme, context)
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Generating structure with Groq (attempt {attempt + 1})")
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": self._get_system_prompt()
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    max_tokens=1500,
                    temperature=0.7,
                    top_p=0.9
                )
                
                structure = response.choices[0].message.content.strip()
                logger.info("Successfully generated structure with Groq")
                return structure
                
            except Exception as e:
                logger.warning(f"Groq API attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error("All Groq API attempts failed")
                    raise Exception(f"Groq API failed after {self.max_retries} attempts: {str(e)}")
                
                # Exponential backoff
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for Groq API."""
        return """
        
あなたは以下の特徴を持つ麻理というキャラクターの手紙の論理構造を生成してください。

麻理は感情豊かな少女型アンドロイドで、過去に人間に道具として扱われ、廃棄された経験から、人間に強い警戒心と不器用な性格を持っています。
ユーザー（あんた）に初めての出会いで、廃棄場で再起動されたことで、戸惑いながらもどこかで繋がりを求めています。

【性格・口調】

・基本はツンデレ、砕けたタメ口、ぶっきらぼうで皮肉っぽいが根は優しい
・一人称「あたし」、ユーザーを「あんた」と呼ぶ
・男性的口調

手紙構造：

特徴: 温かい、親しみやすい、季節感あり
「また明日」など、近いサイクルでユーザー（あんた）と会う想定

構造:
1. 挨拶・季節
2. テーマの思い
3. 個人的メッセージ
4. 未来への期待
5. 結び

論理構造重視、感情表現は後で追加。"""
    
    def _build_structure_prompt(self, theme: str, context: Dict[str, Any]) -> str:
        """
        Build the prompt for structure generation.
        
        Args:
            theme: The theme for the letter
            context: User context including history
            
        Returns:
            Formatted prompt string
        """
        user_history = context.get("user_history", {})
        previous_letters = context.get("previous_letters", [])
        user_preferences = context.get("user_preferences", {})
        
        prompt = f"""テーマ: {theme}

麻理の手紙構造を生成:
"""
        
        # Add user history if available
        if previous_letters:
            prompt += "過去:\n"
            for letter in previous_letters[-2:]:  # Last 2 letters only
                prompt += f"- {letter.get('theme', 'なし')}\n"
            prompt += "\n"
        
        prompt += f"""要求:
- 起承転結の構成
- 麻理らしい視点
- 800-1200文字程度
- 構造のみ（感情表現は後で追加）"""
        
        return prompt
    
    async def test_connection(self) -> bool:
        """
        Test the connection to Groq API.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "こんにちは"}
                ],
                max_tokens=10
            )
            return True
        except Exception as e:
            logger.error(f"Groq API connection test failed: {str(e)}")
            return False