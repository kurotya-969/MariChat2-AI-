"""
レート制限モジュール
APIの過度な使用を防ぐためのレート制限機能
"""
import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class RateLimiter:
    """レート制限を管理するクラス"""
    
    def __init__(self, max_requests: int = 15, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
    
    def create_limiter_state(self) -> Dict[str, Any]:
        """レートリミッター状態を作成（型安全）"""
        return {
            "timestamps": [],
            "is_blocked": False
        }
    
    def check_limiter(self, limiter_state: Dict[str, Any]) -> bool:
        """レート制限をチェックする"""
        # limiter_stateが辞書であることを確認。そうでなければ、エラーを防ぐために再初期化。
        if not isinstance(limiter_state, dict):
            logger.error(f"limiter_stateが辞書ではありません: {type(limiter_state)}. 再初期化します。")
            limiter_state.clear()
            limiter_state.update(self.create_limiter_state())
        
        if limiter_state.get("is_blocked", False):
            return False  # ブロック状態を示すためにFalseを返す
        
        now = time.time()
        timestamps = limiter_state.get("timestamps", [])
        if not isinstance(timestamps, list):
            timestamps = []
            limiter_state["timestamps"] = timestamps
        
        # 時間窓外のタイムスタンプを削除
        limiter_state["timestamps"] = [
            t for t in timestamps if now - t < self.time_window
        ]
        
        # リクエスト数が上限を超えているかチェック
        if len(limiter_state["timestamps"]) >= self.max_requests:
            logger.warning("レートリミット超過")
            limiter_state["is_blocked"] = True
            return False
        
        # 新しいリクエストのタイムスタンプを追加
        limiter_state["timestamps"].append(now)
        return True
    
    def reset_limiter(self, limiter_state: Dict[str, Any]):
        """レートリミッターをリセットする"""
        if isinstance(limiter_state, dict):
            limiter_state["timestamps"] = []
            limiter_state["is_blocked"] = False