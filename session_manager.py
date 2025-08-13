"""
セッション管理クラス - セッション分離強化のための基盤実装

このモジュールは、Streamlitアプリケーションにおけるセッション分離を強化し、
複数ユーザー間でのデータ混在を防ぐためのSessionManagerクラスを提供します。
"""

import streamlit as st
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import uuid

logger = logging.getLogger(__name__)


class SessionManager:
    """
    セッション管理クラス
    
    各ユーザーセッションの独立性を保証し、セッション整合性の検証、
    セッション情報の取得機能を提供します。
    """
    
    def __init__(self):
        """
        SessionManagerを初期化する
        
        セッションID、ユーザーID、作成時刻を記録し、
        セッション固有の識別子を設定します。
        """
        # セッション固有の識別子を生成
        self.session_id = id(st.session_state)
        self.user_id = None  # 後でユーザーIDが設定される
        self.created_at = datetime.now()
        self.last_validated = datetime.now()
        self.validation_count = 0
        self.recovery_count = 0
        
        # 検証履歴の記録用リスト
        self.validation_history: List[Dict[str, Any]] = []
        self.recovery_history: List[Dict[str, Any]] = []
        
        logger.info(f"SessionManager initialized - Session ID: {self.session_id}")
    
    def set_user_id(self, user_id: str):
        """
        ユーザーIDを設定する
        
        Args:
            user_id (str): 設定するユーザーID
        """
        self.user_id = user_id
        logger.debug(f"User ID set for session {self.session_id}: {user_id}")
    
    def validate_session_integrity(self) -> bool:
        """
        セッション整合性を検証する
        
        現在のセッションIDと初期化時のセッションIDを比較し、
        セッションの整合性を確認します。
        
        Returns:
            bool: セッションが整合している場合True、そうでなければFalse
        """
        current_session_id = id(st.session_state)
        stored_session_id = st.session_state.get('_session_id')
        is_consistent = self.session_id == current_session_id
        
        # 検証回数をカウント
        self.validation_count += 1
        validation_time = datetime.now()
        self.last_validated = validation_time
        
        # 検証履歴を記録
        validation_record = {
            "timestamp": validation_time.isoformat(),
            "validation_count": self.validation_count,
            "original_session_id": self.session_id,
            "current_session_id": current_session_id,
            "stored_session_id": stored_session_id,
            "is_consistent": is_consistent,
            "session_keys_count": len(st.session_state.keys()),
            "user_id": self.user_id
        }
        
        # 履歴リストのサイズ制限（最新100件まで保持）
        if len(self.validation_history) >= 100:
            self.validation_history.pop(0)
        
        self.validation_history.append(validation_record)
        
        if not is_consistent:
            logger.warning(
                f"Session integrity violation detected! "
                f"Original: {self.session_id}, Current: {current_session_id}, "
                f"Stored: {stored_session_id}, Validation #{self.validation_count}"
            )
        else:
            logger.debug(f"Session integrity validated successfully (count: {self.validation_count})")
        
        return is_consistent
    
    def recover_session(self):
        """
        セッション不整合時の復旧処理
        
        セッションIDの不整合が検出された場合に、
        セッション状態を修正して整合性を回復します。
        """
        old_session_id = self.session_id
        new_session_id = id(st.session_state)
        recovery_time = datetime.now()
        
        # セッションIDを現在の値に更新
        self.session_id = new_session_id
        self.recovery_count += 1
        self.last_validated = recovery_time
        
        # 復旧履歴を記録
        recovery_record = {
            "timestamp": recovery_time.isoformat(),
            "recovery_count": self.recovery_count,
            "old_session_id": old_session_id,
            "new_session_id": new_session_id,
            "user_id": self.user_id,
            "recovery_type": "session_id_mismatch",
            "session_keys_count": len(st.session_state.keys()),
            "validation_count_at_recovery": self.validation_count
        }
        
        # 履歴リストのサイズ制限（最新50件まで保持）
        if len(self.recovery_history) >= 50:
            self.recovery_history.pop(0)
        
        self.recovery_history.append(recovery_record)
        
        logger.info(
            f"Session recovered - Old ID: {old_session_id}, "
            f"New ID: {new_session_id}, Recovery count: {self.recovery_count}"
        )
        
        # セッション状態にも新しいIDを記録
        st.session_state._session_id = new_session_id
    
    def get_session_info(self) -> Dict[str, Any]:
        """
        セッション情報を取得する
        
        現在のセッション状態、検証履歴、復旧履歴などの
        詳細な情報を辞書形式で返します。
        
        Returns:
            Dict[str, Any]: セッション情報を含む辞書
        """
        current_session_id = id(st.session_state)
        is_consistent = self.session_id == current_session_id
        
        session_info = {
            "session_id": self.session_id,
            "current_session_id": current_session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_validated": self.last_validated.isoformat(),
            "validation_count": self.validation_count,
            "recovery_count": self.recovery_count,
            "is_consistent": is_consistent,
            "session_age_seconds": (datetime.now() - self.created_at).total_seconds(),
            "stored_session_id": st.session_state.get('_session_id'),
            "session_keys": list(st.session_state.keys()),
            "validation_history_count": len(self.validation_history),
            "recovery_history_count": len(self.recovery_history),
            "last_validation_result": self.validation_history[-1] if self.validation_history else None,
            "last_recovery_result": self.recovery_history[-1] if self.recovery_history else None
        }
        
        return session_info
    
    def get_validation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        検証履歴を取得する
        
        Args:
            limit (int): 取得する履歴の最大件数（デフォルト: 10）
        
        Returns:
            List[Dict[str, Any]]: 検証履歴のリスト（新しい順）
        """
        return self.validation_history[-limit:] if self.validation_history else []
    
    def get_recovery_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        復旧履歴を取得する
        
        Args:
            limit (int): 取得する履歴の最大件数（デフォルト: 10）
        
        Returns:
            List[Dict[str, Any]]: 復旧履歴のリスト（新しい順）
        """
        return self.recovery_history[-limit:] if self.recovery_history else []
    
    def get_isolation_status(self) -> Dict[str, Any]:
        """
        セッション分離状態を取得する
        
        各コンポーネントの分離状態を確認し、
        セッション独立性の詳細情報を返します。
        
        Returns:
            Dict[str, Any]: 分離状態情報を含む辞書
        """
        isolation_status = {
            "session_isolation": {
                "session_manager_present": hasattr(st.session_state, '_session_manager'),
                "session_id_consistent": self.validate_session_integrity(),
                "user_id_set": self.user_id is not None
            },
            "component_isolation": {
                "chat_isolated": 'chat' in st.session_state,
                "memory_isolated": 'memory_manager' in st.session_state,
                "notifications_isolated": all(key in st.session_state for key in [
                    'memory_notifications', 'affection_notifications'
                ]),
                "rate_limit_isolated": 'chat' in st.session_state and 
                                     'limiter_state' in st.session_state.get('chat', {})
            },
            "data_integrity": {
                "chat_messages_count": len(st.session_state.get('chat', {}).get('messages', [])),
                "memory_cache_size": len(getattr(st.session_state.get('memory_manager'), 'important_words_cache', [])),
                "special_memories_count": len(getattr(st.session_state.get('memory_manager'), 'special_memories', [])),
                "pending_notifications": {
                    "memory": len(st.session_state.get('memory_notifications', [])),
                    "affection": len(st.session_state.get('affection_notifications', []))
                }
            }
        }
        
        return isolation_status
    
    def reset_session_data(self):
        """
        セッションデータを完全にリセットする
        フルリセット時に使用
        """
        try:
            # 検証履歴をクリア
            self.validation_history.clear()
            self.recovery_history.clear()
            
            # カウンターをリセット
            self.validation_count = 0
            self.recovery_count = 0
            
            # タイムスタンプを更新
            self.last_validated = datetime.now()
            
            logger.info("SessionManagerのデータをリセットしました")
            
        except Exception as e:
            logger.error(f"SessionManagerリセットエラー: {e}")
    
    def get_isolation_summary(self) -> Dict[str, str]:
        """
        セッション分離状態のサマリーを取得する
        
        デバッグ表示用の簡潔な分離状態情報を返します。
        
        Returns:
            Dict[str, str]: 分離状態のサマリー情報
        """
        isolation_status = self.get_isolation_status()
        
        # 各カテゴリの状態を評価
        session_ok = all(isolation_status["session_isolation"].values())
        components_ok = all(isolation_status["component_isolation"].values())
        
        # データ整合性の評価
        data_integrity = isolation_status["data_integrity"]
        data_ok = (
            data_integrity["chat_messages_count"] >= 0 and
            data_integrity["memory_cache_size"] >= 0 and
            data_integrity["special_memories_count"] >= 0
        )
        
        # 全体的な健全性評価
        overall_health = "正常" if (session_ok and components_ok and data_ok) else "要注意"
        
        summary = {
            "overall_health": overall_health,
            "session_isolation": "✅ 正常" if session_ok else "❌ 問題あり",
            "component_isolation": "✅ 正常" if components_ok else "❌ 問題あり", 
            "data_integrity": "✅ 正常" if data_ok else "❌ 問題あり",
            "validation_status": f"検証{self.validation_count}回/復旧{self.recovery_count}回",
            "session_age": f"{round((datetime.now() - self.created_at).total_seconds() / 60, 1)}分",
            "last_check": f"{round((datetime.now() - self.last_validated).total_seconds(), 1)}秒前"
        }
        
        return summary
    
    def __str__(self) -> str:
        """
        SessionManagerの文字列表現
        
        Returns:
            str: セッション情報の要約
        """
        return (
            f"SessionManager(session_id={self.session_id}, "
            f"user_id={self.user_id}, "
            f"validations={self.validation_count}, "
            f"recoveries={self.recovery_count})"
        )
    
    def __repr__(self) -> str:
        """
        SessionManagerの詳細な文字列表現
        
        Returns:
            str: セッション情報の詳細
        """
        return self.__str__()


def get_session_manager() -> SessionManager:
    """
    現在のセッションのSessionManagerインスタンスを取得する
    
    セッション状態にSessionManagerが存在しない場合は新規作成します。
    
    Returns:
        SessionManager: 現在のセッションのSessionManagerインスタンス
    """
    if '_session_manager' not in st.session_state:
        st.session_state._session_manager = SessionManager()
        logger.info("New SessionManager created for current session")
    
    return st.session_state._session_manager


def validate_session_state() -> bool:
    """
    現在のセッション状態を検証する
    
    SessionManagerを使用してセッション整合性をチェックし、
    不整合が検出された場合は自動復旧を試行します。
    
    Returns:
        bool: セッションが整合している場合True、復旧に失敗した場合False
    """
    try:
        session_manager = get_session_manager()
        
        # 基本的なセッション整合性チェック
        if not session_manager.validate_session_integrity():
            logger.warning("Session inconsistency detected, attempting recovery...")
            session_manager.recover_session()
            
            # 復旧後に再度検証
            if session_manager.validate_session_integrity():
                logger.info("Session recovery successful")
            else:
                logger.error("Session recovery failed")
                return False
        
        # 追加の整合性チェック
        validation_results = perform_detailed_session_validation(session_manager)
        
        # 重要な不整合が検出された場合はログに記録
        critical_issues = [issue for issue in validation_results if issue.get('severity') == 'critical']
        if critical_issues:
            logger.error(f"Critical session validation issues detected: {critical_issues}")
            return False
        
        # 警告レベルの問題はログに記録するが、処理は継続
        warning_issues = [issue for issue in validation_results if issue.get('severity') == 'warning']
        if warning_issues:
            logger.warning(f"Session validation warnings: {warning_issues}")
        
        return True
        
    except Exception as e:
        logger.error(f"Session validation failed with exception: {e}")
        return False


def perform_detailed_session_validation(session_manager: SessionManager) -> List[Dict[str, Any]]:
    """
    詳細なセッション検証を実行する
    
    Args:
        session_manager (SessionManager): 検証対象のSessionManagerインスタンス
    
    Returns:
        List[Dict[str, Any]]: 検証結果のリスト（問題が検出された場合のみ）
    """
    validation_issues = []
    
    try:
        # 1. セッションID比較による整合性チェック
        current_session_id = id(st.session_state)
        stored_session_id = st.session_state.get('_session_id')
        
        if session_manager.session_id != current_session_id:
            validation_issues.append({
                "type": "session_id_mismatch",
                "severity": "critical",
                "description": "SessionManager session_id does not match current session",
                "details": {
                    "manager_session_id": session_manager.session_id,
                    "current_session_id": current_session_id
                }
            })
        
        if stored_session_id and stored_session_id != current_session_id:
            validation_issues.append({
                "type": "stored_session_id_mismatch",
                "severity": "warning",
                "description": "Stored session_id does not match current session",
                "details": {
                    "stored_session_id": stored_session_id,
                    "current_session_id": current_session_id
                }
            })
        
        # 2. 必須セッション状態の存在チェック（初期化後のみ）
        # 初期化中の場合はこのチェックをスキップ
        if st.session_state.get('_initialization_complete', False):
            required_keys = ['user_id', 'chat', 'memory_manager']
            for key in required_keys:
                if key not in st.session_state:
                    validation_issues.append({
                        "type": "missing_required_key",
                        "severity": "critical",
                        "description": f"Required session state key '{key}' is missing",
                        "details": {"missing_key": key}
                    })
        
        # 3. チャット状態の整合性チェック
        if 'chat' in st.session_state:
            chat_state = st.session_state.chat
            required_chat_keys = ['messages', 'affection', 'scene_params', 'limiter_state']
            
            for key in required_chat_keys:
                if key not in chat_state:
                    validation_issues.append({
                        "type": "missing_chat_key",
                        "severity": "warning",
                        "description": f"Required chat state key '{key}' is missing",
                        "details": {"missing_chat_key": key}
                    })
            
            # 好感度の範囲チェック
            affection = chat_state.get('affection', 0)
            if not (0 <= affection <= 100):
                validation_issues.append({
                    "type": "invalid_affection_value",
                    "severity": "warning",
                    "description": "Affection value is out of valid range (0-100)",
                    "details": {"affection_value": affection}
                })
        
        # 4. MemoryManagerの整合性チェック
        if 'memory_manager' in st.session_state:
            memory_manager = st.session_state.memory_manager
            if not hasattr(memory_manager, 'important_words_cache'):
                validation_issues.append({
                    "type": "invalid_memory_manager",
                    "severity": "warning",
                    "description": "MemoryManager instance is missing required attributes",
                    "details": {"missing_attribute": "important_words_cache"}
                })
        
        # 5. ユーザーIDの整合性チェック
        session_user_id = st.session_state.get('user_id')
        manager_user_id = session_manager.user_id
        
        if session_user_id and manager_user_id and session_user_id != manager_user_id:
            validation_issues.append({
                "type": "user_id_mismatch",
                "severity": "warning",
                "description": "User ID mismatch between session state and SessionManager",
                "details": {
                    "session_user_id": session_user_id,
                    "manager_user_id": manager_user_id
                }
            })
        
        # 6. 通知リストの整合性チェック
        notification_keys = ['memory_notifications', 'affection_notifications']
        for key in notification_keys:
            if key in st.session_state:
                notifications = st.session_state[key]
                if not isinstance(notifications, list):
                    validation_issues.append({
                        "type": "invalid_notification_type",
                        "severity": "warning",
                        "description": f"Notification key '{key}' is not a list",
                        "details": {"key": key, "type": type(notifications).__name__}
                    })
        
    except Exception as e:
        validation_issues.append({
            "type": "validation_exception",
            "severity": "critical",
            "description": "Exception occurred during detailed validation",
            "details": {"exception": str(e)}
        })
    
    return validation_issues