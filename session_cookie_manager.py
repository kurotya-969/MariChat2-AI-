"""
セッションCookie管理システム
UUIDベースのユーザー識別とCookie管理を提供する
"""
import uuid
import json
import os
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import streamlit as st
import logging

logger = logging.getLogger(__name__)

class SessionCookieManager:
    """セッションCookie管理クラス"""
    
    def __init__(self, storage_path: str = "session_data"):
        """
        初期化
        
        Args:
            storage_path: セッションデータの保存パス
        """
        self.storage_path = storage_path
        self.cookie_name = "mari_session_id"
        self.session_duration_days = 7
        self.cleanup_interval_hours = 24
        
        # ストレージディレクトリを作成
        os.makedirs(self.storage_path, exist_ok=True)
        
        # 最後のクリーンアップ時刻を記録するファイル
        self.cleanup_file = os.path.join(self.storage_path, "last_cleanup.json")
    
    def get_or_create_session_id(self) -> str:
        """
        セッションIDを取得または新規作成
        
        Returns:
            セッションID（UUID4形式）
        """
        try:
            # 既存のセッションIDを確認
            session_id = self._get_session_id_from_state()
            
            if session_id and self._is_valid_session(session_id):
                # 有効なセッションIDが存在する場合
                self._update_session_access_time(session_id)
                logger.info(f"既存セッションID使用: {session_id[:8]}...")
                return session_id
            
            # 新しいセッションIDを生成
            session_id = str(uuid.uuid4())
            self._create_new_session(session_id)
            logger.info(f"新規セッションID生成: {session_id[:8]}...")
            
            return session_id
            
        except Exception as e:
            logger.error(f"セッションID取得エラー: {e}")
            # フォールバック：一時的なセッションID
            return str(uuid.uuid4())
    
    def _get_session_id_from_state(self) -> Optional[str]:
        """
        Streamlitの状態からセッションIDを取得
        
        Returns:
            セッションID（存在しない場合はNone）
        """
        # Streamlitのセッション状態から取得
        session_id = st.session_state.get('mari_session_id')
        
        if session_id:
            return session_id
        
        # URLパラメータから取得を試行（フォールバック）
        query_params = st.query_params
        if 'session_id' in query_params:
            session_id = query_params['session_id']
            if self._is_valid_uuid(session_id):
                return session_id
        
        return None
    
    def _is_valid_uuid(self, uuid_string: str) -> bool:
        """
        UUIDの形式が正しいかチェック
        
        Args:
            uuid_string: チェックするUUID文字列
            
        Returns:
            有効な場合True
        """
        try:
            uuid.UUID(uuid_string, version=4)
            return True
        except (ValueError, TypeError):
            return False
    
    def _is_valid_session(self, session_id: str) -> bool:
        """
        セッションが有効かチェック
        
        Args:
            session_id: チェックするセッションID
            
        Returns:
            有効な場合True
        """
        try:
            session_file = os.path.join(self.storage_path, f"{session_id}.json")
            
            if not os.path.exists(session_file):
                return False
            
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # 最終アクセス時刻をチェック
            last_access = datetime.fromisoformat(session_data.get('last_access', ''))
            expiry_time = last_access + timedelta(days=self.session_duration_days)
            
            return datetime.now() < expiry_time
            
        except Exception as e:
            logger.warning(f"セッション検証エラー: {e}")
            return False
    
    def _create_new_session(self, session_id: str) -> None:
        """
        新しいセッションを作成
        
        Args:
            session_id: 新しいセッションID
        """
        try:
            # セッションデータを作成
            session_data = {
                'session_id': session_id,
                'created_at': datetime.now().isoformat(),
                'last_access': datetime.now().isoformat(),
                'user_agent': self._get_user_agent(),
                'ip_hash': self._get_ip_hash()
            }
            
            # セッションファイルに保存
            session_file = os.path.join(self.storage_path, f"{session_id}.json")
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            
            # Streamlitの状態に保存
            st.session_state.mari_session_id = session_id
            
            # Cookie設定のJavaScriptを生成（セキュア設定）
            self._set_secure_cookie(session_id)
            
        except Exception as e:
            logger.error(f"新規セッション作成エラー: {e}")
    
    def _update_session_access_time(self, session_id: str) -> None:
        """
        セッションの最終アクセス時刻を更新
        
        Args:
            session_id: 更新するセッションID
        """
        try:
            session_file = os.path.join(self.storage_path, f"{session_id}.json")
            
            if os.path.exists(session_file):
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                session_data['last_access'] = datetime.now().isoformat()
                
                with open(session_file, 'w', encoding='utf-8') as f:
                    json.dump(session_data, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.warning(f"セッションアクセス時刻更新エラー: {e}")
    
    def _set_secure_cookie(self, session_id: str) -> None:
        """
        セキュアなCookieを設定
        
        Args:
            session_id: 設定するセッションID
        """
        try:
            # セキュアCookie設定のJavaScript
            cookie_js = f"""
            <script>
            // セキュアCookieの設定
            function setSecureCookie() {{
                const sessionId = '{session_id}';
                const expiryDays = {self.session_duration_days};
                const expiryDate = new Date();
                expiryDate.setTime(expiryDate.getTime() + (expiryDays * 24 * 60 * 60 * 1000));
                
                // セキュア属性付きCookieを設定
                let cookieString = `{self.cookie_name}=${{sessionId}}; expires=${{expiryDate.toUTCString()}}; path=/; SameSite=Strict`;
                
                // HTTPS環境の場合はSecure属性を追加
                if (location.protocol === 'https:') {{
                    cookieString += '; Secure';
                }}
                
                // HttpOnly属性は JavaScript では設定できないため、サーバーサイドで設定が必要
                document.cookie = cookieString;
                
                console.log('セッションCookie設定完了:', sessionId.substring(0, 8) + '...');
            }}
            
            // ページ読み込み時にCookieを設定
            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', setSecureCookie);
            }} else {{
                setSecureCookie();
            }}
            </script>
            """
            
            st.markdown(cookie_js, unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"セキュアCookie設定エラー: {e}")
    
    def _get_user_agent(self) -> str:
        """
        ユーザーエージェントを取得（可能な場合）
        
        Returns:
            ユーザーエージェント文字列
        """
        try:
            # Streamlitでは直接取得できないため、JavaScriptで取得
            return "Streamlit-Client"
        except Exception:
            return "Unknown"
    
    def _get_ip_hash(self) -> str:
        """
        IPアドレスのハッシュを取得（プライバシー保護）
        
        Returns:
            IPアドレスのハッシュ
        """
        try:
            import hashlib
            # 実際のIPアドレスは取得せず、セッション識別用のハッシュのみ
            session_hash = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
            return session_hash
        except Exception:
            return "unknown"
    
    def cleanup_expired_sessions(self) -> None:
        """
        期限切れセッションのクリーンアップ
        """
        try:
            # 前回のクリーンアップ時刻をチェック
            if not self._should_cleanup():
                return
            
            current_time = datetime.now()
            cleaned_count = 0
            
            # セッションファイルをスキャン
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.json') and filename != 'last_cleanup.json':
                    session_file = os.path.join(self.storage_path, filename)
                    
                    try:
                        with open(session_file, 'r', encoding='utf-8') as f:
                            session_data = json.load(f)
                        
                        last_access = datetime.fromisoformat(session_data.get('last_access', ''))
                        expiry_time = last_access + timedelta(days=self.session_duration_days)
                        
                        if current_time > expiry_time:
                            os.remove(session_file)
                            cleaned_count += 1
                            logger.info(f"期限切れセッション削除: {filename}")
                    
                    except Exception as e:
                        logger.warning(f"セッションファイル処理エラー {filename}: {e}")
            
            # クリーンアップ時刻を記録
            self._update_cleanup_time()
            
            if cleaned_count > 0:
                logger.info(f"セッションクリーンアップ完了: {cleaned_count}件削除")
            
        except Exception as e:
            logger.error(f"セッションクリーンアップエラー: {e}")
    
    def _should_cleanup(self) -> bool:
        """
        クリーンアップが必要かチェック
        
        Returns:
            クリーンアップが必要な場合True
        """
        try:
            if not os.path.exists(self.cleanup_file):
                return True
            
            with open(self.cleanup_file, 'r', encoding='utf-8') as f:
                cleanup_data = json.load(f)
            
            last_cleanup = datetime.fromisoformat(cleanup_data.get('last_cleanup', ''))
            next_cleanup = last_cleanup + timedelta(hours=self.cleanup_interval_hours)
            
            return datetime.now() > next_cleanup
            
        except Exception:
            return True
    
    def _update_cleanup_time(self) -> None:
        """
        クリーンアップ時刻を更新
        """
        try:
            cleanup_data = {
                'last_cleanup': datetime.now().isoformat()
            }
            
            with open(self.cleanup_file, 'w', encoding='utf-8') as f:
                json.dump(cleanup_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.warning(f"クリーンアップ時刻更新エラー: {e}")
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        セッション情報を取得
        
        Args:
            session_id: セッションID
            
        Returns:
            セッション情報の辞書
        """
        try:
            session_file = os.path.join(self.storage_path, f"{session_id}.json")
            
            if os.path.exists(session_file):
                with open(session_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            return {}
            
        except Exception as e:
            logger.error(f"セッション情報取得エラー: {e}")
            return {}
    
    def delete_session(self, session_id: str) -> bool:
        """
        セッションを削除
        
        Args:
            session_id: 削除するセッションID
            
        Returns:
            削除成功時True
        """
        try:
            session_file = os.path.join(self.storage_path, f"{session_id}.json")
            
            if os.path.exists(session_file):
                os.remove(session_file)
                logger.info(f"セッション削除: {session_id[:8]}...")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"セッション削除エラー: {e}")
            return False