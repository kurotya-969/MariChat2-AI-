"""
FastAPIセッション管理クライアント
HttpOnlyクッキーによるセキュアなセッション管理のクライアント側実装
"""
import requests
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any
import streamlit as st
import logging

logger = logging.getLogger(__name__)

class SessionAPIClient:
    """FastAPIセッションサーバーとの通信クライアント"""
    
    def __init__(self, api_base_url: str = None):
        """
        初期化
        
        Args:
            api_base_url: FastAPIサーバーのベースURL（Noneの場合は自動決定）
        """
        # Hugging Face Spacesでの実行を考慮してベースURLを自動決定
        if api_base_url is None:
            is_spaces = os.getenv("SPACE_ID") is not None
            if is_spaces:
                # Hugging Face Spacesでは同一コンテナ内なのでlocalhostを使用
                api_base_url = "http://localhost:8000"
            else:
                api_base_url = "http://127.0.0.1:8000"
        
        self.api_base_url = api_base_url
        self.session = requests.Session()
        
        # リクエストタイムアウト設定
        self.timeout = 10
        
        # セッション情報をStreamlitの状態に保存
        if 'session_info' not in st.session_state:
            st.session_state.session_info = {}
    
    def create_session(self) -> Optional[str]:
        """
        新しいセッションを作成
        
        Returns:
            セッションID（成功時）、None（失敗時）
        """
        try:
            url = f"{self.api_base_url}/session/create"
            response = self.session.post(url, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                session_id = data.get('session_id')
                
                # セッション情報を保存
                st.session_state.session_info = {
                    'session_id': session_id,
                    'created_at': datetime.now().isoformat(),
                    'status': 'active'
                }
                
                logger.info(f"新規セッション作成成功: {session_id[:8]}...")
                return session_id
            else:
                logger.error(f"セッション作成失敗: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"セッション作成リクエストエラー: {e}")
            return None
        except Exception as e:
            logger.error(f"セッション作成エラー: {e}")
            return None
    
    def validate_session(self) -> bool:
        """
        現在のセッションの有効性を検証
        
        Returns:
            有効な場合True
        """
        try:
            # session_infoが存在しない場合は無効とみなす
            if 'session_info' not in st.session_state:
                logger.debug("session_info未存在 - セッション無効")
                return False
            
            url = f"{self.api_base_url}/session/validate"
            response = self.session.post(url, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                is_valid = data.get('valid', False)
                
                if is_valid:
                    # セッション情報を更新
                    session_id = data.get('session_id')
                    st.session_state.session_info.update({
                        'session_id': session_id,
                        'last_validated': datetime.now().isoformat(),
                        'status': 'active'
                    })
                    logger.debug(f"セッション検証成功: {session_id[:8]}...")
                else:
                    # 無効なセッション
                    if 'session_info' in st.session_state:
                        st.session_state.session_info['status'] = 'invalid'
                    logger.warning("セッション検証失敗: 無効なセッション")
                
                return is_valid
            else:
                logger.error(f"セッション検証失敗: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"セッション検証リクエストエラー: {e}")
            return False
        except Exception as e:
            logger.error(f"セッション検証エラー: {e}")
            return False
    
    def get_session_info(self) -> Optional[Dict[str, Any]]:
        """
        セッション情報を取得
        
        Returns:
            セッション情報（成功時）、None（失敗時）
        """
        try:
            url = f"{self.api_base_url}/session/info"
            response = self.session.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                
                # セッション情報を更新
                st.session_state.session_info.update({
                    'session_id': data.get('session_id'),
                    'created_at': data.get('created_at'),
                    'last_access': data.get('last_access'),
                    'status': 'active'
                })
                
                return data
            else:
                logger.error(f"セッション情報取得失敗: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"セッション情報取得リクエストエラー: {e}")
            return None
        except Exception as e:
            logger.error(f"セッション情報取得エラー: {e}")
            return None
    
    def delete_session(self) -> bool:
        """
        現在のセッションを削除
        
        Returns:
            削除成功時True
        """
        try:
            url = f"{self.api_base_url}/session/delete"
            response = self.session.delete(url, timeout=self.timeout)
            
            if response.status_code == 200:
                # セッション情報をクリア
                st.session_state.session_info = {
                    'status': 'deleted',
                    'deleted_at': datetime.now().isoformat()
                }
                
                logger.info("セッション削除成功")
                return True
            else:
                logger.error(f"セッション削除失敗: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"セッション削除リクエストエラー: {e}")
            return False
        except Exception as e:
            logger.error(f"セッション削除エラー: {e}")
            return False
    
    def get_or_create_session_id(self) -> str:
        """
        セッションIDを取得または新規作成
        複数セッション生成を防ぐため、既存セッション情報を最優先でチェック
        
        Returns:
            セッションID
        """
        try:
            # 既存のセッション情報をチェック
            existing_session_info = st.session_state.get('session_info', {})
            existing_session_id = existing_session_info.get('session_id')
            
            # 既存セッションがある場合は、基本的にそれを使用（検証は最小限に）
            if existing_session_id:
                logger.debug(f"既存セッション情報発見: {existing_session_id[:8]}...")
                
                # セッション状態をチェック
                session_status = existing_session_info.get('status', 'unknown')
                
                # 明示的に無効とマークされていない限り、既存セッションを使用
                if session_status != 'invalid':
                    logger.debug(f"既存セッション使用: {existing_session_id[:8]}... (status: {session_status})")
                    return existing_session_id
                else:
                    logger.info(f"無効セッション検出: {existing_session_id[:8]}... - 新規作成")
            
            # 新しいセッションを作成（一度だけ）
            logger.info("新規セッション作成開始...")
            
            # セッション作成中フラグを設定（重複作成防止）
            if st.session_state.get('session_creating', False):
                logger.warning("セッション作成中 - 待機")
                # 既存のセッション情報があればそれを返す
                if existing_session_id:
                    return existing_session_id
                # なければフォールバック
                import uuid
                return str(uuid.uuid4())
            
            st.session_state.session_creating = True
            
            try:
                if self.is_server_available():
                    session_id = self.create_session()
                    if session_id:
                        logger.info(f"新規セッション作成成功: {session_id[:8]}...")
                        return session_id
                
                # フォールバック: ローカルセッションID生成
                import uuid
                fallback_id = str(uuid.uuid4())
                logger.warning(f"フォールバックセッションID生成: {fallback_id[:8]}...")
                
                # フォールバックセッション情報を保存
                st.session_state.session_info = {
                    'session_id': fallback_id,
                    'created_at': datetime.now().isoformat(),
                    'fallback_mode': True,
                    'server_available': False,
                    'status': 'fallback'
                }
                
                return fallback_id
                
            finally:
                # セッション作成中フラグをクリア
                st.session_state.session_creating = False
            
        except Exception as e:
            logger.error(f"セッションID取得エラー: {e}")
            # セッション作成中フラグをクリア
            st.session_state.session_creating = False
            
            # 最終フォールバック
            import uuid
            fallback_id = str(uuid.uuid4())
            logger.error(f"最終フォールバックセッションID: {fallback_id[:8]}...")
            return fallback_id
    
    def is_server_available(self) -> bool:
        """
        FastAPIサーバーが利用可能かチェック
        
        Returns:
            利用可能な場合True
        """
        try:
            url = f"{self.api_base_url}/health"
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_session_status(self) -> Dict[str, Any]:
        """
        現在のセッション状態を取得
        
        Returns:
            セッション状態の辞書
        """
        session_info = st.session_state.get('session_info', {})
        
        return {
            'session_id': session_info.get('session_id', 'unknown')[:8] + "..." if session_info.get('session_id') else 'none',
            'status': session_info.get('status', 'unknown'),
            'created_at': session_info.get('created_at'),
            'last_access': session_info.get('last_access'),
            'last_validated': session_info.get('last_validated'),
            'server_available': self.is_server_available()
        }
    
    def reset_session(self) -> str:
        """
        セッションをリセット（削除して新規作成）
        
        Returns:
            新しいセッションID
        """
        try:
            # 既存セッションを削除
            self.delete_session()
            
            # 新しいセッションを作成
            new_session_id = self.create_session()
            if new_session_id:
                logger.info(f"セッションリセット完了: {new_session_id[:8]}...")
                return new_session_id
            
            # フォールバック
            import uuid
            fallback_id = str(uuid.uuid4())
            logger.warning(f"セッションリセット失敗、フォールバック使用: {fallback_id[:8]}...")
            return fallback_id
            
        except Exception as e:
            logger.error(f"セッションリセットエラー: {e}")
            import uuid
            return str(uuid.uuid4())
    
    def get_cookie_status(self) -> Dict[str, Any]:
        """
        現在のCookie状態を取得
        
        Returns:
            Cookie状態の辞書
        """
        try:
            cookie_info = {
                'count': len(self.session.cookies),
                'cookies': [],
                'has_session_cookie': False,
                'timestamp': datetime.now().isoformat()
            }
            
            for cookie in self.session.cookies:
                cookie_data = {
                    'name': cookie.name,
                    'domain': cookie.domain,
                    'path': cookie.path,
                    'secure': cookie.secure,
                    'expires': cookie.expires
                }
                cookie_info['cookies'].append(cookie_data)
                
                # セッション関連のCookieをチェック
                if 'session' in cookie.name.lower():
                    cookie_info['has_session_cookie'] = True
            
            return cookie_info
            
        except Exception as e:
            logger.error(f"Cookie状態取得エラー: {e}")
            return {
                'count': 0,
                'cookies': [],
                'has_session_cookie': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def full_reset_session(self) -> Dict[str, Any]:
        """
        フルリセット（Cookie削除 + 新規セッション作成）
        サーバー接続エラーでも動作するフォールバック機能付き
        
        Returns:
            リセット結果の辞書
        """
        try:
            result = {
                'success': False,
                'old_session_id': None,
                'new_session_id': None,
                'message': '',
                'timestamp': datetime.now().isoformat(),
                'cookie_reset': False,
                'session_created': False,
                'server_available': False,
                'fallback_mode': False
            }
            
            # 現在のセッションIDを記録
            current_session_info = st.session_state.get('session_info', {})
            old_session_id = current_session_info.get('session_id', st.session_state.get('user_id', 'unknown'))
            result['old_session_id'] = old_session_id[:8] + "..." if len(old_session_id) > 8 else old_session_id
            
            logger.info(f"フルリセット開始 - 旧セッション: {result['old_session_id']}")
            
            # 1. サーバー接続テスト
            server_available = self._test_server_connection()
            result['server_available'] = server_available
            
            if server_available:
                # サーバーが利用可能な場合の通常処理
                logger.info("サーバー利用可能 - 通常のフルリセット実行")
                delete_success = self.delete_session()
                logger.info(f"セッション削除結果: {delete_success}")
            else:
                # サーバーが利用できない場合のフォールバック処理
                logger.warning("サーバー接続不可 - フォールバックモードでリセット実行")
                result['fallback_mode'] = True
            
            # 2. セッション情報を完全クリア
            if 'session_info' in st.session_state:
                del st.session_state.session_info
            
            # 3. 新しいrequestsセッションを作成（Cookie完全クリア）
            old_session = self.session
            cookie_count_before = len(old_session.cookies)
            
            self.session.close()
            self.session = requests.Session()
            
            # Cookieが完全にクリアされたことを確認
            cookie_count_after = len(self.session.cookies)
            result['cookie_reset'] = cookie_count_after == 0
            
            logger.info(f"Cookie状態 - 削除前: {cookie_count_before}個, 削除後: {cookie_count_after}個")
            
            # 4. 新しいセッションを作成
            if server_available:
                # サーバー経由でセッション作成
                new_session_id = self.create_session()
            else:
                # フォールバック: ローカルでセッションID生成
                import uuid
                new_session_id = str(uuid.uuid4())
                logger.info(f"フォールバックモード: ローカルセッションID生成 - {new_session_id[:8]}...")
            
            if new_session_id:
                result['success'] = True
                result['session_created'] = True
                result['new_session_id'] = new_session_id[:8] + "..."
                
                if result['fallback_mode']:
                    result['message'] = 'フルリセット成功（フォールバックモード） - Cookie削除＆ローカルセッション作成完了'
                else:
                    result['message'] = 'フルリセット成功 - Cookie削除＆新規セッション作成完了'
                
                logger.info(f"フルリセット成功: {result['old_session_id']} → {result['new_session_id']}")
                
                # 新しいセッション情報をStreamlitセッションに保存
                st.session_state.session_info = {
                    'session_id': new_session_id,
                    'created_at': datetime.now().isoformat(),
                    'reset_count': st.session_state.get('reset_count', 0) + 1,
                    'fallback_mode': result['fallback_mode'],
                    'server_available': result['server_available']
                }
                
            else:
                result['message'] = 'Cookie削除成功、セッション作成失敗'
                logger.error("フルリセット: 新規セッション作成失敗")
            
            return result
            
        except Exception as e:
            logger.error(f"フルリセットエラー: {e}")
            return {
                'success': False,
                'old_session_id': 'error',
                'new_session_id': None,
                'message': f'エラー: {str(e)}',
                'timestamp': datetime.now().isoformat(),
                'cookie_reset': False,
                'session_created': False,
                'server_available': False,
                'fallback_mode': True
            }
    
    def _test_server_connection(self) -> bool:
        """
        サーバー接続をテストする
        
        Returns:
            接続可能かどうか
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"サーバー接続テスト失敗: {e}")
            return False