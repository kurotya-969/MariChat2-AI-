"""
Hugging Face Spaces永続ストレージ対応ユーザー管理システム
Cookieベースのユーザー識別と/mnt/dataでの状態永続化を提供
"""
import os
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager

logger = logging.getLogger(__name__)

class PersistentUserManager:
    """永続ストレージ対応ユーザー管理クラス"""
    
    def __init__(self, storage_base_path: str = "/mnt/data"):
        """
        初期化
        
        Args:
            storage_base_path: 永続ストレージのベースパス（HF Spacesでは/mnt/data）
        """
        # Hugging Face Spacesの永続ストレージパス
        self.storage_base_path = storage_base_path
        self.user_data_dir = os.path.join(storage_base_path, "mari_users")
        self.session_data_dir = os.path.join(storage_base_path, "mari_sessions")
        
        # 永続Cookie管理設定（完全にユニークなキー名）
        self.cookie_name = "hf_spaces_persistent_session_id"  # HF Spaces専用の完全にユニークなキー
        self.cookie_expiry_days = 365  # 1年間有効な永続Cookie
        
        # ディレクトリ作成
        self._ensure_directories()
        
        # Cookie管理の遅延初期化（初回使用時に初期化）
        self.cookies = None
        self._cookie_initialized = False
        
        logger.info(f"永続ユーザー管理システム初期化: {self.user_data_dir}")
    
    def _ensure_directories(self):
        """必要なディレクトリを作成"""
        try:
            os.makedirs(self.user_data_dir, exist_ok=True)
            os.makedirs(self.session_data_dir, exist_ok=True)
            logger.info(f"ストレージディレクトリ確認完了: {self.user_data_dir}")
        except Exception as e:
            logger.error(f"ディレクトリ作成エラー: {e}")
            # フォールバック: ローカルディレクトリを使用
            self.user_data_dir = "local_mari_users"
            self.session_data_dir = "local_mari_sessions"
            os.makedirs(self.user_data_dir, exist_ok=True)
            os.makedirs(self.session_data_dir, exist_ok=True)
            logger.warning(f"フォールバック: ローカルディレクトリを使用 {self.user_data_dir}")
    
    def _ensure_cookie_manager(self, force_init: bool = False):
        """Cookie管理システムの初期化（永続Cookie対応）"""
        # 既に初期化済みで、強制初期化でない場合はスキップ
        if self._cookie_initialized and not force_init and self.cookies is not None:
            logger.debug("Cookie管理システム既に初期化済み - スキップ")
            return
        
        # 固定キーでCookie初期化チェック（永続化対応）
        browser_session_key = "hf_persistent_cookie_initialized"
        
        # より厳密な重複チェック
        if (not force_init and 
            st.session_state.get(browser_session_key, False) and 
            self.cookies is not None and
            hasattr(self.cookies, 'ready') and
            self.cookies.ready()):
            self._cookie_initialized = True
            logger.debug("永続Cookie管理システム既に準備完了 - スキップ")
            return
        
        # 初期化処理中フラグを設定（重複初期化防止）
        init_in_progress_key = "hf_persistent_cookie_initializing"
        if st.session_state.get(init_in_progress_key, False):
            logger.warning("永続Cookie初期化処理中 - 重複初期化を防止")
            return
        
        st.session_state[init_in_progress_key] = True
        
        try:
            logger.info("HF Spaces永続Cookie管理システム初期化開始（固定prefix使用）...")
            
            # セキュアなパスワードを生成（環境変数から取得、なければ生成）
            cookie_password = os.getenv("MARI_COOKIE_PASSWORD", "mari_chat_secure_key_2024")
            
            # 固定のprefixを使用（永続Cookie読み込みのため）
            # リロード後も同じCookieを読み込めるよう、固定のprefixを使用
            fixed_prefix = "hf_persistent_mari_"
            
            logger.debug(f"Cookie初期化: prefix={fixed_prefix} (固定prefix使用)")
            
            # EncryptedCookieManagerを初期化（永続Cookie対応）
            cookies = EncryptedCookieManager(
                prefix=fixed_prefix,  # 固定prefixで永続Cookie読み込み対応
                password=cookie_password
            )
            
            # Cookieが準備できるまで待機（より長い待機時間で既存Cookie読み込み対応）
            max_attempts = 10  # 試行回数を増加
            total_wait_time = 0
            max_wait_time = 3.0  # 最大3秒待機
            
            for attempt in range(max_attempts):
                if cookies.ready():
                    logger.info(f"Cookie準備完了 ({attempt + 1}回目の試行で成功)")
                    break
                
                wait_time = 0.2 + (attempt * 0.1)  # 段階的に待機時間を延長
                total_wait_time += wait_time
                
                if total_wait_time > max_wait_time:
                    logger.warning(f"Cookie準備タイムアウト ({total_wait_time:.1f}秒経過)")
                    break
                
                logger.debug(f"Cookie準備待機中... ({attempt + 1}/{max_attempts}, {wait_time:.1f}秒待機)")
                import time
                time.sleep(wait_time)
            
            # Cookie準備ができていない場合でも、一度試行してみる
            if not cookies.ready():
                logger.warning("Cookie準備未完了だが、既存Cookie読み込みを試行")
                # フォールバックモードにせず、Cookieオブジェクトは保持
                # 既存のCookieがある場合は読み込める可能性がある
            
            self.cookies = cookies
            self._cookie_initialized = True
            
            # 永続化フラグを設定
            st.session_state[browser_session_key] = True
            
            # 初期化処理中フラグをクリア
            if init_in_progress_key in st.session_state:
                del st.session_state[init_in_progress_key]
            
            # 既存Cookieの確認
            try:
                existing_cookies = dict(self.cookies)
                logger.info(f"HF Spaces永続Cookie管理システム初期化完了 - prefix: {fixed_prefix}, 既存Cookie数: {len(existing_cookies)}")
                
                # 既存のユーザーIDがあるかチェック
                if self.cookie_name in existing_cookies:
                    existing_user_id = existing_cookies[self.cookie_name]
                    logger.info(f"既存の永続Cookie発見: ユーザーID {existing_user_id[:8]}...")
                else:
                    logger.info("新規ユーザー: 永続Cookieなし")
                    
            except Exception as cookie_check_e:
                logger.debug(f"Cookie確認エラー（通常動作に影響なし）: {cookie_check_e}")
                logger.info(f"HF Spaces永続Cookie管理システム初期化完了 - prefix: {fixed_prefix}")
            
        except Exception as e:
            logger.error(f"永続Cookie管理初期化エラー: {e}")
            
            # 初期化処理中フラグをクリア
            if init_in_progress_key in st.session_state:
                del st.session_state[init_in_progress_key]
            
            # フォールバック: セッション状態のみ使用
            self.cookies = None
            self._cookie_initialized = True
            st.session_state[browser_session_key] = True
    
    def get_or_create_user_id(self, force_reset: bool = False) -> str:
        """
        Cookie認証ベースのユーザーID取得（ブラウザごとに一つのCookie）
        
        Args:
            force_reset: フルリセット時のみTrue
        
        Returns:
            ユーザーID
        """
        try:
            # 重複呼び出し防止: 処理中フラグをチェック
            if st.session_state.get('user_id_processing', False):
                logger.debug("ユーザーID取得処理中 - 待機")
                # 既存のuser_idがあればそれを返す
                if 'user_id' in st.session_state:
                    return st.session_state.user_id
                # なければ一時的なIDを返す
                return f"temp_{id(st.session_state)}"
            
            # 処理中フラグを設定
            st.session_state.user_id_processing = True
            
            try:
                # Cookie管理システムを初期化（最優先）
                self._ensure_cookie_manager(force_init=force_reset)
                
                # 1. CookieからユーザーIDを取得（最優先）
                user_id = self._get_user_id_from_cookie()
                
                if user_id and self._is_valid_user_id(user_id):
                    # 有効なCookieベースのユーザーIDが存在
                    self._update_user_access_time(user_id)
                    st.session_state.user_id = user_id  # セッション状態に保存
                    logger.info(f"HF Spaces永続Cookie認証成功: {user_id[:8]}...")
                    return user_id
                
                # 2. フルリセット時以外で、セッション状態にuser_idがある場合はCookieと照合
                if not force_reset and 'user_id' in st.session_state:
                    existing_id = st.session_state.user_id
                    if existing_id and self._is_valid_uuid(existing_id) and self._is_valid_user_id(existing_id):
                        # セッション状態のIDが有効な場合、Cookieにも保存
                        self._set_user_id_cookie(existing_id)
                        logger.info(f"セッション状態からユーザーID復元: {existing_id[:8]}...")
                        return existing_id
                
                # 2. Cookieが無効または存在しない場合は新規作成
                if force_reset or not user_id:
                    user_id = self._create_new_user_with_cookie()
                    logger.info(f"新規Cookie認証ユーザー作成: {user_id[:8]}...")
                    return user_id
                
                # 3. 最終フォールバック（Cookieが無効だが何らかのIDがある場合）
                if user_id and self._is_valid_uuid(user_id):
                    # UUIDとして有効だが、ユーザーファイルが存在しない場合は新規作成
                    user_id = self._create_new_user_with_cookie()
                    logger.info(f"フォールバック新規ユーザー作成: {user_id[:8]}...")
                    return user_id
                
                # 4. 完全フォールバック（Cookie無効時）
                logger.warning("Cookie認証失敗 - 一時的なセッションIDを使用")
                temp_id = str(uuid.uuid4())
                st.session_state.user_id = temp_id
                return temp_id
                
            finally:
                # 処理中フラグをクリア
                st.session_state.user_id_processing = False
                
        except Exception as e:
            logger.error(f"Cookie認証エラー: {e}")
            # 処理中フラグをクリア
            st.session_state.user_id_processing = False
            # 完全フォールバック: 一時的なIDを生成（Cookieに保存しない）
            temp_id = str(uuid.uuid4())
            st.session_state.user_id = temp_id
            logger.warning(f"一時的なセッションID使用: {temp_id[:8]}...")
            return temp_id
    
    def _get_user_id_from_cookie(self) -> Optional[str]:
        """CookieからユーザーIDを取得（永続Cookie対応・準備未完了でも試行）"""
        try:
            # Cookie管理の初期化（初回のみ）
            self._ensure_cookie_manager()
            
            if self.cookies is None:
                logger.debug("Cookie管理システム無効 - None返却")
                return None
            
            # Cookie準備状態をチェックするが、準備未完了でも読み込みを試行
            is_ready = self.cookies.ready()
            if not is_ready:
                logger.debug("Cookie準備未完了だが、既存Cookie読み込みを試行")
            else:
                logger.debug("Cookie準備完了 - 通常読み込み")
            
            # 既存Cookieの読み込みを試行（準備状態に関係なく）
            try:
                # 全てのCookieを確認（デバッグ用）
                all_cookies = dict(self.cookies)
                logger.debug(f"利用可能なCookie: {list(all_cookies.keys())} (準備状態: {is_ready})")
                
                user_id = self.cookies.get(self.cookie_name)
                if user_id:
                    logger.debug(f"Cookie値取得: {user_id[:8] if len(user_id) >= 8 else user_id}...")
                    
                    if self._is_valid_uuid(user_id):
                        logger.info(f"HF Spaces永続CookieからユーザーID取得成功: {user_id[:8]}... (キー: {self.cookie_name}, 準備状態: {is_ready})")
                        return user_id
                    else:
                        logger.warning(f"Cookie内のユーザーIDが無効な形式: {user_id}")
                        return None
                
                logger.debug(f"Cookie内に有効なユーザーIDなし (準備状態: {is_ready})")
                return None
                
            except Exception as cookie_read_e:
                logger.warning(f"Cookie読み込み試行エラー: {cookie_read_e}")
                return None
            
        except Exception as e:
            logger.warning(f"Cookie取得エラー: {e}")
            return None
    
    def _set_user_id_cookie(self, user_id: str):
        """ユーザーIDを永続的なクッキーに設定（準備未完了でも試行）"""
        try:
            # Cookie管理システムが初期化されていない場合はスキップ
            if not self._cookie_initialized:
                logger.debug("Cookie管理システム未初期化 - Cookie設定スキップ")
                return
            
            if self.cookies is None:
                logger.debug("Cookie管理システム無効 - Cookie設定スキップ")
                return
            
            # Cookie準備状態をチェックするが、準備未完了でも設定を試行
            is_ready = self.cookies.ready()
            if not is_ready:
                logger.debug("Cookie準備未完了だが、Cookie設定を試行")
            
            # 永続的なCookieを設定
            self.cookies[self.cookie_name] = user_id
            
            # Cookieを保存（streamlit-cookies-managerは自動的に永続化される）
            self.cookies.save()
            
            logger.info(f"HF Spaces永続クッキーに保存: {user_id[:8]}... (キー: {self.cookie_name}, 準備状態: {is_ready})")
            
        except Exception as e:
            logger.warning(f"永続Cookie設定エラー: {e}")
            # エラーが発生してもアプリケーションは継続
    
    def _is_valid_uuid(self, uuid_string: str) -> bool:
        """UUIDの形式チェック"""
        try:
            uuid.UUID(uuid_string, version=4)
            return True
        except (ValueError, TypeError):
            return False
    
    def _is_valid_user_id(self, user_id: str) -> bool:
        """ユーザーIDの有効性チェック"""
        try:
            if not self._is_valid_uuid(user_id):
                return False
            
            user_file = os.path.join(self.user_data_dir, f"{user_id}.json")
            return os.path.exists(user_file)
            
        except Exception as e:
            logger.warning(f"ユーザーID検証エラー: {e}")
            return False
    
    def _create_new_user_with_cookie(self) -> str:
        """Cookie認証ベースの新規ユーザーを作成"""
        try:
            user_id = str(uuid.uuid4())
            
            # ユーザーデータを作成
            user_data = {
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "last_access": datetime.now().isoformat(),
                "version": "1.0",
                "browser_fingerprint": self._generate_browser_fingerprint(),
                "game_data": {
                    "affection": 30,
                    "messages": [{"role": "assistant", "content": "何の用？遊びに来たの？", "is_initial": True}],
                    "scene_params": {"theme": "default"},
                    "ura_mode": False
                },
                "settings": {
                    "notifications_enabled": True,
                    "auto_save": True
                }
            }
            
            # ファイルに保存
            user_file = os.path.join(self.user_data_dir, f"{user_id}.json")
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)
            
            # Cookieに保存（ブラウザごとに一つ）
            self._set_user_id_cookie(user_id)
            
            # セッション状態に保存
            st.session_state.user_id = user_id
            
            logger.info(f"Cookie認証ベース新規ユーザー作成完了: {user_id[:8]}...")
            return user_id
            
        except Exception as e:
            logger.error(f"Cookie認証ベース新規ユーザー作成エラー: {e}")
            # フォールバック: 一時的なIDを生成（Cookieに保存しない）
            temp_id = str(uuid.uuid4())
            st.session_state.user_id = temp_id
            logger.warning(f"フォールバック一時ID: {temp_id[:8]}...")
            return temp_id
    
    def _create_new_user(self) -> str:
        """従来の新規ユーザー作成（後方互換性のため残す）"""
        return self._create_new_user_with_cookie()
    
    def _generate_browser_fingerprint(self) -> str:
        """ブラウザフィンガープリントを生成（簡易版）"""
        try:
            # Streamlitのセッション情報を使用してフィンガープリントを生成
            import hashlib
            
            # セッション固有の情報を組み合わせ
            session_info = f"{id(st.session_state)}_{datetime.now().strftime('%Y%m%d')}"
            fingerprint = hashlib.md5(session_info.encode()).hexdigest()[:16]
            
            logger.debug(f"ブラウザフィンガープリント生成: {fingerprint}")
            return fingerprint
            
        except Exception as e:
            logger.warning(f"ブラウザフィンガープリント生成エラー: {e}")
            return "unknown_browser"
    
    def _update_user_access_time(self, user_id: str):
        """ユーザーの最終アクセス時刻を更新"""
        try:
            user_file = os.path.join(self.user_data_dir, f"{user_id}.json")
            
            if os.path.exists(user_file):
                with open(user_file, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)
                
                user_data["last_access"] = datetime.now().isoformat()
                
                with open(user_file, 'w', encoding='utf-8') as f:
                    json.dump(user_data, f, ensure_ascii=False, indent=2)
                
                logger.debug(f"ユーザーアクセス時刻更新: {user_id[:8]}...")
            
        except Exception as e:
            logger.warning(f"アクセス時刻更新エラー: {e}")
    
    def load_user_game_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        ユーザーのゲームデータを読み込み
        
        Args:
            user_id: ユーザーID
            
        Returns:
            ゲームデータ（存在しない場合はNone）
        """
        try:
            user_file = os.path.join(self.user_data_dir, f"{user_id}.json")
            
            if not os.path.exists(user_file):
                logger.info(f"ユーザーファイルが存在しません: {user_id[:8]}...")
                return None
            
            with open(user_file, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
            
            game_data = user_data.get("game_data", {})
            logger.info(f"ゲームデータ読み込み完了: {user_id[:8]}... (データサイズ: {len(str(game_data))}文字)")
            
            return game_data
            
        except Exception as e:
            logger.error(f"ゲームデータ読み込みエラー: {e}")
            return None
    
    def save_user_game_data(self, user_id: str, game_data: Dict[str, Any]) -> bool:
        """
        ユーザーのゲームデータを永続ストレージ(/mnt/data)に保存
        
        Args:
            user_id: ユーザーID
            game_data: 保存するゲームデータ
            
        Returns:
            保存成功時True
        """
        try:
            user_file = os.path.join(self.user_data_dir, f"{user_id}.json")
            
            # 保存先パスをログ出力（永続ストレージ確認用）
            logger.info(f"ゲームデータ保存先: {user_file}")
            logger.info(f"永続ストレージベースパス: {self.storage_base_path}")
            
            # 既存データを読み込み
            if os.path.exists(user_file):
                with open(user_file, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)
                logger.info(f"既存ユーザーデータを更新: {user_id[:8]}...")
            else:
                # 新規ユーザーデータを作成
                user_data = {
                    "user_id": user_id,
                    "created_at": datetime.now().isoformat(),
                    "version": "1.0"
                }
                logger.info(f"新規ユーザーデータを作成: {user_id[:8]}...")
            
            # ゲームデータを更新
            user_data["game_data"] = game_data
            user_data["last_access"] = datetime.now().isoformat()
            user_data["last_save"] = datetime.now().isoformat()
            
            # ファイルに保存
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)
            
            # 保存後のファイルサイズを確認
            file_size = os.path.getsize(user_file)
            logger.info(f"永続ストレージにゲームデータ保存完了: {user_id[:8]}... (ファイルサイズ: {file_size}バイト)")
            
            return True
            
        except Exception as e:
            logger.error(f"永続ストレージへのゲームデータ保存エラー: {e}")
            return False
    
    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        ユーザー情報を取得
        
        Args:
            user_id: ユーザーID
            
        Returns:
            ユーザー情報
        """
        try:
            user_file = os.path.join(self.user_data_dir, f"{user_id}.json")
            
            if os.path.exists(user_file):
                with open(user_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            return None
            
        except Exception as e:
            logger.error(f"ユーザー情報取得エラー: {e}")
            return None
    
    def delete_user_data(self, user_id: str) -> bool:
        """
        ユーザーデータを削除
        
        Args:
            user_id: 削除するユーザーID
            
        Returns:
            削除成功時True
        """
        try:
            user_file = os.path.join(self.user_data_dir, f"{user_id}.json")
            
            if os.path.exists(user_file):
                os.remove(user_file)
                logger.info(f"ユーザーデータ削除: {user_id[:8]}...")
            
            # 永続Cookieも削除
            try:
                # Cookie管理の強制初期化（フルリセット時）
                self._ensure_cookie_manager(force_init=True)
                
                if self.cookies:
                    if self.cookie_name in self.cookies:
                        # 永続Cookieを削除（有効期限を過去に設定）
                        del self.cookies[self.cookie_name]
                        self.cookies.save()
                        logger.info(f"HF Spaces永続Cookie削除完了 (キー: {self.cookie_name})")
            except Exception as e:
                logger.warning(f"HF Spaces永続Cookie削除エラー: {e}")
            
            # セッション状態からも削除
            if 'persistent_user_id' in st.session_state:
                del st.session_state.persistent_user_id
            if 'persistent_user_id_checked' in st.session_state:
                del st.session_state.persistent_user_id_checked
            
            return True
            
        except Exception as e:
            logger.error(f"ユーザーデータ削除エラー: {e}")
            return False
    
    def full_reset_user_session(self) -> str:
        """
        フルリセット（Cookie初期化を含む）
        
        Returns:
            新しいユーザーID
        """
        try:
            logger.info("フルリセット開始 - Cookie初期化を含む")
            
            # セッション状態をクリア
            if 'persistent_user_id' in st.session_state:
                del st.session_state.persistent_user_id
            if 'persistent_user_id_checked' in st.session_state:
                del st.session_state.persistent_user_id_checked
            if 'cookie_manager_initialized' in st.session_state:
                del st.session_state.cookie_manager_initialized
            
            # Cookie初期化フラグをリセット
            self._cookie_initialized = False
            self.cookies = None
            
            # 新しいユーザーIDを作成（強制リセット）
            new_user_id = self.get_or_create_user_id(force_reset=True)
            
            logger.info(f"フルリセット完了: {new_user_id[:8]}...")
            return new_user_id
            
        except Exception as e:
            logger.error(f"フルリセットエラー: {e}")
            # フォールバック
            import uuid
            fallback_id = str(uuid.uuid4())
            st.session_state.persistent_user_id = fallback_id
            return fallback_id
    
    def list_all_users(self) -> List[Dict[str, Any]]:
        """
        全ユーザーの一覧を取得（管理用）
        
        Returns:
            ユーザー情報のリスト
        """
        try:
            users = []
            
            for filename in os.listdir(self.user_data_dir):
                if filename.endswith('.json'):
                    user_file = os.path.join(self.user_data_dir, filename)
                    
                    try:
                        with open(user_file, 'r', encoding='utf-8') as f:
                            user_data = json.load(f)
                        
                        # 基本情報のみ抽出
                        user_info = {
                            "user_id": user_data.get("user_id", "unknown")[:8] + "...",
                            "created_at": user_data.get("created_at", "unknown"),
                            "last_access": user_data.get("last_access", "unknown"),
                            "has_game_data": "game_data" in user_data,
                            "file_size": os.path.getsize(user_file)
                        }
                        
                        users.append(user_info)
                        
                    except Exception as e:
                        logger.warning(f"ユーザーファイル読み込みエラー {filename}: {e}")
            
            return users
            
        except Exception as e:
            logger.error(f"ユーザー一覧取得エラー: {e}")
            return []
    
    def cleanup_old_users(self, days_threshold: int = 30) -> int:
        """
        古いユーザーデータをクリーンアップ
        
        Args:
            days_threshold: 削除対象の日数閾値
            
        Returns:
            削除されたユーザー数
        """
        try:
            current_time = datetime.now()
            deleted_count = 0
            
            for filename in os.listdir(self.user_data_dir):
                if filename.endswith('.json'):
                    user_file = os.path.join(self.user_data_dir, filename)
                    
                    try:
                        with open(user_file, 'r', encoding='utf-8') as f:
                            user_data = json.load(f)
                        
                        last_access = datetime.fromisoformat(user_data.get("last_access", ""))
                        if (current_time - last_access).days > days_threshold:
                            os.remove(user_file)
                            deleted_count += 1
                            logger.info(f"古いユーザーデータ削除: {filename}")
                    
                    except Exception as e:
                        logger.warning(f"クリーンアップ処理エラー {filename}: {e}")
            
            logger.info(f"ユーザーデータクリーンアップ完了: {deleted_count}件削除")
            return deleted_count
            
        except Exception as e:
            logger.error(f"クリーンアップエラー: {e}")
            return 0
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        ストレージ使用状況を取得
        
        Returns:
            ストレージ統計情報
        """
        try:
            stats = {
                "user_count": 0,
                "total_size": 0,
                "storage_path": self.user_data_dir,
                "cookie_enabled": self.cookies is not None
            }
            
            if os.path.exists(self.user_data_dir):
                for filename in os.listdir(self.user_data_dir):
                    if filename.endswith('.json'):
                        user_file = os.path.join(self.user_data_dir, filename)
                        stats["user_count"] += 1
                        stats["total_size"] += os.path.getsize(user_file)
            
            return stats
            
        except Exception as e:
            logger.error(f"ストレージ統計取得エラー: {e}")
            return {"error": str(e)}