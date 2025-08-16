
"""
麻理チャット＆手紙生成 統合アプリケーション
"""
import streamlit as st
import logging
import os
import asyncio
import sys
import time
import streamlit.components.v1 as components
from huggingface_hub import HfApi, whoami
from datetime import datetime
from dotenv import load_dotenv
from contextlib import contextmanager

# --- HFユーザーIDによる永続ストレージ管理 ---
import json
import uuid

st.set_page_config(
        page_title="麻理プロジェクト", page_icon="🤖",
        layout="centered", initial_sidebar_state="auto"
    )


# 永続化用ディレクトリ
USER_DATA_DIR = "/mnt/data/mari_users"
os.makedirs(USER_DATA_DIR, exist_ok=True)

# --- 1. HF トークン取得 ---
# secrets.toml に HF_TOKEN="your_token_here" として保存しておく
hf_token = st.secrets.get("HF_TOKEN")

# --- 2. ユーザー ID 判定 ---
if hf_token:
    try:
        api = HfApi()
        user_info = whoami(token=hf_token)
        user_id = user_info["id"]
        st.session_state.is_hf_user = True
    except Exception as e:
        st.warning(f"HF API エラー: {e}. 一時 UUID を使用します。")
        if "temp_user_id" not in st.session_state:
            st.session_state.temp_user_id = str(uuid.uuid4())
        user_id = st.session_state.temp_user_id
        st.session_state.is_hf_user = False
else:
    # トークンなし → UUID フォールバック
    if "temp_user_id" not in st.session_state:
        st.session_state.temp_user_id = str(uuid.uuid4())
    user_id = st.session_state.temp_user_id
    st.session_state.is_hf_user = False

user_file = os.path.join(USER_DATA_DIR, f"{user_id}.json")
st.write(f"ユーザー ID: {user_id} (HF ログイン: {st.session_state.is_hf_user})")

# --- 基本設定 ---
# 非同期処理の問題を解決 (Windows向け)
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# .envファイルから環境変数を読み込み
load_dotenv()

# ロガー設定（PyTorch初期化より前に設定）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PyTorchのクラス登録問題を回避するための設定
try:
    import torch
    # スレッド数を制限してクラス登録問題を回避
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
    # 警告を抑制
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    logger.info("PyTorch初期化設定完了")
except ImportError:
    logger.info("PyTorchが利用できません - ルールベース処理を使用")
except Exception as e:
    logger.warning(f"PyTorch初期化設定エラー: {e}")

# --- セッション管理サーバー自動起動 ---
def start_session_server():
    """
    セッション管理サーバーを自動起動する
    Hugging Face Spacesでの実行時に必要
    """
    import subprocess
    import threading
    import requests
    import time
    
    def check_server_running(timeout=2):
        """サーバーが起動しているかチェック"""
        # Hugging Face Spacesでの実行を考慮してホストを動的に決定
        hosts_to_try = ["127.0.0.1", "localhost", "0.0.0.0"]
        port = 8000
        
        for host in hosts_to_try:
            try:
                response = requests.get(f"http://{host}:{port}/health", timeout=timeout)
                if response.status_code == 200:
                    logger.debug(f"サーバー接続成功: {host}:{port}")
                    return True
            except Exception:
                continue
        return False
    
    def run_server():
        """バックグラウンドでサーバーを起動"""
        try:
            logger.info("セッション管理サーバーをバックグラウンドで起動中...")
            
            # uvicornでサーバー起動（Hugging Face Spaces対応）
            import uvicorn
            
            # 実行環境に応じてホストを決定
            is_spaces = os.getenv("SPACE_ID") is not None
            host = "0.0.0.0" if is_spaces else "127.0.0.1"
            
            uvicorn.run(
                "session_api_server:app",
                host=host,
                port=8000,
                log_level="warning",  # ログレベルを下げてStreamlitログと混在を防ぐ
                access_log=False      # アクセスログを無効化
            )
        except Exception as e:
            logger.error(f"サーバー起動エラー: {e}")
    
    # 既にサーバーが起動しているかチェック
    if check_server_running():
        logger.info("✅ セッション管理サーバーは既に起動しています")
        return True
    
    try:
        # バックグラウンドでサーバー起動
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # サーバー起動待機（最大15秒、Hugging Face Spacesでは時間がかかる場合がある）
        max_wait = 15
        for i in range(max_wait):
            time.sleep(1)
            if check_server_running():
                logger.info(f"✅ セッション管理サーバー起動成功 ({i+1}秒)")
                return True
            if i < max_wait - 1:  # 最後の試行以外でログ出力
                logger.info(f"サーバー起動待機中... ({i+1}/{max_wait})")
        
        logger.warning("⚠️ セッション管理サーバー起動タイムアウト - フォールバックモードで継続")
        return False
        
    except Exception as e:
        logger.error(f"サーバー起動処理エラー: {e}")
        return False

# アプリケーション起動時にサーバーを自動起動
if 'server_started' not in st.session_state:
    st.session_state.server_started = start_session_server()
    if st.session_state.server_started:
        logger.info("🚀 セッション管理サーバー起動完了")
    else:
        logger.warning("⚠️ セッション管理サーバー起動失敗 - フォールバックモードで動作")


# --- 必要なモジュールのインポート ---

# << 麻理チャット用モジュール >>
from core_dialogue import DialogueGenerator
from core_sentiment import SentimentAnalyzer
from core_rate_limiter import RateLimiter
from core_scene_manager import SceneManager  # 復元したモジュール
from core_memory_manager import MemoryManager
from components_chat_interface import ChatInterface
from components_status_display import StatusDisplay
from components_dog_assistant import DogAssistant
from components_tutorial import TutorialManager
from session_manager import SessionManager, get_session_manager, validate_session_state, perform_detailed_session_validation
from session_api_client import SessionAPIClient
from user_id_manager import UserIDManager  # ユーザーID永続化管理
## persistent_user_manager importは廃止
# << 手紙生成用モジュール >>
from letter_config import Config
from letter_logger import setup_logger as setup_letter_logger
from letter_generator import LetterGenerator
from letter_request_manager import RequestManager
from letter_user_manager import UserManager
from async_storage_manager import AsyncStorageManager
from async_rate_limiter import AsyncRateLimitManager

# --- 定数 ---
MAX_INPUT_LENGTH = 200
MAX_HISTORY_TURNS = 50

def get_event_loop():
    """
    セッションごとに単一のイベントループを取得または作成する
    """
    try:
        # 既に実行中のループがあればそれを返す
        return asyncio.get_running_loop()
    except RuntimeError:
        # 実行中のループがなければ、新しく作成
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

def run_async(coro):
    """
    Streamlitのセッションで共有されたイベントループを使って非同期関数を実行する
    """
    try:
        # 既存のループがあるかチェック
        loop = asyncio.get_running_loop()
        # 既存のループがある場合は、新しいタスクとして実行
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # 実行中のループがない場合は、新しいループで実行
        return asyncio.run(coro)

# --- ▼▼▼ 1. 初期化処理の一元管理 ▼▼▼ ---

@st.cache_resource
def initialize_cached_managers():
    """
    キャッシュ可能な管理クラスのみを初期化する
    Streamlitコンポーネントを使用しないクラスのみ
    
    Note: 新しいStreamlitキャッシュシステムを使用
    - st.cache_resource: オブジェクト/リソース用（データベース接続、モデルなど）
    - st.cache_data: データ用（DataFrame、辞書、リストなど）
    - 旧st.cacheは非推奨
    """
    logger.info("Initializing cached managers...")
    # --- 手紙機能の依存モジュール ---
    letter_storage = AsyncStorageManager(Config.STORAGE_PATH)
    letter_rate_limiter = AsyncRateLimitManager(letter_storage, max_requests=Config.MAX_DAILY_REQUESTS)
    user_manager = UserManager(letter_storage)
    letter_request_manager = RequestManager(letter_storage, letter_rate_limiter)
    letter_generator = LetterGenerator()

    request_manager = RequestManager(letter_storage, letter_rate_limiter)

    # --- チャット機能の依存モジュール ---
    dialogue_generator = DialogueGenerator()
    sentiment_analyzer = SentimentAnalyzer()
    rate_limiter = RateLimiter()
    scene_manager = SceneManager()
    # memory_manager は セッション単位で作成するため、ここでは作成しない
    chat_interface = ChatInterface(max_input_length=MAX_INPUT_LENGTH)
    status_display = StatusDisplay()
    dog_assistant = DogAssistant()
    tutorial_manager = TutorialManager()
    session_api_client = SessionAPIClient()
    user_id_manager = UserIDManager()  # ユーザーID永続化管理

    logger.info("Cached managers initialized.")
    return {
        # 手紙用
        "user_manager": user_manager,
        "request_manager": request_manager,
        "letter_generator": letter_generator,
        # チャット用
        "dialogue_generator": dialogue_generator,
        "sentiment_analyzer": sentiment_analyzer,
        "rate_limiter": rate_limiter,
        "scene_manager": scene_manager,
        # memory_manager は セッション単位で作成
        "chat_interface": chat_interface,
        "status_display": status_display,
        "dog_assistant": dog_assistant,
        "tutorial_manager": tutorial_manager,
        "session_api_client": session_api_client,
        "user_id_manager": user_id_manager,  # ユーザーID永続化管理
    }

def initialize_session_managers():
    """
    セッション単位で初期化が必要な管理クラスを初期化する
    Streamlitコンポーネントを使用するクラス
    """
    # セッション状態でマネージャーをキャッシュ（重複初期化防止）
    if 'session_managers_initialized' not in st.session_state:
        logger.info("Initializing session managers...")
        st.session_state.session_managers = {}  # persistent_user_managerは廃止
        st.session_state.session_managers_initialized = True
        logger.info("Session managers initialized.")
    else:
        logger.debug("Session managers already initialized - using cached version")
    return st.session_state.session_managers

# initialize_all_managers()は削除 - main()で直接初期化

def initialize_session_state(managers, force_reset_override=False):
    """
    アプリケーション全体のセッションステートを初期化する
    SessionManagerを使用してセッション分離を強化
    
    Args:
        managers: 管理クラスの辞書
        force_reset_override: 強制リセットフラグ（フルリセット時に使用）
    """

      # 基本的なセッション状態を最初に初期化（Cookie処理より前）
    logger.info("🚀 基本セッション状態の早期初期化開始")
    
    if 'chat_initialized' not in st.session_state:
        st.session_state.chat_initialized = False
        logger.info("✅ chat_initialized フラグを初期化（Cookie処理前）")
    
    if 'memory_notifications' not in st.session_state:
        st.session_state.memory_notifications = []
        logger.info("✅ memory_notifications を初期化")
    
    if 'affection_notifications' not in st.session_state:
        st.session_state.affection_notifications = []
        logger.info("✅ affection_notifications を初期化")
    
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
        logger.info(f"✅ debug_mode を初期化: {st.session_state.debug_mode}")
    
    logger.info("✅ 基本セッション状態の早期初期化完了")

    # 強制リセットフラグ（開発時用または明示的な指定）
    force_reset = force_reset_override or os.getenv("FORCE_SESSION_RESET", "false").lower() == "true"
    
    # 関数開始をログ出力（実行確認用）
    logger.info("🚀 initialize_session_state 開始（基本状態は既に初期化済み）")
    
    # 基本セッション状態は既にmain()で初期化済み
    logger.info(f"✅ chat_initialized 現在値: {st.session_state.chat_initialized}")
    
    if force_reset:
        st.session_state.chat_initialized = False
        st.session_state.memory_notifications = []
        st.session_state.affection_notifications = []
        logger.info("🔄 強制リセット: 基本セッション状態をリセット")
    
    # 初回起動時はセッション検証をスキップ
    is_first_run = 'user_id' not in st.session_state
    
    # SessionManagerの初期化（セッション分離強化）
    session_manager = get_session_manager()
    
    # 初回起動時以外でセッション整合性チェックを実行
    if not is_first_run and not validate_session_state():
        logger.error("Session validation failed during initialization")
        # 復旧に失敗した場合は強制リセット
        force_reset = True
    
    # HF Spaces永続ストレージ対応ユーザー管理システムを使用
    # persistent_user_manager 完全廃止
    user_id_manager = managers["user_id_manager"]  # フォールバック用
    session_api_client = managers["session_api_client"]
    
    # Cookie認証ベースのユーザーID取得（起動時uuid4()生成を排除）
    # 重複ID生成防止: 既にuser_idがあり、force_resetでない場合は既存IDを使用
    if 'user_id' in st.session_state and not force_reset:
        session_id = st.session_state.user_id
        logger.debug(f"既存ユーザーID使用（rerun対応）: {session_id[:8]}...")
    else:
        # Cookie認証ベースでユーザーIDを取得（新規作成またはリセット時のみ）
        # persistent_user_manager 完全廃止
        try:
            session_id = user_id_manager.get_or_create_user_id()
            logger.info(f"ユーザーID取得: {session_id[:8]}...")
        except Exception as e2:
            logger.error(f"ユーザーID取得失敗: {e2}")
            session_id = f"temp_{id(st.session_state)}"
            logger.warning(f"最終フォールバック: セッション固有一時ID: {session_id}")
    
    # ユーザーIDとしてセッションIDを使用
    # 安全にuser_idの存在をチェック
    current_user_id = getattr(st.session_state, 'user_id', None)
    session_changed = (current_user_id is None or 
                      current_user_id != session_id or 
                      force_reset)
    
    if session_changed:
        st.session_state.user_id = session_id
        
        # SessionManagerにユーザーIDを設定
        session_manager.set_user_id(session_id)
        
        # セッション情報をログ出力
        from datetime import datetime  # ローカルインポートで確実に利用可能にする
        session_info = {
            "user_id": session_id[:8] + "...",  # session_idを直接使用（既に設定済み）
            "session_id": id(st.session_state),
            "force_reset": force_reset,
            "session_changed": session_changed,
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"FastAPIセッション管理でユーザーセッション設定: {session_info}")
        
        # セッション固有の識別子を保存
        st.session_state._session_id = id(st.session_state)
    else:
        # 既存セッションの場合もSessionManagerにユーザーIDを設定
        if session_manager.user_id != session_id:
            session_manager.set_user_id(session_id)
        
        logger.debug(f"既存セッション継続使用: {st.session_state.user_id[:8]}...")

    # chat_initialized フラグは既に初期化済み（Cookie処理前に実行済み）

    # チャット初期化が必要かどうかを判定
    needs_chat_initialization = (
        not st.session_state.chat_initialized or
        'chat' not in st.session_state or
        force_reset
    )

    if needs_chat_initialization:
        # 保存されたゲームデータを読み込み（重複防止・統一化）
        saved_game_data = None
        
        # 永続ストレージを優先、失敗時のみフォールバック
        # persistent_user_manager 完全廃止
        try:
            saved_game_data = user_id_manager.load_game_data(st.session_state.user_id)
            if saved_game_data:
                logger.info(f"ユーザーファイルからゲームデータを読み込み: {st.session_state.user_id[:8]}...")
            else:
                logger.debug("ユーザーファイルにもゲームデータなし")
        except Exception as e2:
            logger.error(f"ゲームデータ読み込み失敗: {e2}")
            saved_game_data = None
        
        if saved_game_data and not force_reset:
            # 保存データから復元
            logger.info(f"保存されたゲームデータを復元: {st.session_state.user_id[:8]}...")
            
            initial_message = "何の用？遊びに来たの？"
            st.session_state.chat = {
                "messages": saved_game_data.get("messages", [{"role": "assistant", "content": initial_message, "is_initial": True}]),
                "affection": saved_game_data.get("affection", 30),
                "scene_params": saved_game_data.get("scene_params", {"theme": "default"}),
                "limiter_state": managers["rate_limiter"].create_limiter_state(),
                "scene_change_pending": None,
                "ura_mode": saved_game_data.get("ura_mode", False)
            }
            st.session_state.memory_notifications = []
            st.session_state.affection_notifications = []
            st.session_state.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
            
            # メモリマネージャーも復元
            st.session_state.memory_manager = MemoryManager(history_threshold=10)
            if "memory_data" in saved_game_data:
                # メモリデータがあれば復元（実装は後で）
                pass
                
            logger.info(f"ゲームデータ復元完了 - 好感度: {st.session_state.chat['affection']}, メッセージ数: {len(st.session_state.chat['messages'])}")
        else:
            # 新規初期化
            initial_message = "何の用？遊びに来たの？"
            st.session_state.chat = {
                "messages": [{"role": "assistant", "content": initial_message, "is_initial": True}],
                "affection": 30,
                "scene_params": {"theme": "default"},
                "limiter_state": managers["rate_limiter"].create_limiter_state(),
                "scene_change_pending": None,
                "ura_mode": False
            }
            st.session_state.memory_notifications = []
            st.session_state.affection_notifications = []
            st.session_state.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
            st.session_state.memory_manager = MemoryManager(history_threshold=10)
            logger.info(f"新規チャット初期化完了 - 初期メッセージ: '{initial_message}'")
        
        st.session_state.rerun_count = 0  # rerun回数カウンター初期化
        st.session_state.chat_initialized = True
        logger.info("Chat session state initialized with SessionManager.")
    else:
        # 既存セッション継続（ループ防止のため状態変更を最小限に）
        logger.debug("チャット既に初期化済み - 状態チェックのみ実行")
        
        # 重要: 初期メッセージの復元は特別な場合のみ実行
        restore_needed = False
        
        # チュートリアル保護フラグがある場合のみ初期メッセージを復元
        if st.session_state.get('preserve_initial_message', False):
            if 'chat' in st.session_state and 'messages' in st.session_state.chat:
                initial_message = "何の用？遊びに来たの？"
                initial_msg = {"role": "assistant", "content": initial_message, "is_initial": True}
                st.session_state.chat['messages'].insert(0, initial_msg)
                logger.info("初期メッセージを復元しました（チュートリアル保護）")
                # 保護フラグをクリア
                st.session_state.preserve_initial_message = False
                restore_needed = True
        
        # 完全に空のメッセージリストの場合のみ復元（異常状態の修復）
        elif ('chat' in st.session_state and 
              'messages' in st.session_state.chat and 
              len(st.session_state.chat['messages']) == 0):
            initial_message = "何の用？遊びに来たの？"
            initial_msg = {"role": "assistant", "content": initial_message, "is_initial": True}
            st.session_state.chat['messages'] = [initial_msg]
            logger.warning("異常状態検出: 空のメッセージリストを修復しました")
            restore_needed = True
        
        # 状態変更があった場合のみログ出力
        if not restore_needed:
            # 完全に何も変更しない（ループ防止）
            pass

    if force_reset:
        logger.info("Session force reset - all data cleared")


    # セッション状態の補完（初期化時のみ実行してループを防止）
    if not st.session_state.get('_state_supplemented', False):
        state_updated = False
        
        # MemoryManagerがセッション状態にない場合は作成
        if 'memory_manager' not in st.session_state:
            st.session_state.memory_manager = MemoryManager(history_threshold=10)
            state_updated = True
        
        # 通知用リストは既に早期初期化済み（重複初期化を回避）
        
        # 裏モードフラグが存在しない場合は作成（chatが存在する場合のみ）
        if 'chat' in st.session_state and 'ura_mode' not in st.session_state.chat:
            st.session_state.chat['ura_mode'] = False
            state_updated = True
        
        # セッション整合性チェック（初回のみ）
        if not st.session_state.get('_initialization_complete', False):
            if not session_manager.validate_session_integrity():
                logger.warning("Session integrity check failed after initialization")
                session_manager.recover_session()
            
            # 初期化完了フラグを設定（一度だけ）
            st.session_state._initialization_complete = True
            state_updated = True
        
        # 状態補完完了フラグを設定（一度だけ実行）
        st.session_state._state_supplemented = True
        
        # 状態更新があった場合のみログ出力
        if state_updated:
            logger.debug("Session state補完完了（初回のみ）")
    else:
        # 既に補完済みの場合は何もしない（ループ防止）
        logger.debug("Session state補完済み - スキップ")

    # 手紙機能用のセッションは特に追加の初期化は不要
    # (各関数内で必要なデータは都度非同期で取得するため)

def save_game_data_to_file(managers):
    """現在のゲームデータをファイルに保存（永続ストレージ対応）"""
    try:
        if 'user_id' not in st.session_state or 'chat' not in st.session_state:
            return False
            
        # persistent_user_manager 完全廃止
        user_id_manager = managers["user_id_manager"]  # フォールバック用
        
        # 保存するゲームデータを構築
        from datetime import datetime  # ローカルインポートで確実に利用可能にする
        game_data = {
            "messages": st.session_state.chat.get("messages", []),
            "affection": st.session_state.chat.get("affection", 30),
            "scene_params": st.session_state.chat.get("scene_params", {"theme": "default"}),
            "ura_mode": st.session_state.chat.get("ura_mode", False),
            "saved_at": datetime.now().isoformat()
        }
        
        # メモリマネージャーのデータも保存（可能であれば）
        if hasattr(st.session_state, 'memory_manager'):
            try:
                # メモリデータを辞書形式で保存
                memory_data = {
                    "important_memories": getattr(st.session_state.memory_manager, 'important_memories', []),
                    "conversation_summary": getattr(st.session_state.memory_manager, 'conversation_summary', "")
                }
                game_data["memory_data"] = memory_data
            except Exception as e:
                logger.warning(f"メモリデータ保存エラー: {e}")
        
        # 永続ストレージに保存を試行
        success = False
        # persistent_user_manager 完全廃止
        success = user_id_manager.save_game_data(st.session_state.user_id, game_data)
        if success:
            logger.info(f"ユーザーファイルにゲームデータ保存成功")
        
        return success
        
    except Exception as e:
        logger.error(f"ゲームデータ保存エラー: {e}")
        return False


# --- ▼▼▼ 2. UIコンポーネントの関数化 ▼▼▼ ---

@st.cache_data
def load_css_content(file_path: str) -> str:
    """
    CSSファイルの内容をキャッシュして読み込む
    
    Note: Streamlit 4.44.1対応
    - st.cache_data: データ（文字列、辞書など）のキャッシュに使用
    - 旧st.cacheは非推奨のため新しいAPIを使用
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"CSSファイルが見つかりません: {file_path}")
        return ""
    except Exception as e:
        logger.error(f"CSS読み込みエラー: {e}")
        return ""

def inject_custom_css(file_path="streamlit_styles.css"):
    # キャッシュされたCSS読み込み
    css_content = load_css_content(file_path)
    if css_content:
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        logger.info(f"CSSファイルを読み込みました: {file_path}")
    else:
        # フォールバック用の基本スタイルを適用
        apply_fallback_css()
    
   

def apply_fallback_css():
    """フォールバック用の基本CSSを適用"""
    fallback_css = """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }
    .stApp > div:first-child {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(5px);
        min-height: 100vh;
    }
    .stChatMessage {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
        margin: 8px 0 !important;
    }
    </style>
    """
    st.markdown(fallback_css, unsafe_allow_html=True)
    logger.info("フォールバック用CSSを適用しました")



def show_memory_notification(message: str):
    """特別な記憶の通知をポップアップ風に表示する"""
    notification_css = """
    <style>
    .memory-notification {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 12px;
        border: 2px solid #ffffff40;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        margin: 10px 0;
        animation: slideIn 0.5s ease-out;
        text-align: center;
        font-weight: 500;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .memory-notification .icon {
        font-size: 1.2em;
        margin-right: 8px;
    }
    </style>
    """
    
    notification_html = f"""
    <div class="memory-notification">
        <span class="icon">🧠✨</span>
        {message}
    </div>
    """
    
    st.markdown(notification_css + notification_html, unsafe_allow_html=True)

def check_affection_milestone(old_affection: int, new_affection: int) -> str:
    """好感度のマイルストーンに到達したかチェックする"""
    milestones = {
        40: "🌸 麻理があなたに心を開き始めました！手紙をリクエストできるようになりました。",
        60: "💖 麻理があなたを信頼するようになりました！より深い会話ができるようになります。",
        80: "✨ 麻理があなたを大切な人だと思っています！特別な反応が増えるでしょう。",
        100: "🌟 麻理があなたを心から愛しています！最高の関係に到達しました！"
    }
    
    for milestone, message in milestones.items():
        if old_affection < milestone <= new_affection:
            return message
    
    return ""

def show_affection_notification(change_amount: int, change_reason: str, new_affection: int, is_milestone: bool = False):
    """好感度変化の通知を表示する（Streamlit標準コンポーネント使用）"""
    # 好感度変化がない場合は通知しない（マイルストーン以外）
    if change_amount == 0 and not is_milestone:
        return
    
    # マイルストーン通知の場合
    if is_milestone:
        st.balloons()  # 特別な演出
        st.success(f"🎉 **マイルストーン達成！** {change_reason} (現在の好感度: {new_affection}/100)")
    elif change_amount > 0:
        # 好感度上昇
        st.success(f"💕 **+{change_amount}** {change_reason} (現在の好感度: {new_affection}/100)")
    else:
        # 好感度下降
        st.info(f"💔 **{change_amount}** {change_reason} (現在の好感度: {new_affection}/100)")

def show_cute_thinking_animation():
    """かわいらしい考え中アニメーションを表示する"""
    thinking_css = """
    <style>
    .thinking-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 20px;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        border: 2px solid rgba(255, 182, 193, 0.6);
        box-shadow: 0 8px 32px rgba(255, 182, 193, 0.3);
        backdrop-filter: blur(10px);
        margin: 20px 0;
        animation: containerPulse 2s ease-in-out infinite;
    }
    
    .thinking-face {
        font-size: 3em;
        margin-bottom: 15px;
        animation: faceRotate 3s ease-in-out infinite;
        filter: drop-shadow(0 0 10px rgba(255, 105, 180, 0.5));
    }
    
    .thinking-text {
        font-size: 1.2em;
        color: #ff69b4;
        font-weight: 600;
        margin-bottom: 15px;
        animation: textGlow 1.5s ease-in-out infinite alternate;
    }
    
    .thinking-dots {
        display: flex;
        gap: 8px;
    }
    
    .thinking-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: linear-gradient(45deg, #ff69b4, #ff1493);
        animation: dotBounce 1.4s ease-in-out infinite;
    }
    
    .thinking-dot:nth-child(1) { animation-delay: 0s; }
    .thinking-dot:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes containerPulse {
        0%, 100% { transform: scale(1); box-shadow: 0 8px 32px rgba(255, 182, 193, 0.3); }
        50% { transform: scale(1.02); box-shadow: 0 12px 40px rgba(255, 182, 193, 0.5); }
    }
    
    @keyframes faceRotate {
        0%, 100% { transform: rotate(0deg); }
        25% { transform: rotate(-5deg); }
        75% { transform: rotate(5deg); }
    }
    
    @keyframes textGlow {
        0% { text-shadow: 0 0 5px rgba(255, 105, 180, 0.5); }
        100% { text-shadow: 0 0 20px rgba(255, 105, 180, 0.8), 0 0 30px rgba(255, 105, 180, 0.6); }
    }
    
    @keyframes dotBounce {
        0%, 80%, 100% { transform: translateY(0); opacity: 0.7; }
        40% { transform: translateY(-15px); opacity: 1; }
    }
    
    .sound-wave {
        display: flex;
        gap: 3px;
        margin-top: 10px;
    }
    
    .sound-bar {
        width: 4px;
        height: 20px;
        background: linear-gradient(to top, #ff69b4, #ff1493);
        border-radius: 2px;
        animation: soundWave 1s ease-in-out infinite;
    }
    
    .sound-bar:nth-child(1) { animation-delay: 0s; }
    .sound-bar:nth-child(2) { animation-delay: 0.1s; }
    .sound-bar:nth-child(3) { animation-delay: 0.2s; }
    .sound-bar:nth-child(4) { animation-delay: 0.3s; }
    .sound-bar:nth-child(5) { animation-delay: 0.4s; }
    
    @keyframes soundWave {
        0%, 100% { height: 20px; }
        50% { height: 35px; }
    }
    </style>
    """
    
    thinking_html = """
    <div class="thinking-container">
        <div class="thinking-face">🤔</div>
        <div class="thinking-text">麻理が考え中...</div>
        <div class="thinking-dots">
            <div class="thinking-dot"></div>
            <div class="thinking-dot"></div>
            <div class="thinking-dot"></div>
        </div>
        <div class="sound-wave">
            <div class="sound-bar"></div>
            <div class="sound-bar"></div>
            <div class="sound-bar"></div>
            <div class="sound-bar"></div>
            <div class="sound-bar"></div>
        </div>
        <div style="margin-top: 10px; font-size: 0.9em; color: #ff69b4; opacity: 0.8;">
            💭 あんたのために一生懸命考えてるんだから...
        </div>
    </div>
    """
    
    # 音効果のJavaScript（Web Audio APIを使用した実際の音生成）
    sound_js = """
    <script>
    // Web Audio APIを使用した音効果生成
    function playThinkingSound() {
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // 柔らかい思考音を生成
            const oscillator1 = audioContext.createOscillator();
            const oscillator2 = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            
            // 周波数設定（優しい音色）
            oscillator1.frequency.setValueAtTime(220, audioContext.currentTime); // A3
            oscillator2.frequency.setValueAtTime(330, audioContext.currentTime); // E4
            
            // 波形設定（柔らかいサイン波）
            oscillator1.type = 'sine';
            oscillator2.type = 'sine';
            
            // 音量設定（控えめに）
            gainNode.gain.setValueAtTime(0, audioContext.currentTime);
            gainNode.gain.linearRampToValueAtTime(0.1, audioContext.currentTime + 0.1);
            gainNode.gain.linearRampToValueAtTime(0.05, audioContext.currentTime + 0.5);
            gainNode.gain.linearRampToValueAtTime(0, audioContext.currentTime + 1.0);
            
            // 接続
            oscillator1.connect(gainNode);
            oscillator2.connect(gainNode);
            gainNode.connect(audioContext.destination);
            
            // 再生
            oscillator1.start(audioContext.currentTime);
            oscillator2.start(audioContext.currentTime);
            oscillator1.stop(audioContext.currentTime + 1.0);
            oscillator2.stop(audioContext.currentTime + 1.0);
            
            console.log("🎵 麻理の思考音を再生中...");
            
        } catch (error) {
            console.log("音声再生はサポートされていません:", error);
        }
    }
    
    // 視覚的な音波エフェクトの強化
    setTimeout(() => {
        const soundBars = document.querySelectorAll('.sound-bar');
        soundBars.forEach((bar, idx) => {{
            bar.style.animationDuration = (0.8 + Math.random() * 0.4) + 's';
        }});
        
        // 音効果を再生（ユーザーインタラクション後のみ）
        playThinkingSound();
    }, 100);
    
    // 定期的な音波効果
    setInterval(() => {
        const soundBars = document.querySelectorAll('.sound-bar');
        if (soundBars.length > 0) {
            soundBars.forEach((bar, idx) => {{
                const randomHeight = 15 + Math.random() * 25;
                bar.style.height = randomHeight + 'px';
            }});
        }
    }, 200);
    </script>
    """
    
    return st.markdown(thinking_css + thinking_html + sound_js, unsafe_allow_html=True)

@contextmanager
def cute_thinking_spinner():
    """かわいらしい考え中アニメーション付きコンテキストマネージャー"""
    # アニメーション表示用のプレースホルダー
    placeholder = st.empty()
    
    try:
        # アニメーション開始
        with placeholder.container():
            show_cute_thinking_animation()
        
        yield
        
    finally:
        # アニメーション終了
        placeholder.empty()

def render_custom_chat_history(messages, chat_interface):
    """カスタムチャット履歴表示エリア（マスク機能付き）"""
    # チュートリアルダイアログ表示中のみ処理を一時停止（チュートリアル実施中は通常通り表示）
    if st.session_state.get('tutorial_dialog_showing', False):
        logger.info("チュートリアルダイアログ表示中のため、チャット履歴表示を一時停止")
        return
    
    # チュートリアル開始/スキップ処理の瞬間のみ一時停止
    if st.session_state.get('tutorial_processing', False):
        logger.info("チュートリアル開始/スキップ処理中のため、チャット履歴表示を一時停止")
        return
    
    # デバッグ: 受け取ったメッセージの状態をログ出力
    logger.info(f"🔍 render_custom_chat_history 呼び出し: messages={len(messages) if messages else 0}件")
    
    # セッション状態から直接メッセージを取得（参照渡しの問題を回避）
    if 'chat' not in st.session_state or 'messages' not in st.session_state.chat:
        logger.error("チャットセッション状態が存在しません")
        # チュートリアル中でない場合のみエラー表示
        if not st.session_state.get('tutorial_start_requested', False) and not st.session_state.get('tutorial_skip_requested', False):
            st.error("チャットセッションが初期化されていません。ページを再読み込みしてください。")
        return
    
    # セッション状態から直接取得
    session_messages = st.session_state.chat['messages']
    logger.info(f"🔍 セッション状態のメッセージ: {len(session_messages)}件")
    
    # 初期メッセージの存在確認と復元
    initial_messages = [msg for msg in session_messages if msg.get('is_initial', False)]
    
    if not initial_messages:
        logger.warning("初期メッセージが見つかりません - 即座に復元します")
        # 初期メッセージを先頭に挿入
        initial_message = {"role": "assistant", "content": "何の用？遊びに来たの？", "is_initial": True}
        st.session_state.chat['messages'].insert(0, initial_message)
        session_messages = st.session_state.chat['messages']
        logger.info(f"初期メッセージを復元しました - 現在のメッセージ数: {len(session_messages)}")
    else:
        logger.debug(f"初期メッセージ確認: {len(initial_messages)}件 - 内容: '{initial_messages[0].get('content', '')}'")
    
    # 最終的なメッセージ数をチェック
    if not session_messages:
        logger.error("メッセージリストが依然として空です")
        # 強制的に初期メッセージを作成
        initial_message = {"role": "assistant", "content": "何の用？遊びに来たの？", "is_initial": True}
        st.session_state.chat['messages'] = [initial_message]
        session_messages = st.session_state.chat['messages']
        logger.info("強制的に初期メッセージを作成しました")
    
    logger.info(f"🎯 最終的に表示するメッセージ数: {len(session_messages)}")
    
    # 拡張されたチャットインターフェースを使用（マスク機能付き）
    chat_interface.render_chat_history(session_messages)


# === チャットタブの描画関数 ===
def render_chat_tab(managers):
    """「麻理と話す」タブのUIを描画する"""
    logger.info("🎯 render_chat_tab 開始")
    
    # 現在のタブを記録（タブ切り替え検出用）
    st.session_state.last_active_tab = "chat"
    
    # 背景更新はタブ切り替え時のみ実行（ループ防止）
    # Streamlitのタブは毎回全て実行されるため、実際にタブがアクティブかチェック
    should_update_background = (
        st.session_state.get('_last_rendered_tab') != "chat" or
        st.session_state.get('_background_needs_update', False)
    )
    
    if should_update_background:
        try:
            current_theme = st.session_state.chat['scene_params'].get("theme", "default")
            logger.info(f"チャットタブ背景更新: テーマ '{current_theme}' (タブ切り替え時)")
            update_background(managers['scene_manager'], current_theme)
            st.session_state._last_rendered_tab = "chat"
            st.session_state._background_needs_update = False
        except Exception as e:
            logger.error(f"チャットタブ背景更新でエラーが発生: {e}")
            import traceback
            logger.error(f"チャットタブ背景更新エラーの詳細: {traceback.format_exc()}")
            # エラーが発生してもアプリケーションは継続
    
    # チャットタブのメインコンテンツを描画
    render_chat_tab_content(managers)
    
# ### ▼▼▼【CSS連携版】背景画像設定関数 ▼▼▼ ###
# ### ▼▼▼【最終改善版・JS堅牢化】背景画像設定関数 ▼▼▼ ###
def update_background(scene_manager: SceneManager, theme: str):
    """
    指定されたテーマに応じてbodyにクラスを追加し、CSS変数で背景URLを渡す。
    ガード句を外し、毎回JSを実行しません！確実性を高める。
    """
    try:
        logger.info(f"背景更新（ガード句なし）開始 - テーマ: {theme}")
        css_dict = {
            "default": "background-image: url('https://huggingface.co/spaces/sirochild/mari-chat-3/resolve/main/static/ribinngu-hiru.jpg');",
            "room_night": "background-image: url('https://huggingface.co/spaces/sirochild/mari-chat-3/resolve/main/static/ribinngu-yoru-on.jpg');",
            "beach_sunset": "background-image: url('https://huggingface.co/spaces/sirochild/mari-chat-3/resolve/main/static/sunahama-hiru.jpg');",
            "festival_night": "background-image: url('https://huggingface.co/spaces/sirochild/mari-chat-3/resolve/main/static/maturi-yoru.jpg');",
            "shrine_day": "background-image: url('https://huggingface.co/spaces/sirochild/mari-chat-3/resolve/main/static/jinnjya-hiru.jpg');",
            "cafe_afternoon": "background-image: url('https://huggingface.co/spaces/sirochild/mari-chat-3/resolve/main/static/kissa-hiru.jpg');",
            "art_museum_night": "background-image: url('https://huggingface.co/spaces/sirochild/mari-chat-3/resolve/main/static/bijyutukann-yoru.jpg');",
        }
        
        # デバッグ用：使用する背景画像URLをログ出力
        selected_bg = css_dict.get(theme, css_dict['default'])
        logger.info(f"背景画像設定: {selected_bg}")
        # 一意なIDを生成してCSSの重複を防ぐ
        import time
        css_id = f"bg-{int(time.time() * 1000)}"
        
        css = f"""
        <style id="{css_id}">
        /* 背景画像を確実に適用するための強力なセレクタ - 優先度を最大化 */
        .stApp.stApp.stApp {{
            {selected_bg}
            background-size: cover !important;
            background-position: center !important;
            background-attachment: fixed !important;
            background-repeat: no-repeat !important;
            min-height: auto !important;
        }}
        
        /* フォールバック用 - より広範囲に適用 */
        .stApp > div:first-child,
        [data-testid="stAppViewContainer"] {{
            {selected_bg}
            background-size: cover !important;
            background-position: center !important;
            background-attachment: fixed !important;
            background-repeat: no-repeat !important;
            min-height: auto !important;
        }}

        /* bodyとhtmlレベルでも背景を設定（rerun対策） */
        html {{
            {selected_bg}
            background-size: cover !important;
            background-position: center !important;
            background-attachment: fixed !important;
            background-repeat: no-repeat !important;
        }}
        
        body {{
            {selected_bg}
            background-size: cover !important;
            background-position: center !important;
            background-attachment: fixed !important;
            background-repeat: no-repeat !important;
            min-height: auto !important;
        }}
        
        /* 他の要素は透過にして背景を見せる */
        [data-testid="stAppViewContainer"] > section,
        [data-testid="stAppViewContainer"] > section > div.block-container,
        .main .block-container {{
            background: transparent !important;
        }}
        </style>
        
        <script>
        // JavaScript側でも背景を強制適用（rerun対策）
        (function() {{
            const applyBackground = () => {{
                const elements = [
                    document.documentElement,
                    document.body,
                    document.querySelector('.stApp'),
                    document.querySelector('[data-testid="stAppViewContainer"]')
                ];
                
                elements.forEach(el => {{
                    if (el) {{
                        el.style.backgroundImage = '{selected_bg.split(": ")[1].replace(";", "")}';
                        el.style.backgroundSize = 'cover';
                        el.style.backgroundPosition = 'center';
                        el.style.backgroundAttachment = 'fixed';
                        el.style.backgroundRepeat = 'no-repeat';
                        el.style.minHeight = 'auto';
                    }}
                }});
            }};
            
            // 即座に適用
            applyBackground();
            
            // DOM変更時にも再適用
            if (window.MutationObserver) {{
                const observer = new MutationObserver(applyBackground);
                observer.observe(document.body, {{ childList: true, subtree: true }});
                
                // 5秒後に監視を停止（パフォーマンス対策）
                setTimeout(() => observer.disconnect(), 5000);
            }}
            
            // 定期的にも適用（最後の手段）
            setTimeout(applyBackground, 100);
            setTimeout(applyBackground, 500);
        }})();
        </script>
        """
        st.markdown(css, unsafe_allow_html=True)
            # Python側の状態管理は継続する
        st.session_state.current_background_theme = theme
        logger.info(f"背景更新完了（ガード句なし） - テーマ: {theme}")

    except Exception as e:
        logger.error(f"背景更新エラー（ガード句なし）: {e}")
        import traceback
        logger.error(f"背景更新エラーの詳細: {traceback.format_exc()}")


def render_chat_tab_content(managers):
    """チャットタブのメインコンテンツを描画する"""
    logger.info("🎯 render_chat_tab_content 開始")
    # チュートリアル機能の自動チェック
    tutorial_manager = managers['tutorial_manager']
    tutorial_manager.auto_check_completions()
    
    # チュートリアル案内をチャットタブに表示（テスト中）
    tutorial_manager.render_chat_tutorial_guide()

    # --- サイドバー ---
    with st.sidebar:
        # セーフティ機能を左サイドバーに統合
        current_mode = st.session_state.chat.get('ura_mode', False)
        safety_color = "#ff4757" if current_mode else "#2ed573"  # 赤：解除、緑：有効
        safety_text = "セーフティ解除" if current_mode else "セーフティ有効"
        safety_icon = "🔓" if current_mode else "🔒"
        
        # セーフティボタンのカスタムCSS
        safety_css = f"""
        <style>
        .safety-button {{
            background-color: {safety_color};
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 15px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            width: 100%;
            margin-bottom: 15px;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        .safety-button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }}
        </style>
        """
        st.markdown(safety_css, unsafe_allow_html=True)
        
        if st.button(f"{safety_icon} {safety_text}", type="primary" if current_mode else "secondary", 
                    help="麻理のセーフティ機能を切り替えます", use_container_width=True):
            st.session_state.chat['ura_mode'] = not current_mode
            new_mode = st.session_state.chat['ura_mode']
            
            # チュートリアルステップ3を完了
            tutorial_manager.check_step_completion(3, True)
            
            if new_mode:
                st.success("🔓 セーフティ解除モードに切り替えました！")
            else:
                st.info("🔒 セーフティ有効モードに戻しました。")
            st.rerun()

        # 好感度に応じた現在の色を計算する関数（スコープ問題回避のため外部定義）
        def get_affection_color(affection_val):
            if affection_val < 20:
                return "#0284c7"  # 青（寒色）
            elif affection_val < 40:
                # 青から緑へのグラデーション
                ratio = (affection_val - 20) / 20
                return f"color-mix(in srgb, #0284c7 {100-ratio*100}%, #16a34a {ratio*100}%)" if ratio > 0 else "#0284c7"
            elif affection_val < 60:
                # 緑から黄緑へのグラデーション
                ratio = (affection_val - 40) / 20
                return f"color-mix(in srgb, #16a34a {100-ratio*100}%, #65a30d {ratio*100}%)" if ratio > 0 else "#16a34a"
            elif affection_val < 80:
                # 黄緑からオレンジへのグラデーション
                ratio = (affection_val - 60) / 20
                return f"color-mix(in srgb, #65a30d {100-ratio*100}%, #d97706 {ratio*100}%)" if ratio > 0 else "#65a30d"
            else:
                # オレンジから赤へのグラデーション
                ratio = (affection_val - 80) / 20
                return f"color-mix(in srgb, #d97706 {100-ratio*100}%, #dc2626 {ratio*100}%)" if ratio > 0 else "#d97706"

        with st.expander("📊 ステータス", expanded=True):
            affection = st.session_state.chat['affection']
            
            # 好感度の文字を白くするためのCSS
            affection_css = """
            <style>
            .affection-label {
                color: white !important;
                font-weight: bold;
                font-size: 16px;
                margin-bottom: 10px;
            }
            </style>
            """
            st.markdown(affection_css, unsafe_allow_html=True)
            st.markdown('<div class="affection-label">好感度</div>', unsafe_allow_html=True)
            
            st.metric(label="好感度", value=f"{affection} / 100")
            
            # 好感度に応じた動的プログレスバー
            progress_value = affection / 100.0
            
            # 好感度範囲に応じたdata-value属性を設定するためのCSS
            if affection < 20:
                value_range = "0-20"
            elif affection < 40:
                value_range = "20-40"
            elif affection < 60:
                value_range = "40-60"
            elif affection < 80:
                value_range = "60-80"
            else:
                value_range = "80-100"
            
            current_color = get_affection_color(affection)
            
            # 動的プログレスバーのCSS
            dynamic_progress_css = f"""
            <style>
            .dynamic-affection-progress {{
                width: 100%;
                height: 24px;
                border-radius: 8px;
                border: 3px solid rgba(0, 0, 0, 0.4);
                background: linear-gradient(90deg, #0284c7 0%, #16a34a 20%, #65a30d 40%, #d97706 60%, #ea580c 80%, #dc2626 100%);
                box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
                position: relative;
                overflow: hidden;
                margin: 10px 0;
            }}
            
            .dynamic-affection-progress::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                height: 100%;
                width: {progress_value * 100}%;
                background: linear-gradient(90deg, #0284c7 0%, #16a34a 20%, #65a30d 40%, #d97706 60%, #ea580c 80%, #dc2626 100%);
                border-radius: 5px;
                transition: width 0.5s ease-in-out;
                box-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
                opacity: 0.9;
            }}
            
            .dynamic-affection-progress::after {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: repeating-linear-gradient(45deg,
                        transparent,
                        transparent 4px,
                        rgba(255, 255, 255, 0.2) 4px,
                        rgba(255, 255, 255, 0.2) 8px);
                animation: progress-stripes 1s linear infinite;
                border-radius: 5px;
                width: {progress_value * 100}%;
                transition: width 0.5s ease-in-out;
            }}
            
            /* 現在の好感度値を表示 */
            .dynamic-affection-progress .affection-value {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                color: white;
                font-weight: bold;
                font-size: 12px;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.7);
                z-index: 10;
            }}
            </style>
            """
            
            st.markdown(dynamic_progress_css, unsafe_allow_html=True)
            st.markdown(f'<div class="dynamic-affection-progress"><div class="affection-value">{affection}%</div></div>', unsafe_allow_html=True)
            
            stage_name = managers['sentiment_analyzer'].get_relationship_stage(affection)
            st.markdown(f"**関係性**: {stage_name}")
            
            # SceneManagerから現在のテーマ名を取得
            current_theme_name = st.session_state.chat['scene_params'].get("theme", "default")
            st.markdown(f"**現在のシーン**: {current_theme_name}")



        with st.expander("💾 データ保存"):
            # 保存データの存在確認（永続ストレージ優先）
            # persistent_user_manager 完全廃止
            user_id_manager = managers["user_id_manager"]  # フォールバック用
            
            has_saved_data = False
            user_info = None
            
            try:
                # 永続ストレージから確認
                # persistent_user_manager 完全廃止
                has_saved_data = user_info is not None and "game_data" in user_info
                if has_saved_data:
                    logger.debug("永続ストレージに保存データを確認")
            except Exception as e:
                logger.warning(f"永続ストレージ確認エラー、フォールバック使用: {e}")
                # フォールバック: 従来のローカルファイル方式
                has_saved_data = user_id_manager.is_user_data_exists()
                if has_saved_data:
                    user_info = user_id_manager.get_user_info()
                    logger.debug("フォールバック: ローカルファイルに保存データを確認")
            
            if has_saved_data and user_info and "game_data" in user_info:
                # 保存データがある場合の情報表示
                game_data = user_info["game_data"]
                if game_data:
                    saved_affection = game_data.get("affection", "不明")
                    saved_messages = len(game_data.get("messages", []))
                    saved_at = game_data.get("saved_at", "不明")
                    if saved_at != "不明":
                        try:
                            from datetime import datetime
                            saved_time = datetime.fromisoformat(saved_at.replace('Z', '+00:00'))
                            saved_at = saved_time.strftime("%m/%d %H:%M")
                        except:
                            pass
                    
                    # ストレージタイプを表示
                    storage_type = "🌐 永続ストレージ" if user_info.get("storage_type") != "local" else "📁 ローカル"
                    st.info(f"💾 保存データあり ({storage_type})\n好感度: {saved_affection}/100\nメッセージ: {saved_messages}件\n保存日時: {saved_at}")
            
            if st.button("💾 ゲームデータを保存", help="現在の進行状況（好感度、チャット履歴など）をファイルに保存します", use_container_width=True):
                success = save_game_data_to_file(managers)
                if success:
                    st.success("✅ ゲームデータを保存しました！")
                    if 'chat' in st.session_state:
                        affection = st.session_state.chat.get('affection', 30)
                        message_count = len(st.session_state.chat.get('messages', []))
                        st.info(f"📊 保存内容\n好感度: {affection}/100\nメッセージ: {message_count}件")
                else:
                    st.error("❌ ゲームデータの保存に失敗しました。")

        with st.expander("⚙️ 設定"):
            # 設定ボタン内の表示を大きくするCSS
            settings_css = """
            <style>
            .settings-content {
                font-size: 18px !important;
            }
            .settings-content .stButton > button {
                font-size: 18px !important;
                padding: 12px 20px !important;
                height: auto !important;
            }
            .settings-content .stButton > button div {
                font-size: 18px !important;
            }
            </style>
            """
            st.markdown(settings_css, unsafe_allow_html=True)
            
            # 設定コンテンツをラップ
            st.markdown('<div class="settings-content">', unsafe_allow_html=True)
            
            # ... (エクスポートやリセットボタンのロジックは省略) ...
            if st.button("🔄 会話をリセット", type="secondary", use_container_width=True, help="あなたの会話履歴のみをリセットします（他のユーザーには影響しません）"):
                # チャット履歴を完全にリセット
                st.session_state.chat['messages'] = [{"role": "assistant", "content": "何の用？遊びに来たの？", "is_initial": True}]
                st.session_state.chat['affection'] = 30
                st.session_state.chat['scene_params'] = {"theme": "default"}
                st.session_state.chat['limiter_state'] = managers['rate_limiter'].create_limiter_state()
                st.session_state.chat['ura_mode'] = False  # 裏モードもリセット
                
                # メモリマネージャーをクリア
                st.session_state.memory_manager.clear_memory()
                
                # Streamlitの内部チャット状態もクリア
                if 'messages' in st.session_state:
                    del st.session_state.messages
                if 'last_sent_message' in st.session_state:
                    del st.session_state.last_sent_message
                if 'user_message_input' in st.session_state:
                    del st.session_state.user_message_input
                if 'message_flip_states' in st.session_state:
                    del st.session_state.message_flip_states
                
                # 新しいセッションIDを生成（完全リセット）
                session_api_client = managers["session_api_client"]
                
                # セッションをリセット
                new_session_id = session_api_client.reset_session()
                st.session_state.user_id = new_session_id
                
                st.success("会話を完全にリセットしました（新しいセッションとして開始）")
                st.rerun()
            
            # フルリセットボタン（Cookie含む完全リセット）
            st.markdown("---")
            st.markdown("**⚠️ 危険な操作**")
            
            if st.button("🔥 フルリセット（Cookie含む）", 
                        type="secondary", 
                        use_container_width=True, 
                        help="Cookie含む全データを完全にリセットします（ブラウザセッションも新規作成）"):
                
                # 確認ダイアログ
                if 'full_reset_confirm' not in st.session_state:
                    st.session_state.full_reset_confirm = False
                
                if not st.session_state.full_reset_confirm:
                    st.session_state.full_reset_confirm = True
                    st.warning("⚠️ 本当にフルリセットしますか？この操作は取り消せません。")
                    st.info("Cookie削除→新規セッション作成を実行します。もう一度ボタンを押してください。")
                    st.rerun()
                else:
                    # フルリセット実行
                    session_api_client = managers["session_api_client"]
                    
                    try:
                        # プログレスバーで進行状況を表示
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("🔄 フルリセット開始...")
                        progress_bar.progress(10)
                        
                        # 1. 永続ストレージとローカルユーザーデータ削除
                        status_text.text("🗑️ ユーザーデータ削除中...")
                        progress_bar.progress(20)
                        
                        # persistent_user_manager 完全廃止
                        user_id_manager = managers["user_id_manager"]
                        
                        # 永続ストレージから削除
                        persistent_data_deleted = False
                    
                        
                        # ローカルファイルからも削除（フォールバック）
                        local_data_deleted = user_id_manager.delete_user_data()
                        
                        user_data_deleted = persistent_data_deleted or local_data_deleted
                        
                        # 2. フルリセット実行（Cookie削除→新規セッション作成）
                        status_text.text("🍪 Cookie削除中...")
                        progress_bar.progress(40)
                        
                        reset_result = session_api_client.full_reset_session()
                        
                        if reset_result['success']:
                            status_text.text("✅ Cookie削除完了、新規セッション作成中...")
                            progress_bar.progress(60)
                            
                            # 2. Streamlitセッション状態を完全クリア
                            keys_to_clear = list(st.session_state.keys())
                            for key in keys_to_clear:
                                if key not in ['_session_id', 'session_info']:  # 必要なキーは保持
                                    del st.session_state[key]
                            
                            # CSS読み込みフラグもリセット
                            st.session_state.css_loaded = False
                            st.session_state.last_background_theme = ''
                            st.session_state._initialization_complete = False
                            
                            # メッセージ処理キャッシュもクリア
                            cache_keys_to_clear = [key for key in st.session_state.keys() if key.startswith('processed_')]
                            for cache_key in cache_keys_to_clear:
                                del st.session_state[cache_key]
                            
                            status_text.text("🔄 セッション状態初期化中...")
                            progress_bar.progress(80)
                            
                            # 3. 新しいユーザーIDを設定
                            if reset_result.get('new_session_id'):
                                # 完全なセッションIDを取得（表示用は短縮版）
                                full_session_id = st.session_state.session_info.get('session_id')
                                st.session_state.user_id = full_session_id
                            
                            # 4. 初期状態を再構築（強制リセット）
                            initialize_session_state(managers, force_reset_override=True)
                            
                            # 5. MemoryManagerの完全クリア（念のため）
                            if hasattr(st.session_state, 'memory_manager'):
                                st.session_state.memory_manager.clear_memory()
                                logger.info("MemoryManager完全クリア実行")
                            
                            # 6. SessionManagerのデータもリセット
                            session_manager = get_session_manager()
                            session_manager.reset_session_data()
                            logger.info("SessionManagerデータリセット実行")
                            
                            status_text.text("🎉 フルリセット完了！")
                            progress_bar.progress(100)
                            
                            # 成功メッセージ
                            st.success(f"🔥 フルリセット完了！")
                            st.info(f"📁 ローカルユーザーデータ削除: {'✅' if user_data_deleted else '❌'}")
                            st.info(f"📊 Cookie削除: {'✅' if reset_result.get('cookie_reset') else '❌'}")
                            st.info(f"🆕 新規セッション: {'✅' if reset_result.get('session_created') else '❌'}")
                            st.info(f"🔄 旧→新: {reset_result.get('old_session_id')} → {reset_result.get('new_session_id')}")
                            
                            # 自動リロード
                            st.info("⏳ 3秒後に自動でページを再読み込みします...")
                            reload_js = """
                            <script>
                            setTimeout(function() {
                                window.location.reload();
                            }, 3000);
                            </script>
                            """
                            st.markdown(reload_js, unsafe_allow_html=True)
                            
                        else:
                            st.error(f"❌ フルリセット失敗: {reset_result.get('message', '不明なエラー')}")
                            st.info("通常のリセットを試すか、ページを手動で再読み込みしてください。")
                        
                    except Exception as e:
                        logger.error(f"フルリセットエラー: {e}")
                        st.error(f"❌ フルリセットに失敗しました: {str(e)}")
                        st.info("通常のリセットを試すか、ページを手動で再読み込みしてください。")
                    
                    # 確認フラグをリセット
                    st.session_state.full_reset_confirm = False
            
            # 設定コンテンツのHTMLタグを閉じる
            st.markdown('</div>', unsafe_allow_html=True)
        
        # チュートリアル案内をサイドバーに表示（無効化）
        # tutorial_manager.render_tutorial_sidebar()

        if False:  # デバッグモードを完全に無効化
            with st.expander("🛠️ デバッグ情報", expanded=False):
                # SessionManagerから詳細情報を取得
                session_manager = get_session_manager()
                session_info = session_manager.get_session_info()
                isolation_status = session_manager.get_isolation_status()
                
                # 検証履歴と復旧履歴を取得
                validation_history = session_manager.get_validation_history(limit=10)
                recovery_history = session_manager.get_recovery_history(limit=10)
                
                # セッション分離詳細情報を構築
                session_isolation_details = {
                    "session_integrity": {
                        "status": "✅ 正常" if session_info["is_consistent"] else "❌ 不整合",
                        "session_id_match": session_info["session_id"] == session_info["current_session_id"],
                        "original_session_id": session_info["session_id"],
                        "current_session_id": session_info["current_session_id"],
                        "stored_session_id": session_info["stored_session_id"],
                        "user_id": session_info["user_id"],
                        "session_age_minutes": round(session_info["session_age_seconds"] / 60, 2),
                        "last_validated": session_info["last_validated"]
                    },
                    "validation_metrics": {
                        "total_validations": session_info["validation_count"],
                        "total_recoveries": session_info["recovery_count"],
                        "validation_history_size": session_info["validation_history_count"],
                        "recovery_history_size": session_info["recovery_history_count"],
                        "success_rate": round((session_info["validation_count"] - session_info["recovery_count"]) / max(session_info["validation_count"], 1) * 100, 2) if session_info["validation_count"] > 0 else 100
                    },
                    "component_isolation": isolation_status["component_isolation"],
                    "data_integrity": isolation_status["data_integrity"]
                }
                
                # FastAPIセッション状態を取得
                session_api_client = managers.get("session_api_client")
                api_session_status = session_api_client.get_session_status() if session_api_client else {}
                
                # Cookie状態を取得
                cookie_status = session_api_client.get_cookie_status() if session_api_client else {}
                
                # 拡張されたデバッグ情報
                enhanced_debug_info = {
                    "session_isolation_details": session_isolation_details,
                    "isolation_status": isolation_status,
                    "session_manager_info": {
                        "session_id": session_info["session_id"],
                        "current_session_id": session_info["current_session_id"],
                        "user_id": session_info["user_id"],
                        "is_consistent": session_info["is_consistent"],
                        "validation_count": session_info["validation_count"],
                        "recovery_count": session_info["recovery_count"],
                        "session_age_seconds": session_info["session_age_seconds"],
                        "created_at": session_info["created_at"],
                        "last_validated": session_info["last_validated"]
                    },
                    "fastapi_session_info": api_session_status,
                    "cookie_status": cookie_status,
                    "chat_state": {
                        "affection": st.session_state.chat['affection'],
                        "theme": st.session_state.chat['scene_params']['theme'],
                        "messages_count": len(st.session_state.chat['messages']),
                        "ura_mode": st.session_state.chat.get('ura_mode', False),
                        "limiter_state_present": 'limiter_state' in st.session_state.chat,
                        "scene_change_pending": st.session_state.chat.get('scene_change_pending')
                    },
                    "memory_state": {
                        "cache_size": len(st.session_state.memory_manager.important_words_cache),
                        "special_memories": len(st.session_state.memory_manager.special_memories),
                        "memory_manager_type": type(st.session_state.memory_manager).__name__,
                        "memory_manager_id": id(st.session_state.memory_manager)
                    },
                    "system_state": {
                        "session_keys": list(st.session_state.keys()),
                        "session_keys_count": len(st.session_state.keys()),
                        "notifications_pending": {
                            "affection": len(st.session_state.affection_notifications),
                            "memory": len(st.session_state.memory_notifications)
                        },
                        "streamlit_session_id": st.session_state.get('_session_id', 'unknown')
                    }
                }
                
                # タブ形式でデバッグ情報を整理（拡張版）
                debug_tab1, debug_tab2, debug_tab3, debug_tab4, debug_tab5, debug_tab6 = st.tabs([
                    "🔍 セッション分離", "📊 基本情報", "🍪 Cookie状態", "✅ 検証履歴", "🔧 復旧履歴", "⚙️ システム詳細"
                ])
                
                with debug_tab1:
                    st.markdown("### 🔒 セッション分離状態")
                    
                    # 手動検証ボタン
                    col_btn1, col_btn2, col_btn3 = st.columns(3)
                    with col_btn1:
                        if st.button("🔍 手動検証実行", help="セッション整合性を手動で検証します"):
                            validation_result = validate_session_state()
                            if validation_result:
                                st.success("✅ セッション検証成功")
                            else:
                                st.error("❌ セッション検証失敗")
                            st.rerun()
                    
                    with col_btn2:
                        if st.button("📋 詳細検証実行", help="詳細なセッション検証を実行します"):
                            detailed_issues = perform_detailed_session_validation(session_manager)
                            if not detailed_issues:
                                st.success("✅ 詳細検証: 問題なし")
                            else:
                                st.warning(f"⚠️ 詳細検証: {len(detailed_issues)}件の問題を検出")
                                for issue in detailed_issues:
                                    severity_icon = "🔴" if issue['severity'] == 'critical' else "🟡"
                                    st.write(f"{severity_icon} **{issue['type']}**: {issue['description']}")
                    
                    with col_btn3:
                        if st.button("🔄 強制復旧実行", help="セッション状態を強制的に復旧します"):
                            session_manager.recover_session()
                            st.info("🔄 セッション復旧を実行しました")
                            st.rerun()
                    
                    st.markdown("---")
                    
                    # セッション整合性ステータス
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "セッション整合性",
                            session_isolation_details["session_integrity"]["status"],
                            delta=None
                        )
                        st.metric(
                            "検証成功率",
                            f"{session_isolation_details['validation_metrics']['success_rate']}%",
                            delta=None
                        )
                    
                    with col2:
                        st.metric(
                            "総検証回数",
                            session_isolation_details["validation_metrics"]["total_validations"],
                            delta=None
                        )
                        st.metric(
                            "復旧実行回数",
                            session_isolation_details["validation_metrics"]["total_recoveries"],
                            delta=None
                        )
                    
                    # コンポーネント分離状態
                    st.markdown("#### 🧩 コンポーネント分離状態")
                    isolation_data = session_isolation_details["component_isolation"]
                    
                    for component, is_isolated in isolation_data.items():
                        status_icon = "✅" if is_isolated else "❌"
                        component_name = {
                            "chat_isolated": "チャット機能",
                            "memory_isolated": "メモリ管理",
                            "notifications_isolated": "通知システム",
                            "rate_limit_isolated": "レート制限"
                        }.get(component, component)
                        
                        st.write(f"{status_icon} **{component_name}**: {'分離済み' if is_isolated else '未分離'}")
                    
                    # データ整合性
                    st.markdown("#### 📋 データ整合性")
                    integrity_data = session_isolation_details["data_integrity"]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("チャットメッセージ数", integrity_data["chat_messages_count"])
                        st.metric("メモリキャッシュサイズ", integrity_data["memory_cache_size"])
                    
                    with col2:
                        st.metric("特別な記憶数", integrity_data["special_memories_count"])
                        pending_total = sum(integrity_data["pending_notifications"].values())
                        st.metric("保留中通知数", pending_total)
                    
                    # セッションID詳細
                    st.markdown("#### 🆔 セッションID詳細")
                    session_id_info = session_isolation_details["session_integrity"]
                    
                    st.text(f"""
セッション整合性: {session_id_info['status']}
オリジナルID: {session_id_info['original_session_id']}
現在のID: {session_id_info['current_session_id']}
保存されたID: {session_id_info['stored_session_id']}
ユーザーID: {session_id_info['user_id']}
セッション継続時間: {session_id_info['session_age_minutes']} 分
最終検証時刻: {session_id_info['last_validated'][:19]}
                    """)
                
                with debug_tab2:
                    st.markdown("### 📊 基本セッション情報")
                    st.text(str({
                        "session_manager": enhanced_debug_info["session_manager_info"],
                        "chat_state": enhanced_debug_info["chat_state"],
                        "memory_state": enhanced_debug_info["memory_state"]
                    }))
                
                with debug_tab3:
                    st.markdown("### 🍪 Cookie状態")
                    
                    if cookie_status:
                        # Cookie概要
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Cookie数", cookie_status.get('count', 0))
                        with col2:
                            has_session = cookie_status.get('has_session_cookie', False)
                            st.metric("セッションCookie", "✅ あり" if has_session else "❌ なし")
                        with col3:
                            if st.button("🔄 Cookie状態更新", help="Cookie状態を再取得します"):
                                st.rerun()
                        
                        # Cookie詳細
                        if cookie_status.get('cookies'):
                            st.markdown("#### Cookie詳細")
                            for i, cookie in enumerate(cookie_status['cookies']):
                                with st.expander(f"Cookie {i+1}: {cookie.get('name', 'unknown')}"):
                                    st.text(str(cookie))
                        else:
                            st.info("現在Cookieは設定されていません")
                        
                        # Cookie操作ボタン
                        st.markdown("#### Cookie操作")
                        col_cookie1, col_cookie2 = st.columns(2)
                        with col_cookie1:
                            if st.button("🗑️ Cookie削除テスト", help="Cookieを削除してテストします"):
                                try:
                                    session_api_client.session.cookies.clear()
                                    st.success("Cookie削除完了")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Cookie削除エラー: {e}")
                        
                        with col_cookie2:
                            if st.button("🔥 フルリセットテスト", help="フルリセット機能をテストします"):
                                try:
                                    reset_result = session_api_client.full_reset_session()
                                    if reset_result['success']:
                                        st.success(f"フルリセット成功: {reset_result['message']}")
                                    else:
                                        st.error(f"フルリセット失敗: {reset_result['message']}")
                                    st.text(str(reset_result))
                                except Exception as e:
                                    st.error(f"フルリセットテストエラー: {e}")
                    else:
                        st.warning("Cookie状態を取得できませんでした")
                
                with debug_tab4:
                    st.markdown("### ✅ セッション検証履歴")
                    if validation_history:
                        st.write(f"**最新の検証履歴（最大10件）:** 総検証回数 {session_info['validation_count']} 回")
                        
                        # 検証履歴のサマリー
                        recent_validations = validation_history[-5:] if len(validation_history) >= 5 else validation_history
                        success_count = sum(1 for v in recent_validations if v['is_consistent'])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("直近5回の成功率", f"{success_count}/{len(recent_validations)}")
                        with col2:
                            st.metric("最新検証結果", "✅ 成功" if validation_history[-1]['is_consistent'] else "❌ 失敗")
                        with col3:
                            st.metric("検証間隔", f"約{round((datetime.now() - datetime.fromisoformat(validation_history[-1]['timestamp'].replace('Z', '+00:00').replace('+00:00', ''))).total_seconds() / 60, 1)}分前")
                        
                        # 詳細な検証履歴
                        for i, record in enumerate(reversed(validation_history)):
                            status_icon = "✅" if record['is_consistent'] else "❌"
                            timestamp = record['timestamp'][:19].replace('T', ' ')
                            
                            with st.expander(f"{status_icon} 検証 #{record['validation_count']} - {timestamp}", expanded=(i==0)):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**基本情報:**")
                                    st.write(f"- 検証時刻: {timestamp}")
                                    st.write(f"- 検証回数: #{record['validation_count']}")
                                    st.write(f"- 結果: {'✅ 整合性OK' if record['is_consistent'] else '❌ 不整合検出'}")
                                    st.write(f"- ユーザーID: {record['user_id']}")
                                
                                with col2:
                                    st.write("**セッションID情報:**")
                                    st.write(f"- オリジナル: {record['original_session_id']}")
                                    st.write(f"- 現在: {record['current_session_id']}")
                                    st.write(f"- 保存済み: {record['stored_session_id']}")
                                    st.write(f"- セッションキー数: {record['session_keys_count']}")
                    else:
                        st.info("検証履歴がありません")
                
                with debug_tab4:
                    st.markdown("### 🔧 セッション復旧履歴")
                    if recovery_history:
                        st.write(f"**復旧履歴:** 総復旧回数 {session_info['recovery_count']} 回")
                        
                        # 復旧履歴のサマリー
                        if recovery_history:
                            last_recovery = recovery_history[-1]
                            time_since_recovery = (datetime.now() - datetime.fromisoformat(last_recovery['timestamp'].replace('Z', '+00:00').replace('+00:00', ''))).total_seconds() / 60
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("最新復旧", f"約{round(time_since_recovery, 1)}分前")
                            with col2:
                                st.metric("復旧タイプ", last_recovery.get('recovery_type', 'unknown'))
                        
                        # 詳細な復旧履歴
                        for record in reversed(recovery_history):
                            timestamp = record['timestamp'][:19].replace('T', ' ')
                            recovery_type = record.get('recovery_type', 'unknown')
                            
                            with st.expander(f"🔧 復旧 #{record['recovery_count']} - {timestamp} ({recovery_type})", expanded=True):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**復旧情報:**")
                                    st.write(f"- 復旧時刻: {timestamp}")
                                    st.write(f"- 復旧回数: #{record['recovery_count']}")
                                    st.write(f"- 復旧タイプ: {recovery_type}")
                                    st.write(f"- ユーザーID: {record['user_id']}")
                                
                                with col2:
                                    st.write("**セッションID変更:**")
                                    st.write(f"- 変更前: {record['old_session_id']}")
                                    st.write(f"- 変更後: {record['new_session_id']}")
                                    st.write(f"- セッションキー数: {record['session_keys_count']}")
                                    st.write(f"- 復旧時検証回数: {record['validation_count_at_recovery']}")
                    else:
                        st.success("復旧履歴がありません（正常な状態です）")
                
                with debug_tab5:
                    st.markdown("### 🔧 復旧履歴")
                    if recovery_history:
                        st.write(f"**復旧履歴（最大10件）:** 総復旧回数 {session_info['recovery_count']} 回")
                        
                        # 復旧履歴のサマリー
                        recent_recoveries = recovery_history[-5:] if len(recovery_history) >= 5 else recovery_history
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("直近の復旧回数", len(recent_recoveries))
                        with col2:
                            if recent_recoveries:
                                last_recovery = recent_recoveries[-1]
                                st.metric("最終復旧", last_recovery['timestamp'][:19])
                        
                        # 復旧履歴の詳細表示
                        for i, recovery in enumerate(reversed(recovery_history)):
                            with st.expander(f"復旧 #{len(recovery_history)-i}: {recovery['timestamp'][:19]}"):
                                st.text(str(recovery))
                    else:
                        st.success("復旧履歴がありません（正常な状態です）")
                
                with debug_tab6:
                    st.markdown("### ⚙️ システム詳細情報")
                    st.text(str(enhanced_debug_info["system_state"]))
                    
                    
                    # ポチ機能の統計（本格実装）
                    st.markdown("---")
                    st.markdown("### 🐕 ポチ機能統計")
                    flip_states = st.session_state.get('message_flip_states', {})
                    st.markdown(f"**フリップ状態数**: {len(flip_states)}")
                    if flip_states:
                        st.text(str(flip_states))
                    
                    # 追加のシステム情報
                    st.markdown("#### 🔧 技術詳細")
                    st.text(f"""
Python オブジェクトID:
- st.session_state: {id(st.session_state)}
- SessionManager: {id(session_manager)}
- MemoryManager: {enhanced_debug_info['memory_state']['memory_manager_id']}

環境変数:
- DEBUG_MODE: {os.getenv('DEBUG_MODE', 'false')}
- FORCE_SESSION_RESET: {os.getenv('FORCE_SESSION_RESET', 'false')}

Streamlit情報:
- セッション状態キー数: {len(st.session_state.keys())}
- 内部セッションID: {st.session_state.get('_session_id', 'unknown')}
                    """)

    # --- メインコンテンツ ---
    st.title("💬 麻理チャット")
    st.markdown("*捨てられたアンドロイド「麻理」との対話*")
    
    # 好感度変化の通知を表示（テスト中）
    if st.session_state.get('affection_notifications'):
        for notification in st.session_state.affection_notifications:
            show_affection_notification(
                notification["change_amount"],
                notification["change_reason"],
                notification["new_affection"],
                notification.get("is_milestone", False)
            )
        # 通知を表示したらクリア
        st.session_state.affection_notifications = []
    
    # 特別な記憶の通知を表示（テスト中）
    if st.session_state.get('memory_notifications'):
        for notification in st.session_state.memory_notifications:
            show_memory_notification(notification)
        # 通知を表示したらクリア
        st.session_state.memory_notifications = []
    
    # チュートリアル用のexpander（テスト中）
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.expander("📖 初めてチャットする人へ", expanded=False):
            st.markdown("""
            ### 🤖 麻理について
            麻理は感情を持つアンドロイドです。あなたとの会話を通じて関係性が変化していきます。
            
            ### 💝 好感度システム
            - **会話の内容**によって好感度が上下します
            - **優しい言葉**をかけると好感度が上がります
            - **冷たい態度**だと好感度が下がることも...
            - サイドバーで現在の好感度を確認できます
            
            ### 🐕 本音表示機能
            特定の場所について話すと、背景が自動的に変わります：
            - 🏖️ **ビーチ**や**海**の話 → 夕日のビーチ
            - ⛩️ **神社**や**お参り**の話 → 神社の境内
            - ☕ **カフェ**や**コーヒー**の話 → 午後のカフェ
            - 🎨 **美術館**や**アート**の話 → 夜の美術館
            - 🎆 **お祭り**や**花火**の話 → 夜祭り
            
            ### 💬 会話のコツ
            1. **自然な会話**を心がけてください
            2. **質問**をすると麻理が詳しく答えてくれます
            3. **感情**を込めた言葉は特に反応が良いです
            4. **200文字以内**でメッセージを送ってください
            
            ### ⚙️ 便利な機能
            - **サイドバー**：好感度やシーン情報を確認
            - **会話履歴**：過去の会話を振り返り
            - **リセット機能**：新しい関係から始めたい時に
            
            ---
            **準備ができたら、下のチャット欄で麻理に話しかけてみてください！** 😊
            """)
    
    st.markdown("---")
    
    # カスタムチャット履歴表示エリア（マスク機能付き）
    # セッション状態の安全性チェック
    if 'chat' not in st.session_state:
        logger.error("チャットセッション状態が存在しません - 初期化を実行")
        st.error("チャットセッションが初期化されていません。ページを再読み込みしてください。")
        return
    
    if 'messages' not in st.session_state.chat:
        logger.warning("メッセージリストが存在しません - 初期化します")
        initial_message = {"role": "assistant", "content": "何の用？遊びに来たの？", "is_initial": True}
        st.session_state.chat['messages'] = [initial_message]
        logger.info("メッセージリストを初期化しました")
    
    # チャット履歴を常に表示（応答処理中でない時）
    if not st.session_state.get('awaiting_response', False):
        logger.info(f"🎯 通常時のチャット履歴表示: {len(st.session_state.chat['messages'])}件のメッセージ")
        for message in st.session_state.chat['messages']:
            role = message.get("role", "user")
            content = message.get("content", "")
            is_initial = message.get("is_initial", False)
            
            # HTMLタグとStreamlitクラス名を完全に除去
            import re
            import html
            clean_content = re.sub(r'<[^>]*>', '', content)
            clean_content = re.sub(r'st-emotion-cache-[a-zA-Z0-9]+', '', clean_content)
            clean_content = re.sub(r'class="[^"]*"', '', clean_content)
            clean_content = re.sub(r'data-[^=]*="[^"]*"', '', clean_content)
            clean_content = re.sub(r'\s+', ' ', clean_content).strip()
            
            with st.chat_message(role):
                if is_initial:
                    st.markdown(f"**[初期メッセージ]** {clean_content}")
                else:
                    # 隠された真実の処理
                    has_hidden_content, visible_content, hidden_content = managers['chat_interface']._detect_hidden_content(clean_content)
                    if has_hidden_content and role == "assistant":
                        show_all_hidden = st.session_state.get('show_all_hidden', False)
                        if show_all_hidden:
                            st.markdown(f"**表面:** {visible_content}")
                            st.markdown(f"🐕 **本音:** {hidden_content}")
                        else:
                            st.markdown(visible_content)
                    else:
                        st.markdown(clean_content)
        logger.debug("通常時のチャット履歴表示完了")
    else:
        logger.debug("応答処理中のため通常時の履歴表示をスキップ")

    # メッセージ処理ロジック
    def process_chat_message(message: str):
        response = None
        try:
            logger.info(f"🚀 process_chat_message開始 - メッセージ: '{message}'")
            # チュートリアルステップ1を完了（メッセージ送信）
            tutorial_manager.check_step_completion(1, True)
            
            # セッション検証を処理開始時に実行
            if not validate_session_state():
                logger.error("Session validation failed at message processing start")
                return "（申し訳ありません。システムに問題が発生しました。ページを再読み込みしてください。）"
            
            # レート制限チェック
            if 'limiter_state' not in st.session_state.chat:
                st.session_state.chat['limiter_state'] = managers['rate_limiter'].create_limiter_state()
            
            limiter_state = st.session_state.chat['limiter_state']
            if not managers['rate_limiter'].check_limiter(limiter_state):
                st.session_state.chat['limiter_state'] = limiter_state
                return "（…少し話すのが速すぎる。もう少し、ゆっくり話してくれないか？）"
            
            st.session_state.chat['limiter_state'] = limiter_state

            # 会話履歴を正しく構築（現在のメッセージは含まない）
            # 注意: この時点では現在のユーザーメッセージはまだ履歴に追加されていない
            # 初期メッセージ（is_initial=True）を履歴から除外して会話ペアを構築
            non_initial_messages = [msg for msg in st.session_state.chat['messages'] 
                                  if not msg.get('is_initial', False)]
            
            # 会話ペアを時系列順に構築
            history = []
            user_msgs = []
            assistant_msgs = []
            
            for msg in non_initial_messages:
                if msg['role'] == 'user':
                    user_msgs.append(msg['content'])
                elif msg['role'] == 'assistant':
                    assistant_msgs.append(msg['content'])
            
            # ユーザーとアシスタントのメッセージをペアにする（最大5ターン）
            max_turns = min(5, min(len(user_msgs), len(assistant_msgs)))
            for i in range(max_turns):
                if i < len(user_msgs) and i < len(assistant_msgs):
                    history.append((user_msgs[i], assistant_msgs[i]))
            
            # 初回の場合は空の履歴になる（これが正しい動作）
            logger.info(f"📚 構築された履歴: {len(history)}ターン")
            if st.session_state.get('debug_mode', False):
                logger.info(f"🔍 全メッセージ数: {len(st.session_state.chat['messages'])}")
                logger.info(f"🔍 非初期メッセージ数: {len(non_initial_messages)}")
                logger.info(f"🔍 ユーザーメッセージ数: {len(user_msgs)}")
                logger.info(f"🔍 アシスタントメッセージ数: {len(assistant_msgs)}")

            # 好感度更新（初期メッセージを除外）
            old_affection = st.session_state.chat['affection']
            non_initial_messages = [msg for msg in st.session_state.chat['messages'] 
                                  if not msg.get('is_initial', False)]
            affection, change_amount, change_reason = managers['sentiment_analyzer'].update_affection(
                message, st.session_state.chat['affection'], non_initial_messages
            )
            st.session_state.chat['affection'] = affection
            stage_name = managers['sentiment_analyzer'].get_relationship_stage(affection)
            
            # 好感度変化があった場合は通知を追加
            if change_amount != 0:
                affection_notification = {
                    "change_amount": change_amount,
                    "change_reason": change_reason,
                    "new_affection": affection,
                    "old_affection": old_affection
                }
                st.session_state.affection_notifications.append(affection_notification)
                
                # 特定の好感度レベルに到達した時の特別な通知
                milestone_reached = check_affection_milestone(old_affection, affection)
                if milestone_reached:
                    milestone_notification = {
                        "change_amount": 0,  # マイルストーン通知は変化量0で特別扱い
                        "change_reason": milestone_reached,
                        "new_affection": affection,
                        "old_affection": old_affection,
                        "is_milestone": True
                    }
                    st.session_state.affection_notifications.append(milestone_notification)
            
            # シーン変更検知（強化版 + デバッグ）
            current_theme = st.session_state.chat['scene_params']['theme']
            logger.info(f"🎬 シーン変更検知開始 - 現在のテーマ: {current_theme}")
            logger.info(f"📝 最新のユーザーメッセージ: '{message}'")
            logger.info(f"📚 会話履歴件数: {len(history)}")
            
            # 履歴をテキスト形式で確認
            if history:
                history_text = "\n".join([f"ユーザー: {u}\n麻理: {m}" for u, m in history])
                logger.info(f"📜 履歴テキスト: {history_text}")
            else:
                logger.info("📜 履歴テキスト: 空")
            
            # 現在のメッセージも含めてキーワードチェック
            full_text = message + "\n" + "\n".join([f"{u} {m}" for u, m in history])
            logger.info(f"🔍 キーワード検索対象テキスト: '{full_text}'")
            
            # Groq APIクライアントの状態確認
            if hasattr(managers['scene_manager'], 'groq_client') and managers['scene_manager'].groq_client:
                logger.info("✅ Groq APIクライアント利用可能")
            else:
                logger.warning("⚠️ Groq APIクライアント利用不可 - フォールバックモードのみ")
            
            # 強制的にシーン変更検知を実行（履歴に現在のメッセージを含める）
            extended_history = history + [(message, "")]  # 現在のメッセージも含める
            new_theme = managers['scene_manager'].detect_scene_change(extended_history, current_theme=current_theme)
            logger.info(f"🎯 シーン変更検知結果: {new_theme}")
            
            instruction = None
            if new_theme:
                logger.info(f"🎬 シーン変更検出! '{current_theme}' → '{new_theme}'")
                st.session_state.chat['scene_params'] = managers['scene_manager'].update_scene_params(st.session_state.chat['scene_params'], new_theme)
                instruction = managers['scene_manager'].get_scene_transition_message(current_theme, new_theme)
                st.session_state.scene_change_flag = True
                # 背景更新フラグを設定
                st.session_state._background_needs_update = True
                logger.info("シーン変更により背景更新フラグを設定")
                
                # シーン変更時に即座に背景を更新
                try:
                    logger.info(f"シーン変更時の背景更新: テーマ '{new_theme}' (即座に実行)")
                    update_background(managers['scene_manager'], new_theme)
                except Exception as e:
                    logger.error(f"シーン変更時の背景更新でエラーが発生: {e}")
                    import traceback
                    logger.error(f"シーン変更時の背景更新エラーの詳細: {traceback.format_exc()}")
                    # エラーが発生してもアプリケーションは継続
                
                # デバッグモード時にユーザーに通知
                if st.session_state.get("debug_mode", False):
                    st.success(f"🎬 シーン変更: {current_theme} → {new_theme}")
            else:
                logger.info(f"シーン変更なし - 現在のテーマを維持: {current_theme}")
                # デバッグモード時にシーン変更なしの理由を表示
                if st.session_state.get("debug_mode", False):
                    st.info(f"🎬 シーン変更なし: {current_theme} を維持")
                
                # シーン変更なしの場合も現在のテーマで背景を更新（初回表示など）
                try:
                    logger.info(f"現在テーマでの背景更新: テーマ '{current_theme}' (維持)")
                    update_background(managers['scene_manager'], current_theme)
                except Exception as e:
                    logger.error(f"現在テーマでの背景更新でエラーが発生: {e}")
                    import traceback
                    logger.error(f"現在テーマでの背景更新エラーの詳細: {traceback.format_exc()}")
                    # エラーが発生してもアプリケーションは継続

            # メモリ圧縮とサマリー取得（初期メッセージを除外）
            non_initial_messages = [msg for msg in st.session_state.chat['messages'] 
                                  if not msg.get('is_initial', False)]
            compressed_messages, important_words = st.session_state.memory_manager.compress_history(
                non_initial_messages
            )
            memory_summary = st.session_state.memory_manager.get_memory_summary()
            
            # デバッグ: メモリサマリーの内容をログ出力
            if memory_summary:
                logger.warning(f"🧠 メモリサマリーが存在: {memory_summary[:100]}...")
            else:
                logger.info("🧠 メモリサマリーは空です（初対面状態）")
            
            # 対話生成（隠された真実機能統合済み）
            logger.info("🤖 AI対話生成開始")
            logger.info(f"🤖 パラメータ - 好感度: {affection}, ステージ: {stage_name}, テーマ: {st.session_state.chat['scene_params'].get('theme', 'default')}")
            logger.info(f"🤖 URAモード: {st.session_state.chat['ura_mode']}")
            
            # APIクライアントの状態確認
            dialogue_generator = managers['dialogue_generator']
            if dialogue_generator.client:
                logger.info("🔗 Together.ai APIクライアント利用可能")
            else:
                logger.warning("⚠️ Together.ai APIクライアント利用不可 - デモモードで動作")
            
            response = dialogue_generator.generate_dialogue(
                history, message, affection, stage_name, st.session_state.chat['scene_params'], instruction, memory_summary, st.session_state.chat['ura_mode']
            )
            
            logger.info("🤖 AI対話生成完了")
            
            # デバッグ: AI応答の形式をチェック
            if response:
                logger.info(f"🤖 AI応答: '{response[:100]}...'")
                if '[HIDDEN:' in response:
                    logger.info("✅ HIDDEN形式を検出")
                else:
                    logger.warning("⚠️ HIDDEN形式が見つからない - フォールバック処理を実行")
                    # HIDDEN形式でない場合は、強制的にHIDDEN形式に変換
                    response = f"[HIDDEN:（本当の気持ちは...）]{response}"
                    logger.info(f"🔧 フォールバック後: '{response[:100]}...'")
            
            return response if response else "[HIDDEN:（言葉が出てこない...）]…なんて言えばいいか分からない。"

        except Exception as e:
            # ★★★ ここからがデバッグ用のコード ★★★
            import traceback
        
            # ターミナルに強制的にエラーの詳細を出力する
            print("--- !!! PROCESS_CHAT_MESSAGE CRASHED !!! ---")
            print(f"ERROR TYPE: {type(e)}")
            print(f"ERROR DETAILS: {e}")
            traceback.print_exc() # これが最も重要な行！
            print("---------------------------------------------")

            # Streamlitのログにも詳細を記録する
            logger.error(f"チャットメッセージ処理で致命的なエラーが発生", exc_info=True)
            
            # 重要: awaiting_response状態をリセットして無限読み込みを防ぐ
            if 'awaiting_response' in st.session_state:
                st.session_state.awaiting_response = False
                logger.error("エラー発生により awaiting_response をリセットしました")
            
            # 画面にはエラーメッセージを返す
            return "（ごめん、システムの内部で深刻なエラーが起きたみたい。ターミナルを確認して。）"

    # 初期化（セッションステートに必要な変数を登録）
    if "awaiting_response" not in st.session_state:
        st.session_state.awaiting_response = False
    if "response_start_time" not in st.session_state:
        st.session_state.response_start_time = None
    if "latest_prompt" not in st.session_state:
        st.session_state.latest_prompt = ""
    if "chat" not in st.session_state:
        st.session_state.chat = {}
    if "messages" not in st.session_state.chat:
        st.session_state.chat['messages'] = []
    if "affection_notifications" not in st.session_state:
        st.session_state.affection_notifications = []

    # チャット入力フォーム（CSSで下部固定）
    with st.container():
        # フォームを使用してウィジェットの状態管理を改善
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                prompt = st.text_input(
                    "麻理にメッセージを送信...", 
                    placeholder="ここにメッセージを入力してください", 
                    label_visibility="collapsed"
                )

            with col2:
                # 送信ボタン
                submitted = st.form_submit_button("送信", use_container_width=True, type="primary")
                
            if submitted and prompt: # フォームが送信され、何か入力されている場合のみ処理
                # 応答処理のロジックをトリガーする
                st.session_state.awaiting_response = True
                st.session_state.latest_prompt = prompt
                st.session_state.response_start_time = time.time()  # 応答開始時刻を記録
                
                # フォームのclear_on_submit=Trueにより自動的に入力欄がクリアされる
                # rerunは不要 - フォーム送信により自動的に再描画される


    # 応答処理のタイムアウトチェック（30秒でタイムアウト）
    if st.session_state.awaiting_response and st.session_state.response_start_time:
        elapsed_time = time.time() - st.session_state.response_start_time
        if elapsed_time > 30:  # 30秒でタイムアウト
            logger.error(f"応答処理がタイムアウトしました（{elapsed_time:.1f}秒経過）")
            st.session_state.awaiting_response = False
            st.session_state.latest_prompt = ""
            st.session_state.response_start_time = None
            st.error("応答処理がタイムアウトしました。もう一度お試しください。")
    
    # 応答処理（rerun 後にこのブロックが実行される）
    if st.session_state.awaiting_response:
        try:
            prompt = st.session_state.latest_prompt

            # 1. 入力チェック
            if len(prompt) > MAX_INPUT_LENGTH:
                st.error(f"⚠️ メッセージは{MAX_INPUT_LENGTH}文字以内で入力してください。")
                st.session_state.awaiting_response = False
                st.session_state.latest_prompt = ""
            else:
                # 2. ユーザーのメッセージを表示に追加
                from datetime import datetime  # ローカルインポートで確実に利用可能にする
                user_message_id = f"user_{len(st.session_state.chat['messages'])}"
                user_message = {
                    "role": "user",
                    "content": prompt,
                    "timestamp": datetime.now().isoformat(),
                    "message_id": user_message_id
                }
                st.session_state.chat['messages'].append(user_message)
                logger.info(f"ユーザーメッセージを追加: '{prompt}' (ID: {user_message_id})")

                # 3. AI応答生成（スピナー付き + タイムアウト対策）
                response = None
                try:
                    with cute_thinking_spinner():
                        response = process_chat_message(prompt)
                except Exception as e:
                    logger.error(f"AI応答生成でエラーが発生: {e}")
                    response = "（ごめん、今ちょっと調子が悪いみたい...もう一度話しかけてくれる？）"

                # 4. AI応答を履歴に追加
                if response:
                    assistant_message_id = f"assistant_{len(st.session_state.chat['messages'])}"
                    assistant_message = {
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now().isoformat(),
                        "message_id": assistant_message_id
                    }
                    st.session_state.chat['messages'].append(assistant_message)
                    logger.info(f"AI応答を追加: '{response[:50]}...' (ID: {assistant_message_id})")

                # 5. シーン変更フラグがあればリセット
                if st.session_state.get('scene_change_flag', False):
                    del st.session_state['scene_change_flag']

                # 6. 応答完了状態に戻す（必ず実行）
                st.session_state.awaiting_response = False
                st.session_state.latest_prompt = ""
                st.session_state.response_start_time = None
                
                # rerun回数をリセット
                st.session_state.rerun_count = 0
                
                # ゲームデータを保存
                try:
                    save_game_data_to_file(managers)
                except Exception as e:
                    logger.error(f"ゲームデータ保存でエラーが発生: {e}")
                
                # 応答完了 - 画面を更新して新しいメッセージを表示
                logger.info("🎯 応答完了 - 画面更新を実行")
                st.rerun()
                
        except Exception as e:
            # 応答処理全体でエラーが発生した場合の緊急処理
            logger.error(f"応答処理全体でエラーが発生: {e}")
            import traceback
            logger.error(f"応答処理エラーの詳細: {traceback.format_exc()}")
            
            # 必ず awaiting_response をリセット
            st.session_state.awaiting_response = False
            st.session_state.latest_prompt = ""
            st.session_state.response_start_time = None
            
            # エラーメッセージを表示
            st.error("応答処理中にエラーが発生しました。もう一度お試しください。")
            for message in st.session_state.chat['messages']:
                role = message.get("role", "user")
                content = message.get("content", "")
                is_initial = message.get("is_initial", False)
                
                # HTMLタグとStreamlitクラス名を完全に除去
                import re
                import html
                clean_content = re.sub(r'<[^>]*>', '', content)
                clean_content = re.sub(r'st-emotion-cache-[a-zA-Z0-9]+', '', clean_content)
                clean_content = re.sub(r'class="[^"]*"', '', clean_content)
                clean_content = re.sub(r'data-[^=]*="[^"]*"', '', clean_content)
                clean_content = re.sub(r'\s+', ' ', clean_content).strip()
                
                with st.chat_message(role):
                    if is_initial:
                        st.markdown(f"**[初期メッセージ]** {clean_content}")
                    else:
                        # 隠された真実の処理
                        has_hidden_content, visible_content, hidden_content = managers['chat_interface']._detect_hidden_content(clean_content)
                        if has_hidden_content and role == "assistant":
                            show_all_hidden = st.session_state.get('show_all_hidden', False)
                            if show_all_hidden:
                                st.markdown(f"**表面:** {visible_content}")
                                st.markdown(f"🐕 **本音:** {hidden_content}")
                            else:
                                st.markdown(visible_content)
                        else:
                            st.markdown(clean_content)
            
            logger.info("AI応答完了 - 強制チャット履歴表示完了")

    # 応答完了後の処理（rerunは削除）
    if st.session_state.get('response_completed', False):
        st.session_state.response_completed = False  # フラグをクリア
        logger.info("応答完了後の処理完了")



    # ポチのコンポーネント（チャット履歴表示後に配置）
    # 犬のボタン表示前にチャットセッション状態を確認
    if 'chat' not in st.session_state:
        logger.warning("犬のボタン表示前にチャットセッションが存在しません - 初期化します")
        initial_message = {"role": "assistant", "content": "何の用？遊びに来たの？", "is_initial": True}
        st.session_state.chat = {
            "messages": [initial_message],
            "affection": 30,
            "scene_params": {"theme": "default"},
            "limiter_state": managers["rate_limiter"].create_limiter_state(),
            "scene_change_pending": None,
            "ura_mode": False
        }
        logger.info("犬のボタン表示前にチャットセッションを初期化しました")
    elif 'messages' not in st.session_state.chat:
        logger.warning("犬のボタン表示前にメッセージリストが存在しません - 初期化します")
        initial_message = {"role": "assistant", "content": "何の用？遊びに来たの？", "is_initial": True}
        st.session_state.chat['messages'] = [initial_message]
        logger.info("犬のボタン表示前にメッセージリストを初期化しました")
    elif not any(msg.get('is_initial', False) for msg in st.session_state.chat['messages']):
        logger.warning("犬のボタン表示前に初期メッセージが見つかりません - 復元します")
        initial_message = {"role": "assistant", "content": "何の用？遊びに来たの？", "is_initial": True}
        st.session_state.chat['messages'].insert(0, initial_message)
        logger.info("犬のボタン表示前に初期メッセージを復元しました")
    
    # 犬のボタン状態変更の検知と処理（rerunは犬のボタン側で実行済み）
    if st.session_state.get('dog_button_clicked', False):
        # フラグをクリアして重複処理を防ぐ
        st.session_state.dog_button_clicked = False
        # 背景更新を強制
        st.session_state.force_background_update = True
        logger.info("犬のボタン状態変更を検知しました - 背景更新フラグを設定")
        # rerunは犬のボタン側で既に実行済みなので、ここでは実行しない
    
    # DogAssistantコンポーネント（チャットタブ内で表示）
    try:
        managers['dog_assistant'].render_dog_component(tutorial_manager)
        logger.debug("ポチのコンポーネントをチャットタブ内に表示しました")
    except Exception as e:
        logger.error(f"DogAssistant描画エラー: {e}")
        # フォールバックボタンを削除（重複ボタンが原因で2回クリックが必要になっていた）
        st.error("ポチのコンポーネントの表示でエラーが発生しました。")

# === 手紙タブの描画関数 ===
async def generate_tutorial_letter_async(theme: str, managers) -> str:
    """チュートリアル用の短縮版手紙を非同期で生成する（Groq + Qwen使用）"""
    try:
        # 現在の好感度と関係性を取得
        current_affection = st.session_state.chat.get('affection', 30)
        stage_name = managers['sentiment_analyzer'].get_relationship_stage(current_affection)
        user_id = st.session_state.user_id
        
        # チュートリアル用のユーザー履歴を構築
        tutorial_user_history = {
            'profile': {
                'total_letters': 0,
                'affection': current_affection,
                'stage': stage_name
            },
            'letters': {}
        }
        
        # 手紙生成器を使用（Groq + Qwen の2段階プロセス）
        letter_generator = managers.get('letter_generator')
        if not letter_generator:
            # フォールバック：直接生成
            return await generate_tutorial_letter_fallback(theme, current_affection, stage_name)
        
        # 通常の手紙生成システムを使用
        letter_result = await letter_generator.generate_letter(user_id, theme, tutorial_user_history)
        
        # 600文字以内に制限
        letter_content = letter_result['content']
        if len(letter_content) > 600:
            # 文の区切りで切り詰める
            sentences = letter_content.split('。')
            truncated_content = ""
            for sentence in sentences:
                if len(truncated_content + sentence + '。') <= 600:
                    truncated_content += sentence + '。'
                else:
                    break
            letter_content = truncated_content.rstrip('。') + '……'
        
        return letter_content
        
    except Exception as e:
        logger.error(f"チュートリアル手紙生成エラー: {e}")
        # フォールバック手紙
        return await generate_tutorial_letter_fallback(theme, current_affection, stage_name)

async def generate_tutorial_letter_fallback(theme: str, current_affection: int, stage_name: str) -> str:
    """チュートリアル手紙生成のフォールバック"""
    return f"""いつもありがとう。

{theme}のこと……書こうと思ったんだけど、なんか恥ずかしくて。
あんたと話してると、いつもと違う自分になれる気がするの。
それって、きっと特別なことよね。

私、まだまだ素直になれないけれど……
少しずつ、あんたのことを知りたいと思ってる。

……ま、忘れて。バカじゃないの。
でも、ありがとう。"""


def generate_tutorial_letter(theme: str, managers) -> str:
    """チュートリアル用手紙生成の同期ラッパー"""
    logger.info(f"📝 generate_tutorial_letter開始: theme='{theme}'")
    try:
        result = run_async(generate_tutorial_letter_async(theme, managers))
        logger.info(f"📝 generate_tutorial_letter成功: 文字数={len(result) if result else 0}")
        return result
    except Exception as e:
        logger.error(f"チュートリアル手紙生成同期ラッパーエラー: {e}")
        current_affection = st.session_state.chat.get('affection', 30)
        stage_name = managers['sentiment_analyzer'].get_relationship_stage(current_affection)
        logger.info(f"📝 フォールバック手紙生成開始")
        result = run_async(generate_tutorial_letter_fallback(theme, current_affection, stage_name))
        logger.info(f"📝 フォールバック手紙生成完了: 文字数={len(result) if result else 0}")
        return result

def render_letter_tab(managers):
    """「手紙を受け取る」タブのUIを描画する"""
    st.title("✉️ おやすみ前の、一通の手紙")
    
    # タブ切り替え検出とフォーム状態のリセット
    current_tab = "letter"
    last_active_tab = st.session_state.get('last_active_tab', '')
    
    # 初回訪問時の処理
    if last_active_tab == '':
        logger.info("📝 手紙タブ初回訪問")
    
    # 他のタブから手紙タブに切り替わった場合の処理（rerunなし）
    elif last_active_tab != current_tab and last_active_tab not in ['', 'letter']:
        # フォーム関連の状態をリセット（ただし手紙生成完了状態は保持）
        logger.info(f"🔄 タブ切り替え検出: {last_active_tab} → {current_tab}")
        
        # フォーム送信状態もリセット（新しいキー形式に対応）
        form_keys_to_clear = [key for key in st.session_state.keys() if 'letter_form' in key or 'letter_request_form' in key]
        for key in form_keys_to_clear:
            del st.session_state[key]
            logger.info(f"フォームキーをクリア: {key}")
        
        # 手紙表示関連のセッション状態もクリア
        if 'letter_reflected' in st.session_state:
            del st.session_state.letter_reflected
            
        # 手紙フォーム関連の状態もクリア
        if 'letter_form_submitted' in st.session_state:
            del st.session_state.letter_form_submitted
        if 'letter_form_theme' in st.session_state:
            del st.session_state.letter_form_theme
        if 'letter_form_generation_hour' in st.session_state:
            del st.session_state.letter_form_generation_hour
        if 'letter_form_is_instant' in st.session_state:
            del st.session_state.letter_form_is_instant
        
        # ユーザーデータキャッシュをリフレッシュ
        st.session_state.force_refresh_user_data = True
    
    # 現在のタブを記録
    st.session_state.last_active_tab = current_tab
    
    # チュートリアル案内（ステップ4の場合のみ）
    tutorial_manager = managers.get('tutorial_manager')
    if tutorial_manager and tutorial_manager.get_current_step() == 4:
        # 手紙タブに到達したことを祝福
        st.success("🎉 素晴らしい！手紙タブに到達しました！")
        # 手紙タブ専用の案内を表示
        st.info("📝 下のフォームから手紙をリクエストしてみてください。手紙をリクエストするとステップ4が完了します。")
    st.write("今日の終わりに、あなたのためだけにAIが手紙を綴ります。伝えたいテーマと時間を選ぶと、あなたがログインした時に手紙が届きます。")
    
    # 手紙機能のチュートリアル
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.expander("📝 手紙機能の使い方", expanded=False):
            st.markdown("""
            ### ✉️ 手紙機能について
            麻理があなたのために、心を込めて手紙を書いてくれる特別な機能です。
            
            ### 📅 利用方法
            1. **好感度を上げる**：手紙をリクエストするには好感度40以上が必要です
            2. **テーマを入力**：手紙に書いてほしい内容やテーマを入力
            3. **時間を選択**：手紙を書いてほしい深夜の時間を選択
            4. **リクエスト送信**：「この内容でお願いする」ボタンを押す
            5. **手紙を受け取り**：指定した時間に手紙が生成されます
            6. **会話に反映**：手紙を読んだ後、「会話に反映」ボタンで麻理との会話で話題にできます
            
            ### 💡 テーマの例
            - 「今日見た美しい夕日について」
            - 「最近読んだ本の感想」
            - 「季節の変わり目の気持ち」
            - 「大切な人への想い」
            - 「将来への希望や不安」
            
            ### ⏰ 生成時間
            - **深夜2時〜4時**の間で選択可能
            - 静かな夜の時間に、ゆっくりと手紙を綴ります
            - **1日1通まで**リクエスト可能
            
            ### 💝 利用条件
            - **好感度40以上**が必要です
            - 麻理との会話を重ねて関係を深めてください
            
            ### 📖 手紙の確認
            - 生成された手紙は下の「あなたへの手紙」で確認できます
            - 過去の手紙も保存されているので、いつでも読み返せます
            
            ---
            **心に残るテーマを入力して、麻理からの特別な手紙を受け取ってみてください** 💌
            """)

    user_id = st.session_state.user_id
    user_manager = managers['user_manager']
    request_manager = managers['request_manager']

    st.divider()

    # --- 手紙のリクエストフォーム ---
    st.subheader("新しい手紙をリクエストする")
    
    # 現在の好感度を取得
    current_affection = st.session_state.chat['affection']
    required_affection = 40
    
    # 手紙生成回数を取得して即時生成可能かチェック（キャッシュ機能付き）
    cache_key = f"user_data_{user_id}"
    
    # セッション内でのキャッシュをチェック（データベースアクセスを最小化）
    if cache_key in st.session_state and not st.session_state.get('force_refresh_user_data', False):
        user_data = st.session_state[cache_key]
        logger.info(f"📋 キャッシュからユーザーデータを取得: {user_id[:8]}...")
    else:
        try:
            user_data = run_async(user_manager.storage.get_user_data(user_id))
            st.session_state[cache_key] = user_data  # キャッシュに保存
            st.session_state.force_refresh_user_data = False  # フラグをリセット
            logger.info(f"📋 データベースからユーザーデータを取得: {user_id[:8]}...")
        except Exception as e:
            logger.error(f"ユーザーデータ取得エラー: {e}")
            user_data = {"letters": {}}  # エラー時のフォールバック
    
    letters_generated = len(user_data.get("letters", {}))
    can_instant_generate = letters_generated == 0  # 初回のみ即時生成可能
    
    # デバッグ情報（簡潔版）
    logger.info(f"📊 ユーザー {user_id[:8]}... の手紙生成回数: {letters_generated}, 初回判定: {can_instant_generate}")
    
    # デバッグモードの場合のみ詳細表示
    if st.session_state.get('debug_mode', False):
        st.info(f"🔍 デバッグ: ユーザーID={user_id[:8]}..., 手紙数={letters_generated}, 初回={can_instant_generate}")
        st.info(f"🔍 データベースパス: {user_manager.storage.file_path}")
        if user_data.get("letters"):
            st.info(f"🔍 既存の手紙: {list(user_data['letters'].keys())}")
        else:
            st.info("🔍 このユーザーには手紙データがありません（初回ユーザー）")

    
    # 手紙リクエスト可能かどうかの判定
    can_request_letter = current_affection >= required_affection or can_instant_generate
    
    # リクエスト状況をチェック
    try:
        request_status = run_async(request_manager.get_user_request_status(user_id))
    except Exception as e:
        logger.error(f"リクエスト状況取得エラー: {e}")
        request_status = {"has_request": False}

    # 既にリクエスト済みの場合
    if request_status.get("has_request"):
        status = request_status.get('status', 'unknown')
        hour = request_status.get('generation_hour')
        if status == 'pending':
            st.info(f"本日分のリクエストは受付済みです。深夜{hour}時頃に手紙が生成されます。")
        else:
            st.success("本日分の手紙は処理済みです。下記の一覧からご確認ください。")
        can_request_letter = False  # 既にリクエスト済みの場合は新規リクエスト不可
    
    # 好感度不足の場合の警告（履歴は表示するためreturnしない）
    elif not can_request_letter:
        st.warning(f"💔 手紙をリクエストするには好感度が{required_affection}以上必要です。現在の好感度: {current_affection}")
        st.info("麻理ともっと会話して、関係を深めてから手紙をお願いしてみてください。")
        st.info("💡 過去の手紙は下の履歴から確認できます。")
    
    # 初回の特別案内
    elif can_instant_generate:
        st.info("📘 **初回特典**: 初回のみ好感度に関係なく手紙をリクエストできます！")

    # 📖 **手紙生成完了後の読む画面（通常手紙に切り替え）**
    if st.session_state.get('letter_generation_completed', False):
        st.success("✉️ 麻理からの手紙が届きました！")
        st.success("🎉 初回手紙の生成が完了しました！")
        
        # チュートリアル完了メッセージ（該当する場合）
        tutorial_manager = managers.get('tutorial_manager')
        if tutorial_manager and tutorial_manager.get_current_step() <= 4:
            st.success("🎉 チュートリアルステップ4完了！")
        
        # 生成された手紙を表示
        if 'generated_letter_content' in st.session_state:
            with st.expander("📝 生成された手紙", expanded=True):
                st.markdown(st.session_state.generated_letter_content.replace("\n", "\n\n"))

        st.info("💡 次回からは通常の手紙リクエスト（時間指定あり）になります。")
        st.info("💡 手紙は履歴に保存されました。下の履歴から確認できます。")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📮 通常の手紙リクエストに切り替える", type="primary", use_container_width=True):
                # 手紙生成完了状態をクリア
                st.session_state.letter_generation_completed = False
                
                # 生成された手紙データをクリア
                if 'generated_letter_content' in st.session_state:
                    del st.session_state.generated_letter_content
                if 'generated_letter_theme' in st.session_state:
                    del st.session_state.generated_letter_theme
                
                # フォーム状態をクリア
                if 'letter_form_submitted' in st.session_state:
                    del st.session_state.letter_form_submitted
                if 'letter_form_theme' in st.session_state:
                    del st.session_state.letter_form_theme
                if 'letter_form_generation_hour' in st.session_state:
                    del st.session_state.letter_form_generation_hour
                if 'letter_form_is_instant' in st.session_state:
                    del st.session_state.letter_form_is_instant
                
                # ユーザーデータキャッシュをリフレッシュ
                st.session_state.force_refresh_user_data = True
                
                logger.info("📮 通常手紙リクエストに切り替えボタンが押されました")
                st.success("✅ 通常の手紙リクエストが利用できます！")
                st.rerun()
        
        st.divider()
        
        # 手紙生成完了後は履歴のみ表示してフォームは非表示
        show_letter_form = False
    else:
        show_letter_form = True
    
    # 手紙リクエストフォーム（生成完了後は非表示）
    if show_letter_form and can_request_letter:
        # 好感度が十分な場合または初回の場合のフォーム表示
        if not can_instant_generate:
            st.success(f"💝 好感度{current_affection}で手紙をリクエストできます！")
        

        
        # 📝 **フォームをページに直接埋め込み**
        
        # テーマ入力フィールドのautocomplete属性を有効化
        theme_input_css = """
        <style>
        .stTextInput input[type="text"] {
            autocomplete: on !important;
        }
        </style>
        """
        st.markdown(theme_input_css, unsafe_allow_html=True)
        
        # 条件分岐による表示切り替え
        if can_instant_generate:
            # 初回ユーザー向け
            st.success("🎉 **初回特典**: 好感度に関係なく手紙をリクエストできます！")
            st.write("💌 **初回手紙生成**")
            
            theme = st.text_input(
                "手紙に書いてほしいテーマ",
                placeholder="例: 今日見た美しい夕日について",
                help="麻理に書いてほしい内容やテーマを入力してください",
                key="instant_theme_input"
            )
            
            st.info("📘 初回は即座に手紙を生成します（時間選択不要）")
            
            if st.button("💌 初回手紙を生成する", type="primary", use_container_width=True):
                if theme:
                    st.session_state.letter_form_submitted = True
                    st.session_state.letter_form_theme = theme
                    st.session_state.letter_form_generation_hour = None
                    st.session_state.letter_form_is_instant = True
                    st.success("✅ 初回手紙リクエストを受付しました！")
                    st.rerun()
                else:
                    st.error("❌ テーマを入力してください。")
        
        else:
            # 通常ユーザー向け
            st.success(f"💝 好感度{current_affection}で手紙をリクエストできます！")
            st.write("📮 **手紙のリクエスト**")
            
            theme = st.text_input(
                "手紙に書いてほしいテーマ",
                placeholder="例: 今日見た美しい夕日について",
                help="麻理に書いてほしい内容やテーマを入力してください",
                key="scheduled_theme_input"
            )
            
            generation_hour = st.selectbox(
                "手紙を書いてほしい時間",
                options=[2, 3, 4],
                format_func=lambda x: f"深夜{x}時頃",
                help="静かな夜の時間に、ゆっくりと手紙を綴ります"
            )
            
            if st.button("📮 この内容でお願いする", type="primary", use_container_width=True):
                if theme:
                    st.session_state.letter_form_submitted = True
                    st.session_state.letter_form_theme = theme
                    st.session_state.letter_form_generation_hour = generation_hour
                    st.session_state.letter_form_is_instant = False
                    st.success("✅ 手紙リクエストを受付しました！")
                    st.rerun()
                else:
                    st.error("❌ テーマを入力してください。")
        
        # 📋 **統一的なフォーム処理（条件分岐なし）**
        
        # セッション状態から統一的に取得
        form_submitted = st.session_state.get('letter_form_submitted', False)
        form_theme = st.session_state.get('letter_form_theme', '')
        form_generation_hour = st.session_state.get('letter_form_generation_hour', None)
        form_is_instant = st.session_state.get('letter_form_is_instant', False)
        
        # 処理完了チェックは上部の条件分岐で既に処理済み
        
        # 統一的な処理ロジック
        if form_submitted:
            
            # テーマ検証
            if not form_theme:
                st.error("❌ テーマを入力してください。")
            else:
                st.success(f"✅ テーマ検証OK: '{form_theme}'")
                
                # データベース状態確認
                try:
                    current_user_data = run_async(user_manager.storage.get_user_data(user_id))
                    current_letters_count = len(current_user_data.get("letters", {}))
                    is_database_first_time = current_letters_count == 0
                    
                    st.info(f"📊 データベース状態: 手紙数={current_letters_count}, DB初回判定={is_database_first_time}")
                    
                    # 処理タイプの決定（フォームの判定とDBの判定を照合）
                    if form_is_instant and is_database_first_time:
                        # 初回手紙生成（自動実行）
                        st.info("🚀 **初回手紙生成を実行中...**")
                        
                        with st.spinner("初回手紙を生成中..."):
                            try:
                                # 手紙生成
                                instant_letter = generate_tutorial_letter(form_theme, managers)
                                
                                # データベースに保存
                                from datetime import datetime
                                current_date = datetime.now().strftime("%Y-%m-%d")
                                
                                if "letters" not in current_user_data:
                                    current_user_data["letters"] = {}
                                
                                current_user_data["letters"][current_date] = {
                                    "theme": form_theme,
                                    "content": instant_letter,
                                    "status": "completed",
                                    "generation_hour": "instant",
                                    "created_at": datetime.now().isoformat(),
                                    "type": "instant"
                                }
                                
                                # プロフィール更新
                                if "profile" not in current_user_data:
                                    current_user_data["profile"] = {}
                                current_user_data["profile"]["total_letters"] = len(current_user_data["letters"])
                                current_user_data["profile"]["last_letter_date"] = current_date
                                
                                # 保存
                                run_async(user_manager.storage.update_user_data(user_id, current_user_data))
                                
                                # チュートリアル完了
                                tutorial_manager = managers.get('tutorial_manager')
                                if tutorial_manager and tutorial_manager.get_current_step() <= 4:
                                    tutorial_manager.complete_step(4)
                                
                                # 生成された手紙をセッション状態に保存（rerun後も表示するため）
                                st.session_state.generated_letter_content = instant_letter
                                st.session_state.generated_letter_theme = form_theme
                                
                                # 処理完了フラグを設定
                                st.session_state.letter_generation_completed = True
                                
                                # セッション状態クリア
                                st.session_state.letter_form_submitted = False
                                for key in ['letter_form_theme', 'letter_form_generation_hour', 'letter_form_is_instant']:
                                    if key in st.session_state:
                                        del st.session_state[key]
                                
                                # キャッシュ更新
                                cache_key = f"user_data_{user_id}"
                                st.session_state[cache_key] = current_user_data
                                
                                # 手紙生成完了後は必ずrerunして状態を更新
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ 手紙生成エラー: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                    
                    elif not form_is_instant and form_generation_hour:
                        # 通常の手紙リクエスト（自動実行）
                        st.info("📮 **通常の手紙リクエストを実行中...**")
                        
                        with st.spinner("リクエストを送信中..."):
                            try:
                                success, message = run_async(
                                    request_manager.submit_request(
                                        user_id, form_theme, form_generation_hour, affection=current_affection
                                    )
                                )
                                
                                if success:
                                    st.success(message)
                                    st.info("💡 手紙は指定した時間に生成されます。履歴から確認してください。")
                                    
                                    # セッション状態クリア
                                    st.session_state.letter_form_submitted = False
                                    for key in ['letter_form_theme', 'letter_form_generation_hour', 'letter_form_is_instant']:
                                        if key in st.session_state:
                                            del st.session_state[key]
                                    
                                    # 通常手紙リクエスト完了後もrerunして状態を更新
                                    st.rerun()
                                else:
                                    st.error(message)
                            
                            except Exception as e:
                                st.error(f"❌ リクエスト送信エラー: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                    
                    else:
                        # 不整合状態
                        st.warning("⚠️ フォーム状態とデータベース状態に不整合があります")
                        st.write(f"フォーム即時判定: {form_is_instant}, DB初回判定: {is_database_first_time}")
                        
                        if st.button("🔄 状態をリセット", key="reset_inconsistent_state"):
                            st.session_state.letter_form_submitted = False
                            for key in ['letter_form_theme', 'letter_form_generation_hour', 'letter_form_is_instant']:
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.success("✅ 状態をリセットしました。次回アクセス時に反映されます。")
                
                except Exception as e:
                    st.error(f"❌ データベースアクセスエラー: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    
    # 好感度に関係なく履歴は常に表示
    st.divider()

    # --- 過去の手紙一覧 ---
    st.subheader("あなたへの手紙")
    with st.spinner("手紙の履歴を読み込んでいます..."):
        try:
            history = run_async(user_manager.get_user_letter_history(user_id, limit=10))
        except Exception as e:
            logger.error(f"手紙履歴取得エラー: {e}")
            history = []

    if not history:
        st.info("まだ手紙はありません。最初の手紙をリクエストしてみましょう。")
    else:
        for letter_info in history:
            date = letter_info.get("date")
            theme = letter_info.get("theme")
            status = letter_info.get("status", "unknown")

            with st.expander(f"{date} - テーマ: {theme} ({status})"):
                if status == "completed":
                    try:
                        user_data = run_async(user_manager.storage.get_user_data(user_id))
                        content = user_data.get("letters", {}).get(date, {}).get("content", "内容の取得に失敗しました。")
                        st.markdown(content.replace("\n", "\n\n"))
                        
                        # 手紙を会話に反映するボタン
                        col1, col2 = st.columns([3, 1])
                        with col2:
                            if st.button(f"💬 会話に反映", key=f"reflect_{date}", help="この手紙の内容を麻理との会話で話題にします"):
                                # 手紙の内容をメモリに追加
                                letter_summary = f"手紙のテーマ「{theme}」について麻理が書いた内容: {content[:200]}..."
                                memory_notification = st.session_state.memory_manager.add_important_memory("letter_content", letter_summary)
                                
                                # 手紙について話すメッセージを自動追加
                                letter_message = f"この前書いてくれた「{theme}」についての手紙、読ませてもらったよ。"
                                st.session_state.chat['messages'].append({"role": "user", "content": letter_message})
                                
                                # 麻理の応答を生成
                                response = f"あの手紙、読んでくれたんだ...。「{theme}」について書いたとき、あなたのことを思いながら一生懸命考えたんだ。どう思った？"
                                st.session_state.chat['messages'].append({"role": "assistant", "content": response})
                                
                                # 特別な記憶の通知をセッション状態に保存
                                st.session_state.memory_notifications.append(memory_notification)
                                st.success("手紙の内容が会話に反映されました！チャットタブで確認してください。")
                                st.rerun()
                                
                    except Exception as e:
                        logger.error(f"手紙内容取得エラー: {e}")
                        st.error("手紙の内容を読み込めませんでした。")
                elif status == "pending":
                    st.write("この手紙はまだ生成中です。")
                else:
                    st.write("この手紙は生成に失敗しました。")


# --- ▼▼▼ 3. メイン実行ブロック ▼▼▼ ---

def main():
    """メイン関数"""
    # event loopの安全な設定
    try:
        # 既存のループがあるかチェック
        asyncio.get_running_loop()
    except RuntimeError:
        # 実行中のループがない場合のみ新しいループを設定
        if sys.platform.startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # rerun検出機能（ループ防止版）
    current_run_id = id(st.session_state)
    last_run_id = st.session_state.get('last_run_id', None)
    
    # 初回実行時のみrerun検出を記録
    if last_run_id != current_run_id:
        if last_run_id is not None:  # 初回実行時はログ出力しない
            logger.debug(f"Rerun検出: 前回={last_run_id}, 今回={current_run_id}")
        st.session_state.last_run_id = current_run_id
    
    # 全ての依存モジュールを初期化（Cookie処理を含む）
    cached_managers = initialize_cached_managers()
    session_managers = initialize_session_managers()  # ここでCookie処理が実行される
    managers = {**cached_managers, **session_managers}
    
    # セッションステートを初期化（残りの処理）
    initialize_session_state(managers)

    # CSSを適用
    inject_custom_css()

    # 背景を毎回更新（rerun時の白背景問題を解決）
    try:
        current_theme = st.session_state.chat['scene_params'].get('theme', 'default')
        
        # 背景更新が必要かチェック
        last_theme = st.session_state.get('_last_background_theme', None)
        force_background_update = st.session_state.get('force_background_update', False)
        
        if last_theme != current_theme or force_background_update or not st.session_state.get('_initial_background_set', False):
            logger.info(f"背景更新: テーマ '{current_theme}' (前回: {last_theme})")
            
            # 背景を更新
            update_background(managers['scene_manager'], current_theme)
            st.session_state._initial_background_set = True
            st.session_state._last_background_theme = current_theme
            st.session_state._last_rendered_tab = None  # タブ切り替え検出をリセット
            
            # 強制更新フラグをクリア
            if force_background_update:
                st.session_state.force_background_update = False
        else:
            # テーマが変わっていない場合でも、rerun時は背景を再適用
            logger.debug(f"背景再適用: テーマ '{current_theme}'")
            update_background(managers['scene_manager'], current_theme)
                
    except Exception as e:
        logger.error(f"背景設定でエラーが発生: {e}")
        import traceback
        logger.error(f"背景設定エラーの詳細: {traceback.format_exc()}")
        # エラーが発生してもアプリケーションは継続

    # チュートリアル機能の初期化
    tutorial_manager = managers['tutorial_manager']
    
    # 初回訪問時のウェルカムダイアログ
    if tutorial_manager.should_show_tutorial():
        # チュートリアルダイアログ表示前にチャットセッション状態を保護
        if 'chat' not in st.session_state:
            logger.warning("チュートリアルダイアログ表示前にチャットセッションが存在しません - 初期化します")
            initial_message = {"role": "assistant", "content": "何の用？遊びに来たの？", "is_initial": True}
            st.session_state.chat = {
                "messages": [initial_message],
                "affection": 30,
                "scene_params": {"theme": "default"},
                "limiter_state": managers["rate_limiter"].create_limiter_state(),
                "scene_change_pending": None,
                "ura_mode": False
            }
            logger.info("チュートリアルダイアログ表示前にチャットセッションを初期化しました")
        elif 'messages' not in st.session_state.chat:
            logger.warning("チュートリアルダイアログ表示前にメッセージリストが存在しません - 初期化します")
            initial_message = {"role": "assistant", "content": "何の用？遊びに来たの？", "is_initial": True}
            st.session_state.chat['messages'] = [initial_message]
            logger.info("チュートリアルダイアログ表示前にメッセージリストを初期化しました")
        elif not any(msg.get('is_initial', False) for msg in st.session_state.chat['messages']):
            logger.warning("チュートリアルダイアログ表示前に初期メッセージが見つかりません - 復元します")
            initial_message = {"role": "assistant", "content": "何の用？遊びに来たの？", "is_initial": True}
            st.session_state.chat['messages'].insert(0, initial_message)
            logger.info("チュートリアルダイアログ表示前に初期メッセージを復元しました")
        
        # チュートリアル表示中フラグを設定
        st.session_state.tutorial_dialog_showing = True
        
        tutorial_manager.render_welcome_dialog()
        tutorial_manager.mark_tutorial_shown()
        
        # チュートリアル表示完了フラグをクリア
        st.session_state.tutorial_dialog_showing = False
    
    # チュートリアル開始/スキップの処理
    if st.session_state.get('tutorial_start_requested', False):
        st.session_state.tutorial_start_requested = False
        
        # チュートリアル処理中フラグを設定
        st.session_state.tutorial_processing = True
        
        # チャットセッション状態を確実に初期化
        if 'chat' not in st.session_state:
            logger.warning("チュートリアル開始時にチャットセッションが存在しません - 初期化します")
            initial_message = {"role": "assistant", "content": "何の用？遊びに来たの？", "is_initial": True}
            st.session_state.chat = {
                "messages": [initial_message],
                "affection": 30,
                "scene_params": {"theme": "default"},
                "limiter_state": managers["rate_limiter"].create_limiter_state(),
                "scene_change_pending": None,
                "ura_mode": False
            }
            logger.info("チュートリアル開始時にチャットセッションを初期化しました")
        elif 'messages' not in st.session_state.chat:
            logger.warning("チュートリアル開始時にメッセージリストが存在しません - 初期化します")
            initial_message = {"role": "assistant", "content": "何の用？遊びに来たの？", "is_initial": True}
            st.session_state.chat['messages'] = [initial_message]
            logger.info("チュートリアル開始時にメッセージリストを初期化しました")
        else:
            # 初期メッセージを確実に保護
            messages = st.session_state.chat['messages']
            if not any(msg.get('is_initial', False) for msg in messages):
                initial_message = {"role": "assistant", "content": "何の用？遊びに来たの？", "is_initial": True}
                st.session_state.chat['messages'].insert(0, initial_message)
                logger.info("チュートリアル開始時に初期メッセージを復元しました")
        
        # チュートリアル処理完了フラグをクリア（チャット履歴表示を再開）
        st.session_state.tutorial_processing = False
        
        # チュートリアル開始の案内
        st.success("📘 チュートリアルを開始しました！青い案内ボックスに従って進めてください。")
    
    if st.session_state.get('tutorial_skip_requested', False):
        st.session_state.tutorial_skip_requested = False
        
        # チュートリアル処理中フラグを設定
        st.session_state.tutorial_processing = True
        
        # チャットセッション状態を確実に初期化
        if 'chat' not in st.session_state:
            logger.warning("チュートリアルスキップ時にチャットセッションが存在しません - 初期化します")
            initial_message = {"role": "assistant", "content": "何の用？遊びに来たの？", "is_initial": True}
            st.session_state.chat = {
                "messages": [initial_message],
                "affection": 30,
                "scene_params": {"theme": "default"},
                "limiter_state": managers["rate_limiter"].create_limiter_state(),
                "scene_change_pending": None,
                "ura_mode": False
            }
            logger.info("チュートリアルスキップ時にチャットセッションを初期化しました")
        elif 'messages' not in st.session_state.chat:
            logger.warning("チュートリアルスキップ時にメッセージリストが存在しません - 初期化します")
            initial_message = {"role": "assistant", "content": "何の用？遊びに来たの？", "is_initial": True}
            st.session_state.chat['messages'] = [initial_message]
            logger.info("チュートリアルスキップ時にメッセージリストを初期化しました")
        else:
            # 初期メッセージを確実に保護
            messages = st.session_state.chat['messages']
            if not any(msg.get('is_initial', False) for msg in messages):
                initial_message = {"role": "assistant", "content": "何の用？遊びに来たの？", "is_initial": True}
                st.session_state.chat['messages'].insert(0, initial_message)
                logger.info("チュートリアルスキップ時に初期メッセージを復元しました")
        
        # チュートリアル処理完了フラグをクリア（チャット履歴表示を再開）
        st.session_state.tutorial_processing = False
        
        st.success("⏭️ チュートリアルをスキップしました。麻理との会話をお楽しみください！")
    
    # チュートリアルステップ4の場合、手紙タブを強調表示
    tutorial_manager = managers['tutorial_manager']
    current_step = tutorial_manager.get_current_step()
    
    # タブ名を動的に設定
    if current_step == 4 and not tutorial_manager.is_step_completed(4):
        # ステップ4の場合、手紙タブを強調
        letter_tab_name = "👉 ✉️ 手紙を受け取る ← ここをクリック！"
        
        # 手紙タブ強調のCSS
        tab_highlight_css = """
        <style>
        /* 手紙タブの強調表示 */
        .stTabs [data-baseweb="tab-list"] button:nth-child(2) {
            background: linear-gradient(45deg, #ff6b6b, #feca57) !important;
            color: white !important;
            font-weight: bold !important;
            animation: tabPulse 2s ease-in-out infinite !important;
            border: 2px solid #ff6b6b !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4) !important;
        }
        
        .stTabs [data-baseweb="tab-list"] button:nth-child(2):hover {
            transform: scale(1.05) !important;
            box-shadow: 0 6px 20px rgba(255, 107, 107, 0.6) !important;
        }
        
        @keyframes tabPulse {
            0%, 100% { 
                transform: scale(1);
                box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
            }
            50% { 
                transform: scale(1.02);
                box-shadow: 0 6px 25px rgba(255, 107, 107, 0.7);
            }
        }
        
        /* 矢印アニメーション */
        .tutorial-arrow {
            position: fixed;
            top: 60px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1000;
            font-size: 30px;
            color: #ff6b6b;
            animation: arrowBounce 1.5s ease-in-out infinite;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        @keyframes arrowBounce {
            0%, 100% { transform: translateX(-50%) translateY(0px); }
            50% { transform: translateX(-50%) translateY(-10px); }
        }
        
        @media (max-width: 768px) {
            .tutorial-arrow {
                top: 50px;
                font-size: 24px;
            }
        }
        </style>
        """
        
        st.markdown(tab_highlight_css, unsafe_allow_html=True)
        
        # 矢印の表示
        arrow_html = """
        <div class="tutorial-arrow">
            ↓ 手紙タブをクリック！ ↓
        </div>
        """
        
        st.markdown(arrow_html, unsafe_allow_html=True)
    else:
        letter_tab_name = "✉️ 手紙を受け取る"
    
    # タブ名を定義
    tab_names = ["💬 麻理と話す", letter_tab_name, "📘 チュートリアル"]
    
    # タブ作成（Streamlitの標準動作に任せる）
    chat_tab, letter_tab, tutorial_tab = st.tabs(tab_names)

    with chat_tab:
        # 現在のタブを記録（タブ切り替え検出用）
        if st.session_state.get('last_active_tab') != "chat":
            st.session_state.last_active_tab = "chat"
            # タブ切り替え時は背景更新フラグを設定
            st.session_state._background_needs_update = True
            logger.debug("チャットタブがアクティブになりました")
        
        render_chat_tab(managers)

    with letter_tab:
        # 現在のタブを記録（タブ切り替え検出用）
        if st.session_state.get('last_active_tab') != "letter":
            st.session_state.last_active_tab = "letter"
            # 手紙タブでは背景更新は不要
            st.session_state._last_rendered_tab = "letter"
            logger.debug("手紙タブがアクティブになりました")
        
        render_letter_tab(managers)
    
    with tutorial_tab:
        # 現在のタブを記録（タブ切り替え検出用）
        if st.session_state.get('last_active_tab') != "tutorial":
            st.session_state.last_active_tab = "tutorial"
            logger.debug("チュートリアルタブがアクティブになりました")
        
        tutorial_manager.render_tutorial_tab()

    # ポチのコンポーネントはチャットタブ内に移動

# 重複するmain関数を削除（最初のmain関数のみ使用）

if __name__ == "__main__":
    if not Config.validate_config():
        logger.critical("手紙機能の設定にエラーがあります。アプリケーションを起動できません。")
    else:
        main()