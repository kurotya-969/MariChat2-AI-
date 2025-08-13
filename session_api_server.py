"""
FastAPIベースのセッション管理サーバー
HttpOnlyクッキーによるセキュアなセッション管理を提供
"""
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid
import json
import os
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mari Session API", version="1.0.0")

# CORS設定（Streamlitからのアクセスを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501", 
        "https://localhost:8501",
        "http://127.0.0.1:8501",
        "https://127.0.0.1:8501",
        "*"  # Hugging Face Spacesでの実行を考慮
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

class SessionManager:
    """セッション管理クラス"""
    
    def __init__(self, storage_path: str = "session_data"):
        self.storage_path = storage_path
        self.cookie_name = "mari_session_id"
        self.session_duration_days = 7
        self.cleanup_interval_hours = 24
        
        # ストレージディレクトリを作成
        os.makedirs(self.storage_path, exist_ok=True)
        
        # 最後のクリーンアップ時刻を記録するファイル
        self.cleanup_file = os.path.join(self.storage_path, "last_cleanup.json")
    
    def create_session(self) -> str:
        """新しいセッションを作成"""
        session_id = str(uuid.uuid4())
        
        session_data = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'last_access': datetime.now().isoformat(),
            'user_data': {}
        }
        
        # セッションファイルに保存
        session_file = os.path.join(self.storage_path, f"{session_id}.json")
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"新規セッション作成: {session_id[:8]}...")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """セッションデータを取得"""
        try:
            session_file = os.path.join(self.storage_path, f"{session_id}.json")
            
            if not os.path.exists(session_file):
                return None
            
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # 期限チェック
            last_access = datetime.fromisoformat(session_data.get('last_access', ''))
            expiry_time = last_access + timedelta(days=self.session_duration_days)
            
            if datetime.now() > expiry_time:
                # 期限切れセッションを削除
                os.remove(session_file)
                logger.info(f"期限切れセッション削除: {session_id[:8]}...")
                return None
            
            return session_data
            
        except Exception as e:
            logger.error(f"セッション取得エラー: {e}")
            return None
    
    def update_session_access(self, session_id: str) -> bool:
        """セッションの最終アクセス時刻を更新"""
        try:
            session_file = os.path.join(self.storage_path, f"{session_id}.json")
            
            if not os.path.exists(session_file):
                return False
            
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            session_data['last_access'] = datetime.now().isoformat()
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"セッションアクセス時刻更新エラー: {e}")
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """セッションを削除"""
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
    
    def cleanup_expired_sessions(self):
        """期限切れセッションのクリーンアップ"""
        try:
            current_time = datetime.now()
            cleaned_count = 0
            
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
                    
                    except Exception as e:
                        logger.warning(f"セッションファイル処理エラー {filename}: {e}")
            
            if cleaned_count > 0:
                logger.info(f"期限切れセッション {cleaned_count}件を削除")
                
        except Exception as e:
            logger.error(f"セッションクリーンアップエラー: {e}")

# セッションマネージャーのインスタンス
session_manager = SessionManager()

@app.post("/session/create")
async def create_session(response: Response):
    """新しいセッションを作成してCookieを設定"""
    try:
        # 期限切れセッションのクリーンアップ
        session_manager.cleanup_expired_sessions()
        
        # 新しいセッションを作成
        session_id = session_manager.create_session()
        
        # HttpOnlyクッキーを設定（Hugging Face Spaces対応）
        is_production = os.getenv("SPACE_ID") is not None  # Hugging Face Spacesの環境変数
        
        response.set_cookie(
            key=session_manager.cookie_name,
            value=session_id,
            max_age=session_manager.session_duration_days * 24 * 60 * 60,  # 7日間
            httponly=True,
            secure=is_production,  # 本番環境（HTTPS）でのみSecure属性を有効
            samesite="lax" if is_production else "strict"  # 本番環境では緩和
        )
        
        return {
            "status": "success",
            "session_id": session_id,
            "message": "セッションが作成されました"
        }
        
    except Exception as e:
        logger.error(f"セッション作成エラー: {e}")
        raise HTTPException(status_code=500, detail="セッション作成に失敗しました")

@app.get("/session/info")
async def get_session_info(request: Request):
    """現在のセッション情報を取得"""
    try:
        # CookieからセッションIDを取得
        session_id = request.cookies.get(session_manager.cookie_name)
        
        if not session_id:
            raise HTTPException(status_code=401, detail="セッションが見つかりません")
        
        # セッションデータを取得
        session_data = session_manager.get_session(session_id)
        
        if not session_data:
            raise HTTPException(status_code=401, detail="無効なセッションです")
        
        # アクセス時刻を更新
        session_manager.update_session_access(session_id)
        
        return {
            "status": "success",
            "session_id": session_id,
            "created_at": session_data.get('created_at'),
            "last_access": session_data.get('last_access')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"セッション情報取得エラー: {e}")
        raise HTTPException(status_code=500, detail="セッション情報の取得に失敗しました")

@app.post("/session/validate")
async def validate_session(request: Request):
    """セッションの有効性を検証"""
    try:
        # CookieからセッションIDを取得
        session_id = request.cookies.get(session_manager.cookie_name)
        
        if not session_id:
            return {"valid": False, "message": "セッションが見つかりません"}
        
        # セッションデータを取得
        session_data = session_manager.get_session(session_id)
        
        if not session_data:
            return {"valid": False, "message": "無効なセッションです"}
        
        # アクセス時刻を更新
        session_manager.update_session_access(session_id)
        
        return {
            "valid": True,
            "session_id": session_id,
            "message": "有効なセッションです"
        }
        
    except Exception as e:
        logger.error(f"セッション検証エラー: {e}")
        return {"valid": False, "message": "セッション検証に失敗しました"}

@app.delete("/session/delete")
async def delete_session(request: Request, response: Response):
    """セッションを削除"""
    try:
        # CookieからセッションIDを取得
        session_id = request.cookies.get(session_manager.cookie_name)
        
        if session_id:
            # セッションファイルを削除
            session_manager.delete_session(session_id)
        
        # Cookieを削除（Hugging Face Spaces対応）
        is_production = os.getenv("SPACE_ID") is not None
        
        response.delete_cookie(
            key=session_manager.cookie_name,
            httponly=True,
            secure=is_production,
            samesite="lax" if is_production else "strict"
        )
        
        return {
            "status": "success",
            "message": "セッションが削除されました"
        }
        
    except Exception as e:
        logger.error(f"セッション削除エラー: {e}")
        raise HTTPException(status_code=500, detail="セッション削除に失敗しました")

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")