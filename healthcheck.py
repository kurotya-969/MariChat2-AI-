#!/usr/bin/env python3
"""
Streamlitアプリケーション用ヘルスチェックスクリプト
Docker環境でのヘルスチェックに使用
"""

import sys
import urllib.request
import urllib.error
import json
import time

def check_streamlit_health(host="localhost", port=8501, timeout=10):
    """
    Streamlitアプリケーションのヘルスチェックを実行
    
    Args:
        host (str): ホスト名
        port (int): ポート番号
        timeout (int): タイムアウト秒数
    
    Returns:
        bool: ヘルスチェック成功時True
    """
    try:
        # Streamlitのヘルスチェックエンドポイントを確認
        health_url = f"http://{host}:{port}/_stcore/health"
        
        request = urllib.request.Request(health_url)
        request.add_header('User-Agent', 'HealthCheck/1.0')
        
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status == 200:
                print(f"✅ Streamlitアプリケーションは正常に動作しています (ポート: {port})")
                return True
            else:
                print(f"❌ ヘルスチェック失敗: HTTPステータス {response.status}")
                return False
                
    except urllib.error.URLError as e:
        print(f"❌ 接続エラー: {e}")
        return False
    except Exception as e:
        print(f"❌ ヘルスチェックエラー: {e}")
        return False

def check_app_responsiveness(host="localhost", port=8501, timeout=10):
    """
    アプリケーションの応答性をチェック
    
    Args:
        host (str): ホスト名
        port (int): ポート番号
        timeout (int): タイムアウト秒数
    
    Returns:
        bool: 応答性チェック成功時True
    """
    try:
        # メインページへのアクセスを試行
        main_url = f"http://{host}:{port}/"
        
        request = urllib.request.Request(main_url)
        request.add_header('User-Agent', 'HealthCheck/1.0')
        
        start_time = time.time()
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_time = time.time() - start_time
            
            if response.status == 200:
                print(f"✅ アプリケーション応答時間: {response_time:.2f}秒")
                return True
            else:
                print(f"❌ アプリケーション応答エラー: HTTPステータス {response.status}")
                return False
                
    except urllib.error.URLError as e:
        print(f"❌ アプリケーション接続エラー: {e}")
        return False
    except Exception as e:
        print(f"❌ アプリケーション応答性チェックエラー: {e}")
        return False

def main():
    """メイン関数"""
    print("🔍 Streamlitアプリケーション ヘルスチェック開始...")
    
    # 基本的なヘルスチェック
    health_ok = check_streamlit_health()
    
    # アプリケーションの応答性チェック
    app_ok = check_app_responsiveness()
    
    # 結果の判定
    if health_ok and app_ok:
        print("🎉 全てのヘルスチェックが成功しました！")
        sys.exit(0)
    else:
        print("💥 ヘルスチェックに失敗しました")
        sys.exit(1)

if __name__ == "__main__":
    main()