"""
ポチ（犬）アシスタントコンポーネント
画面右下に固定配置され、本音表示機能を提供する
"""
import streamlit as st
import logging

logger = logging.getLogger(__name__)

class DogAssistant:
    """ポチ（犬）アシスタントクラス"""
    
    def __init__(self):
        """初期化"""
        self.default_message = "ポチは麻理の本音を察知したようだ・・・"
        self.active_message = "ワンワン！本音が見えてるワン！"
    
    def render_dog_component(self, tutorial_manager=None):
        """画面右下に固定配置される犬のコンポーネントを描画"""
        try:
            # 犬のボタン表示前にチャットセッション状態を確認
            if 'chat' not in st.session_state:
                logger.warning("犬のコンポーネント表示前にチャットセッションが存在しません - 初期化します")
                initial_message = {"role": "assistant", "content": "何の用？遊びに来たの？", "is_initial": True}
                st.session_state.chat = {
                    "messages": [initial_message],
                    "affection": 30,
                    "scene_params": {"theme": "default"},
                    "limiter_state": {},
                    "scene_change_pending": None,
                    "ura_mode": False
                }
                logger.info("犬のコンポーネント表示前にチャットセッションを初期化しました")
            elif 'messages' not in st.session_state.chat:
                logger.warning("犬のコンポーネント表示前にメッセージリストが存在しません - 初期化します")
                initial_message = {"role": "assistant", "content": "何の用？遊びに来たの？", "is_initial": True}
                st.session_state.chat['messages'] = [initial_message]
                logger.info("犬のコンポーネント表示前にメッセージリストを初期化しました")
            elif not any(msg.get('is_initial', False) for msg in st.session_state.chat['messages']):
                logger.warning("犬のコンポーネント表示前に初期メッセージが見つかりません - 復元します")
                initial_message = {"role": "assistant", "content": "何の用？遊びに来たの？", "is_initial": True}
                st.session_state.chat['messages'].insert(0, initial_message)
                logger.info("犬のコンポーネント表示前に初期メッセージを復元しました")
            # 犬のコンポーネントのCSS（レスポンシブ対応）
            dog_css = """
            <style>
            .dog-assistant-container {
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 1000;
                display: flex;
                flex-direction: column;
                align-items: flex-start;
                pointer-events: none;
                transform: translateX(-50px);
            }
            
            .dog-speech-bubble {
                background-color: rgba(255, 255, 255, 0.95);
                color: #333;
                padding: 10px 15px;
                border-radius: 20px;
                font-size: 13px;
                margin-bottom: 10px;
                position: relative;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(0,0,0,0.1);
                max-width: 200px;
                word-wrap: break-word;
                animation: bubbleFloat 3s ease-in-out infinite;
                pointer-events: auto;
            }
            
            .dog-speech-bubble::after {
                content: '';
                position: absolute;
                bottom: -8px;
                left: 20px;
                width: 0;
                height: 0;
                border-left: 8px solid transparent;
                border-right: 8px solid transparent;
                border-top: 8px solid rgba(255, 255, 255, 0.95);
            }
            
            .dog-button {
                background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
                border: none;
                border-radius: 50%;
                width: 70px;
                height: 70px;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(255, 154, 158, 0.4);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 35px;
                pointer-events: auto;
                animation: dogBounce 2s ease-in-out infinite;
            }
            
            .dog-button:hover {
                transform: scale(1.1);
                box-shadow: 0 6px 20px rgba(255, 154, 158, 0.6);
                background: linear-gradient(135deg, #ff6b6b 0%, #feca57 50%, #ff9ff3 100%);
            }
            
            .dog-button:active {
                transform: scale(0.95);
            }
            
            .dog-button.active {
                background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
                animation: dogActive 1s ease-in-out infinite;
            }
            
            @keyframes bubbleFloat {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-5px); }
            }
            
            @keyframes dogBounce {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-3px); }
            }
            
            @keyframes dogActive {
                0%, 100% { transform: scale(1) rotate(0deg); }
                25% { transform: scale(1.05) rotate(-2deg); }
                75% { transform: scale(1.05) rotate(2deg); }
            }
            
            /* レスポンシブ対応 */
            @media (max-width: 768px) {
                .dog-assistant-container {
                    bottom: 15px;
                    right: 15px;
                    transform: translateX(-40px);
                }
                
                .dog-speech-bubble {
                    max-width: 150px;
                    font-size: 12px;
                    padding: 8px 12px;
                }
                
                .dog-button {
                    width: 60px;
                    height: 60px;
                    font-size: 30px;
                }
            }
            
            @media (max-width: 480px) {
                .dog-assistant-container {
                    bottom: 10px;
                    right: 10px;
                    transform: translateX(-30px);
                }
                
                .dog-speech-bubble {
                    max-width: 120px;
                    font-size: 11px;
                    padding: 6px 10px;
                }
                
                .dog-button {
                    width: 50px;
                    height: 50px;
                    font-size: 25px;
                }
            }
            
            /* 画面が非常に小さい場合は吹き出しを非表示 */
            @media (max-width: 320px) {
                .dog-speech-bubble {
                    display: none;
                }
            }
            </style>
            """
            
            # 現在の状態を取得
            is_active = st.session_state.get('show_all_hidden', False)
            bubble_text = self.active_message if is_active else self.default_message
            button_class = "dog-button active" if is_active else "dog-button"
            
            # JavaScriptでクリックイベントを処理
            dog_js = f"""
            <script>
            function toggleDogMode() {{
                // Streamlitのセッション状態を更新するためのトリガー
                const event = new CustomEvent('dogButtonClick', {{
                    detail: {{ active: {str(is_active).lower()} }}
                }});
                window.dispatchEvent(event);
                
                // ボタンの状態を即座に更新
                const button = document.querySelector('.dog-button');
                const bubble = document.querySelector('.dog-speech-bubble');
                
                if (button && bubble) {{
                    if ({str(is_active).lower()}) {{
                        button.classList.remove('active');
                        bubble.textContent = '{self.default_message}';
                    }} else {{
                        button.classList.add('active');
                        bubble.textContent = '{self.active_message}';
                    }}
                }}
            }}
            
            // ページ読み込み時にイベントリスナーを設定
            document.addEventListener('DOMContentLoaded', function() {{
                const button = document.querySelector('.dog-button');
                if (button) {{
                    button.addEventListener('click', toggleDogMode);
                }}
            }});
            
            // Streamlitの再描画後にもイベントリスナーを再設定
            setTimeout(function() {{
                const button = document.querySelector('.dog-button');
                if (button) {{
                    button.addEventListener('click', toggleDogMode);
                }}
            }}, 100);
            </script>
            """
            
            # HTMLコンポーネント
            dog_html = f"""
            <div class="dog-assistant-container">
                <div class="dog-speech-bubble">
                    {bubble_text}
                </div>
                <button class="{button_class}" title="ポチが麻理の本音を察知します" onclick="toggleDogMode()">
                    🐕
                </button>
            </div>
            """
            
            # HTMLコンポーネント（ボタン以外）を表示
            dog_display_html = f"""
            <div class="dog-assistant-container">
                <div class="dog-speech-bubble">
                    {bubble_text}
                </div>
                <div style="width: 70px; height: 70px; display: flex; align-items: center; justify-content: center;">
                    <!-- Streamlitボタンがここに配置される -->
                </div>
            </div>
            """
            
            st.markdown(dog_css + dog_display_html, unsafe_allow_html=True)
            
            # Streamlitボタンを固定位置に配置
            button_css = """
            <style>
            .dog-button-overlay {
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 1001;
                pointer-events: auto;
            }
            
            .dog-button-overlay .stButton > button {
                background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
                border: none;
                border-radius: 50%;
                width: 70px;
                height: 70px;
                font-size: 35px;
                color: white;
                box-shadow: 0 4px 15px rgba(255, 154, 158, 0.4);
                transition: all 0.3s ease;
                animation: dogBounce 2s ease-in-out infinite;
            }
            
            .dog-button-overlay .stButton > button:hover {
                transform: scale(1.1);
                box-shadow: 0 6px 20px rgba(255, 154, 158, 0.6);
            }
            
            @keyframes dogBounce {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-3px); }
            }
            
            @media (max-width: 768px) {
                .dog-button-overlay {
                    bottom: 15px;
                    right: 15px;
                }
                
                .dog-button-overlay .stButton > button {
                    width: 60px;
                    height: 60px;
                    font-size: 30px;
                }
            }
            
            @media (max-width: 480px) {
                .dog-button-overlay {
                    bottom: 10px;
                    right: 10px;
                }
                
                .dog-button-overlay .stButton > button {
                    width: 50px;
                    height: 50px;
                    font-size: 25px;
                }
            }
            </style>
            """
            
            st.markdown(button_css, unsafe_allow_html=True)
            st.markdown('<div class="dog-button-overlay">', unsafe_allow_html=True)
            
            # ボタンクリック処理
            button_key = f"dog_fixed_{is_active}"
            button_help = "本音を隠す" if is_active else "本音を見る"
            if st.button("🐕", key=button_key, help=button_help):
                self.handle_dog_button_click(tutorial_manager)
                logger.info("右下の犬のボタンがクリックされました")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            logger.debug(f"犬のコンポーネントを描画しました (active: {is_active})")
            
        except Exception as e:
            logger.error(f"犬のコンポーネント描画エラー: {e}")
    
    def handle_dog_button_click(self, tutorial_manager=None):
        """犬のボタンクリック処理（無限ループ防止版）"""
        try:
            # 本音表示機能のトリガー
            if 'show_all_hidden' not in st.session_state:
                st.session_state.show_all_hidden = False
            
            # 現在の状態を取得
            current_state = st.session_state.show_all_hidden
            new_state = not current_state
            
            # 状態を更新
            st.session_state.show_all_hidden = new_state
            # チャット履歴の強制再表示フラグを設定
            st.session_state.show_all_hidden_changed = True
            logger.info(f"犬のボタン状態変更: {current_state} -> {new_state}")
            
            # 全メッセージのフリップ状態を即座に更新
            if 'message_flip_states' not in st.session_state:
                st.session_state.message_flip_states = {}
            
            # 現在のメッセージに対してフリップ状態を設定
            if 'chat' in st.session_state and 'messages' in st.session_state.chat:
                # 初期メッセージが存在することを確認
                messages = st.session_state.chat['messages']
                if not any(msg.get('is_initial', False) for msg in messages):
                    logger.warning("犬のボタン押下時に初期メッセージが見つかりません - 復元します")
                    initial_message = {"role": "assistant", "content": "何の用？遊びに来たの？", "is_initial": True}
                    st.session_state.chat['messages'].insert(0, initial_message)
                    logger.info("犬のボタン押下時に初期メッセージを復元しました")
                
                for i, message in enumerate(st.session_state.chat['messages']):
                    if message['role'] == 'assistant':
                        message_id = f"msg_{i}"
                        st.session_state.message_flip_states[message_id] = new_state
            else:
                logger.warning("犬のボタン押下時にチャットセッションが存在しません - 初期化します")
                # チャットセッションが存在しない場合は初期化
                initial_message = {"role": "assistant", "content": "何の用？遊びに来たの？", "is_initial": True}
                if 'chat' not in st.session_state:
                    st.session_state.chat = {
                        "messages": [initial_message],
                        "affection": 30,
                        "scene_params": {"theme": "default"},
                        "limiter_state": {},
                        "scene_change_pending": None,
                        "ura_mode": False
                    }
                else:
                    st.session_state.chat['messages'] = [initial_message]
                logger.info("犬のボタン押下時にチャットセッションを初期化しました")
            
            # チュートリアルステップ2を完了（tutorial_managerが渡された場合）
            if tutorial_manager:
                tutorial_manager.check_step_completion(2, True)
            
            # 通知メッセージ（一度だけ表示）
            if new_state:
                st.success("🐕 ポチが麻理の本音を察知しました！")
            else:
                st.info("🐕 ポチが通常モードに戻りました。")
            
            logger.info(f"犬のボタン状態変更完了: {current_state} → {new_state}")
            
            # 状態変更を確実に反映するため、強制的に再実行
            st.rerun()
            
        except Exception as e:
            logger.error(f"犬のボタンクリック処理エラー: {e}")
    
    def render_with_streamlit_button(self):
        """Streamlitのボタンを使用した代替実装（フォールバック用）"""
        try:
            # 固定位置のCSS
            fallback_css = """
            <style>
            .dog-fallback-container {
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 1000;
                background: rgba(255, 255, 255, 0.9);
                border-radius: 15px;
                padding: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                backdrop-filter: blur(10px);
            }
            
            @media (max-width: 768px) {
                .dog-fallback-container {
                    bottom: 15px;
                    right: 15px;
                    padding: 8px;
                }
            }
            </style>
            """
            
            st.markdown(fallback_css, unsafe_allow_html=True)
            
            # コンテナの開始
            st.markdown('<div class="dog-fallback-container">', unsafe_allow_html=True)
            
            # 状態表示
            is_active = st.session_state.get('show_all_hidden', False)
            status_text = "本音モード中" if is_active else "通常モード"
            st.caption(f"🐕 {status_text}")
            
            # ボタン
            button_text = "🔄 戻す" if is_active else "🐕 本音を見る"
            if st.button(button_text, key="dog_assistant_btn"):
                # チュートリアルマネージャーを取得（可能な場合）
                tutorial_manager = None
                try:
                    # セッション状態からチュートリアルマネージャーを取得する試み
                    # （完全ではないが、フォールバック用）
                    pass
                except:
                    pass
                
                self.handle_dog_button_click(tutorial_manager)
                # st.rerun()を削除 - 状態変更により自動的に再描画される
            
            # コンテナの終了
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"犬のフォールバック描画エラー: {e}")
    
    def get_current_state(self):
        """現在の犬の状態を取得"""
        return {
            'is_active': st.session_state.get('show_all_hidden', False),
            'message': self.active_message if st.session_state.get('show_all_hidden', False) else self.default_message
        }