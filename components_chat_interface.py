"""
チャットインターフェースコンポーネント
Streamlitのチャット機能を使用したメッセージ表示と入力処理
マスクアイコンとフリップアニメーション機能を含む
"""
import streamlit as st
import logging
import re
import uuid
from typing import List, Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class ChatInterface:
    """チャットインターフェースを管理するクラス"""
    
    def __init__(self, max_input_length: int = 1000):
        """
        Args:
            max_input_length: 入力メッセージの最大長
        """
        self.max_input_length = max_input_length
    
    def render_chat_history(self, messages: List[Dict[str, str]], 
                          memory_summary: str = "") -> None:
        """
        チャット履歴を表示する（マスク機能付き、最適化版）
        
        Args:
            messages: チャットメッセージのリスト
            memory_summary: メモリサマリー（重要単語から生成）
        """
        logger.info(f"🎯 render_chat_history 開始: {len(messages) if messages else 0}件のメッセージ")
        try:
            # 初期メッセージの存在確認と復元
            if messages:
                initial_messages = [msg for msg in messages if msg.get('is_initial', False)]
                if not initial_messages:
                    logger.warning("初期メッセージが見つかりません - 復元を試行")
                    # 初期メッセージが存在しない場合は先頭に追加
                    initial_message = {"role": "assistant", "content": "何の用？遊びに来たの？", "is_initial": True}
                    messages.insert(0, initial_message)
                    logger.info("初期メッセージを復元しました")
            
            # 履歴表示の重複実行を防ぐ（改良版）
            # セッション固有のレンダリング状態を管理
            if 'chat_render_state' not in st.session_state:
                st.session_state.chat_render_state = {
                    'last_messages_hash': None,
                    'render_count': 0,
                    'last_render_time': 0
                }
            
            # メッセージ内容とポチモード状態を含むハッシュを生成
            show_all_hidden = st.session_state.get('show_all_hidden', False)
            messages_with_state = {
                'messages': [{'role': msg.get('role'), 'content': msg.get('content', '')[:100]} for msg in messages],  # 内容を短縮してハッシュ計算を軽量化
                'show_all_hidden': show_all_hidden,
                'message_count': len(messages)
            }
            
            import time
            current_time = time.time()
            messages_hash = hash(str(messages_with_state))
            last_render_hash = st.session_state.chat_render_state['last_messages_hash']
            
            # 時間制限を一時的に無効化（デバッグ用）
            time_since_last_render = current_time - st.session_state.chat_render_state['last_render_time']
            # if time_since_last_render < 0.5 and last_render_hash == messages_hash:
            #     logger.debug(f"短時間での重複レンダリングをスキップ（{time_since_last_render:.2f}秒前）")
            #     return
            logger.debug(f"時間制限無効化 - 前回から{time_since_last_render:.2f}秒経過")
            
            # 強制表示条件をより厳密に制御
            force_render_conditions = [
                st.session_state.get('tutorial_start_requested', False),
                st.session_state.get('tutorial_skip_requested', False), 
                st.session_state.get('show_all_hidden_changed', False),
                st.session_state.get('force_chat_rerender', False)  # 明示的な強制レンダリングフラグ
            ]
            
            should_force_render = any(force_render_conditions)
            
            # キャッシュロジックを一時的に無効化（デバッグ用）
            # if last_render_hash == messages_hash and not should_force_render:
            #     logger.debug("チャット履歴表示をスキップ（変更なし）")
            #     return
            logger.debug("キャッシュスキップを無効化 - 常に表示を実行")
            
            # レンダリング実行（ログ削除で軽量化）
            
            # レンダリング状態を更新
            st.session_state.chat_render_state['last_messages_hash'] = messages_hash
            st.session_state.chat_render_state['render_count'] += 1
            st.session_state.chat_render_state['last_render_time'] = current_time
            
            # メモリサマリーがある場合は表示
            if memory_summary:
                with st.expander("💭 過去の会話の記憶", expanded=False):
                    st.info(memory_summary)
            
            # 独自のチャット表示（st.chat_messageを使わない安定版）
            if not messages:
                st.info("まだメッセージがありません。下のチャット欄で麻理に話しかけてみてください。")
                return
            
            for i, message in enumerate(messages):
                role = message.get("role", "user")
                content = message.get("content", "")
                timestamp = message.get("timestamp")
                is_initial = message.get("is_initial", False)
                message_id = message.get("message_id", f"msg_{i}")
                
                # 独自のチャットバブル表示
                self._render_custom_chat_bubble(role, content, is_initial, message_id, timestamp)
            
            # 履歴表示完了をマーク
            # 既にレンダリング状態は上で更新済み
            
            logger.debug(f"チャット履歴表示完了（{len(messages)}件）")
            
            # 強制表示フラグをクリア（表示完了後に実行）
            if st.session_state.get('show_all_hidden_changed', False):
                st.session_state.show_all_hidden_changed = False
                logger.info("犬のボタン状態変更フラグをクリアしました")
            
            if st.session_state.get('force_chat_rerender', False):
                st.session_state.force_chat_rerender = False
                logger.info("強制チャット再レンダリングフラグをクリアしました")
                        
        except Exception as e:
            logger.error(f"チャット履歴表示エラー: {e}")
            st.error("チャット履歴の表示中にエラーが発生しました。")
    
    def _render_custom_chat_bubble(self, role: str, content: str, is_initial: bool, message_id: str, timestamp: str = None):
        """独自のチャットバブル表示（st.chat_messageを使わない安定版）"""
        logger.info(f"🎨 カスタムチャットバブル開始: {role} - '{content[:30]}...' - 初期:{is_initial}")
        try:
            # チャットバブルのCSS
            bubble_css = """
            <style>
            .custom-chat-container {
                margin: 10px 0;
                display: flex;
                flex-direction: column;
            }
            
            .custom-chat-bubble {
                max-width: 80%;
                padding: 12px 16px;
                border-radius: 18px;
                margin: 4px 0;
                word-wrap: break-word;
                line-height: 1.5;
                font-size: 18px;
            }
            
            .user-bubble {
                background: #007bff;
                color: white;
                align-self: flex-end;
                margin-left: auto;
            }
            
            .assistant-bubble {
                background: #f1f3f4;
                color: #333;
                align-self: flex-start;
                margin-right: auto;
                border: 1px solid #e0e0e0;
            }
            
            .initial-message-bubble {
                background: #e8f5e8 !important;
                color: #2d5a2d !important;
                font-weight: 500 !important;
                border: 2px solid #4caf50 !important;
            }
            
            .chat-avatar {
                width: 32px;
                height: 32px;
                border-radius: 50%;
                margin: 0 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 16px;
                flex-shrink: 0;
            }
            
            .user-avatar {
                background: #007bff;
                color: white;
            }
            
            .assistant-avatar {
                background: #ff69b4;
                color: white;
            }
            
            .chat-row {
                display: flex;
                align-items: flex-start;
                margin: 8px 0;
            }
            
            .user-row {
                flex-direction: row-reverse;
            }
            
            .assistant-row {
                flex-direction: row;
            }
            
            .timestamp {
                font-size: 0.8em;
                color: #666;
                margin-top: 4px;
                text-align: center;
            }
            </style>
            """
            
            st.markdown(bubble_css, unsafe_allow_html=True)
            
            # アバターとバブルのスタイル決定
            if role == "user":
                avatar_class = "user-avatar"
                bubble_class = "user-bubble"
                row_class = "user-row"
                avatar_icon = "👤"
            else:
                avatar_class = "assistant-avatar"
                bubble_class = "assistant-bubble"
                row_class = "assistant-row"
                avatar_icon = "🤖"
                
                # 初期メッセージの場合は特別なスタイル
                if is_initial:
                    bubble_class += " initial-message-bubble"
                    avatar_icon = "💬"
            
            # コンテンツのHTMLエスケープ処理（HTMLタグとStreamlitクラス名を完全に除去）
            import html
            import re
            
            # HTMLタグとStreamlitの内部クラス名を完全に除去
            # 1. HTMLタグを除去（開始・終了タグ両方）
            clean_content = re.sub(r'<[^>]*>', '', content)
            
            # 2. Streamlitの内部クラス名を除去（より包括的）
            clean_content = re.sub(r'st-emotion-cache-[a-zA-Z0-9]+', '', clean_content)
            clean_content = re.sub(r'class="[^"]*st-emotion-cache[^"]*"', '', clean_content)
            clean_content = re.sub(r'class="[^"]*st-[^"]*"', '', clean_content)
            clean_content = re.sub(r"class='[^']*st-emotion-cache[^']*'", '', clean_content)
            clean_content = re.sub(r"class='[^']*st-[^']*'", '', clean_content)
            
            # 3. HTML属性を除去
            clean_content = re.sub(r'data-testid="[^"]*"', '', clean_content)
            clean_content = re.sub(r'data-[^=]*="[^"]*"', '', clean_content)
            clean_content = re.sub(r'class="[^"]*"', '', clean_content)
            clean_content = re.sub(r"class='[^']*'", '', clean_content)
            clean_content = re.sub(r'id="[^"]*"', '', clean_content)
            clean_content = re.sub(r"id='[^']*'", '', clean_content)
            
            # 4. その他のHTML関連文字列を除去
            clean_content = re.sub(r'&[a-zA-Z0-9#]+;', '', clean_content)  # HTMLエンティティ
            clean_content = re.sub(r'[<>]', '', clean_content)  # 残った角括弧
            
            # 5. 余分な空白を除去
            clean_content = re.sub(r'\s+', ' ', clean_content).strip()
            
            # 6. HTMLエスケープ
            escaped_content = html.escape(clean_content)
            
            if content != clean_content:
                logger.error(f"🚨 HTMLタグ混入検出! 元の内容: '{content}'")
                logger.error(f"🚨 クリーン後: '{clean_content}'")
                # スタックトレースを出力して呼び出し元を特定
                import traceback
                logger.error(f"🚨 呼び出しスタック: {traceback.format_stack()}")
            else:
                logger.debug(f"通常テキストをHTMLエスケープ: '{content[:30]}...'")
            
            # チャットバブルのHTML生成
            chat_html = f"""
            <div class="custom-chat-container">
                <div class="chat-row {row_class}">
                    <div class="chat-avatar {avatar_class}">
                        {avatar_icon}
                    </div>
                    <div class="custom-chat-bubble {bubble_class}">
                        {escaped_content}
                        {f'<div class="timestamp">{html.escape(timestamp)}</div>' if timestamp and st.session_state.get("debug_mode", False) else ''}
                    </div>
                </div>
            </div>
            """
            
            st.markdown(chat_html, unsafe_allow_html=True)
            
            # 麻理のメッセージで隠された真実がある場合の処理
            if role == "assistant" and not is_initial:
                has_hidden_content, visible_content, hidden_content = self._detect_hidden_content(content)
                if has_hidden_content:
                    # 犬のボタンの状態に応じて表示を切り替え
                    show_all_hidden = st.session_state.get('show_all_hidden', False)
                    logger.debug(f"隠された真実の表示判定: show_all_hidden={show_all_hidden}, has_hidden={has_hidden_content}")
                    
                    if show_all_hidden:
                        # 本音表示モードの場合は隠された内容を表示
                        # HTMLタグとStreamlitクラス名を除去してからエスケープ
                        clean_hidden_content = re.sub(r'<[^>]*>', '', hidden_content)
                        clean_hidden_content = re.sub(r'st-emotion-cache-[a-zA-Z0-9]+', '', clean_hidden_content)
                        clean_hidden_content = re.sub(r'class="[^"]*"', '', clean_hidden_content)
                        clean_hidden_content = re.sub(r"class='[^']*'", '', clean_hidden_content)
                        clean_hidden_content = re.sub(r'data-[^=]*="[^"]*"', '', clean_hidden_content)
                        clean_hidden_content = re.sub(r'&[a-zA-Z0-9#]+;', '', clean_hidden_content)
                        clean_hidden_content = re.sub(r'[<>]', '', clean_hidden_content)
                        clean_hidden_content = re.sub(r'\s+', ' ', clean_hidden_content).strip()
                        escaped_hidden_content = html.escape(clean_hidden_content)
                        
                        hidden_html = f"""
                        <div class="custom-chat-container">
                            <div class="chat-row assistant-row">
                                <div class="chat-avatar assistant-avatar">
                                    🐕
                                </div>
                                <div class="custom-chat-bubble assistant-bubble" style="background: #fff8e1 !important; border: 2px solid #ffc107 !important;">
                                    <strong>🐕 ポチの本音翻訳:</strong><br>
                                    {escaped_hidden_content}
                                </div>
                            </div>
                        </div>
                        """
                        st.markdown(hidden_html, unsafe_allow_html=True)
                        logger.debug(f"隠された真実を表示: '{clean_hidden_content[:30]}...'")
                    else:
                        logger.debug("通常モードのため隠された真実は非表示")
            
            logger.debug(f"カスタムチャットバブル表示完了: {role} - {message_id}")
            
        except Exception as e:
            logger.error(f"カスタムチャットバブル表示エラー: {e}")
            logger.error(f"エラー詳細: role={role}, content_len={len(content)}, is_initial={is_initial}, message_id={message_id}")
            import traceback
            logger.error(f"スタックトレース: {traceback.format_exc()}")
            # フォールバック: シンプルなテキスト表示
            st.markdown(f"**{role}**: {content}")
            logger.info("フォールバック表示を実行しました")
    
    def _render_mari_message_with_mask(self, message_id: str, content: str, is_initial: bool = False) -> None:
        """
        麻理のメッセージをマスク機能付きで表示する（廃止予定）
        
        Args:
            message_id: メッセージの一意ID
            content: メッセージ内容
            is_initial: 初期メッセージかどうか
        """
        logger.warning("⚠️ 廃止予定のメソッドが呼ばれました: _render_mari_message_with_mask")
        # カスタムチャットバブルに移行
        self._render_custom_chat_bubble("assistant", content, is_initial, message_id)
        return
        
        try:
            # メッセージ処理キャッシュをチェック（重複処理防止）
            cache_key = f"processed_{message_id}_{hash(content)}"
            if cache_key in st.session_state:
                # キャッシュから結果を取得
                cached_result = st.session_state[cache_key]
                has_hidden_content = cached_result['has_hidden']
                visible_content = cached_result['visible_content']
                hidden_content = cached_result['hidden_content']
                logger.debug(f"キャッシュからメッセージ処理結果を取得: {message_id}")
            else:
                # 隠された真実を検出
                has_hidden_content, visible_content, hidden_content = self._detect_hidden_content(content)
                
                # 結果をキャッシュに保存
                st.session_state[cache_key] = {
                    'has_hidden': has_hidden_content,
                    'visible_content': visible_content,
                    'hidden_content': hidden_content
                }
                logger.debug(f"メッセージ処理結果をキャッシュに保存: {message_id}")
            
            # 隠された真実が検出されない場合のフォールバック処理
            if not has_hidden_content:
                logger.warning(f"隠された真実が検出されませんでした: '{content[:50]}...'")
                # AIが[HIDDEN:...]形式で応答していない場合は通常表示
            
            # セッション状態でフリップ状態を管理
            if 'message_flip_states' not in st.session_state:
                st.session_state.message_flip_states = {}
            
            is_flipped = st.session_state.message_flip_states.get(message_id, False)
            
            if has_hidden_content:
                # マスクアイコン付きメッセージを表示
                self._render_message_with_flip_animation(
                    message_id, visible_content, hidden_content, is_flipped, is_initial
                )
            else:
                # 通常のメッセージ表示
                if is_initial:
                    # 初期メッセージは確実に黒文字で表示（強制スタイル適用）
                    initial_message_html = f'''
                    <div class="mari-initial-message" style="
                        color: #333333 !important; 
                        font-weight: 500 !important;
                        background: #f5f5f5 !important;
                        padding: 15px !important;
                        border-radius: 12px !important;
                        border: 1px solid rgba(0,0,0,0.1) !important;
                        margin: 8px 0 !important;
                        font-family: var(--mari-font) !important;
                        line-height: 1.7 !important;
                    ">
                        {content}
                    </div>
                    '''
                    st.markdown(initial_message_html, unsafe_allow_html=True)
                    logger.debug(f"初期メッセージを表示: '{content}'")
                else:
                    st.markdown(content)
                    
        except Exception as e:
            logger.error(f"マスク付きメッセージ表示エラー: {e}")
            # フォールバック: 通常のメッセージ表示
            st.markdown(content)
    
    def _detect_hidden_content(self, content: str) -> Tuple[bool, str, str]:
        """
        メッセージから隠された真実を検出する
        
        Args:
            content: メッセージ内容
            
        Returns:
            (隠された内容があるか, 表示用内容, 隠された内容)
        """
        try:
            # デバッグ用ログ（重複実行防止）
            logger.debug(f"🔍 隠された内容検出中: '{content[:50]}...'")
            
            # 隠された真実のマーカーを検索
            # 形式: [HIDDEN:隠された内容]表示される内容
            hidden_pattern = r'\[HIDDEN:(.*?)\](.*)'
            match = re.search(hidden_pattern, content)
            
            if match:
                hidden_content = match.group(1).strip()
                visible_content = match.group(2).strip()
                
                # 複数HIDDENをチェック
                additional_hidden = re.findall(r'\[HIDDEN:(.*?)\]', visible_content)
                if additional_hidden:
                    logger.warning(f"⚠️ 複数HIDDEN検出: {len(additional_hidden) + 1}個のHIDDENが見つかりました")
                    # 2番目以降のHIDDENを表示内容から除去
                    visible_content = re.sub(r'\[HIDDEN:.*?\]', '', visible_content).strip()
                    logger.info(f"🔧 複数HIDDEN除去後: 表示='{visible_content}'")
                
                logger.info(f"🐕 隠された真実を検出: 表示='{visible_content}', 隠し='{hidden_content}'")
                return True, visible_content, hidden_content
            
            # マーカーがない場合は通常のメッセージ
            logger.debug(f"📝 通常メッセージ: '{content[:30]}...'")
            return False, content, ""
            
        except Exception as e:
            logger.error(f"隠された内容検出エラー: {e}")
            return False, content, ""
    
    def _render_message_with_flip_animation(self, message_id: str, visible_content: str, 
                                          hidden_content: str, is_flipped: bool, is_initial: bool = False) -> None:
        """
        フリップアニメーション付きメッセージを表示する
        
        Args:
            message_id: メッセージID
            visible_content: 表示用内容
            hidden_content: 隠された内容
            is_flipped: 現在フリップされているか
            is_initial: 初期メッセージかどうか
        """
        try:
            logger.info(f"🐕 ポチモード付きメッセージを表示: ID={message_id}, フリップ={is_flipped}")
            # フリップアニメーション用CSS
            flip_css = f"""
            <style>
            .message-container-{message_id} {{
                position: relative;
                perspective: 1000px;
                margin: 10px 0;
            }}
            
            .message-flip-{message_id} {{
                position: relative;
                width: 100%;
                height: auto;
                min-height: 60px;
                transform-style: preserve-3d;
                transition: transform 0.4s ease-in-out;
                transform: {'rotateY(180deg)' if is_flipped else 'rotateY(0deg)'};
            }}
            
            .message-side-{message_id} {{
                position: absolute;
                width: 100%;
                backface-visibility: hidden;
                padding: 15px 45px 15px 15px;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                font-family: var(--mari-font);
                line-height: 1.7;
                min-height: 50px;
            }}
            
            .message-front-{message_id} {{
                background: var(--mari-bubble-bg);
                border: 1px solid rgba(0, 0, 0, 0.1);
                color: var(--text-color);
                transform: rotateY(0deg);
            }}
            
            .message-back-{message_id} {{
                background: var(--hidden-bubble-bg);
                border: 1px solid rgba(255, 248, 225, 0.7);
                color: var(--text-color);
                transform: rotateY(180deg);
                box-shadow: 0 2px 8px rgba(255, 248, 225, 0.3);
            }}
            
            .mask-icon-{message_id} {{
                position: absolute;
                bottom: 12px;
                right: 12px;
                font-size: 20px;
                cursor: pointer;
                padding: 6px;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.9);
                transition: all 0.3s ease;
                z-index: 10;
                user-select: none;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
            }}
            
            .mask-icon-{message_id}:hover {{
                background: rgba(255, 255, 255, 1);
                transform: scale(1.1);
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            }}
            
            .mask-icon-{message_id}:active {{
                transform: scale(0.95);
            }}
            
            .tutorial-pulse-{message_id} {{
                animation: tutorialPulse 2s ease-in-out infinite;
            }}
            
            @keyframes tutorialPulse {{
                0%, 100% {{ 
                    transform: scale(1);
                    box-shadow: 0 0 0 0 rgba(255, 105, 180, 0.7);
                }}
                50% {{ 
                    transform: scale(1.1);
                    box-shadow: 0 0 0 10px rgba(255, 105, 180, 0);
                }}
            }}
            </style>
            """
            
            # 犬のボタンの状態を事前にチェックして即座に反映（無限ループ防止）
            show_all_hidden = st.session_state.get('show_all_hidden', False)
            
            # 犬のボタンの状態に従って表示を切り替え（状態変更時のみ）
            current_flip_state = st.session_state.message_flip_states.get(message_id, False)
            if show_all_hidden != current_flip_state:
                st.session_state.message_flip_states[message_id] = show_all_hidden
                is_flipped = show_all_hidden
                logger.debug(f"メッセージ {message_id} のフリップ状態を更新: {is_flipped}")
            else:
                is_flipped = current_flip_state
            
            # 現在表示するコンテンツを決定
            current_content = hidden_content if is_flipped else visible_content
            
            # 初期メッセージの場合は確実に黒文字で表示
            if is_initial:
                initial_style = "color: #333333 !important; font-weight: 500 !important;"
                initial_class = "mari-initial-message"
                # 初期メッセージは背景色を固定
                bg_color = "#F5F5F5"
                logger.debug(f"初期メッセージをフリップ表示: '{current_content}'")
            else:
                initial_style = ""
                initial_class = ""
                # 通常メッセージは背景色を動的に設定
                bg_color = "#FFF8E1" if is_flipped else "#F5F5F5"
            
            # メッセージを全幅で表示（ボタンは削除）
            message_style = f"""
            <div style="
                padding: 15px; 
                background: {bg_color}; 
                border-radius: 12px; 
                border: 1px solid rgba(0,0,0,0.1); 
                min-height: 50px;
                font-family: var(--mari-font);
                line-height: 1.7;
                margin: 8px 0;
            ">
                <div class="{initial_class}" style="{initial_style}">{current_content}</div>
            </div>
            """
            st.markdown(message_style, unsafe_allow_html=True)
            
            # 本音表示機能の状態表示（開発用）
            if st.session_state.get("debug_mode", False):
                st.caption(f"🐕 Dog Mode: ID={message_id}, Hidden={len(hidden_content)>0}, Showing={is_flipped}")
                
        except Exception as e:
            logger.error(f"フリップアニメーション表示エラー: {e}")
            # フォールバック: 通常のメッセージ表示
            st.markdown(visible_content)
    
    def _is_tutorial_message(self, message_id: str) -> bool:
        """
        チュートリアル用のメッセージかどうかを判定する
        
        Args:
            message_id: メッセージID
            
        Returns:
            チュートリアルメッセージかどうか
        """
        # 初回のマスク付きメッセージの場合はチュートリアル扱い
        tutorial_completed = st.session_state.get('mask_tutorial_completed', False)
        return not tutorial_completed and message_id == "msg_0"
    
    def validate_input(self, message: str) -> Tuple[bool, str]:
        """
        入力メッセージの検証
        
        Args:
            message: 入力メッセージ
            
        Returns:
            (検証結果, エラーメッセージ)
        """
        if not message or not message.strip():
            return False, "メッセージが空です。"
        
        if len(message) > self.max_input_length:
            return False, f"メッセージが長すぎます。{self.max_input_length}文字以内で入力してください。"
        
        # 不正な文字のチェック
        if any(ord(char) < 32 and char not in ['\n', '\r', '\t'] for char in message):
            return False, "不正な文字が含まれています。"
        
        return True, ""
    
    def sanitize_message(self, message: str) -> str:
        """
        メッセージをサニタイズする
        
        Args:
            message: 入力メッセージ
            
        Returns:
            サニタイズされたメッセージ
        """
        try:
            # 基本的なサニタイズ
            sanitized = message.strip()
            
            # HTMLエスケープ（Streamlitが自動で行うが念のため）
            sanitized = sanitized.replace("<", "&lt;").replace(">", "&gt;")
            
            # 連続する空白を単一の空白に変換
            import re
            sanitized = re.sub(r'\s+', ' ', sanitized)
            
            return sanitized
            
        except Exception as e:
            logger.error(f"メッセージサニタイズエラー: {e}")
            return message
    
    def add_message(self, role: str, content: str, 
                   messages: Optional[List[Dict[str, str]]] = None, 
                   message_id: Optional[str] = None) -> List[Dict[str, str]]:
        """
        メッセージをリストに追加する（マスク機能対応）
        
        Args:
            role: メッセージの役割 ('user' or 'assistant')
            content: メッセージ内容
            messages: メッセージリスト（Noneの場合はsession_stateから取得）
            message_id: メッセージの一意ID（Noneの場合は自動生成）
            
        Returns:
            更新されたメッセージリスト
        """
        try:
            if messages is None:
                messages = st.session_state.get('messages', [])
            
            # メッセージIDを生成または使用
            if message_id is None:
                message_id = f"msg_{len(messages)}_{uuid.uuid4().hex[:8]}"
            
            # メッセージオブジェクトを作成
            message = {
                "role": role,
                "content": self.sanitize_message(content),
                "timestamp": datetime.now().isoformat(),
                "message_id": message_id
            }
            
            messages.append(message)
            
            # セッション状態を更新
            st.session_state.messages = messages
            
            logger.info(f"メッセージを追加: {role} - {len(content)}文字 (ID: {message_id})")
            return messages
            
        except Exception as e:
            logger.error(f"メッセージ追加エラー: {e}")
            return messages or []
    
    def create_hidden_content_message(self, visible_content: str, hidden_content: str) -> str:
        """
        隠された真実を含むメッセージを作成する
        
        Args:
            visible_content: 表示される内容
            hidden_content: 隠された内容
            
        Returns:
            マーカー付きメッセージ
        """
        return f"[HIDDEN:{hidden_content}]{visible_content}"
    
    def generate_mock_hidden_content(self, visible_content: str) -> str:
        """
        テスト用のモック隠された内容を生成する
        
        Args:
            visible_content: 表示される内容
            
        Returns:
            隠された内容
        """
        # 簡単なモック生成ロジック
        mock_patterns = {
            "何の用？": "（本当は嬉しいけど...素直になれない）",
            "別に": "（実はすごく気になってる）",
            "そうね": "（もっと話していたい）",
            "まあまあ": "（とても楽しい！）",
            "普通": "（特別な時間だと思ってる）",
            "いいんじゃない": "（すごく良いと思う！）",
            "そんなことない": "（本当はそう思ってる）"
        }
        
        for pattern, hidden in mock_patterns.items():
            if pattern in visible_content:
                return hidden
        
        # デフォルトの隠された内容
        return "（本当の気持ちは...）"
    
    def render_input_area(self, placeholder: str = "メッセージを入力してください...") -> Optional[str]:
        """
        入力エリアをレンダリングし、入力を取得する
        
        Args:
            placeholder: 入力フィールドのプレースホルダー
            
        Returns:
            入力されたメッセージ（入力がない場合はNone）
        """
        try:
            # レート制限チェック
            if st.session_state.get('limiter_state', {}).get('is_blocked', False):
                st.warning("⏰ レート制限中です。しばらくお待ちください。")
                st.chat_input(placeholder, disabled=True)
                return None
            
            # 通常の入力フィールド（autocomplete属性付き）
            # Streamlitのchat_inputにautocomplete属性を追加するためのCSS
            chat_input_css = """
            <style>
            /* チャット入力フィールドのautocomplete属性を有効化 */
            .stChatInput input[type="text"] {
        
            }
            .stChatInput textarea {
            }
            </style>
            """
            st.markdown(chat_input_css, unsafe_allow_html=True)
            
            user_input = st.chat_input(placeholder)
            
            if user_input:
                # 入力検証
                is_valid, error_msg = self.validate_input(user_input)
                if not is_valid:
                    st.error(error_msg)
                    return None
                
                return user_input
            
            return None
            
        except Exception as e:
            logger.error(f"入力エリア表示エラー: {e}")
            st.error("入力エリアの表示中にエラーが発生しました。")
            return None
    
    def show_typing_indicator(self, message: str = "考え中...") -> None:
        """
        タイピングインジケーターを表示する
        
        Args:
            message: 表示するメッセージ
        """
        return st.spinner(message)
    
    def clear_chat_history(self) -> None:
        """チャット履歴をクリアする"""
        try:
            st.session_state.messages = []
            logger.info("チャット履歴をクリアしました")
            
        except Exception as e:
            logger.error(f"チャット履歴クリアエラー: {e}")
    
    def get_chat_stats(self) -> Dict[str, int]:
        """
        チャットの統計情報を取得する
        
        Returns:
            統計情報の辞書
        """
        try:
            messages = st.session_state.get('messages', [])
            
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
            
            total_chars = sum(len(msg.get("content", "")) for msg in messages)
            
            return {
                "total_messages": len(messages),
                "user_messages": len(user_messages),
                "assistant_messages": len(assistant_messages),
                "total_characters": total_chars,
                "average_message_length": total_chars // len(messages) if messages else 0
            }
            
        except Exception as e:
            logger.error(f"統計情報取得エラー: {e}")
            return {
                "total_messages": 0,
                "user_messages": 0,
                "assistant_messages": 0,
                "total_characters": 0,
                "average_message_length": 0
            }
    
    def export_chat_history(self) -> str:
        """
        チャット履歴をエクスポート用の文字列として取得する
        
        Returns:
            エクスポート用の文字列
        """
        try:
            messages = st.session_state.get('messages', [])
            
            if not messages:
                return "チャット履歴がありません。"
            
            export_lines = []
            export_lines.append("=== 麻理チャット履歴 ===")
            export_lines.append(f"エクスポート日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            export_lines.append("")
            
            for i, message in enumerate(messages, 1):
                role = "ユーザー" if message.get("role") == "user" else "麻理"
                content = message.get("content", "")
                timestamp = message.get("timestamp", "")
                
                export_lines.append(f"[{i}] {role}: {content}")
                if timestamp:
                    export_lines.append(f"    時刻: {timestamp}")
                export_lines.append("")
            
            return "\n".join(export_lines)
            
        except Exception as e:
            logger.error(f"履歴エクスポートエラー: {e}")
            return "エクスポート中にエラーが発生しました。"