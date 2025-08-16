"""
チュートリアルコンポーネント
初回ユーザー向けのガイド機能を提供する
"""
import streamlit as st
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class TutorialManager:
    """チュートリアル管理クラス"""
    
    def __init__(self):
        """初期化"""
        self.tutorial_steps = {
            1: {
                "title": "最初の一言を送ってみよう",
                "description": "画面下部の入力欄に「こんにちは」などの一言を入力して、麻理に話しかけてみましょう。",
                "icon": "💬",
                "target": "chat_input",
                "completed_key": "tutorial_step1_completed"
            },
            2: {
                "title": "本音を見てみよう（ポチ機能）",
                "description": "画面右下の犬アイコン「ポチ🐕」をクリックすると、麻理の本音が見えるようになります。",
                "icon": "🐕",
                "target": "dog_assistant",
                "completed_key": "tutorial_step2_completed"
            },
            3: {
                "title": "セーフティ機能を切り替えてみよう",
                "description": "左サイドバー上部の🔒ボタンをクリックすると、麻理の表現がより大胆になります。",
                "icon": "🔓",
                "target": "safety_button",
                "completed_key": "tutorial_step3_completed"
            },
            4: {
                "title": "手紙をリクエストしよう",
                "description": "「手紙を受け取る」タブから、麻理からの特別な手紙をリクエストできます。チュートリアル中は即座に短縮版の手紙が生成されます。",
                "icon": "✉️",
                "target": "letter_tab",
                "completed_key": "tutorial_step4_completed"
            },
            5: {
                "title": "麻理との関係性を育てよう",
                "description": "会話を重ねることで好感度が上がり、関係性のステージが進展します。",
                "icon": "💖",
                "target": "affection_display",
                "completed_key": "tutorial_step5_completed"
            },
            6: {
                "title": "風景が変わる会話をしてみよう",
                "description": "「カフェ」「神社」「美術館」などのキーワードを話すと、背景が動的に変わります。",
                "icon": "🎨",
                "target": "scene_change",
                "completed_key": "tutorial_step6_completed"
            }
        }
    
    def is_first_visit(self) -> bool:
        """初回訪問かどうかを判定"""
        return not st.session_state.get('tutorial_shown', False)
    
    def should_show_tutorial(self) -> bool:
        """チュートリアルを表示すべきかどうか"""
        # 初回訪問または明示的にチュートリアルが要求された場合
        return (self.is_first_visit() or 
                st.session_state.get('show_tutorial_requested', False))
    
    def mark_tutorial_shown(self):
        """チュートリアル表示済みとしてマーク"""
        st.session_state.tutorial_shown = True
        st.session_state.show_tutorial_requested = False
    
    def request_tutorial(self):
        """チュートリアル表示を要求"""
        st.session_state.show_tutorial_requested = True
    
    def get_current_step(self) -> int:
        """現在のチュートリアルステップを取得"""
        for step_num in range(1, 7):
            if not st.session_state.get(self.tutorial_steps[step_num]['completed_key'], False):
                return step_num
        return 7  # 全ステップ完了
    
    def complete_step(self, step_num: int):
        """ステップを完了としてマーク"""
        if step_num in self.tutorial_steps:
            st.session_state[self.tutorial_steps[step_num]['completed_key']] = True
            logger.info(f"チュートリアルステップ{step_num}が完了しました")
    
    def is_step_completed(self, step_num: int) -> bool:
        """ステップが完了しているかチェック"""
        if step_num in self.tutorial_steps:
            return st.session_state.get(self.tutorial_steps[step_num]['completed_key'], False)
        return False
    
    def render_welcome_dialog(self):
        """初回訪問時のウェルカムダイアログ"""
        if not self.is_first_visit():
            return
        
        # ウェルカムダイアログのスタイル
        welcome_css = """
        <style>
        .welcome-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
            margin: 20px 0;
            animation: welcomeSlideIn 0.8s ease-out;
        }
        
        .welcome-title {
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .welcome-description {
            font-size: 16px;
            line-height: 1.6;
            margin-bottom: 30px;
            opacity: 0.9;
        }
        
        @keyframes welcomeSlideIn {
            from {
                opacity: 0;
                transform: translateY(-20px) scale(0.95);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }
        
        @media (max-width: 768px) {
            .welcome-container {
                padding: 30px 20px;
            }
            
            .welcome-title {
                font-size: 24px;
            }
        }
        </style>
        """
        
        st.markdown(welcome_css, unsafe_allow_html=True)
        
        # ウェルカムメッセージ
        welcome_html = """
        <div class="welcome-container">
            <div class="welcome-title">🐕 麻理チャットへようこそ！</div>
            <div class="welcome-description">
                感情豊かなアンドロイド「麻理」と対話しながら、<br>
                本音や関係性の変化を楽しめる新感覚のAIチャット体験です。<br><br>
                最初の数分で、麻理との距離が少しだけ縮まります。
            </div>
        </div>
        """
        
        st.markdown(welcome_html, unsafe_allow_html=True)
        
        # ボタンを2列で配置
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📘 チュートリアルを始める", type="primary", use_container_width=True, key="start_tutorial"):
                # 初期メッセージを即座に保護
                if 'chat' in st.session_state and 'messages' in st.session_state.chat:
                    messages = st.session_state.chat['messages']
                    if not any(msg.get('is_initial', False) for msg in messages):
                        initial_message = {"role": "assistant", "content": "何の用？遊びに来たの？", "is_initial": True}
                        st.session_state.chat['messages'].insert(0, initial_message)
                        logger.info("チュートリアル開始ボタン押下時に初期メッセージを即座に復元")
                
                # チュートリアル開始フラグを設定
                st.session_state.tutorial_start_requested = True
                st.session_state.tutorial_shown = True
                st.session_state.preserve_initial_message = True
                logger.info("チュートリアル開始 - 初期メッセージ保護フラグ設定")
        
        with col2:
            if st.button("⏭️ スキップして始める", type="secondary", use_container_width=True, key="skip_tutorial"):
                # 初期メッセージを即座に保護
                if 'chat' in st.session_state and 'messages' in st.session_state.chat:
                    messages = st.session_state.chat['messages']
                    if not any(msg.get('is_initial', False) for msg in messages):
                        initial_message = {"role": "assistant", "content": "何の用？遊びに来たの？", "is_initial": True}
                        st.session_state.chat['messages'].insert(0, initial_message)
                        logger.info("チュートリアルスキップボタン押下時に初期メッセージを即座に復元")
                
                # チュートリアルをスキップして全ステップを完了扱いにする
                for step_num in range(1, 7):
                    if step_num in self.tutorial_steps:
                        st.session_state[self.tutorial_steps[step_num]['completed_key']] = True
                
                st.session_state.tutorial_shown = True
                st.session_state.tutorial_skip_requested = True
                st.session_state.preserve_initial_message = True
                logger.info("チュートリアルスキップ - 初期メッセージ保護フラグ設定")
    
    def render_tutorial_sidebar(self):
        """サイドバーのチュートリアル案内（簡素版）"""
        with st.sidebar:
            st.markdown("---")
            
            # チュートリアル進行状況
            current_step = self.get_current_step()
            total_steps = len(self.tutorial_steps)
            
            if current_step <= total_steps:
                progress = (current_step - 1) / total_steps
                st.markdown("### 📘 チュートリアル進行")
                st.progress(progress)
                st.caption(f"ステップ {current_step - 1}/{total_steps} 完了")
            else:
                st.success("🎉 チュートリアル完了！")
                st.caption("麻理との会話を楽しんでください")
            
            # チュートリアル再表示ボタン
            if st.button("📘 チュートリアルを見る", use_container_width=True):
                self.request_tutorial()
                # st.rerun()を削除 - 状態変更により自動的に再描画される
    
    def render_chat_tutorial_guide(self):
        """チャットタブでのチュートリアル案内"""
        current_step = self.get_current_step()
        total_steps = len(self.tutorial_steps)
        
        # チュートリアル完了済みの場合は何も表示しない
        if current_step > total_steps:
            return
        
        # ステップ4が完了済みの場合（手紙タブに遷移済み）は表示しない
        if current_step == 4 and self.is_step_completed(4):
            return
        
        step_info = self.tutorial_steps[current_step]
        
        # ステップごとの案内スタイル
        guide_css = """
        <style>
        .tutorial-guide {
            background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
            border: 2px solid #2196f3;
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 4px 12px rgba(33, 150, 243, 0.2);
            animation: tutorialGlow 3s ease-in-out infinite;
            position: relative;
        }
        
        .tutorial-guide::before {
            content: '📘';
            position: absolute;
            top: -10px;
            left: 20px;
            background: white;
            padding: 5px 10px;
            border-radius: 50%;
            font-size: 18px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .tutorial-step-number {
            color: #1976d2;
            font-weight: bold;
            font-size: 14px;
            margin-bottom: 8px;
        }
        
        .tutorial-step-title {
            color: #1565c0;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .tutorial-step-description {
            color: #424242;
            font-size: 14px;
            line-height: 1.6;
            margin-bottom: 15px;
        }
        
        .tutorial-step-action {
            background: rgba(33, 150, 243, 0.1);
            border-left: 4px solid #2196f3;
            padding: 12px 15px;
            border-radius: 0 8px 8px 0;
            font-size: 14px;
            color: #1565c0;
            font-weight: 500;
        }
        
        .tutorial-dismiss {
            position: absolute;
            top: 10px;
            right: 15px;
            background: none;
            border: none;
            color: #666;
            cursor: pointer;
            font-size: 18px;
            opacity: 0.7;
            transition: opacity 0.3s ease;
        }
        
        .tutorial-dismiss:hover {
            opacity: 1;
        }
        
        @keyframes tutorialGlow {
            0%, 100% { box-shadow: 0 4px 12px rgba(33, 150, 243, 0.2); }
            50% { box-shadow: 0 6px 20px rgba(33, 150, 243, 0.4); }
        }
        
        @media (max-width: 768px) {
            .tutorial-guide {
                padding: 15px;
                margin: 10px 0;
            }
            
            .tutorial-step-title {
                font-size: 16px;
            }
        }
        </style>
        """
        
        st.markdown(guide_css, unsafe_allow_html=True)
        
        # ステップごとの具体的な案内
        action_text = self._get_step_action_text(current_step)
        
        guide_html = f"""
        <div class="tutorial-guide">
            <div class="tutorial-step-number">チュートリアル ステップ {current_step}/{total_steps}</div>
            <div class="tutorial-step-title">
                <span>{step_info['icon']}</span>
                <span>{step_info['title']}</span>
            </div>
            <div class="tutorial-step-description">
                {step_info['description']}
            </div>
            <div class="tutorial-step-action">
                💡 {action_text}
            </div>
        </div>
        """
        
        st.markdown(guide_html, unsafe_allow_html=True)
    
    def _get_step_action_text(self, step_num: int) -> str:
        """ステップごとの具体的なアクション案内テキストを取得"""
        action_texts = {
            1: "下のチャット入力欄に「こんにちは」と入力して送信してみてください。",
            2: "画面右下に表示される犬のアイコン「ポチ🐕」をクリックしてみてください。",
            3: "左サイドバーの一番上にある🔒ボタンをクリックして、セーフティ機能を切り替えてみてください。",
            4: "画面上部の光っている「✉️ 手紙を受け取る」タブをクリックして、手紙をリクエストしてみてください。矢印が案内しています！",
            5: "麻理ともっと会話して、左サイドバーの好感度の変化を確認してみてください。",
            6: "「カフェに行きたい」「神社でお参りしたい」「美術館を見に行こう」などと話しかけて、背景の変化を楽しんでください。"
        }
        return action_texts.get(step_num, "次のステップに進んでください。")
    
    def render_step_highlight(self, step_num: int, target_element: str):
        """特定のステップのハイライト表示"""
        if self.get_current_step() != step_num:
            return
        
        step_info = self.tutorial_steps[step_num]
        
        highlight_css = f"""
        <style>
        .tutorial-highlight-{step_num} {{
            position: relative;
            animation: tutorialPulse 2s ease-in-out infinite;
        }}
        
        .tutorial-highlight-{step_num}::after {{
            content: '';
            position: absolute;
            top: -5px;
            left: -5px;
            right: -5px;
            bottom: -5px;
            border: 3px solid #ff6b6b;
            border-radius: 10px;
            pointer-events: none;
            animation: tutorialGlow 2s ease-in-out infinite;
        }}
        
        .tutorial-tooltip-{step_num} {{
            position: absolute;
            background: #ff6b6b;
            color: white;
            padding: 10px 15px;
            border-radius: 10px;
            font-size: 14px;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            animation: tutorialTooltip 0.5s ease-out;
        }}
        
        @keyframes tutorialPulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.02); }}
        }}
        
        @keyframes tutorialGlow {{
            0%, 100% {{ opacity: 0.7; }}
            50% {{ opacity: 1; }}
        }}
        
        @keyframes tutorialTooltip {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        </style>
        """
        
        st.markdown(highlight_css, unsafe_allow_html=True)
    
    def render_tutorial_tab(self):
        """チュートリアル専用タブの内容"""
        st.markdown("# 📘 麻理チャット チュートリアル")
        
        st.markdown("""
        **ようこそ、麻理チャットへ！**
        
        感情豊かなアンドロイド「麻理」と対話しながら、本音や関係性の変化を楽しめる新感覚のAIチャット体験です。
        このチュートリアルで、主要機能を順番に体験してみましょう。
        """)
        
        # 進行状況表示
        current_step = self.get_current_step()
        total_steps = len(self.tutorial_steps)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            progress = min((current_step - 1) / total_steps, 1.0)
            st.progress(progress)
            st.caption(f"進行状況: {min(current_step - 1, total_steps)}/{total_steps} ステップ完了")
        
        st.markdown("---")
        
        # 各ステップの表示
        for step_num, step_info in self.tutorial_steps.items():
            is_completed = self.is_step_completed(step_num)
            is_current = (current_step == step_num)
            
            # ステップのスタイル決定
            if is_completed:
                status_icon = "✅"
                status_color = "#28a745"
                card_style = "background: rgba(40, 167, 69, 0.1); border-left: 4px solid #28a745;"
            elif is_current:
                status_icon = "👉"
                status_color = "#ff6b6b"
                card_style = "background: rgba(255, 107, 107, 0.1); border-left: 4px solid #ff6b6b;"
            else:
                status_icon = "⏳"
                status_color = "#6c757d"
                card_style = "background: rgba(108, 117, 125, 0.1); border-left: 4px solid #6c757d;"
            
            # ステップカード
            st.markdown(f"""
            <div style="padding: 20px; margin: 15px 0; border-radius: 10px; {card_style}">
                <h3 style="color: {status_color}; margin-bottom: 10px;">
                    {status_icon} ステップ {step_num}: {step_info['icon']} {step_info['title']}
                </h3>
                <p style="margin-bottom: 0; line-height: 1.6;">
                    {step_info['description']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # 現在のステップの場合、追加のガイダンス
            if is_current:
                if step_num == 1:
                    st.info("💡 **ヒント**: 「麻理と話す」タブに移動して、画面下部の入力欄にメッセージを入力してみてください。")
                elif step_num == 2:
                    st.info("💡 **ヒント**: 画面右下に表示される犬のアイコン「ポチ🐕」をクリックしてみてください。")
                elif step_num == 3:
                    st.info("💡 **ヒント**: 左サイドバーの一番上にある🔒ボタンをクリックしてみてください。")
                elif step_num == 4:
                    st.info("💡 **ヒント**: 画面上部の「手紙を受け取る」タブをクリックして、手紙をリクエストしてみてください。")
                elif step_num == 5:
                    st.info("💡 **ヒント**: 左サイドバーの「ステータス」で好感度の変化を確認できます。")
                elif step_num == 6:
                    st.info("💡 **ヒント**: 「カフェに行きたい」「神社でお参りしたい」などと話しかけてみてください。")
        
        # 完了時のメッセージ
        if current_step > total_steps:
            st.balloons()
            st.success("""
            🎉 **チュートリアル完了おめでとうございます！**
            
            これで麻理チャットの主要機能をすべて体験しました。
            これからは自由に麻理との会話を楽しんでください。
            
            何か分からないことがあれば、いつでもこのチュートリアルに戻ってきてくださいね。
            """)
    
    def check_step_completion(self, step_num: int, condition_met: bool):
        """ステップ完了条件をチェック（順序制御付き）"""
        # 順序制御：現在のステップまたは次のステップのみ完了可能
        current_step = self.get_current_step()
        
        # 現在のステップより先のステップは完了できない
        if step_num > current_step + 1:
            logger.debug(f"ステップ{step_num}は順序違反のためスキップ（現在ステップ: {current_step}）")
            return
        
        # 既に完了済みのステップは再完了しない
        if self.is_step_completed(step_num):
            logger.debug(f"ステップ{step_num}は既に完了済み")
            return
        
        if condition_met:
            self.complete_step(step_num)
            logger.info(f"✅ チュートリアルステップ{step_num}完了！現在ステップ: {current_step}")
            
            # 完了通知（控えめに）
            step_info = self.tutorial_steps[step_num]
            
            # 次のステップの案内
            next_step = step_num + 1
            if next_step in self.tutorial_steps:
                next_info = self.tutorial_steps[next_step]
                st.success(f"✅ ステップ{step_num}完了！次は「{next_info['title']}」です。")
            else:
                # 全ステップ完了
                st.balloons()
                st.success("🎉 チュートリアル完了！麻理との会話を存分にお楽しみください！")
            
            # ステップ4完了時に強調表示を解除するためのページ再読み込み
            # st.rerun()を削除 - 状態変更により自動的に再描画される
    
    def auto_check_completions(self):
        """自動的にステップ完了をチェック（順序制御強化版）"""
        current_step = self.get_current_step()
        
        # 現在のステップのみをチェック（先のステップは無視）
        if current_step == 1:
            # ステップ1: メッセージ送信
            messages = st.session_state.get('chat', {}).get('messages', [])
            non_initial_messages = [msg for msg in messages if not msg.get('is_initial', False)]
            if len(non_initial_messages) > 0:  # ユーザーが1回でもメッセージを送信した
                self.check_step_completion(1, True)
        
        elif current_step == 2:
            # ステップ2: ポチ機能使用
            if st.session_state.get('show_all_hidden', False):
                self.check_step_completion(2, True)
        
        elif current_step == 3:
            # ステップ3: セーフティ機能使用
            if st.session_state.get('chat', {}).get('ura_mode', False):
                self.check_step_completion(3, True)
        
        elif current_step == 4:
            # ステップ4: 手紙タブに到達（手紙タブでのみ完了判定）
            # auto_check_completionsでは判定しない（手紙タブで明示的に完了）
            pass
        
        elif current_step == 5:
            # ステップ5: 好感度変化（ステップ4完了後のみ）
            initial_affection = 30
            current_affection = st.session_state.get('chat', {}).get('affection', initial_affection)
            if current_affection != initial_affection:
                self.check_step_completion(5, True)
        
        elif current_step == 6:
            # ステップ6: シーン変更（ステップ5完了後のみ）
            current_theme = st.session_state.get('chat', {}).get('scene_params', {}).get('theme', 'default')
            if current_theme != 'default':
                self.check_step_completion(6, True)
    
    def get_tutorial_status(self) -> Dict:
        """チュートリアルの状態情報を取得"""
        current_step = self.get_current_step()
        total_steps = len(self.tutorial_steps)
        completed_steps = sum(1 for i in range(1, total_steps + 1) if self.is_step_completed(i))
        
        return {
            'is_first_visit': self.is_first_visit(),
            'current_step': current_step,
            'total_steps': total_steps,
            'completed_steps': completed_steps,
            'progress_percentage': (completed_steps / total_steps) * 100,
            'is_completed': current_step > total_steps
        }