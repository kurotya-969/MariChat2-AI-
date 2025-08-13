"""
ステータス表示コンポーネント
好感度ゲージと関係ステージの表示を担当する
"""
import streamlit as st
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class StatusDisplay:
    """ステータス表示を管理するクラス"""
    
    def __init__(self):
        """ステータス表示の初期化"""
        self.stage_colors = {
            "敵対": {"color": "#ff4757", "emoji": "🔴", "bg_color": "rgba(255, 71, 87, 0.1)"},
            "警戒": {"color": "#ff6348", "emoji": "🟠", "bg_color": "rgba(255, 99, 72, 0.1)"},
            "中立": {"color": "#ffa502", "emoji": "🟡", "bg_color": "rgba(255, 165, 2, 0.1)"},
            "好意": {"color": "#2ed573", "emoji": "🟢", "bg_color": "rgba(46, 213, 115, 0.1)"},
            "親密": {"color": "#a55eea", "emoji": "💜", "bg_color": "rgba(165, 94, 234, 0.1)"}
        }
    
    def get_affection_color(self, affection: int) -> str:
        """
        好感度に基づいて色を取得する
        
        Args:
            affection: 好感度値 (0-100)
            
        Returns:
            色のHEXコード
        """
        if affection < 20:
            return "#ff4757"  # 赤
        elif affection < 40:
            return "#ff6348"  # オレンジ
        elif affection < 60:
            return "#ffa502"  # 黄色
        elif affection < 80:
            return "#2ed573"  # 緑
        else:
            return "#a55eea"  # 紫
    
    def get_relationship_stage_info(self, affection: int) -> Dict[str, str]:
        """
        好感度から関係性ステージの情報を取得する
        
        Args:
            affection: 好感度値 (0-100)
            
        Returns:
            ステージ情報の辞書
        """
        if affection < 20:
            stage = "敵対"
        elif affection < 40:
            stage = "警戒"
        elif affection < 60:
            stage = "中立"
        elif affection < 80:
            stage = "好意"
        else:
            stage = "親密"
        
        # 古いキー形式との互換性を保つため、新しいキーで検索し、見つからない場合は中立を返す
        stage_info = None
        for key, value in self.stage_colors.items():
            if stage in key:
                stage_info = value
                break
        
        return stage_info or self.stage_colors.get("中立", {"color": "#ffa502", "emoji": "🟡", "bg_color": "rgba(255, 165, 2, 0.1)"})
    
    def render_affection_gauge(self, affection: int) -> None:
        """
        好感度ゲージを表示する
        
        Args:
            affection: 好感度値 (0-100)
        """
        try:
            # 好感度の値を0-100の範囲に制限
            affection = max(0, min(100, affection))
            
            # 好感度メトリック表示
            col1, col2 = st.columns([2, 1])
            with col1:
                st.metric("好感度", f"{affection}/100")
            with col2:
                # 好感度の変化を表示（前回の値と比較）
                prev_affection = st.session_state.get('prev_affection', affection)
                delta = affection - prev_affection
                if delta != 0:
                    st.metric("変化", f"{delta:+d}")
                st.session_state.prev_affection = affection
            
            # プログレスバー
            progress_value = affection / 100.0
            affection_color = self.get_affection_color(affection)
            
            # カスタムプログレスバーのCSS
            progress_css = f"""
            <style>
            .affection-progress {{
                width: 100%;
                height: 25px;
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                overflow: hidden;
                margin: 10px 0;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }}
            .affection-fill {{
                height: 100%;
                background: linear-gradient(90deg, 
                    #ff4757 0%, 
                    #ff6348 25%, 
                    #ffa502 50%, 
                    #2ed573 75%, 
                    #a55eea 100%);
                width: {progress_value * 100}%;
                transition: width 0.5s ease-in-out;
                position: relative;
            }}
            .affection-text {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                color: white;
                font-weight: bold;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
                font-size: 12px;
            }}
            </style>
            """
            
            st.markdown(progress_css, unsafe_allow_html=True)
            
            # プログレスバーのHTML
            progress_html = f"""
            <div class="affection-progress">
                <div class="affection-fill">
                    <div class="affection-text">{affection}%</div>
                </div>
            </div>
            """
            
            st.markdown(progress_html, unsafe_allow_html=True)
            
            # Streamlitの標準プログレスバーも表示（フォールバック）
            st.progress(progress_value)
            
        except Exception as e:
            logger.error(f"好感度ゲージ表示エラー: {e}")
            # フォールバック表示
            st.metric("好感度", f"{affection}/100")
            st.progress(affection / 100.0)
    
    def render_relationship_stage(self, affection: int) -> None:
        """
        関係性ステージを表示する
        
        Args:
            affection: 好感度値 (0-100)
        """
        try:
            stage_info = self.get_relationship_stage_info(affection)
            
            # ステージ名を取得
            if affection < 20:
                stage_name = "ステージ1：敵対"
                stage_description = "麻理はあなたを敵視している"
            elif affection < 40:
                stage_name = "ステージ2：警戒"
                stage_description = "麻理はあなたを警戒している"
            elif affection < 60:
                stage_name = "ステージ3：中立"
                stage_description = "麻理はあなたに対して中立的"
            elif affection < 80:
                stage_name = "ステージ4：好意"
                stage_description = "麻理はあなたに好意を持っている"
            else:
                stage_name = "ステージ5：親密"
                stage_description = "麻理はあなたと親密な関係"
            
            # ステージ表示のCSS
            stage_css = f"""
            <style>
            .relationship-stage {{
                background: {stage_info['bg_color']};
                border: 2px solid {stage_info['color']};
                border-radius: 10px;
                padding: 15px;
                margin: 10px 0;
                text-align: center;
            }}
            .stage-emoji {{
                font-size: 24px;
                margin-bottom: 5px;
            }}
            .stage-name {{
                color: {stage_info['color']};
                font-weight: bold;
                font-size: 16px;
                margin-bottom: 5px;
            }}
            .stage-description {{
                color: {stage_info['color']};
                font-size: 12px;
                opacity: 0.8;
            }}
            </style>
            """
            
            st.markdown(stage_css, unsafe_allow_html=True)
            
            # ステージ表示のHTML
            stage_html = f"""
            <div class="relationship-stage">
                <div class="stage-emoji">{stage_info['emoji']}</div>
                <div class="stage-name">{stage_name}</div>
                <div class="stage-description">{stage_description}</div>
            </div>
            """
            
            st.markdown(stage_html, unsafe_allow_html=True)
            
            # フォールバック表示
            st.write(f"{stage_info['emoji']} **関係性**: {stage_name}")
            
        except Exception as e:
            logger.error(f"関係性ステージ表示エラー: {e}")
            # フォールバック表示
            if affection < 20:
                st.write("🔴 **関係性**: ステージ1：敵対")
            elif affection < 40:
                st.write("🟠 **関係性**: ステージ2：中立")
            elif affection < 60:
                st.write("🟡 **関係性**: ステージ3：好意")
            elif affection < 80:
                st.write("🟢 **関係性**: ステージ4：親密")
            else:
                st.write("💜 **関係性**: ステージ5：最接近")
    
    def render_affection_history(self, max_history: int = 10) -> None:
        """
        好感度の履歴を表示する（デバッグモード用）
        
        Args:
            max_history: 表示する履歴の最大数
        """
        try:
            if not st.session_state.get('debug_mode', False):
                return
            
            # 好感度履歴を取得
            affection_history = st.session_state.get('affection_history', [])
            
            if not affection_history:
                st.write("好感度の履歴がありません")
                return
            
            # 最新の履歴を表示
            recent_history = affection_history[-max_history:]
            
            st.subheader("📈 好感度履歴")
            
            for i, entry in enumerate(reversed(recent_history)):
                timestamp = entry.get('timestamp', 'Unknown')
                affection = entry.get('affection', 0)
                change = entry.get('change', 0)
                message = entry.get('message', '')
                
                change_str = f"({change:+d})" if change != 0 else ""
                st.write(f"{i+1}. {affection}/100 {change_str} - {timestamp[:19]}")
                if message:
                    st.caption(f"メッセージ: {message[:50]}...")
            
        except Exception as e:
            logger.error(f"好感度履歴表示エラー: {e}")
    
    def update_affection_history(self, old_affection: int, new_affection: int, 
                               message: str = "") -> None:
        """
        好感度履歴を更新する
        
        Args:
            old_affection: 変更前の好感度
            new_affection: 変更後の好感度
            message: 関連するメッセージ
        """
        try:
            if 'affection_history' not in st.session_state:
                st.session_state.affection_history = []
            
            # 履歴エントリを作成
            history_entry = {
                'timestamp': st.session_state.get('current_timestamp', ''),
                'affection': new_affection,
                'change': new_affection - old_affection,
                'message': message[:100] if message else ''  # メッセージを100文字に制限
            }
            
            st.session_state.affection_history.append(history_entry)
            
            # 履歴の長さを制限（最大50エントリ）
            if len(st.session_state.affection_history) > 50:
                st.session_state.affection_history = st.session_state.affection_history[-50:]
            
        except Exception as e:
            logger.error(f"好感度履歴更新エラー: {e}")
    
    def get_affection_statistics(self) -> Dict[str, float]:
        """
        好感度の統計情報を取得する
        
        Returns:
            統計情報の辞書
        """
        try:
            affection_history = st.session_state.get('affection_history', [])
            
            if not affection_history:
                return {
                    'current': st.session_state.get('affection', 30),
                    'average': 30.0,
                    'max': 30,
                    'min': 30,
                    'total_changes': 0
                }
            
            affections = [entry['affection'] for entry in affection_history]
            changes = [entry['change'] for entry in affection_history if entry['change'] != 0]
            
            return {
                'current': st.session_state.get('affection', 30),
                'average': sum(affections) / len(affections),
                'max': max(affections),
                'min': min(affections),
                'total_changes': len(changes),
                'positive_changes': len([c for c in changes if c > 0]),
                'negative_changes': len([c for c in changes if c < 0])
            }
            
        except Exception as e:
            logger.error(f"好感度統計取得エラー: {e}")
            return {
                'current': st.session_state.get('affection', 30),
                'average': 30.0,
                'max': 30,
                'min': 30,
                'total_changes': 0
            }
    
    def apply_status_styles(self) -> None:
        """
        ステータス表示用のカスタムスタイルを適用する
        """
        try:
            status_css = """
            <style>
            /* ステータス表示全体のスタイル */
            .status-container {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(15px);
                border-radius: 15px;
                padding: 20px;
                margin: 15px 0;
                border: 1px solid rgba(255, 255, 255, 0.2);
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease;
            }
            
            .status-container:hover {
                background: rgba(255, 255, 255, 0.15);
                transform: translateY(-3px);
                box-shadow: 0 12px 35px rgba(0, 0, 0, 0.15);
            }
            
            /* メトリクス表示の改善 */
            .stMetric {
                background: rgba(255, 255, 255, 0.05);
                padding: 10px;
                border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .stMetric > div {
                color: white !important;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.7);
            }
            
            /* 好感度ゲージのアニメーション */
            .affection-progress {
                position: relative;
                overflow: hidden;
            }
            
            .affection-progress::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, 
                    transparent, 
                    rgba(255, 255, 255, 0.3), 
                    transparent);
                animation: shimmer 2s infinite;
            }
            
            @keyframes shimmer {
                0% { left: -100%; }
                100% { left: 100%; }
            }
            
            /* 関係性ステージのアニメーション */
            .relationship-stage {
                animation: stageGlow 3s ease-in-out infinite alternate;
            }
            
            @keyframes stageGlow {
                0% { box-shadow: 0 0 5px rgba(255, 255, 255, 0.2); }
                100% { box-shadow: 0 0 20px rgba(255, 255, 255, 0.4); }
            }
            
            /* ステージ変更時のエフェクト */
            .stage-change-effect {
                animation: stageChange 1s ease-in-out;
            }
            
            @keyframes stageChange {
                0% { transform: scale(1); opacity: 1; }
                50% { transform: scale(1.05); opacity: 0.8; }
                100% { transform: scale(1); opacity: 1; }
            }
            
            /* 好感度変化のエフェクト */
            .affection-change-positive {
                animation: positiveChange 0.8s ease-out;
            }
            
            .affection-change-negative {
                animation: negativeChange 0.8s ease-out;
            }
            
            @keyframes positiveChange {
                0% { color: #2ed573; transform: scale(1); }
                50% { color: #2ed573; transform: scale(1.1); }
                100% { color: inherit; transform: scale(1); }
            }
            
            @keyframes negativeChange {
                0% { color: #ff4757; transform: scale(1); }
                50% { color: #ff4757; transform: scale(1.1); }
                100% { color: inherit; transform: scale(1); }
            }
            
            /* デバッグ情報のスタイル */
            .debug-info {
                background: rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 8px;
                padding: 10px;
                margin: 10px 0;
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }
            
            .debug-info pre {
                color: #00ff00;
                margin: 0;
            }
            
            /* 履歴表示のスタイル */
            .history-item {
                background: rgba(255, 255, 255, 0.05);
                border-left: 3px solid rgba(255, 255, 255, 0.3);
                padding: 8px 12px;
                margin: 5px 0;
                border-radius: 0 8px 8px 0;
                transition: all 0.3s ease;
            }
            
            .history-item:hover {
                background: rgba(255, 255, 255, 0.1);
                border-left-color: rgba(255, 255, 255, 0.5);
                transform: translateX(5px);
            }
            
            .history-positive {
                border-left-color: #2ed573;
            }
            
            .history-negative {
                border-left-color: #ff4757;
            }
            
            .history-neutral {
                border-left-color: #ffa502;
            }
            </style>
            """
            
            st.markdown(status_css, unsafe_allow_html=True)
            logger.debug("ステータス表示用スタイルを適用しました")
            
        except Exception as e:
            logger.error(f"ステータススタイル適用エラー: {e}")
    
    def render_enhanced_status_display(self, affection: int) -> None:
        """
        拡張されたステータス表示を描画する
        
        Args:
            affection: 現在の好感度
        """
        try:
            # カスタムスタイルを適用
            self.apply_status_styles()
            
            # ステータスコンテナの開始
            st.markdown('<div class="status-container">', unsafe_allow_html=True)
            
            # 好感度ゲージ
            self.render_affection_gauge(affection)
            
            # 関係性ステージ
            self.render_relationship_stage(affection)
            
            # ステータスコンテナの終了
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"拡張ステータス表示エラー: {e}")
            # フォールバック：通常の表示
            self.render_affection_gauge(affection)
            self.render_relationship_stage(affection)
    
    def show_affection_change_notification(self, old_affection: int, 
                                         new_affection: int, reason: str = "") -> None:
        """
        好感度変化の通知を表示する
        
        Args:
            old_affection: 変更前の好感度
            new_affection: 変更後の好感度
            reason: 変化の理由
        """
        try:
            change = new_affection - old_affection
            
            if change == 0:
                return
            
            # 変化の方向に応じてスタイルを決定
            if change > 0:
                icon = "📈"
                color = "#2ed573"
                change_text = f"+{change}"
                css_class = "affection-change-positive"
            else:
                icon = "📉"
                color = "#ff4757"
                change_text = str(change)
                css_class = "affection-change-negative"
            
            # 通知メッセージを作成
            notification_html = f"""
            <div class="{css_class}" style="
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid {color};
                border-radius: 8px;
                padding: 10px;
                margin: 10px 0;
                color: {color};
                text-align: center;
                animation: slideInFromTop 0.5s ease-out;
            ">
                {icon} 好感度が変化しました: {change_text}
                {f'<br><small>{reason}</small>' if reason else ''}
            </div>
            """
            
            st.markdown(notification_html, unsafe_allow_html=True)
            
            # 自動で消える通知（JavaScript）
            auto_hide_js = """
            <script>
            setTimeout(function() {
                const notifications = document.querySelectorAll('.affection-change-positive, .affection-change-negative');
                notifications.forEach(function(notification) {
                    notification.style.transition = 'opacity 0.5s ease-out';
                    notification.style.opacity = '0';
                    setTimeout(function() {
                        notification.remove();
                    }, 500);
                });
            }, 3000);
            </script>
            """
            
            st.markdown(auto_hide_js, unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"好感度変化通知エラー: {e}")
    
    def get_status_display_config(self) -> Dict[str, any]:
        """
        ステータス表示の設定情報を取得する
        
        Returns:
            設定情報の辞書
        """
        try:
            current_affection = st.session_state.get('affection', 30)
            stage_info = self.get_relationship_stage_info(current_affection)
            
            return {
                "current_affection": current_affection,
                "affection_color": self.get_affection_color(current_affection),
                "stage_info": stage_info,
                "history_count": len(st.session_state.get('affection_history', [])),
                "statistics": self.get_affection_statistics(),
                "styles_applied": True
            }
            
        except Exception as e:
            logger.error(f"ステータス表示設定取得エラー: {e}")
            return {
                "current_affection": 30,
                "affection_color": "#ffa502",
                "stage_info": self.stage_colors["中立"],
                "history_count": 0,
                "statistics": {},
                "styles_applied": False
            }