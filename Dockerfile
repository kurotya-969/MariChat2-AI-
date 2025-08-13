FROM python:3.10-slim

# システムの依存関係をインストール（最初にrootで実行）
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 非rootユーザーを作成
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 作業ディレクトリを設定
WORKDIR $HOME/app

# Pythonの依存関係をコピーしてインストール
COPY --chown=user requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# アプリケーションファイルをコピー
COPY --chown=user . .

# Streamlit設定ディレクトリを作成
RUN mkdir -p $HOME/.streamlit

# Streamlit設定ファイルを作成
RUN echo '\
[server]\n\
headless = true\n\
address = "0.0.0.0"\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[theme]\n\
primaryColor = "#FF6B6B"\n\
backgroundColor = "#0E1117"\n\
secondaryBackgroundColor = "#262730"\n\
textColor = "#FAFAFA"\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
' > $HOME/.streamlit/config.toml

# 動的ポートを公開（Render対応）
EXPOSE $PORT

# 起動スクリプトを作成
RUN echo '#!/bin/bash\n\
PORT=${PORT:-8501}\n\
streamlit run main_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false\n\
' > start.sh && chmod +x start.sh

# Streamlitアプリケーションを起動
CMD ["./start.sh"]