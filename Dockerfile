FROM python:3.10-slim

# 非rootユーザーを作成
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 作業ディレクトリを設定
WORKDIR $HOME/app

# システムの依存関係をインストール（rootで実行）
USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# userに戻る
USER user

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
port = 7860\n\
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

# ポート7860を公開（Hugging Face Spaces標準）
EXPOSE 7860

# ヘルスチェック
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health

# Streamlitアプリケーションを起動
CMD ["streamlit", "run", "main_app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]