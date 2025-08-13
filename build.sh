#!/bin/bash

# Vercel用ビルドスクリプト
echo "Installing Python dependencies..."

# 依存関係をインストール
pip install -r requirements.txt

echo "Build completed successfully!"