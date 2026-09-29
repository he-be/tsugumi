#!/bin/zsh
# 完成品ウェイトを Hugging Face に上げる (Mac アプリの直DLインストーラの供給源)。
#
# アプリ側のピンは Sources/TsugumiApp/Core/Installation/ にある:
#   - PrebuiltModelSource.swift  … リポジトリ名
#   - PrebuiltFileTables.swift   … ファイルごとの bytes / SHA-256
# ここでアップロードする内容を変えたら、その 2 つを必ず作り直すこと
# (テーブルは shasum -a 256 の出力から生成した。生成手順はファイル冒頭のコメント)。
#
# 必要なもの: 書き込み権限つきの HF トークン (hf auth login --token ...)。
# 手元の ~/.cache/huggingface/token は read 専用なので、そのままでは失敗する。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OWNER="mh73772"

GEMMA_REPO="$OWNER/turbofieldfare-gemma4-qat-sym"

echo "== gemma4-qat-sym (15.7 GB) =="
hf repo create "$GEMMA_REPO" --repo-type model --private 2>/dev/null || true
hf upload "$GEMMA_REPO" "$REPO_ROOT/scratch/gemma4-qat-sym.moepack" . \
    --repo-type model \
    --exclude "verified-install.json" --exclude "*.install.lock" --exclude ".DS_Store"

# Ornith (turbofieldfare-ornith-oq4e-g64) はアプリから外した (2026-09-29)。Hub のリポジトリはそのまま。

echo "done. アプリの直DLはこのリポジトリを SHA-256 ピンで検証して読む。"
echo "公開する場合: hf repo settings で private を外す (アプリは公開でも HF_TOKEN でも動く)。"
