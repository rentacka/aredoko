# 🫧 🌸 あれどこだっけ？（AreDoko）ローカル AI 記憶庫

> **「もう、フォルダ分けしなくていい。『あれ、どこだっけ？』を 0.5 秒で思い出す AI 記憶庫。」**

## ✨ 主な機能
1. **ワンクリック・ブラウザ同期（クロスプラットフォーム対応）**:
   - Windows / macOS / Linux の **Chrome / Edge / Brave** の `Bookmarks` JSON を読み取り専用で安全に取り込み。
   - **Safari / Firefox / Chrome エクスポート HTML** のドラッグ＆ドロップ取込にも対応。
2. **ぽんっと投げ込み箱**:
   - URL・X（旧Twitter）ポスト・雑メモをそのまま保存＆即座にベクトル化。
3. **0.5 秒ハイブリッド意味検索（完全ローカル・API代 0 円）**:
   - **Dense（Port 1919 EmbeddingGemma-2: 768次元）＋ Sparse（日本語 2-gram / BM25）** による高速検索。
   - **Port 1918（ローカル金魚）** による AI 精密リランク ON/OFF 切替（オフライン時は自動で軽量検索へフォールバック）。
4. **AI 要約エンリッチ（Phase 2）**:
   - 未要約のブックマークのページ概要をバックグラウンド取得し、Port 1918 で「1行要約 ＋ 連想タグ」を自動補完。
5. **3 テーマ着せ替え（自動保存）**:
   - **✨ Glass（パステル・グラス / 標準）** / **🎀 Sweet（さくらスイート）** / **🌙 Dark（ミッドナイト・グラス）**。

## 🚀 起動方法
- Windows: [start_aredoko.bat](file:///j:/Antigravity/Aria_Cute💕/aredoko/start_aredoko.bat) をダブルクリック！
- コマンドライン（Windows / macOS / Linux 共通）:
  ```bash
  python app.py
  ```
  起動後、ブラウザで `http://127.0.0.1:1925` を開いてください。

## 🗺️ 今後の展開（ブラウザ拡張・スマホ連携メモ）
- CORS を開放済みのため、将来の **Chrome / Edge 拡張機能（Manifest V3 サイドパネル）** からそのまま `http://127.0.0.1:1925/api/...` へ接続可能。
- PC サーバー無し・スマホ単体動作へ広げる場合は、ブラウザ内軽量推論（WebGPU / ONNX Runtime Web / Transformers.js）への差し替え、または軽量同期モデルで展開予定。
