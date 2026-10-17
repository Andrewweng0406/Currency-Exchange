# 常駐事前驗證：部署操作紀錄

最新部署 `21cac22a-1df1-4f34-9f87-e439efdad163` 已達 Railway `SUCCESS`。首次部署 `12165cf6-ddc0-4c7d-8f6a-0621bb40387b` 的啟動紀錄確認 PostgreSQL 遷移至 `20261016_0002` 並啟動單一 daily_pipeline。155 項測試在本機與實際部署的 Linux/CPU 環境均通過；線上測試明確使用隔離 SQLite、關閉排程與通知，不碰正式資料庫。

服務：<https://twd-fx-monitor-production.up.railway.app/health>。排程：<https://twd-fx-monitor-production.up.railway.app/scheduler/status>。驗證狀態：<https://twd-fx-monitor-production.up.railway.app/forward-validation>。

## 保存與執行設定

沿用 Railway 專案 `twd-fx-monitor` 的 production 環境、既有 PostgreSQL 和 API 服務。沒有用本機 SQLite 覆蓋線上資料。

- PostgreSQL 保存行情、銀行報價、歷史預測與新增的事前驗證表。
- `fx-model-evidence` 磁碟（1,000 MB）掛載 `/app/models/artifacts`，保存模型及按內容指紋封存的驗證檔案。
- Docker 啟動先執行 `alembic upgrade head`，成功後啟動一個 Uvicorn worker；Railway 維持單一 replica 並關閉閒置睡眠。
- Railway 健康檢查使用 `/health`，時限 120 秒；失敗重啟上限 10 次。直接依賴版本固定為本機測試通過的版本；間接依賴尚未完整鎖定。
- `SCHEDULER_ENABLED=true`；`LINE_REPORT_ENABLED=false`。本次只啟用資料與驗證，不寄送訊息。
- 每日 22:30 UTC（台北隔日 06:30）依序匯入、建特徵、訓練全部期間、記錄／評分事前預測、評估模型健康。每一步最長 30 分鐘，失敗後停止相依步驟；獨立排程入口也使用同一流程。
- CBC 與銀行觀測必須成功且非空。其他來源失敗明確記為 `DEGRADED`，保留已成功取得的資料；特徵過期限制與模型建議守門仍生效，不能把降級當成完整資料。
- 新模型先寫入同磁碟暫存檔，成功後才原子替換正式檔，避免 API 在訓練時讀到半份模型。
- Linux 採用同版 `xgboost-cpu`，本機 macOS 保留 `xgboost`；版本及發行類型納入規則指紋，分開統計不同運算環境。[XGBoost 官方安裝說明](https://xgboost.readthedocs.io/en/stable/install.html#minimal-installation-cpu-only)確認 CPU 套件可省去 GPU 依賴。

## 實際來源故障與修正

第一輪流程確認 FRED 資料停在 8 月底，並非單純缺少最新一天：

- 同一個官方網址，原自訂 User-Agent 的請求在部署環境逾時；標準 requests 標頭的單一序列請求約 0.1 秒、完整批次約 2 秒完成。FRED 改用標準標頭，其他提供者維持原設定。
- 批次 ZIP 其實分成 `daily.csv`（利率、美元指數、人民幣）和 `daily,_close.csv`（VIX、美股）。原程式只讀前者，現已讀取所有 CSV 成員並逐序列辨識，避免悄悄少掉三項資料。
- 批次請求失敗或缺少序列時，逐項重試相同官方來源。個別失敗不丟掉其餘成功觀測；最後明列各序列成功／失敗。
- 週末或假日的外資資料查詢向前尋找實際交易日，最多 15 個日曆日；保留原日期，拒絕日期不符或空白 OK 回應，網路錯誤不偽裝成休市。

另直接對線上 138 筆舊預測做了 [時序稽核](PRODUCTION_PREDICTION_AUDIT_2026-10-17.md)：60 筆有基本交易日／建立時間問題，其餘也缺少不可變事前證據，不能用來認證模型準確率。

## 更新前備份內容

`data/production_before_forward_20261017.json.gz`：4,365,164 bytes，SHA-256 `f662003a2c17f4d190704e709839e93119de58a9fcbfb48728f805bcdc0098d5`。

這是以 PostgreSQL repeatable-read 交易取得的應用資料邏輯備份，包含 12 張公開資料表的完整列、建表／索引 DDL 及 3 個舊模型檔；不是 PostgreSQL 實體快照或 PITR 備份。檔案只保存在本機，權限 0600，已排除 Git、Docker 與 Railway 上傳。

備份列數：FX 2,708；銀行報價 52；外資 295；市場資料 94,771；特徵 2,654；預測 138；模型評估 9,468；警示 9；換匯計畫 2；AI 解讀 57；macro/event 各 0。

這也確認線上另有本機沒有的銀行與預測歷史，應保留兩份來源，不能直接用本機資料庫取代。

## 驗證方式

1. 以本次部署 ID 追蹤到 Railway `SUCCESS`，不能把上傳完成視為上線。
2. 確認 `/health`、`/scheduler/status`、`/forward-validation`、`/overview`；查看單一相依排程與停用的模型建議。
3. 在服務內執行 `python scripts/run_daily_pipeline.py`，確認所有資料與驗證步驟完成且沒有 LINE 步驟。
4. 檢查官方資料來源、模型版本、封存檔指紋、資料庫遷移版本與新預測實際記錄時間；同日重跑不得灌大樣本。
5. 重啟後確認資料庫紀錄與磁碟檔案仍存在、排程重新註冊。這只能驗證恢復能力；仍要等待未來每日執行和真正到期樣本，不能提前宣稱長期驗證成功。

## 已完成的實際檢查

- 必要程式更新的重新部署後，逐一比對：15 張表的列數、首批 3 筆完整事前紀錄、26 個模型／封存檔的內容指紋完全相同，確認資料並未隨容器替換遺失。沒有額外重啟資料庫服務。
- 2026-10-17 06:45:25 UTC，本次修正後的完整匯入回報 `status=OK`，21 個來源項目全數成功、無缺少必要來源；處理 96,372 筆觀測，包含既有資料的重新匯入，並非全數新增列。
- 三個舊 v1 模型也已從更新前備份保存到持久化磁碟，內容 SHA-256 與備份一致。舊版本行情、特徵、預測與 AI 紀錄未用本機資料庫覆蓋。
- 06:49:34 UTC 完整流程結束，ingestion、features、models、forward_validation、model_health 五步均回傳 0，整體 `ok=true`，沒有 LINE 步驟。
- 新規則 `992c4840a14ebdaca5571277d1762558c7f3bea3a5e695fcbb772c42e40eb778` 於 06:49:30 UTC 記錄 1／5／20 日三筆預測：0 到期、3 待評分。舊規則 `fa83f35a…` 的三筆另組保留；合計六筆仍只有一個來源交易日，不能視為六個獨立交易日。
- 逐筆檢查六份 payload、規則 JSON、模型、相關來源程式檔的 SHA-256 均相符，所有輸入欄位齊全。首批三筆完整紀錄與 23 個應維持不變的舊模型／封存檔均未改動；三個最新版 v3 模型檔則按預期更新。磁碟合計 32 個模型／封存檔。
- 同日再執行事前記錄：新增 0、重複 3、缺少期間 0、新結果 0；`COLLECTING`、指紋異常 0。
- `/health`、`/scheduler/status`、`/forward-validation`、`/overview` 均回傳 200。`advice_available=false`、`exchange=null`；API 與 PostgreSQL 各一個正常 replica，Railway 無待套用工作或服務問題。

## 完成時的資料庫快照

FX 7,323（其中 CBC 4,669）；銀行報價 53（華南 1、歷史土銀 52）；市場資料 95,018；外資 295；特徵 7,323（daily_v1 2,654、daily_v3 4,669）；原預測 141（v1 138、v3 3）；模型評估 9,648；事前紀錄 6；事前結果 0。macro_data／economic_events 仍為空，不宣稱這兩類已補齊。

FRED 七項最新來源日期：US_2Y／US_10Y／NASDAQ／VIX 為 10/15；SP500 為 10/16；BROAD_USD_INDEX／USD_CNY 為 10/09。日期依來源實際提供內容保留，H.10 仍按隔週發布規則對齊可用時間，沒有把資料重標成今天。

下一次排程：2026-10-17 22:30 UTC，即台北 10/18 06:30。尚未等待這次未來排程真正觸發，也沒有尚未到期預測的實測成績；本輪驗證的是已上線流程、手動完整執行、重新部署後保存能力與同日冪等性。
