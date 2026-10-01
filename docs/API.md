# Finpre API v1

每個訊號都附成績單的 AI 投資研究 API。互動式文件（OpenAPI）：啟動後開 `http://<host>:8000/docs`。

```bash
uvicorn fintech_agent.api.main:app --host 0.0.0.0 --port 8000      # 或 docker compose up -d api
python scripts/manage_tenants.py create --name "王小明" --plan research   # 金鑰只顯示一次
```

## 認證

每個請求帶 API 金鑰：`X-API-Key: fp_...` 或 `Authorization: Bearer fp_...`。金鑰只以雜湊保存，遺失請由管理者 `rotate-key` 重發（舊金鑰立即失效）。

## 合規模式

金鑰所屬的租戶決定模式，客戶端無法切換：

| 模式 | 適用 | 回傳內容 |
|---|---|---|
| `research`（預設） | 所有非持牌客戶 | 訊號、分數、信心、報酬率區間（%）、波動、模型成績。**不含**買賣建議、價位、部位 |
| `advisor` | 企業版 **且** 持牌證券投資顧問事業 | 另含動作、部位、進場區間、停損、停利 |

研究模式回應中的 `compliance` 區塊只告訴你過濾了幾句、幾個欄位（`removed_sentences`、`removed_fields`），被過濾的原文只保存在稽核紀錄。

## 端點

| 方法 | 路徑 | 說明 | 計量 |
|---|---|---|---|
| GET | `/v1/health` | 健康檢查 | 否 |
| GET | `/v1/plans` | 方案與限制 | 否 |
| GET | `/v1/me` | 目前租戶、方案、本月用量 | 否 |
| POST | `/v1/analyze` | 多代理完整分析（技術、基本面／籌碼／情緒、量化 → 整合結論） | 是，計入每日完整分析上限 |
| POST | `/v1/forecast` | champion 模型的報酬率分位數、上漲機率 | 是 |
| GET | `/v1/ranking/{TW\|US}?horizon=5` | 最新選股排序（研究清單）與其回測成績 | 是 |
| POST | `/v1/portfolio/risk` | 風險雷達：組合報酬區間、VaR／ES、集中度、尾端損失貢獻 | 是 |
| GET | `/v1/report/{TW\|US}?watchlist=2330,2317&format=json\|html\|md` | 每日盤後研究報告 | 是 |
| GET | `/v1/scorecard` | 成績單：回測＋上線後追蹤 | 是 |
| GET | `/v1/usage?month=2026-10` | 用量明細 | 否 |
| GET | `/v1/audit/export` | 本租戶的稽核紀錄（JSON Lines） | 企業版 |

### 範例

```bash
H="X-API-Key: $FINPRE_KEY"
curl -s -H "$H" -H 'content-type: application/json' -d '{"ticker":"2330","horizon":5,"risk":"穩健"}' localhost:8000/v1/analyze
curl -s -H "$H" -H 'content-type: application/json' -d '{"ticker":"NVDA","horizon":20}' localhost:8000/v1/forecast
curl -s -H "$H" -H 'content-type: application/json' \
     -d '{"holdings":{"2330":1000,"2317":2000,"NVDA":30},"horizon":5,"cash":200000,"base_currency":"TWD"}' \
     localhost:8000/v1/portfolio/risk
curl -s -H "$H" "localhost:8000/v1/report/TW?watchlist=2330,2454&format=html" > report.html
```

`/v1/analyze` 研究模式回應（節錄）：

```json
{
  "ticker": "2330", "name": "台積電", "market": "TW", "close": "…", "as_of": "…", "mode": "research",
  "decision": {
    "signal": "偏多", "signal_score": 0.88, "conviction": 63, "return_band_pct": [-3.5, 4.1], "hv20_ann_pct": 19.2,
    "client_message": "綜合技術、基本面／籌碼與量化三位分析師，台積電 目前的綜合訊號為「偏多」…以上為統計模型的研究結果，並非買賣建議。",
    "watch_items": ["量化模型的上漲機率跌破 45% 或升破 60%", "…"], "disclaimer": "本服務為投資研究工具…"
  },
  "agents": {"technical": {"stance": "看多", "score": 0.9, "summary": "…"}, "fundamental": {…}, "quant": {…}},
  "compliance": {"mode": "research", "removed_sentences": 0, "removed_fields": 9},
  "elapsed_s": 10.6, "cached": false
}
```

同一模式、同一請求在 30 分鐘內重複呼叫會由快取回覆（`"cached": true`，仍計量、仍寫稽核）。

## 錯誤

| 狀態碼 | 意義 |
|---|---|
| 400 | 輸入錯誤（例如代號不存在、資料不足）——**不計量** |
| 401 | 缺少或無效的 API 金鑰 |
| 403 | 超過方案限制（觀察清單、持股數）或功能不在方案內（稽核匯出） |
| 404 | 尚未產生排名 |
| 422 | 參數格式錯誤（例如持股股數 ≤ 0） |
| 429 | 超過每日完整分析或每月呼叫上限 |

## 方案限制

| 方案 | 每日完整分析 | 每月呼叫 | 觀察清單 | 風險雷達持股 | 排名 | 顧問模式 | 稽核匯出 |
|---|---|---|---|---|---|---|---|
| 免費 | 3 | 300 | 5 | 5 | 前 5 名 | – | – |
| 研究版 | 50 | 3,000 | 30 | 30 | 全部 | – | – |
| 專業版 | 200 | 5,000 | 200 | 60 | 全部 | – | – |
| 企業版 | 10,000 | 200,000 | 2,000 | 500 | 全部 | 持牌機構 | ✓ |

> 所有回應皆為研究資訊，不構成投資建議。過去績效不代表未來表現。
