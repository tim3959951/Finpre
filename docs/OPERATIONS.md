# Finpre 維運手冊（Runbook）

> 對象：部署與維運 Finpre 的工程師、處理客戶帳號的營運人員、需要稽核資料的法遵人員。

## 1. 元件

| 服務 | 指令 | 連接埠 | 說明 |
|---|---|---|---|
| API | `uvicorn fintech_agent.api.main:app --workers 2` | 8000 | B2B API，`/docs` 為 OpenAPI 文件 |
| 網頁 | `streamlit run fintech_agent/ui/app.py` | 8501 | 客戶以 API 金鑰登入（`FA_UI_AUTH=apikey`） |
| 排程 | `python scripts/scheduler.py` | – | 台北時間 15:40（台股，週一至五）、06:30（美股，週二至六） |

狀態都在磁碟上（Docker 以 volume 掛載）：

| 路徑 | 內容 | 備份 |
|---|---|---|
| `runs/audit.sqlite`、`runs/audit_anchors.jsonl` | 稽核紀錄與每日錨點 | **必須**，保存年限依合約與法規 |
| `runs/tenants.sqlite` | 客戶、方案、金鑰雜湊、用量 | 必須 |
| `runs/experiments.sqlite`、`runs/champion.json` | 回測、線上預測與到期結算、champion | 建議 |
| `runs/ranking_*.json`、`runs/ranking_history/` | 最新排名與每日排名封存（算上線後 IC） | 建議 |
| `checkpoints/` | 訓練好的 LightGBM | 可重建（`train_models.py`） |
| `reports/` | 每日報告 JSON／HTML／Markdown | 建議 |
| `data_cache/` | 市場資料快取 | 可重建 |

## 2. 部署

```bash
cp .env.example .env            # 至少設定 FA_AUDIT_KEY、FINMIND_TOKEN；要 LLM 時設定 ANTHROPIC_API_KEY 等
docker compose build
docker compose run --rm api python scripts/train_models.py --universe tw50 --models lgbm --horizons 5 20 --years 13
docker compose run --rm api python scripts/train_models.py --universe us50 --models lgbm --horizons 5 20 --years 13
docker compose run --rm api python scripts/daily_job.py --market TW --force     # 第一次產生排名、報告、成績單
docker compose up -d
curl -s localhost:8000/v1/health
```

上線前檢查清單：
- [ ] `FA_AUDIT_KEY` 已設定為長隨機字串，且不在版本庫中。
- [ ] 網頁 `FA_UI_AUTH=apikey`、`FA_UI_ADMIN=0`（compose 預設如此）。
- [ ] 前面加上 HTTPS 反向代理（Caddy／Nginx／雲端負載平衡），API 與網頁都不直接暴露 HTTP。
- [ ] `python scripts/compliance_check.py` 通過（CI 每次推送也會跑）。
- [ ] 資料源授權：yfinance 不可商用，正式上線前換成授權資料源（見 PRODUCT.md §8）。
- [ ] 法律意見：研究模式的呈現方式已請熟悉證券法規的律師確認。

## 3. 客戶與金鑰

```bash
python scripts/manage_tenants.py create --name "王小明" --plan research          # 金鑰只顯示一次
python scripts/manage_tenants.py create --name "某某投顧" --plan enterprise --licensed --mode advisor
python scripts/manage_tenants.py list
python scripts/manage_tenants.py set-plan t_xxxx --plan pro                     # 升降級
python scripts/manage_tenants.py set-plan t_xxxx --plan enterprise --no-licensed  # 撤銷持牌 → 自動回到研究模式
python scripts/manage_tenants.py rotate-key t_xxxx                              # 金鑰外洩：立即換發
python scripts/manage_tenants.py usage t_xxxx --month 2026-10                   # 對帳
python scripts/manage_tenants.py deactivate t_xxxx                              # 停權
```

顧問模式只有在「企業版＋持牌」兩個條件都成立時才能開啟；程式會拒絕其他組合。開通前須取得對方的證券投資顧問事業許可證影本並簽約。

## 4. 每日排程

`daily_job.py --market TW|US` 依序執行：休市判斷 → 更新排名（h5、h20）→ 追蹤股票池預測 → 結算到期預測 → 每日報告（並寫入稽核）→ 對發布的報告檔做合規掃描 → 成績單 → 驗證並錨定稽核紀錄。

| 退出碼 | 意義 | 處理 |
|---|---|---|
| 0 | 正常（或休市日略過） | – |
| 2 | 稽核鏈驗證失敗，或發布的報告含禁用內容 | **立即處理**，見 §6 |
| 3 | 某一步失敗（資料源、排名、報告…） | 看日誌，修正後 `--force` 重跑；其他步驟已照常完成 |

手動重跑：`python scripts/daily_job.py --market TW --force`（`--skip-ranking` 可省下約 10 分鐘）。

## 5. 監控

| 檢查 | 方式 | 警戒 |
|---|---|---|
| API 存活 | `GET /v1/health` | 連續 3 次失敗 |
| 每日排程 | 排程日誌的 `daily job finished with exit code` | 非 0 |
| 稽核鏈 | `python scripts/manage_tenants.py audit-verify` | 非 0 |
| 合規 | `python scripts/compliance_check.py --reports reports --audit 1000` | 非 0 |
| 模型品質 | `runs/scorecard.json`：80% 區間覆蓋、方向命中、相對 random walk 誤差；排名上線 IC | 覆蓋 < 70% 或 > 90%；滾動 IC 連續 8 週 < 0 |
| 用量 | `manage_tenants.py usage` | 接近方案上限的客戶 → 業務跟進 |

## 6. 事件處理

**研究模式出現禁用內容**（compliance_check 或 daily_job 退出碼 2）
1. 找出內容：`compliance_check.py` 會列出報告檔或稽核列編號。
2. 若是已發布的報告：從發送管道撤下，保留稽核紀錄（不要刪）。
3. 把該句加入 `tests/test_hardening.py` 的案例，修正 `product/compliance.py`，測試通過後部署。
4. 記錄事件：時間、影響的客戶（稽核紀錄的 actor）、原因、修正。

**稽核鏈驗證失敗**
1. 不要再寫入：停止 API 與排程（`docker compose stop api scheduler`）。
2. `audit-verify` 給出第一個不一致的列號；與備份及錨點檔比對，判斷是損毀還是遭竄改。
3. 從最近一次乾淨備份還原，或把損毀檔封存後另開新檔（新檔第一列記錄事件），通知法遵。

**金鑰外洩**：`rotate-key` → 檢查 `audit/export` 與用量是否有異常呼叫。

**資料源中斷**（FinMind／Yahoo）：daily_job 退出碼 3；API 會回 400「資料不足」且不計量。恢復後 `--force` 重跑。

## 7. 模型維運

- 每月（或成績單顯示品質下滑時）重新訓練：`train_models.py --universe tw50|us50 --models lgbm --horizons 5 20 --years 13`。
- 換 champion 必須通過 A/B（`scripts/benchmark.py --promote`，或網頁「模型實驗室」——僅營運者可見），紀錄寫入 `runs/champion.json`。
- 非商用授權模型（TimesFM 3.0）預設不會出現在可用清單，對話工具也無法呼叫。
