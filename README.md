# Fintech Agent — TimesFM 驅動的多代理 AI 投資分析系統

台灣 + 美國市場。四個 Agent 協作：**技術分析師**、**基本面／籌碼／情緒分析師**、**量化 ML 工程師**（TimesFM 2.5 為 v1 基準 champion；v0.2 的 50 檔回測後，由加入籌碼共變數的本地 LightGBM 接任 champion，Chronos-2 / TimesFM 3.0 等持續當 challenger，可做 A/B 測試與模型切換），由 **首席投資顧問** 統籌做最終決策並與客戶對話。

詳細設計見 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## 快速開始（M2 Mac）

```bash
git clone git@github.com:tim3959951/Finpre.git && cd Finpre
uv venv --python python3.11 .venv && source .venv/bin/activate
uv pip install -e ".[models,llm,dev]"
cp .env.example .env        # 填入 ANTHROPIC_API_KEY / OPENAI_API_KEY / FINMIND_TOKEN（皆為選填）

pytest -q                                   # 單元測試（離線、合成資料）
python scripts/analyze.py 2330 --provider none          # 規則模式，不需任何 LLM
python scripts/analyze.py NVDA --provider anthropic     # Anthropic API 當大腦
python scripts/analyze.py 2330 --provider ollama --model qwen3:8b   # 本地 LLM（先 ollama pull qwen3:8b）
streamlit run fintech_agent/ui/app.py       # 網頁介面：對話 / 個股分析 / 模型實驗室
```

首次執行會從 Hugging Face 下載模型權重（TimesFM 2.5 ≈ 0.9 GB、Chronos-2 ≈ 0.5 GB、Chronos-Bolt small ≈ 0.2 GB）。

## 模型 benchmark 與 A/B 測試

```bash
# Walk-forward 回測，只評估模型發布後的區間（避免預訓練資料洩漏）
python scripts/benchmark.py --market TW --horizon 5 --windows 60 --min-origin 2025-10-01
python scripts/benchmark.py --market US --horizon 20 --windows 40 --step 20
# 挑戰者顯著勝出時自動升級 champion
python scripts/benchmark.py --market TW --horizon 5 --promote
# 50 檔 + 籌碼/大盤/匯率共變數（台股需 FinMind，建議設定 FINMIND_TOKEN 提高額度）
python scripts/benchmark.py --universe tw50 --horizon 5 --windows 60 --min-origin 2025-10-01 \
  --models naive timesfm-2.5 timesfm-2.5-xreg chronos-2 chronos-2-cov lgbm lgbm-cov ensemble ensemble-cov
# 各市場整合結果成 markdown 表格
python scripts/summarize_benchmarks.py TW:5:logs/b50_tw50_h5.csv US:5:logs/b50_us50_h5.csv

# 訓練本地模型（量化 Agent 會自動載入 checkpoints/；LightGBM 每個約 5–15 秒）
python scripts/train_models.py --universe tw50 --models lgbm lgbm-cov --horizons 5 20
python scripts/train_models.py --universe us50 --models lgbm lgbm-cov --horizons 5 20
python scripts/train_models.py --universe us50 --models dlinear --horizons 20 --epochs 20   # M2 GPU (MPS)
```

Champion 查找順序：本機升級紀錄 `runs/champion.json` → `settings.yaml` 的 `forecasting.champions`（v0.2 的回測結果）→ 預設 `forecasting.champion`（TimesFM 2.5）。若 champion 是 LightGBM 但本機尚未訓練 checkpoint，量化 Agent 會自動退回 TimesFM 2.5 並在 log 提示執行 `train_models.py`。

### v0.2 結果摘要（台股 50 + 美股 50 檔，2025-10 → 2026-09，CRPS 相對 random walk）

| | 台股 5 日 | 台股 20 日 | 美股 5 日 | 美股 20 日 |
|---|---|---|---|---|
| 最佳模型 | **lgbm-cov +3.4%**（72% 個股勝出，p=0.003） | **lgbm +3.6%**（80%，p<0.001） | lgbm +0.5%（不顯著） | lgbm-cov +0.8%（不顯著） |
| TimesFM 2.5 | −0.3% | −1.6% | −2.3% | −4.0% |
| Chronos-2 | −0.5% | −0.9% | −1.7% | 0.0% |

台股比美股更有可預測結構；籌碼共變數只對「本地訓練」的 LightGBM 有幫助（台股 5 日 IC 由 0.01 升到 0.11），對 zero-shot 基礎模型反而有害。完整表格與解讀見 `docs/ARCHITECTURE.md` §11，原始報告在 `docs/benchmarks/v0.2/`。

| 模型 | 類型 | 授權 | 備註 |
|---|---|---|---|
| `timesfm-2.5` | 基礎模型 | Apache-2.0 | v1 champion / 預設退回模型，200M，16k context，分位數輸出；80% 區間覆蓋最準 |
| `timesfm-3.0` | 基礎模型 | **非商用** | 僅供研究比較；`allow_noncommercial_models: true` 才會在 UI 出現 |
| `chronos-2` / `chronos-bolt-small` / `chronos-bolt-base` | 基礎模型 | Apache-2.0 | Amazon |
| `tirex` | 基礎模型 | NXAI 社群授權 | 選配：`pip install tirex-ts` |
| `naive` / `drift` / `arima` | 統計基準 | – | naive（random walk）是必須打敗的基準 |
| `dlinear` / `lgbm` | 本地訓練 | – | DLinear 在 MPS 上訓練；LightGBM 分位數迴歸（**v0.2 champion：台股 20 日、美股 5 日**）|
| `ensemble` | 集成 | – | 成員見 `forecasting.ensemble_members` |
| `chronos-2-cov` / `timesfm-2.5-xreg` / `lgbm-cov` / `ensemble-cov` | 共變數版 | 同原模型 | 加入三大法人、融資、大盤、費半、匯率（美股：S&P 500、VIX、利率、美元）；**lgbm-cov 為 v0.2 champion：台股 5 日、美股 20 日** |
| `timesfm-3.0-cov` | 共變數版 | **非商用** | 僅研究比較 |

## 專案結構

```
fintech_agent/
  data/          yfinance + FinMind 資料、快取、籌碼/營收/估值摘要
  features/      技術指標、技術/基本面/籌碼/情緒規則評分
  forecasting/   統一 Forecaster 介面、TimesFM/Chronos/TiRex、基準、DLinear/LGBM、registry
  evaluation/    walk-forward 回測、指標、Diebold-Mariano A/B、實驗紀錄與 champion registry、shadow 線上 A/B
  llm/           可插拔 LLM（Anthropic / OpenAI 相容 / Ollama / 規則模式）+ tool-calling loop
  agents/        三位專家 + 首席投資顧問（調度、決策、對話）
  ui/            Streamlit 介面
scripts/         analyze / benchmark / summarize_benchmarks / train_models
config/settings.yaml
```

> 本系統輸出僅供研究與教育用途，不構成投資建議。若要對外提供個股買賣建議服務，在台灣需具證券投資顧問事業許可。
