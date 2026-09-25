# Fintech Agent — TimesFM 驅動的多代理 AI 投資分析系統

台灣 + 美國市場。四個 Agent 協作：**技術分析師**、**基本面／籌碼／情緒分析師**、**量化 ML 工程師**（TimesFM 2.5 為 v1 champion，Chronos-2 等為 benchmark，可做 A/B 測試與模型切換），由 **首席投資顧問** 統籌做最終決策並與客戶對話。

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
# 在 M2 GPU (MPS) 上訓練 DLinear 全域模型
python scripts/train_models.py --market ALL --epochs 20
```

| 模型 | 類型 | 授權 | 備註 |
|---|---|---|---|
| `timesfm-2.5` | 基礎模型 | Apache-2.0 | **v1 champion**，200M，16k context，分位數輸出 |
| `timesfm-3.0` | 基礎模型 | **非商用** | 僅供研究比較；`allow_noncommercial_models: true` 才會在 UI 出現 |
| `chronos-2` / `chronos-bolt-small` / `chronos-bolt-base` | 基礎模型 | Apache-2.0 | Amazon |
| `tirex` | 基礎模型 | NXAI 社群授權 | 選配：`pip install tirex-ts` |
| `naive` / `drift` / `arima` | 統計基準 | – | naive（random walk）是必須打敗的基準 |
| `dlinear` / `lgbm` | 本地訓練 | – | DLinear 在 MPS 上訓練；LightGBM 分位數迴歸 |
| `ensemble` | 集成 | – | 成員見 `forecasting.ensemble_members` |

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
scripts/         analyze / benchmark / train_models
config/settings.yaml
```

> 本系統輸出僅供研究與教育用途，不構成投資建議。若要對外提供個股買賣建議服務，在台灣需具證券投資顧問事業許可。
