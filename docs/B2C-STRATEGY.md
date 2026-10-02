# Finpre B2C App 策略與競品研究（2026-10-01）

> 本文整理台灣與國外投資研究 App 的功能、定價與合規做法，作為 Finpre 直接面向一般投資人的 App 規劃依據。法規部分是資料整理，不是法律意見；標「推論」者需律師確認。

## 1. 結論

1. **B2C 可以做，而且市場夠大**：台灣證券開戶 1,376.78 萬人（2025 年底），單季有交易的散戶約 583 萬人；複委託開戶約 700 萬戶（美股約占 8 成）；2025 年 12 月新開戶中 30 歲以下占 62.4%。投信投顧公會《2025 年臺灣民眾對於共同基金投資之問卷調查》（2026/4/30 發布，1,493 份）：民眾最期待的 AI 理財功能是「風險預警與市場分析」（61%），其次是個人化投資組合（52%）。
2. **B2B 不是穩的**：大型券商都在自建 AI（富邦 AI PRO、國泰 AI 台股日報、永豐大戶投、凱基 AI 助理）；與投顧合作時，依「投顧與網路業者合作三不」，技術方**不得分潤顧問費**，只能收固定費用。
3. **市場領先者的 B2C 都走「資訊工具」路線**：財報狗（自稱會員 100 萬以上，NT$599／799 每月，含 AI 助理 60／120 次）、CMoney 籌碼K線（Google Play 下載 100 萬以上，NT$680／月）、三竹智選股（NT$290／月）、Yahoo 股市 VIP（NT$299／月）。公開頁面都沒有揭露投顧執照，靠「僅供參考」免責。
4. **法規紅線很清楚**：被判刑的案例都是「收費＋個股買賣建議或價位＋反覆提供」（例：2025/4 PressPlay 收月費提供盤前預測與買賣建議，判 6 月併科 100 萬；2026/6 Discord 收費群，判 6 月併科 100 萬緩刑 3 年）。金管會監理門診 FAQ：提供個股「合理價位」即屬投顧業務；SITCA 問答 Q7：支撐壓力、停損停利、買賣價位可能違法；Q6：市場資訊基本統計分析、技術理論教學不算。2023–2026 年沒有找到以 AI 選股 App 為對象的處分。
5. **Finpre 的 B2C 設計原則**：把「意見」改寫成「可驗證的統計事實」，把付費放在「工具」而不是「對個股的看法」。

## 2. 競品有、Finpre 還沒有的功能

| 功能 | 誰在做 | Finpre 做法 | 法規風險 |
|---|---|---|---|
| 「為什麼動了」自選股／持股每日推播 | Robinhood Cortex Digests（約 100 萬人用過）、Toss AI Signal、國泰 AI 庫存管家（開啟率約一般推播 20 倍） | 盤後報告改成個人化推播：漲跌原因＝新聞＋法人＋大盤歸因 | 低（事實歸因） |
| 持股匯入＋組合健檢 | 富邦 AI PRO（截圖匯入）、Simply Wall St、TipRanks、Webull Vega Portfolio | 截圖／CSV 匯入 → 風險雷達（VaR、集中度、匯率） | 低 |
| 事件警示 | 三竹 36 種警示、大戶投 30+ 因子推播、富果到價提醒 | 籌碼異常、波動放大、風險雷達旗標變化、到價 | 低 |
| 使用者自訂選股（含自然語言） | XQ、三竹、財報狗 100+ 條件、同花順问财 | 「外資連買 3 天且營收年增 > 20%」直接轉成條件 | 低（使用者自訂） |
| 法說會逐字稿與摘要 | 優分析、XQ、Perplexity、Toss AI Earnings Call | 台股法說會摘要＋重點數字 | 低 |
| 回答附來源 | Fiscal.ai、Perplexity | AI 每句話連到原始數據或公告 | 低（降低誤導） |
| 即時題材排行＋供應鏈連動 | Toss Real-Time Issues | 題材 → 相關個股 → 法人動向 | 低 |
| 白話籌碼＋歷史類似情境統計 | 沒有人做好 | 「過去 10 年出現類似籌碼型態後 20 日報酬分佈」 | 低～中（統計，不下結論） |
| 公開成績單／審計頁 | Danelfin Audit、Zacks、Seeking Alpha Quant 績效 | 已有，強化成獨立頁面，含失敗年份 | 中（績效宣傳要誠實） |
| 透過 ChatGPT 等 AI 助理的 MCP 散布 | 富果 Fugle.AI、Fiscal.ai、Webull、moomoo | 提供 Finpre MCP 伺服器當獲客管道 | 中 |
| 估值視覺化（本益比河流圖） | 財報狗 | 只呈現歷史本益比區間，不給「合理價」 | 中（不可給合理價） |
| 社群／使用者論點 | 爆料同學會、Simply Wall St Narratives | 暫不做（炒作與「主力坑殺」風險） | 高 |
| 個股評分／買賣評等、目標價 | Seeking Alpha、Danelfin、AI股神（支撐壓力） | 不做（台灣無美國出版商豁免） | 高 |

## 3. 競品的弱點（用戶抱怨）＝ Finpre 的機會

- 付費仍有大量廣告（籌碼K線）→ Finpre 付費版零廣告。
- 開盤當機、閃退（三竹、富邦）→ 盤中只讀快取，重運算放盤後。
- 新手看不懂籌碼（籌碼K線、XQ）→ 白話解釋＋歷史統計。
- AI 宣稱準確率卻不揭露（凱基 5 日漲跌機率、小 App 宣稱 90% 命中）→ Finpre 公開成績單，包含 2022 年全部落後 random walk。
- 社團誇大報酬、對帳單不一致（玩股網）→ 不做喊單、不曬績效。
- 券商 AI 綁自家客戶、只涵蓋熱門股（大戶投 168 檔）→ 中立、不綁券商、台美股都有。

## 4. 合規設計（B2C 版本）

| 層級 | 內容 | 收費 |
|---|---|---|
| 公開資訊與統計 | 個股資料、財報、法人買賣超、技術指標計算、新聞摘要、法說會摘要、歷史統計（類似情境後的報酬分佈、模型歷史成績） | 免費（含每日次數上限） |
| 使用者自己的工具 | 持股匯入、風險雷達、自訂選股、警示與推播、每日個人化摘要、AI 解釋資料與概念 | 付費 |
| 不提供 | 買賣建議、買進／賣出評等、目標價、合理價、支撐壓力、停損停利、「該不該買」的回答、喊單社群 | – |

- 現在的「綜合訊號偏多／偏空」屬於對個股的分析意見，收費時是灰色地帶（推論）。B2C 版改為：呈現各面向的客觀統計與模型歷史成績，或只放在免費層、不廣告導流。最終界線請律師與金管會 FinTechSpace 監理門診確認。
- 免費＋廣告不是安全港（法條含「間接」與「第三人」報酬）；不接券商或金融商品廣告。
- 不宣稱「高準確率」（參考 SEC「AI washing」罰款案、FTC Operation AI Comply）。

## 5. 定價參考

- 台灣：入門 NT$250–300、中階 NT$500–800／月；年繳約 10 個月價。財報狗用 AI 次數分級（60／120 次）。
- 國外：研究訂閱 US$10–40／月為主流；Robinhood Gold US$5／月，Gold 訂閱 430 萬。
- 轉換率參考（RevenueCat 2026，全品類）：硬付費牆 D35 轉換 10.7%、freemium 2.1%；17–32 天試用轉換 42.5%。同花順问财：媒體推算付費轉換約 5.3%，AI 付費用戶超過 180 萬。
- 建議：免費（每日 3 次分析、自選 10 檔）／標準 NT$299（持股風險雷達、警示、每日推播、AI 問答 100 次）／進階 NT$690（無限分析、自訂選股、法說會摘要、美股）。年繳 10 個月價，14 天試用。

## 6. 產品路線（B2C）

1. **MVP（6–8 週）**：PWA（手機網頁可加到主畫面），LINE／Google／Apple 登入；自選股與持股（截圖／CSV 匯入）；每日「為什麼動了」推播（LINE 或 Web Push）；風險雷達；白話籌碼＋歷史統計；AI 問答（附來源、受合規過濾）；成績單頁；金流（綠界／藍新，避開 App 內購抽成）。
2. **第二階段**：原生 App（React Native／Flutter）、自然語言選股、法說會摘要、題材排行、MCP 伺服器。
3. **同步進行**：律師與 FinTechSpace 監理門診確認合規界線；授權資料源（TEJ／FinMind 付費方案）；內容獲客（YouTube／Threads 上的「誠實成績單」系列）。

## 主要來源

- 證交所開戶統計（中央社）：https://www.cna.com.tw/news/afe/202601020033.aspx
- 12 月新開戶年齡（工商時報）：https://www.ctee.com.tw/news/20260103700034-439901
- SITCA 認定經營證券投資顧問業務之問答集：https://www.sitca.org.tw/ROC/Legal/files/%E8%AA%8D%E5%AE%9A%E7%B6%93%E7%87%9F%E8%AD%89%E5%88%B8%E6%8A%95%E8%B3%87%E9%A1%A7%E5%95%8F%E6%A5%AD%E5%8B%99%E4%B9%8B%E5%95%8F%E7%AD%94%E9%9B%86.pdf
- 金管會監理門診 FAQ：https://www.fsc.gov.tw/websitedowndoc?file=chfsc/202210031521410.pdf&filedisplay=%E7%9B%A3%E7%90%86%E9%96%80%E8%A8%BAFAQ%E7%AC%AC%E5%85%AD%E7%89%88.pdf
- 判決案例：https://www.ettoday.net/news/20260618/3185434.htm 、https://news.ltn.com.tw/news/society/breakingnews/5003978
- 投顧與網路業者合作三不：https://www.ctee.com.tw/news/20220719700155-430298
- 財報狗方案：https://statementdog.com/pricing
- CMoney 籌碼K線：https://www.cmoney.tw/app/landing_page/chipk/
- 國泰 AI 庫存管家：https://www.ctee.com.tw/news/20260722700913-430201
- 富邦 AI PRO：https://apps.apple.com/tw/app/%E5%AF%8C%E9%82%A6-ai-pro/id6751573234
- Robinhood 2026 Q1：https://investors.robinhood.com/news-releases/news-release-details/robinhood-reports-first-quarter-2026-results
- Toss Real-Time Issues：https://en.sedaily.com/finance/2026/03/25/toss-securities-launches-ai-powered-real-time-issues-service
- Simply Wall St 合規做法：https://support.simplywall.st/hc/en-us/articles/360000338455-Does-Simply-Wall-St-provides-financial-advice-or-stock-recommendations
- Danelfin：https://danelfin.com/how-it-works
- RevenueCat 2026：https://www.revenuecat.com/blog/growth/subscription-app-trends-benchmarks-2026
- 投信投顧公會基金投資行為調查新聞稿（2026/4/30）：https://www.sitca.org.tw/CWEB/1150430%E6%8A%95%E4%BF%A1%E6%8A%95%E9%A1%A7%E5%85%AC%E6%9C%83%E6%96%B0%E8%81%9E%E7%A8%BF.pdf
