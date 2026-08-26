# Data Source Audit

This project uses real providers only. If a provider fails or a series is unavailable, ingestion records the failure and continues with remaining sources; missing data must lower model confidence in later phases.

| Variable | Provider | API URL / method | Frequency | Timezone | Cost | Auth | Historical coverage | Fallback | Limitations |
|---|---|---|---|---|---|---|---|---|---|
| MARKET_USDTWD OHLC | Yahoo Finance | `yfinance` symbol `USDTWD=X` | Daily/business day | Source timestamps normalized to UTC | Free | None | Usually multi-year, varies by Yahoo | CBC official closing rate | Yahoo is not an official exchange/central bank source; use for OHLC convenience, validate gaps. |
| USD/TWD official close | Central Bank of Taiwan | `https://www.cbc.gov.tw/en/lp-700-2.html` HTML table | Daily | Taiwan local publication, stored UTC | Free | None | Page currently lists 3,000+ daily observations | Yahoo `USDTWD=X` close | Official page exposes close only, not OHLC. HTML structure can change. |
| BANK_USD_SELL_RATE | Bank of Taiwan | `https://rate.bot.com.tw/xrt?Lang=en-US` HTML table / CSV link | Intraday during bank quotation updates | Asia/Taipei display, stored UTC | Free | None | Current page; historical inquiry exists but needs separate implementation | Land Bank public rate page | BOT note says displayed info is reference only; direct programmatic access currently returns challenge validation in local tests, so it is not dependable without an official API/feed. |
| BANK_USD_SELL_RATE fallback | Land Bank of Taiwan | `https://rate.landbank.com.tw/en-US/Foreign?mid=69` HTML table | Intraday during bank quotation updates | Asia/Taipei display, stored UTC | Free | None | Current page; historical page exists separately | User-configured family bank later | Public webpage parsing; actual transaction price still follows bank channel quote. |
| DXY | Yahoo Finance | `DX-Y.NYB` | Daily | Exchange calendar, stored UTC | Free | None | Varies by Yahoo | FRED `DTWEXBGS` broad USD index | Yahoo is unofficial; ICE DXY official data may require paid licensing. |
| Broad USD index proxy | FRED | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS` | Daily | FRED dates, stored UTC | Free | None for CSV | 2006+ for broad dollar index | Yahoo DXY | It is not classic ICE DXY; it is a broad nominal USD index proxy. |
| US 2Y yield | FRED | `...fredgraph.csv?id=DGS2` | Daily | FRED dates, stored UTC | Free | None for CSV | 1976+ | US Treasury CSV provider later | Some holidays/missing values represented as `.`. |
| US 10Y yield | FRED | `...fredgraph.csv?id=DGS10` | Daily | FRED dates, stored UTC | Free | None for CSV | 1962+ | US Treasury CSV provider later | Same as above. |
| 2s10s spread | Derived | `DGS10 - DGS2` | Daily | UTC | Free | None | Depends on yield series | FRED `T10Y2Y` | Derived feature in Phase 2. |
| VIX | FRED | `...fredgraph.csv?id=VIXCLS` | Daily close | FRED dates, stored UTC | Free | None | 1990+ | Yahoo `^VIX` later | FRED notes CBOE copyright; use citation. |
| S&P 500 | FRED | `...fredgraph.csv?id=SP500` | Daily close | FRED dates, stored UTC | Free | None | 2012+ in FRED series | Yahoo `^GSPC` later | Close only from FRED. |
| NASDAQ Composite | FRED | `...fredgraph.csv?id=NASDAQCOM` | Daily close | FRED dates, stored UTC | Free | None | 1971+ | Yahoo `^IXIC` later | Close only from FRED. |
| USD/CNH | Yahoo Finance | `USDCNH=X` | Daily | Source timestamps normalized UTC | Free | None | Varies; current test returned latest row only | Paid FX API later | Non-official; CNH long history is unreliable through Yahoo and should be replaced before modeling. |
| USD/KRW | Yahoo Finance | `USDKRW=X` | Daily | Source timestamps normalized UTC | Free | None | Varies | Paid FX API later | Non-official; may have gaps. |
| USD/JPY | Yahoo Finance | `USDJPY=X` | Daily | Source timestamps normalized UTC | Free | None | Varies | Paid FX API later | Non-official; may have gaps. |
| TWSE foreign investor net buy/sell | TWSE official JSON | `https://www.twse.com.tw/rwd/zh/fund/T86?date=YYYYMMDD&selectType=ALLBUT0999&response=json` | Daily after market | Asia/Taipei, stored UTC | Free | None | Endpoint supports historical date queries, practical coverage to be verified month-by-month | Paid TW market API | Returned unit is shares by security; Phase 1 stores aggregate shares. NT$ amount needs another TWSE table or computation. |
| TAIEX | TWSE official JSON | `https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST?date=YYYYMMDD&response=json` | Daily monthly file | Asia/Taipei, stored UTC | Free | None | Historical monthly queries available | Yahoo `^TWII` later | ROC calendar date format requires conversion. |
| TSMC 2330 | TWSE official JSON | `https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date=YYYYMMDD&stockNo=2330&response=json` | Daily monthly file | Asia/Taipei, stored UTC | Free | None | Historical monthly queries available | Yahoo `2330.TW` later | ROC calendar date format; official adjusted prices not included. |
| TSM ADR | Yahoo Finance | `TSM` | Daily | US exchange calendar, stored UTC | Free | None | Multi-year, varies | Paid market data later | Non-official. |
| Fed events / CPI / PCE / NFP calendar | Not implemented Phase 1 | BLS, BEA, Federal Reserve, Census, ISM calendars; optional paid calendar API | Event-based | America/New_York, stored UTC | Free and paid mix | Mostly none for official pages | Manual/event calendar provider | Official sources publish actuals, but consensus forecasts usually require paid or licensed source; no fake consensus data. |
| Fed expectations | Optional future provider | CME FedWatch or paid market data | Intraday/daily | America/New_York, stored UTC | Often licensed/limited | Varies | Disable feature | Do not fabricate implied policy path; use only if provider terms/API are acceptable. |
| Taiwan fundamentals | Future provider | Taiwan official statistics/CBC/MOEA/DGBAS | Monthly/quarterly | Asia/Taipei, stored UTC by release timestamp | Free | None | Varies by series | Manual official CSV | Must align by release timestamp, not reference month, to avoid look-ahead bias. |

## Phase 1 Decision

The first production-ready slice will ingest reliable free/public data for core daily monitoring:

- Official/primary: CBC USD/TWD close, Bank of Taiwan USD spot selling, TWSE foreign flow, TAIEX, 2330, FRED yields/VIX/S&P/Nasdaq/broad USD index.
- Non-official fallback/supplement: Yahoo Finance for USD/TWD OHLC, DXY, Asian FX pairs, and TSM ADR.
- Deferred: economic surprise data, Fed expectations, Taiwan macro fundamentals, and bank historical spot selling by arbitrary family bank.

Deferred items are intentionally not mocked.
