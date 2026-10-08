# Báo cáo đánh giá dự án `score` (Value Betting Engine)

- Ngày lập: 2026-10-08
- Phạm vi: toàn bộ repo tại commit `41d3a2e` + thư mục chưa commit `dashboard/`, `models/*.pkl`
- Phương pháp: đọc mã nguồn theo 4 mảng (model/quant, data, API/ops, UI), chạy test suite, kiểm tra trực tiếp DB `data/global_matches.db`, và đối chiếu lại các lỗi quan trọng trên code thật.

---

## 1. Tổng quan dự án

Dự án dự đoán kết quả bóng đá (1X2, Tài/Xỉu, Kèo châu Á, phạt góc) bằng mô hình Dixon–Coles kết hợp LightGBM, so sánh với tỷ lệ của nhà cái để tìm kèo có giá trị (EV ≥ 5%) và đặt cược theo phân số Kelly. Ban đầu dự án chỉ làm cho EPL, sau đó mở rộng ra nhiều giải: EPL, La Liga, UWCL, J.League Cup và Emperor's Cup.

| Thành phần | Vị trí | Vai trò |
|---|---|---|
| Streamlit UI | `app.py` (5.223 dòng) | Giao diện chính: scan kèo, Top-20, tab Nhật Bản, journal, backtest |
| Mô hình | `src/dixon_coles.py`, `src/ml_model.py`, `src/corner_model.py`, `src/models/*` | Dixon–Coles, LightGBM, Poisson phạt góc, ensemble v2, khử vig |
| Chiến lược / rủi ro | `src/recommender.py`, `src/strategy.py`, `src/risk/*` | EV, fair odds, Kelly, giới hạn exposure |
| Dữ liệu | `src/data_loader.py`, `src/global_db.py`, `src/fetchers/*`, `src/data/*` | Lấy dữ liệu từ football-data, Flashscore, Fotmob, ESPN; lưu SQLite |
| Backtest | `src/backtester.py` (legacy), `src/backtest/*` (v2), `src/validation/*` | Walk-forward, metrics, báo cáo |
| API | `src/api/*`, `run_api.py` | FastAPI v1/v2, xác thực bằng Telegram WebApp |
| Bot / Worker | `src/bot/*`, `src/workers/*`, `src/services/*`, `src/execution/*` | Gửi tín hiệu Telegram, tất toán cược, giám sát dữ liệu |
| Triển khai | `Dockerfile`, `docker-compose.yml`, `nginx/`, `deploy.sh`, `.github/workflows/daily_scan.yml` | VPS 24/7 và cron trên GitHub Actions |
| Dashboard mới | `dashboard/` (React + Vite + Tailwind, **chưa commit**) | UI thay thế, hiện mới chạy trên mock data |

Quy mô: khoảng 38.000 dòng Python, 161 file đã track, 7 commit trong khoảng 2026-09-22 → 2026-10-07. Dự án phát triển rất nhanh nên có nhiều lớp code trùng lặp (v1/v2, legacy/new).

### Trạng thái test

```
python -m pytest tests  →  1 failed, 299 passed (28s)
python -m pytest        →  dừng ở lỗi collection (trùng tên module)
```

1. `tests/test_sufficient_team_data.py::test_fujieda_rest_days_four_when_data_present` **FAIL**: `assert nan == 4.0`. Test này phụ thuộc DB thật, và phía sau còn có một lỗi logic thật (xem lỗi **C5**).
2. Chạy `pytest` từ thư mục gốc bị lỗi `import file mismatch`, vì `scripts/test_rotation_features.py` và `tests/test_rotation_features.py` trùng tên. Repo không có `pytest.ini` hay `testpaths`.
3. Dashboard: `npx tsc -b --noEmit` qua, không có lỗi type.

---

## 2. Các lỗi hiện tại

Mức độ được phân như sau:

- **Critical**: sai tiền, sai kết quả, hoặc lỗ hổng bảo mật trong production.
- **High**: sai số liệu đáng kể, hoặc có nguy cơ mất dữ liệu.
- **Medium**: sai lệch, khó vận hành.
- **Low**: chất lượng code, vệ sinh repo.

### 2.1 Critical

**C1. Đội mới hoặc đội ít dữ liệu bị mô hình coi là đội mạnh.**
- Vị trí: `src/dixon_coles.py:47-48`.
- Prior "weak tier" đặt α = 0.4 và δ = 0.4. Trên mô hình đã fit cho EPL, α trung bình khoảng 0 (cao nhất 0.49) và δ trung bình −0.13 (cao nhất 0.45, là Arsenal). Như vậy đội mới được gán hàng công gần top và hàng thủ ngang Arsenal.
- Đo thực tế: một đội giả lập mới, đá sân nhà gặp Coventry, nhận P(H) = 87%.
- Lỗi này ảnh hưởng mọi đội thăng hạng, đội lần đầu dự UWCL hoặc cúp, nên sinh EV giả rất lớn.
- Giới hạn stake bảo vệ cho đội mới cũng không có tác dụng: `NEW_TEAM_KELLY_CAP == MAX_STAKE_PCT` (`src/models/league_fit.py:38`).

**C2. Backtest v2 thực chất không đánh giá mô hình nào.**
- Vị trí: `src/backtest/run_backtest.py`.
- Xác suất được hardcode 0.40/0.30/0.30 (L362-363).
- Khi thiếu odds, code dùng odds giả 2.10/3.40/3.60. Closing odds được copy từ odds lúc vào lệnh, nên CLV luôn bằng 0 (L349-350).
- Các split walk-forward được tính rồi bỏ đi (`_ = splits`, L445).
- Khi DB rỗng, code âm thầm chuyển sang dữ liệu synthetic nhưng vẫn gắn nhãn `mode="live"`.
- Hệ quả: mọi con số ROI hay CLV từ backtest v2 đều không có ý nghĩa.

**C3. API và bot dùng odds cũ không giới hạn thời gian.**
- Vị trí: `src/api/services/value_bets.py:176-183`.
- `_load_fixtures` trả về cache `upcoming_fixtures` bất cứ khi nào cache khác rỗng. Không có `max_age`, và chỉ gọi mạng khi cache rỗng.
- Trong Docker không có process nào ghi lại cache này. Vì vậy odds cũ nhiều ngày vẫn được chấm như tín hiệu live.
- Khi mọi trận trong cache đã đá xong, cache vẫn không rỗng, nên bot im lặng không gửi gì.
- Cửa sổ scan bắt đầu từ 0h giờ VN, nên còn liệt kê cả trận đã bắt đầu.

**C4. Xác thực API bị tắt mặc định khi deploy.**
- Vị trí: `.env.example:17` đặt `TELEGRAM_AUTH_DISABLED=true`.
- `DEPLOY.md` hướng dẫn `cp .env.example .env`, nên giá trị mặc định `false` trong `docker-compose.yml:30` không bao giờ được dùng.
- Hệ quả: mọi endpoint `/api/v1` và `/api/v2`, kể cả endpoint ghi dữ liệu, đều công khai qua nginx.

**C5. `get_rest_days` không có tính point-in-time.**
- Vị trí: `src/features.py:448-520`.
- Hàm lấy `teams.last_match_date` hoặc `MAX(match_date)` của toàn bộ lịch sử, không lọc `< upcoming_date`.
- Khi đã có trận sau ngày cần tính, `rest_days` âm và hàm trả về `NaN`. Ví dụ: Fujieda có trận ngày 23, 26 và 29/09, nên tính cho ngày 23/09 ra NaN. Đây chính là test đang fail.
- Hệ quả:
  - Feature rest-days trong backtest hoặc khi tính lại lịch sử bị sai, hoặc thành NaN.
  - `teams.last_match_date` (26/09) lệch với bảng `matches` (29/09), tức là cache bị stale.

### 2.2 High

**H1. EV, fair odds và Kelly bỏ qua tiền hoàn khi kèo hòa (push) hoặc thua nửa.**
- Vị trí: `src/dixon_coles.py:199-206`, `src/recommender.py:296,340`, `src/strategy.py:142-149`.
- Code tính xác suất thắng là `win + 0.5·half_win` rồi áp `EV = P·odds − 1`.
- Ví dụ Over 2.0 với odds 2.0: hệ thống tính EV = −12,5%, trong khi EV thật là +13,8%. Lỗi làm sai lệch cả kèo châu Á và Tài/Xỉu ở line nguyên hoặc line ¼.

**H2. Kèo chấp phạt góc bị đảo dấu.**
- Vị trí: `src/corner_model.py:445-453`.
- Code tính P(HC−AC > h) thay vì P(HC−AC+h > 0). Ở line nguyên, kết quả hòa bị tính là Away thắng.
- O/U phạt góc dùng Poisson, trong khi số phạt góc biến động lớn hơn Poisson (overdispersion), nên xác suất ở hai đuôi quá tự tin.

**H3. Telegram gửi kèo phạt góc với odds giả.**
- Vị trí: `app.py:2464-2474`.
- Khi không có odds phạt góc, code tự gán 1.90/1.90 cho line 10.5, rồi gửi kết quả như một value bet thật.

**H4. Tất toán có thể dùng nhầm trận.**
- Vị trí: `src/journal.py:993-1006`.
- Khi không tìm thấy kết quả đúng ngày, hàm fallback sang bất kỳ trận nào có cùng cặp đội (`iloc[-1]`).
- Ví dụ: trận bị hoãn hoặc chưa được nạp sẽ được tất toán bằng tỷ số mùa trước.

**H5. Có race condition khi tất toán và khi ghi journal.**
- `settle_live_bet` chạy `UPDATE … WHERE id=?` mà không có `AND status='PENDING'` (`src/journal.py:770-797`). Có 3 nơi cùng tất toán: settlement_worker, `scripts/recheck_and_retrain.py` và nút trong Streamlit.
- `add_to_journal` kiểm tra rồi mới insert trên hai kết nối khác nhau, và index dedup không phải UNIQUE (`src/journal.py:164-168, 313-340`). Hệ quả là cùng một kèo có thể bị ghi và đặt hai lần.

**H6. SQLite bị nhiều process ghi đồng thời mà không có cấu hình phù hợp.**
- uvicorn chạy `--workers 2`, cùng với data_monitor, settlement_worker, telegram_bot, Streamlit và các script.
- Không chỗ nào bật WAL hay `busy_timeout`.
- `connect_global_db` chạy `init_schema` (executescript + commit) ở mỗi lần kết nối, kể cả khi chỉ đọc.
- `ensure_live_bets_table` chạy `DROP VIEW` và `CREATE VIEW` ở mỗi lần gọi.
- Hệ quả: lỗi "database is locked" khi tải cao.

**H7. Mất dữ liệu khi fetch thất bại hoặc ghi đồng thời.**
- Fetch Flashscore lỗi thì nuốt exception và trả về DataFrame rỗng. Sau đó `save_upcoming_to_db` chạy `DELETE` toàn bộ fixtures của giải và ghi timestamp "fresh" (`src/fetchers/flashscore_league.py:838-847, 1026`; `src/global_db.py:362-370`).
- `save_matches_to_db` đọc toàn bảng, nối thêm dữ liệu rồi `to_sql(if_exists="replace")` (`src/data_loader.py:640-660`). Hai lần refresh chạy đồng thời sẽ làm mất dữ liệu.

**H8. Parser odds 1X2 phụ thuộc thứ tự phần tử.**
- Vị trí: `src/data_loader.py:1667-1679`.
- Parser giả định thứ tự home, away, draw và không kiểm tra `eventParticipantId`. Kèo châu Á cũng map theo vị trí.
- Nếu Flashscore đổi thứ tự, giá home và giá hòa bị hoán đổi mà không báo lỗi, sinh EV giả rất lớn.

**H9. API có xác thực nhưng không phân quyền.**
- Bất kỳ user Telegram nào cũng gọi được:
  - `POST /api/v2/bets/place` (`src/api/routes_v2.py:407-436`). Endpoint nhận xác suất, bankroll và `as_of_time` do client gửi lên, nên ghi được cược giả hoặc cược lùi ngày vào phân tích PnL/CLV.
  - `POST /api/v1/teams/import-url` (`src/api/routes.py:77-96`), có thể đổi hash Flashscore của đội bất kỳ.

**H10. Lộ token bot trong log.**
- `src/bot/telegram_bot.py:60` đặt log ở mức INFO, nên httpx ghi URL `https://api.telegram.org/bot<TOKEN>/...` vào log container.
- `src/notifier.py:83-86` đẩy lỗi `requests` có chứa URL kèm token lên Streamlit.

**H11. Cache SQLite trong CI bị đóng băng.**
- Vị trí: `.github/workflows/daily_scan.yml:65-72`.
- Workflow dùng key cố định `epl-sqlite-Linux-v1`, mà cache của Actions không ghi đè được. Mọi lần chạy đều khôi phục DB từ lần chạy đầu tiên.
- Hệ quả: alert bị gửi trùng, và journal cùng CLV mới bị mất.

**H12. Backtest legacy không phản ánh code chạy live.**
- Calibrator isotonic fit trên xác suất 1X2 nhưng lại áp dụng cho O/U và AH (`src/backtester.py:835-836`). Đường live không calibrate.
- Backtest bỏ qua đội lạ, trong khi live vẫn đặt cược cho đội lạ.
- "CLV" thực chất so B365 với giá trung bình cùng thời điểm (`src/backtester.py:357-385`), không phải closing odds. Không chỗ nào đọc `PSCH`/`B365CH`.

**H13. Streamlit, API và bot cho ra EV khác nhau với cùng một trận.**
- Streamlit fit lại mô hình ngay trong process, theo giá trị các slider. API và bot thì load `models/*_latest.pkl`.
- Trong `app.py` có 3 đường tính EV khác nhau:
  - Scan Telegram (L2436): hardcode B365, không có ML.
  - Top-20 (L1838).
  - Tab Nhật Bản (L2884).

**H14. Guard chống chạy thread nền trùng lặp trong Streamlit không có tác dụng.**
- Vị trí: `app.py:1634-1635, 1982-1983`.
- `_FX_BG_LOCK` và `_FX_BG_RUNNING` là biến global của script chính, nên bị tạo lại ở mỗi lần rerun.
- Khi cache cũ hơn 15 phút, mỗi lần click lại sinh thêm một thread scrape Flashscore.

### 2.3 Medium

| # | Lỗi | Vị trí |
|---|---|---|
| M1 | `LeagueWeightDixonColesProxy`: 1X2 dùng λ/μ có trọng số giải, còn O/U và AH dùng ma trận chưa có trọng số. Giải quốc nội bị nhân trọng số của chính nó, nên lệch về phía Xỉu và hòa | `src/models/league_fit.py:200-284` |
| M2 | Ensemble v2: thiếu một thành phần thì thay bằng phân phối đều 1/3 với nguyên trọng số, làm phồng xác suất kèo dưới và hòa | `src/models/ensemble_v2.py:35-36, 234-253` |
| M3 | LightGBM: `CalibratedClassifierCV(cv=3)` không chia fold theo thời gian, nên bị leakage. Feature rotation có trong train nhưng không có khi predict | `src/ml_model.py:250, 529-537, 609-615` |
| M4 | Features: thiếu số liệu thì ghi 0.0, và impute bằng mean của toàn frame (look-ahead) | `src/features.py:241-245, 297` |
| M5 | Cho phép nhiều kèo tương quan trên cùng trận, ví dụ 1X2 Home và AH −0.5 Home | `src/strategy.py:717` |
| M6 | Chính sách stake mâu thuẫn. README ghi "Quarter-Kelly", config dùng 0.10 với trần `MAX_KELLY_FRACTION=0.10`. Match card in "Kelly (25%)". API mặc định `min_ev=3.0`, trái với quy tắc 5%. `min_ev_pct=1.0` bị hiểu thành 100% | `src/config.py:14-18`, `src/recommender.py:999`, `src/api/routes_v2.py:335`, `src/api/services/value_bets.py:300` |
| M7 | Hai engine tất toán với nhãn khác nhau (WIN/LOSS và WON/LOST). v2 không tất toán phạt góc, không void trận hoãn, và có thể tất toán bằng tỷ số đang đá | `src/services/settlement_service.py`, `src/execution/settlement_v2.py:253` |
| M8 | Alert bị đánh dấu "đã gửi" trước khi gửi, nên gửi lỗi thì mất luôn. Không có throttle khi gặp 429 | `src/api/services/scheduler.py:290-300` |
| M9 | Migration schema có thể xóa toàn bộ `matches` và `teams` khi không đọc được `schema_version` (mất dòng meta hoặc lỗi đọc) | `src/global_db.py:803-819` |
| M10 | Odds từ bookmaker bất kỳ đều ghi vào cột `B365H/D/A`, trộn lẫn với dữ liệu lịch sử Bet365 thật | `src/data_loader.py:1652-1664, 1826` |
| M11 | Store v2 (`quant_engine_v2.db`): không có unique key cho trận. Giờ UK được lưu như UTC (lệch 1 giờ vào mùa hè). Unique constraint trên cột nullable không chặn được trùng lặp | `src/data/bulk_importer.py:331-336, 462-492`, `src/db/schema_v2.py:123-128` |
| M12 | `run_pipeline` bắt lỗi flush nhưng không rollback, nên lần dùng session tiếp theo lỗi `PendingRollbackError` | `src/pipeline/run_pipeline.py:266-272` |
| M13 | Không có rate limit. Cold path của `/api/v1/value-bets` fit Dixon–Coles mà không có lock. `_MODEL_CACHE` không bao giờ invalidate, nên phải restart mới nhận model mới | `src/api/services/value_bets.py:135-175` |
| M14 | Streamlit cache: `force_refresh` nằm trong cache key nên mỗi lần refresh train 2 lần. `_cache_ver` có tiền tố `_` nên không được hash. Model cache không giới hạn số entry. `_clear_caches()` xóa cache của mọi session | `app.py:1577, 1717, 1755, 2024, 2282-2294` |
| M15 | Có 40 khối `except Exception` trong `app.py`. Fit ML lỗi thì âm thầm chỉ còn Dixon–Coles, giải fit lỗi thì bị bỏ khỏi kết quả scan mà không báo | `app.py:1628, 1836, 1898, 1906` |
| M16 | HTTP helpers không retry, không bọc `TimeoutError` hay `JSONDecodeError`. `FLASHSCORE_FSIGN` bị hardcode | `src/data_loader.py:47, 325-364` |
| M17 | `requirements.txt` ghi `streamlit>=1.30`, nhưng code dùng `st.pills(required=True)` (cần bản ≥ 1.40). Mọi dependency khác cũng chỉ có `>=`, không có lockfile | `requirements.txt`, `app.py:2602` |
| M18 | Dashboard chỉ dùng mock data (`dashboard/src/App.tsx:7,54`). Nhãn "Realtime Active" và "5 phút trước" là hardcode. Danh sách giải có UCL và SERIE_A (không tồn tại) nhưng thiếu EMPERORS_CUP | `dashboard/src/*` |

### 2.4 Low / vệ sinh repo

- **Chưa ignore:** repo không có `.gitignore` cho `dashboard/node_modules` (111 MB, khoảng 7.200 file), `dist/`, `tsconfig.tsbuildinfo` và `models/*.pkl`. Chạy `git add .` lúc này sẽ commit tất cả. Ngoài ra `.dockerignore` không loại `data/`, `models`, `node_modules`, nên DB journal và chat ID bị đóng vào image.
- **Test giả làm script:**
  - `test_backtest.py` và `test_recommender.py` ở thư mục gốc không có hàm test.
  - `test_backtest.py` chạy `replace_history=True` trên DB thật.
  - `scripts/test_*.py` mặc định ghi vào DB production.
- **Code trùng hoặc code chết:**
  - Hai backtester, hai ensemble, ba bản tất toán line ¼, hai notifier Telegram.
  - `src/data/odds_engine.py` chỉ được test dùng tới.
  - Package `src/pipeline/` che module `src/pipeline.py` (phải load qua `importlib`).
- **Docker / nginx:** không có healthcheck. nginx không redirect HTTP sang HTTPS, không có security headers. Lỗi exception trả thẳng cho client. CORS đặt `*`.
- **Rò kết nối:** `with sqlite3.connect()` không đóng kết nối (gây vấn đề trên Windows). `src/notifier.py` không escape HTML tên đội.
- **Giả định EPL:** `_season_label` giả định mùa giải tháng 8 đến tháng 5, nên tách sai mùa J-League. Backtest gán mọi trận vào giải "EPL". `predict_next_game.py` chỉ chạy cho EPL.
- **Tài liệu lỗi thời:** `README.md`, `.cursorrules`, `PORTFOLIO.md` vẫn mô tả dự án chỉ có EPL và Streamlit, không nhắc tới đa giải, API, worker hay dashboard.

---

## 3. Đề xuất hướng phát triển

### Giai đoạn 0: chặn rủi ro ngay (1–3 ngày)

Mục tiêu: không mất tiền vì tín hiệu sai, không lộ hệ thống.

1. Đặt `TELEGRAM_AUTH_DISABLED=false` trong `.env.example`. Thêm `APP_ENV=prod` để app từ chối khởi động nếu auth bị tắt. Thêm allowlist `ADMIN_TELEGRAM_IDS` cho các endpoint ghi.
2. Sửa prior cho đội mới thành giá trị tương đối với mean đã fit, ví dụ `α = mean(α) − 0.3`, `δ = mean(δ) − 0.3`. Thêm test kiểm tra đội mới yếu hơn trung bình giải.
3. Thêm `max_age` cho cache fixtures và odds. Lọc `kickoff > now`. Thêm worker ingest định kỳ trong Docker. Fetch lỗi thì không được xóa cache.
4. Không gửi alert khi dùng odds giả (phạt góc 1.90) hoặc khi có fallback.
5. Tắt log INFO của httpx để không lộ token.
6. Sửa `get_rest_days`: lọc `match_date < upcoming_date`, và làm test này hermetic (dùng DB tạm).
7. Bổ sung `.gitignore` cho `dashboard/node_modules`, `dist`, `*.tsbuildinfo`, `models/*.pkl`. Thêm `pytest.ini` với `testpaths = tests`. Đổi tên hoặc di chuyển các `scripts/test_*.py`.

### Giai đoạn 1: đúng về toán và tiền (1–2 tuần)

1. **Một module pricing duy nhất:** EV tính chính xác theo settlement (`Σ mass × payoff`), fair odds = `1 + loss/win` cho line ¼ và line nguyên, Kelly theo cùng phân phối đó. Gộp ba bản tất toán line ¼ làm một.
2. **Một module settlement duy nhất:** idempotent (`UPDATE … WHERE status='PENDING'` có kiểm tra rowcount), dùng chung một bộ nhãn status, void khi trận hoãn, chỉ tất toán khi trạng thái là FINISHED. Bỏ fallback "trùng cặp đội". Thêm unique index cho kèo PENDING.
3. Sửa dấu kèo chấp phạt góc. Chuyển O/U phạt góc sang Negative Binomial.
4. Sửa proxy trọng số giải để mọi thị trường dùng chung λ/μ. Không áp trọng số cho giải quốc nội.
5. Ensemble: khi thiếu thành phần thì chia lại trọng số cho các thành phần còn lại, không thay bằng phân phối đều.
6. Thống nhất chính sách stake (EV ≥ 5%, phân số Kelly, trần 1%) trong `src/config.py`, expose qua API, hiển thị nhất quán. Quyết định chính thức dùng 0.10 hay 0.25 Kelly và cập nhật tài liệu theo quyết định đó.
7. Chặn đặt nhiều kèo tương quan trên cùng một trận, hoặc áp chiết khấu tương quan.

### Giai đoạn 2: dữ liệu và hạ tầng (2–4 tuần)

1. **Chuyển sang Postgres** (`DATABASE_URL` đã được hỗ trợ trong `src/db/session.py`). Chọn một nguồn sự thật duy nhất: gộp `global_matches.db` và `quant_engine_v2.db`. Nếu tạm giữ SQLite thì bật WAL và `busy_timeout`, chỉ để một process ghi, và chỉ chạy schema khi deploy.
2. **Lớp ingest riêng:** HTTP client dùng chung có retry, backoff và circuit breaker. Parse odds theo `eventParticipantId`. Thêm cột `bookmaker` thay cho việc ghi đè `B365*`. Thêm nguồn odds dự phòng (The Odds API hoặc Betfair).
3. **Theo dõi chất lượng dữ liệu:** tuổi odds, số fixture mỗi giải mỗi ngày, overround 1X2 bất thường (phát hiện giá bị hoán đổi), số tên đội chưa map, số trận trùng.
4. **Entity resolution tập trung:** bảng `team_aliases` trong DB, dùng chung cho mọi store, kèm hàng đợi review cho tên chưa map.
5. **Thời gian:** lưu mọi timestamp ở dạng UTC có timezone, chỉ đổi sang JST hoặc giờ VN khi hiển thị. Thêm lint rule cấm datetime naive.
6. **Vận hành:**
   - Thêm `/health/ready` và healthcheck trong compose. Worker ghi heartbeat.
   - Alert đi qua outbox, chỉ đánh dấu SENT sau khi gửi thành công.
   - Thêm `limit_req` trong nginx, redirect HTTPS, security headers.
   - Pin dependency bằng `uv` hoặc pip-tools, chạy `pip-audit`, bật Dependabot.
   - Bỏ cron CI scan, hoặc cho nó dùng chung DB với VPS.
7. **Model artifacts:** train offline, version theo hash, API tự reload khi file model thay đổi, lưu artifact ngoài git.

### Giai đoạn 3: chất lượng mô hình và đo edge thật (liên tục)

1. **Backtest trung thực:**
   - Refit Dixon–Coles và LightGBM trong từng split walk-forward.
   - Báo lỗi rõ ràng thay vì dùng dữ liệu synthetic.
   - Dùng chung calibrator giữa backtest và live, calibrate riêng từng thị trường.
   - Calibrate theo fold thời gian.
   - Báo cáo kết quả theo từng giải.
2. **CLV là thước đo chính:** lưu closing price (Pinnacle hoặc `PSCH`) cho mỗi cược, dùng một công thức CLV duy nhất (log, sau khi khử vig), báo cáo CLV theo thị trường và theo giải.
3. **Cải thiện Dixon–Coles:**
   - Shrink sức mạnh đội về mean (ridge hoặc prior Bayes) thay cho ngưỡng cứng 5 trận.
   - Fit ρ và ξ riêng cho từng giải theo log-loss out-of-sample.
   - Dùng gradient giải tích để tăng tốc fit.
   - Cache ma trận tỷ số thay vì tính lại cho từng thị trường.
4. **Bổ sung test cho phần lõi:** τ của Dixon–Coles, xác suất cộng lại bằng 1, bảng settlement, EV khi push, kèo chấp phạt góc, leakage của LightGBM, ensemble khi thiếu thành phần.
5. **Mùa giải theo lịch từng giải:** J-League và các cúp chạy theo năm dương lịch.

### Giai đoạn 4: sản phẩm và UI (song song với giai đoạn 2–3)

1. **Một nguồn sự thật cho số liệu:** mọi UI (Streamlit, WebApp, dashboard, bot) đều gọi `/api/v2` với cùng model và pipeline. Bỏ việc fit lại mô hình ngay trong Streamlit.
2. **Nối dashboard React với API:** client có type, biến `VITE_API_BASE`, Vite dev proxy, polling bằng React Query, thời điểm "last updated" lấy từ API. Chọn cách xác thực cho web (Telegram Login hoặc token). Sau đó dùng dashboard thay cho `src/webapp`, với Node build stage trong Dockerfile.
3. **Danh sách giải lấy từ `config/leagues.json` qua API.** Top-N scan mọi giải đang bật.
4. **Hiển thị độ tin cậy trên mỗi card:** nguồn odds, tuổi odds, cờ fallback, cờ "đội ít dữ liệu", cờ "ML bị tắt".
5. **Thêm màn hình Journal, CLV, PnL và tất toán** vào dashboard. Theo dõi kết quả là giá trị cốt lõi của sản phẩm.
6. **Tách nhỏ `app.py` trong thời gian chuyển đổi:** dùng Streamlit multipage (`pages/`) và chuyển logic scan và cache vào `src/services`.
7. **Viết lại README** cho đa giải, API, worker và dashboard. Cập nhật `.cursorrules` và `DEPLOY.md`.

---

## 4. Thứ tự ưu tiên tóm tắt

| Ưu tiên | Việc | Lý do |
|---|---|---|
| 1 | C4 auth, H9 phân quyền, H10 token | Bảo mật production |
| 2 | C1 prior đội mới, H1 EV khi push, H2 và H3 phạt góc | Đang sinh EV giả, đặt sai tiền |
| 3 | C3 odds cũ, H7 xóa cache khi fetch lỗi, H8 parser odds | Tín hiệu dựa trên dữ liệu sai |
| 4 | H4 và H5 tất toán, H6 khóa SQLite | Sai PnL và CLV, trùng cược |
| 5 | C2 và H12 backtest, C5 rest days | Không đo được edge thật |
| 6 | Gộp code trùng, Postgres, dashboard nối API | Giảm nợ kỹ thuật, mở rộng sản phẩm |

---

## 5. Trạng thái sửa lỗi (cập nhật 2026-10-08)

Đã sửa trên working tree, **chưa commit**: 80 file sửa, khoảng 6.100 dòng thêm và 1.500 dòng xoá, cùng 13 file test mới.

- `python -m pytest` từ thư mục gốc: **465 passed, 0 failed**. Trước khi sửa là 299 passed, 1 failed và 1 lỗi collection.
- `npx tsc -b --noEmit` (dashboard): qua.
- `compileall` với `src`, `scripts`, `app.py`: qua. API khởi động bình thường với `APP_ENV=test`, và từ chối khởi động khi `APP_ENV=prod` mà auth bị tắt.
- DB thật trong `data/` không mất dữ liệu: `global_matches.db` vẫn có 2467 trận và 214 đội. DB đã được chuyển sang chế độ WAL và có thêm cột `flashscore_event_id`.

### 5.1 Đã sửa

| Lỗi | Kết quả |
|---|---|
| C1 prior đội mới | Prior tính tương đối: mean(α) − 0.3 và mean(δ) − 0.3. `NEW_TEAM_KELLY_CAP` = 0,5% |
| C2 backtest v2 | Mỗi split walk-forward fit Dixon–Coles riêng. Không còn odds giả hay xác suất hardcode. CLV tính theo closing odds thật. DB rỗng thì báo lỗi `BacktestDataError`; dữ liệu synthetic chỉ chạy khi có `--synthetic` |
| C3 odds cũ | API lọc các trận có kickoff ≤ now và kiểm tra tuổi cache (`FIXTURES_MAX_AGE_MINUTES`). Thêm worker `fixtures_worker` trong compose. Scheduler không gửi tín hiệu khi odds đã cũ |
| C4 auth | `.env.example` đặt `TELEGRAM_AUTH_DISABLED=false` và `APP_ENV=prod`. App không khởi động nếu tắt auth trong môi trường prod |
| C5 rest days | Tính point-in-time (`match_date < upcoming`). Test chạy trên DB tạm, không còn phụ thuộc DB thật |
| H1 EV khi hoà tiền | EV, fair odds và Kelly tính chính xác theo bảng settlement (cả push và thua nửa) |
| H2 phạt góc | Sửa dấu kèo chấp, thêm khối lượng push. O/U dùng Negative Binomial khi dữ liệu có overdispersion |
| H3 odds giả | Không gửi kèo phạt góc khi thiếu odds thật, ở cả `app.py` và `scripts/auto_scanner.py` |
| H4–H5 tất toán và journal | Chỉ chấp nhận kết quả trong ±1 ngày. `UPDATE … WHERE status='PENDING'`. Insert trong `BEGIN IMMEDIATE`, kèm UNIQUE partial index |
| H6 SQLite | Bật WAL và `busy_timeout`. Schema chỉ khởi tạo một lần mỗi process. Đóng kết nối đầy đủ |
| H7 mất cache | Fetch lỗi hoặc frame rỗng thì không ghi đè. `save_matches_to_db` chuyển sang upsert |
| H8 parser odds | Map selection theo `eventParticipantId`. Kiểm tra overround nằm trong [1.0, 1.25] |
| H9 phân quyền | Thêm `ADMIN_TELEGRAM_IDS` cho endpoint ghi. `persist` mặc định là false. Giới hạn độ lệch của `as_of_time` |
| H10 token | Log httpx đặt mức WARNING. Che token trong thông báo lỗi. Escape HTML. Xử lý 429 |
| H11 cache CI | Key cache gắn `run_id` |
| H12 backtest legacy | Calibrate riêng từng thị trường. CLV dùng closing thật (PS/AvgC). Dùng chung chính sách Kelly với live |
| H13 EV lệch nhau | Scan Telegram giờ dùng chung đường `scan_top_value_bets` với Top-20 |
| H14 thread nền | Guard chuyển vào `st.cache_resource`. Chờ 5 phút trước khi thử lại sau khi lỗi |
| M1–M18 | Đã sửa toàn bộ (xem `git diff` và các file `tests/test_fix_*.py`) |
| Vệ sinh repo | Thêm `.gitignore` (node_modules, dist, models), `pytest.ini`. Đổi tên `scripts/test_*.py` thành `scripts/check_*.py`. Chuyển test giả ở thư mục gốc vào `scripts/`. Viết lại README |

### 5.2 Thay đổi hành vi cần biết

1. **Phải cấu hình khi deploy:**
   - Đặt `ADMIN_TELEGRAM_IDS` để dùng các endpoint ghi. Nếu để trống, mọi request ghi bị từ chối.
   - Chạy local thì đặt `APP_ENV=dev`.
2. **Biến môi trường mới:** `APP_ENV`, `ADMIN_TELEGRAM_IDS`, `ENABLE_DOCS`, `PLACE_AS_OF_MAX_SKEW_SECONDS`, `FIXTURES_MAX_AGE_MINUTES`, `FIXTURES_REFRESH_INTERVAL_MINUTES`, `FIXTURES_REFRESH_LEAGUES`, `FIXTURES_REFRESH_BACKOFF_SECONDS`, `MODEL_COLD_FIT_TTL_MINUTES`, `TELEGRAM_SEND_RATE`, `NGINX_FORCE_HTTPS`, `PUBLIC_BASE_URL`, `FLASHSCORE_FSIGN`.
3. **Thang đo metric thay đổi:**
   - `multiclass_brier` theo định nghĩa chuẩn: giá trị 1X2 tăng gấp 3 so với trước.
   - CLV chuyển sang `log(taken / closing đã khử vig)`.
   - `closing_ref_family` mặc định chuyển sang `"PS"`.
4. **Hành vi chọn kèo và quét:**
   - `select_value_bets` mặc định chỉ chọn 1 kèo mỗi trận.
   - Scan Telegram trong Streamlit chỉ quét hôm nay và ngày mai (giờ VN).
   - Kèo phạt góc gần như tắt hẳn, vì chưa có feed nào cung cấp odds phạt góc.
5. **Backtest:** `--dry-run` không còn tự dùng dữ liệu synthetic; phải thêm `--synthetic`.
6. **Journal:** `settle_live_bet` raise `RuntimeError` khi kèo đã được tất toán trước đó.
7. **Đổi tên script:** `scripts/test_*.py` thành `scripts/check_*.py`. Hai script này mặc định chạy dry-run; muốn ghi vào `data/` thì phải thêm `--write`.

### 5.3 Còn mở (chưa sửa hoặc ngoài phạm vi)

1. **Tab Nhật Bản chưa blend LightGBM**, nên vẫn khác Top-20 ở phần ML. Phần Kelly đã được giới hạn qua recommender.
2. **Kèo Corners trong `settlement_v2` chưa tất toán được**, vì `CanonicalMatch` chưa có cột `home_corners`/`away_corners`. Thêm cột cần migration cho DB đang có.
3. **Unique index cho alias** (khi không có source id) chỉ áp dụng cho DB v2 mới. DB đã tồn tại vẫn dựa vào bước resolver kiểm tra trước khi insert.
4. **Code đọc `B365H` cho fixture sắp đá** (`app.py`, `recommender`, `scanner`, `japan_schedule`, `journal`) nên chuyển sang `OddsH/D/A` + `OddsBookmaker`. Hiện vẫn chạy đúng, vì `B365*` vẫn được ghi để tương thích ngược.
5. **Chưa chạy `nginx -t`**, vì Docker daemon không chạy trên máy này.
6. **Hạng mục kiến trúc trong mục 3 chưa làm:**
   - Chuyển sang Postgres.
   - Gộp hai backtester, hai ensemble, ba bản settlement.
   - Nối dashboard với `/api/v2`.
   - Thêm lockfile cho dependency.
