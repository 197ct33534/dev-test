# ⚽ Premier League Value Betting & Corner Recommender System

> **Project Portfolio & Technical Presentation**  
> Hệ thống dự đoán xác suất bóng đá Ngoại hạng Anh, phát hiện Value Bet và quản trị vốn bằng toán thống kê.

---

## 📑 Mục lục

1. [🎯 Mục tiêu Dự án](#1--mục-tiêu-dự-án-project-goal--overview)
2. [🛠️ Kiến trúc Công nghệ & Dữ liệu](#2-️-kiến-trúc-công-nghệ--dữ-liệu-tech-stack--architecture)
3. [⚡ Các Chức năng Cốt lõi](#3--các-chức-năng-cốt-lõi-của-hệ-thống-core-features)
4. [📊 Điểm Nổi Bật Kỹ Thuật](#4--điểm-nổi-bật-kỹ-thuật-technical-highlights--edge)
5. [🚀 Cách chạy nhanh](#5--cách-chạy-nhanh)
6. [📎 Phụ lục Công thức](#6--phụ-lục-công-thức)

---

## 1. 🎯 Mục tiêu Dự án (Project Goal & Overview)

| | |
|---|---|
| **Tên dự án** | Premier League Value Betting & Corner Recommender System |
| **Lĩnh vực** | Sports Analytics · Statistical Modelling · Decision Support |
| **Đối tượng** | Người phân tích kèo / nghiên cứu mô hình bóng đá / portfolio kỹ thuật |

### Bài toán giải quyết

Nhà cái không bán “xác suất thật” — họ bán **giá** đã gắn biên lợi nhuận (overround). Người chơi thường quyết định bằng cảm xúc, tin đồn hoặc “cảm giác trận đấu”.

Dự án xây một **pipeline end-to-end** để:

1. **Ước lượng độc lập** xác suất các kết quả bóng đá bằng mô hình toán–thống kê **Dixon–Coles Poisson** (không phụ thuộc bảng kèo nhà cái).
2. **So sánh** xác suất mô hình \(P_{\text{model}}\) với tỷ lệ cược nhà cái \(\text{Odds}\) để tìm các cửa có **Giá trị Kỳ vọng dương**.
3. **Lọc nghiêm** chỉ khuyến nghị khi:

\[
\text{EV} = (P_{\text{model}} \times \text{Odds}) - 1 \;\ge\; 0.05 \quad (5\%)
\]

4. **Quản trị vốn** bằng **Fractional Kelly (25% Kelly)** — quy đổi lợi thế thống kê thành mức stake kỷ luật, tránh all-in cảm xúc.
5. **Bổ sung thị trường phạt góc** bằng Poisson GLM trên thống kê `HC` / `AC` nhiều mùa.

### Giá trị mang lại

| Góc nhìn | Lợi ích |
|----------|---------|
| 🔬 Khoa học dữ liệu | Ứng dụng tối ưu hóa hợp lý cực đại, time-decay, ma trận tỷ số đầy đủ |
| 💰 Ra quyết định | Tách “xác suất mô hình” khỏi “giá nhà cái” → bắt lệch kèo có kiểm chứng |
| 🧭 Kỷ luật vốn | Kelly phân đoạn giúp kiểm soát rủi ro chuỗi thua |
| 🖥️ Sản phẩm | Dashboard Streamlit realtime — không chỉ notebook nghiên cứu |

---

## 2. 🛠️ Kiến trúc Công nghệ & Dữ liệu (Tech Stack & Architecture)

### 2.1 Tech Stack

| Tầng | Công nghệ | Vai trò |
|------|-----------|---------|
| 🐍 Ngôn ngữ | **Python 3.10+** | Toàn bộ pipeline |
| 📊 Xử lý dữ liệu | **Pandas**, **NumPy** | CSV/API → bảng trận sạch, vector hóa xác suất |
| 🧮 Tối ưu & thống kê | **SciPy** (`optimize.minimize`), **Statsmodels** | Fit Dixon–Coles; Poisson GLM phạt góc |
| 🌐 UI | **Streamlit** | Dashboard tương tác tiếng Việt |
| 📈 Trực quan | **Plotly** | So sánh \(P_{\text{model}}\) vs \(P_{\text{implied}}\) |

### 2.2 Nguồn dữ liệu

```text
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│ football-data.co.uk │     │ Fotmob / Flashscore  │     │ ESPN Scoreboard API │
│  3 mùa E0 (CSV)     │     │  lịch trận sắp đá    │     │  odds DraftKings    │
│  FT + Corners + Odds│     │  (calendar sync)     │     │  (+ FD fixtures.csv)│
└─────────┬───────────┘     └──────────┬───────────┘     └──────────┬──────────┘
          │                            │                            │
          └──────────────┬─────────────┴────────────────────────────┘
                         ▼
              src/data_loader.py  →  DataFrame chuẩn hóa
                         ▼
         Dixon–Coles  ·  Recommender  ·  Corner Model  ·  Streamlit UI
```

| Feed | Nội dung chính |
|------|----------------|
| `mmz4281/{season}/E0.csv` | Kết quả Full-time, phạt góc (`HC`/`AC`), odds Bet365 / Avg / Max |
| Flashscore calendar *(via Fotmob API)* | Lịch Premier League còn lại trong mùa |
| ESPN `eng.1` scoreboard | Odds live 1X2 / O/U / Asian Handicap (không cần API key) |
| `fixtures.csv` | Bổ sung odds khi còn trong feed Fri/Tue của football-data |

> **Mặc định:** huấn luyện trên **3 mùa gần nhất** để cân bằng kích thước mẫu và tính hiện đại của phong độ.

### 2.3 Cấu trúc thư mục (Modular Architecture)

```text
score/
├── app.py                  # 🖥️  Streamlit UI — tách biệt khỏi logic toán
├── src/
│   ├── data_loader.py      # 📥  Tải / làm sạch / gộp lịch + odds API
│   ├── dixon_coles.py      # 📐  Fit model + ma trận tỷ số + 1X2/O–U/AH
│   ├── corner_model.py     # 🚩  Poisson GLM dự đoán phạt góc
│   ├── recommender.py      # 💎  Fair Odds · EV · Kelly · Value cards
│   └── __init__.py
├── requirements.txt
├── README.md               # Hướng dẫn chạy nhanh
└── PORTFOLIO.md            # Báo cáo / bài trình bày (file này)
```

**Nguyên tắc thiết kế:** UI (`app.py`) không chứa công thức tối ưu; mọi \(\lambda, \mu, \rho, \xi\), EV, Kelly nằm trong `src/` — dễ kiểm thử, tái sử dụng CLI và mở rộng API sau này.

---

## 3. ⚡ Các Chức năng Cốt lõi của Hệ thống (Core Features)

### Chức năng 1 — Pipeline Thu thập & Tiền xử lý Tự động  
📁 `src/data_loader.py`

| Bước | Mô tả |
|------|--------|
| ⬇️ **Download** | Tự động kéo CSV nhiều mùa từ `football-data.co.uk` |
| 🧹 **Clean** | Chuẩn hóa tên cột, parse ngày (day-first), ép kiểu số cho bàn thắng / góc / odds |
| 🚫 **NaN policy** | Loại bỏ trận thiếu trường bắt buộc (`FTHG`, `FTAG`, `FTR`); odds tùy chọn được giữ dạng thiếu để UI nhập tay |
| 🏷️ **Team normalize** | Map tên Fotmob/ESPN → chuẩn football-data (`Man United`, `Nott'm Forest`, …) |
| 📅 **Upcoming** | Gộp lịch sắp đá + odds ESPN (ưu tiên) + `fixtures.csv` (bổ sung) |

**Output:** một `DataFrame` sẵn sàng fit Dixon–Coles và quét value bet hàng loạt.

---

### Chức năng 2 — Mô hình Dự đoán Dixon–Coles  
📁 `src/dixon_coles.py`

Mô hình Poisson độc lập cổ điển giả định \(X \perp Y\), thường **thổi phồng** xác suất các tỷ số thấp. Dixon–Coles hiệu chỉnh bằng hệ số \(\tau(x,y;\rho)\).

#### 2.1 Tham số đội bóng

Với mỗi đội \(i\):

| Ký hiệu | Ý nghĩa |
|---------|---------|
| \(\alpha_i\) | Sức **tấn công** |
| \(\beta_i\) | Sức **phòng thủ** |
| \(\gamma\) | Hệ số **lợi thế sân nhà** (home advantage) |
| \(\rho\) | Tham số tương quan tỷ số thấp |
| \(\xi\) | **Time-decay** — trọng số giảm dần theo thời gian |

Cường độ bàn thắng kỳ vọng:

\[
\lambda = \alpha_{\text{home}} \cdot \beta_{\text{away}} \cdot \gamma, \qquad
\mu = \alpha_{\text{away}} \cdot \beta_{\text{home}}
\]

(\(\lambda, \mu\) chính là “xG mô hình” theo nghĩa bàn thắng kỳ vọng của từng đội.)

#### 2.2 Time-decay \(\xi\)

Trận càng cũ càng ít ảnh hưởng tới likelihood. Phong độ gần đây được ưu tiên — phù hợp bóng đá hiện đại (thay HLV, chuyển nhượng, form ngắn hạn).

#### 2.3 Hiệu chỉnh \(\tau(x,y)\)

Điều chỉnh xác suất các ô **0–0, 1–0, 0–1, 1–1** trong ma trận tỷ số, khắc phục thiên lệch Poisson thuần.

#### 2.4 Đầu ra dự đoán

- Ma trận xác suất tỷ số \(P(X=x, Y=y)\) đầy đủ (cắt ngưỡng bàn hợp lý).
- Biên xác suất suy ra từ ma trận:

| Thị trường | Phủ sóng |
|------------|----------|
| 🏛️ **1X2** | Home / Draw / Away |
| ⚽ **Tài–Xỉu** | Mọi mốc phổ biến: 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5… (kể cả **quarter lines**) |
| ⚖️ **Asian Handicap** | −1.5 … +1.5 theo bước 0.25 (settle win / half / push / lose) |

---

### Chức năng 3 — Động cơ Value Bet & Quản trị Vốn  
📁 `src/recommender.py`

```text
P_model  ──►  Fair Odds = 1 / P_model
                │
Odds_book ──────┼──►  EV = P_model × Odds_book − 1
                │
                └──►  nếu EV ≥ 5%  →  Fractional Kelly (f = 0.25)
                                    stake* = f × EV / (Odds − 1) × Bankroll
```

| Thành phần | Công thức / quy tắc |
|------------|---------------------|
| Fair Odds | \(1 / P_{\text{model}}\) |
| Expected Value | \(\text{EV} = (P_{\text{model}} \times \text{Odds}) - 1\) |
| Ngưỡng khuyến nghị | Chỉ đề xuất khi \(\text{EV} \ge 0.05\) |
| Vốn | **Quarter Kelly** (\(f=0.25\)) — giảm biến động so với Full Kelly |
| So sánh ngầm định | \(P_{\text{implied}} = 1 / \text{Odds}\) (chưa bỏ overround từng cửa) |

**ValueBet card** đóng gói: thị trường, lựa chọn, mốc, \(P_{\text{model}}\), fair odds, odds nhà cái, EV%, Kelly%, số tiền gợi ý theo bankroll.

---

### Chức năng 4 — Web Dashboard Real-time  
📁 `app.py`

| Tính năng UI | Chi tiết |
|--------------|----------|
| 🇻🇳 Giao diện tiếng Việt | Thuật ngữ kèo / EV / Kelly dễ trình bày demo |
| 📋 Chọn trận sắp đá | Selectbox + thẻ nhanh theo vòng gần nhất |
| 🔌 Odds từ API | ESPN/DraftKings (+ fallback football-data); cảnh báo khi phải nhập tay |
| 🔑 Dynamic widget keys | Key gắn `Home__Away` → đổi trận **nạp odds mới**, không dính `session_state` trận cũ |
| 📊 Plotly comparison | Cột \(P_{\text{model}}\) vs \(P_{\text{implied}}\) |
| 💎 Recommendation cards | Nổi bật khi \(\text{EV} \ge 5\%\) |
| 📦 Tab phụ | Quét hàng loạt · Lịch sử · Backtest · Phạt góc · Bảng tấn công/phòng thủ |

**Sidebar cấu hình:** số mùa, \(\xi\), ngưỡng EV tối thiểu, fraction Kelly, bankroll.

---

### Chức năng bổ sung — Corner Recommender  
📁 `src/corner_model.py`

- Hồi quy Poisson trên thống kê phạt góc sân nhà / sân khách qua nhiều mùa.
- Dự đoán \(\mathbb{E}[\text{corners}]\) và hỗ trợ phân tích thị trường góc trên dashboard.

---

## 4. 📊 Điểm Nổi Bật Kỹ Thuật (Technical Highlights & Edge)

### ✅ 1. Vượt Poisson truyền thống bằng Dixon–Coles

Poisson độc lập thường **overestimate** các tỷ số thấp. Hệ số \(\tau\) + ước lượng \(\rho\) đồng thời với \(\alpha, \beta, \gamma\) giúp phân phối tỷ số thực tế hơn — từ đó 1X2 / O–U / AH tin cậy hơn.

### ✅ 2. Time-decay \(\xi\) — “bộ nhớ có trọng số”

Không xem trận cách đây 3 năm ngang trọng số với vòng gần nhất. Đây là điểm khác biệt quan trọng so với bảng xếp hạng tấn công–phòng thủ tĩnh.

### ✅ 3. Quarter lines & Asian settlement đúng nghĩa

Không chỉ “trên/dưới 2.5”. Hệ thống settle **half-win / half-lose / push** cho mốc 0.25 / 0.75 — sát thực tế sàn châu Á.

### ✅ 4. Streamlit state & realtime odds

Bài toán kinh điển: widget giữ giá trị cũ khi đổi selectbox. Giải pháp **dynamic keys theo cặp đấu** + reload khi đổi pick → odds API luôn khớp trận đang phân tích.

### ✅ 5. Tách UI khỏi core toán

`src/` có thể unit-test / chạy CLI (`predict_upcoming_match`, quét fixtures) độc lập Streamlit — sẵn sàng mở rộng FastAPI hoặc batch job.

### ✅ 6. Kỷ luật đầu tư bằng toán, không bằng cảm xúc

Ngưỡng EV cố định + Kelly phân đoạn tạo **khung ra quyết định lặp lại được** — phù hợp tinh thần portfolio kỹ thuật và quản trị rủi ro.

---

## 5. 🚀 Cách chạy nhanh

```bash
cd d:\nghia\score
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`.

| Bước demo gợi ý | Hành động |
|-----------------|-----------|
| 1 | Sidebar: chọn 3 mùa, chỉnh \(\xi\), EV min = 5%, Kelly = 25% |
| 2 | Tab **Chi tiết kèo sắp tới** → chọn trận → kiểm tra nguồn odds API |
| 3 | Đọc thẻ Value / biểu đồ Model vs Nhà cái |
| 4 | Tab **Phạt góc** / **Sức mạnh đội** để bổ sung narrative |

---

## 6. 📎 Phụ lục Công thức

### Likelihood (ý tưởng)

Mỗi trận đóng góp log-likelihood của cặp \((x,y)\) dưới phân phối Dixon–Coles có trọng số thời gian \(e^{-\xi \cdot \Delta t}\). Tham số được ước lượng bằng tối ưu số (`scipy.optimize.minimize`).

### Fair Odds & EV

\[
\text{Fair Odds} = \frac{1}{P_{\text{model}}}, \qquad
\text{EV} = P_{\text{model}} \cdot \text{Odds}_{\text{book}} - 1
\]

### Fractional Kelly

\[
f^{\star} = f_{\text{kelly}} \cdot \frac{\text{EV}}{\text{Odds}-1}, \quad f_{\text{kelly}} = 0.25
\]

Stake gợi ý \(= f^{\star} \times \text{Bankroll}\) (cắt về \([0,1]\)).

### Quy tắc khuyến nghị (từ `.cursorrules`)

| Điều kiện | Hành động |
|-----------|-----------|
| \(\text{EV} \ge 0.05\) | Hiển thị **Value Bet** + stake Kelly |
| \(\text{EV} < 0.05\) | Không khuyến nghị (có thể vẫn hiện so sánh xác suất) |

---

## 🏁 Kết luận

**Premier League Value Betting & Corner Recommender System** không phải máy “bắt chắc thắng”, mà là **hệ hỗ trợ quyết định**: ước lượng xác suất độc lập → đo lệch giá nhà cái → chỉ xuống tiền khi EV vượt ngưỡng → kiểm soát quy mô bằng Kelly.

Đó là sự kết hợp của:

- 📐 Mô hình thống kê có nền tảng học thuật (Dixon–Coles),
- 🔌 Pipeline dữ liệu / odds tự động,
- 🖥️ Sản phẩm tương tác Streamlit sẵn sàng demo,
- 🧭 Tư duy quản trị rủi ro có kỷ luật.

---

<div align="center">

**Built with Python · Pandas · SciPy · Statsmodels · Streamlit · Plotly**

*Educational / research use — không đảm bảo lợi nhuận cá cược.*

</div>
