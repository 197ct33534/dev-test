# VALUE BETTING — Pro Sports Analytics Dashboard

Giao diện dark Fintech cho feed **Top Kèo Hời (+EV)**: React 18 + Vite + Tailwind +
React Query, đọc dữ liệu thật từ FastAPI (`/api/v1`).

## Chạy local

Terminal 1 — API (tắt auth chỉ khi chạy local):

```bash
APP_ENV=dev TELEGRAM_AUTH_DISABLED=true python run_api.py
```

Terminal 2 — dashboard (Vite proxy `/api` → `http://127.0.0.1:8000`):

```bash
cd dashboard
npm install
npm run dev
```

Mở `http://localhost:5173`. API chạy cổng khác thì đặt
`VITE_API_PROXY_TARGET=http://127.0.0.1:<port>` trong `dashboard/.env.local`.

Làm UI offline không cần backend: `VITE_USE_MOCK=true npm run dev`.

## Biến môi trường

| Biến | Ý nghĩa |
|------|---------|
| `VITE_API_BASE` | Origin của API khi dashboard host ở domain khác. Để trống = cùng origin |
| `VITE_API_PROXY_TARGET` | Đích proxy của dev server (mặc định `http://127.0.0.1:8000`) |
| `VITE_USE_MOCK` | `true` → dùng `src/data/mockMatches.ts`, không gọi API |

## Xác thực

- Trong Telegram: tự gửi `Authorization: tma <initData>`.
- Ngoài Telegram: server đặt `DASHBOARD_API_TOKENS` (mỗi token ≥ 16 ký tự). Khi API trả
  401, dashboard hiện ô nhập token và lưu vào `localStorage`. Token chỉ có quyền đọc —
  endpoint ghi vẫn cần Telegram admin (`ADMIN_TELEGRAM_IDS`).

## Production

`npm run build` sinh `dist/` với base `/dashboard/`. FastAPI tự mount `dashboard/dist`
tại `/dashboard` khi thư mục tồn tại (`DASHBOARD_DIST` để đổi đường dẫn). Docker build
dashboard trong stage Node riêng (xem `Dockerfile`).

## Dữ liệu

| Endpoint | Dùng cho |
|----------|----------|
| `GET /api/v1/value-bets?leagues=&markets=&min_ev=&kelly_fraction=` | Feed, poll 60 s. Trả `generated_at` và `league_meta` (nguồn odds, tuổi odds, stale, ML bật/tắt) |
| `GET /api/v1/fixtures?leagues=&min_ev=` | Chế độ **Lịch đấu**: mọi trận trong khung hôm nay/mai kèm odds 1X2, AH, O/U và trạng thái (`value` / `below` / `gated` / `no_odds` / `no_model`) |
| `GET /api/v1/leagues` | Danh sách giải trong sidebar (`config/leagues.json`) |
| `GET /api/v1/config` | Chính sách stake: EV tối thiểu, Kelly fraction và trần, trần stake |
| `GET /api/v1/teams/{id}/profile` | Phong độ 5 trận và H2H trong drawer chi tiết |

EV và stake luôn lấy từ backend (EV theo bảng tất toán cho AH/O-U, Kelly có trần đội
mới); dashboard không tự tính lại. Giờ thi đấu luôn hiển thị theo giờ VN.
Mặc định quét mọi giải trong `config/leagues.json`. Đổi Kelly fraction sẽ gọi lại API (debounce 400 ms).

## Cấu trúc

| Path | Vai trò |
|------|---------|
| `src/api/client.ts` | `fetch` có type, header auth, `ApiError` |
| `src/api/hooks.ts` | React Query hooks (polling, cache) |
| `src/api/adapters.ts` | API → `MatchCardData`, phong độ/H2H từ team profile |
| `src/App.tsx` | Layout, bộ lọc, trạng thái loading/lỗi/401/odds cũ |
| `src/components/MatchCard.tsx` | Thẻ kèo + chip độ tin cậy |
| `src/components/MatchDetailDrawer.tsx` | Drawer: phong độ, công thức EV, lý do mô hình, H2H |
| `src/components/TokenGate.tsx` | Nhập token khi API trả 401 |

## Kiểm tra

```bash
npm test          # vitest (adapter, query builder)
npx tsc -b --noEmit
```
