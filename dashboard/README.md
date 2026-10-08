# VALUE BETTING — Pro Sports Analytics Dashboard

Giao diện dark Fintech cho feed **Top Kèo Hời (+EV)**: React 18 + Vite + Tailwind.

> **Trạng thái: WIP — chỉ dữ liệu mẫu (mock).** Chưa nối API; header hiển thị
> badge "Dữ liệu mẫu (mock)". Giờ kickoff mock được sinh tương đối so với lúc
> mở trang; EV/stake tính từ `modelProb × odd` theo policy backend
> (Kelly fraction 0.10, stake tối đa 1% bankroll).

## Chạy local

```bash
cd dashboard
npm install
npm run dev
```

Mở `http://localhost:5173`.

## Cấu trúc

| Path | Vai trò |
|------|---------|
| `src/App.tsx` | Layout + realtime filter |
| `src/components/Header.tsx` | Logo, nav, realtime badge, Nhanh/Nâng cao |
| `src/components/Sidebar.tsx` | Control panel (EV, Kelly, markets, leagues) |
| `src/components/MatchCard.tsx` | Thẻ trận +EV / OU lệch / actions |
| `src/components/MatchDetailDrawer.tsx` | Drawer: form 5 trận, EV math, H2H |
| `src/data/mockMatches.ts` | 6 J.League Cup + 1 EPL mock; `LEAGUE_OPTIONS` = keys trong `config/leagues.json` |

## Design tokens

- Canvas `#090D16` / raised `#0F172A`
- Accent emerald `#10B981`, cyan `#06B6D4`, amber `#F59E0B`, red `#EF4444`
- Cards `rounded-xl`, border `slate-800`

## Tích hợp backend

Thay `MOCK_MATCHES` bằng API FastAPI (`run_api.py`, v1/v2) và bỏ prop `isMock`
ở `Header`. Giữ shape `MatchCardData` trong `src/types.ts`.
