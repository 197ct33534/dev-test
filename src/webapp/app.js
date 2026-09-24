/* Telegram WebApp — Lite value-bet cards + journal performance */
(function () {
  const tg = window.Telegram && window.Telegram.WebApp;
  if (tg) {
    tg.ready();
    try { tg.expand(); } catch (_) {}
  }

  const state = {
    bets: [],
    market: "ALL",
    view: "bets",
    loading: false,
    perf: null,
  };

  const el = {
    user: document.getElementById("user-name"),
    status: document.getElementById("status"),
    bets: document.getElementById("bets"),
    tabs: document.getElementById("tabs"),
    refresh: document.getElementById("btn-refresh"),
    modal: document.getElementById("modal"),
    modalTitle: document.getElementById("modal-title"),
    modalBody: document.getElementById("modal-body"),
    modalClose: document.getElementById("modal-close"),
    mainNav: document.getElementById("main-nav"),
    viewBets: document.getElementById("view-bets"),
    viewJournal: document.getElementById("view-journal"),
    perfStatus: document.getElementById("perf-status"),
    perfMetrics: document.getElementById("perf-metrics"),
    perfWinrate: document.getElementById("perf-winrate"),
    perfPnl: document.getElementById("perf-pnl"),
    perfRoi: document.getElementById("perf-roi"),
    perfMeta: document.getElementById("perf-meta"),
  };

  const user = tg && tg.initDataUnsafe && tg.initDataUnsafe.user;
  if (user) {
    const name = [user.first_name, user.last_name].filter(Boolean).join(" ") || user.username || "Bạn";
    el.user.textContent = name;
  }

  /** Auth headers for API calls (Telegram Mini App initData). */
  function authHeaders() {
    const headers = {};
    const initData = tg && tg.initData;
    if (initData) {
      headers["Authorization"] = `tma ${initData}`;
      headers["X-Telegram-Init-Data"] = initData;
    }
    return headers;
  }

  function apiFetch(url, options) {
    const opts = options || {};
    const headers = Object.assign({}, authHeaders(), opts.headers || {});
    return fetch(url, Object.assign({}, opts, { headers }));
  }

  function esc(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function fmtEv(v) {
    if (v == null || Number.isNaN(Number(v))) return "—";
    return `${Number(v).toFixed(1)}%`;
  }

  function fmtOdds(v) {
    if (v == null || Number.isNaN(Number(v))) return "n/a";
    return Number(v).toFixed(2);
  }

  function signedColor(n) {
    if (n == null || Number.isNaN(Number(n))) return "text-tg-hint";
    const x = Number(n);
    if (x > 0) return "text-emerald-400";
    if (x < 0) return "text-rose-400";
    return "text-tg-text";
  }

  function restLine(b) {
    if (b.fatigue_label) return b.fatigue_label;
    const hr = b.home_rest_days;
    const ar = b.away_rest_days;
    if (hr == null && ar == null) return "";
    const h = hr != null ? `${b.home || "Chủ"} nghỉ ${Number(hr).toFixed(0)}n` : "";
    const a = ar != null ? `${b.away || "Khách"} nghỉ ${Number(ar).toFixed(0)}n` : "";
    return [h, a].filter(Boolean).join(" · ");
  }

  function filteredBets() {
    if (state.market === "ALL") return state.bets;
    return state.bets.filter((b) => String(b.market || "") === state.market);
  }

  function setView(view) {
    state.view = view === "journal" ? "journal" : "bets";
    el.viewBets.classList.toggle("hidden", state.view !== "bets");
    el.viewJournal.classList.toggle("hidden", state.view !== "journal");
    el.tabs.classList.toggle("hidden", state.view !== "bets");
    el.mainNav.querySelectorAll(".nav-tab").forEach((btn) => {
      const active = btn.dataset.view === state.view;
      btn.classList.toggle("nav-active", active);
      btn.classList.toggle("text-tg-hint", !active);
    });
    if (state.view === "journal") loadPerformance();
  }

  function renderCards() {
    const list = filteredBets();
    if (!list.length) {
      el.bets.innerHTML = "";
      el.status.textContent = state.loading
        ? "Đang tải…"
        : "Không có kèo phù hợp bộ lọc.";
      return;
    }
    el.status.textContent = `${list.length} kèo · EV ≥ ngưỡng`;
    el.bets.innerHTML = list
      .map((b, i) => {
        const reasons = (b.ai_reasons || [])
          .map((r) => `<li class="text-tg-hint text-xs">${esc(r)}</li>`)
          .join("");
        const rest = restLine(b);
        return `
        <article class="card rounded-xl p-3.5 shadow-sm border border-white/5" data-idx="${i}">
          <div class="flex justify-between gap-2 items-start">
            <div>
              <p class="font-semibold text-[15px] leading-snug">${esc(b.home)} vs ${esc(b.away)}</p>
              <p class="text-xs text-tg-hint mt-0.5">${esc(b.competition || b.league || "")} · ${esc(b.kickoff_vn || b.kickoff || "")}</p>
            </div>
            <span class="shrink-0 text-xs font-bold px-2 py-1 rounded-md bg-emerald-500/15 text-emerald-300">EV ${esc(fmtEv(b.ev_pct))}</span>
          </div>
          <p class="mt-2 text-sm text-tg-link font-medium">${esc(b.pick || `${b.market} · ${b.selection}`)} · @${esc(fmtOdds(b.odds))}</p>
          ${b.model_fair_line || b.bookie_market_line || b.line_edge
            ? `<p class="mt-1 text-xs text-sky-300/90">${esc(
                [b.model_fair_line && `Model ${b.model_fair_line}`,
                 b.bookie_market_line && `NC ${b.bookie_market_line}`,
                 b.line_edge && `lệch ${b.line_edge}`]
                  .filter(Boolean)
                  .join(" · ")
              )}</p>`
            : ""}
          ${rest ? `<p class="mt-1 text-xs text-amber-300/90">${esc(rest)}</p>` : ""}
          ${reasons ? `<ul class="mt-2 space-y-0.5 list-disc list-inside">${reasons}</ul>` : ""}
          <div class="mt-3 flex gap-2">
            <button type="button" class="btn-profile flex-1 text-xs py-2 rounded-lg bg-white/5 border border-white/10"
              data-home="${esc(b.home)}" data-away="${esc(b.away)}">📊 Xem Hồ Sơ Đội</button>
          </div>
        </article>`;
      })
      .join("");
  }

  function renderPerformance(data) {
    state.perf = data;
    if (!data) {
      el.perfStatus.classList.remove("hidden");
      el.perfStatus.textContent = "Không có dữ liệu.";
      el.perfMetrics.classList.add("hidden");
      el.perfMeta.classList.add("hidden");
      return;
    }
    el.perfStatus.classList.add("hidden");
    el.perfMetrics.classList.remove("hidden");

    const wr = data.win_rate_percent;
    const pnl = data.net_pnl;
    const roi = data.realized_roi_percent;

    el.perfWinrate.textContent = wr == null ? "—" : `${Number(wr).toFixed(1)}%`;
    el.perfWinrate.className = `text-lg font-semibold mt-0.5 ${signedColor(wr - 50)}`;

    el.perfPnl.textContent =
      pnl == null ? "—" : `${pnl >= 0 ? "+" : ""}${Number(pnl).toFixed(2)}$`;
    el.perfPnl.className = `text-lg font-semibold mt-0.5 ${signedColor(pnl)}`;

    el.perfRoi.textContent =
      roi == null ? "—" : `${roi >= 0 ? "+" : ""}${Number(roi).toFixed(1)}%`;
    el.perfRoi.className = `text-lg font-semibold mt-0.5 ${signedColor(roi)}`;

    const bits = [
      data.total_bets_settled != null && `${data.total_bets_settled} đã settle`,
      data.total_bets_placed != null && `${data.total_bets_placed} đã đặt`,
      data.brier_score != null && `Brier ${Number(data.brier_score).toFixed(3)}`,
      data.brier_note,
      data.ev_vs_realized_gap != null &&
        `EV−ROI gap ${Number(data.ev_vs_realized_gap).toFixed(1)}pp`,
    ].filter(Boolean);
    if (bits.length) {
      el.perfMeta.classList.remove("hidden");
      el.perfMeta.textContent = bits.join(" · ");
    } else {
      el.perfMeta.classList.add("hidden");
    }
  }

  async function loadBets() {
    state.loading = true;
    el.status.textContent = "Đang tải kèo hời…";
    el.refresh.disabled = true;
    try {
      const res = await apiFetch("/api/v1/value-bets?min_ev=5&limit=20&markets=1X2,AH,OU,Corners");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      state.bets = data.bets || [];
      renderCards();
    } catch (err) {
      el.status.textContent = `Lỗi tải kèo: ${err.message || err}`;
      el.bets.innerHTML = "";
    } finally {
      state.loading = false;
      el.refresh.disabled = false;
    }
  }

  async function loadPerformance() {
    el.perfStatus.classList.remove("hidden");
    el.perfStatus.textContent = "Đang tải hiệu suất…";
    el.perfMetrics.classList.add("hidden");
    try {
      const res = await apiFetch("/api/v1/analytics/performance");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      renderPerformance(data);
    } catch (err) {
      el.perfStatus.textContent = `Lỗi: ${err.message || err}`;
      el.perfMetrics.classList.add("hidden");
    }
  }

  function profileBlock(title, data) {
    if (!data) return `<p class="text-tg-hint">${esc(title)}: không có dữ liệu</p>`;
    const past = (data.past_matches || [])
      .map(
        (m) =>
          `<li><span class="text-tg-hint">${esc(m.date)}</span> · ${esc(m.competition)} — ${esc(m.home)} ${esc(m.score)} ${esc(m.away)} (${esc(m.result || "")})</li>`
      )
      .join("") || "<li class='text-tg-hint'>Chưa có trận gần đây</li>";
    const up = (data.upcoming_matches || [])
      .map(
        (m) =>
          `<li><span class="text-tg-hint">${esc(m.kickoff)}</span> · ${esc(m.competition)} — vs ${esc(m.opponent)}</li>`
      )
      .join("") || "<li class='text-tg-hint'>Chưa có lịch sắp tới</li>";
    return `
      <section>
        <h3 class="font-semibold mb-1">${esc(data.display_name || title)}</h3>
        <p class="text-xs text-tg-hint mb-2">5 trận vừa qua</p>
        <ul class="space-y-1 mb-3">${past}</ul>
        <p class="text-xs text-tg-hint mb-2">5 trận sắp tới</p>
        <ul class="space-y-1">${up}</ul>
      </section>`;
  }

  async function openProfiles(home, away) {
    el.modalTitle.textContent = `${home} · ${away}`;
    el.modalBody.innerHTML = `<p class="text-tg-hint">Đang tải hồ sơ…</p>`;
    el.modal.classList.remove("hidden");
    try {
      const [rh, ra] = await Promise.all([
        apiFetch(`/api/v1/teams/${encodeURIComponent(home)}/profile`),
        apiFetch(`/api/v1/teams/${encodeURIComponent(away)}/profile`),
      ]);
      const dh = rh.ok ? await rh.json() : null;
      const da = ra.ok ? await ra.json() : null;
      el.modalBody.innerHTML =
        profileBlock(home, dh) +
        `<hr class="border-white/10" />` +
        profileBlock(away, da);
    } catch (err) {
      el.modalBody.innerHTML = `<p class="text-red-300">Lỗi: ${esc(err.message || err)}</p>`;
    }
  }

  el.mainNav.addEventListener("click", (e) => {
    const btn = e.target.closest(".nav-tab");
    if (!btn) return;
    setView(btn.dataset.view || "bets");
  });

  el.tabs.addEventListener("click", (e) => {
    const btn = e.target.closest(".tab");
    if (!btn) return;
    state.market = btn.dataset.market || "ALL";
    el.tabs.querySelectorAll(".tab").forEach((t) => {
      t.classList.remove("tab-active", "bg-white/5", "text-tg-hint");
      if (t === btn) t.classList.add("tab-active");
      else t.classList.add("bg-white/5", "text-tg-hint");
    });
    renderCards();
  });

  el.bets.addEventListener("click", (e) => {
    const btn = e.target.closest(".btn-profile");
    if (!btn) return;
    openProfiles(btn.dataset.home || "", btn.dataset.away || "");
  });

  el.modalClose.addEventListener("click", () => el.modal.classList.add("hidden"));
  el.modal.addEventListener("click", (e) => {
    if (e.target === el.modal) el.modal.classList.add("hidden");
  });
  el.refresh.addEventListener("click", () => {
    if (state.view === "journal") loadPerformance();
    else loadBets();
  });

  // Deep-link: /webapp/?tab=journal or #journal
  const params = new URLSearchParams(window.location.search);
  const hash = (window.location.hash || "").replace(/^#/, "");
  if (params.get("tab") === "journal" || hash === "journal") {
    setView("journal");
  }

  loadBets();
})();
