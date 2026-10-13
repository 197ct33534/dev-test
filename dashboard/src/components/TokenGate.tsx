import { useState } from "react";
import { KeyRound } from "lucide-react";

interface TokenGateProps {
  /** True when a stored token was rejected (vs. no token yet). */
  rejected: boolean;
  onSubmit: (token: string) => void;
}

/** Shown on HTTP 401: asks for a DASHBOARD_API_TOKENS value (stored locally). */
export function TokenGate({ rejected, onSubmit }: TokenGateProps) {
  const [value, setValue] = useState("");
  return (
    <div className="vb-card mx-auto mt-10 max-w-md p-6">
      <div className="flex items-center gap-2">
        <KeyRound className="h-4 w-4 text-accent-cyan" />
        <h2 className="font-display text-base font-semibold text-ink">
          Cần token truy cập
        </h2>
      </div>
      <p className="mt-2 text-sm text-ink-muted">
        {rejected
          ? "Token hiện tại không hợp lệ. Nhập token khác."
          : "Ngoài Telegram, dashboard cần một token đọc (biến DASHBOARD_API_TOKENS trên server)."}
      </p>
      <form
        className="mt-4 flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          const token = value.trim();
          if (token) onSubmit(token);
        }}
      >
        <input
          className="vb-input"
          type="password"
          autoComplete="off"
          placeholder="Dán token…"
          value={value}
          onChange={(e) => setValue(e.target.value)}
        />
        <button type="submit" className="vb-btn-primary shrink-0">
          Lưu
        </button>
      </form>
      <p className="mt-3 text-[11px] text-ink-faint">
        Token chỉ lưu trong trình duyệt này và chỉ cho quyền đọc.
      </p>
    </div>
  );
}
