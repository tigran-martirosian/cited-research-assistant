export function seconds(value: number): string {
  if (value < 60) return `${value.toFixed(1)}s`;
  const minutes = Math.floor(value / 60);
  return `${minutes}m ${Math.round(value - minutes * 60)}s`;
}

export function count(value: number): string {
  return value.toLocaleString("en-US");
}

export function chars(value: number): string {
  if (value >= 10_000) return `${(value / 1000).toFixed(1)}k chars`;
  return `${count(value)} chars`;
}

export function cost(value: number | null | undefined): string {
  if (value == null) return "—";
  return value < 0.01 ? `$${value.toFixed(4)}` : `$${value.toFixed(2)}`;
}

export function dateTime(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

const DAY = 24 * 60 * 60 * 1000;

/** History group label for a timestamp: Today, Yesterday, Previous 7 days, Month Year. */
export function historyGroup(epochSeconds: number, now = new Date()): string {
  const date = new Date(epochSeconds * 1000);
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const t = date.getTime();
  if (t >= today) return "Today";
  if (t >= today - DAY) return "Yesterday";
  if (t >= today - 7 * DAY) return "Previous 7 days";
  return date.toLocaleString(undefined, { month: "long", year: "numeric" });
}

/** "18.4k tokens" */
export function tokens(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M tokens`;
  if (value >= 1000) return `${(value / 1000).toFixed(1)}k tokens`;
  return `${value} tokens`;
}

export interface TokenTotals {
  input: number;
  cacheWrite: number;
  cacheRead: number;
  output: number;
  total: number;
}

function tokenCount(usage: Record<string, unknown> | null, key: string): number {
  const value = usage?.[key];
  return typeof value === "number" ? value : 0;
}

/** One Claude stage's reported usage. The four Anthropic fields are disjoint (input excludes
 * cache reads and writes; output includes thinking), so their sum counts nothing twice. It is a
 * raw token count, not cost-weighted: a cache read costs a tenth of fresh input. */
export function stageTokens(usage: Record<string, unknown> | null): TokenTotals {
  const input = tokenCount(usage, "input_tokens");
  const cacheWrite = tokenCount(usage, "cache_creation_input_tokens");
  const cacheRead = tokenCount(usage, "cache_read_input_tokens");
  const output = tokenCount(usage, "output_tokens");
  return { input, cacheWrite, cacheRead, output, total: input + cacheWrite + cacheRead + output };
}

/** Claude usage summed over every Claude stage of a run (Gemini rows, whose fields mean
 * something else, are left out). */
export function runTokens(stages: { usage: Record<string, unknown> | null; provider?: string }[]): TokenTotals {
  const sum: TokenTotals = { input: 0, cacheWrite: 0, cacheRead: 0, output: 0, total: 0 };
  for (const s of stages) {
    if (s.provider === "gemini") continue;
    const t = stageTokens(s.usage);
    sum.input += t.input;
    sum.cacheWrite += t.cacheWrite;
    sum.cacheRead += t.cacheRead;
    sum.output += t.output;
    sum.total += t.total;
  }
  return sum;
}
