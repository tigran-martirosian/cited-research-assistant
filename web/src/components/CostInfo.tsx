import { useState } from "react";
import type { Details } from "../types";
import { cost, runTokens, tokens } from "../format";

/** A small (i) that reveals a run's estimated Claude cost and tokens (the answer header
 * shows only the time). */
export function CostInfo({ details }: { details: Details }) {
  const [open, setOpen] = useState(false);
  const usage = runTokens(details.claude_usage);
  return (
    <span className="cost-info">
      <button
        type="button"
        className={`info-button ${open ? "open" : ""}`}
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-label="Cost and tokens"
        title="Cost and tokens"
      >
        i
      </button>
      {open && (
        <span className="cost-info-text">
          Claude {cost(details.claude_cost_usd)} est. · {tokens(usage.total)} ({tokens(usage.input + usage.cacheWrite + usage.cacheRead)} in,{" "}
          {tokens(usage.output)} out)
        </span>
      )}
    </span>
  );
}
