import { useCallback, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Citation } from "../types";
import { CheckIcon, CopyIcon } from "./Icons";

/** A citation group in the answer's text: "[h15]", "[s2]", "[h3, h7]". */
const CITE = /\[((?:[hs]\d+)(?:\s*[,;]\s*[hs]\d+)*)\]/g;
const CITE_HREF = /^#cite-(\d+)$/;
const POPOVER_CHARS = 700; // passage text shown in a citation's hover popover

function ids(group: string): string[] {
  return group.split(/[,;]/).map((s) => s.trim());
}

/** Evidence ids the text cites, in order of first appearance. */
function citationOrder(text: string): string[] {
  const seen: string[] = [];
  for (const m of text.matchAll(CITE)) for (const id of ids(m[1])) if (!seen.includes(id)) seen.push(id);
  return seen;
}

/** The citation groups as markdown links to "#cite-N" (N = the citation's number); ids without a
 * number (not in the evidence) are dropped. */
function linkCitations(text: string, numbers: Map<string, number>): string {
  return text.replace(CITE, (_m, group: string) =>
    ids(group)
      .filter((id) => numbers.has(id))
      .map((id) => `[${numbers.get(id)}](#cite-${numbers.get(id)})`)
      .join(""),
  );
}

/** Plain text of rendered children (for matching a paragraph's bold lead). */
function textOf(node: unknown): string {
  if (!node || typeof node !== "object") return "";
  const n = node as { type?: string; value?: string; children?: unknown[] };
  if (n.type === "text") return n.value ?? "";
  return (n.children ?? []).map(textOf).join("");
}

/** A paragraph that opens with a bold label for one person's case gets a callout. */
const CASE_LEAD = /^(individual (case|advice)|one person|a case|case note|in one consult|for one person|his advice to)/i;

/** A citation number with its passage in a popover on hover or keyboard focus. */
function CiteRef({ n, href, citation, onCite, children }: {
  n: number;
  href: string;
  citation?: Citation;
  onCite?: (n: number) => void;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ left: number; top: number; above: boolean } | null>(null);
  const anchor = useRef<HTMLAnchorElement>(null);
  const timer = useRef<number | undefined>(undefined);

  const show = useCallback(() => {
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setOpen(true), 120);
  }, []);
  const hide = useCallback(() => {
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setOpen(false), 160);
  }, []);

  useLayoutEffect(() => {
    if (!open || !anchor.current) return;
    const r = anchor.current.getBoundingClientRect();
    const width = Math.min(380, window.innerWidth - 24);
    const left = Math.max(12, Math.min(r.left + r.width / 2 - width / 2, window.innerWidth - width - 12));
    const above = r.top > window.innerHeight * 0.55;
    setPos({ left, top: above ? r.top - 8 : r.bottom + 8, above });
  }, [open]);

  const text = citation
    ? [citation.preceding, citation.text, citation.continuation].filter(Boolean).join(" ")
    : "";
  const clipped = text.length > POPOVER_CHARS ? `${text.slice(0, POPOVER_CHARS).trimEnd()}…` : text;
  return (
    <sup className="cite" onMouseEnter={show} onMouseLeave={hide}>
      <a
        ref={anchor}
        href={href}
        onClick={(e) => {
          e.preventDefault();
          setOpen(false);
          onCite?.(n);
        }}
        onFocus={show}
        onBlur={hide}
        aria-label={`Source ${n}${citation ? `: ${citation.name}` : ""}`}
        aria-describedby={open && citation ? `cite-pop-${n}` : undefined}
      >
        {children}
      </a>
      {open && citation && pos && (
        <span
          id={`cite-pop-${n}`}
          role="tooltip"
          className={`cite-popover ${citation.kind === "community" ? "community" : ""} ${pos.above ? "above" : ""}`}
          style={{ left: pos.left, top: pos.top }}
          onMouseEnter={show}
          onMouseLeave={hide}
        >
          <span className="cite-popover-head">
            <span className="cite-popover-n">{n}</span>
            <span className="cite-popover-source">
              {citation.kind === "community" ? "Community material" : citation.name}
            </span>
            {citation.date && <span className="cite-popover-date">{citation.date}</span>}
          </span>
          <span className="cite-popover-text">{clipped}</span>
          <span className="cite-popover-hint">Click to open in Sources</span>
        </span>
      )}
    </sup>
  );
}

function makeComponents(onCite?: (n: number) => void, byNumber?: Map<number, Citation>): Components {
  return {
    // Wide tables scroll inside their own box instead of widening the page.
    table: ({ node: _node, ...props }) => (
      <div className="table-scroll">
        <table {...props} />
      </div>
    ),
    a: ({ node: _node, href, children, ...props }) => {
      const cite = href?.match(CITE_HREF);
      if (cite && href) {
        const n = Number(cite[1]);
        return (
          <CiteRef n={n} href={href} citation={byNumber?.get(n)} onCite={onCite}>
            {children}
          </CiteRef>
        );
      }
      return <a href={href} {...props} target="_blank" rel="noreferrer noopener">{children}</a>;
    },
    // A paragraph that is only bold text is a block heading; one that opens with a bold
    // individual-case label is a case callout.
    p: ({ node, ...props }) => {
      const kids = (node?.children ?? []).filter((c) => !(c.type === "text" && !c.value.trim()));
      const first = kids[0];
      const lead = kids.length === 1 && first?.type === "element" && first.tagName === "strong";
      const caseNote = !lead && first?.type === "element" && first.tagName === "strong" && CASE_LEAD.test(textOf(first));
      return <p {...props} className={lead ? "lead-heading" : caseNote ? "callout callout-case" : undefined} />;
    },
  };
}

const plain = makeComponents();

export function Markdown({ text, components = plain }: { text: string; components?: Components }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
      {text}
    </ReactMarkdown>
  );
}

/** Split the answer at its ##/### headings (outside code fences) so the Short answer can get its
 * own treatment. Answers without those headings render as one plain section. A
 * "Community practice" section and an individual-case section are callouts. */
function sections(text: string): { kind: string; body: string }[] {
  const out: { kind: string; body: string }[] = [];
  let current: string[] = [];
  let kind = "lead";
  let fenced = false;
  const flush = () => {
    const body = current.join("\n");
    if (body.trim()) out.push({ kind, body });
  };
  for (const line of text.split("\n")) {
    if (/^\s*(```|~~~)/.test(line)) fenced = !fenced;
    const heading = !fenced && line.match(/^#{2,3}\s+(.+?)\s*#*\s*$/);
    if (heading) {
      flush();
      current = [];
      const title = heading[1].replace(/[*_:]/g, "").trim().toLowerCase();
      kind =
        title === "short answer" ? "short"
        : title === "evidence" ? "evidence"
        : title === "conclusion" ? "conclusion"
        : /^community\b/.test(title) ? "community"
        : /\b(individual (case|cases|advice)|one person|case notes?)\b/.test(title) ? "case"
        : "section";
    }
    current.push(line);
  }
  flush();
  return out;
}

/** Markdown reduced to plain text for the clipboard: no emphasis marks, citations as [n]. */
function plainText(md: string): string {
  return md.replace(/\*\*|__/g, "").replace(/(^|\s)[*_](\S[^*_]*\S)[*_](?=\s|$)/g, "$1$2").trim();
}

const AMOUNT = /\d\s*(?:[-–]\s*\d+(?:\.\d+)?\s*)?(?:%|tbsp|tsp|tablespoons?|teaspoons?|cups?|oz|ounces?|g\b|ml|quarts?|l\b|lb)/gi;

/** The recipe line of the short answer (a rewrite puts it there): the line with the most
 * amounts, when it has at least three. */
function recipeLine(short: string | undefined): string | null {
  if (!short) return null;
  let best: string | null = null;
  let most = 2;
  for (const line of short.split("\n").slice(1)) {
    const n = line.match(AMOUNT)?.length ?? 0;
    if (n > most) {
      most = n;
      best = line;
    }
  }
  return best ? plainText(best.replace(/^\s*(?:[-*+]|>)\s+/, "")) : null;
}

function useCopy(): [string | null, (key: string, text: string) => void] {
  const [copied, setCopied] = useState<string | null>(null);
  const copy = useCallback(async (key: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(key);
      setTimeout(() => setCopied(null), 1500);
    } catch {
      /* clipboard unavailable */
    }
  }, []);
  return [copied, copy];
}

interface Props {
  text: string;
  streaming?: boolean;
  /** The answer's cited passages; while streaming (none yet) ids are numbered as they appear. */
  citations?: Citation[];
  onCite?: (n: number) => void;
}

/** The answer, rendered the same way while it streams and when it is final (streaming only
 * disables Copy, so nothing moves when the stream ends). Citations render as numbered
 * superscripts; hovering one shows its passage, clicking it opens that source in the sources
 * panel. */
export function Answer({ text, streaming = false, citations, onCite }: Props) {
  const [copied, copy] = useCopy();
  const byNumber = useMemo(() => new Map((citations ?? []).map((c) => [c.n, c])), [citations]);
  const components = useMemo(() => makeComponents(onCite, byNumber), [onCite, byNumber]);
  const numbers = useMemo(() => {
    const map = new Map<string, number>();
    if (citations) for (const c of citations) map.set(c.id, c.n);
    else citationOrder(text).forEach((id, i) => map.set(id, i + 1));
    return map;
  }, [citations, text]);
  const shown = useMemo(() => linkCitations(text, numbers), [text, numbers]);
  // Copy gives the text with plain [n] numbers rather than evidence ids.
  const copyText = useMemo(
    () => text.replace(CITE, (_m, g: string) => ids(g).filter((id) => numbers.has(id)).map((id) => `[${numbers.get(id)}]`).join("")),
    [text, numbers],
  );
  const parts = useMemo(() => sections(shown), [shown]);
  const recipe = useMemo(
    () => recipeLine(sections(copyText).find((s) => s.kind === "short")?.body),
    [copyText],
  );

  return (
    <section className="answer">
      <div className="prose">
        {parts.map((s, i) => (
          <section key={i} className={`answer-section answer-${s.kind}`}>
            <Markdown text={s.body} components={components} />
          </section>
        ))}
      </div>
      <div className="answer-actions">
        <button className="ghost-button small" onClick={() => copy("answer", copyText)} disabled={streaming}>
          {copied === "answer" ? <CheckIcon /> : <CopyIcon />}
          {copied === "answer" ? "Copied" : "Copy answer"}
        </button>
        {recipe && (
          <button
            className="ghost-button small"
            onClick={() => copy("recipe", recipe)}
            disabled={streaming}
            title={recipe}
          >
            {copied === "recipe" ? <CheckIcon /> : <CopyIcon />}
            {copied === "recipe" ? "Copied" : "Copy recipe"}
          </button>
        )}
      </div>
    </section>
  );
}
