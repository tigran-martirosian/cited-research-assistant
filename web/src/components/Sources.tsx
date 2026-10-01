import { useEffect, useRef, useState } from "react";
import type { Citation, Excerpt, Source } from "../types";
import { chars } from "../format";
import { BookIcon, ChevronIcon } from "./Icons";

function Toggle({ label, text }: { label: string; text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="subtoggle">
      <button className={`subtoggle-button ${open ? "open" : ""}`} onClick={() => setOpen(!open)} aria-expanded={open}>
        <ChevronIcon />
        {open ? `Hide ${label}` : `Show ${label}`}
        <span className="muted"> · {chars(text.length)}</span>
      </button>
      {open && <div className="source-text extra">{text}</div>}
    </div>
  );
}

function ExcerptView({ excerpt }: { excerpt: Excerpt }) {
  const queries = [...new Set(excerpt.passages.flatMap((p) => p.queries))];
  return (
    <div className="excerpt">
      {excerpt.passages.map((passage) => (
        <blockquote key={passage.hit_id} className="source-text">
          {passage.text}
        </blockquote>
      ))}
      {excerpt.incomplete && (
        <p className="excerpt-note">
          This passage stops at a question or cut-off point; the text that follows was not fully retrieved.
        </p>
      )}
      {excerpt.continuation && <Toggle label="continuation" text={excerpt.continuation} />}
      {excerpt.context && <Toggle label="surrounding context" text={excerpt.context} />}
      {queries.length > 0 && (
        <p className="found-by">
          Found by {queries.map((q, i) => (
            <span key={q}>
              {i > 0 && ", "}
              <q>{q}</q>
            </span>
          ))}
        </p>
      )}
    </div>
  );
}

function SourceCard({ source, index }: { source: Source; index: number }) {
  const [open, setOpen] = useState(false);
  const preview = source.excerpts[0]?.passages[0]?.text ?? "";
  const extras = source.excerpts.filter((e) => e.context || e.continuation).length;
  return (
    <li className={`source-card ${open ? "open" : ""}`}>
      <button className="source-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="source-index">{index}</span>
        <span className="source-main">
          <span className="source-title">{source.name || source.title || "Untitled source"}</span>
          {!open && <span className="source-preview">{preview}</span>}
        </span>
        <span className="source-meta">
          {source.excerpts.length > 1 && <span>{source.excerpts.length} excerpts</span>}
          {extras > 0 && <span className="badge">context</span>}
          <ChevronIcon className="chevron" />
        </span>
      </button>
      {open && (
        <div className="source-body">
          {source.excerpts.map((excerpt, i) => (
            <ExcerptView key={i} excerpt={excerpt} />
          ))}
        </div>
      )}
    </li>
  );
}

/** Every retrieved source of the evidence (Research Details, and older saved results). */
export function SourceList({ sources }: { sources: Source[] }) {
  return (
    <ol className="source-list">
      {sources.map((source, i) => (
        <SourceCard key={source.source_id} source={source} index={i + 1} />
      ))}
    </ol>
  );
}

/** The date, unless the readable name already shows its year. */
function dateNote(c: Citation): string | null {
  if (!c.date || c.name.includes(c.date.slice(0, 4))) return null;
  return c.date;
}

function CitationCard({ citation, open, onToggle, anchor }: {
  citation: Citation;
  open: boolean;
  onToggle: () => void;
  anchor: string;
}) {
  const date = dateNote(citation);
  return (
    <li id={anchor} className={`source-card ${open ? "open" : ""}`}>
      <button className="source-head" onClick={onToggle} aria-expanded={open}>
        <span className="source-index">{citation.n}</span>
        <span className="source-main">
          <span className="source-title">{citation.name}</span>
          {date && <span className="source-date">{date}</span>}
          {!open && <span className="source-preview">{citation.text}</span>}
        </span>
        <span className="source-meta">
          {citation.kind === "community" && <span className="badge community">community</span>}
          <ChevronIcon className="chevron" />
        </span>
      </button>
      {open && (
        <div className="source-body">
          <div className="excerpt">
            {citation.preceding && <Toggle label="text before" text={citation.preceding} />}
            <blockquote className="source-text">{citation.text}</blockquote>
            {citation.continuation && <Toggle label="continuation" text={citation.continuation} />}
            {citation.context && (
              <Toggle label={citation.kind === "community" ? "original wording" : "surrounding context"} text={citation.context} />
            )}
          </div>
        </div>
      )}
    </li>
  );
}

interface Props {
  sources: Source[];
  citations?: Citation[];
  /** The citation last clicked in the answer (seq changes on every click). */
  focus?: { n: number; seq: number } | null;
  /** Prefix for the entries' element ids (several answers can be on one page). */
  idPrefix: string;
}

/** The sources behind an answer, collapsed to one line until clicked. Only the passages the
 * answer cites, numbered to match its superscripts; clicking a superscript opens its entry.
 * Results saved before citations existed (or an answer that cites nothing) list every source. */
export function Sources({ sources, citations, focus, idPrefix }: Props) {
  const [open, setOpen] = useState(false);
  const [expanded, setExpanded] = useState<Set<number>>(new Set());
  const list = useRef<HTMLOListElement>(null);
  const cited = citations && citations.length > 0 ? citations : null;

  useEffect(() => {
    if (!focus || !cited) return;
    setOpen(true);
    setExpanded((prev) => new Set(prev).add(focus.n));
    // After the panel and entry render open.
    requestAnimationFrame(() =>
      document.getElementById(`${idPrefix}-cite-${focus.n}`)?.scrollIntoView({ behavior: "smooth", block: "center" }),
    );
  }, [focus, cited, idPrefix]);

  if (!cited && sources.length === 0) return null;
  const excerpts = sources.reduce((n, s) => n + s.excerpts.length, 0);
  const toggle = (n: number) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(n)) next.delete(n);
      else next.add(n);
      return next;
    });

  return (
    <section className={`sources ${open ? "open" : ""}`}>
      <button className="sources-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <ChevronIcon className="chevron" />
        <BookIcon />
        <span>Sources</span>
        <span className="muted sources-summary">
          {cited
            ? `${cited.length} cited ${cited.length === 1 ? "passage" : "passages"}`
            : `${sources.length} ${sources.length === 1 ? "source" : "sources"} · ${excerpts} ${excerpts === 1 ? "excerpt" : "excerpts"}`}
        </span>
      </button>
      {open &&
        (cited ? (
          <ol className="source-list" ref={list}>
            {cited.map((c) => (
              <CitationCard
                key={c.n}
                citation={c}
                anchor={`${idPrefix}-cite-${c.n}`}
                open={expanded.has(c.n)}
                onToggle={() => toggle(c.n)}
              />
            ))}
          </ol>
        ) : (
          <SourceList sources={sources} />
        ))}
    </section>
  );
}
