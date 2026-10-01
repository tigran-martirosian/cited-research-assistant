import { useLayoutEffect, useRef, useState } from "react";
import { CheckIcon, CopyIcon } from "./Icons";

/** The question as a compact chat bubble: body font size, line breaks kept, clamped to
 * about four lines with Show more / Show less when it is longer. Copy puts the whole
 * question on the clipboard. */
export function QuestionBubble({ text }: { text: string }) {
  const body = useRef<HTMLParagraphElement>(null);
  const [expanded, setExpanded] = useState(false);
  const [overflows, setOverflows] = useState(false);
  const [copied, setCopied] = useState(false);

  useLayoutEffect(() => {
    const el = body.current;
    if (!el || expanded) return;
    const measure = () => setOverflows(el.scrollHeight > el.clientHeight + 1);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [text, expanded]);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  };

  return (
    <div className="question-bubble">
      <p ref={body} className={`question ${expanded ? "" : "clamped"}`}>
        {text}
      </p>
      <div className="question-actions">
        {(overflows || expanded) && (
          <button className="question-more" onClick={() => setExpanded(!expanded)} aria-expanded={expanded}>
            {expanded ? "Show less" : "Show more"}
          </button>
        )}
        <button className="question-more question-copy" onClick={copy} title="Copy question">
          {copied ? <CheckIcon /> : <CopyIcon />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
    </div>
  );
}
