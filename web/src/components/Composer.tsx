import { forwardRef, useImperativeHandle, useLayoutEffect, useRef, useState } from "react";
import { ArrowRightIcon, PlusIcon } from "./Icons";

export interface ComposerHandle {
  focus: () => void;
  /** Put text in the box (without submitting) and focus it. */
  fill: (text: string) => void;
}

interface Props {
  onSubmit: (question: string) => Promise<void>;
  placeholder: string;
  /** Inline under the empty-state intro instead of docked at the bottom. */
  inline?: boolean;
  /** In an open conversation the box asks follow-ups; this starts a new question. */
  onNewQuestion?: () => void;
}

const MAX_HEIGHT = 220;

export const Composer = forwardRef<ComposerHandle, Props>(function Composer(
  { onSubmit, placeholder, inline, onNewQuestion },
  ref,
) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const input = useRef<HTMLTextAreaElement>(null);

  useImperativeHandle(
    ref,
    () => ({
      focus: () => input.current?.focus(),
      fill: (value: string) => {
        setText(value);
        setError(null);
        setTimeout(() => {
          const el = input.current;
          if (!el) return;
          el.focus();
          el.setSelectionRange(value.length, value.length);
        }, 0);
      },
    }),
    [],
  );

  // Grow with the text, up to MAX_HEIGHT, then scroll.
  useLayoutEffect(() => {
    const el = input.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT)}px`;
  }, [text]);

  const question = text.trim();

  async function submit() {
    if (!question || busy) return;
    setBusy(true);
    setError(null);
    try {
      await onSubmit(question);
      setText("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      input.current?.focus();
    }
  }

  return (
    <div className={`composer-dock ${inline ? "inline" : ""}`}>
      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <textarea
          ref={input}
          rows={1}
          value={text}
          placeholder={placeholder}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submit();
            }
          }}
          aria-label="Question"
          autoFocus
        />
        <button
          type="submit"
          className="send"
          disabled={!question || busy}
          title="Research (Enter)"
        >
          <span className="send-label">Research</span>
          <ArrowRightIcon />
        </button>
      </form>
      <div className="composer-foot">
        <p className={`composer-hint ${error ? "error" : ""}`}>
          {error ??
            (onNewQuestion
              ? "Enter to ask · Shift+Enter for a new line · follows up the answer above"
              : "Enter to ask · Shift+Enter for a new line")}
        </p>
        {onNewQuestion && (
          <button type="button" className="ghost-button small composer-new" onClick={onNewQuestion}>
            <PlusIcon />
            New question
          </button>
        )}
      </div>
    </div>
  );
});
