import { TEXT_SIZES, useTextSize } from "../textSize";

/** Smaller / larger reading text, beside the theme switch. */
export function TextSize() {
  const [size, setSize] = useTextSize();
  const i = TEXT_SIZES.indexOf(size as (typeof TEXT_SIZES)[number]);
  return (
    <div className="text-size" role="group" aria-label="Text size">
      <span className="text-size-label">Text</span>
      <button
        onClick={() => setSize(TEXT_SIZES[i - 1])}
        disabled={i <= 0}
        aria-label="Smaller text"
        title="Smaller text"
      >
        <span className="text-size-a small">A</span>
      </button>
      <span className="text-size-value" aria-live="polite">
        {Math.round(size * 100)}%
      </span>
      <button
        onClick={() => setSize(TEXT_SIZES[i + 1])}
        disabled={i >= TEXT_SIZES.length - 1}
        aria-label="Larger text"
        title="Larger text"
      >
        <span className="text-size-a large">A</span>
      </button>
    </div>
  );
}
