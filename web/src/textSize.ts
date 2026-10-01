import { useEffect, useState } from "react";

/** The reader's text size for answers and sources, a scale on --text-scale. */
export const TEXT_SIZES = [0.9, 1, 1.12, 1.25] as const;

const KEY = "cra-text-size";

function readSize(): number {
  try {
    const v = Number(localStorage.getItem(KEY));
    return TEXT_SIZES.includes(v as (typeof TEXT_SIZES)[number]) ? v : 1;
  } catch {
    return 1;
  }
}

/** The text size, persisted in localStorage and applied as --text-scale on <html>. index.html
 * applies the stored size before first paint. */
export function useTextSize(): [number, (size: number) => void] {
  const [size, setSize] = useState<number>(readSize);

  useEffect(() => {
    document.documentElement.style.setProperty("--text-scale", String(size));
    try {
      if (size === 1) localStorage.removeItem(KEY);
      else localStorage.setItem(KEY, String(size));
    } catch {
      /* storage unavailable: the size lasts for this page only */
    }
  }, [size]);

  return [size, setSize];
}
