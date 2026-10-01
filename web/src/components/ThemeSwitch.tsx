import { useTheme, type ThemePref } from "../theme";
import { MonitorIcon, MoonIcon, SunIcon } from "./Icons";

const OPTIONS: { value: ThemePref; label: string; Icon: typeof SunIcon }[] = [
  { value: "light", label: "Light", Icon: SunIcon },
  { value: "dark", label: "Dark", Icon: MoonIcon },
  { value: "system", label: "System", Icon: MonitorIcon },
];

/** Compact Light / Dark / System segmented control. */
export function ThemeSwitch() {
  const [pref, setPref] = useTheme();
  return (
    <div className="theme-switch" role="radiogroup" aria-label="Theme">
      {OPTIONS.map(({ value, label, Icon }) => (
        <button
          key={value}
          role="radio"
          aria-checked={pref === value}
          className={pref === value ? "on" : ""}
          onClick={() => setPref(value)}
          title={`${label} theme`}
        >
          <Icon />
          <span className="theme-label">{label}</span>
        </button>
      ))}
    </div>
  );
}
