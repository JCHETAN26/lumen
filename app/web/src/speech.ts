// Optional spoken answers with the browser's speech synthesis (on macOS, the system's voices).
// Off by default: screen reader users already hear answers through the live region, and two
// voices at once are worse than one.

export function canSpeak(): boolean {
  return typeof window !== "undefined" && "speechSynthesis" in window;
}

export function speak(text: string): void {
  if (!canSpeak()) return;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
}

const KEY = "lumen.readAloud";

export function loadReadAloud(): boolean {
  try {
    return window.localStorage.getItem(KEY) === "1";
  } catch {
    return false;
  }
}

export function saveReadAloud(on: boolean): void {
  try {
    window.localStorage.setItem(KEY, on ? "1" : "0");
  } catch {
    // storage blocked: the setting lasts for this visit only
  }
}
