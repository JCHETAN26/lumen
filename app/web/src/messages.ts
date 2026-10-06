import type { AskResult } from "./api";

// What the app says. The model can't tell *why* a photo is unanswerable, so the tips are general
// advice, never a diagnosis of this photo.
export const CANT_TELL = "I can't tell from this photo.";
export const TIPS = [
  "Add more light, or turn toward a window.",
  "Hold the camera steady for a second before taking the photo.",
  "Move back a little so the whole object is in the photo.",
  "Check that the question is about something in the photo.",
];

export function spokenResult(r: AskResult): string {
  if (r.unanswerable || !r.answer) return `${CANT_TELL} Try: ${TIPS.join(" ")}`;
  return `Answer: ${r.answer}`;
}
