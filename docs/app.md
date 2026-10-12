# Lumen app (Stage 5)

A web page served by a small local server on the Mac. Take or choose a photo, type (or dictate) a
question, and hear or read a short answer, or "I can't tell from this photo" with tips.

## Run it

```bash
cd app/web && npm ci && npm run build && cd ../..
uv run --group app --group mlx python -m lumen.app.server --model lr4e-4-e2 --variant 4bit \
    --threshold 0.672   # dev-chosen, results/finetune/choice.json
# then open http://localhost:8765
```

`--fake` runs the same server with a fixed fake answerer (no model), for UI work and the end-to-end
tests. `npm run dev` serves the page with hot reload and proxies `/api` to the server.

## Privacy and safety

- The server binds to 127.0.0.1 and rejects any other Host header (DNS rebinding). Every API call
  needs an `X-Lumen` header, which cross-site forms can't send, and no CORS is granted.
- Photos and questions stay in memory: never written, logged, or uploaded; access logs are off.
- Strict Content-Security-Policy (`'self'` only, no inline script or style), no third-party code
  or fonts at run time, `no-store` on API responses.
- **No in-app speech recognition.** Chrome's Web Speech API sends audio to Google's servers. Users
  dictate with macOS Dictation instead, which the page points to.
- Model output is rendered as plain text (React escapes it); nothing is interpreted as HTML.

## Accessibility (WCAG 2.2 AA, screen-reader first)

- Native controls only (buttons, labelled file input, textarea, checkbox); one `h1`, numbered
  `h2` steps, a `main` landmark.
- The answer area is a polite, atomic live region that is always in the page, so screen readers
  announce answers without moving focus. Errors use `role="alert"`, start with "Error:" (never
  color alone), and move focus to the field that needs fixing (`aria-invalid` +
  `aria-describedby`).
- The Ask button stays focusable while busy (`aria-disabled`, not `disabled`), so focus is never
  lost. The camera keeps focus inside its controls and lands on "Take photo" when it is ready;
  closing it returns focus to "Take a photo".
- Visible 3 px focus outline on everything (including the visually hidden file input's label),
  44 px targets (WCAG 2.5.8 asks for 24), high-contrast light and dark themes, reduced motion
  respected, text resizes with the browser.
- "Read answers aloud" (system voices) is off by default: VoiceOver users already hear the live
  region, and two voices at once are worse than one.
- "Can't tell" gives general tips. The model doesn't say *why* a photo can't answer (blur,
  framing, light), so the app doesn't pretend to.

## Verification

Automated (CI on every push):
- `eslint-plugin-jsx-a11y` (strict) on all components.
- Component tests (Vitest + Testing Library) with axe-core on the initial and answered states.
- End-to-end (Playwright, Chromium) on the built page with the real server: the answer and
  can't-tell flows, keyboard only, the camera flow (synthetic camera), the header guard, and axe
  WCAG 2.0/2.1/2.2 A+AA in light and dark mode, color contrast included.

Manual, on the Mac with VoiceOver (Cmd+F5), Safari. Record the date, macOS version, and result of
each step in the devlog:
1. Tab from the top: every control is reached in order and announced with its label.
2. "Take a photo": permission prompt, then focus on "Take photo"; take a photo; focus returns.
3. "Choose a photo": the file picker opens; after choosing, "Photo ready." is announced.
4. Ask with no question: the error is announced and focus is in the question field.
5. Ask: "Looking at the photo…" then the answer is announced without moving focus.
6. A dark or blurry photo: "I can't tell from this photo." and the tips are announced.
7. Dictate a question with macOS Dictation into the question field.
8. Zoom to 200% (Cmd +): nothing is cut off or overlaps; no horizontal scrolling.
9. Turn on "Read answers aloud" with VoiceOver off: the answer is spoken once.
