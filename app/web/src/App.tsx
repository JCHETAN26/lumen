import { useEffect, useId, useRef, useState, type FormEvent } from "react";
import { ask, AskError, type AskResult } from "./api";
import Camera from "./Camera";
import { CANT_TELL, spokenResult, TIPS } from "./messages";
import { canSpeak, loadReadAloud, saveReadAloud, speak } from "./speech";

const MAX_QUESTION = 500;

type Status =
  | { kind: "idle" }
  | { kind: "asking" }
  | { kind: "done"; result: AskResult; question: string }
  | { kind: "error"; message: string };

export default function App() {
  const [photo, setPhotoState] = useState<{ blob: Blob; url: string } | null>(null);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [problem, setProblem] = useState<{ field: "photo" | "question"; text: string } | null>(
    null,
  );
  const [readAloud, setReadAloud] = useState(loadReadAloud);
  const fileInput = useRef<HTMLInputElement>(null);
  const cameraButton = useRef<HTMLButtonElement>(null);
  const questionBox = useRef<HTMLTextAreaElement>(null);
  const ids = { photoHelp: useId(), questionHelp: useId(), problem: useId() };
  const hasCamera = typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia;

  function setPhoto(blob: Blob | null) {
    if (photo) URL.revokeObjectURL(photo.url);
    setPhotoState(blob ? { blob, url: URL.createObjectURL(blob) } : null);
  }

  function acceptPhoto(blob: Blob) {
    setPhoto(blob);
    setStatus({ kind: "idle" });
    if (problem?.field === "photo") setProblem(null);
  }

  function onFile(files: FileList | null) {
    const f = files?.[0];
    if (f) acceptPhoto(f);
  }

  // When the camera closes, its controls disappear: give focus back to the button that opened
  // it, after React has put that button back in the page.
  const refocusCameraButton = useRef(false);
  useEffect(() => {
    if (!cameraOpen && refocusCameraButton.current) {
      refocusCameraButton.current = false;
      cameraButton.current?.focus();
    }
  }, [cameraOpen]);

  function closeCamera() {
    refocusCameraButton.current = true;
    setCameraOpen(false);
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (status.kind === "asking") return;
    const q = question.trim();
    if (!photo) {
      setProblem({ field: "photo", text: "Choose or take a photo first." });
      (hasCamera ? cameraButton : fileInput).current?.focus();
      return;
    }
    if (!q) {
      setProblem({ field: "question", text: "Type a question about the photo." });
      questionBox.current?.focus();
      return;
    }
    setProblem(null);
    setStatus({ kind: "asking" });
    try {
      const result = await ask(photo.blob, q);
      setStatus({ kind: "done", result, question: q });
      if (readAloud) speak(spokenResult(result));
    } catch (err) {
      const message = err instanceof AskError ? err.message : "Something went wrong. Try again.";
      setStatus({ kind: "error", message });
    }
  }

  function toggleReadAloud(on: boolean) {
    setReadAloud(on);
    saveReadAloud(on);
  }

  const asking = status.kind === "asking";

  return (
    <main>
      <h1>Lumen</h1>
      <p className="lede">
        Ask a question about a photo. Everything stays on this computer: the photo and your
        question are never uploaded or saved.
      </p>

      <form onSubmit={onSubmit} noValidate>
        <section aria-labelledby="photo-heading">
          <h2 id="photo-heading">1. Photo</h2>
          <p id={ids.photoHelp} className="help">
            Take a photo with the camera, or choose one from this computer.
          </p>
          {cameraOpen ? (
            <Camera
              onCapture={(blob) => {
                acceptPhoto(blob);
                closeCamera();
              }}
              onClose={closeCamera}
            />
          ) : (
            <div className="row">
              {hasCamera && (
                <button
                  ref={cameraButton}
                  type="button"
                  onClick={() => setCameraOpen(true)}
                  aria-describedby={ids.photoHelp}
                >
                  Take a photo
                </button>
              )}
              <label className="file-button">
                Choose a photo
                <input
                  ref={fileInput}
                  type="file"
                  accept="image/*"
                  className="visually-hidden"
                  aria-describedby={problem?.field === "photo" ? ids.problem : ids.photoHelp}
                  aria-invalid={problem?.field === "photo" || undefined}
                  onChange={(e) => onFile(e.target.files)}
                />
              </label>
            </div>
          )}
          <p role="status" className="help">
            {photo ? "Photo ready." : ""}
          </p>
          {photo && (
            <div className="photo">
              <img src={photo.url} alt="Preview of what Lumen will look at" />
              <button type="button" onClick={() => setPhoto(null)}>
                Remove photo
              </button>
            </div>
          )}
        </section>

        <section aria-labelledby="question-heading">
          <h2 id="question-heading">2. Question</h2>
          <label htmlFor="question">Your question</label>
          <p id={ids.questionHelp} className="help">
            For example: “What does this label say?” To dictate, use macOS Dictation (press the
            Microphone key, or Fn twice).
          </p>
          <textarea
            id="question"
            ref={questionBox}
            rows={3}
            maxLength={MAX_QUESTION}
            value={question}
            onChange={(e) => {
              setQuestion(e.target.value);
              if (problem?.field === "question") setProblem(null);
            }}
            aria-describedby={
              problem?.field === "question" ? `${ids.problem} ${ids.questionHelp}` : ids.questionHelp
            }
            aria-invalid={problem?.field === "question" || undefined}
          />
        </section>

        {problem && (
          <p id={ids.problem} role="alert" className="error">
            <strong>Error:</strong> {problem.text}
          </p>
        )}

        <div className="row">
          <button type="submit" className="primary" aria-disabled={asking || undefined}>
            {asking ? "Asking…" : "Ask"}
          </button>
        </div>
      </form>

      <section aria-labelledby="answer-heading" className="answer">
        <h2 id="answer-heading">Answer</h2>
        {/* Always in the page, so screen readers announce what appears in it. */}
        <div role="status" aria-live="polite" aria-atomic="true" className="answer-text">
          {status.kind === "asking" && <p>Looking at the photo…</p>}
          {status.kind === "done" && <Result result={status.result} question={status.question} />}
        </div>
        {status.kind === "error" && (
          <p role="alert" className="error">
            <strong>Error:</strong> {status.message}
          </p>
        )}
      </section>

      <section aria-labelledby="settings-heading">
        <h2 id="settings-heading">Settings</h2>
        {canSpeak() ? (
          <label className="check">
            <input
              type="checkbox"
              checked={readAloud}
              onChange={(e) => toggleReadAloud(e.target.checked)}
            />
            Read answers aloud (leave off if you use a screen reader)
          </label>
        ) : (
          <p>This browser can't read answers aloud; your screen reader will still announce them.</p>
        )}
      </section>
    </main>
  );
}

function Result({ result, question }: { result: AskResult; question: string }) {
  if (result.unanswerable || !result.answer) {
    return (
      <>
        <p className="result-main">{CANT_TELL}</p>
        <p>Try:</p>
        <ul>
          {TIPS.map((t) => (
            <li key={t}>{t}</li>
          ))}
        </ul>
      </>
    );
  }
  return (
    <>
      <p className="result-main">
        <span className="visually-hidden">Answer: </span>
        {result.answer}
      </p>
      <p className="help">You asked: {question}</p>
    </>
  );
}
