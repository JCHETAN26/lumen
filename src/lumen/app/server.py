"""Lumen's local answer server: one photo and one question in, a short answer (or "can't tell") out.

Privacy and safety, by construction:
- Binds to 127.0.0.1 only and rejects any other Host header (stops DNS-rebinding pages from
  reaching it). Every API call must carry the `X-Lumen` header, which a cross-site form can't
  send, and no CORS is granted, so other websites can't use it.
- Photos and questions are processed in memory and never written, logged, or uploaded; access
  logs are off. Answers are returned as plain JSON text, and the page renders them as text.
- One question at a time (an 8 GB Mac runs one model).

The same prompt, decoding, answer extractor, and P(unanswerable) as every evaluated run. The
"can't tell" decision is the model answering "unanswerable", or, with `--threshold` (the
answerability threshold chosen on dev in Stage 3), P(unanswerable) above it.

    uv run --group app --group mlx python -m lumen.app.server --model lr2e-4-e1 --variant 4bit
    uv run --group app python -m lumen.app.server --fake     # no model: UI development and tests
"""

from __future__ import annotations

import argparse
import io
import threading
import time
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from lumen.zeroshot.extract import UNANSWERABLE, extract_answer
from lumen.zeroshot.prompts import chat_messages

HOST = "127.0.0.1"
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
MAX_IMAGE_BYTES = 20 * 2**20
MAX_QUESTION_CHARS = 500
REQUIRED_HEADER = "x-lumen"
WEB_DIST = Path(__file__).resolve().parents[3] / "app" / "web" / "dist"

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; "
        "style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; "
        "base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(self), microphone=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}


def is_unanswerable(answer: str, p_unanswerable: float, threshold: float | None) -> bool:
    if answer.strip().lower() == UNANSWERABLE:
        return True
    return threshold is not None and p_unanswerable > threshold


def read_photo(data: bytes) -> Image.Image:
    """Decode an uploaded photo, upright (phone photos store rotation in EXIF), in RGB."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            return ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise ValueError("that file isn't a photo Lumen can read (try a JPEG or PNG)") from e


def create_app(
    backend: Any,
    *,
    prompt_key: str,
    threshold: float | None = None,
    model_info: dict[str, Any] | None = None,
    static_dir: Path | None = WEB_DIST,
) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    lock = threading.Lock()

    def error(status: int, message: str) -> JSONResponse:
        return JSONResponse({"error": message}, status_code=status)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = request.headers.get("host", "")
        name = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
        if name not in ALLOWED_HOSTS:
            return error(403, "Lumen only answers on this computer (localhost)")
        if request.url.path.startswith("/api/") and REQUIRED_HEADER not in request.headers:
            return error(400, "missing X-Lumen header")
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ready": True,
            "model": model_info or {},
            "unanswerable_threshold": threshold,
            "max_image_bytes": MAX_IMAGE_BYTES,
            "max_question_chars": MAX_QUESTION_CHARS,
        }

    @app.post("/api/ask")
    def ask(image: Annotated[UploadFile, File()], question: Annotated[str, Form()]):
        question = " ".join(question.split())
        if not question:
            return error(422, "please type a question")
        if len(question) > MAX_QUESTION_CHARS:
            return error(422, f"please keep the question under {MAX_QUESTION_CHARS} characters")
        data = image.file.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            return error(413, f"the photo is over {MAX_IMAGE_BYTES // 2**20} MB")
        try:
            photo = read_photo(data)
        except ValueError as e:
            return error(415, str(e))
        del data
        t0 = time.perf_counter()
        with lock:
            out = backend.run_batch([photo], [chat_messages(question, prompt_key)])[0]
        answer = extract_answer(out.raw).answer
        unans = is_unanswerable(answer, out.p_unanswerable, threshold)
        return {
            "answer": None if unans else answer,
            "unanswerable": unans,
            "p_unanswerable": round(float(out.p_unanswerable), 4),
            "seconds": round(time.perf_counter() - t0, 2),
        }

    if static_dir is not None and static_dir.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=static_dir, html=True), name="web")
    return app


def main() -> int:
    ap = argparse.ArgumentParser(description="Lumen's local answer server (localhost only)")
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--model", help="an adapter key in configs/pins.yaml, or 'base'")
    who.add_argument("--fake", action="store_true", help="no model: a fixed fake answerer")
    ap.add_argument("--variant", default="4bit", choices=["fp16", "8bit", "4bit"])
    ap.add_argument("--threshold", type=float, default=None, help="dev-chosen P(unanswerable)")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    import uvicorn

    if args.fake:
        from lumen.zeroshot.backends import FakeBackend

        backend, prompt_key, info = FakeBackend(), "abstain", {"fake": True}
    else:
        from lumen.ondevice.models import load_backend

        backend, s, load_s, _ = load_backend(args.model, args.variant)
        prompt_key = s["prompt_key"]
        info = {"key": args.model, "variant": args.variant, "load_seconds": round(load_s, 1)}
    if not WEB_DIST.is_dir():
        print(f"note: {WEB_DIST} not built; run `npm run build` in app/web (API only for now)")
    app = create_app(backend, prompt_key=prompt_key, threshold=args.threshold, model_info=info)
    print(f"Lumen is running at http://localhost:{args.port} (this computer only)")
    uvicorn.run(app, host=HOST, port=args.port, access_log=False, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
