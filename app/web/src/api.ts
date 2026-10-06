// The local answer server (lumen.app.server). Same origin; every call carries X-Lumen, which the
// server requires so that other websites can't use it.

export interface AskResult {
  answer: string | null;
  unanswerable: boolean;
  p_unanswerable: number;
  seconds: number;
}

export class AskError extends Error {}

export const NOT_RUNNING =
  "Lumen's answer server isn't running on this computer. Start it, then ask again.";

export async function ask(photo: Blob, question: string): Promise<AskResult> {
  const form = new FormData();
  form.append("image", photo, "photo");
  form.append("question", question);
  let res: Response;
  try {
    res = await fetch("/api/ask", { method: "POST", body: form, headers: { "X-Lumen": "1" } });
  } catch {
    throw new AskError(NOT_RUNNING);
  }
  const body = (await res.json().catch(() => ({}))) as Partial<AskResult> & { error?: string };
  if (!res.ok) {
    throw new AskError(body.error ? capitalize(body.error) : `Something went wrong (${res.status}).`);
  }
  return body as AskResult;
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1) + (/[.!?]$/.test(s) ? "" : ".");
}
