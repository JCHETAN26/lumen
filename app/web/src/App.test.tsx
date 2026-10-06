import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import axe from "axe-core";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { NOT_RUNNING } from "./api";
import { CANT_TELL, TIPS } from "./messages";

async function expectNoAxeViolations(container: HTMLElement) {
  const results = await axe.run(container, { rules: { "color-contrast": { enabled: false } } });
  expect(results.violations.map((v) => `${v.id}: ${v.help}`)).toEqual([]);
}

function reply(body: object, status = 200) {
  return vi.fn(async () => new Response(JSON.stringify(body), { status }));
}

const photo = () => new File([new Uint8Array([1, 2, 3])], "photo.jpg", { type: "image/jpeg" });

beforeEach(() => {
  URL.createObjectURL = vi.fn(() => "blob:preview");
  URL.revokeObjectURL = vi.fn();
  window.localStorage.clear();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

async function askAbout(question: string) {
  const user = userEvent.setup();
  await user.upload(screen.getByLabelText("Choose a photo"), photo());
  await user.type(screen.getByRole("textbox", { name: "Your question" }), question);
  await user.click(screen.getByRole("button", { name: "Ask" }));
  return user;
}

describe("Lumen", () => {
  it("has a clear structure and no axe violations", async () => {
    const { container } = render(<App />);
    expect(screen.getByRole("heading", { level: 1, name: "Lumen" })).toBeInTheDocument();
    for (const name of ["1. Photo", "2. Question", "Answer", "Settings"]) {
      expect(screen.getByRole("heading", { level: 2, name })).toBeInTheDocument();
    }
    expect(screen.getByRole("main")).toBeInTheDocument();
    await expectNoAxeViolations(container);
  });

  it("asks for a photo first and moves focus to it", async () => {
    const fetch = reply({});
    vi.stubGlobal("fetch", fetch);
    render(<App />);
    await userEvent.setup().click(screen.getByRole("button", { name: "Ask" }));
    expect(screen.getByRole("alert")).toHaveTextContent("Error: Choose or take a photo first.");
    expect(screen.getByLabelText("Choose a photo")).toHaveFocus();
    expect(fetch).not.toHaveBeenCalled();
  });

  it("asks for a question and marks the field invalid", async () => {
    render(<App />);
    const user = userEvent.setup();
    await user.upload(screen.getByLabelText("Choose a photo"), photo());
    expect(screen.getByText("Photo ready.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Ask" }));
    const box = screen.getByRole("textbox", { name: "Your question" });
    expect(box).toHaveFocus();
    expect(box).toHaveAttribute("aria-invalid", "true");
    expect(box).toHaveAccessibleDescription(/Type a question about the photo/);
  });

  it("announces the answer in a live region", async () => {
    const fetch = reply({ answer: "dog", unanswerable: false, p_unanswerable: 0.1, seconds: 1.2 });
    vi.stubGlobal("fetch", fetch);
    const { container } = render(<App />);
    await askAbout("What is this?");
    const live = container.querySelector('[aria-live="polite"]')!;
    await waitFor(() => expect(live).toHaveTextContent("Answer: dog"));
    const [url, init] = fetch.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/ask");
    expect(init.headers).toEqual({ "X-Lumen": "1" });
    expect((init.body as FormData).get("question")).toBe("What is this?");
    await expectNoAxeViolations(container);
  });

  it("says it can't tell, with tips, when the photo can't answer", async () => {
    vi.stubGlobal(
      "fetch",
      reply({ answer: null, unanswerable: true, p_unanswerable: 0.9, seconds: 1 }),
    );
    render(<App />);
    await askAbout("What is this?");
    await waitFor(() => expect(screen.getByText(CANT_TELL)).toBeInTheDocument());
    for (const tip of TIPS) expect(screen.getByText(tip)).toBeInTheDocument();
  });

  it("explains when the answer server isn't running", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new TypeError("failed"))));
    render(<App />);
    await askAbout("What is this?");
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(NOT_RUNNING));
  });

  it("shows the server's own error message", async () => {
    vi.stubGlobal("fetch", reply({ error: "the photo is over 20 MB" }, 413));
    render(<App />);
    await askAbout("What is this?");
    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent("Error: The photo is over 20 MB."),
    );
  });

  it("reads answers aloud only when turned on, and remembers the choice", async () => {
    const speak = vi.fn();
    vi.stubGlobal("speechSynthesis", { speak, cancel: vi.fn() });
    vi.stubGlobal("SpeechSynthesisUtterance", function (this: { text: string }, t: string) {
      this.text = t;
    });
    vi.stubGlobal(
      "fetch",
      reply({ answer: "dog", unanswerable: false, p_unanswerable: 0.1, seconds: 1 }),
    );
    render(<App />);
    const toggle = screen.getByRole("checkbox", { name: /Read answers aloud/ });
    expect(toggle).not.toBeChecked();
    await userEvent.setup().click(toggle);
    expect(window.localStorage.getItem("lumen.readAloud")).toBe("1");
    await askAbout("What is this?");
    await waitFor(() => expect(speak).toHaveBeenCalledTimes(1));
    expect((speak.mock.calls[0][0] as { text: string }).text).toBe("Answer: dog");
  });

  it("offers the camera when the browser has one", () => {
    vi.stubGlobal("navigator", { ...navigator, mediaDevices: { getUserMedia: vi.fn() } });
    render(<App />);
    expect(screen.getByRole("button", { name: "Take a photo" })).toBeInTheDocument();
  });
});
