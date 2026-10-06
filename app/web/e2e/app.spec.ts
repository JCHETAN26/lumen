import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

// A tiny valid PNG (1x1), so the real server decodes a real image.
const PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC",
  "base64",
);

async function axeClean(page: Page) {
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"])
    .analyze();
  expect(results.violations.map((v) => `${v.id}: ${v.help}`)).toEqual([]);
}

async function choosePhoto(page: Page) {
  await page
    .getByLabel("Choose a photo")
    .setInputFiles({ name: "photo.png", mimeType: "image/png", buffer: PNG });
  await expect(page.getByText("Photo ready.")).toBeVisible();
}

for (const colorScheme of ["light", "dark"] as const) {
  test(`answers a question, accessibly (${colorScheme})`, async ({ page }) => {
    await page.emulateMedia({ colorScheme });
    await page.goto("/");
    await axeClean(page);
    await choosePhoto(page);
    await page.getByRole("textbox", { name: "Your question" }).fill("What is this?");
    await page.getByRole("button", { name: "Ask" }).click();
    await expect(page.getByRole("status").filter({ hasText: "Answer:" })).toContainText("dog");
    await axeClean(page);
  });
}

test("says it can't tell, with tips", async ({ page }) => {
  await page.goto("/");
  await choosePhoto(page);
  await page.getByRole("textbox", { name: "Your question" }).fill("Is this too blurry to read?");
  await page.getByRole("button", { name: "Ask" }).click();
  await expect(page.getByText("I can't tell from this photo.")).toBeVisible();
  await expect(page.getByText("Add more light, or turn toward a window.")).toBeVisible();
  await axeClean(page);
});

test("works with the keyboard alone", async ({ page }) => {
  await page.goto("/");
  await page.keyboard.press("Tab");
  await expect(page.getByRole("button", { name: "Take a photo" })).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(page.getByLabel("Choose a photo")).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(page.getByRole("textbox", { name: "Your question" })).toBeFocused();
  await page.keyboard.type("What is this?");
  await page.keyboard.press("Tab");
  await expect(page.getByRole("button", { name: "Ask" })).toBeFocused();
  await page.keyboard.press("Enter");
  const alert = page.getByRole("alert");
  await expect(alert).toContainText("Choose or take a photo first.");
  // Focus goes to the first way of fixing the problem.
  await expect(page.getByRole("button", { name: "Take a photo" })).toBeFocused();
});

test("the API refuses calls without the X-Lumen header", async ({ request }) => {
  const r = await request.get("/api/health");
  expect(r.status()).toBe(400);
  const ok = await request.get("/api/health", { headers: { "X-Lumen": "1" } });
  expect((await ok.json()).ready).toBe(true);
});

test("takes a photo with the camera, keyboard only", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Take a photo" }).focus();
  await page.keyboard.press("Enter");
  await expect(page.getByText("Camera on.")).toBeVisible();
  const shutter = page.getByRole("button", { name: "Take photo" });
  await expect(shutter).toBeFocused(); // ready to press at once
  await axeClean(page);
  await page.keyboard.press("Enter");
  await expect(page.getByText("Photo ready.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Take a photo" })).toBeFocused();
  await page.getByRole("textbox", { name: "Your question" }).fill("What is this?");
  await page.getByRole("button", { name: "Ask" }).click();
  await expect(page.getByRole("status").filter({ hasText: "Answer:" })).toContainText("dog");
});
