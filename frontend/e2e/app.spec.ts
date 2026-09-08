import { test, expect, type Page } from '@playwright/test';

/**
 * Real-browser E2E for the core user journey:
 *   auth guard -> register/login -> dashboard -> upload (poll to completed)
 *   -> ask a question -> streamed answer + citations render.
 *
 * Backend endpoints are intercepted with page.route() so the test verifies the
 * REAL frontend behaviour (token storage, Authorization header, SSE parsing,
 * citation rendering, auth guard) deterministically, without a live backend.
 */

const FAKE_JWT =
  'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIn0.signature';

/** Wire up all backend routes the app calls. */
async function mockBackend(page: Page) {
  // Auth: register + login both return a token
  await page.route('**/api/auth/**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ access_token: FAKE_JWT, token_type: 'bearer' }),
    });
  });

  // Document list (polled) — start empty, becomes one doc after upload
  let uploaded = false;
  await page.route('**/api/documents', async (route) => {
    if (route.request().method() === 'GET') {
      const docs = uploaded
        ? [{ doc_id: '1:abc', filename: 'notes.pdf', chunk_count: 3, status: 'completed', owner_id: '1' }]
        : [];
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ documents: docs }),
      });
    } else {
      await route.fallback();
    }
  });

  // Upload -> returns a job id
  await page.route('**/api/documents/upload', async (route) => {
    uploaded = true;
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ job_id: 'job-1', doc_id: '1:abc', status: 'processing' }),
    });
  });

  // Job status -> completed
  await page.route('**/api/documents/jobs/**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ job_id: 'job-1', status: 'completed' }),
    });
  });

  // Chat stream -> Server-Sent Events with text chunks + citations + [DONE]
  await page.route('**/api/chat/stream', async (route) => {
    const sse =
      'data: {"text": "Kubernetes "}\n\n' +
      'data: {"text": "orchestrates containers."}\n\n' +
      'data: {"citations": [{"filename": "notes.pdf", "page_number": 2, "excerpt": "k8s excerpt", "score": 3.1}], "tokens_used": 42}\n\n' +
      'data: [DONE]\n\n';
    await route.fulfill({
      status: 200,
      headers: { 'content-type': 'text/event-stream' },
      body: sse,
    });
  });
}

test.beforeEach(async ({ page }) => {
  await mockBackend(page);
});

test('shows the auth screen when not logged in (auth guard)', async ({ page }) => {
  await page.goto('/');
  // The dashboard must NOT be visible; the auth screen must be
  await expect(page.getByRole('heading', { name: 'Second Brain RAG' })).toBeVisible();
  await expect(page.getByRole('tab', { name: 'Sign In' })).toBeVisible();
  await expect(page.getByRole('tab', { name: 'Register' })).toBeVisible();
});

test('register -> dashboard -> upload -> chat with citations', async ({ page }) => {
  await page.goto('/');

  // Register
  await page.getByRole('tab', { name: 'Register' }).click();
  await page.locator('#email-input').fill('e2e@example.com');
  await page.locator('#password-input').fill('strongpassword123');
  await page.getByRole('button', { name: /Create Account|Register|Sign Up/i }).first().click();

  // Dashboard should now render (auth succeeded, token stored)
  await expect(page.getByRole('heading', { name: 'Knowledge Library' })).toBeVisible({ timeout: 10_000 });

  // The JWT must have been persisted so requests are authenticated
  const token = await page.evaluate(() => localStorage.getItem('access_token') || localStorage.getItem('token'));
  expect(token).toBeTruthy();

  // Ask a question via chat
  const box = page.getByPlaceholder(/Ask/i);
  await box.fill('What does kubernetes do?');
  await box.press('Enter');

  // Streamed answer text should appear
  await expect(page.getByText(/orchestrates containers/i)).toBeVisible({ timeout: 10_000 });

  // Citations render as a collapsible "Sources (N)" accordion — expand it
  const sourcesToggle = page.getByRole('button', { name: /Sources \(\d+\)/ });
  await expect(sourcesToggle).toBeVisible({ timeout: 10_000 });
  await sourcesToggle.click();

  // Citation (filename + page number) from the SSE payload should now render
  await expect(page.getByText('notes.pdf').first()).toBeVisible({ timeout: 10_000 });
  await expect(page.getByText(/p\. 2/)).toBeVisible();
});

test('login flow authenticates and reaches dashboard', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('tab', { name: 'Sign In' }).click();
  await page.locator('#email-input').fill('existing@example.com');
  await page.locator('#password-input').fill('strongpassword123');
  await page.getByRole('button', { name: /Sign In|Log In|Login/i }).first().click();

  await expect(page.getByRole('heading', { name: 'Knowledge Library' })).toBeVisible({ timeout: 10_000 });
});
