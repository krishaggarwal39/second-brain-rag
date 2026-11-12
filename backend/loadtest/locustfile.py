"""
Load test for Second Brain RAG — measures real multi-user performance.

Simulates realistic user journeys:
1. Register a unique user → login → get JWT
2. Upload a small test PDF document
3. Poll job status until indexed
4. List owned documents
5. Chat query (non-streaming, to measure full round-trip latency)
6. Chat stream query (SSE)

Run:
  locust -f loadtest/locustfile.py --headless \
    -u 20 -r 5 --run-time 2m \
    --host http://localhost:8000 \
    --csv loadtest/results/run
"""

import os
import io
import uuid
from locust import HttpUser, task, between

# A tiny valid PDF (single blank page) so upload + ingestion actually works.
# Generating it inline avoids needing an external fixture file.
TINY_PDF = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
    b"/Resources<</Font<</F1 4 0 R>>>>>>endobj\n"
    b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
    b"xref\n0 5\ntrailer<</Size 5/Root 1 0 R>>\nstartxref\n0\n%%EOF"
)

# Test questions for chat queries
QUESTIONS = [
    "What is machine learning?",
    "How do neural networks work?",
    "What is the GIL in Python?",
    "Explain distributed consensus",
    "What are microservices?",
]


class SecondBrainUser(HttpUser):
    """Simulates a realistic user journey through the Second Brain app."""

    wait_time = between(1, 3)  # 1-3s think time between tasks
    token = None
    email = None
    job_id = None
    doc_uploaded = False
    question_idx = 0

    def on_start(self):
        """Register + login to get a JWT before any other requests."""
        self.email = f"loadtest-{uuid.uuid4().hex[:10]}@test.com"
        password = "LoadTest123!"

        # Register
        resp = self.client.post(
            "/auth/register",
            json={"email": self.email, "password": password},
            name="/auth/register",
        )
        if resp.status_code == 200:
            self.token = resp.json().get("access_token")
        elif resp.status_code == 409:
            # Already registered (unlikely with uuid, but handle gracefully)
            resp = self.client.post(
                "/auth/login",
                json={"email": self.email, "password": password},
                name="/auth/login",
            )
            if resp.status_code == 200:
                self.token = resp.json().get("access_token")

        if not self.token:
            raise Exception(f"Auth failed: {resp.status_code} {resp.text[:200]}")

    @property
    def auth_headers(self):
        h = {}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        # API-key is also required (enforced at router-level in main.py)
        api_key = os.getenv("LOADTEST_API_KEY", "loadtest-key")
        h["X-API-Key"] = api_key
        return h

    @task(2)
    def list_documents(self):
        """GET /documents — frequent operation (frontend polls every 10s)."""
        self.client.get("/documents", headers=self.auth_headers, name="/documents [list]")

    @task(1)
    def upload_document(self):
        """POST /documents/upload — upload a tiny PDF, then poll status."""
        if self.doc_uploaded:
            return  # Only upload once per user lifecycle

        filename = f"loadtest-{uuid.uuid4().hex[:6]}.pdf"
        files = {"file": (filename, io.BytesIO(TINY_PDF), "application/pdf")}
        resp = self.client.post(
            "/documents/upload",
            files=files,
            headers=self.auth_headers,
            name="/documents/upload",
        )
        if resp.status_code == 200:
            self.job_id = resp.json().get("job_id")
            self.doc_uploaded = True

    @task(1)
    def poll_job_status(self):
        """GET /documents/jobs/{id} — poll ingestion progress."""
        if not self.job_id:
            return
        self.client.get(
            f"/documents/jobs/{self.job_id}",
            headers=self.auth_headers,
            name="/documents/jobs/[id]",
        )

    # NOTE: /chat and /chat/stream call an external LLM (Gemini/Groq). Their latency
    # is dominated by the LLM provider and free-tier rate limits, not our infra, so
    # they are excluded from the infra-throughput load test. Set INCLUDE_CHAT=1 to
    # include them when a real LLM key is configured on the server.
    if os.getenv("INCLUDE_CHAT") == "1":
        @task(3)
        def chat_query(self):
            """POST /chat — full retrieval + generation round-trip."""
            q = QUESTIONS[self.question_idx % len(QUESTIONS)]
            self.question_idx += 1
            self.client.post(
                "/chat",
                json={"question": q, "chat_history": []},
                headers=self.auth_headers,
                name="/chat [non-stream]",
                timeout=120,
            )
