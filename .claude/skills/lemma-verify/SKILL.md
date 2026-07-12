---
name: lemma-verify
description: Verify LEMMA actually works end-to-end — run the test suite, boot the real server, and probe the live endpoints. Use this whenever LEMMA's code changes (before any commit or push), after pulling, before or after a Railway deploy, or whenever anyone asks "does lemma work", "is lemma healthy", "did the change break anything", or wants proof the demo is up. Don't settle for "tests pass" alone — this skill exists because a green suite with a dead server is still a broken product.
---

# lemma-verify

Prove LEMMA works at three levels: unit (pytest), process (server boots), and product (endpoints answer). Run all three — each catches failures the others can't.

## 1 · Test suite (~5s, no network, no API key)

```
.venv\Scripts\python.exe -m pytest -q        # Windows
.venv/bin/python -m pytest -q                # POSIX
```

Expect **all green** (23 tests as of 2026-07). Tests inject fake embedders, so a pass here says the logic is right — not that the real models or server work. Any failure: stop, fix, rerun before proceeding.

## 2 · Boot the real server

```
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8765
```

Run it in the background and poll `http://127.0.0.1:8765/health` until it answers (up to ~60s on a cold cache — see Gotchas). Use an uncommon port (8765) so a dev instance on 8000 doesn't collide.

## 3 · Probe the product surface

| Probe | Expect |
|---|---|
| `GET /health` | `status: "ok"`, a `public_badge` string (never a raw model id), `documents ≥ 1` (sample doc auto-seeds) |
| `GET /documents` | JSON list including the seeded handbook |
| `POST /ask` (only if `ANTHROPIC_API_KEY` is set) | Grounded answer with `sources`; without a key, a graceful error is CORRECT behavior, not a failure |

Optionally compare prod: `GET https://lemma-production-b84f.up.railway.app/health` — if its response is missing fields that local has (e.g. `public_badge`), prod is running an older build; report the lag, don't "fix" the code.

**Always kill the server when done** and report a pass/fail line per level, with the failing output verbatim if anything failed.

## Gotchas (each earned from a real failure)

- **System Python has no pytest.** The suite only runs via `.venv` — `python -m pytest` at global scope fails with "No module named pytest".
- **First boot downloads embedding weights** (~100MB to `%TEMP%\fastembed_cache`) — a "hung" server on first run is usually just downloading. Windows symlink warnings during this are benign.
- **`.env` does not autoload** — python-dotenv isn't wired. `ANTHROPIC_API_KEY` must be exported in the environment or `/ask` runs keyless (gracefully).
- **429s during probing are the rate limiter working** (`/ingest`, `/ask`, `DELETE /documents` are throttled per-IP) — don't report them as product failures.
- **Prod can lag `main`** — Railway isn't wired to auto-deploy from GitHub, so a healthy-but-stale prod is a deploy task, not a code bug.
