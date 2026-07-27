---
name: lemma-verify
description: Verify LEMMA actually works end-to-end — run the test suite, boot the real server, and probe the live endpoints. Use this whenever LEMMA's code changes (before any commit or push), after pulling, before or after a Railway deploy, or whenever anyone asks "does lemma work", "is lemma healthy", "did the change break anything", or wants proof the demo is up. Don't settle for "tests pass" alone — this skill exists because a green suite with a dead server is still a broken product.
---

# lemma-verify

Prove LEMMA works at three levels: unit (pytest), process (server boots), and product (endpoints answer). Run all three — each catches failures the others can't.

## 1 · Test suite (~5s, no network, no API key)

```
.venv\Scripts\python.exe -W error -m pytest -q        # Windows
.venv/bin/python -W error -m pytest -q                # POSIX
```

Expect **the current count printed by pytest to be all green, with no warnings**. Tests inject fake embedders, so a pass here says the logic is right — not that the real models or server work. Any failure or warning: stop, fix, rerun before proceeding.

## 2 · Boot the real server

```
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8765
```

Run it in the background with `LEMMA_DATA_DIR` set to a new isolated temporary
directory, then poll `http://127.0.0.1:8765/health` until it answers (up to ~60s
on a cold cache — see Gotchas). Use an uncommon port (8765) so a dev instance
on 8000 doesn't collide. Record the spawned PID and stop only that process.

## 3 · Probe the product surface

| Probe | Expect |
|---|---|
| `GET /health` | `status: "ok"`, a `public_badge` string (never a raw model id), `documents ≥ 1` (sample doc auto-seeds) |
| `GET /documents` | JSON list including the seeded handbook |
| `GET /` | `200`; canonical plus complete Open Graph/Twitter image metadata point to `/lemma-social-card.png` |
| `GET /lemma-social-card.png` | `200 image/png`; 1200×630; bytes match `static/lemma-social-card.png` |
| `POST /ask` (only if `ANTHROPIC_API_KEY` is set) | Grounded answer with `sources`; without a key, a graceful error is CORRECT behavior, not a failure |

For a release, compare the same read-only GETs against
`https://lemma-production-b84f.up.railway.app`. A branch candidate may correctly
be newer than production; report that split as **repository-ready, not deployed**.
After an authorized deployment, production must serve the new root metadata and
the exact PNG bytes as well as a healthy `/health`.

**Always kill the server when done** and report a pass/fail line per level, with the failing output verbatim if anything failed.

## Gotchas (each earned from a real failure)

- **System Python has no pytest.** The suite only runs via `.venv` — `python -m pytest` at global scope fails with "No module named pytest".
- **First boot downloads embedding weights** (~100MB to `%TEMP%\fastembed_cache`) — a "hung" server on first run is usually just downloading. Windows symlink warnings during this are benign.
- **`.env` does not autoload** — python-dotenv isn't wired. `ANTHROPIC_API_KEY` must be exported in the environment or `/ask` runs keyless (gracefully).
- **429s during probing are the rate limiter working** (`/ingest`, `/ask`, `DELETE /documents` are throttled per-IP) — don't report them as product failures.
- **Prod can lag `main`** — Railway isn't wired to auto-deploy from GitHub, so a healthy-but-stale prod is a deploy task, not a code bug.
