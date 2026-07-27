"""Contracts for LEMMA's public evidence surface.

These tests keep visual polish subordinate to product truth: the cinematic
hero must remain self-contained, motion-safe, and must not rename the DOM
hooks used by the upload, ask, and live-eval behavior.
"""

from pathlib import Path
import struct
import zlib


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
HANDBOOK = (ROOT / "sample_docs" / "lemma-handbook.md").read_text(encoding="utf-8")
WORKFLOW = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
VERIFY_SKILL = (ROOT / ".claude" / "skills" / "lemma-verify" / "SKILL.md").read_text(
    encoding="utf-8"
)
DEV_REQUIREMENTS = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
CARD = ROOT / "static" / "lemma-social-card.png"
CARD_URL = "https://lemma-production-b84f.up.railway.app/lemma-social-card.png"
CARD_ALT = (
    "LEMMA evidence card: dense and sparse retrieval merge with reciprocal rank "
    "fusion; bounded re-planning may run for up to three hops before exact cited "
    "passages, while a separate write-retrieve-verify-clean-up recall probe "
    "reports pass or fail."
)


def test_cinematic_hero_explains_the_proof_pipeline():
    assert '<header class="hero">' in HTML
    assert 'class="proof-constellation" aria-hidden="true"' in HTML
    assert '<ol class="proof-rail" aria-label="How LEMMA proves an answer">' in HTML
    for step in ("Retrieve", "Re-plan", "Cite", "Evaluate"):
        assert f"<b>{step}</b>" in HTML


def test_cinematic_motion_has_a_reduced_motion_contract():
    assert "@media (prefers-reduced-motion: reduce)" in HTML
    assert "animation: none !important" in HTML
    assert "scroll-behavior: auto" in HTML


def test_surface_remains_self_contained():
    assert "<script src=" not in HTML
    assert 'rel="stylesheet"' not in HTML
    assert "url(http" not in HTML


def test_live_behavior_hooks_survive_visual_work():
    for element_id in (
        "modelBadge",
        "file",
        "uploadBtn",
        "upStatus",
        "docs",
        "chat",
        "messages",
        "question",
        "askBtn",
        "evalBtn",
        "llmCheck",
        "evalResults",
    ):
        assert f'id="{element_id}"' in HTML


def test_document_discovery_metadata_stays_present():
    assert '<html lang="en">' in HTML
    assert 'name="viewport"' in HTML
    assert '<meta name="description"' in HTML
    assert '<meta name="theme-color" content="#070a10">' in HTML
    assert 'rel="icon"' in HTML
    assert 'href="data:image/svg+xml,' in HTML


def test_social_card_is_a_real_1200_by_630_png():
    data = CARD.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(data) >= 100_000

    offset = 8
    chunks: list[bytes] = []
    while offset < len(data):
        assert offset + 12 <= len(data)
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_end = offset + 12 + length
        assert chunk_end <= len(data)
        payload = data[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", data[offset + 8 + length : chunk_end])[0]
        assert zlib.crc32(chunk_type + payload) & 0xFFFFFFFF == expected_crc
        chunks.append(chunk_type)
        if chunk_type == b"IHDR":
            assert length == 13
            assert struct.unpack(">II", payload[:8]) == (1200, 630)
        offset = chunk_end

    assert offset == len(data)
    assert chunks[0] == b"IHDR"
    assert b"IDAT" in chunks
    assert chunks[-1] == b"IEND"


def test_social_card_metadata_and_readme_are_complete():
    assert (
        '<link rel="canonical" '
        'href="https://lemma-production-b84f.up.railway.app/">' in HTML
    )
    expected = {
        "og:image": CARD_URL,
        "og:image:secure_url": CARD_URL,
        "og:image:type": "image/png",
        "og:image:width": "1200",
        "og:image:height": "630",
        "og:image:alt": CARD_ALT,
        "twitter:image": CARD_URL,
        "twitter:image:alt": CARD_ALT,
    }
    for field, value in expected.items():
        key = "name" if field.startswith("twitter:") else "property"
        assert f'<meta {key}="{field}" content="{value}">' in HTML
    assert '<meta name="twitter:card" content="summary_large_image">' in HTML
    assert (
        "![LEMMA evidence card](static/lemma-social-card.png)" in README
    )
    assert "proves its answers" not in HTML.lower()
    assert "proves</em> its answers" not in README
    assert "when citations are absent, the interface marks the answer" in README
    assert "marks it unverified instead of" in HANDBOOK
    for anchor in ("#architecture", "#why-this-project-exists", "#quickstart"):
        assert f'href="{anchor}"' in README


def test_social_card_route_serves_the_exact_asset():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    try:
        response = client.get("/lemma-social-card.png")
    finally:
        client.close()
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.content == CARD.read_bytes()


def test_release_workflow_and_verification_instructions_are_current():
    assert "workflow_dispatch:" in WORKFLOW
    assert "permissions:\n  contents: read" in WORKFLOW
    assert "actions/checkout@v5" in WORKFLOW
    assert "actions/setup-python@v6" in WORKFLOW
    assert "pip install -r requirements-dev.txt" in WORKFLOW
    assert "python -W error -m pytest -q" in WORKFLOW
    assert "httpx2>=2.7,<3" in DEV_REQUIREMENTS
    assert "23 tests" not in VERIFY_SKILL
    assert ".venv\\Scripts\\python.exe -W error -m pytest -q" in VERIFY_SKILL
    assert ".venv/Scripts/python -W error -m pytest -q" in README
    assert "GET /lemma-social-card.png" in VERIFY_SKILL
    assert "1200×630" in VERIFY_SKILL
    assert "repository-ready, not deployed" in VERIFY_SKILL
