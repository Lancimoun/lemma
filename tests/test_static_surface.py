"""Contracts for LEMMA's public evidence surface.

These tests keep visual polish subordinate to product truth: the cinematic
hero must remain self-contained, motion-safe, and must not rename the DOM
hooks used by the upload, ask, and live-eval behavior.
"""

from pathlib import Path


HTML = (Path(__file__).resolve().parents[1] / "static" / "index.html").read_text(
    encoding="utf-8"
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
