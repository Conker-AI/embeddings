"""Image and compose configuration tests."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_the_image_does_not_run_as_root():
    """Read Dockerfile and verify the last USER line exists and is not root."""
    with open(ROOT / "Dockerfile", encoding="utf-8") as f:
        lines = f.readlines()

    user_lines = [line.strip() for line in lines if line.strip().startswith("USER ")]
    assert user_lines, "No USER instruction found in Dockerfile"

    last_user = user_lines[-1]
    user_spec = last_user.split(" ", 1)[1]  # Remove "USER " prefix

    # Check it's not root or uid 0
    assert user_spec not in ("root", "0"), f"Image runs as {user_spec}, should be non-root"
    assert not user_spec.startswith("0:"), f"Image runs as {user_spec}, should be non-root"


def test_the_dev_ollama_is_pinned_by_digest():
    """Read docker-compose.yml and verify the ollama service uses a digest."""
    with open(ROOT / "docker-compose.yml", encoding="utf-8") as f:
        content = f.read()

    # Find the ollama service section
    lines = content.split("\n")
    in_ollama = False
    image_line = None

    for line in lines:
        stripped = line.strip()
        if stripped == "ollama:":
            in_ollama = True
            continue
        if in_ollama and stripped.startswith("image:"):
            image_line = stripped
            break
        if in_ollama and not line.startswith("  ") and stripped:
            # We've moved to the next service
            break

    assert image_line, "No image line found in ollama service"
    assert "@sha256:" in image_line, f"Ollama image not pinned by digest: {image_line}"
