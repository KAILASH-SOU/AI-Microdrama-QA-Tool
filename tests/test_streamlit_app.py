"""Automated test for Streamlit app rendering and session flow."""

from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_without_exceptions(monkeypatch):
    """Verifies that app.py compiles, runs end-to-end, and renders all tabs without exceptions."""
    # Use mock provider so AppTest doesn't trigger live network calls in unit testing
    monkeypatch.setenv("MOCK_MODE", "true")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=30)

    # Assert no uncaught exceptions in the app run
    assert len(at.exception) == 0, f"Streamlit app raised exceptions: {at.exception}"

    # Verify key UI components exist
    assert len(at.tabs) == 4
    assert len(at.sidebar.button) >= 1


def test_streamlit_app_switch_to_upload_mode():
    """Verifies that switching to upload mode without files doesn't throw KeyError or exceptions."""
    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=10)
    assert len(at.exception) == 0

    # Select upload mode in screenplay source radio
    source_radio = next(r for r in at.sidebar.radio if "Screenplay Source" in r.label)
    source_radio.set_value("Upload Screenplays (.zip or .txt)").run(timeout=10)
    assert len(at.exception) == 0


def test_streamlit_app_dark_theme_css():
    """Verifies that the dedicated dark theme styling renders cleanly."""
    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=10)
    assert len(at.exception) == 0
    # Confirm dark mode styling is injected
    assert any("#0E131F" in str(m.value) for m in at.markdown)



