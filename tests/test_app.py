"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden Slider an seine
Grenzen, wählt jede Option jeder Auswahl und prüft, dass kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere st.plotly_chart-Aufrufe
ohne eindeutiges key=) und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    at = _fresh_app()
    assert at.title and at.markdown


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_every_option_of_every_selection_does_not_raise():
    """select_slider (Grenzen) und selectbox (Kosten, Start, Terminalmenge): jede Option einzeln."""
    base = _fresh_app()
    for kind in ("select_slider", "selectbox"):
        for index in range(len(getattr(base, kind))):
            for opt_index in range(len(getattr(base, kind)[index].options)):
                at = _fresh_app()
                w = getattr(at, kind)[index]
                w.set_value(w.options[opt_index]).run(timeout=TIMEOUT)
                assert not at.exception, (kind, index, opt_index, [str(e) for e in at.exception])


def test_checkbox_and_seed_input_do_not_raise():
    at = _fresh_app()
    at.checkbox[0].set_value(True).run(timeout=TIMEOUT)
    assert not at.exception
    at = _fresh_app()
    at.number_input[0].set_value(0).run(timeout=TIMEOUT)
    assert not at.exception
    at = _fresh_app()
    at.number_input[0].set_value(999999).run(timeout=TIMEOUT)
    assert not at.exception


def test_every_verdict_box_is_present_and_free_of_unresolved_placeholders():
    at = _fresh_app()
    texts = [m.value for kind in ("info", "success", "warning", "error") for m in getattr(at, kind)]
    assert len(texts) >= 8                       # fünf Experiment-Verdikte und drei Kernfragen
    joined = " ".join(texts)
    for bad in ("nan", "None", "TODO", "{"):
        assert bad not in joined, bad
