"""Permalink (Begrenzen, Einrasten), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben) und Zufalls-Seed-Button.

Alle Regler stehen in SETTING_SPECS (auch die im Hauptbereich beim Experiment): ein Preset legt nur fest, was es braucht, alles Übrige gilt als Standardwert
(`apply_preset` schreibt für jeden Regler `preset.get(key, Standard)`). Der Test `tests/test_layout.py` prüft, dass jeder Regler der Seite hier steht und seine Änderung
die Markierung des Presets aufhebt."""

import math
import time
from dataclasses import dataclass

import streamlit as st

import mzl_constants as C
from mzl_rng import SplitMix64


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None      # erlaubte Werte (Auswahlregler); Permalink rastet darauf ein


SETTING_SPECS = {
    "umax": SettingSpec("umax", float, C.DEFAULT_UMAX, min(C.UMAX_OPTIONS), max(C.UMAX_OPTIONS), C.UMAX_OPTIONS),
    "s1": SettingSpec("s1", float, C.DEFAULT_S1, min(C.S_OPTIONS), max(C.S_OPTIONS), C.S_OPTIONS),
    "s2": SettingSpec("s2", float, C.DEFAULT_S2, min(C.S_OPTIONS), max(C.S_OPTIONS), C.S_OPTIONS),
    "sk_N": SettingSpec("skn", int, C.SK_DEFAULT_N, C.SK_N_MIN, C.SK_N_MAX),
    "sk_x0": SettingSpec("skx", float, C.SK_DEFAULT_X0, C.SK_X0_MIN, C.SK_X0_MAX),
    "sk_term": SettingSpec("skt", bool, False),
    "sk_cost": SettingSpec("skk", str, C.SK_COST_KEYS[0], options=C.SK_COST_KEYS),
    "wg_N": SettingSpec("wgn", int, C.DEFAULT_WG_N, C.WG_N_MIN, C.WG_N_MAX),
    "wg_start": SettingSpec("wgs", str, C.DEFAULT_START, options=C.START_KEYS),
    "wf_N": SettingSpec("wfn", int, C.DEFAULT_WF_N, C.WF_N_MIN, C.WF_N_MAX),
    "wf_cost": SettingSpec("wfk", str, C.DEFAULT_COST_FALL, options=C.COST_KEYS),
    "wf_n": SettingSpec("wfm", int, C.DEFAULT_WF_STARTS, C.WF_STARTS_MIN, C.WF_STARTS_MAX),
    "wf_seed": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "tm_term": SettingSpec("tmt", str, C.DEFAULT_TERM, options=C.TERM_KEYS),
    "tm_N": SettingSpec("tmn", int, C.DEFAULT_TM_N, C.TM_N_MIN, C.TM_N_MAX),
}
PRESET_KEYS = {k: k for k in SETTING_SPECS}        # Preset-Schlüssel gleich Reglerschlüssel


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _parse(spec: SettingSpec, raw: str):
    """Wert aus der Adresszeile: auf die Grenzen begrenzt, bei Auswahlreglern auf die nächste erlaubte Stufe eingerastet; None bei Unsinn."""
    if spec.caster is bool:
        return str(raw).lower() in ("1", "true", "ja")
    if spec.caster is str:
        return raw if spec.options is None or raw in spec.options else None
    try:
        value = spec.caster(raw)
    except (ValueError, TypeError):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    if spec.options is not None:
        value = min(spec.options, key=lambda o: abs(o - value))
    if spec.caster is float and spec.options is None:
        value = round(value / C.SK_X0_STEP) * C.SK_X0_STEP      # Skalar-Start: Vielfache von 1/8 (exakt als Bruch lesbar)
        value = max(spec.lo, min(spec.hi, value))
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            value = _parse(spec, qp[spec.url_param])
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(int(value)) if isinstance(value, bool) else str(value)
    except Exception:
        pass


def apply_preset(name):
    """Alle Regler setzen: was das Preset nicht nennt, bekommt den Standardwert."""
    preset = C.PRESETS[name]
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = preset.get(state_key, spec.default)


def _gleich(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return a == b


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen (Gleitkommawerte auf 1e-9), sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State."""
    for name, preset in C.PRESETS.items():
        if all(_gleich(st.session_state.get(state_key, spec.default), preset.get(state_key, spec.default)) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    """Neuer Zufalls-Seed für die Startzustände; die Auswahl selbst läuft über SplitMix64 (aus der Uhrzeit), nicht über Python-`random`."""
    st.session_state["wf_seed"] = SplitMix64(time.time_ns()).below(C.SEED_MAX + 1)
