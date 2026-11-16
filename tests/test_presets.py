"""Presets: Vollständigkeit, gültige Werte, Standard-Preset gleich Standardeinstellungen, Unterscheidbarkeit; Permalink begrenzt und rastet ein."""

import pytest

import mzl_constants as C
import mzl_presets as P


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.SETTING_SPECS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_options():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[key]
            if spec.options is not None:
                assert value in spec.options, (name, key)
            elif spec.lo is not None:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_first_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS.values()))
    assert all(first.get(key, spec.default) == spec.default for key, spec in P.SETTING_SPECS.items())


def test_presets_differ_pairwise():
    """Kein zwei Presets haben dieselben Werte: sonst könnte die Hervorhebung nur das erste markieren."""
    vecs = [tuple(preset.get(key, spec.default) for key, spec in P.SETTING_SPECS.items()) for preset in C.PRESETS.values()]
    assert len(set(vecs)) == len(vecs)


def test_setting_specs_cover_every_preset_key_and_every_constant_default():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert P.SETTING_SPECS["wf_seed"].default == C.DEFAULT_SEED == 716


def test_permalink_parsing_limits_and_snaps():
    spec = P.SETTING_SPECS
    assert P._parse(spec["umax"], "0.6") == 0.5 and P._parse(spec["umax"], "99") == 5.0 and P._parse(spec["umax"], "-1") == 0.25      # einrasten auf die Stufen
    assert P._parse(spec["s1"], "7") in (5.0, 10.0) and P._parse(spec["wf_N"], "99") == C.WF_N_MAX and P._parse(spec["wf_N"], "-4") == C.WF_N_MIN
    assert P._parse(spec["sk_x0"], "1.51") == 1.5 and P._parse(spec["sk_x0"], "9") == C.SK_X0_MAX
    assert P._parse(spec["wf_cost"], "nonsense") is None and P._parse(spec["wf_cost"], "pos") == "pos"
    assert P._parse(spec["sk_term"], "1") is True and P._parse(spec["sk_term"], "0") is False
    assert P._parse(spec["wf_N"], "abc") is None and P._parse(spec["sk_x0"], "nan") is None and P._parse(spec["sk_x0"], "inf") is None


def test_preset_help_texts_agree_with_the_measurements():
    """Die Hilfetexte nennen Zahlen: N̄ = 1 bei |u| ≤ 5 und Horizont 7 (N̄ + 1) für die Hauptgrenzen (exakt)."""
    import mzl_regions as R

    assert R.regions(5.0, 5.0, 5.0).n_bar == 1 and "N̄ = 1" in C.PRESET_HELP["Grobe Stellgrenze (|u| ≤ 5)"]
    assert R.regions(0.5, 5.0, 5.0).n_bar + 1 == 7 and "F₇" in C.PRESET_HELP["Wagen: sicher ab N = 7"]
    assert C.PRESETS["Wagen: sicher ab N = 7"]["wf_N"] == 7
