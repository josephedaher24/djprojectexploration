from __future__ import annotations

import math

import numpy as np
import pytest

from djprojectexploration.transition_preview import (
    FILTER_EDGE_KILL_END,
    ISOLATOR_KILL_DB,
    MIX_HEADROOM_GAIN,
    OVERLAP_CROSSFADE_FLOOR_GAIN,
    TRUE_PEAK_CEILING_DBTP,
    _apply_bipolar_filter,
    _apply_schroeder_reverb,
    _apply_tempo_delay,
    _apply_final_limiter,
    _apply_mix_headroom,
    _apply_isolator_eq,
    _automation_curves,
    _db_to_fader_position,
    _evaluate_points,
    _explicit_automation_curves,
    _fader_position_to_db,
    _normalize_automation,
    _resolve_modes,
)


def _automation(total_beats: float = 16.0) -> dict[str, dict[str, list[dict[str, float | str]]]]:
    return {
        deck: {
            lane: [
                {"beat": 0.0, "value": 0.0, "curve": "linear"},
                {"beat": total_beats, "value": 0.0, "curve": "linear"},
            ]
            for lane in ("volume", "eq_low", "eq_mid", "eq_high", "filter")
        }
        for deck in ("from", "to")
    }


def test_explicit_automation_requires_overlap_endpoints() -> None:
    spec = _automation()
    spec["from"]["volume"][0]["beat"] = 1.0
    with pytest.raises(ValueError, match="start at beat 0"):
        _normalize_automation(spec, total_beats=16.0)


def test_auto_preset_defaults_to_overlap_crossfade() -> None:
    preset, volume, eq, filt = _resolve_modes(
        preset="auto",
        volume_mode=None,
        eq_mode=None,
        filter_mode=None,
    )
    assert (preset, volume, eq, filt) == ("auto", "overlap-crossfade", "center-bass-swap", "none")


def test_automation_curves_respect_shapes_and_kill() -> None:
    points = [
        {"beat": 0.0, "value": 0.0, "curve": "linear"},
        {"beat": 8.0, "value": 8.0, "curve": "smooth"},
        {"beat": 16.0, "value": 0.0, "curve": "exponential"},
    ]
    values = _evaluate_points(points, beats=np.array([0.0, 4.0, 8.0, 12.0, 16.0], dtype=np.float32))
    assert values[0] == pytest.approx(0.0)
    assert values[2] == pytest.approx(8.0)
    assert values[-1] == pytest.approx(0.0)

    spec = _automation()
    spec["from"]["volume"] = [
        {"beat": 0.0, "value": 0.0, "curve": "linear"},
        {"beat": 16.0, "value": -60.0, "curve": "linear"},
    ]
    spec["from"]["eq_low"] = [
        {"beat": 0.0, "value": ISOLATOR_KILL_DB, "curve": "linear"},
        {"beat": 16.0, "value": ISOLATOR_KILL_DB, "curve": "linear"},
    ]
    curves = _explicit_automation_curves(_normalize_automation(spec, total_beats=16.0) or {}, samples=9, total_beats=16.0)
    assert curves["a_gain"][0] == pytest.approx(1.0)
    assert curves["a_gain"][-1] < 0.002
    assert np.all(curves["a_eq_low_db"] == ISOLATOR_KILL_DB)


def test_explicit_volume_accepts_daw_boost_and_digital_mute() -> None:
    spec = _automation()
    spec["from"]["volume"] = [
        {"beat": 0.0, "value": -96.0, "curve": "linear"},
        {"beat": 16.0, "value": 6.0, "curve": "linear"},
    ]
    curves = _explicit_automation_curves(
        _normalize_automation(spec, total_beats=16.0) or {},
        samples=3,
        total_beats=16.0,
    )
    assert curves["a_gain"][0] == pytest.approx(0.0)
    assert curves["a_gain"][-1] == pytest.approx(10.0 ** (6.0 / 20.0))


def test_delay_and_reverb_depth_lanes_are_normalized_and_audible() -> None:
    spec = _automation(total_beats=4.0)
    for deck in ("from", "to"):
        spec[deck]["delay"] = [
            {"beat": 0.0, "value": 0.0, "curve": "linear"},
            {"beat": 4.0, "value": 1.0, "curve": "linear"},
        ]
        spec[deck]["reverb"] = [
            {"beat": 0.0, "value": 0.0, "curve": "linear"},
            {"beat": 4.0, "value": 1.0, "curve": "linear"},
        ]
    curves = _explicit_automation_curves(_normalize_automation(spec, total_beats=4.0) or {}, samples=5, total_beats=4.0)
    assert curves["a_delay_depth"].tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert curves["b_reverb_depth"][-1] == pytest.approx(1.0)

    sample_rate = 8000
    impulse = np.zeros((2, sample_rate), dtype=np.float32)
    impulse[:, 0] = 1.0
    on = np.ones(sample_rate, dtype=np.float32)
    delayed = _apply_tempo_delay(impulse, sample_rate=sample_rate, depth=on, beat_seconds=0.5, beats=0.125, tone=0.5)
    reverbed = _apply_schroeder_reverb(impulse, sample_rate=sample_rate, depth=on, decay_seconds=0.5, tone=0.5)
    assert float(np.max(np.abs(delayed[:, 500:]))) > 0.0
    assert float(np.max(np.abs(reverbed[:, 200:]))) > 0.0


def test_isolator_flat_kill_and_filter_center_bypass() -> None:
    sample_rate = 44100
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    low = np.sin(2.0 * math.pi * 90.0 * t)
    high = np.sin(2.0 * math.pi * 7000.0 * t)
    audio = np.stack([low + high, low + high]).astype(np.float32)
    zeros = np.zeros(audio.shape[1], dtype=np.float32)
    flat = _apply_isolator_eq(audio, sample_rate=sample_rate, low_db=zeros, mid_db=zeros, high_db=zeros)
    assert np.allclose(flat, audio)
    killed = _apply_isolator_eq(
        audio,
        sample_rate=sample_rate,
        low_db=np.full(audio.shape[1], ISOLATOR_KILL_DB, dtype=np.float32),
        mid_db=zeros,
        high_db=zeros,
    )
    assert float(np.std(killed)) < float(np.std(audio)) * 0.9
    assert np.allclose(_apply_bipolar_filter(audio, sample_rate=sample_rate, amount=zeros), audio)


def test_final_limiter_has_ceiling_without_quiet_normalization() -> None:
    sample_rate = 44100
    ceiling = 10.0 ** (TRUE_PEAK_CEILING_DBTP / 20.0)
    loud = np.stack([np.full(sample_rate, 4.0, dtype=np.float32)] * 2)
    limited = _apply_final_limiter(loud, sample_rate=sample_rate)
    assert float(np.max(np.abs(limited))) <= ceiling + 1e-6
    quiet = np.stack([np.full(sample_rate, 0.1, dtype=np.float32)] * 2)
    assert np.allclose(_apply_final_limiter(quiet, sample_rate=sample_rate), quiet, atol=2e-4)


def test_legacy_overlap_crossfade_floor_and_hard_center_bass_swap() -> None:
    overlap = _automation_curves(
        samples=5,
        sample_rate=44100,
        volume_mode="overlap-crossfade",
        eq_mode="center-bass-swap",
        filter_mode="none",
    )
    assert overlap["a_gain"][0] == pytest.approx(1.0)
    assert overlap["a_gain"][-1] == pytest.approx(OVERLAP_CROSSFADE_FLOOR_GAIN)
    assert overlap["b_gain"][0] == pytest.approx(OVERLAP_CROSSFADE_FLOOR_GAIN)
    assert overlap["b_gain"][-1] == pytest.approx(1.0)
    assert overlap["a_low_gain"][1] == pytest.approx(1.0)
    assert overlap["a_low_gain"][2] == pytest.approx(0.0)
    assert overlap["b_low_gain"][1] == pytest.approx(0.0)
    assert overlap["b_low_gain"][2] == pytest.approx(1.0)


def test_legacy_cut_and_fade_half_transition_modes() -> None:
    cut = _automation_curves(
        samples=5,
        sample_rate=44100,
        volume_mode="cut-in-fade-out",
        eq_mode="none",
        filter_mode="none",
    )
    assert np.allclose(cut["b_gain"], 1.0)
    assert cut["a_gain"][0] == pytest.approx(1.0)
    assert cut["a_gain"][2] == pytest.approx(1.0)
    assert cut["a_gain"][-1] == pytest.approx(0.0)

    fade = _automation_curves(
        samples=5,
        sample_rate=44100,
        volume_mode="fade-in-cut-out",
        eq_mode="none",
        filter_mode="none",
    )
    assert np.allclose(fade["a_gain"], 1.0)
    assert fade["b_gain"][0] == pytest.approx(0.0)
    assert fade["b_gain"][2] == pytest.approx(1.0)
    assert fade["b_gain"][-1] == pytest.approx(1.0)


def test_explicit_cut_and_fade_templates_hold_their_intended_half() -> None:
    """The UI sends dB points, so verify those explicit envelope semantics too."""
    cut_spec = _automation()
    cut_spec["from"]["volume"] = [
        {"beat": 0.0, "value": 0.0, "curve": "linear"},
        {"beat": 8.0, "value": 0.0, "curve": "linear"},
        {"beat": 16.0, "value": -60.0, "curve": "linear"},
    ]
    cut = _explicit_automation_curves(
        _normalize_automation(cut_spec, total_beats=16.0) or {},
        samples=5,
        total_beats=16.0,
    )
    assert np.allclose(cut["a_volume_db"][:3], 0.0)
    assert cut["a_volume_db"][-1] == pytest.approx(-60.0)
    assert np.allclose(cut["b_volume_db"], 0.0)

    fade_spec = _automation()
    fade_spec["to"]["volume"] = [
        {"beat": 0.0, "value": -60.0, "curve": "linear"},
        {"beat": 8.0, "value": 0.0, "curve": "linear"},
        {"beat": 16.0, "value": 0.0, "curve": "linear"},
    ]
    fade = _explicit_automation_curves(
        _normalize_automation(fade_spec, total_beats=16.0) or {},
        samples=5,
        total_beats=16.0,
    )
    assert np.allclose(fade["a_volume_db"], 0.0)
    assert fade["b_volume_db"][0] == pytest.approx(-60.0)
    assert np.allclose(fade["b_volume_db"][2:], 0.0)


def test_same_beat_jump_is_preserved_and_right_continuous() -> None:
    spec = _automation()
    spec["from"]["eq_low"] = [
        {"beat": 0.0, "value": 0.0, "curve": "linear"},
        {"beat": 8.0, "value": 0.0, "curve": "linear"},
        {"beat": 8.0, "value": ISOLATOR_KILL_DB, "curve": "linear"},
        {"beat": 16.0, "value": ISOLATOR_KILL_DB, "curve": "linear"},
    ]
    normalized = _normalize_automation(spec, total_beats=16.0) or {}
    points = normalized["from"]["eq_low"]
    assert len(points) == 4
    values = _evaluate_points(points, beats=np.array([7.99, 8.0, 8.01], dtype=np.float32))
    assert values[0] == pytest.approx(0.0, abs=0.1)
    assert values[1] == pytest.approx(ISOLATOR_KILL_DB)
    assert values[2] == pytest.approx(ISOLATOR_KILL_DB)

    spec["to"]["eq_low"] = [
        {"beat": 0.0, "value": ISOLATOR_KILL_DB, "curve": "linear"},
        {"beat": 8.0, "value": ISOLATOR_KILL_DB, "curve": "linear"},
        {"beat": 8.0, "value": 0.0, "curve": "linear"},
        {"beat": 16.0, "value": 0.0, "curve": "linear"},
    ]
    incoming = _normalize_automation(spec, total_beats=16.0) or {}
    assert _evaluate_points(
        incoming["to"]["eq_low"], beats=np.array([7.99, 8.0, 8.01], dtype=np.float32)
    ).tolist() == pytest.approx([ISOLATOR_KILL_DB, 0.0, 0.0], abs=0.1)

    spec["from"]["eq_low"].insert(2, {"beat": 8.0, "value": -48.0, "curve": "linear"})
    with pytest.raises(ValueError, match="at most two"):
        _normalize_automation(spec, total_beats=16.0)


def test_cubic_fader_anchors_inverse_and_fader_domain_interpolation() -> None:
    positions = np.asarray([0.0, 0.25, 0.5, 0.85, 1.0], dtype=np.float32)
    expected = np.asarray([-96.0, -30.0, -13.0, 0.0, 6.0], dtype=np.float32)
    assert _fader_position_to_db(positions).tolist() == pytest.approx(expected.tolist(), abs=1e-5)
    assert _db_to_fader_position(expected).tolist() == pytest.approx(positions.tolist(), abs=1e-5)
    dense = _fader_position_to_db(np.linspace(0.0, 1.0, 1001, dtype=np.float32))
    assert np.all(np.diff(dense) >= 0.0)

    points = [
        {"beat": 0.0, "value": -30.0, "curve": "linear"},
        {"beat": 16.0, "value": 0.0, "curve": "linear"},
    ]
    midpoint = _evaluate_points(points, beats=np.asarray([8.0], dtype=np.float32), lane="volume")[0]
    assert midpoint == pytest.approx(float(_fader_position_to_db(np.asarray([0.55]))[0]), abs=1e-5)
    assert midpoint != pytest.approx(-15.0, abs=0.1)
    shaped = [
        {"beat": 0.0, "value": -30.0, "curve": "smooth"},
        {"beat": 8.0, "value": 0.0, "curve": "exponential"},
        {"beat": 16.0, "value": 6.0, "curve": "linear"},
    ]
    shaped_values = _evaluate_points(shaped, beats=np.asarray([0.0, 4.0, 8.0, 12.0, 16.0], dtype=np.float32), lane="eq_mid")
    assert shaped_values.tolist() == pytest.approx([-30.0, -10.7228, 0.0, 0.6973, 6.0], abs=0.02)


def test_global_mix_headroom_and_filter_terminal_kills() -> None:
    assembled = np.ones((2, 9), dtype=np.float32)
    assert np.allclose(_apply_mix_headroom(assembled), MIX_HEADROOM_GAIN)

    sample_rate = 44100
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = np.stack([np.sin(2.0 * math.pi * 90.0 * t) + np.sin(2.0 * math.pi * 9000.0 * t)] * 2).astype(np.float32)
    lpf_kill = _apply_bipolar_filter(audio, sample_rate=sample_rate, amount=np.full(audio.shape[1], -1.0, dtype=np.float32))
    hpf_kill = _apply_bipolar_filter(audio, sample_rate=sample_rate, amount=np.full(audio.shape[1], 1.0, dtype=np.float32))
    assert np.max(np.abs(lpf_kill)) == pytest.approx(0.0, abs=1e-7)
    assert np.max(np.abs(hpf_kill)) == pytest.approx(0.0, abs=1e-7)
    near_edge = _apply_bipolar_filter(
        audio,
        sample_rate=sample_rate,
        amount=np.full(audio.shape[1], FILTER_EDGE_KILL_END - 0.001, dtype=np.float32),
    )
    assert np.max(np.abs(near_edge)) > 0.0
