"""Export a standalone HTML visualizer for rendered transition previews."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_NAME = "transition_visualizer.html"


def _json_script_payload(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def _resolve_transition_paths(
    transition_dir: Path | None,
    *,
    audio_file: Path | None,
    metadata_file: Path | None,
    waveform_file: Path | None,
) -> tuple[Path, Path, Path, Path]:
    if transition_dir is not None:
        base = transition_dir.expanduser().resolve()
        audio_file = audio_file or (base / "preview.wav")
        metadata_file = metadata_file or (base / "transition.json")
        waveform_file = waveform_file or (base / "transition_waveforms.json")
    if audio_file is None or metadata_file is None or waveform_file is None:
        raise ValueError("Need either transition_dir or all of --audio-file, --metadata-file, and --waveform-file.")

    audio = audio_file.expanduser().resolve()
    metadata = metadata_file.expanduser().resolve()
    waveform = waveform_file.expanduser().resolve()
    for path in (audio, metadata, waveform):
        if not path.exists():
            raise FileNotFoundError(f"Required transition artifact not found: {path}")

    output_base = transition_dir.expanduser().resolve() if transition_dir is not None else metadata.parent
    return output_base, audio, metadata, waveform


def _relative_src(path: Path, *, html_dir: Path) -> str:
    return Path(os.path.relpath(path.resolve(), html_dir.resolve())).as_posix()


def build_transition_visualizer_html(
    *,
    audio_src: str,
    metadata: dict[str, Any],
    waveform: dict[str, Any],
    title: str,
) -> str:
    payload_scripts = "\n".join(
        [
            f'<script id="transition-metadata-json" type="application/json">{_json_script_payload(metadata)}</script>',
            f'<script id="transition-waveform-json" type="application/json">{_json_script_payload(waveform)}</script>',
        ]
    )

    style = """
<style>
  :root {
    color-scheme: dark;
    --bg: #111;
    --panel: #1b1b1b;
    --panel-2: #242424;
    --line: #343434;
    --text: #f2f2f2;
    --muted: #a8a8a8;
    --blue: #46c7f3;
    --yellow: #ffd13d;
    --pink: #e58cff;
    --orange: #d99036;
    --white: #f6f2e8;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  main {
    width: min(1180px, calc(100vw - 32px));
    margin: 18px auto 28px;
    display: grid;
    gap: 14px;
  }
  header {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 12px;
    align-items: end;
  }
  h1 {
    margin: 0;
    font-size: 22px;
    line-height: 1.2;
    font-weight: 650;
  }
  .meta {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.5;
  }
  .badge-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: flex-end;
  }
  .badge {
    border: 1px solid var(--line);
    background: var(--panel);
    border-radius: 4px;
    padding: 5px 7px;
    font-size: 12px;
    color: var(--muted);
  }
  audio { width: 100%; display: block; }
  .surface {
    border: 1px solid var(--line);
    background: var(--panel);
    border-radius: 6px;
    overflow: hidden;
  }
  .toolbar {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px;
    border-bottom: 1px solid var(--line);
    background: var(--panel-2);
    flex-wrap: wrap;
  }
  button {
    background: #f2f2f2;
    color: #111;
    border: 0;
    border-radius: 4px;
    padding: 7px 10px;
    font: inherit;
    cursor: pointer;
  }
  label {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--muted);
    font-size: 12px;
    user-select: none;
  }
  input[type="checkbox"] { accent-color: var(--blue); }
  .canvas-stack {
    position: relative;
    width: 100%;
    height: 500px;
    background: #121212;
  }
  canvas {
    position: absolute;
    inset: 0;
    display: block;
    width: 100%;
    height: 100%;
  }
  .legend {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 8px;
    padding: 10px;
    border-top: 1px solid var(--line);
    color: var(--muted);
    font-size: 12px;
  }
  .legend-item { display: inline-flex; align-items: center; gap: 7px; min-width: 0; }
  .swatch { width: 18px; height: 3px; border-radius: 2px; background: var(--blue); flex: 0 0 auto; }
  .swatch.wave { background: var(--orange); height: 10px; opacity: .8; }
  .swatch.volume { background: var(--blue); }
  .swatch.eq { background: var(--yellow); }
  .swatch.filter { background: var(--pink); }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 10px;
  }
  .stat {
    border: 1px solid var(--line);
    background: var(--panel);
    border-radius: 6px;
    padding: 10px;
  }
  .stat b { display: block; margin-bottom: 4px; font-size: 13px; }
  .stat span { color: var(--muted); font-size: 12px; line-height: 1.45; }
  @media (max-width: 760px) {
    header { grid-template-columns: 1fr; }
    .badge-row { justify-content: flex-start; }
    .canvas-stack { height: 580px; }
  }
</style>
"""

    script = """
<script>
(function() {
  const metadata = JSON.parse(document.getElementById('transition-metadata-json').textContent);
  const waveform = JSON.parse(document.getElementById('transition-waveform-json').textContent);

  const bgCanvas = document.getElementById('transition-bg-canvas');
  const bgCtx = bgCanvas.getContext('2d');
  const fgCanvas = document.getElementById('transition-fg-canvas');
  const fgCtx = fgCanvas.getContext('2d');
  const audio = document.getElementById('transition-audio');
  const playButton = document.getElementById('play-toggle');
  let layout = null;
  let backgroundDirty = true;
  const toggles = {
    volume: document.getElementById('toggle-volume'),
    eq: document.getElementById('toggle-eq'),
    filter: document.getElementById('toggle-filter'),
    playhead: document.getElementById('toggle-playhead'),
  };

  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  }
  function fmt(v, d=2) {
    const n = Number(v);
    return Number.isFinite(n) ? n.toFixed(d) : '';
  }
  function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.innerHTML = value;
  }
  function xFor(t, left, width) {
    return left + (Math.max(0, Math.min(1, Number(t) || 0)) * width);
  }
  function yForLine(value, laneTop, laneHeight) {
    return laneTop + ((1 - Math.max(0, Math.min(1, Number(value) || 0))) * laneHeight);
  }
  function drawRoundedRect(x, y, w, h, r) {
    bgCtx.beginPath();
    bgCtx.moveTo(x + r, y);
    bgCtx.lineTo(x + w - r, y);
    bgCtx.quadraticCurveTo(x + w, y, x + w, y + r);
    bgCtx.lineTo(x + w, y + h - r);
    bgCtx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    bgCtx.lineTo(x + r, y + h);
    bgCtx.quadraticCurveTo(x, y + h, x, y + h - r);
    bgCtx.lineTo(x, y + r);
    bgCtx.quadraticCurveTo(x, y, x + r, y);
    bgCtx.closePath();
  }
  function smoothValue(points, key, i) {
    const a = points[Math.max(0, i - 2)];
    const b = points[Math.max(0, i - 1)];
    const c = points[i];
    const d = points[Math.min(points.length - 1, i + 1)];
    const e = points[Math.min(points.length - 1, i + 2)];
    return (
      (Number(a[key]) || 0) +
      (2 * (Number(b[key]) || 0)) +
      (3 * (Number(c[key]) || 0)) +
      (2 * (Number(d[key]) || 0)) +
      (Number(e[key]) || 0)
    ) / 9;
  }
  function drawEnvelope(points, lane, key, color, scale) {
    const mid = lane.y + (lane.h * 0.5);
    const amp = lane.h * 0.46 * scale;
    bgCtx.save();
    bgCtx.fillStyle = color;
    bgCtx.beginPath();
    points.forEach((p, i) => {
      const x = xFor(p.t, lane.x, lane.w);
      const y = mid - (Math.min(1, smoothValue(points, key, i)) * amp);
      if (i === 0) bgCtx.moveTo(x, y);
      else bgCtx.lineTo(x, y);
    });
    for (let i = points.length - 1; i >= 0; i -= 1) {
      const p = points[i];
      const x = xFor(p.t, lane.x, lane.w);
      const y = mid + (Math.min(1, smoothValue(points, key, i)) * amp);
      bgCtx.lineTo(x, y);
    }
    bgCtx.closePath();
    bgCtx.fill();
    bgCtx.restore();
  }
  function drawMonoFallback(points, lane) {
    const mid = lane.y + (lane.h * 0.5);
    const amp = lane.h * 0.42;
    if (!points || !points.length) return;

    bgCtx.save();
    bgCtx.beginPath();
    points.forEach((p, i) => {
      const x = xFor(p.t, lane.x, lane.w);
      const y = mid - ((Number(p.max) || 0) * amp);
      if (i === 0) bgCtx.moveTo(x, y);
      else bgCtx.lineTo(x, y);
    });
    for (let i = points.length - 1; i >= 0; i -= 1) {
      const p = points[i];
      const x = xFor(p.t, lane.x, lane.w);
      const y = mid - ((Number(p.min) || 0) * amp);
      bgCtx.lineTo(x, y);
    }
    bgCtx.closePath();
    bgCtx.fillStyle = 'rgba(217,144,54,.70)';
    bgCtx.fill();
    bgCtx.restore();
  }
  function drawWave(points, lane) {
    if (!points || !points.length) return;
    const hasBands = points.some(p => p.low_amp != null || p.mid_amp != null || p.high_amp != null);
    if (!hasBands) {
      drawMonoFallback(points, lane);
      return;
    }

    const mid = lane.y + (lane.h * 0.5);
    const amp = lane.h * 0.42;
    drawEnvelope(points, lane, 'low_amp', 'rgba(24,75,216,.88)', 1.0);
    drawEnvelope(points, lane, 'mid_amp', 'rgba(211,128,39,.82)', 0.70);
    drawEnvelope(points, lane, 'high_amp', 'rgba(248,242,226,.92)', 0.36);

    bgCtx.strokeStyle = 'rgba(255,255,255,.12)';
    bgCtx.lineWidth = 1;
    bgCtx.beginPath();
    points.forEach((p, i) => {
      const x = xFor(p.t, lane.x, lane.w);
      const y = mid - ((Number(p.rms) || 0) * amp * 0.85);
      if (i === 0) bgCtx.moveTo(x, y);
      else bgCtx.lineTo(x, y);
    });
    bgCtx.stroke();
  }
  function drawLine(points, lane, color, width, valueKey='value', normalizedKey=null, dash=null) {
    if (!points || !points.length) return;
    bgCtx.save();
    bgCtx.strokeStyle = color;
    bgCtx.lineWidth = width;
    bgCtx.lineCap = 'round';
    bgCtx.lineJoin = 'round';
    if (dash) bgCtx.setLineDash(dash);
    bgCtx.beginPath();
    points.forEach((p, i) => {
      const value = normalizedKey ? p[normalizedKey] : p[valueKey];
      const x = xFor(p.t, lane.x, lane.w);
      const y = yForLine(value, lane.y + 8, lane.h - 16);
      if (i === 0) bgCtx.moveTo(x, y);
      else bgCtx.lineTo(x, y);
    });
    bgCtx.stroke();
    bgCtx.restore();
  }
  function drawGrid(lane) {
    const render = metadata.render || {};
    const frontBars = Math.max(0, Math.round(Number(render.front_padding_bars) || 0));
    const overlapBars = Math.max(1, Math.round(Number(render.overlap_bars) || 1));
    const backBars = Math.max(0, Math.round(Number(render.back_padding_bars) || 0));
    const bars = Math.max(1, Math.round(Number(render.timeline_bars) || (frontBars + overlapBars + backBars)));
    bgCtx.save();
    bgCtx.textAlign = 'center';
    bgCtx.textBaseline = 'top';
    bgCtx.font = '10px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    for (let i = 0; i <= bars; i += 1) {
      const x = lane.x + (lane.w * i / bars);
      const relativeBar = i - frontBars;
      const isBoundary = i === frontBars || i === frontBars + overlapBars;
      const isPhrase = relativeBar % 4 === 0;
      bgCtx.strokeStyle = isPhrase ? '#444' : '#2d2d2d';
      bgCtx.lineWidth = isPhrase ? 1.2 : 1;
      if (isBoundary) {
        bgCtx.strokeStyle = '#666';
        bgCtx.lineWidth = 1.4;
      }
      bgCtx.beginPath();
      bgCtx.moveTo(x, lane.y);
      bgCtx.lineTo(x, lane.y + lane.h);
      bgCtx.stroke();
      if (i < bars && (bars <= 16 || i % 2 === 0)) {
        const label = i - frontBars + 1;
        bgCtx.fillStyle = isPhrase ? '#9f9f9f' : '#666';
        bgCtx.fillText(String(label), x + (lane.w / bars / 2), lane.y + 6);
      }
    }
    bgCtx.restore();
  }
  function laneLabel(track, cueName, lane) {
    const title = track.title || lane.name;
    const artist = track.artists || '';
    const cue = cueName ? ' / ' + cueName : '';
    return { title, sub: artist + cue };
  }
  function resizeCanvases() {
    const rect = bgCanvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(320, Math.floor(rect.width * dpr));
    const height = Math.max(360, Math.floor(rect.height * dpr));
    for (const canvas of [bgCanvas, fgCanvas]) {
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        backgroundDirty = true;
      }
    }
    bgCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    fgCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = width / dpr;
    const h = height / dpr;
    const pad = 18;
    const labelW = Math.min(250, Math.max(142, w * 0.24));
    const gap = 18;
    const laneH = (h - (pad * 2) - gap) / 2;
    const from = metadata.from_track || {};
    const to = metadata.to_track || {};
    layout = {
      w, h,
      lanes: [
        { name: 'Track 1', key: 'outgoing', track: from, cue: from.cue_name, x: pad + labelW, y: pad, w: w - (pad * 2) - labelW, h: laneH },
        { name: 'Track 2', key: 'incoming', track: to, cue: to.cue_name, x: pad + labelW, y: pad + laneH + gap, w: w - (pad * 2) - labelW, h: laneH },
      ],
    };
  }
  function drawBackground() {
    resizeCanvases();
    if (!backgroundDirty || !layout) return;
    const { w, h, lanes } = layout;
    bgCtx.clearRect(0, 0, w, h);
    lanes.forEach(lane => {
      bgCtx.fillStyle = '#1d1d1d';
      drawRoundedRect(lane.x, lane.y, lane.w, lane.h, 8);
      bgCtx.fill();
      drawGrid(lane);
      const text = laneLabel(lane.track || {}, lane.cue, lane);
      bgCtx.textAlign = 'right';
      bgCtx.textBaseline = 'middle';
      bgCtx.fillStyle = '#f2f2f2';
      bgCtx.font = '13px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      bgCtx.fillText(text.title, lane.x - 12, lane.y + lane.h * 0.44);
      bgCtx.fillStyle = '#8f8f8f';
      bgCtx.font = '11px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      bgCtx.fillText(text.sub, lane.x - 12, lane.y + lane.h * 0.56);
      drawWave(waveform.waveforms[lane.key] || [], lane);
    });

    const auto = waveform.automation || {};
    if (toggles.volume.checked && auto.volume) {
      drawLine(auto.volume.outgoing, lanes[0], '#46c7f3', 3);
      drawLine(auto.volume.incoming, lanes[1], '#46c7f3', 3);
    }
    if (toggles.eq.checked && auto.eq) {
      drawLine(auto.eq.outgoing_low_gain, lanes[0], '#ffd13d', 2.5);
      drawLine(auto.eq.incoming_low_gain, lanes[1], '#ffd13d', 2.5);
    }
    if (toggles.filter.checked && auto.filter) {
      drawLine(auto.filter.outgoing_cutoff_hz, lanes[0], '#e58cff', 2.5, 'value', 'normalized');
      drawLine(auto.filter.incoming_cutoff_hz, lanes[1], '#e58cff', 2.5, 'value', 'normalized');
    }
    backgroundDirty = false;
  }
  function drawForeground() {
    resizeCanvases();
    if (!layout) return;
    const { w, h, lanes } = layout;
    fgCtx.clearRect(0, 0, w, h);
    if (toggles.playhead.checked && audio.duration) {
      const t = Math.max(0, Math.min(1, audio.currentTime / audio.duration));
      const x = xFor(t, lanes[0].x, lanes[0].w);
      fgCtx.strokeStyle = '#f6f2e8';
      fgCtx.lineWidth = 1.5;
      fgCtx.beginPath();
      fgCtx.moveTo(x, lanes[0].y);
      fgCtx.lineTo(x, lanes[1].y + lanes[1].h);
      fgCtx.stroke();
    }
  }
  function renderText() {
    const from = metadata.from_track || {};
    const to = metadata.to_track || {};
    const render = metadata.render || {};
    setText('subtitle', '<b>' + esc(from.title) + '</b> -> <b>' + esc(to.title) + '</b><br>' +
      esc(from.artists) + ' / ' + esc(to.artists));
    const padding = (Number(render.front_padding_bars) || 0) || (Number(render.back_padding_bars) || 0)
      ? ' + ' + esc(render.front_padding_bars || 0) + '/' + esc(render.back_padding_bars || 0) + ' padding'
      : '';
    setText('timing', fmt(render.duration_seconds, 2) + 's / ' + esc(render.overlap_bars) + ' bars' + padding);
    const fromPitch = Number(render.from_pitch_shift) || Number(from.pitch_shift) || 0;
    const toPitch = Number(render.to_pitch_shift) || Number(to.pitch_shift) || 0;
    const fromKey = from.shifted_key || from.key || '';
    const toKey = to.shifted_key || to.key || '';
    const pitchText = (fromKey || toKey)
      ? ' / ' + esc(fromKey) + (fromPitch ? ' ' + (fromPitch > 0 ? '+' : '') + fromPitch : '') +
        ' -> ' + esc(toKey) + (toPitch ? ' ' + (toPitch > 0 ? '+' : '') + toPitch : '')
      : '';
    setText('tempo', fmt(from.bpm, 2) + ' -> ' + fmt(to.bpm, 2) + ' BPM' + pitchText);
    setText('preset', esc(render.preset || '') + ' / ' + esc(render.volume_mode || '') + ' / ' + esc(render.eq_mode || '') + ' / ' + esc(render.filter_mode || ''));
    setText('cues', esc(from.cue_name || 'bar offset') + ' @ ' + fmt(render.from_start_seconds, 2) + 's<br>' +
      esc(to.cue_name || 'bar offset') + ' @ ' + fmt(render.to_start_seconds, 2) + 's');
  }

  playButton.addEventListener('click', () => {
    if (audio.paused) audio.play();
    else audio.pause();
  });
  audio.addEventListener('play', () => { playButton.textContent = 'Pause'; });
  audio.addEventListener('pause', () => { playButton.textContent = 'Play'; });
  audio.addEventListener('ended', () => { playButton.textContent = 'Play'; });
  Object.values(toggles).forEach(el => el.addEventListener('change', () => {
    backgroundDirty = true;
    drawBackground();
    drawForeground();
  }));
  window.addEventListener('resize', () => {
    backgroundDirty = true;
    drawBackground();
    drawForeground();
  });
  function tick() {
    drawBackground();
    drawForeground();
    requestAnimationFrame(tick);
  }
  renderText();
  tick();
})();
</script>
"""

    body = f"""
<main>
  <header>
    <div>
      <h1>{title}</h1>
      <div id="subtitle" class="meta"></div>
    </div>
    <div class="badge-row">
      <span id="timing" class="badge"></span>
      <span id="tempo" class="badge"></span>
    </div>
  </header>

  <section class="surface">
    <div class="toolbar">
      <button id="play-toggle" type="button">Play</button>
      <audio id="transition-audio" controls src="{audio_src}"></audio>
      <label><input id="toggle-volume" type="checkbox" checked> Volume</label>
      <label><input id="toggle-eq" type="checkbox" checked> EQ</label>
      <label><input id="toggle-filter" type="checkbox" checked> Filter</label>
      <label><input id="toggle-playhead" type="checkbox" checked> Playhead</label>
    </div>
    <div class="canvas-stack">
      <canvas id="transition-bg-canvas"></canvas>
      <canvas id="transition-fg-canvas"></canvas>
    </div>
    <div class="legend">
      <span class="legend-item"><span class="swatch wave"></span> Waveform</span>
      <span class="legend-item"><span class="swatch volume"></span> Volume automation</span>
      <span class="legend-item"><span class="swatch eq"></span> Low-band EQ gain</span>
      <span class="legend-item"><span class="swatch filter"></span> Filter cutoff</span>
    </div>
  </section>

  <section class="grid">
    <div class="stat"><b>Automation</b><span id="preset"></span></div>
    <div class="stat"><b>Cues</b><span id="cues"></span></div>
  </section>
</main>
"""

    return "\n".join(
        [
            "<!doctype html>",
            "<html>",
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width,initial-scale=1">',
            f"<title>{title}</title>",
            style,
            "</head>",
            "<body>",
            body,
            payload_scripts,
            script,
            "</body>",
            "</html>",
        ]
    )


def export_transition_visualizer(
    *,
    transition_dir: Path | None = None,
    audio_file: Path | None = None,
    metadata_file: Path | None = None,
    waveform_file: Path | None = None,
    output_file: Path | None = None,
    title: str | None = None,
) -> Path:
    output_base, audio, metadata_path, waveform_path = _resolve_transition_paths(
        transition_dir,
        audio_file=audio_file,
        metadata_file=metadata_file,
        waveform_file=waveform_file,
    )
    output = (output_file or (output_base / DEFAULT_OUTPUT_NAME)).expanduser().resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    waveform = json.loads(waveform_path.read_text(encoding="utf-8"))
    html_title = title or "Transition Preview"
    html = build_transition_visualizer_html(
        audio_src=_relative_src(audio, html_dir=output.parent),
        metadata=metadata,
        waveform=waveform,
        title=html_title,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a standalone HTML visualizer for a rendered transition.")
    parser.add_argument("transition_dir", type=Path, nargs="?", default=None)
    parser.add_argument("--audio-file", type=Path, default=None)
    parser.add_argument("--metadata-file", type=Path, default=None)
    parser.add_argument("--waveform-file", type=Path, default=None)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)

    output = export_transition_visualizer(
        transition_dir=args.transition_dir,
        audio_file=args.audio_file,
        metadata_file=args.metadata_file,
        waveform_file=args.waveform_file,
        output_file=args.output_file,
        title=args.title,
    )
    print(f"Wrote transition visualizer HTML: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
