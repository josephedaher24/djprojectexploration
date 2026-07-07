(() => {
  const state = { tracks: [], options: null, lastRender: null, lastRenderedBody: null };
  const els = {
    fromTrack: document.getElementById('from-track'),
    toTrack: document.getElementById('to-track'),
    fromCue: document.getElementById('from-cue'),
    toCue: document.getElementById('to-cue'),
    overlapBars: document.getElementById('overlap-bars'),
    paddingBars: document.getElementById('padding-bars'),
    frontPaddingBars: document.getElementById('front-padding-bars'),
    backPaddingBars: document.getElementById('back-padding-bars'),
    fromNudgeBeats: document.getElementById('from-nudge-beats'),
    toNudgeBeats: document.getElementById('to-nudge-beats'),
    fromPitchShift: document.getElementById('from-pitch-shift'),
    toPitchShift: document.getElementById('to-pitch-shift'),
    keyPreview: document.getElementById('key-preview'),
    preset: document.getElementById('preset'),
    volumeMode: document.getElementById('volume-mode'),
    eqMode: document.getElementById('eq-mode'),
    filterMode: document.getElementById('filter-mode'),
    render: document.getElementById('render'),
    swap: document.getElementById('swap'),
    status: document.getElementById('status'),
    preview: document.getElementById('preview'),
    resultMeta: document.getElementById('result-meta'),
    resultLinks: document.getElementById('result-links'),
  };
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  }
  function setStatus(text, error=false) {
    els.status.textContent = text;
    els.status.classList.toggle('error', error);
  }
  function trackLabel(track) {
    const bpm = track.bpm ? ' / ' + Number(track.bpm).toFixed(1) + ' BPM' : '';
    const key = track.key ? ' / ' + track.key : '';
    const source = track.source_label ? track.source_label + ' / ' : '';
    return source + String(track.track_number).padStart(2, '0') + ' - ' + track.title + ' - ' + track.artists + bpm + key;
  }
  function pitchLabel(value) {
    const n = Number(value) || 0;
    return n === 0 ? '0 st' : (n > 0 ? '+' : '') + n + ' st';
  }
  function shiftCamelot(key, semitones) {
    const text = String(key || '').trim().toUpperCase();
    const match = text.match(/^(\\d{1,2})([AB])$/);
    if (!match) return text;
    const number = Number(match[1]);
    if (number < 1 || number > 12) return text;
    const shifted = ((number - 1 + (7 * Number(semitones || 0))) % 12 + 12) % 12 + 1;
    return String(shifted) + match[2];
  }
  function fillSelect(select, items, valueFn, labelFn, selected=null) {
    select.innerHTML = '';
    for (const item of items) {
      const option = document.createElement('option');
      option.value = valueFn(item);
      option.textContent = labelFn(item);
      select.appendChild(option);
    }
    if (selected != null) select.value = String(selected);
  }
  function selectedTrack(select) {
    return state.tracks.find(track => track.id === select.value) || null;
  }
  function cuesFor(track, role) {
    if (!track) return [];
    const cues = Array.isArray(track.cues) ? track.cues : [];
    return cues;
  }
  function fillCues() {
    const from = selectedTrack(els.fromTrack);
    const to = selectedTrack(els.toTrack);
    const fromCues = cuesFor(from, 'out');
    const toCues = cuesFor(to, 'in');
    const defaultFrom = fromCues.find(cue => String(cue.role || '').toLowerCase() === 'out')?.name || fromCues[0]?.name;
    const defaultTo = toCues.find(cue => String(cue.role || '').toLowerCase() === 'in')?.name || toCues[0]?.name;
    fillSelect(els.fromCue, fromCues, cue => cue.name, cue => cue.name + ' / ' + (cue.role || 'cue') + (cue.start_seconds == null ? '' : ' @ ' + Number(cue.start_seconds).toFixed(2) + 's'), defaultFrom);
    fillSelect(els.toCue, toCues, cue => cue.name, cue => cue.name + ' / ' + (cue.role || 'cue') + (cue.start_seconds == null ? '' : ' @ ' + Number(cue.start_seconds).toFixed(2) + 's'), defaultTo);
    updateKeyPreview();
  }
  function presetDetails(name) {
    return (state.options.preset_details || {})[name] || {};
  }
  function applyPresetToControls(name) {
    const preset = presetDetails(name || 'auto');
    els.volumeMode.value = preset.volume_mode || els.volumeMode.value || 'crossfade';
    els.eqMode.value = preset.eq_mode || els.eqMode.value || 'none';
    els.filterMode.value = preset.filter_mode || els.filterMode.value || 'none';
  }
  function setCustomPreset() {
    if (els.preset.value !== 'custom') els.preset.value = 'custom';
  }
  function fillModes() {
    fillSelect(els.preset, state.options.presets, x => x, x => x, 'auto');
    fillSelect(els.volumeMode, state.options.volume_modes, x => x, x => x);
    fillSelect(els.eqMode, state.options.eq_modes, x => x, x => x);
    fillSelect(els.filterMode, state.options.filter_modes, x => x, x => x);
    applyPresetToControls(els.preset.value);
    const pitchValues = [-3, -2, -1, 0, 1, 2, 3];
    fillSelect(els.fromPitchShift, pitchValues, x => x, pitchLabel, 0);
    fillSelect(els.toPitchShift, pitchValues, x => x, pitchLabel, 0);
  }
  function updateKeyPreview() {
    const from = selectedTrack(els.fromTrack);
    const to = selectedTrack(els.toTrack);
    const fromShift = Number(els.fromPitchShift.value) || 0;
    const toShift = Number(els.toPitchShift.value) || 0;
    const fromKey = shiftCamelot(from?.key, fromShift);
    const toKey = shiftCamelot(to?.key, toShift);
    els.keyPreview.textContent = 'Keys: ' + (from?.key || '-') + ' -> ' + (fromKey || '-') +
      ' / ' + (to?.key || '-') + ' -> ' + (toKey || '-');
  }
  async function loadInitial() {
    const [tracksRes, optionsRes] = await Promise.all([fetch('/api/tracks'), fetch('/api/options')]);
    const tracks = await tracksRes.json();
    const options = await optionsRes.json();
    if (!tracks.ok) throw new Error(tracks.error || 'Track load failed');
    if (!options.ok) throw new Error(options.error || 'Option load failed');
    state.tracks = tracks.tracks;
    state.options = options;
    fillSelect(els.fromTrack, state.tracks, t => t.id, trackLabel, state.tracks[0]?.id);
    fillSelect(els.toTrack, state.tracks, t => t.id, trackLabel, state.tracks[1]?.id || state.tracks[0]?.id);
    fillModes();
    fillCues();
    setStatus(state.tracks.length + ' tracks loaded from ' + (tracks.sources || []).map(source => source.label).join(' + '));
  }
  function renderBody() {
    return {
      from_track: els.fromTrack.value,
      to_track: els.toTrack.value,
      from_cue: els.fromCue.value,
      to_cue: els.toCue.value,
      overlap_bars: Number(els.overlapBars.value),
      front_padding_bars: Number(els.frontPaddingBars.value),
      back_padding_bars: Number(els.backPaddingBars.value),
      from_nudge_beats: Number(els.fromNudgeBeats.value),
      to_nudge_beats: Number(els.toNudgeBeats.value),
      from_pitch_shift: Number(els.fromPitchShift.value),
      to_pitch_shift: Number(els.toPitchShift.value),
      preset: els.preset.value,
      volume_mode: els.volumeMode.value,
      eq_mode: els.eqMode.value,
      filter_mode: els.filterMode.value,
      overwrite: true,
    };
  }
  async function renderTransition() {
    els.render.disabled = true;
    setStatus('Rendering...');
    const body = renderBody();
    try {
      const res = await fetch('/api/render', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || 'Render failed');
      state.lastRender = data;
      state.lastRenderedBody = body;
      els.preview.className = '';
      els.preview.innerHTML = '<div class="preview-toolbar"><audio id="preview-audio" controls src="' + esc(data.urls.preview) + '?t=' + Date.now() + '"></audio></div><div class="canvas-wrap"><canvas id="preview-bg"></canvas><canvas id="preview-fg"></canvas></div>';
      const audio = document.getElementById('preview-audio');
      if (audio) {
        audio.addEventListener('timeupdate', drawPreview);
        audio.addEventListener('seeked', drawPreview);
        audio.addEventListener('play', tickPlayhead);
        audio.addEventListener('pause', drawPreview);
      }
      const t = data.transition;
      els.resultMeta.textContent = t.from_track_number + ' ' + t.from_title + ' -> ' + t.to_track_number + ' ' + t.to_title + ' / ' + Number(t.duration_seconds).toFixed(2) + 's';
      els.resultLinks.innerHTML = '<a href="' + esc(data.urls.visualizer) + '" target="_blank">Visualizer</a>' +
        '<a href="' + esc(data.urls.preview) + '" target="_blank">WAV</a>' +
        '<a href="' + esc(data.urls.metadata) + '" target="_blank">Metadata</a>';
      setStatus('Rendered ' + t.from_title + ' -> ' + t.to_title);
      await loadPreview(data);
    } catch (err) {
      setStatus(err.message || String(err), true);
    } finally {
      els.render.disabled = false;
    }
  }
  function smoothstep(x) {
    return x * x * (3 - (2 * x));
  }
  function resolvedModes(body) {
    return {
      volume: body.volume_mode || 'crossfade',
      eq: body.eq_mode || 'none',
      filter: body.filter_mode || 'none',
    };
  }
  function currentSignature(body) {
    const modes = resolvedModes(body);
    return JSON.stringify({
      from_track: body.from_track,
      to_track: body.to_track,
      from_cue: body.from_cue,
      to_cue: body.to_cue,
      overlap_bars: body.overlap_bars,
      front_padding_bars: body.front_padding_bars,
      back_padding_bars: body.back_padding_bars,
      from_nudge_beats: body.from_nudge_beats,
      to_nudge_beats: body.to_nudge_beats,
      from_pitch_shift: body.from_pitch_shift,
      to_pitch_shift: body.to_pitch_shift,
      volume_mode: modes.volume,
      eq_mode: modes.eq,
      filter_mode: modes.filter,
    });
  }
  function renderedSignature() {
    return state.lastRenderedBody ? currentSignature(state.lastRenderedBody) : '';
  }
  function valueFor(mode, deck, x) {
    const half = x < 0.5;
    if (mode === 'volume:crossfade') return deck === 'a' ? Math.cos(x * Math.PI * 0.5) : Math.sin(x * Math.PI * 0.5);
    if (mode === 'volume:overlap-crossfade') {
      const floor = Math.pow(10, -6 / 20);
      const curve = deck === 'a' ? Math.cos(x * Math.PI * 0.5) : Math.sin(x * Math.PI * 0.5);
      return floor + ((1 - floor) * curve);
    }
    if (mode === 'volume:smooth-crossfade') {
      const t = smoothstep(x);
      return deck === 'a' ? Math.sqrt(Math.max(0, 1 - t)) : Math.sqrt(Math.max(0, t));
    }
    if (mode === 'volume:overlap') return 1;
    if (mode === 'volume:fade-in-fade-out') return deck === 'a' ? (half ? 1 : 1 - ((x - 0.5) * 2)) : (half ? x * 2 : 1);
    if (mode === 'volume:cut-in-fade-out') return deck === 'a' ? (half ? 1 : 1 - ((x - 0.5) * 2)) : 1;
    if (mode === 'volume:fade-in-cut-out') return deck === 'a' ? 1 : (half ? x * 2 : 1);
    if (mode === 'volume:center-cut') return deck === 'a' ? (half ? 1 : 0) : (half ? 0 : 1);
    if (mode === 'eq:none') return 1;
    if (mode === 'eq:start-bass-swap') return deck === 'a' ? 0.15 : 1;
    if (mode === 'eq:end-bass-swap') return deck === 'b' ? 0.15 : 1;
    if (mode === 'eq:center-bass-swap') return deck === 'a' ? (half ? 1 : 0.15) : (half ? 0.15 : 1);
    if (mode === 'eq:long-bass-cut') return 0.15;
    if (mode === 'filter:none') return null;
    if (mode === 'filter:low-pass-filter-out') return deck === 'a' ? 1 - x : null;
    if (mode === 'filter:low-pass-filter-in') return deck === 'b' ? x : null;
    if (mode === 'filter:high-pass-filter-out') return deck === 'a' ? x : null;
    if (mode === 'filter:high-pass-filter-in') return deck === 'b' ? 1 - x : null;
    return null;
  }
  async function loadPreview(data) {
    const [metadata, waveform] = await Promise.all([
      fetch(data.urls.metadata + '?t=' + Date.now()).then(r => r.json()),
      fetch(data.urls.waveform + '?t=' + Date.now()).then(r => r.json()),
    ]);
    state.preview = { metadata, waveform };
    drawPreview();
  }
  function xFor(t, lane) {
    return lane.x + (Math.max(0, Math.min(1, Number(t) || 0)) * lane.w);
  }
  function yFor(v, lane) {
    return lane.y + ((1 - Math.max(0, Math.min(1, Number(v) || 0))) * lane.h);
  }
  function smoothPoint(points, key, i) {
    const a = points[Math.max(0, i - 1)];
    const b = points[i];
    const c = points[Math.min(points.length - 1, i + 1)];
    return ((Number(a[key]) || 0) + (2 * (Number(b[key]) || 0)) + (Number(c[key]) || 0)) / 4;
  }
  function drawEnvelope(ctx, points, lane, key, color, scale) {
    if (!points || !points.length) return;
    const mid = lane.y + lane.h * 0.5;
    const amp = lane.h * 0.46 * scale;
    ctx.fillStyle = color;
    ctx.beginPath();
    points.forEach((p, i) => {
      const x = xFor(p.t, lane);
      const y = mid - (Math.min(1, smoothPoint(points, key, i)) * amp);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    for (let i = points.length - 1; i >= 0; i -= 1) {
      const p = points[i];
      const x = xFor(p.t, lane);
      const y = mid + (Math.min(1, smoothPoint(points, key, i)) * amp);
      ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
  }
  function drawWave(ctx, points, lane) {
    drawEnvelope(ctx, points, lane, 'low_amp', 'rgba(24,75,216,.86)', 1.0);
    drawEnvelope(ctx, points, lane, 'mid_amp', 'rgba(211,128,39,.80)', 0.70);
    drawEnvelope(ctx, points, lane, 'high_amp', 'rgba(248,242,226,.92)', 0.36);
  }
  function drawLine(ctx, points, lane, color, width, dashed=false, valueKey='value', normalizedKey=null) {
    if (!points || !points.length) return;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    if (dashed) ctx.setLineDash([7, 6]);
    ctx.beginPath();
    points.forEach((p, i) => {
      const v = normalizedKey ? p[normalizedKey] : p[valueKey];
      const x = xFor(p.t, lane);
      const y = yFor(v, { ...lane, y: lane.y + 8, h: lane.h - 16 });
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.restore();
  }
  function drawDashedLine(ctx, lane, color, fn) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.setLineDash([7, 6]);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    let started = false;
    for (let i = 0; i <= 120; i += 1) {
      const t = i / 120;
      const v = fn(t);
      if (v == null || Number.isNaN(v)) {
        started = false;
        continue;
      }
      const x = lane.x + (t * lane.w);
      const y = lane.y + ((1 - Math.max(0, Math.min(1, v))) * lane.h);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
    ctx.restore();
  }
  function drawPlayhead(ctx, lanes) {
    const audio = document.getElementById('preview-audio');
    const duration = Number(audio?.duration) || Number(state.preview?.metadata?.render?.duration_seconds) || 0;
    if (!audio || !duration || !lanes.length) return;
    const t = Math.max(0, Math.min(1, audio.currentTime / duration));
    const x = xFor(t, lanes[0]);
    ctx.save();
    ctx.strokeStyle = 'rgba(246,242,232,.92)';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(x, lanes[0].y);
    ctx.lineTo(x, lanes[lanes.length - 1].y + lanes[lanes.length - 1].h);
    ctx.stroke();
    ctx.restore();
  }
  function tickPlayhead() {
    drawPreview();
    const audio = document.getElementById('preview-audio');
    if (audio && !audio.paused && !audio.ended) requestAnimationFrame(tickPlayhead);
  }
  function drawAutomationPreview() {
    updateKeyPreview();
    drawPreview();
  }
  function drawPreview() {
    updateKeyPreview();
    const bg = document.getElementById('preview-bg');
    const fg = document.getElementById('preview-fg');
    if (!bg || !fg || !state.preview) return;
    const body = renderBody();
    const dirty = currentSignature(body) !== renderedSignature();
    const rect = bg.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    for (const canvas of [bg, fg]) {
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    }
    const ctx = bg.getContext('2d');
    const fctx = fg.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    fctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    fctx.clearRect(0, 0, rect.width, rect.height);
    const metadata = state.preview.metadata;
    const waveform = state.preview.waveform;
    const pad = 18;
    const labelW = Math.min(250, Math.max(142, rect.width * 0.24));
    const gap = 18;
    const laneH = (rect.height - (pad * 2) - gap) / 2;
    const laneW = rect.width - (pad * 2) - labelW;
    const lanes = [
      { x: pad + labelW, y: pad, w: laneW, h: laneH, deck: 'a', key: 'outgoing', track: metadata.from_track || {} },
      { x: pad + labelW, y: pad + laneH + gap, w: laneW, h: laneH, deck: 'b', key: 'incoming', track: metadata.to_track || {} },
    ];
    const render = metadata.render || {};
    const bars = Math.max(1, Number(render.timeline_bars || ((render.front_padding_bars || 0) + (render.overlap_bars || 1) + (render.back_padding_bars || 0))));
    ctx.fillStyle = '#121212';
    ctx.fillRect(0, 0, rect.width, rect.height);
    lanes.forEach(lane => {
      ctx.fillStyle = '#1d1d1d';
      ctx.fillRect(lane.x, lane.y, lane.w, lane.h);
      for (let i = 0; i <= bars; i += 1) {
        const x = lane.x + (lane.w * i / bars);
        ctx.strokeStyle = i % 4 === 0 ? '#444' : '#2d2d2d';
        ctx.lineWidth = i % 4 === 0 ? 1.2 : 1;
        ctx.beginPath();
        ctx.moveTo(x, lane.y);
        ctx.lineTo(x, lane.y + lane.h);
        ctx.stroke();
      }
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = '#f2f2f2';
      ctx.font = '13px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.fillText(lane.track.title || lane.key, lane.x - 12, lane.y + lane.h * 0.44);
      ctx.fillStyle = '#8f8f8f';
      ctx.font = '11px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.fillText((lane.track.artists || '') + (lane.track.cue_name ? ' / ' + lane.track.cue_name : ''), lane.x - 12, lane.y + lane.h * 0.56);
      drawWave(ctx, (waveform.waveforms || {})[lane.key] || [], lane);
    });
    const auto = waveform.automation || {};
    if (auto.volume) {
      drawLine(ctx, auto.volume.outgoing, lanes[0], '#46c7f3', 3);
      drawLine(ctx, auto.volume.incoming, lanes[1], '#46c7f3', 3);
    }
    if (auto.eq) {
      drawLine(ctx, auto.eq.outgoing_low_gain, lanes[0], '#ffd13d', 2.5);
      drawLine(ctx, auto.eq.incoming_low_gain, lanes[1], '#ffd13d', 2.5);
    }
    if (auto.filter) {
      drawLine(ctx, auto.filter.outgoing_cutoff_hz, lanes[0], '#e58cff', 2.5, false, 'value', 'normalized');
      drawLine(ctx, auto.filter.incoming_cutoff_hz, lanes[1], '#e58cff', 2.5, false, 'value', 'normalized');
    }
    if (dirty) {
      const modes = resolvedModes(body);
      const front = Math.max(0, Number(body.front_padding_bars) || 0);
      const overlap = Math.max(1, Number(body.overlap_bars) || 1);
      const back = Math.max(0, Number(body.back_padding_bars) || 0);
      const total = front + overlap + back;
      const timelineValue = (mode, deck, t, frontValue, backValue) => {
        const bar = t * total;
        if (bar < front) return frontValue;
        if (bar > front + overlap) return backValue;
        const x = Math.max(0, Math.min(1, (bar - front) / overlap));
        return valueFor(mode, deck, x);
      };
      const volumeMode = 'volume:' + modes.volume;
      const eqMode = 'eq:' + modes.eq;
      const filterMode = 'filter:' + modes.filter;
      drawDashedLine(fctx, lanes[0], '#48c8f2', t => timelineValue(volumeMode, 'a', t, 1, 0));
      drawDashedLine(fctx, lanes[1], '#48c8f2', t => timelineValue(volumeMode, 'b', t, 0, 1));
      drawDashedLine(fctx, lanes[0], '#ffd13d', t => timelineValue(eqMode, 'a', t, 1, 1));
      drawDashedLine(fctx, lanes[1], '#ffd13d', t => timelineValue(eqMode, 'b', t, 1, 1));
      drawDashedLine(fctx, lanes[0], '#e58cff', t => timelineValue(filterMode, 'a', t, null, null));
      drawDashedLine(fctx, lanes[1], '#e58cff', t => timelineValue(filterMode, 'b', t, null, null));
      fctx.fillStyle = 'rgba(242,242,242,.82)';
      fctx.font = '12px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      fctx.fillText('unrendered settings preview', lanes[0].x, Math.max(18, lanes[0].y - 12));
    }
    drawPlayhead(fctx, lanes);
  }
  els.fromTrack.addEventListener('change', () => { fillCues(); drawAutomationPreview(); });
  els.toTrack.addEventListener('change', () => { fillCues(); drawAutomationPreview(); });
  els.fromCue.addEventListener('change', drawAutomationPreview);
  els.toCue.addEventListener('change', drawAutomationPreview);
  els.paddingBars.addEventListener('change', () => {
    els.frontPaddingBars.value = els.paddingBars.value;
    els.backPaddingBars.value = els.paddingBars.value;
    drawAutomationPreview();
  });
  els.preset.addEventListener('change', () => {
    applyPresetToControls(els.preset.value);
    drawAutomationPreview();
  });
  els.preset.addEventListener('input', () => {
    applyPresetToControls(els.preset.value);
    drawAutomationPreview();
  });
  [els.overlapBars, els.frontPaddingBars, els.backPaddingBars, els.fromNudgeBeats, els.toNudgeBeats, els.fromPitchShift, els.toPitchShift].forEach(el => {
    el.addEventListener('change', drawAutomationPreview);
    el.addEventListener('input', drawAutomationPreview);
  });
  [els.volumeMode, els.eqMode, els.filterMode].forEach(el => {
    el.addEventListener('change', () => { setCustomPreset(); drawAutomationPreview(); });
    el.addEventListener('input', () => { setCustomPreset(); drawAutomationPreview(); });
  });
  window.addEventListener('resize', drawAutomationPreview);
  els.render.addEventListener('click', renderTransition);
  els.swap.addEventListener('click', () => {
    const fromTrack = els.fromTrack.value;
    const fromCue = els.fromCue.value;
    els.fromTrack.value = els.toTrack.value;
    els.toTrack.value = fromTrack;
    fillCues();
    if ([...els.toCue.options].some(option => option.value === fromCue)) els.toCue.value = fromCue;
  });
  loadInitial().catch(err => setStatus(err.message || String(err), true));
})();
