(function() {
  const records = JSON.parse(document.getElementById('seq-records-json').textContent);
  const simMap = JSON.parse(document.getElementById('seq-sim-json').textContent);
  const idxToPoint = JSON.parse(document.getElementById('seq-points-json').textContent);
  const layoutEntries = JSON.parse(document.getElementById('seq-layout-entries-json').textContent);
  const config = JSON.parse(document.getElementById('seq-config-json').textContent);
  const plot = document.getElementById('__PLOT_ID__');
  const appSettings = config.app_settings || {};
  appSettings.defaults = Object.assign({
    latent_links_per_track: 3,
    recommended_links_highlight: 25,
    point_color: 'genre',
    loudness_match_mode: 'auditions_transitions',
    map_fx: true,
    show_score_values: true,
    figure_light_mode: false,
  }, appSettings.defaults || {});
  appSettings.current = Object.assign({}, appSettings.defaults, appSettings.current || {});
  appSettings.behavior = Object.assign({
    latent_links: 'Top-K weighted candidate links per track, using the current transition weight sliders.',
    recommended_links: "Current selected track's ranked next-track recommendations highlighted on the map.",
  }, appSettings.behavior || {});
  config.app_settings = appSettings;

  const roleColors = {
    track1: '#38bdf8',
    track1Fill: 'rgba(56, 189, 248, .16)',
    track1Line: 'rgba(56, 189, 248, __A__)',
    track2: '#c084fc',
    track2Fill: 'rgba(192, 132, 252, .16)',
    track2Line: 'rgba(192, 132, 252, __A__)',
    recommendation: '#34d399',
    recommendationLine: 'rgba(52, 211, 153, __A__)',
    pinned: '#fbbf24',
    selected: '#f8fafc',
    sequence: '#fb923c',
    sequenceLine: 'rgba(251, 146, 60, __A__)',
    selectedSlot: '#94a3b8',
    reference: '#fbbf24',
  };
  const DEFAULT_TARGET_ENERGY = null;
  const PLANNING_HISTORY_LIMIT = 100;
  const configuredRhythmTempoWeight = Number(config.weights && config.weights.rhythm_tempo_weight);
  const RHYTHM_TEMPO_WEIGHT = Number.isFinite(configuredRhythmTempoWeight)
    ? clamp(configuredRhythmTempoWeight, 0, 1)
    : 0.70;
  const ENERGY_AXIS_TICKS = [1, 3, 5, 7, 9];
  const MAP_MIN_ZOOM_FRACTION = 0.045;
  const MAP_MAX_ZOOM_FRACTION = 1.35;
  const MAP_WHEEL_ZOOM_IN_FACTOR = 0.945;
  const MAP_WHEEL_ZOOM_OUT_FACTOR = 1.06;
  const MAP_BUTTON_ZOOM_IN_FACTOR = 0.92;
  const MAP_BUTTON_ZOOM_OUT_FACTOR = 1.09;
  const MAP_DARK_THEME = {
    bg: '#0b1020',
    panel: '#111827',
    panel2: '#151f32',
    ink: '#e5e7eb',
    muted: '#9ca3af',
    grid: '#263244',
    hoverBg: '#0f172a',
    hoverBorder: '#475569',
    mapClear: [0.066, 0.094, 0.145, 1],
    mapLineRgb: [0.9725, 0.9804, 0.9882],
    miniBg: 'rgba(15, 23, 42, .92)',
    miniViewStroke: 'rgba(248, 250, 252, .88)',
    staticBadgeFill: 'rgba(15, 23, 42, .82)',
    staticBadgeLabel: '#f8fafc',
    selected: '#f8fafc',
  };
  const MAP_LIGHT_THEME = {
    bg: '#f8fafc',
    panel: '#ffffff',
    panel2: '#f1f5f9',
    ink: '#0f172a',
    muted: '#475569',
    grid: '#cbd5e1',
    hoverBg: '#ffffff',
    hoverBorder: '#94a3b8',
    mapClear: [0.9725, 0.9804, 0.9882, 1],
    mapLineRgb: [0.0588, 0.0902, 0.1647],
    miniBg: 'rgba(255, 255, 255, .92)',
    miniViewStroke: 'rgba(15, 23, 42, .74)',
    staticBadgeFill: 'rgba(255, 255, 255, .88)',
    staticBadgeLabel: '#0f172a',
    selected: '#0f172a',
  };
  const DIAGNOSTIC_CHROMA_HEATMAP_STOPS = [
    [0, '#000004'],
    [0.18, '#3b0f70'],
    [0.42, '#8c2981'],
    [0.66, '#de4968'],
    [0.84, '#fe9f6d'],
    [1, '#fcfdbf'],
  ];
  const DIAGNOSTIC_GROOVE_HEATMAP_STOPS = DIAGNOSTIC_CHROMA_HEATMAP_STOPS;

  const byIdx = new Map(records.map(r => [Number(r.idx), r]));
  const initialRecommendationOrder = records.map(r => Number(r.idx)).filter(Number.isFinite);
  for (let i = initialRecommendationOrder.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmp = initialRecommendationOrder[i];
    initialRecommendationOrder[i] = initialRecommendationOrder[j];
    initialRecommendationOrder[j] = tmp;
  }
  let selectedIdx = null;
  let recommendationReferenceIdx = null;
  let hoveredIdx = null;
  let selectedSlot = null;
  let sequenceLength = Number(config.default_length || 10);
  let sequence = Array.from({ length: sequenceLength }, () => null);
  let targetValues = Array.from({ length: sequenceLength }, () => DEFAULT_TARGET_ENERGY);
  let planningUndoStack = [];
  let planningRedoStack = [];
  let planningEditSnapshot = null;
  let planningSaveTimer = null;
  let sequenceDragSlot = null;
  let draggingEnergySlot = null;
  let currentPoints = [];
  let activePane = 'explore';
  let transitionFromIdx = null;
  let transitionToIdx = null;
  let webglMap = null;
  let mapRelayoutClampActive = false;
  const pinnedRecommendationIdxs = new Set();
  let lastClickedIdx = null;
  let lastClickMs = 0;
  let lastPlotPointClickMs = 0;
  let libraryQuery = '';
  let librarySort = null;
  let libraryDir = null;
  let appTracks = [];
  let appTracksById = new Map();
  let appOptions = null;
  let appLoadStarted = false;
  let appLoadDone = false;
  let transitionRender = null;
  let transitionRenderedState = null;
  let transitionRenderPending = false;
  let transitionRenderMessage = '';
  let transitionRenderWarn = false;
  let transitionRenderRequestId = 0;
  let transitionPreviewMode = 'verified';
  let liveTransition = null;
  let liveTransitionPending = false;
  let liveTransitionMessage = '';
  let liveTransitionError = false;
  let liveTransitionRequestId = 0;
  let livePreparationTimer = null;
  let liveAudioContext = null;
  let liveMasterGain = null;
  let liveCompressor = null;
  let livePlayback = null;
  let liveTransportPosition = 0;
  let liveAutomationRescheduleFrame = null;
  let liveSeekFrame = null;
  let livePlaybackStartRequestId = 0;
  let liveSeekPending = null;
  let liveSeekInFlight = false;
  let transitionLoopEnabled = true;
  let liveMixerOpen = true;
  let liveCrossfader = 0;
  const LIVE_MIXER_LANES = ['volume', 'eq_low', 'eq_mid', 'eq_high', 'filter', 'delay', 'reverb'];
  const LIVE_CHANNEL_LANES = ['volume', 'eq_low', 'eq_mid', 'eq_high', 'filter'];
  const LIVE_FX_LANES = ['delay', 'reverb'];
  // Rotary controls use relative movement, so their full range takes a
  // deliberate, encoder-like gesture instead of the knob diameter.
  const LIVE_KNOB_TRAVEL_PX = 130;
  const BEAT_JUMP_SIZES = [.125, .25, .5, 1, 2, 4, 8, 16, 32, 64];
  const transitionBeatJumpSizes = { from: 4, to: 4 };
  const liveRoughOffsets = { from: 0, to: 0 };
  const LIVE_AUTOMATION_HORIZON_SECONDS = 12;
  const LIVE_AUTOMATION_REFRESH_SECONDS = 3;
  const liveManualOverrides = { from: {}, to: {} };
  const liveBufferCache = new Map();
  let sharedAudioContext = null;
  let sharedAudioSource = null;
  let sharedAudioLoudnessGain = null;
  let sharedAudioMasterGain = null;
  let mapTrailSegments = [];
  let mapEffectsFrame = null;
  let previewAudioIdx = null;
  let previewAudioPlaying = false;
  let previewAudioContext = '';
  let previewAudioUrl = '';
  let previewAudioStart = 0;
  let masterVolume = 0.85;
  let lastNonzeroVolume = 0.85;
  let rowScrubPositions = new Map();
  let transitionScrubFrame = null;
  let transitionEditorFrame = null;
  let transitionScrubDragging = false;
  let transitionSnapToBeat = true;
  let transitionEditorDrag = null;
  let transitionViewport = null;
  let transitionReferenceLocked = false;
  let recFilters = { query: '', sameKey: false, bpmRange: '', energyRange: '', excludeUsed: true, genreMode: 'any' };
  let waveformState = { idx: null, url: '', payload: null, loading: false, error: '' };
  const waveformCache = new Map();
  let waveformPointerActive = false;
  let rowWaveformPointerIdx = null;
  let diagnosticWaveformPointerIdx = null;
  let audioProgressFrame = null;
  let lastPointDoubleClickMs = 0;
  let lastPointDoubleClickIdx = null;
  let previewFormState = {
    from_cue: '',
    to_cue: '',
    overlap_bars: 16,
    front_padding_bars: 2,
    back_padding_bars: 2,
    padding_bars: 2,
    from_nudge_beats: 0,
    to_nudge_beats: 0,
    from_beatgrid_ms: 0,
    to_beatgrid_ms: 0,
    from_pitch_shift: 0,
    to_pitch_shift: 0,
    preset: 'auto',
    quality: 'preview',
    volume_mode: 'overlap-crossfade',
    eq_mode: 'center-bass-swap',
    filter_mode: 'none',
    effects: {
      from: { delay_beats: .5, delay_tone: .5, reverb_decay: 1.2, reverb_tone: .5 },
      to: { delay_beats: .5, delay_tone: .5, reverb_decay: 1.2, reverb_tone: .5 },
    },
  };
  let transitionAutomation = null;
  let transitionAutomationPreset = 'auto';
  let transitionAutomationActiveLane = null;
  let transitionAutomationSelectedPoint = null;
  let transitionAutomationSelectedSegment = null;
  let transitionAutomationDrag = null;
  let transitionAutomationHistory = [];
  let transitionAutomationHistoryIndex = -1;
  let transitionAutomationSnapTime = 'beat';
  let transitionAutomationSnapValue = '1';

  const els = {
    sequenceLength: document.getElementById('sequence-length'),
    energySource: document.getElementById('energy-source'),
    colorMode: document.getElementById('color-mode'),
    mapEffectsEnabled: document.getElementById('map-effects-enabled'),
    showScoreValues: document.getElementById('show-score-values'),
    figureLightMode: document.getElementById('figure-light-mode'),
    loudnessMatchSetting: document.getElementById('loudness-match-setting'),
    loudnessMatchMode: document.getElementById('loudness-match-mode'),
    latentLinksPerTrack: document.getElementById('latent-links-per-track'),
    latentLinksPerTrackVal: document.getElementById('latent-links-per-track-val'),
    recommendedLinksHighlight: document.getElementById('recommended-links-highlight'),
    recommendedLinksHighlightVal: document.getElementById('recommended-links-highlight-val'),
    penaltyScale: document.getElementById('energy-penalty-scale'),
    penaltyScaleVal: document.getElementById('energy-penalty-scale-val'),
    weightStyle: document.getElementById('weight-style'),
    weightRhythm: document.getElementById('weight-rhythm'),
    weightHarmony: document.getElementById('weight-harmony'),
    weightStyleVal: document.getElementById('weight-style-val'),
    weightRhythmVal: document.getElementById('weight-rhythm-val'),
    weightHarmonyVal: document.getElementById('weight-harmony-val'),
    simplex: document.getElementById('simplex-control'),
    simplexHandle: document.getElementById('simplex-handle'),
    fullscreenToggle: document.getElementById('fullscreen-toggle'),
    helpToggle: document.getElementById('help-toggle'),
    helpClose: document.getElementById('help-close'),
    helpPopover: document.getElementById('help-popover'),
    helpContent: document.getElementById('help-content'),
    settingsToggle: document.getElementById('settings-toggle'),
    settingsClose: document.getElementById('settings-close'),
    settingsPopover: document.getElementById('settings-popover'),
    settingsContent: document.getElementById('settings-content'),
    tabButtons: Array.from(document.querySelectorAll('[data-pane-tab]')),
    panes: {
      explore: document.getElementById('pane-explore'),
      library: document.getElementById('pane-library'),
      diagnostics: document.getElementById('pane-diagnostics'),
      preview: document.getElementById('pane-preview'),
    },
    transitionBadges: document.getElementById('transition-badges'),
    transitionPreview: document.getElementById('transition-preview-panel'),
    setOutgoing: document.getElementById('set-outgoing'),
    setIncoming: document.getElementById('set-incoming'),
    songPopover: document.getElementById('song-popover'),
    selectedTrack: document.getElementById('selected-track'),
    audio: document.getElementById('track-audio'),
    songPlayerToggle: document.getElementById('song-player-toggle'),
    songPlayerStatus: document.getElementById('song-player-status'),
    songPlayerTime: document.getElementById('song-player-time'),
    songWaveform: document.getElementById('song-waveform'),
    appendSelected: document.getElementById('append-selected'),
    clearTransition: document.getElementById('clear-transition'),
    clearLast: document.getElementById('clear-last'),
    resetSequence: document.getElementById('reset-sequence'),
    undoPlanning: document.getElementById('undo-planning'),
    redoPlanning: document.getElementById('redo-planning'),
    applyEnergyGuides: document.getElementById('apply-energy-guides'),
    clearEnergyTargets: document.getElementById('clear-energy-targets'),
    downloadSequence: document.getElementById('download-sequence'),
    planningSaveStatus: document.getElementById('planning-save-status'),
    energyCurve: document.getElementById('energy-curve'),
    sequenceList: document.getElementById('sequence-list'),
    recommendationPanel: document.getElementById('recommendation-panel'),
    transitionDiagnostics: document.getElementById('transition-diagnostics'),
    currentTransitionScore: document.getElementById('current-transition-score'),
    mapEffects: document.getElementById('map-effects-canvas'),
    webglMap: document.getElementById('webgl-map-canvas'),
    mapMiniMap: document.getElementById('map-mini-map'),
    mapZoomControls: document.getElementById('map-zoom-controls'),
    mapColorLegend: document.getElementById('map-color-legend'),
    songHoverCard: document.getElementById('song-hover-card'),
    librarySearch: document.getElementById('library-search'),
    libraryTable: document.getElementById('library-table'),
    globalPlayer: document.getElementById('global-player'),
    globalArt: document.getElementById('global-art'),
    globalTitle: document.getElementById('global-title'),
    globalArtist: document.getElementById('global-artist'),
    globalToggle: document.getElementById('global-toggle'),
    globalScrub: document.getElementById('global-scrub'),
    globalVolume: document.getElementById('global-volume'),
    globalVolumeButton: document.getElementById('global-volume-button'),
    globalTime: document.getElementById('global-time'),
    globalTrack1: document.getElementById('global-track1'),
    globalTrack2: document.getElementById('global-track2'),
  };

  const baseTraceIndices = [];
  const originalBaseMarkers = new Map();
  const baseTraceGenreColors = new Map();
  if (plot && Array.isArray(plot.data)) {
    for (let i = 0; i < plot.data.length; i += 1) {
      const trace = plot.data[i] || {};
      const custom = Array.isArray(trace.customdata) ? trace.customdata : [];
      const isBaseTrackTrace = custom.length > 0 && Array.isArray(custom[0]) && custom[0].length >= 12;
      if (isBaseTrackTrace) {
        baseTraceIndices.push(i);
        originalBaseMarkers.set(i, JSON.parse(JSON.stringify(trace.marker || {})));
        if (trace.name && trace.marker && trace.marker.color) {
          baseTraceGenreColors.set(String(trace.name), String(trace.marker.color));
        }
      }
    }
  }

  function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
  function finiteOr(v, fallback) { const n = Number(v); return Number.isFinite(n) ? n : fallback; }
  function hasNumericValue(value) { return value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value)); }
  function fmt(v, d=3) { return hasNumericValue(v) ? Number(v).toFixed(d) : ''; }
  function fmtEnergy(v) { return hasNumericValue(v) ? Number(v).toFixed(1) : '—'; }
  function planningStorageKey() {
    const identity = records.map(record => String(record.track_id || record.idx)).join('|');
    let hash = 2166136261;
    for (let i = 0; i < identity.length; i += 1) {
      hash ^= identity.charCodeAt(i);
      hash = Math.imul(hash, 16777619);
    }
    return 'djprojectexploration.sequence-plan.v1.' + (hash >>> 0).toString(36);
  }
  function planningSnapshot() {
    return {
      sequenceLength,
      sequence: sequence.map(value => hasNumericValue(value) ? Number(value) : null),
      targetValues: targetValues.map(value => hasNumericValue(value) ? normalizeTargetEnergy(value, null) : null),
      selectedSlot: hasNumericValue(selectedSlot) ? Number(selectedSlot) : null,
      selectedIdx: hasNumericValue(selectedIdx) ? Number(selectedIdx) : null,
      recommendationReferenceIdx: hasNumericValue(recommendationReferenceIdx) ? Number(recommendationReferenceIdx) : null,
      transitionFromIdx: hasNumericValue(transitionFromIdx) ? Number(transitionFromIdx) : null,
      transitionToIdx: hasNumericValue(transitionToIdx) ? Number(transitionToIdx) : null,
    };
  }
  function samePlanningSnapshot(a, b) { return JSON.stringify(a) === JSON.stringify(b); }
  function normalizePlanningTrack(value, seen) {
    if (!hasNumericValue(value)) return null;
    const idx = Number(value);
    if (!Number.isFinite(idx) || !byIdx.has(idx) || seen.has(idx)) return null;
    seen.add(idx);
    return idx;
  }
  function applyPlanningSnapshot(snapshot) {
    if (!snapshot || typeof snapshot !== 'object') return false;
    const length = clamp(Math.round(Number(snapshot.sequenceLength || sequenceLength)), 2, 40);
    const seen = new Set();
    const sourceSequence = Array.isArray(snapshot.sequence) ? snapshot.sequence : [];
    const sourceTargets = Array.isArray(snapshot.targetValues) ? snapshot.targetValues : [];
    sequenceLength = length;
    sequence = Array.from({ length }, (_, slot) => normalizePlanningTrack(sourceSequence[slot], seen));
    targetValues = Array.from({ length }, (_, slot) => normalizeTargetEnergy(sourceTargets[slot], null));
    const savedSlot = hasNumericValue(snapshot.selectedSlot) ? Number(snapshot.selectedSlot) : NaN;
    selectedSlot = Number.isFinite(savedSlot) && savedSlot >= 0 && savedSlot < length ? Math.round(savedSlot) : null;
    const savedSelected = hasNumericValue(snapshot.selectedIdx) ? Number(snapshot.selectedIdx) : NaN;
    selectedIdx = Number.isFinite(savedSelected) && byIdx.has(savedSelected) ? savedSelected : null;
    const savedReference = hasNumericValue(snapshot.recommendationReferenceIdx) ? Number(snapshot.recommendationReferenceIdx) : NaN;
    recommendationReferenceIdx = Number.isFinite(savedReference) && byIdx.has(savedReference) ? savedReference : selectedIdx;
    const savedFrom = hasNumericValue(snapshot.transitionFromIdx) ? Number(snapshot.transitionFromIdx) : NaN;
    const savedTo = hasNumericValue(snapshot.transitionToIdx) ? Number(snapshot.transitionToIdx) : NaN;
    transitionFromIdx = Number.isFinite(savedFrom) && byIdx.has(savedFrom) ? savedFrom : null;
    transitionToIdx = Number.isFinite(savedTo) && byIdx.has(savedTo) && savedTo !== transitionFromIdx ? savedTo : null;
    if (els.sequenceLength) els.sequenceLength.value = String(sequenceLength);
    return true;
  }
  function updatePlanningControls() {
    if (els.undoPlanning) els.undoPlanning.disabled = planningUndoStack.length === 0;
    if (els.redoPlanning) els.redoPlanning.disabled = planningRedoStack.length === 0;
    if (els.applyEnergyGuides) els.applyEnergyGuides.disabled = !targetGuideSlots().some(slot => !hasNumericValue(targetValues[slot]));
    if (els.clearEnergyTargets) els.clearEnergyTargets.disabled = !targetValues.some(hasNumericValue);
  }
  function savePlanningWorkspace() {
    planningSaveTimer = null;
    try {
      window.localStorage.setItem(planningStorageKey(), JSON.stringify(planningSnapshot()));
      if (els.planningSaveStatus) els.planningSaveStatus.textContent = 'Saved locally';
    } catch (err) {
      if (els.planningSaveStatus) els.planningSaveStatus.textContent = 'Local save unavailable';
    }
  }
  function schedulePlanningSave() {
    if (planningSaveTimer !== null) window.clearTimeout(planningSaveTimer);
    planningSaveTimer = window.setTimeout(savePlanningWorkspace, 180);
    if (els.planningSaveStatus) els.planningSaveStatus.textContent = 'Saving…';
    updatePlanningControls();
  }
  function restorePlanningWorkspace() {
    try {
      const raw = window.localStorage.getItem(planningStorageKey());
      if (!raw) return false;
      return applyPlanningSnapshot(JSON.parse(raw));
    } catch (err) {
      return false;
    }
  }
  function commitPlanningChange(before) {
    const after = planningSnapshot();
    if (samePlanningSnapshot(before, after)) return false;
    planningUndoStack.push(before);
    if (planningUndoStack.length > PLANNING_HISTORY_LIMIT) planningUndoStack.shift();
    planningRedoStack = [];
    schedulePlanningSave();
    return true;
  }
  function mutatePlanning(mutator, { render=true } = {}) {
    const before = planningSnapshot();
    mutator();
    const changed = commitPlanningChange(before);
    if (render) renderAll();
    return changed;
  }
  function undoPlanning() {
    if (!planningUndoStack.length) return;
    const current = planningSnapshot();
    const previous = planningUndoStack.pop();
    planningRedoStack.push(current);
    applyPlanningSnapshot(previous);
    schedulePlanningSave();
    renderAll();
    scheduleLivePreparation();
  }
  function redoPlanning() {
    if (!planningRedoStack.length) return;
    const current = planningSnapshot();
    const next = planningRedoStack.pop();
    planningUndoStack.push(current);
    applyPlanningSnapshot(next);
    schedulePlanningSave();
    renderAll();
  }
  function setRangeProgress(el, value, maxValue) {
    if (!el) return;
    const min = Number(el.min || 0);
    const max = Number(maxValue || el.max || 0);
    const pct = max > min ? clamp(((Number(value) || 0) - min) / (max - min), 0, 1) * 100 : 0;
    el.style.setProperty('--progress', pct.toFixed(2) + '%');
  }
  function applyMasterVolume() {
    masterVolume = clamp(Number(masterVolume), 0, 1);
    if (masterVolume > 0.001) lastNonzeroVolume = masterVolume;
    if (els.audio) els.audio.volume = sharedAudioMasterGain ? 1 : masterVolume;
    if (sharedAudioMasterGain && sharedAudioContext) {
      const now = sharedAudioContext.currentTime;
      sharedAudioMasterGain.gain.cancelScheduledValues(now);
      sharedAudioMasterGain.gain.setTargetAtTime(masterVolume, now, .01);
    }
    const transitionAudio = transitionAudioElement();
    if (transitionAudio) transitionAudio.volume = masterVolume;
    if (els.globalVolume) {
      els.globalVolume.value = String(masterVolume);
      setRangeProgress(els.globalVolume, masterVolume, 1);
      els.globalVolume.setAttribute('aria-valuetext', Math.round(masterVolume * 100) + '%');
      els.globalVolume.setAttribute('title', 'Volume ' + Math.round(masterVolume * 100) + '%');
    }
    if (els.globalVolumeButton) {
      const muted = masterVolume <= 0.001;
      els.globalVolumeButton.setAttribute('aria-label', muted ? 'Unmute' : 'Mute');
      els.globalVolumeButton.setAttribute('title', muted ? 'Unmute' : 'Mute');
      els.globalVolumeButton.classList.toggle('is-muted', muted);
    }
  }
  function ensureSharedAudioLoudnessGraph() {
    if (!els.audio || sharedAudioLoudnessGain) return sharedAudioLoudnessGain;
    const Constructor = window.AudioContext || window.webkitAudioContext;
    if (!Constructor) return null;
    sharedAudioContext = new Constructor();
    sharedAudioSource = sharedAudioContext.createMediaElementSource(els.audio);
    sharedAudioLoudnessGain = sharedAudioContext.createGain();
    sharedAudioMasterGain = sharedAudioContext.createGain();
    sharedAudioSource.connect(sharedAudioLoudnessGain);
    sharedAudioLoudnessGain.connect(sharedAudioMasterGain);
    sharedAudioMasterGain.connect(sharedAudioContext.destination);
    applyMasterVolume();
    return sharedAudioLoudnessGain;
  }
  function applySharedAudioLoudness({ smooth=true } = {}) {
    if (!els.audio) return;
    const record = previewAudioIdx === null ? null : byIdx.get(Number(previewAudioIdx));
    const target = loudnessGainLinear(record, 'audition');
    if (target === 1 && !sharedAudioLoudnessGain) return;
    const gain = ensureSharedAudioLoudnessGraph();
    if (!gain || !sharedAudioContext) return;
    const now = sharedAudioContext.currentTime;
    gain.gain.cancelScheduledValues(now);
    gain.gain.setTargetAtTime(target, now, smooth ? .018 : .001);
    const resume = sharedAudioContext.resume();
    if (resume && resume.catch) resume.catch(() => {});
  }
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  }
  function formatSettingValue(value) {
    if (value === null || value === undefined || value === '') return '<span class="muted">None</span>';
    if (typeof value === 'boolean') return value ? 'true' : 'false';
    if (typeof value === 'number') return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(6)));
    if (Array.isArray(value) || typeof value === 'object') {
      return '<code>' + esc(JSON.stringify(value, null, 2)) + '</code>';
    }
    return esc(value);
  }
  function settingsRowsHtml(rows) {
    return '<div class="settings-list">' + rows
      .filter(row => row && row.length >= 2)
      .map(row => '<div class="settings-row"><div class="settings-key">' + esc(row[0]) + '</div><div class="settings-value">' + formatSettingValue(row[1]) + '</div></div>')
      .join('') + '</div>';
  }
  function settingsSectionHtml(title, rows) {
    const filtered = rows.filter(row => row && row.length >= 2);
    if (!filtered.length) return '';
    return '<section class="settings-section"><h3>' + esc(title) + ' <span>' + filtered.length + '</span></h3>' + settingsRowsHtml(filtered) + '</section>';
  }
  function appSetting(key, fallback=null) {
    return appSettings && appSettings.current && appSettings.current[key] !== undefined ? appSettings.current[key] : fallback;
  }
  function setAppSetting(key, value) {
    if (!appSettings.current) appSettings.current = {};
    appSettings.current[key] = value;
  }
  const LOUDNESS_MATCH_MODES = ['off', 'auditions', 'auditions_transitions'];
  const LOUDNESS_TARGET_LUFS = -12;
  const LOUDNESS_MAX_GAIN_DB = 6;
  function loudnessMatchMode() {
    if (!config.app_mode) return 'off';
    const mode = String(appSetting('loudness_match_mode', 'auditions_transitions'));
    return LOUDNESS_MATCH_MODES.includes(mode) ? mode : 'auditions_transitions';
  }
  function loudnessMatchingEnabled(scope='audition') {
    const mode = loudnessMatchMode();
    return mode === 'auditions_transitions' || (scope === 'audition' && mode === 'auditions');
  }
  function loudnessInfo(record) {
    record = canonicalRecord(record);
    const info = record && record.loudness && typeof record.loudness === 'object' ? record.loudness : {};
    const integrated = Number(info.integrated_lufs);
    const shortTermMean = Number(info.short_term_mean_lufs);
    const shortTermStd = Number(info.short_term_std_lu);
    const shortTermRange = Number(info.short_term_range_lu);
    return {
      integrated: Number.isFinite(integrated) ? integrated : null,
      shortTermMean: Number.isFinite(shortTermMean) ? shortTermMean : null,
      shortTermStd: Number.isFinite(shortTermStd) ? shortTermStd : null,
      shortTermRange: Number.isFinite(shortTermRange) ? shortTermRange : null,
    };
  }
  function loudnessGainDb(record, scope='audition') {
    const info = loudnessInfo(record);
    if (!loudnessMatchingEnabled(scope) || info.integrated === null) return 0;
    return clamp(LOUDNESS_TARGET_LUFS - info.integrated, -LOUDNESS_MAX_GAIN_DB, LOUDNESS_MAX_GAIN_DB);
  }
  function loudnessGainLinear(record, scope='audition') {
    return Math.pow(10, loudnessGainDb(record, scope) / 20);
  }
  function loudnessMetaHtml(record, scope='audition') {
    const info = loudnessInfo(record);
    if (info.integrated === null) return '<span class="loudness-meta muted">Loudness unavailable</span>';
    const gain = loudnessGainDb(record, scope);
    const details = [
      'Integrated ' + info.integrated.toFixed(1) + ' LUFS',
      info.shortTermMean === null ? '' : 'short-term mean ' + info.shortTermMean.toFixed(1) + ' LUFS',
      info.shortTermStd === null ? '' : 'short-term variation ' + info.shortTermStd.toFixed(1) + ' LU',
      info.shortTermRange === null ? '' : 'short-term range ' + info.shortTermRange.toFixed(1) + ' LU',
    ].filter(Boolean).join(' · ');
    return '<span class="loudness-meta" title="' + esc(details) + '">Loudness ' + esc(info.integrated.toFixed(1)) + ' LUFS · match ' + esc(signedFmt(gain, 1)) + ' dB <span aria-hidden="true">ⓘ</span></span>';
  }
  function currentUiSettingsRows() {
    const row = (label, value) => [label, value];
    return [
      row('Active pane', activePane),
      row('Track count', records.length),
      row('Sequence length', sequenceLength),
      row('Energy source', els.energySource ? els.energySource.value : ''),
      row('Point color', appSetting('point_color', 'genre')),
      row('Map effects', Boolean(appSetting('map_fx', true))),
      row('Figure light mode', figureLightMode()),
      row('Loudness match', titleCaseOption(loudnessMatchMode())),
      row('Latent links per track', appSetting('latent_links_per_track', 3)),
      row('Recommended links highlighted', appSetting('recommended_links_highlight', 25)),
      row('Show recommendation scores', Boolean(appSetting('show_score_values', true))),
      row('Energy penalty scale', els.penaltyScale ? Number(els.penaltyScale.value) : null),
      row('Current weights', weights()),
      row('Preview preset', titleCaseOption(previewFormState.preset)),
      row('Preview quality', previewFormState.quality),
      row('Track 1 beatgrid offset (ms)', previewFormState.from_beatgrid_ms),
      row('Track 2 beatgrid offset (ms)', previewFormState.to_beatgrid_ms),
      row('Preview volume mode', titleCaseOption(previewFormState.volume_mode)),
      row('Preview Low EQ mode', titleCaseOption(previewFormState.eq_mode)),
      row('Preview filter mode', titleCaseOption(previewFormState.filter_mode)),
    ];
  }
  function renderSettingsPanel() {
    if (!els.settingsContent) return;
    const generator = config.generator_settings || {};
    const sections = [
      settingsSectionHtml('Current UI', currentUiSettingsRows()),
      settingsSectionHtml('Visualization settings payload', [
        ['Preset source', appSettings.preset_source || 'None'],
        ['Defaults', appSettings.defaults || {}],
        ['Current', appSettings.current || {}],
      ]),
      settingsSectionHtml('Map link behavior', [
        ['Latent links', appSettings.behavior && appSettings.behavior.latent_links],
        ['Recommended links', appSettings.behavior && appSettings.behavior.recommended_links],
      ]),
      settingsSectionHtml('Generator arguments', [
        ['Control mode', config.control_mode],
        ['Static layout', Boolean(config.static_layout)],
        ['Default length', config.default_length],
        ['Temperature', config.temperature],
        ['Top recommendation rows', config.top_k_rows],
        ['App mode', Boolean(config.app_mode)],
        ['Layout selection mode', config.layout_selection_mode || 'interpolated'],
        ['Simplex grid step', generator.step],
        ['Mix slugs', generator.mix_slugs],
        ['Default weights', config.weights],
      ]),
      settingsSectionHtml('PaCMAP layout', Object.entries(generator.pacmap || {}).map(([key, value]) => [key, value])),
      settingsSectionHtml('Similarity matrices', Object.entries(generator.similarity || {}).map(([key, value]) => [key, value])),
      settingsSectionHtml('Paths and exports', Object.entries(generator.paths || {}).map(([key, value]) => [key, value])),
      settingsSectionHtml('Runtime', [
        ['Layout entries', Array.isArray(layoutEntries) ? layoutEntries.length : 0],
        ['App runtime loaded', Boolean(appLoadDone)],
        ['Runtime error', appOptions && appOptions.error ? appOptions.error : ''],
      ]),
    ];
    els.settingsContent.innerHTML = sections.join('');
  }
  function helpKeyHtml(key, text) {
    return '<li><span class="help-key">' + esc(key) + '</span><span>' + esc(text) + '</span></li>';
  }
  function helpCardHtml(title, rows) {
    return '<section class="help-card"><h3>' + esc(title) + '</h3><ul class="help-list">' + rows.join('') + '</ul></section>';
  }
  function renderHelpPanel() {
    if (!els.helpContent) return;
    els.helpContent.innerHTML = '<div class="help-grid">' +
      helpCardHtml('Map navigation', [
        helpKeyHtml('wheel', 'Zoom the map under the pointer.'),
        helpKeyHtml('drag', 'Pan the map.'),
        helpKeyHtml('+ / i', 'Zoom in.'),
        helpKeyHtml('-', 'Zoom out.'),
        helpKeyHtml('arrows', 'Pan the map.'),
        helpKeyHtml('R', 'Reset map view to fit the current layout.'),
        helpKeyHtml('mini map', 'Click the mini map to recenter the main map.'),
      ]) +
      helpCardHtml('Track selection', [
        helpKeyHtml('click', 'Select a candidate track and show its detail panel.'),
        helpKeyHtml('double click', 'Set or clear Track 1 as the recommendation source.'),
        helpKeyHtml('1', 'Set the current candidate as Track 1.'),
        helpKeyHtml('2', 'Set the current candidate as Track 2.'),
        helpKeyHtml('hover', 'Show preview without changing selection.'),
        helpKeyHtml('Esc', 'Close open panels.'),
      ]) +
      helpCardHtml('Fullscreen', [
        helpKeyHtml('F', 'Toggle app fullscreen.'),
        helpKeyHtml('F11', 'Toggle fullscreen when the browser allows the shortcut.'),
      ]) +
      helpCardHtml('Sequence workflow', [
        helpKeyHtml('+ button', 'Append or replace with the current candidate.'),
        helpKeyHtml('Track 1', 'Locked source for recommendations and transition scoring.'),
        helpKeyHtml('Track 2', 'Candidate target for detail and transition preview.'),
        helpKeyHtml('Add', 'Promotes the added candidate to Track 1.'),
      ]) +
      helpCardHtml('Display modes', [
        helpKeyHtml('Genre', 'Rolled-up genre categories.'),
        helpKeyHtml('Energy', 'Tagged or auto energy, depending on Energy source.'),
        helpKeyHtml('Target', 'Selected energy minus the target energy for the active slot.'),
        helpKeyHtml('Key wheel', 'Camelot key color; gray means missing or unavailable.'),
        helpKeyHtml('Tempo', 'Estimated BPM.'),
      ]) +
      helpCardHtml('Audio', [
        helpKeyHtml('play', 'Preview the selected or row track.'),
        helpKeyHtml('volume icon', 'Mute or unmute.'),
        helpKeyHtml('volume bar', 'Adjust playback volume.'),
        helpKeyHtml('waveform', 'Click or drag to seek.'),
      ]) +
      '</div>';
  }
  function setHelpPanelOpen(open) {
    if (!els.helpPopover) return;
    if (open) {
      renderHelpPanel();
      els.helpPopover.classList.remove('hidden');
    } else {
      els.helpPopover.classList.add('hidden');
    }
    if (els.helpToggle) els.helpToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
  }
  function toggleHelpPanel() {
    setHelpPanelOpen(!els.helpPopover || els.helpPopover.classList.contains('hidden'));
  }
  function isFullscreenActive() {
    return Boolean(document.fullscreenElement || document.webkitFullscreenElement);
  }
  function syncFullscreenToggle() {
    if (!els.fullscreenToggle) return;
    const active = isFullscreenActive();
    els.fullscreenToggle.textContent = active ? '⤡' : '⛶';
    els.fullscreenToggle.setAttribute('aria-label', active ? 'Exit fullscreen' : 'Enter fullscreen');
    els.fullscreenToggle.setAttribute('aria-pressed', active ? 'true' : 'false');
    els.fullscreenToggle.title = (active ? 'Exit' : 'Enter') + ' fullscreen (F)';
  }
  async function toggleFullscreen() {
    try {
      if (isFullscreenActive()) {
        const exit = document.exitFullscreen || document.webkitExitFullscreen;
        if (!exit) throw new Error('Fullscreen exit is unavailable in this browser.');
        await exit.call(document);
      } else {
        const root = document.documentElement;
        const request = root.requestFullscreen || root.webkitRequestFullscreen;
        if (!request) throw new Error('Fullscreen is unavailable in this browser.');
        await request.call(root);
      }
    } catch (error) {
      console.error('fullscreen toggle:', error);
    }
  }
  function setSettingsPanelOpen(open) {
    if (!els.settingsPopover || !els.settingsToggle) return;
    els.settingsPopover.classList.toggle('hidden', !open);
    els.settingsToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open) renderSettingsPanel();
  }
  function toggleSettingsPanel() {
    const isOpen = els.settingsPopover && !els.settingsPopover.classList.contains('hidden');
    setSettingsPanelOpen(!isOpen);
  }
  function reportUiError(scope, err) {
    const message = err && err.message ? err.message : String(err || 'Unknown error');
    console.error(scope + ': ' + message, err);
    const status = document.getElementById('transition-render-status');
    if (status && String(scope || '').toLowerCase().includes('transition')) {
      status.textContent = scope + ': ' + message;
      status.classList.add('warn');
    }
  }
  function safeUi(scope, fn, fallback=null) {
    try {
      return fn();
    } catch (err) {
      reportUiError(scope, err);
      return fallback;
    }
  }
  function figureLightMode() {
    return Boolean(appSetting('figure_light_mode', els.figureLightMode ? els.figureLightMode.checked : false));
  }
  function mapTheme() {
    return figureLightMode() ? MAP_LIGHT_THEME : MAP_DARK_THEME;
  }
  function applyFigureTheme({ renderEnergy=false } = {}) {
    const theme = mapTheme();
    document.body.classList.toggle('figure-light', figureLightMode());
    if (window.Plotly && plot) {
      Plotly.relayout(plot, {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        'font.color': theme.ink,
        'hoverlabel.bgcolor': theme.hoverBg,
        'hoverlabel.bordercolor': theme.hoverBorder,
        'hoverlabel.font.color': theme.ink,
      });
      restyleTrace('Hovered', {
        marker: [{ size: 18, color: figureLightMode() ? 'rgba(15,23,42,0.03)' : 'rgba(255,255,255,0.02)', line: { color: theme.selected, width: 2 } }],
      });
      restyleTrace('Selected', {
        marker: [{ size: 24, color: figureLightMode() ? 'rgba(15,23,42,0.08)' : 'rgba(255,255,255,0.16)', line: { color: theme.selected, width: 3 } }],
      });
    }
    if (webglMap) webglMap.render();
    safeUi('mini map theme redraw', drawMiniMap);
    safeUi('map effects theme redraw', ensureMapEffectsLoop);
    if (renderEnergy && activePane === 'explore') safeUi('energy curve render', renderEnergyCurve);
  }
  function usingWebglMap() {
    return Boolean(webglMap && webglMap.available());
  }
  function cssColorToRgba(value, alpha=1) {
    const fallback = [0.58, 0.64, 0.72, alpha];
    const s = String(value || '').trim();
    if (!s) return fallback;
    const hex = s.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
    if (hex) {
      let h = hex[1];
      if (h.length === 3) h = h.split('').map(ch => ch + ch).join('');
      return [
        parseInt(h.slice(0, 2), 16) / 255,
        parseInt(h.slice(2, 4), 16) / 255,
        parseInt(h.slice(4, 6), 16) / 255,
        alpha,
      ];
    }
    const rgb = s.match(/^rgba?\(([^)]+)\)$/i);
    if (rgb) {
      const parts = rgb[1].split(',').map(part => Number(part.trim()));
      if (parts.length >= 3 && parts.slice(0, 3).every(Number.isFinite)) {
        return [
          clamp(parts[0] / 255, 0, 1),
          clamp(parts[1] / 255, 0, 1),
          clamp(parts[2] / 255, 0, 1),
          Number.isFinite(parts[3]) ? clamp(parts[3], 0, 1) : alpha,
        ];
      }
    }
    return fallback;
  }
  function createWebglMapRenderer(canvas) {
    if (!canvas) return null;
    const gl = canvas.getContext('webgl', { alpha: true, antialias: true }) || canvas.getContext('experimental-webgl');
    if (!gl) return null;
    const vertexSource = [
      'attribute vec2 a_position;',
      'attribute vec4 a_color;',
      'uniform vec4 u_view;',
      'uniform float u_point_size;',
      'varying vec4 v_color;',
      'void main() {',
      '  float x = ((a_position.x - u_view.x) / max(0.000001, u_view.y - u_view.x)) * 2.0 - 1.0;',
      '  float y = ((a_position.y - u_view.z) / max(0.000001, u_view.w - u_view.z)) * 2.0 - 1.0;',
      '  gl_Position = vec4(x, y, 0.0, 1.0);',
      '  gl_PointSize = u_point_size;',
      '  v_color = a_color;',
      '}',
    ].join('\n');
    const fragmentSource = [
      'precision mediump float;',
      'varying vec4 v_color;',
      'void main() {',
      '  vec2 p = gl_PointCoord * 2.0 - 1.0;',
      '  float d = dot(p, p);',
      '  if (d > 1.0) discard;',
      '  float edge = 1.0 - smoothstep(0.72, 1.0, d);',
      '  gl_FragColor = vec4(v_color.rgb, v_color.a * edge);',
      '}',
    ].join('\n');
    const lineFragmentSource = [
      'precision mediump float;',
      'varying vec4 v_color;',
      'void main() {',
      '  gl_FragColor = v_color;',
      '}',
    ].join('\n');
    const compileShader = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const message = gl.getShaderInfoLog(shader) || 'Unknown shader compile error';
        gl.deleteShader(shader);
        throw new Error(message);
      }
      return shader;
    };
    const vertexShader = compileShader(gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = compileShader(gl.FRAGMENT_SHADER, fragmentSource);
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program) || 'WebGL program link failed');
    }
    const lineFragmentShader = compileShader(gl.FRAGMENT_SHADER, lineFragmentSource);
    const lineProgram = gl.createProgram();
    gl.attachShader(lineProgram, vertexShader);
    gl.attachShader(lineProgram, lineFragmentShader);
    gl.linkProgram(lineProgram);
    if (!gl.getProgramParameter(lineProgram, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(lineProgram) || 'WebGL line program link failed');
    }
    const positionLoc = gl.getAttribLocation(program, 'a_position');
    const colorLoc = gl.getAttribLocation(program, 'a_color');
    const viewLoc = gl.getUniformLocation(program, 'u_view');
    const pointSizeLoc = gl.getUniformLocation(program, 'u_point_size');
    const linePositionLoc = gl.getAttribLocation(lineProgram, 'a_position');
    const lineColorLoc = gl.getAttribLocation(lineProgram, 'a_color');
    const lineViewLoc = gl.getUniformLocation(lineProgram, 'u_view');
    const linePointSizeLoc = gl.getUniformLocation(lineProgram, 'u_point_size');
    const positionBuffer = gl.createBuffer();
    const colorBuffer = gl.createBuffer();
    const linePositionBuffer = gl.createBuffer();
    const lineColorBuffer = gl.createBuffer();
    let view = null;
    let drag = null;
    let movedDuringDrag = false;
    function normalizedView(next) {
      let x0 = Number(next && (next.x0 ?? next.minX));
      let x1 = Number(next && (next.x1 ?? next.maxX));
      let y0 = Number(next && (next.y0 ?? next.minY));
      let y1 = Number(next && (next.y1 ?? next.maxY));
      if (![x0, x1, y0, y1].every(Number.isFinite)) return null;
      if (Math.abs(x1 - x0) < 1e-9) x1 = x0 + 1;
      if (Math.abs(y1 - y0) < 1e-9) y1 = y0 + 1;
      return { x0, x1, y0, y1 };
    }
    function fitBounds() {
      view = normalizedView(mapDataBounds());
      render();
    }
    function resize() {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(1, Math.floor(Math.max(1, rect.width) * dpr));
      const height = Math.max(1, Math.floor(Math.max(1, rect.height) * dpr));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      gl.viewport(0, 0, canvas.width, canvas.height);
      return { width: Math.max(1, rect.width), height: Math.max(1, rect.height), dpr };
    }
    function dataToCanvasPoint(pt) {
      if (!view || !pt) return null;
      const rect = canvas.getBoundingClientRect();
      const x = Number(pt[0]);
      const y = Number(pt[1]);
      if (![x, y].every(Number.isFinite)) return null;
      return {
        x: ((x - view.x0) / Math.max(1e-9, view.x1 - view.x0)) * rect.width,
        y: (1 - ((y - view.y0) / Math.max(1e-9, view.y1 - view.y0))) * rect.height,
      };
    }
    function screenToData(clientX, clientY) {
      if (!view) return null;
      const rect = canvas.getBoundingClientRect();
      const nx = clamp((Number(clientX) - rect.left) / Math.max(1, rect.width), 0, 1);
      const ny = clamp((Number(clientY) - rect.top) / Math.max(1, rect.height), 0, 1);
      return {
        x: view.x0 + nx * (view.x1 - view.x0),
        y: view.y1 - ny * (view.y1 - view.y0),
        nx,
        ny,
      };
    }
    function hitTest(ev, radius=14) {
      if (!view || !ev) return null;
      const rect = canvas.getBoundingClientRect();
      const x = Number(ev.clientX) - rect.left;
      const y = Number(ev.clientY) - rect.top;
      if (![x, y].every(Number.isFinite)) return null;
      const r2 = radius * radius;
      let bestIdx = null;
      let bestD2 = r2;
      for (const record of records) {
        const p = dataToCanvasPoint(currentPoint(record.idx));
        if (!p) continue;
        const dx = p.x - x;
        const dy = p.y - y;
        const d2 = dx * dx + dy * dy;
        if (d2 <= bestD2) {
          bestD2 = d2;
          bestIdx = Number(record.idx);
        }
      }
      return bestIdx;
    }
    function setViewRanges(x0, x1, y0, y1) {
      const next = normalizedView(constrainMapRanges(x0, x1, y0, y1));
      if (!next) return;
      view = next;
      render();
      safeUi('map effects after webgl view change', ensureMapEffectsLoop);
      safeUi('mini map after webgl view change', drawMiniMap);
    }
    function render() {
      resize();
      if (!view) view = normalizedView(mapDataBounds());
      if (!view) return;
      const mode = normalizePointColorMode(appSetting('point_color', els.colorMode ? els.colorMode.value : 'genre'));
      const extent = mode === 'genre' ? null : colorScaleExtent(mode);
      const positions = new Float32Array(records.length * 2);
      const colors = new Float32Array(records.length * 4);
      records.forEach((record, i) => {
        const pt = currentPoint(record.idx);
        positions[i * 2] = pt ? Number(pt[0]) : 0;
        positions[i * 2 + 1] = pt ? Number(pt[1]) : 0;
        const rgba = cssColorToRgba(miniMapMarkerColor(record, mode, extent), pt ? 0.92 : 0);
        colors.set(rgba, i * 4);
      });
      const pairs = strongestConnectionPairs();
      const linePositions = new Float32Array(pairs.length * 4);
      const lineColors = new Float32Array(pairs.length * 8);
      let lineCount = 0;
      pairs.forEach((pair, i) => {
        const a = currentPoint(pair.src);
        const b = currentPoint(pair.dst);
        if (!a || !b) return;
        const alpha = 0.05 + (1 - i / Math.max(1, pairs.length)) * 0.085;
        const offset = lineCount * 4;
        linePositions[offset] = Number(a[0]);
        linePositions[offset + 1] = Number(a[1]);
        linePositions[offset + 2] = Number(b[0]);
        linePositions[offset + 3] = Number(b[1]);
        const colorOffset = lineCount * 8;
        const lineRgb = mapTheme().mapLineRgb;
        const rgba = [lineRgb[0], lineRgb[1], lineRgb[2], alpha];
        lineColors.set(rgba, colorOffset);
        lineColors.set(rgba, colorOffset + 4);
        lineCount += 1;
      });
      gl.useProgram(program);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      const clear = mapTheme().mapClear;
      gl.clearColor(clear[0], clear[1], clear[2], clear[3]);
      gl.clear(gl.COLOR_BUFFER_BIT);
      if (lineCount > 0) {
        gl.useProgram(lineProgram);
        gl.bindBuffer(gl.ARRAY_BUFFER, linePositionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, linePositions.subarray(0, lineCount * 4), gl.DYNAMIC_DRAW);
        gl.enableVertexAttribArray(linePositionLoc);
        gl.vertexAttribPointer(linePositionLoc, 2, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, lineColorBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, lineColors.subarray(0, lineCount * 8), gl.DYNAMIC_DRAW);
        gl.enableVertexAttribArray(lineColorLoc);
        gl.vertexAttribPointer(lineColorLoc, 4, gl.FLOAT, false, 0, 0);
        gl.uniform4f(lineViewLoc, view.x0, view.x1, view.y0, view.y1);
        gl.uniform1f(linePointSizeLoc, 1);
        gl.lineWidth(1);
        gl.drawArrays(gl.LINES, 0, lineCount * 2);
      }
      gl.useProgram(program);
      gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(positionLoc);
      gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, colors, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(colorLoc);
      gl.vertexAttribPointer(colorLoc, 4, gl.FLOAT, false, 0, 0);
      gl.uniform4f(viewLoc, view.x0, view.x1, view.y0, view.y1);
      gl.uniform1f(pointSizeLoc, 10.5 * (window.devicePixelRatio || 1));
      gl.drawArrays(gl.POINTS, 0, records.length);
    }
    canvas.addEventListener('wheel', ev => {
      if (!usingWebglMap() || !view) return;
      const at = screenToData(ev.clientX, ev.clientY);
      if (!at) return;
      const factor = ev.deltaY < 0 ? MAP_WHEEL_ZOOM_IN_FACTOR : MAP_WHEEL_ZOOM_OUT_FACTOR;
      const width = Math.abs(view.x1 - view.x0) * factor;
      const height = Math.abs(view.y1 - view.y0) * factor;
      const x0 = at.x - at.nx * width;
      const x1 = x0 + width;
      const y1 = at.y + at.ny * height;
      const y0 = y1 - height;
      setViewRanges(x0, x1, y0, y1);
      ev.preventDefault();
    }, { passive: false });
    canvas.addEventListener('pointerdown', ev => {
      if (!usingWebglMap() || !view) return;
      drag = {
        x: ev.clientX,
        y: ev.clientY,
        view: { ...view },
      };
      movedDuringDrag = false;
      try { canvas.setPointerCapture(ev.pointerId); } catch (err) {}
    });
    canvas.addEventListener('pointermove', ev => {
      if (!usingWebglMap()) return;
      if (drag && (ev.buttons & 1)) {
        const rect = canvas.getBoundingClientRect();
        const dx = Number(ev.clientX) - drag.x;
        const dy = Number(ev.clientY) - drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) movedDuringDrag = true;
        const width = drag.view.x1 - drag.view.x0;
        const height = drag.view.y1 - drag.view.y0;
        setViewRanges(
          drag.view.x0 - dx / Math.max(1, rect.width) * width,
          drag.view.x1 - dx / Math.max(1, rect.width) * width,
          drag.view.y0 + dy / Math.max(1, rect.height) * height,
          drag.view.y1 + dy / Math.max(1, rect.height) * height,
        );
        ev.preventDefault();
        return;
      }
      const idx = hitTest(ev);
      if (idx !== null && byIdx.has(idx)) showSongHover(byIdx.get(idx), ev);
      else hideSongHover();
    });
    canvas.addEventListener('pointerup', ev => {
      if (drag) {
        try { canvas.releasePointerCapture(ev.pointerId); } catch (err) {}
      }
      drag = null;
    });
    canvas.addEventListener('pointerleave', () => {
      drag = null;
      if (usingWebglMap()) hideSongHover();
    });
    canvas.addEventListener('click', ev => {
      if (!usingWebglMap()) return;
      if (movedDuringDrag || Number(ev.detail || 0) > 1) {
        movedDuringDrag = false;
        return;
      }
      const idx = hitTest(ev);
      if (idx !== null) handleMapTrackClick(idx, ev);
      else handleMapBlankClick(ev);
    });
    canvas.addEventListener('dblclick', ev => {
      if (!usingWebglMap()) return;
      const idx = hitTest(ev);
      if (idx !== null) assignTransitionFromDoubleClick(idx);
      else handlePlotDoubleClick(false);
      ev.preventDefault();
      ev.stopPropagation();
    });
    return {
      available: () => true,
      render,
      resize: () => { resize(); render(); },
      fitBounds: () => { fitBounds(); safeUi('mini map after webgl fit', drawMiniMap); },
      viewRanges: () => view ? { ...view } : null,
      setViewRanges,
      dataToCanvasPoint,
      hitTest,
    };
  }
  function syncMapRenderer({ render=true } = {}) {
    if (!webglMap) {
      try {
        webglMap = createWebglMapRenderer(els.webglMap);
      } catch (err) {
        reportUiError('webgl map init', err);
        webglMap = null;
      }
    }
    if (!webglMap || !webglMap.available()) {
      reportUiError('webgl map init', 'WebGL is unavailable in this browser.');
      return false;
    }
    try {
      if (!webglMap.viewRanges()) webglMap.fitBounds();
      if (render) webglMap.render();
      return true;
    } catch (err) {
      reportUiError('webgl map render', err);
      return false;
    }
  }
  function durationText(record) {
    if (!record) return '';
    if (record.duration_text) return String(record.duration_text);
    const n = Number(record.duration_seconds);
    if (!Number.isFinite(n) || n <= 0) return '';
    const total = Math.round(n);
    return Math.floor(total / 60) + ':' + String(total % 60).padStart(2, '0');
  }
  function roundedBpm(record) {
    const n = Number(record && record.est_bpm);
    return Number.isFinite(n) ? String(Math.round(n)) : '';
  }
  function camelotKeyColor(value) {
    const m = String(value || '').trim().toUpperCase().match(/^(\d{1,2})[AB]?/);
    if (!m) return '#cbd5e1';
    const palette = {
      1:'#ef4444', 2:'#f97316', 3:'#f59e0b', 4:'#eab308',
      5:'#84cc16', 6:'#22c55e', 7:'#14b8a6', 8:'#06b6d4',
      9:'#3b82f6', 10:'#6366f1', 11:'#a855f7', 12:'#ec4899',
    };
    return palette[Number(m[1])] || '#cbd5e1';
  }
  function keyHtml(key) {
    return '<span class="camelot-key" style="color:' + esc(camelotKeyColor(key)) + '">' + esc(key || '') + '</span>';
  }
  function camelotCanonicalName(value) {
    const m = String(value || '').trim().toUpperCase().match(/^(\d{1,2})([AB])$/);
    if (!m) return '';
    const names = {
      A: { 1:'Abm', 2:'Ebm', 3:'Bbm', 4:'Fm', 5:'Cm', 6:'Gm', 7:'Dm', 8:'Am', 9:'Em', 10:'Bm', 11:'F#m', 12:'C#m' },
      B: { 1:'B', 2:'F#', 3:'Db', 4:'Ab', 5:'Eb', 6:'Bb', 7:'F', 8:'C', 9:'G', 10:'D', 11:'A', 12:'E' },
    };
    return names[m[2]] && names[m[2]][Number(m[1])] ? names[m[2]][Number(m[1])] : '';
  }
  function canonicalKeyText(value) {
    const key = String(value || '').trim();
    if (!key || key.toLowerCase() === 'nan') return '';
    const canonical = camelotCanonicalName(key);
    return canonical ? key.toUpperCase() + ' (' + canonical + ')' : key;
  }
  function shiftCamelotKey(value, semitones=0) {
    const match = String(value || '').trim().toUpperCase().match(/^(\d{1,2})([AB])$/);
    if (!match) return String(value || '').trim().toUpperCase();
    const number = Number(match[1]);
    if (!Number.isFinite(number) || number < 1 || number > 12) return String(value || '').trim().toUpperCase();
    // Match the backend renderer: a chromatic semitone moves seven Camelot
    // positions around the wheel while retaining major/minor mode.
    const shifted = ((number - 1 + (7 * Math.round(Number(semitones) || 0))) % 12 + 12) % 12 + 1;
    return String(shifted) + match[2];
  }
  function transitionDeckKeyHtml(record, pitchShift) {
    const shiftedKey = shiftCamelotKey(record && record.key, pitchShift);
    if (!shiftedKey) return '';
    const shift = Math.round(Number(pitchShift) || 0);
    return keyHtml(shiftedKey) + (shift ? ' <span class="transition-key-shift">(' + esc((shift > 0 ? '+' : '') + shift) + ')</span>' : '');
  }
  function transitionDeckMetaHtml(record, pitchShift) {
    record = canonicalRecord(record);
    if (!record) return '';
    return [
      esc(rawGenre(record)),
      transitionDeckKeyHtml(record, pitchShift),
      esc(roundedBpm(record)),
      energyMetaHtml(record, null, false),
      esc(durationText(record)),
    ].filter(Boolean).join(' &sdot; ');
  }
  function genreMarkerColor(value) {
    const mapped = baseTraceGenreColors.get(String(value || ''));
    if (mapped) return mapped;
    const palette = [
      '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
      '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#393b79', '#637939',
    ];
    const s = String(value || '');
    let hash = 0;
    for (let i = 0; i < s.length; i += 1) hash = ((hash * 31) + s.charCodeAt(i)) >>> 0;
    return palette[hash % palette.length];
  }
  function camelotKeyMarkerColor(value) {
    return camelotNumber(value) === null ? '#64748b' : camelotKeyColor(value);
  }
  function hexToRgb(hex) {
    const clean = String(hex || '').replace('#', '');
    if (clean.length !== 6) return [148, 163, 184];
    return [
      parseInt(clean.slice(0, 2), 16),
      parseInt(clean.slice(2, 4), 16),
      parseInt(clean.slice(4, 6), 16),
    ];
  }
  function rgbToHex(rgb) {
    return '#' + rgb.map(v => clamp(Math.round(v), 0, 255).toString(16).padStart(2, '0')).join('');
  }
  function lerpColor(a, b, t) {
    const ar = hexToRgb(a);
    const br = hexToRgb(b);
    const x = clamp(Number(t), 0, 1);
    return rgbToHex(ar.map((v, i) => v + (br[i] - v) * x));
  }
  function energyResidual(record) {
    const predicted = Number(record && record.glm_energy);
    const tagged = Number(record && record.human_energy);
    return Number.isFinite(predicted) && Number.isFinite(tagged) ? predicted - tagged : NaN;
  }
  function sequenceSlotForRecord(record) {
    const idx = Number(record && record.idx);
    if (!Number.isFinite(idx)) return null;
    const slot = sequence.findIndex(value => Number(value) === idx);
    return slot >= 0 ? slot : null;
  }
  function relevantResidualSlot(record, explicitSlot=null) {
    const slot = Number(explicitSlot);
    if (Number.isFinite(slot) && slot >= 0 && slot < sequenceLength) return slot;
    const sequenceSlot = sequenceSlotForRecord(record);
    if (sequenceSlot !== null) return sequenceSlot;
    const activeSlot = targetSlot();
    return activeSlot >= 0 ? activeSlot : null;
  }
  function targetEnergyResidualForSlot(record, slot) {
    const slotNumber = Number(slot);
    if (!Number.isFinite(slotNumber) || slotNumber < 0) return NaN;
    const target = targetForSlot(slotNumber);
    const value = energyOf(record);
    return Number.isFinite(value) && Number.isFinite(target) ? value - target : NaN;
  }
  function targetEnergyResidual(record) {
    return targetEnergyResidualForSlot(record, Math.max(0, targetSlot()));
  }
  function targetEnergyResidualExtentForSlot(slot) {
    const slotNumber = Number(slot);
    if (!Number.isFinite(slotNumber) || slotNumber < 0) return targetEnergyResidualExtent();
    const vals = records
      .map(record => targetEnergyResidualForSlot(record, slotNumber))
      .filter(Number.isFinite)
      .map(Math.abs);
    const maxAbs = vals.length ? Math.max(0.5, Math.min(4, Math.max(...vals))) : 1;
    return { min: -maxAbs, max: maxAbs, title: 'Energy - target' };
  }
  function targetEnergyResidualExtent() {
    const vals = records.map(targetEnergyResidual).filter(Number.isFinite).map(Math.abs);
    const maxAbs = vals.length ? Math.max(0.5, Math.min(4, Math.max(...vals))) : 1;
    return { min: -maxAbs, max: maxAbs, title: 'Energy - target' };
  }
  function twoSidedResidualColor(value, maxAbs) {
    const v = Number(value);
    if (!Number.isFinite(v)) return '#64748b';
    const m = Math.max(0.25, Number(maxAbs) || 1);
    if (Math.abs(v) <= 0.08) return '#f8fafc';
    if (v < 0) return lerpColor('#f8fafc', '#2563eb', Math.min(1, Math.abs(v) / m));
    return lerpColor('#f8fafc', '#ef4444', Math.min(1, v / m));
  }
  function segmentedColor(stops, t) {
    const x = clamp(Number(t), 0, 1);
    for (let i = 0; i < stops.length - 1; i += 1) {
      const a = stops[i];
      const b = stops[i + 1];
      if (x <= b[0]) {
        const local = (x - a[0]) / Math.max(1e-9, b[0] - a[0]);
        return lerpColor(a[1], b[1], local);
      }
    }
    return stops[stops.length - 1][1];
  }
  function energyColor(value) {
    const t = (Number(value) - 1) / 8;
    if (!Number.isFinite(t)) return '#64748b';
    return segmentedColor([
      [0, '#440154'],
      [0.38, '#31688e'],
      [0.72, '#35b779'],
      [1, '#fde725'],
    ], t);
  }
  function energyTextColor(value) {
    const base = energyColor(value);
    return lerpColor(base, '#ffffff', 0.18);
  }
  function signedFmt(value, digits=2) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '';
    return (n > 0 ? '+' : '') + n.toFixed(digits);
  }
  function tempoColor(value, extent) {
    const min = Number(extent && extent.min);
    const max = Number(extent && extent.max);
    const t = (Number(value) - min) / Math.max(1e-9, max - min);
    if (!Number.isFinite(t)) return '#64748b';
    return segmentedColor([
      [0, '#30123b'],
      [0.2, '#4663d8'],
      [0.42, '#1bcfd4'],
      [0.62, '#a4fc3c'],
      [0.82, '#f89540'],
      [1, '#7a0403'],
    ], t);
  }
  function canonicalRecord(record) {
    if (!record) return null;
    const idx = Number(record.idx);
    if (Number.isFinite(idx) && byIdx.has(idx)) {
      return Object.assign({}, record, byIdx.get(idx));
    }
    return record;
  }
  function artworkHtml(record, cls='') {
    record = canonicalRecord(record);
    const extra = cls ? ' ' + cls : '';
    if (record && record.artwork_uri) {
      return '<img class="art-thumb' + extra + '" src="' + esc(record.artwork_uri) + '" alt="">';
    }
    return '<div class="art-thumb art-placeholder' + extra + '">art</div>';
  }
  function rawGenre(record) {
    record = canonicalRecord(record);
    return titleCaseGenre(String((record && (record.raw_genre || record.genre)) || 'Unknown'));
  }
  function titleCaseGenre(value) {
    return String(value || '').split(/(\s+|\/|-)/).map(part => {
      if (/^\s+$|^\/$|^-$/.test(part)) return part;
      if (!part) return part;
      const lower = part.toLowerCase();
      if (lower === 'dj') return 'DJ';
      if (lower === 'edm') return 'EDM';
      return lower.charAt(0).toUpperCase() + lower.slice(1);
    }).join('');
  }
  function energyMetaHtml(record, slot=null, showResidual=true) {
    record = canonicalRecord(record);
    const value = energyOf(record);
    if (!Number.isFinite(value)) return '';
    const resolvedSlot = relevantResidualSlot(record, slot);
    const residual = showResidual ? targetEnergyResidualForSlot(record, resolvedSlot) : NaN;
    const energy = '<span class="track-energy" style="color:' + esc(energyTextColor(value)) + '">' + esc(fmtEnergy(value)) + '</span>';
    if (!Number.isFinite(residual)) return energy;
    const extent = targetEnergyResidualExtentForSlot(resolvedSlot);
    const maxAbs = Math.max(Math.abs(Number(extent.min) || 0), Math.abs(Number(extent.max) || 0), 1);
    return energy + ' <span class="track-energy-residual" style="color:' + esc(twoSidedResidualColor(residual, maxAbs)) + '">(' + esc(signedFmt(residual, 1)) + ')</span>';
  }
  function trackMetaHtml(record, { slot=null, showEnergy=true, showResidual=true } = {}) {
    record = canonicalRecord(record);
    if (!record) return '';
    const energy = showEnergy ? energyMetaHtml(record, slot, showResidual) : '';
    return [
      esc(rawGenre(record)),
      record.key ? keyHtml(record.key) : '',
      esc(roundedBpm(record)),
      energy,
      esc(durationText(record)),
    ].filter(Boolean).join(' &sdot; ');
  }
  function trackSummaryHtml(record, { size='compact', showArt=true, slot=null, showEnergy=true, showResidual=true } = {}) {
    record = canonicalRecord(record);
    if (!record) return '<span class="muted">None selected</span>';
    const body =
      '<div class="track-title">' + esc(record.title) + '</div>' +
      '<div class="track-artist muted">' + esc(record.artists) + '</div>' +
      '<div class="track-meta-line muted">' + trackMetaHtml(record, { slot, showEnergy, showResidual }) + '</div>' +
      (size === 'large' ? '<div class="track-loudness-line">' + loudnessMetaHtml(record) + '</div>' : '');
    if (!showArt) return body;
    return '<div class="track-summary ' + esc(size) + '">' + artworkHtml(record) + '<div>' + body + '</div></div>';
  }
  function closestAttr(target, attr) {
    const el = target && target.closest ? target.closest('[' + attr + ']') : null;
    return el ? el.getAttribute(attr) : null;
  }
  function closestEl(target, selector) {
    return target && target.closest ? target.closest(selector) : null;
  }
  function isPlotPointTarget(target) {
    return !!closestEl(target, '.point, .points');
  }
  function trackId(record) {
    return record ? String(record.track_id || (record.mix_slug + ':' + record.track_number)) : '';
  }
  function resetTransitionRenderState() {
    stopLiveAudition({ clear: true });
    transitionRender = null;
    transitionRenderedState = null;
    transitionRenderPending = false;
    transitionRenderMessage = '';
    transitionRenderWarn = false;
    transitionRenderRequestId += 1;
    transitionViewport = null;
  }
  function resetTransitionAutomationState() {
    transitionAutomation = null;
    transitionAutomationActiveLane = null;
    transitionAutomationSelectedPoint = null;
    transitionAutomationSelectedSegment = null;
    transitionAutomationDrag = null;
    transitionAutomationHistory = [];
    transitionAutomationHistoryIndex = -1;
    previewFormState.automation = null;
  }
  function appTrack(record) {
    const id = trackId(record);
    return id ? appTracksById.get(id) || null : null;
  }
  function optionHtml(value, label, selectedValue) {
    const selected = String(value) === String(selectedValue) ? ' selected' : '';
    return '<option value="' + esc(value) + '"' + selected + '>' + esc(label) + '</option>';
  }
  function actionSymbolButton(attrs, symbol, label, extraClass='') {
    const cls = 'action-symbol-button' + (extraClass ? ' ' + extraClass : '');
    return '<button class="' + esc(cls) + '" ' + attrs + ' aria-label="' + esc(label) + '" title="' + esc(label) + '"><span aria-hidden="true">' + esc(symbol) + '</span></button>';
  }
  function setActionButton(el, symbol, label, extraClass='') {
    if (!el) return;
    el.classList.add('action-symbol-button');
    if (extraClass) extraClass.split(/\s+/).filter(Boolean).forEach(cls => el.classList.add(cls));
    el.innerHTML = '<span aria-hidden="true">' + esc(symbol) + '</span>';
    el.setAttribute('aria-label', label);
    el.setAttribute('title', label);
  }
  function numericInput(id, label, value, min, max, step) {
    return '<label>' + esc(label) + '<input id="' + id + '" data-preview-field="' + id + '" type="number" min="' + min + '" max="' + max + '" step="' + step + '" value="' + esc(value) + '"></label>';
  }
  const TRANSITION_TIMING_STEPS = {
    overlap_bars: [1, 2, 4, 8, 16, 32, 48, 64],
    front_padding_bars: [0, 1, 2, 4, 8, 16],
    back_padding_bars: [0, 1, 2, 4, 8, 16],
    padding_bars: [0, 1, 2, 4, 8, 16],
  };
  function nearestTransitionTimingValue(field, value) {
    const choices = TRANSITION_TIMING_STEPS[field];
    if (!choices) return Number(value) || 0;
    const target = Number(value);
    if (!Number.isFinite(target)) return choices[0];
    return choices.reduce((best, choice) => Math.abs(choice - target) < Math.abs(best - target) ? choice : best, choices[0]);
  }
  function steppedTimingInput(id, label, value) {
    const choices = TRANSITION_TIMING_STEPS[id];
    const normalized = nearestTransitionTimingValue(id, value);
    return '<label class="preview-step-label">' + esc(label) + '<span class="preview-step-input">' +
      '<button type="button" data-preview-step="' + esc(id) + ':-1" aria-label="Decrease ' + esc(label) + '">−</button>' +
      '<input id="' + esc(id) + '" data-preview-field="' + esc(id) + '" type="number" min="' + choices[0] + '" max="' + choices[choices.length - 1] + '" step="1" value="' + esc(normalized) + '">' +
      '<button type="button" data-preview-step="' + esc(id) + ':1" aria-label="Increase ' + esc(label) + '">+</button></span></label>';
  }
  function setTransitionPadding(value) {
    const next = nearestTransitionTimingValue('padding_bars', value);
    const changed = next !== Number(previewFormState.front_padding_bars) || next !== Number(previewFormState.back_padding_bars);
    previewFormState.padding_bars = next;
    previewFormState.front_padding_bars = next;
    previewFormState.back_padding_bars = next;
    syncPreviewFieldElements('padding_bars', next);
    // Padding changes the visible Front → overlap → Back timeline itself.
    // Reset any zoomed shared viewport so the newly added/removed context is
    // immediately reflected in both deck waveforms and the transport bounds.
    if (changed) transitionViewport = null;
    updatePreviewEffectDisplay({ dirty: true });
  }
  function stepTransitionTimingField(field, direction) {
    const choices = TRANSITION_TIMING_STEPS[field];
    if (!choices) return;
    const current = nearestTransitionTimingValue(field, previewFormState[field]);
    const index = Math.max(0, choices.indexOf(current));
    const next = choices[clamp(index + Math.sign(Number(direction) || 0), 0, choices.length - 1)];
    if (field === 'padding_bars') { setTransitionPadding(next); return; }
    previewFormState[field] = next;
    syncPreviewFieldElements(field, next);
    updatePreviewEffectDisplay({ dirty: true });
  }
  function selectInput(id, label, values, selectedValue, labelFn) {
    const labels = labelFn || (v => v);
    return '<label>' + esc(label) + '<select id="' + id + '" data-preview-field="' + id + '">' +
      values.map(v => optionHtml(v, labels(v), selectedValue)).join('') +
      '</select></label>';
  }
  function titleCaseOption(value) {
    return String(value == null ? '' : value).replace(/[-_]+/g, ' ').replace(/\b\w/g, char => char.toUpperCase());
  }
  function cueOptions(record, role, selectedValue) {
    const track = appTrack(record);
    const cues = track && Array.isArray(track.cues) ? track.cues : [];
    if (!cues.length) return optionHtml('', '(no imported cues)', '');
    const preferredRole = String(role || '').toLowerCase();
    let selected = selectedValue;
    if (!selected) {
      const preferred = cues.find(cue => String(cue.role || '').toLowerCase() === preferredRole) || cues[0];
      selected = preferred ? preferred.name : '';
    }
    return cues.map(cue => {
      const time = cue.start_seconds == null ? '' : ' @ ' + Number(cue.start_seconds).toFixed(2) + 's';
      const roleText = cue.role ? ' / ' + cue.role : '';
      return optionHtml(cue.name, cue.name + roleText + time, selected);
    }).join('');
  }
  function cuesForRecord(record) {
    const track = appTrack(record);
    return track && Array.isArray(track.cues) ? track.cues : [];
  }
  function defaultCueFor(record, role) {
    const cues = cuesForRecord(record);
    if (!cues.length) return null;
    const preferredRole = String(role || '').toLowerCase();
    return cues.find(cue => String(cue.role || '').toLowerCase() === preferredRole) || cues[0] || null;
  }
  function cueByName(record, cueName, role) {
    const cues = cuesForRecord(record);
    const normalized = String(cueName || '').trim().toUpperCase();
    if (normalized) {
      const match = cues.find(cue => String(cue.name || '').trim().toUpperCase() === normalized);
      if (match) return match;
    }
    return defaultCueFor(record, role);
  }
  function cueSelectHtml(record, role, field, label='Cue') {
    return '<label class="transition-cue-control"><span>' + esc(label) + '</span><select data-preview-field="' + esc(field) + '">' +
      cueOptions(record, role, previewFormState[field]) +
      '</select></label>';
  }
  function trackBpm(record) {
    const track = appTrack(record);
    const bpm = Number(track && track.bpm);
    if (Number.isFinite(bpm) && bpm > 0) return bpm;
    const est = Number(record && record.est_bpm);
    if (Number.isFinite(est) && est > 0) return est;
    const csv = Number(record && record.csv_bpm);
    return Number.isFinite(csv) && csv > 0 ? csv : 120;
  }
  function transitionDeckConfig(deck) {
    const isFrom = deck === 'from';
    return {
      deck,
      label: isFrom ? 'Track 1' : 'Track 2',
      role: isFrom ? 'out' : 'in',
      cueField: isFrom ? 'from_cue' : 'to_cue',
      nudgeField: isFrom ? 'from_nudge_beats' : 'to_nudge_beats',
      beatgridField: isFrom ? 'from_beatgrid_ms' : 'to_beatgrid_ms',
      record: isFrom ? (transitionFromIdx === null ? null : byIdx.get(transitionFromIdx)) : (transitionToIdx === null ? null : byIdx.get(transitionToIdx)),
      color: isFrom ? roleColors.track1 : roleColors.track2,
    };
  }
  function transitionWindowInfo(deck) {
    const cfg = transitionDeckConfig(deck);
    const record = cfg.record;
    if (!record) return null;
    const bpm = trackBpm(record);
    const beatSec = 60 / Math.max(1, bpm);
    const beatsPerBar = 4;
    const frontBars = Math.max(0, Number(previewFormState.front_padding_bars) || 0);
    const overlapBars = Math.max(1, Number(previewFormState.overlap_bars) || 1);
    const backBars = Math.max(0, Number(previewFormState.back_padding_bars) || 0);
    const totalBars = frontBars + overlapBars + backBars;
    const cue = cueByName(record, previewFormState[cfg.cueField], cfg.role);
    const cueSeconds = Number(cue && cue.start_seconds);
    const anchor = Number.isFinite(cueSeconds) ? cueSeconds : previewStart(record);
    const nudgeBeats = Number(previewFormState[cfg.nudgeField]) || 0;
    const beatgridMs = Math.round((Number(previewFormState[cfg.beatgridField]) || 0) / 5) * 5;
    const transitionStart = anchor + (nudgeBeats * beatSec) + (beatgridMs / 1000);
    const contextStart = transitionStart - (frontBars * beatsPerBar * beatSec);
    const duration = Math.max(beatSec, totalBars * beatsPerBar * beatSec);
    const trackDuration = Math.max(beatSec, Number(record.duration_seconds) || audioDuration(record) || duration);
    return {
      cfg,
      record,
      bpm,
      beatSec,
      beatsPerBar,
      frontBars,
      overlapBars,
      backBars,
      totalBars,
      cue,
      anchor,
      nudgeBeats,
      beatgridMs,
      transitionStart,
      transitionEnd: transitionStart + (overlapBars * beatsPerBar * beatSec),
      contextStart,
      contextEnd: contextStart + duration,
      duration,
      trackDuration,
    };
  }
  function transitionNudgeBounds(info) {
    if (!info) return { min: -999, max: 999 };
    const overlapDuration = Math.max(info.beatSec, info.overlapBars * info.beatsPerBar * info.beatSec);
    const min = (-overlapDuration - info.anchor) / info.beatSec;
    const max = (info.trackDuration - info.anchor) / info.beatSec;
    if (!Number.isFinite(min) || !Number.isFinite(max)) return { min: -999, max: 999 };
    return { min: Math.min(min, max), max: Math.max(min, max) };
  }
  function clampTransitionNudge(deck, value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return 0;
    const info = transitionWindowInfo(deck);
    const bounds = transitionNudgeBounds(info);
    // Serato-style jump sizes go down to one eighth of a beat. Keep the
    // alignment model at that precision rather than silently rounding jumps.
    return Math.round(clamp(n, bounds.min, bounds.max) * 8) / 8;
  }
  function resetDeckAlignment(deck) {
    const cfg = transitionDeckConfig(deck);
    previewFormState[cfg.nudgeField] = 0;
    previewFormState[cfg.beatgridField] = 0;
  }
  function resetBothDeckAlignments() {
    resetDeckAlignment('from');
    resetDeckAlignment('to');
    syncPreviewFieldElements('from_nudge_beats', 0);
    syncPreviewFieldElements('to_nudge_beats', 0);
    syncPreviewFieldElements('from_beatgrid_ms', 0);
    syncPreviewFieldElements('to_beatgrid_ms', 0);
    updatePreviewEffectDisplay({ dirty: true });
    schedulePlanningSave();
  }
  function presetDetails(name) {
    return (appOptions && appOptions.preset_details && appOptions.preset_details[name]) || {};
  }
  function transitionAutomationBeats() {
    return Math.max(1, Number(previewFormState.overlap_bars) || 1) * 4;
  }
  function automationLaneLimits(lane) {
    if (lane === 'volume') return [-96, 6];
    if (lane === 'filter') return [-1, 1];
    if (lane === 'delay' || lane === 'reverb') return [0, 1];
    return [-96, 6];
  }
  // A compact, invertible DAW-style fader response. Values remain dB in the
  // payload; the editor and renderer interpolate their positions on this
  // smooth fader curve instead of directly in dB.
  const AUTOMATION_DAW_SCALE = [[0, -96], [.25, -30], [.5, -13], [.85, 0], [1, 6]];
  function usesDawAutomationScale(lane) {
    return lane === 'volume' || String(lane || '').startsWith('eq_');
  }
  function dawFaderSlopes() {
    const positions = AUTOMATION_DAW_SCALE.map(point => point[0]);
    const values = AUTOMATION_DAW_SCALE.map(point => point[1]);
    const deltas = positions.slice(1).map((position, index) => (values[index + 1] - values[index]) / (position - positions[index]));
    return positions.map((position, index) => {
      if (index === 0) return deltas[0];
      if (index === positions.length - 1) return deltas[deltas.length - 1];
      const left = deltas[index - 1], right = deltas[index];
      if (left * right <= 0) return 0;
      const leftWidth = positions[index] - positions[index - 1];
      const rightWidth = positions[index + 1] - positions[index];
      const firstWeight = (2 * rightWidth) + leftWidth;
      const secondWeight = rightWidth + (2 * leftWidth);
      return (firstWeight + secondWeight) / ((firstWeight / left) + (secondWeight / right));
    });
  }
  const AUTOMATION_DAW_SLOPES = dawFaderSlopes();
  function dawFaderValue(position) {
    const p = clamp(Number(position) || 0, 0, 1);
    let index = AUTOMATION_DAW_SCALE.length - 2;
    for (let i = 1; i < AUTOMATION_DAW_SCALE.length; i += 1) {
      if (p <= AUTOMATION_DAW_SCALE[i][0] || i === AUTOMATION_DAW_SCALE.length - 1) { index = i - 1; break; }
    }
    const [x0, y0] = AUTOMATION_DAW_SCALE[index];
    const [x1, y1] = AUTOMATION_DAW_SCALE[index + 1];
    const width = x1 - x0;
    const t = (p - x0) / width;
    const t2 = t * t, t3 = t2 * t;
    return ((2 * t3 - 3 * t2 + 1) * y0)
      + ((t3 - 2 * t2 + t) * width * AUTOMATION_DAW_SLOPES[index])
      + ((-2 * t3 + 3 * t2) * y1)
      + ((t3 - t2) * width * AUTOMATION_DAW_SLOPES[index + 1]);
  }
  function dawFaderPosition(value) {
    const target = clamp(Number(value) || 0, -96, 6);
    if (target <= -96) return 0;
    if (target >= 6) return 1;
    let low = 0, high = 1;
    for (let i = 0; i < 28; i += 1) {
      const middle = (low + high) * .5;
      if (dawFaderValue(middle) < target) low = middle;
      else high = middle;
    }
    return (low + high) * .5;
  }
  function automationScalePosition(lane, value) {
    if (!usesDawAutomationScale(lane)) {
      const limits = automationLaneLimits(lane);
      return clamp((Number(value) - limits[0]) / Math.max(.000001, limits[1] - limits[0]), 0, 1);
    }
    return dawFaderPosition(value);
  }
  function automationValueFromScalePosition(lane, position) {
    if (!usesDawAutomationScale(lane)) {
      const limits = automationLaneLimits(lane);
      return limits[0] + (clamp(position, 0, 1) * (limits[1] - limits[0]));
    }
    return dawFaderValue(position);
  }
  function cloneAutomation(value) {
    return value ? JSON.parse(JSON.stringify(value)) : null;
  }
  function automationLane(deck, lane) {
    return transitionAutomation && transitionAutomation[deck] && transitionAutomation[deck][lane];
  }
  function automationPoint(beat, value, curve='linear') {
    return { beat: Number(beat), value: Number(value), curve };
  }
  function presetAutomationTemplate(name) {
    const total = transitionAutomationBeats();
    const preset = presetDetails(name);
    const volumeMode = preset.volume_mode || previewFormState.volume_mode || 'overlap-crossfade';
    const eqMode = preset.eq_mode || previewFormState.eq_mode || 'none';
    const filterMode = preset.filter_mode || previewFormState.filter_mode || 'none';
    const blank = () => [automationPoint(0, 0), automationPoint(total, 0)];
    const defaultVolume = () => [automationPoint(0, 0), automationPoint(total, 0)];
    const output = { from: {}, to: {} };
    ['from', 'to'].forEach(deck => {
      output[deck].volume = defaultVolume();
      output[deck].eq_low = blank();
      output[deck].eq_mid = blank();
      output[deck].eq_high = blank();
      output[deck].filter = blank();
      output[deck].delay = blank();
      output[deck].reverb = blank();
    });
    const fromFade = [automationPoint(0, 0, 'smooth'), automationPoint(total, -60, 'smooth')];
    const toFade = [automationPoint(0, -60, 'smooth'), automationPoint(total, 0, 'smooth')];
    if (volumeMode === 'overlap-crossfade') {
      output.from.volume = [automationPoint(0, 0, 'smooth'), automationPoint(total, -6, 'smooth')];
      output.to.volume = [automationPoint(0, -6, 'smooth'), automationPoint(total, 0, 'smooth')];
    } else if (volumeMode !== 'overlap') {
      output.from.volume = fromFade;
      output.to.volume = toFade;
    }
    if (volumeMode === 'center-cut') {
      const before = Math.max(0, total / 2 - 0.01);
      const after = Math.min(total, total / 2 + 0.01);
      output.from.volume = [automationPoint(0, 0), automationPoint(before, 0), automationPoint(after, -60), automationPoint(total, -60)];
      output.to.volume = [automationPoint(0, -60), automationPoint(before, -60), automationPoint(after, 0), automationPoint(total, 0)];
    } else if (volumeMode === 'fade-in-fade-out') {
      output.from.volume = [automationPoint(0, 0), automationPoint(total / 2, 0), automationPoint(total, -60)];
      output.to.volume = [automationPoint(0, -60), automationPoint(total / 2, 0), automationPoint(total, 0)];
    } else if (volumeMode === 'cut-in-fade-out') {
      output.from.volume = [automationPoint(0, 0), automationPoint(total / 2, 0), automationPoint(total, -60)];
      output.to.volume = defaultVolume();
    } else if (volumeMode === 'fade-in-cut-out') {
      output.from.volume = defaultVolume();
      output.to.volume = [automationPoint(0, -60), automationPoint(total / 2, 0), automationPoint(total, 0)];
    }
    if (eqMode === 'start-bass-swap') output.from.eq_low = [automationPoint(0, -96), automationPoint(total, -96)];
    if (eqMode === 'end-bass-swap') output.to.eq_low = [automationPoint(0, -96), automationPoint(total, -96)];
    if (eqMode === 'center-bass-swap') {
      const center = total / 2;
      output.from.eq_low = [automationPoint(0, 0), automationPoint(center, 0), automationPoint(center, -96), automationPoint(total, -96)];
      output.to.eq_low = [automationPoint(0, -96), automationPoint(center, -96), automationPoint(center, 0), automationPoint(total, 0)];
    }
    if (eqMode === 'long-bass-cut') ['from', 'to'].forEach(deck => { output[deck].eq_low = [automationPoint(0, -96), automationPoint(total, -96)]; });
    if (filterMode === 'low-pass-filter-out') output.from.filter = [automationPoint(0, 0), automationPoint(total, -1, 'smooth')];
    if (filterMode === 'low-pass-filter-in') output.to.filter = [automationPoint(0, -1), automationPoint(total, 0, 'smooth')];
    if (filterMode === 'high-pass-filter-out') output.from.filter = [automationPoint(0, 0), automationPoint(total, 1, 'smooth')];
    if (filterMode === 'high-pass-filter-in') output.to.filter = [automationPoint(0, 1), automationPoint(total, 0, 'smooth')];
    return output;
  }
  function normalizeAutomationPoints(points, lane) {
    const total = transitionAutomationBeats();
    const limits = automationLaneLimits(lane);
    const clean = (Array.isArray(points) ? points : []).map((point, order) => ({
      beat: clamp(Number(point.beat) || 0, 0, total),
      value: clamp(Number(point.value) || 0, limits[0], limits[1]),
      curve: ['linear', 'smooth', 'exponential'].includes(point.curve) ? point.curve : 'linear',
      order,
    })).sort((a, b) => a.beat - b.beat || a.order - b.order);
    const normalized = [];
    for (let start = 0; start < clean.length;) {
      const beat = clean[start].beat;
      let end = start + 1;
      while (end < clean.length && Math.abs(clean[end].beat - beat) < .000001) end += 1;
      const group = clean.slice(start, end);
      if (group.length > 2) normalized.push(group[0], group[group.length - 1]);
      else normalized.push(...group);
      start = end;
    }
    if (normalized.length < 2) return [automationPoint(0, 0), automationPoint(total, 0)];
    while (normalized.length > 2 && Math.abs(normalized[0].beat) < .000001 && Math.abs(normalized[1].beat) < .000001) normalized.splice(0, 1);
    while (normalized.length > 2 && Math.abs(normalized[normalized.length - 1].beat - total) < .000001 && Math.abs(normalized[normalized.length - 2].beat - total) < .000001) normalized.splice(normalized.length - 1, 1);
    normalized[0].beat = 0;
    normalized[normalized.length - 1].beat = total;
    return normalized.map(({ order, ...point }) => point);
  }
  function ensureTransitionAutomation({ reset=false } = {}) {
    if (reset || !transitionAutomation) {
      transitionAutomation = presetAutomationTemplate(transitionAutomationPreset || previewFormState.preset || 'auto');
      previewFormState.automation = transitionAutomation;
      return transitionAutomation;
    }
    ['from', 'to'].forEach(deck => LIVE_MIXER_LANES.forEach(lane => {
      transitionAutomation[deck][lane] = normalizeAutomationPoints(transitionAutomation[deck][lane], lane);
    }));
    previewFormState.automation = transitionAutomation;
    return transitionAutomation;
  }
  function snapshotTransitionAutomation({ resetHistory=false } = {}) {
    const snapshot = cloneAutomation(ensureTransitionAutomation());
    if (resetHistory) {
      transitionAutomationHistory = [snapshot];
      transitionAutomationHistoryIndex = 0;
      return;
    }
    transitionAutomationHistory = transitionAutomationHistory.slice(0, transitionAutomationHistoryIndex + 1);
    transitionAutomationHistory.push(snapshot);
    if (transitionAutomationHistory.length > 50) transitionAutomationHistory.shift();
    transitionAutomationHistoryIndex = transitionAutomationHistory.length - 1;
  }
  function markTransitionAutomationEdited({ history=true } = {}) {
    ensureTransitionAutomation();
    if (previewFormState.preset !== 'custom') {
      previewFormState.preset = 'custom';
      const presetEl = document.getElementById('preset');
      if (presetEl) presetEl.value = 'custom';
    }
    if (history) snapshotTransitionAutomation();
    if (livePlayback) requestLiveAutomationReschedule();
    updatePreviewEffectDisplay({ dirty: true });
  }
  function restoreTransitionAutomationHistory(delta) {
    const next = clamp(transitionAutomationHistoryIndex + delta, 0, transitionAutomationHistory.length - 1);
    if (next === transitionAutomationHistoryIndex || !transitionAutomationHistory[next]) return;
    transitionAutomationHistoryIndex = next;
    transitionAutomation = cloneAutomation(transitionAutomationHistory[next]);
    previewFormState.automation = transitionAutomation;
    previewFormState.preset = 'custom';
    transitionAutomationSelectedPoint = null;
    transitionAutomationSelectedSegment = null;
    if (livePlayback) requestLiveAutomationReschedule();
    updatePreviewEffectDisplay({ dirty: true });
  }
  function applyPresetToPreviewState(name) {
    previewFormState.preset = name || 'auto';
    const preset = presetDetails(previewFormState.preset);
    previewFormState.volume_mode = preset.volume_mode || previewFormState.volume_mode || 'overlap-crossfade';
    previewFormState.eq_mode = preset.eq_mode || previewFormState.eq_mode || 'none';
    previewFormState.filter_mode = preset.filter_mode || previewFormState.filter_mode || 'none';
    ['volume_mode', 'eq_mode', 'filter_mode'].forEach(id => {
      const el = document.getElementById(id);
      if (el && previewFormState[id] != null) el.value = previewFormState[id];
    });
    const presetEl = document.getElementById('preset');
    if (presetEl) presetEl.value = previewFormState.preset;
    transitionAutomationPreset = previewFormState.preset;
    transitionAutomationSelectedPoint = null;
    transitionAutomationSelectedSegment = null;
    ensureTransitionAutomation({ reset: true });
    snapshotTransitionAutomation({ resetHistory: true });
    if (livePlayback) requestLiveAutomationReschedule();
  }
  function applyLegacyModeAutomation(field) {
    ensureTransitionAutomation();
    const template = presetAutomationTemplate('custom');
    const lane = field === 'volume_mode' ? 'volume' : field === 'eq_mode' ? 'eq_low' : field === 'filter_mode' ? 'filter' : null;
    if (!lane) return;
    ['from', 'to'].forEach(deck => {
      transitionAutomation[deck][lane] = cloneAutomation(template[deck][lane]);
    });
    transitionAutomationSelectedPoint = null;
    transitionAutomationSelectedSegment = null;
    markTransitionAutomationEdited();
  }
  function resolvedPreviewModes() {
    return {
      volume: previewFormState.volume_mode || 'overlap-crossfade',
      eq: previewFormState.eq_mode || 'none',
      filter: previewFormState.filter_mode || 'none',
    };
  }
  function previewSignature(state) {
    const s = state || {};
    return JSON.stringify({
      from_cue: s.from_cue || '',
      to_cue: s.to_cue || '',
      overlap_bars: Number(s.overlap_bars) || 0,
      front_padding_bars: Number(s.front_padding_bars) || 0,
      back_padding_bars: Number(s.back_padding_bars) || 0,
      from_nudge_beats: Number(s.from_nudge_beats) || 0,
      to_nudge_beats: Number(s.to_nudge_beats) || 0,
      from_beatgrid_ms: Number(s.from_beatgrid_ms) || 0,
      to_beatgrid_ms: Number(s.to_beatgrid_ms) || 0,
      from_pitch_shift: Number(s.from_pitch_shift) || 0,
      to_pitch_shift: Number(s.to_pitch_shift) || 0,
      quality: s.quality || 'preview',
      volume_mode: s.volume_mode || 'overlap-crossfade',
      eq_mode: s.eq_mode || 'none',
      filter_mode: s.filter_mode || 'none',
      loudness_match_mode: s.loudness_match_mode || 'auditions_transitions',
      effects: s.effects || null,
      automation: s.automation || null,
    });
  }
  function previewSettingsDirty() {
    const current = Object.assign({}, previewFormState, { loudness_match_mode: loudnessMatchMode() });
    return !!(transitionRender && transitionRenderedState && previewSignature(current) !== previewSignature(transitionRenderedState));
  }
  function transitionLoadingHtml(active=false, id='transition-loading') {
    const idAttr = id ? ' id="' + esc(id) + '"' : '';
    return '<div' + idAttr + ' class="transition-loading' + (active ? ' active' : '') + '" aria-hidden="' + (active ? 'false' : 'true') + '"><span></span></div>';
  }
  function transitionRenderStatusInfo() {
    const dirty = previewSettingsDirty();
    if (transitionRenderPending) return { text: 'Rendering transition...', warn: false };
    if (transitionRender) {
      if (transitionRenderWarn && transitionRenderMessage) return { text: transitionRenderMessage, warn: true };
      return {
        text: dirty ? 'Settings changed. Render again to update audio and visualizer.' : (transitionRenderMessage || 'Rendered preview is current.'),
        warn: dirty,
      };
    }
    if (transitionRenderMessage) return { text: transitionRenderMessage, warn: transitionRenderWarn };
    return { text: 'Ready to render selected transition.', warn: false };
  }
  function automationSummaryHtml() {
    return '<div id="automation-summary" class="automation-summary">' + automationSummaryInnerHtml() + '</div>';
  }
  function syncPreviewFieldElements(field, value, sourceEl=null) {
    document.querySelectorAll('[data-preview-field="' + field + '"]').forEach(el => {
      if (el === sourceEl) return;
      if (el.type === 'number') el.value = String(Number(value) || 0);
      else el.value = String(value == null ? '' : value);
    });
  }
  function updateTransitionDirtyUi() {
    const status = document.getElementById('transition-render-status');
    const button = document.getElementById('transition-render-button');
    const loading = document.getElementById('transition-loading');
    const dirty = previewSettingsDirty();
    if (button) {
      const seamAction = button.classList.contains('transition-seam-action');
      button.textContent = seamAction ? (transitionRenderPending ? '…' : '↻') : (transitionRenderPending ? 'Rendering...' : (transitionRender && dirty ? 'Render again' : 'Render transition'));
      button.setAttribute('title', transitionRenderPending ? 'Rendering transition' : (transitionRender && dirty ? 'Render transition again' : 'Render transition'));
      button.disabled = transitionRenderPending;
    }
    if (loading) {
      loading.classList.toggle('active', transitionRenderPending);
      loading.setAttribute('aria-hidden', transitionRenderPending ? 'false' : 'true');
    }
    if (status) {
      const info = transitionRenderStatusInfo();
      status.textContent = info.text;
      status.classList.toggle('warn', !!info.warn);
    }
    const editorDirty = document.getElementById('transition-editor-dirty');
    if (editorDirty) {
      editorDirty.textContent = transitionRenderPending ? 'Rendering' : (dirty ? 'Pending render' : (transitionRender ? 'Rendered' : 'Not rendered'));
      editorDirty.classList.toggle('transition-editor-dirty', transitionRenderPending || dirty);
    }
  }
  function showTransitionPendingUi() {
    updateTransitionDirtyUi();
    const transport = document.querySelector('.transition-transport-row');
    if (transport && !transitionRender) transport.outerHTML = transitionTransportHtml();
    const meta = document.querySelector('.transition-meta');
    if (meta && !transitionRender) meta.textContent = 'Rendering transition...';
  }
  function updatePreviewEffectDisplay({ dirty=true } = {}) {
    const summary = document.getElementById('automation-summary');
    if (summary) summary.innerHTML = automationSummaryInnerHtml();
    syncAutomationEditorControls();
    safeUi('transition editor draw', drawTransitionEditor);
    if (dirty) scheduleLivePreparation();
    if (dirty) safeUi('transition dirty ui update', updateTransitionDirtyUi);
  }
  function automationSummaryInnerHtml() {
    const active = transitionAutomationActiveLane ? transitionAutomationActiveLane.replace(':', ' · ') : 'alignment mode';
    return '<b>Offline automation</b><br>' +
      'preset ' + esc(titleCaseOption(previewFormState.preset || 'auto')) +
      ' / active ' + esc(active) +
      ' / 3-band EQ';
  }
  async function loadAppRuntime() {
    if (!config.app_mode || appLoadStarted) return;
    appLoadStarted = true;
    try {
      const [tracksRes, optionsRes] = await Promise.all([fetch('/api/tracks'), fetch('/api/options')]);
      const tracksPayload = await tracksRes.json();
      const optionsPayload = await optionsRes.json();
      if (!tracksPayload.ok) throw new Error(tracksPayload.error || 'Track load failed');
      if (!optionsPayload.ok) throw new Error(optionsPayload.error || 'Option load failed');
      appTracks = tracksPayload.tracks || [];
      appTracksById = new Map(appTracks.map(track => [String(track.id), track]));
      appOptions = optionsPayload;
      applyPresetToPreviewState(previewFormState.preset);
      appLoadDone = true;
    } catch (err) {
      appOptions = { error: err.message || String(err) };
    }
    renderTransitionPreview();
  }
  function setActivePane(pane, { render=true } = {}) {
    activePane = pane || 'explore';
    els.tabButtons.forEach(btn => btn.classList.toggle('active', btn.getAttribute('data-pane-tab') === activePane));
    Object.keys(els.panes).forEach(key => {
      if (els.panes[key]) els.panes[key].classList.toggle('active', key === activePane);
    });
    if (activePane === 'explore') {
      setTimeout(() => safeUi('map renderer sync', () => syncMapRenderer()), 20);
    }
    if (activePane === 'explore') setTimeout(ensureMapEffectsLoop, 40);
    if (activePane === 'explore' && window.Plotly && els.energyCurve) {
      setTimeout(() => Plotly.Plots.resize(els.energyCurve), 30);
    }
    if (activePane === 'preview') {
      setTimeout(() => safeUi('transition editor bind', bindTransitionEditor), 20);
      setTimeout(() => safeUi('transition editor draw', drawTransitionEditor), 30);
    }
    if (render) setTimeout(() => safeUi('pane render', renderAll), 0);
  }
  function shortTrack(record) {
    return trackSummaryHtml(record, { size: 'compact', showArt: false });
  }
  function trackPreviewMetaHtml(record) {
    if (!record) return '<div class="muted">No track assigned.</div>';
    return trackSummaryHtml(record, { size: 'large', showArt: true });
  }
  function timeText(seconds) {
    const n = Number(seconds);
    if (!Number.isFinite(n) || n < 0) return '0:00';
    const total = Math.floor(n);
    return Math.floor(total / 60) + ':' + String(total % 60).padStart(2, '0');
  }
  function audioUri(record) {
    record = canonicalRecord(record);
    return String((record && record.audio_uri) || '');
  }
  function previewStart(record) {
    record = canonicalRecord(record);
    const start = Number(record && record.preview_start_seconds);
    const duration = Number(record && record.duration_seconds);
    if (!Number.isFinite(start) || start < 0) return 0;
    if (Number.isFinite(duration) && duration > 1) return clamp(start, 0, Math.max(0, duration - 1));
    return start;
  }
  function audioDuration(record) {
    const activeDuration = els.audio && Number.isFinite(Number(els.audio.duration)) ? Number(els.audio.duration) : NaN;
    if (record && previewAudioIdx === Number(record.idx) && Number.isFinite(activeDuration) && activeDuration > 0) return activeDuration;
    const duration = Number(record && record.duration_seconds);
    return Number.isFinite(duration) && duration > 0 ? duration : activeDuration;
  }
  function rowScrubValue(record) {
    record = canonicalRecord(record);
    if (!record) return 0;
    const idx = Number(record.idx);
    const stored = rowScrubPositions.get(idx);
    if (Number.isFinite(Number(stored))) return Number(stored);
    if (previewAudioIdx === idx && els.audio && Number.isFinite(Number(els.audio.currentTime))) return Number(els.audio.currentTime);
    return 0;
  }
  function rowWaveformScrubberHtml(record, { showTime=true } = {}) {
    record = canonicalRecord(record);
    if (!record) return '';
    const duration = audioDuration(record);
    const max = Number.isFinite(duration) && duration > 0 ? duration : Math.max(1, Number(record.duration_seconds) || 1);
    const value = clamp(rowScrubValue(record), 0, max);
    return '<div class="row-waveform-scrubber">' +
      '<canvas class="row-waveform" data-library-waveform-idx="' + record.idx + '" data-duration="' + esc(max) + '" height="30" aria-label="Playback scrubber"></canvas>' +
      (showTime ? '<span class="scrub-time" data-library-time-idx="' + record.idx + '">' + esc(timeText(value)) + ' / ' + esc(durationText(record)) + '</span>' : '') +
      '</div>';
  }
  function rowPlayerHtml(record) {
    record = canonicalRecord(record);
    if (!record) return '';
    return '<div class="row-player">' +
      '<button class="play-button" data-play-idx="' + record.idx + '" data-play-mode="library" aria-label="Play">▶</button>' +
      rowWaveformScrubberHtml(record) +
      '</div>';
  }
  function currentAudioRecord() {
    if (previewAudioIdx !== null && byIdx.has(Number(previewAudioIdx))) return byIdx.get(Number(previewAudioIdx));
    if (selectedIdx !== null && byIdx.has(Number(selectedIdx))) return byIdx.get(Number(selectedIdx));
    return null;
  }
  function updatePlayButtons() {
    document.querySelectorAll('[data-play-idx]').forEach(btn => {
      const idx = Number(btn.getAttribute('data-play-idx'));
      const playing = Number.isFinite(idx) && idx === previewAudioIdx && previewAudioPlaying;
      btn.textContent = playing ? '⏸' : '▶';
      const label = (playing ? 'Pause preview for ' : 'Play preview for ') + (byIdx.get(idx)?.title || 'track');
      btn.setAttribute('aria-label', label);
      btn.setAttribute('title', label);
    });
    if (els.songPlayerToggle) {
      const mainPlaying = selectedIdx !== null && previewAudioIdx === selectedIdx && previewAudioPlaying;
      els.songPlayerToggle.textContent = mainPlaying ? '⏸' : '▶';
      els.songPlayerToggle.setAttribute('aria-label', mainPlaying ? 'Pause' : 'Play');
      els.songPlayerToggle.setAttribute('title', mainPlaying ? 'Pause' : 'Play');
      els.songPlayerToggle.disabled = selectedIdx === null || !audioUri(byIdx.get(selectedIdx));
    }
  }
  function updateGlobalPlayer() {
    const record = currentAudioRecord();
    const hasRecord = !!(record && audioUri(record));
    const activeRecord = record && previewAudioIdx === Number(record.idx);
    const duration = audioDuration(record);
    const current = activeRecord && els.audio && Number.isFinite(Number(els.audio.currentTime))
      ? Number(els.audio.currentTime)
      : rowScrubValue(record);
    if (els.globalPlayer) els.globalPlayer.classList.toggle('idle', !hasRecord);
    if (els.globalArt) {
      if (record && record.artwork_uri) {
        els.globalArt.className = 'global-art';
        els.globalArt.innerHTML = '<img src="' + esc(record.artwork_uri) + '" alt="">';
      } else {
        els.globalArt.className = 'global-art art-placeholder';
        els.globalArt.textContent = 'art';
      }
    }
    if (els.globalTitle) els.globalTitle.textContent = record ? String(record.title || 'Untitled') : 'No track playing';
    if (els.globalArtist) {
      els.globalArtist.innerHTML = record
        ? '<span>' + esc(record.artists || '') + '</span><span class="global-meta-line">' + trackMetaHtml(record, { slot: relevantResidualSlot(record) }) + '</span>'
        : 'Select a track to preview';
    }
    if (els.globalToggle) {
      els.globalToggle.disabled = !hasRecord;
      els.globalToggle.textContent = previewAudioPlaying ? '⏸' : '▶';
      els.globalToggle.setAttribute('aria-label', previewAudioPlaying ? 'Pause' : 'Play');
      els.globalToggle.setAttribute('title', previewAudioPlaying ? 'Pause' : 'Play');
    }
    if (els.globalScrub) {
      const max = Number.isFinite(duration) && duration > 0 ? duration : 1;
      els.globalScrub.disabled = !hasRecord;
      els.globalScrub.max = String(max);
      els.globalScrub.value = String(clamp(current || 0, 0, max));
      setRangeProgress(els.globalScrub, current || 0, max);
    }
    if (els.globalTime) els.globalTime.textContent = timeText(current || 0) + ' / ' + timeText(duration);
    if (els.globalTrack1) els.globalTrack1.disabled = !record;
    if (els.globalTrack2) els.globalTrack2.disabled = !record;
  }
  function updateLibraryPlaybackState() {
    document.querySelectorAll('[data-library-row-idx]').forEach(row => {
      const idx = Number(row.getAttribute('data-library-row-idx'));
      const isCurrent = Number.isFinite(idx) && idx === previewAudioIdx;
      row.classList.toggle('now-playing', isCurrent && previewAudioPlaying);
      row.classList.toggle('now-playing-paused', isCurrent && !previewAudioPlaying);
    });
  }
  function updatePlayerTimeLabels() {
    const record = previewAudioIdx === null ? (selectedIdx === null ? null : byIdx.get(selectedIdx)) : byIdx.get(previewAudioIdx);
    const duration = audioDuration(record);
    const current = els.audio && Number.isFinite(Number(els.audio.currentTime)) ? Number(els.audio.currentTime) : 0;
    if (els.songPlayerTime) {
      const displayCurrent = selectedIdx !== null && previewAudioIdx === selectedIdx ? current : previewStart(byIdx.get(selectedIdx));
      const displayDuration = audioDuration(selectedIdx === null ? null : byIdx.get(selectedIdx));
      els.songPlayerTime.textContent = timeText(displayCurrent) + ' / ' + timeText(displayDuration);
    }
    if (els.songPlayerStatus) {
      const activeWaveform = selectedIdx !== null && waveformState.idx === selectedIdx ? waveformState : null;
      if (activeWaveform && activeWaveform.loading) els.songPlayerStatus.textContent = 'Loading waveform';
      else if (activeWaveform && activeWaveform.error) els.songPlayerStatus.textContent = activeWaveform.error;
      else els.songPlayerStatus.textContent = selectedIdx === null ? 'Preview' : 'Full track preview';
    }
    document.querySelectorAll('[data-library-time-idx]').forEach(el => {
      const idx = Number(el.getAttribute('data-library-time-idx'));
      const row = byIdx.get(idx);
      if (!row) return;
      const rowCurrent = rowScrubValue(row);
      el.textContent = timeText(rowCurrent) + ' / ' + durationText(row);
    });
  }
  function updateRowScrubbers() {
    drawLibraryWaveforms();
    drawDiagnosticWaveforms();
    updatePlayerTimeLabels();
    updateLibraryPlaybackState();
  }
  function updateTransitionTrackPlayButtons() {
    document.querySelectorAll('[data-transition-track-play]').forEach(button => {
      const deck = button.getAttribute('data-transition-track-play') || '';
      const info = transitionWindowInfo(deck);
      const active = !!(info && previewAudioIdx === Number(info.record.idx) && previewAudioContext === 'transition-' + deck && previewAudioPlaying);
      button.textContent = active ? '⏸' : '▶';
      button.setAttribute('aria-label', (active ? 'Pause ' : 'Play ') + (info && info.cfg ? info.cfg.label : 'track'));
    });
    document.querySelectorAll('[data-transition-track-scrub]').forEach(scrub => {
      const deck = scrub.getAttribute('data-transition-track-scrub') || '';
      const info = transitionWindowInfo(deck);
      if (!info || !info.record) return;
      const active = previewAudioIdx === Number(info.record.idx) && previewAudioContext === 'transition-' + deck;
      const value = active && els.audio && Number.isFinite(Number(els.audio.currentTime))
        ? Number(els.audio.currentTime)
        : (rowScrubPositions.has(Number(info.record.idx)) ? rowScrubValue(info.record) : info.transitionStart);
      scrub.max = String(info.trackDuration);
      scrub.value = String(clamp(value, 0, info.trackDuration));
      setRangeProgress(scrub, scrub.value, scrub.max);
    });
  }
  function syncAudioUi() {
    previewAudioPlaying = !!(els.audio && !els.audio.paused && !els.audio.ended);
    updatePlayButtons();
    updateRowScrubbers();
    updateTransitionTrackPlayButtons();
    updateGlobalPlayer();
    drawMainWaveform();
    requestTransitionEditorDraw();
  }
  function requestAudioProgressFrame() {
    if (audioProgressFrame !== null || !window.requestAnimationFrame) return;
    audioProgressFrame = window.requestAnimationFrame(() => {
      audioProgressFrame = null;
      syncAudioUi();
      if (previewAudioPlaying) requestAudioProgressFrame();
    });
  }
  function seekSharedAudio(seconds) {
    if (!els.audio) return;
    const duration = Number(els.audio.duration);
    const max = Number.isFinite(duration) && duration > 0 ? duration : Number.POSITIVE_INFINITY;
    const target = clamp(Number(seconds) || 0, 0, max);
    try { els.audio.currentTime = target; } catch (err) {}
  }
  function playRecordAt(record, startSeconds, context='point') {
    record = canonicalRecord(record);
    if (!record || !els.audio) return;
    const renderedTransition = transitionAudioElement();
    if (renderedTransition && !renderedTransition.paused && !renderedTransition.ended) renderedTransition.pause();
    const url = audioUri(record);
    if (!url) return;
    const start = Math.max(0, Number(startSeconds) || 0);
    const idx = Number(record.idx);
    const sourceChanged = previewAudioUrl !== url;
    previewAudioIdx = idx;
    previewAudioContext = context;
    previewAudioStart = start;
    applySharedAudioLoudness({ smooth: !sourceChanged });
    if (sourceChanged) {
      previewAudioUrl = url;
      els.audio.src = url;
      try { els.audio.load(); } catch (err) {}
    }
    applyMasterVolume();
    const doPlay = () => {
      seekSharedAudio(start);
      rowScrubPositions.set(idx, start);
      const promise = els.audio.play();
      if (promise && promise.catch) promise.catch(() => {});
      syncAudioUi();
      requestAudioProgressFrame();
    };
    if (els.audio.readyState >= 1) doPlay();
    else els.audio.addEventListener('loadedmetadata', doPlay, { once: true });
  }
  function pauseSharedAudio() {
    if (!els.audio) return;
    try { els.audio.pause(); } catch (err) {}
    previewAudioPlaying = false;
    syncAudioUi();
  }
  function stopSharedAudio({ clear=false } = {}) {
    if (!els.audio) return;
    pauseSharedAudio();
    if (clear) {
      els.audio.removeAttribute('src');
      try { els.audio.load(); } catch (err) {}
      previewAudioIdx = null;
      previewAudioUrl = '';
      previewAudioContext = '';
      previewAudioStart = 0;
    }
    syncAudioUi();
  }
  function hideSongPopover({ stopAudio=true } = {}) {
    if (els.songPopover) els.songPopover.classList.add('hidden');
    if (stopAudio && previewAudioContext === 'main') stopSharedAudio({ clear: false });
    updatePlayButtons();
  }
  function clearCurrentTrackSelection() {
    selectedIdx = null;
    lastClickedIdx = null;
    lastClickMs = 0;
    hideSongPopover({ stopAudio: false });
    schedulePlanningSave();
    renderSelectionOnly();
  }
  function setCandidateTrack(idx, { autoplay=true, showPopover=true, render=true, setRecommendationReference=true, setTransitionCandidate=true } = {}) {
    idx = Number(idx);
    if (!byIdx.has(idx)) return;
    if (setRecommendationReference && transitionFromIdx === null && recommendationReferenceIdx !== idx) pinnedRecommendationIdxs.clear();
    selectedIdx = idx;
    if (setRecommendationReference) recommendationReferenceIdx = idx;
    if (setTransitionCandidate && transitionFromIdx !== null && Number(transitionFromIdx) !== idx) {
      if (transitionToIdx === null || Number(transitionToIdx) !== idx) resetDeckAlignment('to');
      transitionToIdx = idx;
      resetTransitionRenderState();
      resetTransitionAutomationState();
      previewFormState.to_cue = '';
      scheduleLivePreparation();
    }
    const record = byIdx.get(idx);
    if (showPopover) showSongPopover(record, { autoplay });
    else hideSongPopover({ stopAudio: false });
    schedulePlanningSave();
    if (render) renderSelectionOnly();
  }
  function focusTrack(idx, { autoplay=true, showPopover=true, render=true } = {}) {
    setCandidateTrack(idx, {
      autoplay,
      showPopover,
      render,
      setRecommendationReference: false,
      setTransitionCandidate: false,
    });
  }
  function showSongPopover(record, { autoplay=true } = {}) {
    if (!els.songPopover || !record) return;
    els.songPopover.classList.remove('hidden');
    if (els.selectedTrack) {
      els.selectedTrack.innerHTML = trackSummaryHtml(record, { size: 'large', showArt: true });
    }
    loadMainWaveform(record);
    const start = previewStart(record);
    if (autoplay) {
      playRecordAt(record, start, 'main');
    } else {
      previewAudioIdx = Number(record.idx);
      previewAudioContext = 'main';
      previewAudioStart = start;
      if (els.audio && audioUri(record) && previewAudioUrl !== audioUri(record)) {
        previewAudioUrl = audioUri(record);
        els.audio.src = previewAudioUrl;
        try { els.audio.load(); } catch (err) {}
      }
      applySharedAudioLoudness({ smooth: true });
      if (els.audio) seekSharedAudio(start);
    }
    syncAudioUi();
  }
  function showSongHover(record, nativeEvent) {
    if (record && Number.isFinite(Number(record.idx))) {
      const idx = Number(record.idx);
      if (hoveredIdx !== idx) {
        hoveredIdx = idx;
        safeUi('hover overlay', updatePacmapOverlays);
      }
    }
    if (!els.songHoverCard || !record) return;
    els.songHoverCard.innerHTML = trackSummaryHtml(record, { size: 'large', showArt: true });
    els.songHoverCard.classList.remove('hidden');
    const pane = document.querySelector('.plot-pane');
    if (!pane || !nativeEvent) return;
    const paneRect = pane.getBoundingClientRect();
    const cardRect = els.songHoverCard.getBoundingClientRect();
    let x = Number(nativeEvent.clientX) - paneRect.left + 14;
    let y = Number(nativeEvent.clientY) - paneRect.top + 14;
    x = clamp(x, 8, Math.max(8, paneRect.width - cardRect.width - 8));
    y = clamp(y, 8, Math.max(8, paneRect.height - cardRect.height - 8));
    els.songHoverCard.style.left = x + 'px';
    els.songHoverCard.style.top = y + 'px';
  }
  function hideSongHover() {
    if (hoveredIdx !== null) {
      hoveredIdx = null;
      safeUi('hover overlay', updatePacmapOverlays);
    }
    if (els.songHoverCard) els.songHoverCard.classList.add('hidden');
  }
  function toggleTrackPreview(idx, { mode='point' } = {}) {
    idx = Number(idx);
    if (!byIdx.has(idx)) return;
    if (mode === 'recommendation') {
      // Auditioning a recommendation changes the transient selection only; the
      // recommendation source remains the current anchor track.
      focusTrack(idx, { autoplay: false, showPopover: false });
    }
    const record = byIdx.get(idx);
    const context = mode === 'library' || mode === 'diagnostics' || mode === 'recommendation' ? mode : 'point';
    if (previewAudioIdx === idx && previewAudioContext === context && els.audio && !els.audio.paused) {
      pauseSharedAudio();
      return;
    }
    const start = context === 'library' || context === 'diagnostics' ? rowScrubValue(record) : previewStart(record);
    playRecordAt(record, start, context);
  }
  function toggleTransitionTrackPreview(deck) {
    const info = transitionWindowInfo(deck);
    if (!info || !info.record) return;
    const idx = Number(info.record.idx);
    const context = 'transition-' + String(deck || '');
    if (previewAudioIdx === idx && previewAudioContext === context && els.audio && !els.audio.paused) {
      pauseSharedAudio();
      return;
    }
    const start = rowScrubPositions.has(idx) ? rowScrubValue(info.record) : Math.max(0, Number(info.transitionStart) || 0);
    playRecordAt(info.record, start, context);
  }
  function normalizeWaveformValues(values) {
    if (!Array.isArray(values)) return [];
    const out = [];
    let maxValue = 0;
    values.forEach(value => {
      const n = Number(value);
      if (!Number.isFinite(n)) return;
      if (n > maxValue) maxValue = n;
      out.push(n);
    });
    const scale = maxValue > 1.001 ? 255 : 1;
    return out.map(value => clamp(value / scale, 0, 1));
  }
  function normalizeSignedWaveformValues(values) {
    if (!Array.isArray(values)) return [];
    const out = [];
    let maxAbs = 0;
    values.forEach(value => {
      const n = Number(value);
      if (!Number.isFinite(n)) return;
      maxAbs = Math.max(maxAbs, Math.abs(n));
      out.push(n);
    });
    const scale = maxAbs > 1.001 ? 127 : 1;
    return out.map(value => clamp(value / scale, -1, 1));
  }
  function waveformDetailUri(record) {
    record = canonicalRecord(record);
    return String((record && (record.waveform_detail_uri || record.waveform_uri)) || '');
  }
  function baseWaveformPayload(record) {
    record = canonicalRecord(record);
    return {
      row: normalizeWaveformValues(record && record.waveform_peaks),
      preview: normalizeWaveformValues(record && record.waveform_preview_peaks),
      detail: [],
      previewMin: normalizeSignedWaveformValues(record && record.waveform_preview_min),
      previewMax: normalizeSignedWaveformValues(record && record.waveform_preview_max),
      detailMin: [],
      detailMax: [],
      bands: null,
      drawCache: new Map(),
      loaded: false,
      loading: false,
      error: '',
      promise: null,
    };
  }
  function waveformPayload(record) {
    record = canonicalRecord(record);
    if (!record) return baseWaveformPayload(null);
    const key = trackId(record);
    if (!waveformCache.has(key)) waveformCache.set(key, baseWaveformPayload(record));
    return waveformCache.get(key);
  }
  function applyWaveformDetail(payload, data) {
    if (!payload || !data) return payload;
    const row = normalizeWaveformValues(data.row_peaks);
    const preview = normalizeWaveformValues(data.preview_peaks);
    const detail = normalizeWaveformValues(data.detail_peaks);
    const envelope = data.envelope || {};
    const bands = data.bands || {};
    if (row.length) payload.row = row;
    if (preview.length) payload.preview = preview;
    if (detail.length) payload.detail = detail;
    const previewMin = normalizeSignedWaveformValues(envelope.preview_min || data.preview_min);
    const previewMax = normalizeSignedWaveformValues(envelope.preview_max || data.preview_max);
    const detailMin = normalizeSignedWaveformValues(envelope.detail_min || data.detail_min);
    const detailMax = normalizeSignedWaveformValues(envelope.detail_max || data.detail_max);
    if (previewMin.length && previewMax.length) {
      payload.previewMin = previewMin;
      payload.previewMax = previewMax;
    }
    if (detailMin.length && detailMax.length) {
      payload.detailMin = detailMin;
      payload.detailMax = detailMax;
    }
    const low = normalizeWaveformValues(bands.low);
    const mid = normalizeWaveformValues(bands.mid);
    const high = normalizeWaveformValues(bands.high);
    payload.bands = (low.length || mid.length || high.length) ? { low, mid, high } : null;
    payload.drawCache = new Map();
    payload.loaded = true;
    payload.error = '';
    return payload;
  }
  function waveformSeries(payload, preference='detail') {
    if (!payload) return [];
    if (preference === 'row') return payload.row || [];
    if (preference === 'preview') return (payload.preview && payload.preview.length ? payload.preview : payload.row) || [];
    return (payload.detail && payload.detail.length ? payload.detail : (payload.preview && payload.preview.length ? payload.preview : payload.row)) || [];
  }
  function waveformEnvelopeSeries(payload, preference='detail') {
    if (!payload) return null;
    const detailReady = preference !== 'row' && payload.detailMin && payload.detailMax && payload.detailMin.length && payload.detailMax.length;
    if (detailReady) return { min: payload.detailMin, max: payload.detailMax };
    if (preference !== 'row' && payload.previewMin && payload.previewMax && payload.previewMin.length && payload.previewMax.length) {
      return { min: payload.previewMin, max: payload.previewMax };
    }
    return null;
  }
  function resampleSeries(series, count) {
    const source = normalizeWaveformValues(series);
    const n = Math.max(2, Math.round(count || 2));
    if (!source.length) return [];
    if (source.length === n) return source;
    if (source.length === 1) return Array.from({ length: n }, () => source[0]);
    const out = [];
    const scale = (source.length - 1) / Math.max(1, n - 1);
    for (let i = 0; i < n; i += 1) {
      const pos = i * scale;
      const lo = Math.floor(pos);
      const hi = Math.min(source.length - 1, lo + 1);
      const frac = pos - lo;
      out.push(source[lo] + (source[hi] - source[lo]) * frac);
    }
    return out;
  }
  function resampleSignedSeries(series, count) {
    const source = normalizeSignedWaveformValues(series);
    const n = Math.max(2, Math.round(count || 2));
    if (!source.length) return [];
    if (source.length === n) return source;
    if (source.length === 1) return Array.from({ length: n }, () => source[0]);
    const out = [];
    const scale = (source.length - 1) / Math.max(1, n - 1);
    for (let i = 0; i < n; i += 1) {
      const pos = i * scale;
      const lo = Math.floor(pos);
      const hi = Math.min(source.length - 1, lo + 1);
      const frac = pos - lo;
      out.push(source[lo] + (source[hi] - source[lo]) * frac);
    }
    return out;
  }
  function drawWaveformEnvelope(ctx, samples, width, height, color, scale=1, pow=1) {
    if (!samples || !samples.length) return;
    const mid = height * 0.5;
    const amp = height * 0.47 * scale;
    const denom = Math.max(1, samples.length - 1);
    ctx.save();
    ctx.fillStyle = color;
    ctx.beginPath();
    samples.forEach((value, i) => {
      const x = (i / denom) * width;
      const y = mid - Math.pow(clamp(Number(value) || 0, 0, 1), pow) * amp;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    for (let i = samples.length - 1; i >= 0; i -= 1) {
      const x = (i / denom) * width;
      const y = mid + Math.pow(clamp(Number(samples[i]) || 0, 0, 1), pow) * amp;
      ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }
  function cachedDrawSeries(payload, count) {
    if (!payload) return { peaks: [], low: [], mid: [], high: [], min: [], max: [] };
    if (!payload.drawCache) payload.drawCache = new Map();
    const key = String(Math.max(2, Math.round(count || 2))) + ':' + (payload.loaded ? 'detail' : 'base');
    if (payload.drawCache.has(key)) return payload.drawCache.get(key);
    const peaks = resampleSeries(waveformSeries(payload, 'detail'), count);
    const envelope = waveformEnvelopeSeries(payload, 'detail');
    const bands = payload.bands || null;
    const series = {
      peaks,
      low: bands && bands.low && bands.low.length ? resampleSeries(bands.low, count) : peaks,
      mid: bands && bands.mid && bands.mid.length ? resampleSeries(bands.mid, count) : peaks,
      high: bands && bands.high && bands.high.length ? resampleSeries(bands.high, count) : peaks,
      min: envelope && envelope.min && envelope.min.length ? resampleSignedSeries(envelope.min, count) : [],
      max: envelope && envelope.max && envelope.max.length ? resampleSignedSeries(envelope.max, count) : [],
    };
    payload.drawCache.set(key, series);
    return series;
  }
  function drawDetailedWaveform(ctx, payload, width, height, progressX) {
    const count = Math.max(180, Math.floor(width * 1.35));
    const series = cachedDrawSeries(payload, count);
    const peaks = series.peaks;
    const low = series.low;
    const mid = series.mid;
    const high = series.high;
    if (!peaks.length) {
      drawWaveformBars(ctx, [], width, height, progressX);
      return;
    }
    drawWaveformEnvelope(ctx, low, width, height, 'rgba(31,90,240,.68)', 1.0, 1.03);
    drawWaveformEnvelope(ctx, mid, width, height, 'rgba(211,128,39,.68)', 0.74, 0.92);
    drawWaveformEnvelope(ctx, high, width, height, 'rgba(248,242,226,.82)', 0.38, 0.72);
    ctx.save();
    ctx.globalCompositeOperation = 'screen';
    ctx.beginPath();
    ctx.rect(0, 0, clamp(progressX, 0, width), height);
    ctx.clip();
    drawWaveformEnvelope(ctx, low, width, height, 'rgba(29,185,84,.42)', 1.0, 1.02);
    drawWaveformEnvelope(ctx, mid, width, height, 'rgba(94,234,212,.32)', 0.74, 0.9);
    ctx.restore();
  }
  async function ensureDetailedWaveform(record, { redrawMain=false, redrawTransition=false, redrawDiagnostics=false } = {}) {
    record = canonicalRecord(record);
    if (!record || !window.fetch) return null;
    const payload = waveformPayload(record);
    const url = waveformDetailUri(record);
    if (!url || payload.loaded) return payload;
    if (payload.loading && payload.promise) return payload.promise;
    payload.loading = true;
    payload.error = '';
    payload.promise = fetch(url)
      .then(res => {
        if (!res.ok) throw new Error('Waveform fetch failed');
        return res.json();
      })
      .then(data => applyWaveformDetail(payload, data))
      .catch(err => {
        payload.error = 'Detailed waveform unavailable';
        return payload;
      })
      .finally(() => {
        payload.loading = false;
        if (waveformState.idx === Number(record.idx)) {
          waveformState.loading = false;
          waveformState.error = payload.error || '';
          waveformState.payload = payload;
          if (redrawMain) drawMainWaveform();
        }
        if (redrawTransition) drawTransitionEditor();
        if (redrawDiagnostics) drawDiagnosticWaveforms();
      });
    return payload.promise;
  }
  async function loadMainWaveform(record) {
    record = canonicalRecord(record);
    if (!record || !els.songWaveform) return;
    const url = waveformDetailUri(record);
    const payload = waveformPayload(record);
    waveformState = { idx: Number(record.idx), url, payload, loading: false, error: payload.error || '' };
    drawMainWaveform();
    if (url && !payload.loaded) {
      waveformState.loading = true;
      drawMainWaveform();
      ensureDetailedWaveform(record, { redrawMain: true, redrawTransition: true });
    }
  }
  function drawWaveformBars(ctx, peaks, width, height, progressX) {
    const normalized = normalizeWaveformValues(peaks);
    const bars = normalized.length ? normalized : Array.from({ length: 120 }, (_, i) => 0.18 + 0.12 * Math.sin(i * 0.43));
    const gap = width < 260 ? 1 : 2;
    const barW = Math.max(1, Math.floor(width / bars.length) - gap);
    const step = width / bars.length;
    for (let i = 0; i < bars.length; i += 1) {
      const x = i * step;
      const h = Math.max(4, bars[i] * (height - 14));
      const y = (height - h) * 0.5;
      const active = x <= progressX;
      ctx.fillStyle = active ? '#1db954' : 'rgba(148, 163, 184, 0.52)';
      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(x, y, barW, h, Math.min(3, barW * 0.5));
        ctx.fill();
      } else {
        ctx.fillRect(x, y, barW, h);
      }
    }
  }
  function drawMainWaveform() {
    const canvas = els.songWaveform;
    if (!canvas) return;
    const record = selectedIdx === null ? null : byIdx.get(selectedIdx);
    const rect = canvas.getBoundingClientRect();
    const widthCss = Math.max(1, rect.width || canvas.clientWidth || 320);
    const heightCss = Math.max(1, rect.height || 54);
    const dpr = window.devicePixelRatio || 1;
    const width = Math.floor(widthCss * dpr);
    const height = Math.floor(heightCss * dpr);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, widthCss, heightCss);
    ctx.fillStyle = '#0b1020';
    ctx.fillRect(0, 0, widthCss, heightCss);
    if (!record) {
      ctx.fillStyle = '#64748b';
      ctx.font = '12px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.fillText('Select a track', 12, Math.round(heightCss / 2) + 4);
      return;
    }
    const duration = audioDuration(record);
    const current = previewAudioIdx === Number(record.idx) && els.audio ? Number(els.audio.currentTime || 0) : previewStart(record);
    const progress = Number.isFinite(duration) && duration > 0 ? clamp(current / duration, 0, 1) : 0;
    const progressX = progress * widthCss;
    const activeWaveform = waveformState.idx === Number(record.idx) ? waveformState : { payload: waveformPayload(record), loading: false, error: '' };
    drawDetailedWaveform(ctx, activeWaveform.payload || waveformPayload(record), widthCss, heightCss, progressX);
    const markerTime = previewStart(record);
    if (Number.isFinite(duration) && duration > 0 && markerTime > 0) {
      const x = clamp(markerTime / duration, 0, 1) * widthCss;
      ctx.save();
      ctx.strokeStyle = 'rgba(251, 191, 36, .9)';
      ctx.lineWidth = 1.4;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(x, 5);
      ctx.lineTo(x, heightCss - 5);
      ctx.stroke();
      ctx.restore();
    }
    ctx.strokeStyle = 'rgba(248, 250, 252, .88)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(progressX, 4);
    ctx.lineTo(progressX, heightCss - 4);
    ctx.stroke();
    if (activeWaveform.loading || activeWaveform.error) {
      ctx.fillStyle = 'rgba(15, 23, 42, .78)';
      ctx.fillRect(0, 0, widthCss, heightCss);
      ctx.fillStyle = activeWaveform.error ? '#fbbf24' : '#cbd5e1';
      ctx.font = '12px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.fillText(activeWaveform.error || 'Loading waveform...', 12, Math.round(heightCss / 2) + 4);
    }
  }
  function drawRowWaveform(canvas, record) {
    record = canonicalRecord(record);
    if (!canvas || !record) return;
    const rect = canvas.getBoundingClientRect();
    const widthCss = Math.max(1, rect.width || canvas.clientWidth || 220);
    const heightCss = Math.max(1, rect.height || 30);
    const dpr = window.devicePixelRatio || 1;
    const width = Math.floor(widthCss * dpr);
    const height = Math.floor(heightCss * dpr);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, widthCss, heightCss);
    ctx.fillStyle = '#0b1020';
    ctx.fillRect(0, 0, widthCss, heightCss);
    const duration = audioDuration(record);
    const current = clamp(rowScrubValue(record), 0, Number.isFinite(duration) && duration > 0 ? duration : Number(record.duration_seconds) || 1);
    const progress = Number.isFinite(duration) && duration > 0 ? clamp(current / duration, 0, 1) : 0;
    const peaks = waveformSeries(waveformPayload(record), 'row');
    drawWaveformBars(ctx, peaks, widthCss, heightCss, progress * widthCss);
    const markerTime = previewStart(record);
    if (Number.isFinite(duration) && duration > 0 && markerTime > 0) {
      const x = clamp(markerTime / duration, 0, 1) * widthCss;
      ctx.save();
      ctx.strokeStyle = 'rgba(251, 191, 36, .78)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x, 4);
      ctx.lineTo(x, heightCss - 4);
      ctx.stroke();
      ctx.restore();
    }
    if (previewAudioIdx === Number(record.idx)) {
      const x = progress * widthCss;
      ctx.strokeStyle = previewAudioPlaying ? 'rgba(248,250,252,.96)' : 'rgba(203,213,225,.72)';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(x, 4);
      ctx.lineTo(x, heightCss - 4);
      ctx.stroke();
    }
  }
  function drawLibraryWaveforms() {
    document.querySelectorAll('[data-library-waveform-idx]').forEach(canvas => {
      const idx = Number(canvas.getAttribute('data-library-waveform-idx'));
      const record = byIdx.get(idx);
      if (record) drawRowWaveform(canvas, record);
    });
  }
  function drawDiagnosticWaveform(canvas, record) {
    record = canonicalRecord(record);
    if (!canvas || !record) return;
    const payload = waveformPayload(record);
    const url = waveformDetailUri(record);
    if (url && !payload.loaded && !payload.loading) {
      ensureDetailedWaveform(record, { redrawDiagnostics: true });
    }
    const rect = canvas.getBoundingClientRect();
    const widthCss = Math.max(1, rect.width || canvas.clientWidth || 480);
    const heightCss = Math.max(1, rect.height || 86);
    const dpr = window.devicePixelRatio || 1;
    const width = Math.floor(widthCss * dpr);
    const height = Math.floor(heightCss * dpr);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, widthCss, heightCss);
    ctx.fillStyle = '#0b1020';
    ctx.fillRect(0, 0, widthCss, heightCss);
    const duration = audioDuration(record);
    const current = previewAudioIdx === Number(record.idx) && els.audio
      ? Number(els.audio.currentTime || 0)
      : rowScrubValue(record);
    const progress = Number.isFinite(duration) && duration > 0 ? clamp(current / duration, 0, 1) : 0;
    drawDetailedWaveform(ctx, payload, widthCss, heightCss, progress * widthCss);
    const markerTime = previewStart(record);
    if (Number.isFinite(duration) && duration > 0 && markerTime > 0) {
      const x = clamp(markerTime / duration, 0, 1) * widthCss;
      ctx.save();
      ctx.strokeStyle = 'rgba(251, 191, 36, .82)';
      ctx.lineWidth = 1.2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(x, 5);
      ctx.lineTo(x, heightCss - 5);
      ctx.stroke();
      ctx.restore();
    }
    ctx.strokeStyle = previewAudioIdx === Number(record.idx) && previewAudioPlaying
      ? 'rgba(248,250,252,.96)'
      : 'rgba(203,213,225,.72)';
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(progress * widthCss, 4);
    ctx.lineTo(progress * widthCss, heightCss - 4);
    ctx.stroke();
    if (payload.loading || payload.error) {
      ctx.fillStyle = 'rgba(15, 23, 42, .76)';
      ctx.fillRect(0, 0, widthCss, heightCss);
      ctx.fillStyle = payload.error ? '#fbbf24' : '#cbd5e1';
      ctx.font = '12px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.fillText(payload.error || 'Loading waveform...', 12, Math.round(heightCss / 2) + 4);
    }
  }
  function drawDiagnosticWaveforms() {
    document.querySelectorAll('[data-diagnostic-waveform-idx]').forEach(canvas => {
      const idx = Number(canvas.getAttribute('data-diagnostic-waveform-idx'));
      const record = byIdx.get(idx);
      if (record) drawDiagnosticWaveform(canvas, record);
    });
  }
  function seekDiagnosticWaveformFromEvent(ev, idx, { play=false } = {}) {
    idx = Number(idx);
    const record = byIdx.get(idx);
    if (!record) return;
    const duration = audioDuration(record);
    if (!Number.isFinite(duration) || duration <= 0) return;
    const targetCanvas = closestEl(ev.target, '[data-diagnostic-waveform-idx]');
    const canvas = targetCanvas || (ev.currentTarget && ev.currentTarget.getAttribute && ev.currentTarget.getAttribute('data-diagnostic-waveform-idx')
      ? ev.currentTarget
      : document.querySelector('[data-diagnostic-waveform-idx="' + idx + '"]'));
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = clamp((ev.clientX - rect.left) / Math.max(1, rect.width), 0, 1);
    const target = x * duration;
    rowScrubPositions.set(idx, target);
    if (play) {
      playRecordAt(record, target, 'diagnostics');
    } else {
      if (previewAudioIdx === idx && els.audio) seekSharedAudio(target);
      syncAudioUi();
    }
    drawDiagnosticWaveforms();
  }
  function seekLibraryWaveformFromEvent(ev, idx, { play=false } = {}) {
    idx = Number(idx);
    const record = byIdx.get(idx);
    if (!record) return;
    const duration = audioDuration(record);
    if (!Number.isFinite(duration) || duration <= 0) return;
    const targetCanvas = closestEl(ev.target, '[data-library-waveform-idx]');
    const canvas = targetCanvas || (ev.currentTarget && ev.currentTarget.getAttribute && ev.currentTarget.getAttribute('data-library-waveform-idx')
      ? ev.currentTarget
      : document.querySelector('[data-library-waveform-idx="' + idx + '"]'));
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = clamp((ev.clientX - rect.left) / Math.max(1, rect.width), 0, 1);
    const target = x * duration;
    rowScrubPositions.set(idx, target);
    if (play) {
      playRecordAt(record, target, 'library');
    } else {
      if (previewAudioIdx === idx && els.audio) seekSharedAudio(target);
      syncAudioUi();
    }
  }
  function seekMainWaveformFromEvent(ev, shouldPlay=true) {
    if (selectedIdx === null || !els.songWaveform) return;
    const record = byIdx.get(selectedIdx);
    const duration = audioDuration(record);
    if (!record || !Number.isFinite(duration) || duration <= 0) return;
    const rect = els.songWaveform.getBoundingClientRect();
    const x = clamp((ev.clientX - rect.left) / Math.max(1, rect.width), 0, 1);
    const target = x * duration;
    if (shouldPlay) playRecordAt(record, target, 'main');
    else {
      seekSharedAudio(target);
      syncAudioUi();
    }
  }
  function smoothstep(x) {
    return x * x * (3 - (2 * x));
  }
  function previewValueFor(mode, deck, x) {
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
    if (mode === 'eq:center-bass-swap') return deck === 'a' ? (half ? 1 : 0) : (half ? 0 : 1);
    if (mode === 'eq:long-bass-cut') return 0.15;
    if (mode === 'filter:none') return null;
    if (mode === 'filter:low-pass-filter-out') return deck === 'a' ? 1 - x : null;
    if (mode === 'filter:low-pass-filter-in') return deck === 'b' ? x : null;
    if (mode === 'filter:high-pass-filter-out') return deck === 'a' ? x : null;
    if (mode === 'filter:high-pass-filter-in') return deck === 'b' ? 1 - x : null;
    return null;
  }
  function traceIndexByName(name) {
    if (!plot || !plot.data) return -1;
    for (let i = 0; i < plot.data.length; i += 1) {
      if (plot.data[i] && plot.data[i].name === name) return i;
    }
    return -1;
  }
  function restyleTrace(name, update) {
    if (!window.Plotly || !plot) return;
    const idx = traceIndexByName(name);
    if (idx < 0) return;
    Plotly.restyle(plot, update, [idx]);
  }
  function normalizePointColorMode(mode) {
    return mode === 'energy_residual' ? 'target' : String(mode || 'genre');
  }
  function colorScaleExtent(mode) {
    mode = normalizePointColorMode(mode);
    if (mode === 'energy') return { min: 1, max: 9, title: els.energySource.value === 'glm' ? 'Auto energy' : 'Tagged energy' };
    if (mode === 'target') return targetEnergyResidualExtent();
    if (mode === 'key') return { min: 1, max: 12, title: 'Camelot key wheel' };
    const vals = records.map(r => Number(r.est_bpm)).filter(Number.isFinite);
    if (!vals.length) return { min: 0, max: 1, title: 'Tempo BPM' };
    let min = Math.min(...vals);
    let max = Math.max(...vals);
    if (Math.abs(max - min) < 1e-9) max = min + 1;
    return { min, max, title: 'Estimated BPM' };
  }
  function pointColorValue(idx, mode) {
    mode = normalizePointColorMode(mode);
    const record = byIdx.get(Number(idx));
    if (!record) return NaN;
    if (mode === 'energy') return energyOf(record);
    if (mode === 'target') return targetEnergyResidual(record);
    if (mode === 'key') return camelotNumber(record.key);
    if (mode === 'tempo') return Number(record.est_bpm);
    return NaN;
  }
  function pointMarkerColor(idx, mode, extent=null) {
    mode = normalizePointColorMode(mode);
    const record = byIdx.get(Number(idx));
    if (!record) return '#64748b';
    if (mode === 'key') return camelotKeyMarkerColor(record.key);
    if (mode === 'target') {
      const maxAbs = Math.max(Math.abs(Number(extent && extent.min) || 0), Math.abs(Number(extent && extent.max) || 0), 1);
      return twoSidedResidualColor(targetEnergyResidual(record), maxAbs);
    }
    return pointColorValue(idx, mode);
  }
  function colorLegendGradient(mode) {
    mode = normalizePointColorMode(mode);
    if (mode === 'energy') {
      return 'linear-gradient(90deg, #440154 0%, #31688e 38%, #35b779 72%, #fde725 100%)';
    }
    if (mode === 'target') {
      return 'linear-gradient(90deg, #2563eb 0%, #f8fafc 50%, #ef4444 100%)';
    }
    if (mode === 'key') {
      return 'linear-gradient(90deg, #ef4444 0%, #f97316 8.3%, #f59e0b 16.6%, #eab308 25%, #84cc16 33.3%, #22c55e 41.6%, #14b8a6 50%, #06b6d4 58.3%, #3b82f6 66.6%, #6366f1 75%, #a855f7 83.3%, #ec4899 91.6%, #64748b 100%)';
    }
    return 'linear-gradient(90deg, #30123b 0%, #4663d8 20%, #1bcfd4 42%, #a4fc3c 62%, #f89540 82%, #7a0403 100%)';
  }
  function renderMapColorLegend(mode, extent) {
    mode = normalizePointColorMode(mode);
    if (!els.mapColorLegend) return;
    if (mode === 'genre') {
      if (usingWebglMap()) {
        const genres = Array.from(new Set(records.map(record => String(record.genre || record.raw_genre || 'Unknown'))))
          .sort((a, b) => a.localeCompare(b));
        els.mapColorLegend.classList.add('genre-legend');
        els.mapColorLegend.classList.remove('hidden');
        els.mapColorLegend.innerHTML =
          '<span class="map-color-legend-title">Genre</span>' +
          '<span class="map-genre-legend">' +
          genres.map(genre =>
            '<span class="map-genre-chip"><i style="background:' + esc(genreMarkerColor(genre)) + '"></i>' +
            esc(titleCaseGenre(genre)) + '</span>'
          ).join('') +
          '</span>';
        return;
      }
      els.mapColorLegend.classList.remove('genre-legend');
      els.mapColorLegend.classList.add('hidden');
      els.mapColorLegend.innerHTML = '';
      return;
    }
    els.mapColorLegend.classList.remove('genre-legend');
    const min = Number(extent && extent.min);
    const max = Number(extent && extent.max);
    const mid = (min + max) / 2;
    const places = mode === 'energy' || mode === 'target' ? 1 : 0;
    const label = value => Number.isFinite(value) ? fmt(value, places) : '';
    let rangeHtml = '<span class="map-color-range"><span>' + label(min) + '</span><span>' + label(mid) + '</span><span>' + label(max) + '</span></span>';
    if (mode === 'key') {
      rangeHtml = '<span class="map-color-range"><span>1</span><span>6/7</span><span>12</span><span>gray n/a</span></span>';
    }
    els.mapColorLegend.classList.remove('hidden');
    els.mapColorLegend.innerHTML =
      '<span class="map-color-legend-title">' + esc(extent.title) + '</span>' +
      '<span class="map-color-ramp" style="background:' + colorLegendGradient(mode) + '"></span>' +
      rangeHtml;
  }
  const simplexVertices = {
    style: { x: 180, y: 34 },
    rhythm: { x: 44, y: 270 },
    harmony: { x: 316, y: 270 },
  };
  function componentWeightsRaw() {
    const style = Math.max(0, Number(els.weightStyle ? els.weightStyle.value : 0));
    const rhythm = Math.max(0, Number(els.weightRhythm ? els.weightRhythm.value : 0));
    const harmony = Math.max(0, Number(els.weightHarmony ? els.weightHarmony.value : 0));
    const total = style + rhythm + harmony;
    if (total <= 1e-12) return { style: 0.5, rhythm: 0.3, harmony: 0.2 };
    return { style: style / total, rhythm: rhythm / total, harmony: harmony / total };
  }
  function setComponentSliders(weights) {
    if (els.weightStyle) els.weightStyle.value = String(clamp(weights.style, 0, 1));
    if (els.weightRhythm) els.weightRhythm.value = String(clamp(weights.rhythm, 0, 1));
    if (els.weightHarmony) els.weightHarmony.value = String(clamp(weights.harmony, 0, 1));
  }
  function simplexPoint(weights) {
    return {
      x: weights.style * simplexVertices.style.x + weights.rhythm * simplexVertices.rhythm.x + weights.harmony * simplexVertices.harmony.x,
      y: weights.style * simplexVertices.style.y + weights.rhythm * simplexVertices.rhythm.y + weights.harmony * simplexVertices.harmony.y,
    };
  }
  function updateSimplexHandle(weights) {
    if (!els.simplexHandle) return;
    const p = simplexPoint(weights);
    els.simplexHandle.setAttribute('cx', String(p.x));
    els.simplexHandle.setAttribute('cy', String(p.y));
  }
  function simplexWeightsFromPoint(px, py) {
    const a = simplexVertices.style;
    const b = simplexVertices.rhythm;
    const c = simplexVertices.harmony;
    const denom = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
    let style = ((b.y - c.y) * (px - c.x) + (c.x - b.x) * (py - c.y)) / denom;
    let rhythm = ((c.y - a.y) * (px - c.x) + (a.x - c.x) * (py - c.y)) / denom;
    let harmony = 1 - style - rhythm;
    style = clamp(style, 0, 1); rhythm = clamp(rhythm, 0, 1); harmony = clamp(harmony, 0, 1);
    const total = Math.max(1e-12, style + rhythm + harmony);
    return { style: style / total, rhythm: rhythm / total, harmony: harmony / total };
  }
  function eventToSvgPoint(ev) {
    const rect = els.simplex.getBoundingClientRect();
    return { x: (ev.clientX - rect.left) * 360 / Math.max(1, rect.width), y: (ev.clientY - rect.top) * 320 / Math.max(1, rect.height) };
  }
  function setWeightsFromSimplexEvent(ev) {
    const p = eventToSvgPoint(ev);
    setComponentSliders(simplexWeightsFromPoint(p.x, p.y));
    renderAll();
  }
  function applyPointColorMode() {
    const mode = normalizePointColorMode(appSetting('point_color', els.colorMode ? els.colorMode.value : 'genre'));
    if (els.colorMode && els.colorMode.value !== mode && Array.from(els.colorMode.options).some(opt => opt.value === mode)) {
      els.colorMode.value = mode;
      setAppSetting('point_color', mode);
    }
    if (usingWebglMap()) {
      const extent = mode === 'genre' ? null : colorScaleExtent(mode);
      renderMapColorLegend(mode, extent);
      if (webglMap) webglMap.render();
      safeUi('mini map draw', drawMiniMap);
      return;
    }
    if (!window.Plotly || !plot || !baseTraceIndices.length) return;

    if (mode === 'genre') {
      renderMapColorLegend(mode, null);
      for (const traceIdx of baseTraceIndices) {
        const marker = JSON.parse(JSON.stringify(originalBaseMarkers.get(traceIdx) || {}));
        Plotly.restyle(plot, { marker: [marker], showlegend: [true] }, [traceIdx]);
      }
      Plotly.relayout(plot, { 'legend.title.text': 'Genre' });
      safeUi('mini map draw', drawMiniMap);
      return;
    }

    const extent = colorScaleExtent(mode);
    renderMapColorLegend(mode, extent);
    baseTraceIndices.forEach(traceIdx => {
      const trace = plot.data[traceIdx] || {};
      const custom = Array.isArray(trace.customdata) ? trace.customdata : [];
      const values = custom.map(row => pointMarkerColor(Array.isArray(row) ? row[0] : NaN, mode, extent));
      const marker = mode === 'energy' || mode === 'tempo'
        ? {
            size: 9,
            color: values,
            colorscale: mode === 'energy' ? 'Viridis' : 'Turbo',
            cmin: extent.min,
            cmax: extent.max,
            showscale: false,
            line: { color: 'black', width: 0.5 },
          }
        : {
            size: 9,
            color: values,
            showscale: false,
            line: { color: 'black', width: 0.5 },
          };
      Plotly.restyle(plot, { marker: [marker], showlegend: [false] }, [traceIdx]);
    });
    Plotly.relayout(plot, { 'legend.title.text': extent.title });
    safeUi('mini map draw', drawMiniMap);
  }
  function weights() {
    return componentWeightsRaw();
  }
  function componentScores(c) {
    const styleScore = clamp(Number(c.style_score_norm ?? c.maest_score_norm ?? c.maest_similarity ?? 0), 0, 1);
    const tempoScore = clamp(Number(c.tempo_score_norm ?? c.tempo_similarity ?? 0), 0, 1);
    const grooveScore = clamp(Number(c.groove_score_norm ?? c.groove_similarity ?? 0), 0, 1);
    const computedRhythm = RHYTHM_TEMPO_WEIGHT * tempoScore + (1 - RHYTHM_TEMPO_WEIGHT) * grooveScore;
    const rhythmScore = clamp(Number(c.rhythm_score_norm ?? computedRhythm), 0, 1);
    const harmonyScore = clamp(Number(c.harmony_score_norm ?? c.chroma_score_norm ?? c.chroma_similarity ?? 0), 0, 1);
    return { styleScore, rhythmScore, harmonyScore, tempoScore, grooveScore };
  }
  function latentLinksPerTrack() {
    return clamp(Math.round(Number(appSetting('latent_links_per_track', els.latentLinksPerTrack ? els.latentLinksPerTrack.value : 3))), 0, 8);
  }
  function recommendedLinksHighlight() {
    return clamp(Math.round(Number(appSetting('recommended_links_highlight', els.recommendedLinksHighlight ? els.recommendedLinksHighlight.value : 25))), 1, 25);
  }
  function mapFxEnabled() {
    return !(els.mapEffectsEnabled && !els.mapEffectsEnabled.checked);
  }
  function showScoreValues() {
    return Boolean(appSetting('show_score_values', els.showScoreValues ? els.showScoreValues.checked : true));
  }
  function updateMapLinkSettingLabels() {
    if (els.latentLinksPerTrackVal) els.latentLinksPerTrackVal.textContent = String(latentLinksPerTrack());
    if (els.recommendedLinksHighlightVal) els.recommendedLinksHighlightVal.textContent = String(recommendedLinksHighlight());
  }
  function applyAppSettingsToControls() {
    if (els.colorMode) els.colorMode.value = normalizePointColorMode(appSetting('point_color', 'genre'));
    if (els.mapEffectsEnabled) els.mapEffectsEnabled.checked = Boolean(appSetting('map_fx', true));
    if (els.showScoreValues) els.showScoreValues.checked = showScoreValues();
    if (els.figureLightMode) els.figureLightMode.checked = figureLightMode();
    if (els.loudnessMatchSetting) els.loudnessMatchSetting.hidden = !config.app_mode;
    if (els.loudnessMatchMode) els.loudnessMatchMode.value = loudnessMatchMode();
    if (els.latentLinksPerTrack) els.latentLinksPerTrack.value = String(latentLinksPerTrack());
    if (els.recommendedLinksHighlight) els.recommendedLinksHighlight.value = String(recommendedLinksHighlight());
    applyFigureTheme();
    updateMapLinkSettingLabels();
  }
  function weightedCandidateScore(c) {
    const w = weights();
    const scores = componentScores(c);
    const score = w.style * scores.styleScore + w.rhythm * scores.rhythmScore + w.harmony * scores.harmonyScore;
    return Number.isFinite(score) ? score : 0;
  }
  function layoutInterpolatedPoints(w) {
    if (!Array.isArray(layoutEntries) || !layoutEntries.length) {
      return records.map(r => idxToPoint[String(r.idx)] || [0, 0]);
    }
    const target = [w.style, w.rhythm, w.harmony];
    const ranked = layoutEntries.map(entry => {
      const d2 = entry.weights.reduce((acc, val, idx) => acc + Math.pow(Number(val) - target[idx], 2), 0);
      return { entry, d2 };
    }).sort((a,b) => a.d2 - b.d2).slice(0, 12);
    if (!ranked.length) return records.map(r => idxToPoint[String(r.idx)] || [0, 0]);
    if ((config.layout_selection_mode || 'interpolated') === 'discrete' || ranked[0].d2 <= 1e-12) {
      return ranked[0].entry.points;
    }
    const weightsLocal = ranked.map(r => 1 / Math.max(1e-9, r.d2));
    const sum = weightsLocal.reduce((a,b) => a + b, 0);
    return ranked[0].entry.points.map((_, idx) => {
      let x = 0, y = 0;
      ranked.forEach((r, ridx) => {
        const c = weightsLocal[ridx] / sum;
        x += c * Number(r.entry.points[idx][0]);
        y += c * Number(r.entry.points[idx][1]);
      });
      return [x, y];
    });
  }
  function currentPoint(idx) {
    return currentPoints[Number(idx)] || idxToPoint[String(idx)] || null;
  }
  function recordMapTrails(previousPoints, nextPoints) {
    if (!mapFxEnabled()) return;
    if (!Array.isArray(previousPoints) || !Array.isArray(nextPoints) || previousPoints.length !== nextPoints.length) return;
    const now = performance.now();
    const fresh = [];
    for (let i = 0; i < nextPoints.length; i += 1) {
      const a = previousPoints[i];
      const b = nextPoints[i];
      if (!a || !b) continue;
      const dx = Number(b[0]) - Number(a[0]);
      const dy = Number(b[1]) - Number(a[1]);
      if (!Number.isFinite(dx) || !Number.isFinite(dy)) continue;
      if ((dx * dx + dy * dy) < 1e-6) continue;
      fresh.push({ from: [Number(a[0]), Number(a[1])], to: [Number(b[0]), Number(b[1])], t0: now, life: 780 });
      if (fresh.length >= 120) break;
    }
    if (!fresh.length) return;
    mapTrailSegments = mapTrailSegments.concat(fresh).slice(-220);
    ensureMapEffectsLoop();
  }
  function plotPointPx(pt) {
    if (usingWebglMap() && webglMap) return webglMap.dataToCanvasPoint(pt);
    if (!plot || !els.mapEffects || !plot._fullLayout || !pt) return null;
    const xa = plot._fullLayout.xaxis;
    const ya = plot._fullLayout.yaxis;
    const size = plot._fullLayout._size || { l: 0, t: 0 };
    if (!xa || !ya || typeof xa.l2p !== 'function' || typeof ya.l2p !== 'function') return null;
    const plotRect = plot.getBoundingClientRect();
    const canvasRect = els.mapEffects.getBoundingClientRect();
    const x = plotRect.left - canvasRect.left + Number(size.l || 0) + xa.l2p(Number(pt[0]));
    const y = plotRect.top - canvasRect.top + Number(size.t || 0) + ya.l2p(Number(pt[1]));
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    return { x, y };
  }
  function plotClickNearTrackPoint(ev, radius=18) {
    if (!ev || !els.mapEffects) return false;
    const canvasRect = els.mapEffects.getBoundingClientRect();
    const x = Number(ev.clientX) - canvasRect.left;
    const y = Number(ev.clientY) - canvasRect.top;
    if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
    const r2 = radius * radius;
    for (let i = 0; i < records.length; i += 1) {
      const p = plotPointPx(currentPoint(i));
      if (!p) continue;
      const dx = p.x - x;
      const dy = p.y - y;
      if ((dx * dx + dy * dy) <= r2) return true;
    }
    return false;
  }
  function mapDataBounds() {
    const pts = records.map(r => currentPoint(r.idx)).filter(Boolean);
    if (!pts.length) return null;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    pts.forEach(pt => {
      const x = Number(pt[0]);
      const y = Number(pt[1]);
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    });
    if (!Number.isFinite(minX) || !Number.isFinite(maxX) || !Number.isFinite(minY) || !Number.isFinite(maxY)) return null;
    const padX = Math.max(0.5, (maxX - minX) * 0.08);
    const padY = Math.max(0.5, (maxY - minY) * 0.08);
    return { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY };
  }
  function mapViewRanges() {
    if (usingWebglMap() && webglMap) return webglMap.viewRanges();
    if (!plot || !plot._fullLayout || !plot._fullLayout.xaxis || !plot._fullLayout.yaxis) return null;
    const xr = plot._fullLayout.xaxis.range;
    const yr = plot._fullLayout.yaxis.range;
    if (!Array.isArray(xr) || !Array.isArray(yr)) return null;
    const x0 = Number(xr[0]), x1 = Number(xr[1]), y0 = Number(yr[0]), y1 = Number(yr[1]);
    if (![x0, x1, y0, y1].every(Number.isFinite)) return null;
    return { x0, x1, y0, y1 };
  }
  function constrainMapRanges(x0, x1, y0, y1) {
    x0 = Number(x0);
    x1 = Number(x1);
    y0 = Number(y0);
    y1 = Number(y1);
    if (![x0, x1, y0, y1].every(Number.isFinite)) return null;
    const bounds = mapDataBounds();
    if (!bounds) return { x0, x1, y0, y1 };
    const baseW = Math.max(1e-9, Number(bounds.maxX) - Number(bounds.minX));
    const baseH = Math.max(1e-9, Number(bounds.maxY) - Number(bounds.minY));
    const width = clamp(Math.abs(x1 - x0), baseW * MAP_MIN_ZOOM_FRACTION, baseW * MAP_MAX_ZOOM_FRACTION);
    const height = clamp(Math.abs(y1 - y0), baseH * MAP_MIN_ZOOM_FRACTION, baseH * MAP_MAX_ZOOM_FRACTION);
    const dataCx = (Number(bounds.minX) + Number(bounds.maxX)) / 2;
    const dataCy = (Number(bounds.minY) + Number(bounds.maxY)) / 2;
    let cx = (x0 + x1) / 2;
    let cy = (y0 + y1) / 2;
    if (width <= baseW) cx = clamp(cx, Number(bounds.minX) + width / 2, Number(bounds.maxX) - width / 2);
    else cx = dataCx;
    if (height <= baseH) cy = clamp(cy, Number(bounds.minY) + height / 2, Number(bounds.maxY) - height / 2);
    else cy = dataCy;
    return {
      x0: cx - width / 2,
      x1: cx + width / 2,
      y0: cy - height / 2,
      y1: cy + height / 2,
    };
  }
  function mapRangesDiffer(a, b) {
    if (!a || !b) return false;
    const scale = Math.max(1, Math.abs(a.x1 - a.x0), Math.abs(a.y1 - a.y0));
    return Math.abs(a.x0 - b.x0) > scale * 1e-5
      || Math.abs(a.x1 - b.x1) > scale * 1e-5
      || Math.abs(a.y0 - b.y0) > scale * 1e-5
      || Math.abs(a.y1 - b.y1) > scale * 1e-5;
  }
  function relayoutMapRanges(x0, x1, y0, y1) {
    const next = constrainMapRanges(x0, x1, y0, y1);
    if (!next) return;
    if (usingWebglMap() && webglMap) {
      webglMap.setViewRanges(next.x0, next.x1, next.y0, next.y1);
      safeUi('map overlays after webgl map control', updatePacmapOverlays);
      return;
    }
    if (!window.Plotly || !plot) return;
    Plotly.relayout(plot, {
      'xaxis.range': [next.x0, next.x1],
      'yaxis.range': [next.y0, next.y1],
    }).then(() => {
      safeUi('map overlays after map control', updatePacmapOverlays);
      safeUi('map effects after map control', ensureMapEffectsLoop);
      safeUi('mini map after map control', drawMiniMap);
    });
  }
  function zoomMap(factor) {
    const view = mapViewRanges();
    if (!view) return;
    const cx = (view.x0 + view.x1) / 2;
    const cy = (view.y0 + view.y1) / 2;
    const halfW = Math.abs(view.x1 - view.x0) * Number(factor) / 2;
    const halfH = Math.abs(view.y1 - view.y0) * Number(factor) / 2;
    relayoutMapRanges(cx - halfW, cx + halfW, cy - halfH, cy + halfH);
  }
  function panMap(dxFrac, dyFrac) {
    const view = mapViewRanges();
    if (!view) return;
    const w = view.x1 - view.x0;
    const h = view.y1 - view.y0;
    const dx = w * Number(dxFrac || 0);
    const dy = h * Number(dyFrac || 0);
    relayoutMapRanges(view.x0 + dx, view.x1 + dx, view.y0 + dy, view.y1 + dy);
  }
  function resetMapView() {
    if (usingWebglMap() && webglMap) {
      webglMap.fitBounds();
      safeUi('map overlays after webgl reset', updatePacmapOverlays);
      safeUi('map effects after webgl reset', ensureMapEffectsLoop);
      safeUi('mini map after webgl reset', drawMiniMap);
      return;
    }
    if (!window.Plotly || !plot) return;
    Plotly.relayout(plot, { 'xaxis.autorange': true, 'yaxis.autorange': true }).then(() => {
      safeUi('map overlays after reset', updatePacmapOverlays);
      safeUi('map effects after reset', ensureMapEffectsLoop);
      safeUi('mini map after reset', drawMiniMap);
    });
  }
  function handleMapAction(action) {
    if (action === 'zoom-in') zoomMap(MAP_BUTTON_ZOOM_IN_FACTOR);
    else if (action === 'zoom-out') zoomMap(MAP_BUTTON_ZOOM_OUT_FACTOR);
    else if (action === 'pan-left') panMap(-0.18, 0);
    else if (action === 'pan-right') panMap(0.18, 0);
    else if (action === 'pan-up') panMap(0, 0.18);
    else if (action === 'pan-down') panMap(0, -0.18);
    else if (action === 'reset') resetMapView();
  }
  function miniMapPoint(pt, bounds, w, h) {
    const x = (Number(pt[0]) - bounds.minX) / Math.max(1e-9, bounds.maxX - bounds.minX);
    const y = (Number(pt[1]) - bounds.minY) / Math.max(1e-9, bounds.maxY - bounds.minY);
    return { x: clamp(x, 0, 1) * w, y: (1 - clamp(y, 0, 1)) * h };
  }
  function miniMapMarkerColor(record, mode, extent=null) {
    mode = normalizePointColorMode(mode);
    if (!record) return '#64748b';
    if (mode === 'genre') return genreMarkerColor(record.genre || record.raw_genre);
    if (mode === 'energy') return energyColor(energyOf(record));
    if (mode === 'tempo') return tempoColor(record.est_bpm, extent);
    if (mode === 'key') return camelotKeyMarkerColor(record.key);
    if (mode === 'target') {
      const maxAbs = Math.max(Math.abs(Number(extent && extent.min) || 0), Math.abs(Number(extent && extent.max) || 0), 1);
      return twoSidedResidualColor(targetEnergyResidual(record), maxAbs);
    }
    return '#64748b';
  }
  function drawMiniMap() {
    const canvas = els.mapMiniMap;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const widthCss = Math.max(1, rect.width || 180);
    const heightCss = Math.max(1, rect.height || 120);
    const dpr = window.devicePixelRatio || 1;
    const width = Math.floor(widthCss * dpr);
    const height = Math.floor(heightCss * dpr);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, widthCss, heightCss);
    const theme = mapTheme();
    ctx.fillStyle = theme.miniBg;
    ctx.fillRect(0, 0, widthCss, heightCss);
    const bounds = mapDataBounds();
    if (!bounds) return;
    const mode = normalizePointColorMode(appSetting('point_color', els.colorMode ? els.colorMode.value : 'genre'));
    const extent = mode === 'genre' ? null : colorScaleExtent(mode);
    records.forEach(record => {
      const pt = currentPoint(record.idx);
      if (!pt) return;
      const p = miniMapPoint(pt, bounds, widthCss, heightCss);
      ctx.fillStyle = miniMapMarkerColor(record, mode, extent);
      ctx.globalAlpha = 0.82;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2.1, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.globalAlpha = 1;
    const drawRing = (idx, color, radius) => {
      if (idx === null || idx === undefined) return;
      const pt = currentPoint(idx);
      if (!pt) return;
      const p = miniMapPoint(pt, bounds, widthCss, heightCss);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      ctx.stroke();
    };
    drawRing(selectedIdx, theme.selected, 4.5);
    if (transitionFromIdx === null && recommendationReferenceIdx !== selectedIdx) drawRing(recommendationReferenceIdx, roleColors.reference, 5.2);
    drawRing(transitionFromIdx, roleColors.track1, 5.5);
    drawRing(transitionToIdx, roleColors.track2, 5.5);
    const view = mapViewRanges();
    if (view) {
      const a = miniMapPoint([Math.min(view.x0, view.x1), Math.min(view.y0, view.y1)], bounds, widthCss, heightCss);
      const b = miniMapPoint([Math.max(view.x0, view.x1), Math.max(view.y0, view.y1)], bounds, widthCss, heightCss);
      const x = Math.min(a.x, b.x);
      const y = Math.min(a.y, b.y);
      const w = Math.max(4, Math.abs(b.x - a.x));
      const h = Math.max(4, Math.abs(b.y - a.y));
      ctx.strokeStyle = theme.miniViewStroke;
      ctx.lineWidth = 1.2;
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
    }
  }
  function recenterMapFromMiniMap(ev) {
    const canvas = els.mapMiniMap;
    const bounds = mapDataBounds();
    const view = mapViewRanges();
    if (!canvas || !bounds || !view) return;
    const rect = canvas.getBoundingClientRect();
    const nx = clamp((ev.clientX - rect.left) / Math.max(1, rect.width), 0, 1);
    const ny = clamp((ev.clientY - rect.top) / Math.max(1, rect.height), 0, 1);
    const cx = bounds.minX + nx * (bounds.maxX - bounds.minX);
    const cy = bounds.maxY - ny * (bounds.maxY - bounds.minY);
    const halfW = Math.abs(view.x1 - view.x0) / 2;
    const halfH = Math.abs(view.y1 - view.y0) / 2;
    relayoutMapRanges(cx - halfW, cx + halfW, cy - halfH, cy + halfH);
  }
  function drawGlow(ctx, p, radius, color, alpha) {
    if (!p) return;
    const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, radius);
    g.addColorStop(0, color.replace('__A__', String(alpha)));
    g.addColorStop(0.42, color.replace('__A__', String(alpha * 0.36)));
    g.addColorStop(1, color.replace('__A__', '0'));
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.fill();
  }
  function drawPulseRing(ctx, p, radius, color, phase, label) {
    if (!p) return;
    const pulse = 0.5 + (0.5 * Math.sin(phase));
    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur = 16 + pulse * 12;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2;
    ctx.globalAlpha = 0.62 + pulse * 0.22;
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius + pulse * 5, 0, Math.PI * 2);
    ctx.stroke();
    if (label) {
      ctx.shadowBlur = 8;
      ctx.fillStyle = color;
      ctx.font = '700 11px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(label, p.x, p.y - radius - 10);
    }
    ctx.restore();
  }
  function drawFlowLine(ctx, a, b, color, strength, phase, dotCount=1) {
    if (!a || !b) return;
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const dist = Math.hypot(dx, dy);
    if (!Number.isFinite(dist) || dist < 2) return;
    ctx.save();
    ctx.lineCap = 'round';
    ctx.strokeStyle = color.replace('__A__', String(0.08 + strength * 0.28));
    ctx.lineWidth = 1.2 + strength * 2.2;
    ctx.shadowColor = color.replace('__A__', '0.55');
    ctx.shadowBlur = 10 + strength * 12;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    for (let j = 0; j < dotCount; j += 1) {
      const t = (phase + j / Math.max(1, dotCount)) % 1;
      const ease = t * t * (3 - 2 * t);
      const x = a.x + dx * ease;
      const y = a.y + dy * ease;
      ctx.fillStyle = color.replace('__A__', String(0.45 + strength * 0.42));
      ctx.shadowBlur = 18 + strength * 10;
      ctx.beginPath();
      ctx.arc(x, y, 2.2 + strength * 3.2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }
  function drawBackgroundField(ctx, width, height, time) {
    ctx.save();
    const t = time / 1000;
    const cx = width * 0.52;
    const cy = height * 0.46;
    ctx.globalCompositeOperation = 'screen';
    ctx.strokeStyle = 'rgba(45, 212, 191, 0.045)';
    ctx.lineWidth = 1;
    for (let y = 34; y < height; y += 42) {
      const drift = Math.sin(t * 0.32 + y * 0.015) * 16;
      ctx.beginPath();
      for (let x = -40; x <= width + 40; x += 24) {
        const yy = y + Math.sin((x * 0.012) + t + y * 0.01) * 5;
        if (x === -40) ctx.moveTo(x + drift, yy);
        else ctx.lineTo(x + drift, yy);
      }
      ctx.stroke();
    }
    ctx.strokeStyle = 'rgba(251, 191, 36, 0.026)';
    for (let i = 0; i < 18; i += 1) {
      const y = ((i * 67 + (t * 12)) % (height + 120)) - 60;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y + Math.sin(t + i) * 20);
      ctx.stroke();
    }
    const rings = [0.35, 0.53, 0.72, 0.91];
    rings.forEach((r, i) => {
      const wobble = Math.sin(t * 0.21 + i) * 8;
      ctx.strokeStyle = 'rgba(148, 163, 184, ' + (0.028 - i * 0.004).toFixed(3) + ')';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.ellipse(cx, cy, width * r * 0.58 + wobble, height * r * 0.36, -0.12, 0, Math.PI * 2);
      ctx.stroke();
    });
    ctx.restore();
  }
  function strongestConnectionPairs() {
    const perTrack = latentLinksPerTrack();
    if (perTrack <= 0) return [];
    const byPair = new Map();
    Object.keys(simMap || {}).forEach(srcKey => {
      const src = Number(srcKey);
      const group = simMap[srcKey] || {};
      const candidates = Array.isArray(group.candidates) ? group.candidates : [];
      candidates
        .map(c => ({ c, score: weightedCandidateScore(c) }))
        .sort((a, b) => b.score - a.score)
        .slice(0, perTrack)
        .forEach(item => {
          const c = item.c;
          const dst = Number(c.idx);
          if (!Number.isFinite(src) || !Number.isFinite(dst) || src === dst) return;
          const key = src < dst ? src + ':' + dst : dst + ':' + src;
          const prev = byPair.get(key);
          if (!prev || item.score > prev.score) byPair.set(key, { src, dst, score: item.score });
        });
    });
    const pairs = Array.from(byPair.values());
    pairs.sort((a, b) => b.score - a.score);
    return pairs;
  }
  function drawStrongestConnections(ctx) {
    const pairs = strongestConnectionPairs();
    pairs.forEach((pair, i) => {
      const a = plotPointPx(currentPoint(pair.src));
      const b = plotPointPx(currentPoint(pair.dst));
      if (!a || !b) return;
      const alpha = usingWebglMap()
        ? 0.07 + (1 - i / Math.max(1, pairs.length)) * 0.105
        : 0.035 + (1 - i / Math.max(1, pairs.length)) * 0.055;
      const rgb = figureLightMode() ? '15, 23, 42' : '248, 250, 252';
      ctx.save();
      ctx.strokeStyle = 'rgba(' + rgb + ', ' + alpha.toFixed(3) + ')';
      ctx.lineWidth = usingWebglMap() ? 0.9 : 0.65;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      ctx.restore();
    });
  }
  function drawSequenceTransitionEffects(ctx, time) {
    const pts = sequence
      .map(idx => idx === null ? null : plotPointPx(currentPoint(idx)))
      .filter(Boolean);
    for (let i = 0; i < pts.length - 1; i += 1) {
      drawFlowLine(ctx, pts[i], pts[i + 1], roleColors.sequenceLine, 0.34, (time / 2100 + i * 0.19) % 1, 1);
    }
  }
  function drawSequencePathStatic(ctx) {
    const pts = sequence
      .map(idx => idx === null ? null : plotPointPx(currentPoint(idx)))
      .filter(Boolean);
    if (pts.length < 2) return;
    ctx.save();
    ctx.strokeStyle = 'rgba(251, 146, 60, 0.42)';
    ctx.lineWidth = 1.8;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.shadowColor = 'rgba(251, 146, 60, 0.24)';
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i += 1) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.stroke();
    ctx.restore();
  }
  function drawStaticMapBadge(ctx, p, label, color, radius=12, shape='circle') {
    if (!p) return;
    const theme = mapTheme();
    ctx.save();
    ctx.fillStyle = theme.staticBadgeFill;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.shadowColor = color;
    ctx.shadowBlur = 8;
    ctx.beginPath();
    if (shape === 'square') {
      const half = radius * 0.88;
      ctx.rect(p.x - half, p.y - half, half * 2, half * 2);
    } else {
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    }
    ctx.fill();
    ctx.stroke();
    if (label) {
      ctx.shadowBlur = 0;
      ctx.fillStyle = theme.staticBadgeLabel;
      ctx.font = '800 10px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(label), p.x, p.y + 0.4);
    }
    ctx.restore();
  }
  function drawWebglStaticOverlays(ctx) {
    if (!usingWebglMap()) return;
    const hoverMatchesSelection = hoveredIdx !== null && (
      hoveredIdx === selectedIdx ||
      hoveredIdx === transitionFromIdx ||
      hoveredIdx === transitionToIdx
    );
    if (hoveredIdx !== null && !hoverMatchesSelection) {
      drawStaticMapBadge(ctx, plotPointPx(currentPoint(hoveredIdx)), '', mapTheme().selected, 13);
    }
    if (selectedIdx !== null) drawStaticMapBadge(ctx, plotPointPx(currentPoint(selectedIdx)), '', mapTheme().selected, 13);
    if (transitionFromIdx === null && recommendationReferenceIdx !== null && Number(recommendationReferenceIdx) !== Number(selectedIdx)) {
      drawStaticMapBadge(ctx, plotPointPx(currentPoint(recommendationReferenceIdx)), 'R', roleColors.reference, 13, 'diamond');
    }
    if (transitionFromIdx !== null) drawStaticMapBadge(ctx, plotPointPx(currentPoint(transitionFromIdx)), '1', roleColors.track1, 15);
    if (transitionToIdx !== null) drawStaticMapBadge(ctx, plotPointPx(currentPoint(transitionToIdx)), '2', roleColors.track2, 15);

    sequence.forEach((idx, slot) => {
      if (idx === null) return;
      drawStaticMapBadge(ctx, plotPointPx(currentPoint(idx)), String(slot + 1), roleColors.sequence, 10, 'square');
    });

    const recContext = recommendationContext();
    const rows = preparedRecommendationRows(recommendedLinksHighlight()).rows;
    if (recContext.sourceIdx !== null) {
      rows.forEach((row, i) => {
        drawStaticMapBadge(ctx, plotPointPx(currentPoint(row.idx)), String(i + 1), roleColors.recommendation, 9);
      });
    }
  }
  function drawMapEffectsFrame(time) {
    mapEffectsFrame = null;
    const canvas = els.mapEffects;
    if (!canvas || (!usingWebglMap() && !plot)) return;
    const animated = mapFxEnabled();
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 20 || rect.height < 20) {
      return;
    }
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.floor(rect.width * dpr));
    const height = Math.max(1, Math.floor(rect.height * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    if (animated) drawBackgroundField(ctx, rect.width, rect.height, time);
    if (!usingWebglMap()) drawStrongestConnections(ctx);
    if (animated) drawSequenceTransitionEffects(ctx, time);
    else drawSequencePathStatic(ctx);

    if (animated) {
      mapTrailSegments = mapTrailSegments.filter(seg => (time - seg.t0) < seg.life);
      for (const seg of mapTrailSegments) {
        const a = plotPointPx(seg.from);
        const b = plotPointPx(seg.to);
        if (!a || !b) continue;
        const age = clamp((time - seg.t0) / seg.life, 0, 1);
        const alpha = (1 - age) * 0.22;
        ctx.save();
        ctx.strokeStyle = 'rgba(125, 211, 252, ' + alpha.toFixed(3) + ')';
        ctx.lineWidth = 1.1;
        ctx.shadowColor = 'rgba(45, 212, 191, ' + (alpha * 1.4).toFixed(3) + ')';
        ctx.shadowBlur = 12;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        ctx.restore();
      }
    } else {
      mapTrailSegments = [];
    }

    const recContext = recommendationContext();
    const rows = preparedRecommendationRows(recommendedLinksHighlight()).rows;
    const src = recContext.sourceIdx === null ? null : plotPointPx(currentPoint(recContext.sourceIdx));
    if (src && rows.length) {
      const finals = rows.map(r => Number(r.finalScore)).filter(Number.isFinite);
      const minScore = finals.length ? Math.min(...finals) : 0;
      const maxScore = finals.length ? Math.max(...finals) : 1;
      rows.forEach((row, i) => {
        const dst = plotPointPx(currentPoint(row.idx));
        if (!dst) return;
        const score = Number(row.finalScore);
        const scoreStrength = Number.isFinite(score) && Math.abs(maxScore - minScore) > 1e-9
          ? clamp((score - minScore) / (maxScore - minScore), 0, 1)
          : 1 - (i / Math.max(1, rows.length));
        const rankStrength = 1 - (i / Math.max(1, rows.length));
        const strength = clamp(0.35 * scoreStrength + 0.65 * rankStrength, 0, 1);
        drawFlowLine(ctx, src, dst, roleColors.recommendationLine, strength * 0.72, animated ? (time / 1750 + i * 0.071) % 1 : 0, animated ? (i < 4 ? 2 : 1) : 0);
        if (animated) drawGlow(ctx, dst, 14 + strength * 16, roleColors.recommendationLine, 0.06 + strength * 0.13);
      });
    }

    const fromP = transitionFromIdx === null ? null : plotPointPx(currentPoint(transitionFromIdx));
    const toP = transitionToIdx === null ? null : plotPointPx(currentPoint(transitionToIdx));
    if (fromP && toP) drawFlowLine(ctx, fromP, toP, roleColors.track2Line, 1, animated ? (time / 1300) % 1 : 0, animated ? 3 : 0);
    if (animated) {
      drawPulseRing(ctx, selectedIdx === null ? null : plotPointPx(currentPoint(selectedIdx)), 13, roleColors.selected, time / 520, '');
      drawPulseRing(ctx, fromP, 16, roleColors.track1, time / 470, '');
      drawPulseRing(ctx, toP, 16, roleColors.track2, time / 520 + 1.2, '');
    }
    drawWebglStaticOverlays(ctx);

    if (animated && activePane === 'explore') {
      ensureMapEffectsLoop();
    }
  }
  function ensureMapEffectsLoop() {
    if (mapEffectsFrame !== null || !window.requestAnimationFrame) return;
    mapEffectsFrame = window.requestAnimationFrame(drawMapEffectsFrame);
  }
  function updatePointCoordinates() {
    if (usingWebglMap() && webglMap) {
      webglMap.render();
      return;
    }
    if (!window.Plotly || !plot || !baseTraceIndices.length) return;
    const byGenre = new Map();
    for (const r of records) {
      const pt = currentPoint(r.idx);
      if (!pt) continue;
      const g = String(r.genre);
      if (!byGenre.has(g)) byGenre.set(g, { x: [], y: [] });
      byGenre.get(g).x.push(Number(pt[0]));
      byGenre.get(g).y.push(Number(pt[1]));
    }
    for (const traceIdx of baseTraceIndices) {
      const trace = plot.data[traceIdx] || {};
      const vals = byGenre.get(String(trace.name));
      if (vals) Plotly.restyle(plot, { x: [vals.x], y: [vals.y] }, [traceIdx]);
    }
  }
  function updateWeightLabels() {
    const w = weights();
    if (els.weightStyleVal) els.weightStyleVal.textContent = fmt(w.style, 2);
    if (els.weightRhythmVal) els.weightRhythmVal.textContent = fmt(w.rhythm, 2);
    if (els.weightHarmonyVal) els.weightHarmonyVal.textContent = fmt(w.harmony, 2);
    updateSimplexHandle(w);
    els.penaltyScaleVal.textContent = fmt(Number(els.penaltyScale.value || 0), 2);
  }
  function energyPenaltyScale() {
    return clamp(Number(els.penaltyScale ? els.penaltyScale.value : 0), 0, 1);
  }
  function energyAdjustment(rawError, baseline, { active=true } = {}) {
    const numericBaseline = Number(baseline);
    const hasBaseline = Number.isFinite(numericBaseline);
    if (!active || !Number.isFinite(Number(rawError))) {
      return { energyErrorPower: NaN, penaltyScale: 0, energyScore: 1, penalty: hasBaseline ? 0 : NaN, finalScore: hasBaseline ? numericBaseline : NaN };
    }
    const scale = energyPenaltyScale();
    const errorMagnitude = Math.abs(Number(rawError) || 0);
    const energyErrorPower = errorMagnitude * errorMagnitude;
    const energyScore = 1 / (1 + scale * energyErrorPower);
    const finalScore = hasBaseline ? numericBaseline * energyScore : NaN;
    const penalty = hasBaseline ? numericBaseline - finalScore : NaN;
    return { energyErrorPower, penaltyScale: scale, energyScore, penalty, finalScore };
  }
  function energyOf(record) {
    if (!record) return NaN;
    const source = els.energySource.value;
    const primary = source === 'glm' ? record.glm_energy : record.human_energy;
    const secondary = source === 'glm' ? record.human_energy : record.glm_energy;
    const p = Number(primary);
    if (Number.isFinite(p)) return p;
    const s = Number(secondary);
    if (Number.isFinite(s)) return s;
    return 5;
  }
  function camelotNumber(value) {
    const m = String(value || '').trim().toUpperCase().match(/^(\d{1,2})[AB]?/);
    if (!m) return null;
    const n = Number(m[1]);
    return Number.isFinite(n) ? n : null;
  }
  function recommendationFilterHtml() {
    return '<div class="recommendation-filterbar">' +
      '<label class="recommendation-search">Search<input data-rec-filter="query" type="search" placeholder="title, artist, genre, key, bpm, energy" value="' + esc(recFilters.query) + '"></label>' +
      '<label><span><input data-rec-filter="sameKey" type="checkbox"' + (recFilters.sameKey ? ' checked' : '') + '> Key family</span></label>' +
      '<label>BPM +/-<input data-rec-filter="bpmRange" type="number" min="0" max="60" step="1" placeholder="Any" value="' + esc(recFilters.bpmRange) + '"></label>' +
      '<label>Energy +/-<input data-rec-filter="energyRange" type="number" min="0" max="8" step="0.25" placeholder="Any" value="' + esc(recFilters.energyRange) + '"></label>' +
      '<label><span><input data-rec-filter="excludeUsed" type="checkbox"' + (recFilters.excludeUsed ? ' checked' : '') + '> Exclude used</span></label>' +
      '<label>Genre<select data-rec-filter="genreMode">' +
      optionHtml('any', 'Any', recFilters.genreMode) +
      optionHtml('same', 'Same', recFilters.genreMode) +
      optionHtml('different', 'Different', recFilters.genreMode) +
      '</select></label>' +
      '</div>';
  }
  function updateRecFiltersFromElement(el) {
    const key = el && el.getAttribute && el.getAttribute('data-rec-filter');
    if (!key) return false;
    if (key === 'sameKey' || key === 'excludeUsed') recFilters[key] = !!el.checked;
    else if (key === 'genreMode') recFilters.genreMode = el.value || 'any';
    else recFilters[key] = el.value || '';
    return true;
  }
  function recommendationSearchText(row) {
    const record = byIdx.get(Number(row && row.idx)) || row || {};
    return [
      record.title,
      record.artists,
      record.genre,
      record.raw_genre,
      record.key,
      roundedBpm(record),
      energyOf(record),
      record.mix_slug,
      record.filename,
    ].map(v => String(v == null ? '' : v).toLowerCase()).join(' ');
  }
  function recommendationMatchesQuery(row, query) {
    const tokens = String(query || '').trim().toLowerCase().split(/\s+/).filter(Boolean);
    if (!tokens.length) return true;
    const haystack = recommendationSearchText(row);
    return tokens.every(token => haystack.includes(token));
  }
  function preparedRecommendationRows(limit=25) {
    const allRows = rankedRecommendations().map((row, idx) => ({ ...row, globalRank: idx + 1 }));
    const query = recFilters.query || '';
    const matchingRows = allRows.filter(row => recommendationMatchesQuery(row, query));
    let rows = matchingRows;
    let pinnedOutsideQueryCount = 0;
    if (pinnedRecommendationIdxs.size) {
      const pinnedRows = allRows
        .filter(row => pinnedRecommendationIdxs.has(Number(row.idx)))
        .map(row => ({ ...row, pinned: true, pinnedOutsideQuery: !recommendationMatchesQuery(row, query) }));
      pinnedOutsideQueryCount = pinnedRows.filter(row => row.pinnedOutsideQuery).length;
      const pinnedSet = new Set(pinnedRows.map(row => Number(row.idx)));
      const otherRows = matchingRows.filter(row => !pinnedSet.has(Number(row.idx)));
      rows = pinnedRows.concat(otherRows);
    }
    return {
      allRows,
      rows: rows.slice(0, limit),
      matchedCount: matchingRows.length,
      pinnedOutsideQueryCount,
      query: String(query || '').trim(),
    };
  }
  function selectedIndices() {
    return new Set(sequence.filter(v => v !== null).map(Number));
  }
  function selectedIndicesExcept(slot) {
    const used = new Set();
    for (let i = 0; i < sequence.length; i += 1) {
      if (i === slot) continue;
      if (sequence[i] !== null) used.add(Number(sequence[i]));
    }
    return used;
  }
  function nextEmptySlot() {
    if (selectedSlot !== null && selectedSlot >= 0 && selectedSlot < sequence.length && sequence[selectedSlot] === null) return selectedSlot;
    const i = sequence.findIndex(v => v === null);
    return i >= 0 ? i : -1;
  }
  function targetSlot() {
    if (selectedSlot !== null && selectedSlot >= 0 && selectedSlot < sequence.length) return selectedSlot;
    return nextEmptySlot();
  }
  function previousFilledSlot(slot) {
    slot = Math.round(Number(slot));
    if (!Number.isFinite(slot) || slot <= 0) return -1;
    const start = Math.min(sequence.length - 1, slot - 1);
    for (let i = start; i >= 0; i -= 1) {
      if (sequence[i] !== null && byIdx.has(Number(sequence[i]))) return i;
    }
    return -1;
  }
  function recommendationContext(slot=targetSlot()) {
    if (!Number.isFinite(Number(slot)) || Number(slot) < 0) {
      return { mode: 'full', slot: -1, sourceIdx: null, sourceSlot: null };
    }
    slot = Math.round(Number(slot));
    if (transitionFromIdx !== null && byIdx.has(Number(transitionFromIdx))) {
      return { mode: 'track1', slot, sourceIdx: Number(transitionFromIdx), sourceSlot: null };
    }
    if (recommendationReferenceIdx !== null && byIdx.has(Number(recommendationReferenceIdx))) {
      return { mode: 'selected', slot, sourceIdx: Number(recommendationReferenceIdx), sourceSlot: null };
    }
    const sourceSlot = previousFilledSlot(slot);
    if (sourceSlot >= 0) {
      return { mode: 'sequence', slot, sourceIdx: Number(sequence[sourceSlot]), sourceSlot };
    }
    return { mode: 'initial', slot, sourceIdx: null, sourceSlot: null };
  }
  function recommendationContextLabel(ctx) {
    if (!ctx || ctx.mode === 'full') return 'Sequence is full';
    if (ctx.mode === 'track1') {
      const record = byIdx.get(Number(ctx.sourceIdx));
      return 'Scoring next-track candidates from Track 1' + (record ? ' "' + record.title + '"' : '');
    }
    if (ctx.mode === 'selected') {
      const record = byIdx.get(Number(ctx.sourceIdx));
      return 'Scoring next-track candidates from selected track' + (record ? ' "' + record.title + '"' : '');
    }
    if (ctx.mode === 'sequence') {
      const record = byIdx.get(Number(ctx.sourceIdx));
      return 'No track selected; scoring candidates after slot ' + (Number(ctx.sourceSlot) + 1) + (record ? ' "' + record.title + '"' : '');
    }
    return 'Initial suggestions use the target energy curve';
  }
  function canPlaceTrack(idx, slot) {
    idx = Number(idx);
    if (!Number.isFinite(idx) || !byIdx.has(idx) || slot < 0) return false;
    return !selectedIndicesExcept(slot).has(idx);
  }
  function setTransitionEndpoint(kind, idx, { toggle=true } = {}) {
    idx = Number(idx);
    if (!Number.isFinite(idx) || !byIdx.has(idx)) return;
    resetTransitionRenderState();
    resetTransitionAutomationState();
    if (kind === 'out') {
      const nextFrom = toggle && transitionFromIdx === idx ? null : idx;
      if (transitionFromIdx !== nextFrom) pinnedRecommendationIdxs.clear();
      if (transitionFromIdx !== nextFrom) {
        resetDeckAlignment('from');
        previewFormState.from_cue = '';
      }
      transitionFromIdx = nextFrom;
      if (transitionToIdx === idx) {
        transitionToIdx = null;
        resetDeckAlignment('to');
        previewFormState.to_cue = '';
      }
    } else {
      const nextTo = toggle && transitionToIdx === idx ? null : idx;
      if (transitionToIdx !== nextTo) {
        resetDeckAlignment('to');
        previewFormState.to_cue = '';
      }
      transitionToIdx = nextTo;
      if (transitionFromIdx === idx) {
        transitionFromIdx = null;
        resetDeckAlignment('from');
        previewFormState.from_cue = '';
      }
    }
    schedulePlanningSave();
    renderAll();
  }
  function assignTransitionFromDoubleClick(idx) {
    idx = Number(idx);
    if (!Number.isFinite(idx) || !byIdx.has(idx)) return;
    const now = Date.now();
    if (lastPointDoubleClickIdx === idx && (now - lastPointDoubleClickMs) < 900) return;
    lastPointDoubleClickIdx = idx;
    lastPointDoubleClickMs = now;
    setTransitionEndpoint('out', idx, { toggle: true });
  }
  function handlePlotDoubleClick(fromPoint=false) {
    const now = Date.now();
    if ((now - lastPointDoubleClickMs) < 900) return false;
    if (fromPoint && lastClickedIdx !== null && (now - lastClickMs) < 800) {
      assignTransitionFromDoubleClick(lastClickedIdx);
      return false;
    }
    if (transitionFromIdx !== null || transitionToIdx !== null) clearTransitionPair();
    return false;
  }
  function handleMapTrackClick(idx, nativeEvent=null) {
    idx = Number(idx);
    if (!Number.isFinite(idx) || !byIdx.has(idx)) return;
    const now = Date.now();
    lastPlotPointClickMs = now;
    const nativeDetail = Number(nativeEvent && nativeEvent.detail) || 0;
    const isDouble = nativeDetail >= 2 || (lastClickedIdx === idx && (now - lastClickMs) < 520);
    lastClickedIdx = idx;
    lastClickMs = now;
    selectTrack(idx);
    if (isDouble) assignTransitionFromDoubleClick(idx);
  }
  function handleMapBlankClick() {
    const clearRequestTime = Date.now();
    const selectedAtClick = selectedIdx;
    window.setTimeout(() => {
      if (lastPlotPointClickMs >= clearRequestTime - 40) return;
      if (selectedIdx !== selectedAtClick) return;
      if (selectedIdx === null) return;
      clearCurrentTrackSelection();
    }, 650);
  }
  function swapTransitionPair() {
    const oldFrom = transitionFromIdx;
    transitionFromIdx = transitionToIdx;
    transitionToIdx = oldFrom;
    [['from_cue', 'to_cue'], ['from_nudge_beats', 'to_nudge_beats'], ['from_beatgrid_ms', 'to_beatgrid_ms'], ['from_pitch_shift', 'to_pitch_shift']].forEach(([fromField, toField]) => {
      const value = previewFormState[fromField];
      previewFormState[fromField] = previewFormState[toField];
      previewFormState[toField] = value;
    });
    resetTransitionRenderState();
    resetTransitionAutomationState();
    schedulePlanningSave();
    renderAll();
  }
  function clearTransitionPair() {
    transitionFromIdx = null;
    transitionToIdx = null;
    pinnedRecommendationIdxs.clear();
    lastClickedIdx = null;
    lastClickMs = 0;
    lastPointDoubleClickMs = 0;
    lastPointDoubleClickIdx = null;
    resetTransitionRenderState();
    resetTransitionAutomationState();
    previewFormState.from_cue = '';
    previewFormState.to_cue = '';
    schedulePlanningSave();
    renderAll();
  }
  function setSequenceLength(n, shouldRender=true) {
    n = clamp(Math.round(Number(n || 10)), 2, 40);
    const old = sequence.slice();
    const oldTargets = targetValues.slice();
    sequenceLength = n;
    sequence = Array.from({ length: n }, (_, i) => i < old.length ? old[i] : null);
    targetValues = Array.from({ length: n }, (_, i) => i < oldTargets.length
      ? normalizeTargetEnergy(oldTargets[i], null)
      : DEFAULT_TARGET_ENERGY);
    selectedSlot = selectedSlot !== null && selectedSlot < n ? selectedSlot : null;
    if (shouldRender) renderAll();
  }
  function normalizeTargetEnergy(value, fallback=DEFAULT_TARGET_ENERGY) {
    if (value === null || value === undefined || value === '') return fallback;
    const n = Number(value);
    if (!Number.isFinite(n)) return fallback;
    return Math.round(clamp(n, 1, 9) * 10) / 10;
  }
  function setTargetSlotValue(slot, value, shouldRender=true) {
    slot = Math.round(Number(slot));
    value = normalizeTargetEnergy(value, NaN);
    if (!Number.isFinite(slot) || slot < 0 || slot >= targetValues.length || !Number.isFinite(value)) return;
    targetValues[slot] = value;
    selectedSlot = slot;
    if (shouldRender) renderAll();
  }
  function targetAnchors() {
    const anchors = [];
    for (let slot = 0; slot < targetValues.length; slot += 1) {
      const value = targetValues[slot];
      if (hasNumericValue(value)) anchors.push({ slot, value: clamp(Number(value), 1, 9) });
    }
    return anchors.sort((a,b) => a.slot - b.slot);
  }
  function targetGuide() {
    const anchors = targetAnchors();
    const out = Array.from({ length: sequenceLength }, () => null);
    if (anchors.length < 2) return out;
    for (let i = 0; i < sequenceLength; i += 1) {
      for (let j = 0; j < anchors.length - 1; j += 1) {
        const left = anchors[j];
        const right = anchors[j + 1];
        if (left.slot <= i && i <= right.slot) {
          const t = (i - left.slot) / Math.max(1, right.slot - left.slot);
          out[i] = left.value + t * (right.value - left.value);
          break;
        }
      }
    }
    return out;
  }
  function targetGuideSlots() {
    const guide = targetGuide();
    return guide.map((value, slot) => ({ value, slot }))
      .filter(item => Number.isFinite(item.value) && !hasNumericValue(targetValues[item.slot]))
      .map(item => item.slot);
  }
  function targetCurve() {
    return targetValues.map(value => hasNumericValue(value) ? Number(value) : null);
  }
  function targetForSlot(slot) {
    const value = targetValues[Math.round(Number(slot))];
    return hasNumericValue(value) ? Number(value) : null;
  }
  function clearTargetSlotValue(slot, shouldRender=true) {
    slot = Math.round(Number(slot));
    if (!Number.isFinite(slot) || slot < 0 || slot >= targetValues.length) return;
    targetValues[slot] = null;
    selectedSlot = slot;
    if (shouldRender) renderAll();
  }
  function applyTargetGuides() {
    mutatePlanning(() => {
      const guide = targetGuide();
      guide.forEach((value, slot) => {
        if (!hasNumericValue(targetValues[slot]) && Number.isFinite(value)) {
          targetValues[slot] = normalizeTargetEnergy(value, null);
        }
      });
    });
  }
  function scoreCandidate(c, slot) {
    const w = weights();
    const scores = componentScores(c);
    const baseline = clamp(
      w.style * scores.styleScore + w.rhythm * scores.rhythmScore + w.harmony * scores.harmonyScore,
      0,
      1
    );
    const record = byIdx.get(Number(c.idx));
    const e = energyOf(record);
    const target = targetForSlot(slot);
    const rawError = Number.isFinite(e) && Number.isFinite(target) ? e - target : NaN;
    const adjustment = energyAdjustment(rawError, baseline, { active: Number.isFinite(target) });
    return { baseline, ...scores, energy: e, target, hasEnergyTarget: Number.isFinite(target), rawError, ...adjustment };
  }
  function scoreInitialCandidate(idx, slot) {
    const record = byIdx.get(Number(idx));
    const e = energyOf(record);
    const target = targetForSlot(slot);
    const rawError = Number.isFinite(e) && Number.isFinite(target) ? e - target : NaN;
    const adjustment = energyAdjustment(rawError, 1, { active: Number.isFinite(target) });
    return {
      ...(record || {}),
      idx: Number(idx),
      maest_similarity: NaN,
      chroma_similarity: NaN,
      tempo_similarity: NaN,
      groove_similarity: NaN,
      maest_score_norm: NaN,
      chroma_score_norm: NaN,
      tempo_score_norm: NaN,
      groove_score_norm: NaN,
      baseline: NaN,
      styleScore: NaN,
      rhythmScore: NaN,
      harmonyScore: NaN,
      tempoScore: NaN,
      grooveScore: NaN,
      energy: e,
      target,
      hasEnergyTarget: Number.isFinite(target),
      rawError,
      energyErrorPower: adjustment.energyErrorPower,
      penaltyScale: adjustment.penaltyScale,
      penalty: 1 - adjustment.energyScore,
      energyScore: adjustment.energyScore,
      finalScore: adjustment.energyScore,
      initialRecommendation: true,
      slot,
    };
  }
  function scoreTransition(sourceIdx, destIdx, slot) {
    const entry = simMap[String(sourceIdx)] || {};
    const candidates = Array.isArray(entry.candidates) ? entry.candidates : [];
    const found = candidates.find(c => Number(c.idx) === Number(destIdx));
    if (found) return { ...found, ...scoreCandidate(found, slot), missing: false };

    const record = byIdx.get(Number(destIdx));
    const e = energyOf(record);
    const target = targetForSlot(slot);
    const rawError = Number.isFinite(e) && Number.isFinite(target) ? e - target : NaN;
    const adjustment = energyAdjustment(rawError, NaN, { active: Number.isFinite(target) });
    return {
      idx: Number(destIdx),
      maest_similarity: NaN,
      chroma_similarity: NaN,
      tempo_similarity: NaN,
      groove_similarity: NaN,
      maest_score_norm: NaN,
      chroma_score_norm: NaN,
      tempo_score_norm: NaN,
      groove_score_norm: NaN,
      baseline: NaN,
      styleScore: NaN,
      rhythmScore: NaN,
      harmonyScore: NaN,
      tempoScore: NaN,
      grooveScore: NaN,
      energy: e,
      target,
      hasEnergyTarget: Number.isFinite(target),
      rawError,
      energyErrorPower: adjustment.energyErrorPower,
      penaltyScale: adjustment.penaltyScale,
      penalty: adjustment.penalty,
      energyScore: adjustment.energyScore,
      finalScore: NaN,
      missing: true,
    };
  }
  function transitionRecommendationRank(sourceIdx, destIdx, slot) {
    const entry = simMap[String(sourceIdx)] || {};
    const candidates = Array.isArray(entry.candidates) ? entry.candidates : [];
    const rows = candidates
      .filter(c => Number(c.idx) !== Number(sourceIdx))
      .map(c => ({ idx: Number(c.idx), finalScore: Number(scoreCandidate(c, slot).finalScore) }))
      .filter(row => Number.isFinite(row.idx) && Number.isFinite(row.finalScore))
      .sort((a, b) => b.finalScore - a.finalScore);
    const rankIdx = rows.findIndex(row => Number(row.idx) === Number(destIdx));
    if (rankIdx < 0) return null;
    return { rank: rankIdx + 1, total: rows.length };
  }
  function rankedRecommendations() {
    const slot = targetSlot();
    if (slot < 0) return [];
    const used = selectedIndicesExcept(slot);
    const ctx = recommendationContext(slot);
    if (ctx.sourceIdx === null) {
      const penaltyScale = energyPenaltyScale();
      const rows = [];
      for (const idx of initialRecommendationOrder) {
        if (recFilters.excludeUsed && used.has(idx)) continue;
        const record = byIdx.get(idx);
        if (!record) continue;
        const targetEnergy = targetForSlot(slot);
        const energyRange = Number(recFilters.energyRange);
        const hasEnergyRange = String(recFilters.energyRange || '').trim() !== '' && Number.isFinite(energyRange) && energyRange >= 0;
        if (hasEnergyRange && Number.isFinite(targetEnergy) && Math.abs(energyOf(record) - targetEnergy) > energyRange) continue;
        rows.push({ ...scoreInitialCandidate(idx, slot), recommendationSourceMode: ctx.mode });
      }
      if (penaltyScale > 0) {
        rows.sort((a, b) => b.finalScore - a.finalScore);
      }
      return rows;
    }
    const sourceIdx = Number(ctx.sourceIdx);
    const srcRecord = byIdx.get(sourceIdx);
    const srcKeyFamily = camelotNumber(srcRecord && srcRecord.key);
    const bpmRange = Number(recFilters.bpmRange);
    const hasBpmRange = String(recFilters.bpmRange || '').trim() !== '' && Number.isFinite(bpmRange) && bpmRange >= 0;
    const energyRange = Number(recFilters.energyRange);
    const hasEnergyRange = String(recFilters.energyRange || '').trim() !== '' && Number.isFinite(energyRange) && energyRange >= 0;
    const targetEnergy = targetForSlot(slot);
    const genreMode = recFilters.genreMode || 'any';
    const srcGenre = String((srcRecord && (srcRecord.raw_genre || srcRecord.genre)) || '').toLowerCase();
    const entry = simMap[String(sourceIdx)] || {};
    const candidates = Array.isArray(entry.candidates) ? entry.candidates : [];
    const rows = [];
    for (const c of candidates) {
      const idx = Number(c.idx);
      if (!Number.isFinite(idx) || idx === sourceIdx) continue;
      if (recFilters.excludeUsed && used.has(idx)) continue;
      const record = byIdx.get(idx);
      if (!record) continue;
      if (recFilters.sameKey && srcKeyFamily !== null && camelotNumber(record.key) !== srcKeyFamily) continue;
      if (hasBpmRange) {
        const srcBpm = Number(srcRecord && srcRecord.est_bpm);
        const candBpm = Number(record.est_bpm);
        if (!Number.isFinite(srcBpm) || !Number.isFinite(candBpm) || Math.abs(candBpm - srcBpm) > bpmRange) continue;
      }
      if (hasEnergyRange && Number.isFinite(targetEnergy) && Math.abs(energyOf(record) - targetEnergy) > energyRange) continue;
      if (genreMode !== 'any') {
        const candGenre = String(record.raw_genre || record.genre || '').toLowerCase();
        if (genreMode === 'same' && candGenre !== srcGenre) continue;
        if (genreMode === 'different' && candGenre === srcGenre) continue;
      }
      const s = scoreCandidate(c, slot);
      rows.push({ ...c, ...s, slot, recommendationSourceIdx: sourceIdx, recommendationSourceSlot: ctx.sourceSlot, recommendationSourceMode: ctx.mode });
    }
    rows.sort((a,b) => b.finalScore - a.finalScore);
    return rows;
  }
  function updatePacmapOverlays() {
    if (usingWebglMap()) {
      if (webglMap) webglMap.render();
      safeUi('webgl static overlays', ensureMapEffectsLoop);
      return;
    }
    const hoverMatchesSelection = hoveredIdx !== null && (
      hoveredIdx === selectedIdx ||
      hoveredIdx === transitionFromIdx ||
      hoveredIdx === transitionToIdx
    );
    const hoveredPt = hoveredIdx === null || hoverMatchesSelection ? null : currentPoint(hoveredIdx);
    restyleTrace('Hovered', {
      x: [hoveredPt ? [hoveredPt[0]] : []],
      y: [hoveredPt ? [hoveredPt[1]] : []],
    });
    const selectedPt = selectedIdx === null ? null : currentPoint(selectedIdx);
    restyleTrace('Selected', {
      x: [selectedPt ? [selectedPt[0]] : []],
      y: [selectedPt ? [selectedPt[1]] : []],
    });
    const referenceIdx = transitionFromIdx === null ? recommendationReferenceIdx : null;
    const referencePt = referenceIdx === null || Number(referenceIdx) === Number(selectedIdx) ? null : currentPoint(referenceIdx);
    restyleTrace('Recommendation source', {
      x: [referencePt ? [referencePt[0]] : []],
      y: [referencePt ? [referencePt[1]] : []],
      text: [referencePt ? ['R'] : []],
    });
    const fromPt = transitionFromIdx === null ? null : currentPoint(transitionFromIdx);
    const toPt = transitionToIdx === null ? null : currentPoint(transitionToIdx);
    restyleTrace('Track 1', {
      x: [fromPt ? [fromPt[0]] : []],
      y: [fromPt ? [fromPt[1]] : []],
      text: [fromPt ? ['T1'] : []],
    });
    restyleTrace('Track 2', {
      x: [toPt ? [toPt[0]] : []],
      y: [toPt ? [toPt[1]] : []],
      text: [toPt ? ['T2'] : []],
    });
    restyleTrace('Transition pair', {
      x: [fromPt && toPt ? [fromPt[0], toPt[0]] : []],
      y: [fromPt && toPt ? [fromPt[1], toPt[1]] : []],
    });

    const pathX = [];
    const pathY = [];
    const pathText = [];
    const pathCustom = [];
    for (let i = 0; i < sequence.length; i += 1) {
      const idx = sequence[i];
      if (idx === null) continue;
      const pt = currentPoint(idx);
      if (!pt) continue;
      pathX.push(pt[0]);
      pathY.push(pt[1]);
      pathText.push(String(i + 1));
      pathCustom.push([idx]);
    }
    restyleTrace('Sequence path', {
      x: [pathX],
      y: [pathY],
      text: [pathText],
      customdata: [pathCustom],
      marker: [{ size: 14, color: 'rgba(251,146,60,0.24)', symbol: 'square', line: { color: roleColors.sequence, width: 2 } }],
      line: [{ color: 'rgba(251,146,60,0.58)', width: 2 }],
      textfont: [{ color: '#fed7aa', size: 12 }],
    });

    const rows = preparedRecommendationRows(recommendedLinksHighlight()).rows;
    const linkX = [];
    const linkY = [];
    const recX = [];
    const recY = [];
    const recText = [];
    const recCustom = [];
    const recContext = recommendationContext();
    const src = recContext.sourceIdx === null ? null : currentPoint(recContext.sourceIdx);
    if (src) {
      rows.forEach((row, i) => {
        const dst = currentPoint(row.idx);
        if (!dst) return;
        linkX.push(src[0], dst[0], null);
        linkY.push(src[1], dst[1], null);
        recX.push(dst[0]);
        recY.push(dst[1]);
        recText.push(String(i + 1));
        recCustom.push([Number(row.idx)]);
      });
    }
    restyleTrace('Recommended next links', { x: [linkX], y: [linkY] });
    restyleTrace('Recommended next', {
      x: [recX],
      y: [recY],
      text: [recText],
      customdata: [recCustom],
      marker: [{ size: 15, color: 'rgba(52,211,153,0.12)', line: { color: 'rgba(52,211,153,0.74)', width: 1.5 } }],
      textfont: [{ color: '#d1fae5', size: 11 }],
      hoverinfo: ['skip'],
    });
  }
  function selectTrack(idx) {
    setCandidateTrack(idx, { autoplay: true, showPopover: true, render: true });
  }
  function setCurrentTrack(idx) {
    setCandidateTrack(idx, { autoplay: false, showPopover: false, render: true, setRecommendationReference: true, setTransitionCandidate: false });
  }
  function appendTrack(idx) {
    const slot = targetSlot();
    if (slot < 0) return;
    idx = Number(idx);
    if (!canPlaceTrack(idx, slot)) return;
    mutatePlanning(() => {
      sequence[slot] = idx;
      selectedSlot = null;
      pinnedRecommendationIdxs.clear();
      transitionFromIdx = idx;
      transitionToIdx = null;
      resetDeckAlignment('from');
      resetDeckAlignment('to');
      resetTransitionRenderState();
      resetTransitionAutomationState();
      previewFormState.from_cue = '';
      previewFormState.to_cue = '';
      setCandidateTrack(idx, { autoplay: true, showPopover: true, render: false });
    });
  }
  function removeSlot(slot) {
    slot = Number(slot);
    if (slot < 0 || slot >= sequence.length || sequence[slot] === null) return;
    mutatePlanning(() => { sequence[slot] = null; });
  }
  function moveSequenceSlot(fromSlot, toSlot) {
    fromSlot = Math.round(Number(fromSlot));
    toSlot = Math.round(Number(toSlot));
    if (!Number.isFinite(fromSlot) || !Number.isFinite(toSlot)) return;
    if (fromSlot < 0 || fromSlot >= sequence.length || toSlot < 0 || toSlot >= sequence.length) return;
    if (fromSlot === toSlot) {
      sequenceDragSlot = null;
      safeUi('sequence render', renderSequence);
      return;
    }
    mutatePlanning(() => {
      const movedTrack = sequence.splice(fromSlot, 1)[0];
      const movedTarget = targetValues.splice(fromSlot, 1)[0];
      sequence.splice(toSlot, 0, movedTrack);
      targetValues.splice(toSlot, 0, movedTarget);
      if (selectedSlot === fromSlot) selectedSlot = toSlot;
      else if (selectedSlot !== null) {
        if (fromSlot < selectedSlot && selectedSlot <= toSlot) selectedSlot -= 1;
        else if (toSlot <= selectedSlot && selectedSlot < fromSlot) selectedSlot += 1;
      }
      sequenceDragSlot = null;
    });
  }
  function compareLibraryRecords(a, b) {
    const key = librarySort;
    if (!key) return 0;
    let av = a[key];
    let bv = b[key];
    if (key === 'human_energy' || key === 'glm_energy' || key === 'est_bpm' || key === 'duration_seconds') {
      av = Number(av);
      bv = Number(bv);
      if (!Number.isFinite(av)) av = -Infinity;
      if (!Number.isFinite(bv)) bv = -Infinity;
      return av === bv ? String(a.title).localeCompare(String(b.title)) : av - bv;
    }
    return String(av || '').localeCompare(String(bv || ''), undefined, { numeric: true, sensitivity: 'base' });
  }
  function renderLibrary() {
    if (!els.libraryTable) return;
    const query = String(libraryQuery || '').trim().toLowerCase();
    let rows = records.filter(r => {
      if (!query) return true;
      const haystack = [r.title, r.artists, r.key, r.genre, r.raw_genre, r.mix_slug, r.filename, r.est_bpm, r.duration_text, r.human_energy, r.glm_energy]
        .map(v => String(v == null ? '' : v).toLowerCase()).join(' ');
      return haystack.includes(query);
    });
    if (librarySort) {
      rows = rows.slice().sort(compareLibraryRecords);
      if (libraryDir === 'desc') rows = rows.reverse();
    }
    const sortTh = (key, label, cls='') => {
      const indicator = librarySort === key ? (libraryDir === 'asc' ? '▲' : '▼') : '';
      return '<th ' + (cls ? 'class="' + cls + '" ' : '') + 'data-library-sort="' + esc(key) + '" data-sort-indicator="' + indicator + '">' + label + '</th>';
    };
    let html = '<table><thead><tr>' +
      sortTh('title', 'Track') +
      sortTh('artists', 'Artist') +
      sortTh('key', 'Key', 'short center') +
      sortTh('est_bpm', 'BPM', 'short center') +
      sortTh('human_energy', 'Tagged<br>energy', 'short center wrap-head') +
      sortTh('glm_energy', 'Auto<br>energy', 'short center wrap-head') +
      sortTh('duration_seconds', 'Duration', 'short center') +
      sortTh('raw_genre', 'Genre') +
      '<th>Actions</th></tr></thead><tbody>';
    rows.forEach(r => {
      const rowClasses = [];
      if (Number(r.idx) === selectedIdx) rowClasses.push('selected-slot');
      if (Number(r.idx) === previewAudioIdx) rowClasses.push(previewAudioPlaying ? 'now-playing' : 'now-playing-paused');
      const rowClass = rowClasses.length ? ' class="' + rowClasses.join(' ') + '"' : '';
      const titleAttr = [r.title, r.artists].filter(Boolean).join(' - ');
      html += '<tr data-library-row-idx="' + r.idx + '"' + rowClass + '>' +
        '<td class="library-title-cell" title="' + esc(titleAttr) + '"><div class="library-title-summary">' + artworkHtml(r) + '<span>' + esc(r.title) + '</span></div></td>' +
        '<td>' + esc(r.artists) + '</td>' +
        '<td class="center">' + keyHtml(r.key) + '</td>' +
        '<td class="center">' + roundedBpm(r) + '</td>' +
        '<td class="center">' + fmtEnergy(r.human_energy) + '</td>' +
        '<td class="center">' + fmtEnergy(r.glm_energy) + '</td>' +
        '<td class="center">' + esc(durationText(r)) + '</td>' +
        '<td>' + esc(rawGenre(r)) + '</td>' +
        '<td><div class="library-actions">' +
        rowPlayerHtml(r) +
        '<button data-focus-track="' + r.idx + '">Select</button>' +
        '<button data-library-current="' + r.idx + '">Current</button>' +
        '<button class="track-one" data-library-outgoing="' + r.idx + '">Track 1</button>' +
        '<button class="track-two" data-library-incoming="' + r.idx + '">Track 2</button>' +
        '<button class="primary" data-library-place="' + r.idx + '">Place</button>' +
        '</div></td></tr>';
    });
    html += '</tbody></table>';
    els.libraryTable.innerHTML = html;
    updatePlayButtons();
    updateRowScrubbers();
  }
  function renderSequence() {
    const guide = targetGuide();
    let html = '<table><thead><tr><th class="num slot-col">Slot</th><th>Track</th><th class="target-col">Target</th><th class="num actual-col">Actual</th><th class="actions-col"></th></tr></thead><tbody>';
    for (let i = 0; i < sequence.length; i += 1) {
      const idx = sequence[i];
      const r = idx === null ? null : byIdx.get(idx);
      const classes = [];
      if (selectedSlot === i) classes.push('selected-slot');
      if (sequenceDragSlot === i) classes.push('drag-source');
      const cls = classes.length ? ' class="' + classes.join(' ') + '"' : '';
      html += '<tr' + cls + ' data-sequence-drop-slot="' + i + '"><td class="num"><span class="sequence-drag-handle" draggable="true" data-sequence-drag-slot="' + i + '" title="Drag to reorder">↕</span> ' + (i + 1) + '</td><td>';
      if (r) html += trackSummaryHtml(r, { size: 'compact', showArt: true, slot: i });
      else html += '<span class="muted">empty</span>';
      const targetValue = hasNumericValue(targetValues[i]) ? Number(targetValues[i]) : NaN;
      const guideValue = guide[i];
      const targetInput = '<input class="target-edit' + (Number.isFinite(targetValue) ? '' : ' is-unset') + '" data-target-slot="' + i + '" type="number" min="1" max="9" step="0.1" placeholder="—" title="' + esc(Number.isFinite(guideValue) ? 'Free target. Guide: ' + fmtEnergy(guideValue) : 'No energy target') + '" value="' + (Number.isFinite(targetValue) ? fmtEnergy(targetValue) : '') + '">';
      const clearTarget = Number.isFinite(targetValue)
        ? actionSymbolButton('data-clear-target-slot="' + i + '"', '×', 'Clear energy target', 'target-clear')
        : '';
      html += '</td><td class="target-col"><div class="target-cell">' + targetInput + clearTarget + '</div></td><td class="num actual-col">' + (r ? fmtEnergy(energyOf(r)) : '—') + '</td><td class="actions-col"><div class="table-actions">';
      if (r) html += '<button class="play-button" data-play-idx="' + r.idx + '" aria-label="Play">▶</button> ';
      html += actionSymbolButton('data-seq-slot="' + i + '"', selectedSlot === i ? '●' : '○', selectedSlot === i ? 'Selected slot' : 'Select slot', selectedSlot === i ? 'is-active' : '') + ' ';
      if (selectedIdx !== null && canPlaceTrack(selectedIdx, i)) {
        html += actionSymbolButton('data-place-slot="' + i + '"', r ? '⇄' : '+', r ? 'Replace with selected track' : 'Place selected track', 'primary') + ' ';
      }
      if (r) html += actionSymbolButton('data-remove-slot="' + i + '"', '−', 'Remove track from slot', 'danger');
      html += '</div></td></tr>';
    }
    html += '</tbody></table>';
    els.sequenceList.innerHTML = html;
  }
  function metricValue(v, d=3) {
    const text = fmt(v, d);
    return text || '<span class="muted">n/a</span>';
  }
  function scoreFeatureRows(score, { includeRhythmComponents=false } = {}) {
    const rows = [
      { key: 'style', label: 'Style', value: Number(score && score.styleScore), weight: weights().style },
      { key: 'rhythm', label: 'Rhythm', value: Number(score && score.rhythmScore), weight: weights().rhythm },
      { key: 'harmonic', label: 'Harmony', value: Number(score && score.harmonyScore), weight: weights().harmony },
      { key: 'energy', label: 'Energy fit', value: score && score.hasEnergyTarget ? Number(score.energyScore) : NaN, weight: score && score.hasEnergyTarget ? energyPenaltyScale() : 0 },
    ];
    if (includeRhythmComponents) {
      const rhythmIndex = rows.findIndex(row => row.key === 'rhythm');
      rows.splice(rhythmIndex + 1, 0,
        { key: 'tempo', label: 'Tempo similarity', value: Number(score && score.tempoScore), weight: RHYTHM_TEMPO_WEIGHT, rhythmComponent: true },
        { key: 'groove', label: 'Groove similarity', value: Number(score && score.grooveScore), weight: 1 - RHYTHM_TEMPO_WEIGHT, rhythmComponent: true },
      );
    }
    return rows;
  }
  function featureBarsHtml(score, keys=null, options={}) {
    const rows = scoreFeatureRows(score, options).filter(row => !keys || keys.includes(row.key));
    return '<div class="diagnostic-feature-bars">' + rows.map(row => {
      const finite = Number.isFinite(row.value);
      const pct = finite ? clamp(row.value, 0, 1) * 100 : 0;
      const weightLabel = row.rhythmComponent ? 'within rhythm ' : 'w ';
      return '<div class="diagnostic-feature-bar ' + esc(row.key) + (row.rhythmComponent ? ' rhythm-component' : '') + '">' +
        '<span>' + esc(row.label) + '</span>' +
        '<div class="diagnostic-bar-track"><i style="width:' + pct.toFixed(1) + '%"></i></div>' +
        '<b>' + (finite ? fmt(row.value, 3) : 'n/a') + '</b>' +
        '<em>' + weightLabel + metricValue(row.weight, 2) + '</em>' +
        '</div>';
    }).join('') + '</div>';
  }
  function scoreSummaryBarsHtml(score) {
    const rows = [
      { key: 'final', label: 'Final', value: Number(score && score.finalScore), strength: Number(score && score.finalScore) },
      { key: 'mix', label: 'Mix', value: Number(score && score.baseline), strength: Number(score && score.baseline) },
      { key: 'penalty', label: 'Loss', value: Number(score && score.penalty), strength: Number(score && score.penalty) },
    ];
    return '<div class="diagnostic-feature-bars focused-score-bars">' + rows.map(row => {
      const finite = Number.isFinite(row.value);
      const pct = finite && Number.isFinite(row.strength) ? clamp(row.strength, 0, 1) * 100 : 0;
      return '<div class="diagnostic-feature-bar ' + esc(row.key) + '">' +
        '<span>' + esc(row.label) + '</span>' +
        '<div class="diagnostic-bar-track"><i style="width:' + pct.toFixed(1) + '%"></i></div>' +
        '<b>' + (finite ? fmt(row.value, 2) : 'n/a') + '</b>' +
        '<em></em>' +
        '</div>';
    }).join('') + '</div>';
  }
  function flattenNumericValues(value, out=[]) {
    if (Array.isArray(value)) {
      value.forEach(item => flattenNumericValues(item, out));
      return out;
    }
    const n = Number(value);
    if (Number.isFinite(n)) out.push(n);
    return out;
  }
  function heatmapExtent(...values) {
    const nums = [];
    values.forEach(value => flattenNumericValues(value, nums));
    if (!nums.length) return { min: 0, max: 1 };
    let min = Math.min(...nums);
    let max = Math.max(...nums);
    if (Math.abs(max - min) < 1e-9) {
      min -= 0.5;
      max += 0.5;
    }
    return { min, max };
  }
  function heatmapColor(value, extent, stops=DIAGNOSTIC_GROOVE_HEATMAP_STOPS) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '#1f2937';
    const t = (n - Number(extent.min)) / Math.max(1e-9, Number(extent.max) - Number(extent.min));
    return segmentedColor(stops, t);
  }
  function diagnosticHeatmapGradientCss(stops=DIAGNOSTIC_GROOVE_HEATMAP_STOPS) {
    return 'linear-gradient(90deg, ' + stops.map(stop => stop[1] + ' ' + Math.round(stop[0] * 100) + '%').join(', ') + ')';
  }
  function diagnosticHeatmapLegendHtml(extent, stops=DIAGNOSTIC_GROOVE_HEATMAP_STOPS) {
    const min = Number(extent && extent.min);
    const max = Number(extent && extent.max);
    return '<div class="diagnostic-heatmap-legend" title="Embedding cell color scale">' +
      '<span>low ' + esc(Number.isFinite(min) ? fmt(min, 2) : '') + '</span>' +
      '<i style="background:' + esc(diagnosticHeatmapGradientCss(stops)) + '"></i>' +
      '<span>high ' + esc(Number.isFinite(max) ? fmt(max, 2) : '') + '</span>' +
      '</div>';
  }
  const PITCH_CLASS_LABELS = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#'];
  const PITCH_CLASS_INDEX = {
    A: 0,
    'A#': 1, BB: 1,
    B: 2, CB: 2,
    'B#': 3, C: 3,
    'C#': 4, DB: 4,
    D: 5,
    'D#': 6, EB: 6,
    E: 7, FB: 7,
    'E#': 8, F: 8,
    'F#': 9, GB: 9,
    G: 10,
    'G#': 11, AB: 11,
  };
  const GROOVE_BAND_LABELS = ['Low', 'Mid', 'High'];
  const GROOVE_BAND_DISPLAY_ORDER = [2, 1, 0];
  function normalizePitchName(value) {
    const raw = String(value || '').trim().replace(/♯/g, '#').replace(/♭/g, 'b');
    const m = raw.toUpperCase().match(/^([A-G])([#B]?)/);
    if (!m) return '';
    return m[1] + (m[2] || '');
  }
  function parseTaggedKey(value) {
    const raw = String(value || '').trim();
    if (!raw || raw.toLowerCase() === 'nan') return null;
    const camelot = raw.toUpperCase().match(/^(\d{1,2})([AB])$/);
    const canonical = camelot ? camelotCanonicalName(raw) : '';
    const keyText = canonical || ((raw.match(/\(([A-G][#b♯♭]?m?)\)/i) || [])[1]) || raw;
    const parsed = String(keyText).trim().replace(/\s+/g, '').replace(/♯/g, '#').replace(/♭/g, 'b').match(/^([A-G])([#b]?)(m|min|minor|maj|major)?$/i);
    if (!parsed) return null;
    const pitch = normalizePitchName(parsed[1] + (parsed[2] || ''));
    if (!(pitch in PITCH_CLASS_INDEX)) return null;
    const suffix = String(parsed[3] || '').toLowerCase();
    const mode = camelot
      ? (camelot[2] === 'A' ? 'minor' : 'major')
      : (suffix === 'm' || suffix === 'min' || suffix === 'minor' ? 'minor' : 'major');
    return { tonic: pitch, mode };
  }
  function taggedKeyPitchClassSet(record) {
    const parsed = parseTaggedKey(record && record.key);
    if (!parsed) return new Set();
    const tonic = PITCH_CLASS_INDEX[parsed.tonic];
    const intervals = parsed.mode === 'minor' ? [0, 2, 3, 5, 7, 8, 10] : [0, 2, 4, 5, 7, 9, 11];
    return new Set(intervals.map(interval => PITCH_CLASS_LABELS[(tonic + interval) % 12]));
  }
  function taggedCanonicalKeyText(record) {
    const key = String((record && record.key) || '').trim();
    if (!key || key.toLowerCase() === 'nan') return 'n/a';
    return canonicalKeyText(key) || key;
  }
  function diagnosticTrackHeaderHtml(record, role) {
    return '<div class="diagnostic-embedding-title"><b>' + esc(role) + ': ' + esc(record.title || 'Untitled') + '</b>' +
      '<span>' + esc(record.artists || record.raw_genre || '') + ' / Key ' + esc(taggedCanonicalKeyText(record)) + '</span></div>';
  }
  function diagnosticChromaHtml(record, extent, role='Track') {
    const values = Array.isArray(record && record.diagnostic_chroma) ? record.diagnostic_chroma : [];
    if (!values.length) return '<div class="muted">No chroma embedding available.</div>';
    const cells = Array.from({ length: 12 }, (_, i) => {
      const value = Number(values[i]);
      const label = Number.isFinite(value) ? fmt(value, 3) : 'n/a';
      return '<span class="diagnostic-cell" title="' + esc(PITCH_CLASS_LABELS[i]) + ': ' + esc(label) + '" style="background:' + heatmapColor(value, extent, DIAGNOSTIC_CHROMA_HEATMAP_STOPS) + '"></span>';
    }).join('');
    const inKey = taggedKeyPitchClassSet(record);
    const pitchLabels = PITCH_CLASS_LABELS.map(label => {
      const isInKey = inKey.has(label);
      return '<span' + (isInKey ? ' class="in-key" title="In tagged key"' : '') + '>' + esc(label) + '</span>';
    }).join('');
    return '<div class="diagnostic-embedding-panel">' +
      diagnosticTrackHeaderHtml(record, role) +
      '<div class="diagnostic-embedding-block">' +
      '<div class="diagnostic-embedding-label"><span>Chroma</span><span>Tagged key: ' + esc(taggedCanonicalKeyText(record)) + '</span></div>' +
      '<div class="diagnostic-cell-row">' + cells + '</div>' +
      '<div class="diagnostic-pitch-labels">' + pitchLabels + '</div>' +
      '</div></div>';
  }
  function diagnosticGrooveHtml(record, extent, role='Track') {
    const rows = Array.isArray(record && record.diagnostic_groove) ? record.diagnostic_groove : [];
    if (!rows.length) return '<div class="muted">No groove embedding available.</div>';
    const columnCount = Math.max(1, rows.length || 16);
    const cells = ['<span class="diagnostic-groove-corner"></span>'];
    for (let col = 0; col < columnCount; col += 1) {
      const beatLabel = col % 4 === 0 ? ('Beat ' + (Math.floor(col / 4) + 1)) : '';
      cells.push('<span class="diagnostic-groove-beat">' + esc(beatLabel) + '</span>');
    }
    GROOVE_BAND_DISPLAY_ORDER.forEach(band => {
      cells.push('<span class="diagnostic-groove-band">' + esc(GROOVE_BAND_LABELS[band] || ('B' + (band + 1))) + '</span>');
      for (let col = 0; col < columnCount; col += 1) {
        const row = Array.isArray(rows[col]) ? rows[col] : [];
        const value = Number(row[band]);
        const label = Number.isFinite(value) ? fmt(value, 3) : 'n/a';
        const beat = Math.floor(col / 4) + 1;
        const subdivision = (col % 4) + 1;
        cells.push('<span class="diagnostic-cell" title="Beat ' + beat + ', subdivision ' + subdivision + ', ' + esc(GROOVE_BAND_LABELS[band]) + ': ' + esc(label) + '" style="background:' + heatmapColor(value, extent) + '"></span>');
      }
    });
    return '<div class="diagnostic-embedding-panel">' +
      diagnosticTrackHeaderHtml(record, role) +
      '<div class="diagnostic-embedding-block">' +
      '<div class="diagnostic-embedding-label"><span>Groove</span><span>4 beats x 4 subdivisions x 3 bands</span></div>' +
      '<div class="diagnostic-groove-grid" style="grid-template-columns:34px repeat(' + columnCount + ', minmax(0, 1fr))">' + cells.join('') + '</div>' +
      '</div></div>';
  }
  function diagnosticEmbeddingComparisonHtml(from, to) {
    const extents = {
      chroma: heatmapExtent(from && from.diagnostic_chroma, to && to.diagnostic_chroma),
      groove: heatmapExtent(from && from.diagnostic_groove, to && to.diagnostic_groove),
    };
    return '<div class="diagnostic-embedding-compare">' +
      '<div class="diagnostic-embedding-section"><h4>Harmony / chroma detail</h4>' +
      diagnosticHeatmapLegendHtml(extents.chroma, DIAGNOSTIC_CHROMA_HEATMAP_STOPS) +
      '<div class="diagnostic-embedding-stack">' +
      diagnosticChromaHtml(from, extents.chroma, 'Track 1') +
      diagnosticChromaHtml(to, extents.chroma, 'Track 2') +
      '</div></div>' +
      '<div class="diagnostic-embedding-section"><h4>Groove detail <span class="muted">(part of Rhythm)</span></h4>' +
      diagnosticHeatmapLegendHtml(extents.groove) +
      '<div class="diagnostic-embedding-stack">' +
      diagnosticGrooveHtml(from, extents.groove, 'Track 1') +
      diagnosticGrooveHtml(to, extents.groove, 'Track 2') +
      '</div></div>' +
      '</div>';
  }
  function diagnosticWaveformCardHtml(record, role, slot=null) {
    const idx = Number(record && record.idx);
    const meta = [energyMetaHtml(record, slot), esc(durationText(record))].filter(Boolean).join(' &sdot; ');
    return '<div class="diagnostic-waveform-card">' +
      '<div class="diagnostic-waveform-head"><button class="play-button" data-play-idx="' + idx + '" data-play-mode="diagnostics" aria-label="Play">▶</button><b>' + esc(role) + ': ' + esc(record.title || 'Untitled') + '</b><span>' + meta + '</span></div>' +
      '<canvas class="diagnostic-waveform-canvas" data-diagnostic-waveform-idx="' + idx + '" height="96" aria-label="' + esc(role) + ' full waveform"></canvas>' +
      '</div>';
  }
  function diagnosticWaveformComparisonHtml(from, to, slot=null) {
    return '<div class="diagnostic-waveform-grid">' +
      diagnosticWaveformCardHtml(from, 'Track 1') +
      diagnosticWaveformCardHtml(to, 'Track 2', slot) +
      '</div>';
  }
  function transitionFocusHtml(from, to, score, slot) {
    return '<div class="diagnostic-transition-pair">' +
      '<div>' + trackSummaryHtml(from, { size: 'large', showArt: true }) + '</div>' +
      '<div class="diagnostic-arrow">→</div>' +
      '<div>' + trackSummaryHtml(to, { size: 'large', showArt: true, slot }) + '</div>' +
      '</div>' +
      '<div class="diagnostic-subhead">Slot ' + (slot + 1) + (score && score.missing ? ' <span class="warn">missing direct similarity row</span>' : '') + '</div>' +
      scoreSummaryBarsHtml(score) +
      featureBarsHtml(score, null, { includeRhythmComponents: true }) +
      '<div class="focused-transition-section">' + diagnosticEmbeddingComparisonHtml(from, to) + '</div>' +
      '<div class="focused-transition-waveforms"><h3>Full track waveforms</h3>' +
      '<div class="diagnostic-subhead">Click or drag either waveform to play and scrub the full track.</div>' +
      diagnosticWaveformComparisonHtml(from, to, slot) + '</div>';
  }
  function sequenceTransitionTableHtml() {
    const filled = [];
    for (let i = 0; i < sequence.length; i += 1) {
      if (sequence[i] !== null) filled.push({ slot: i, idx: Number(sequence[i]) });
    }
    if (!filled.length) {
      return '<section class="diagnostic-card wide sequence-score-card"><div class="muted">Add tracks to inspect sequence transition scores.</div></section>';
    }

    const metricCell = (value, cls, strength=value, label='') =>
      recommendationMetricCell(value, cls, strength, 2, label);
    const emptyMetricCell = () =>
      '<td class="num recommendation-score-cell sequence-empty-score"><span class="muted">—</span></td>';
    const rankCell = rankInfo =>
      '<td class="num recommendation-score-cell">' + (rankInfo ? rankMeterHtml(rankInfo) : '<span class="muted">—</span>') + '</td>';

    let html = '<section class="diagnostic-card wide sequence-score-card"><div class="diagnostic-table-wrap"><table><thead><tr>' +
      '<th class="num">Slot</th><th class="num">Actions</th><th>Waveform</th><th>Track</th><th class="num">Rank</th><th class="num">Final</th><th class="num">Mix</th><th class="num">Loss</th>' +
      '<th class="num">Style</th><th class="num">Rhythm</th><th class="num">Harmony</th><th class="num">Energy fit</th>' +
      '</tr></thead><tbody>';

    filled.forEach((current, i) => {
      const record = byIdx.get(current.idx);
      const previous = i > 0 ? filled[i - 1] : null;
      const score = previous ? scoreTransition(previous.idx, current.idx, current.slot) : scoreInitialCandidate(current.idx, current.slot);
      const rankInfo = previous ? transitionRecommendationRank(previous.idx, current.idx, current.slot) : null;
      html += '<tr>' +
        '<td class="num">' + (current.slot + 1) + '</td>' +
        '<td class="num"><div class="table-actions"><button class="play-button" data-play-idx="' + current.idx + '" data-play-mode="library" aria-label="Play">▶</button></div></td>' +
        '<td class="waveform-cell">' + rowWaveformScrubberHtml(record, { showTime: false }) + '</td>' +
        '<td>' + trackSummaryHtml(record, { size: 'compact', showArt: true, slot: current.slot }) +
          (score && score.missing ? '<div class="warn">missing similarity row</div>' : '') + '</td>' +
        rankCell(rankInfo) +
        (previous ? metricCell(score.finalScore, 'final', score.finalScore, 'Final score') : emptyMetricCell()) +
        (previous ? metricCell(score.baseline, 'mix', score.baseline, 'Weighted mix score') : emptyMetricCell()) +
        metricCell(score.penalty, 'penalty', score.penalty, previous ? 'Energy score loss' : 'Initial target loss') +
        (previous ? metricCell(score.styleScore, 'style', score.styleScore, 'Style score') : emptyMetricCell()) +
        (previous ? metricCell(score.rhythmScore, 'rhythm', score.rhythmScore, 'Rhythm score') : emptyMetricCell()) +
        (previous ? metricCell(score.harmonyScore, 'harmonic', score.harmonyScore, 'Harmony score') : emptyMetricCell()) +
        (previous ? metricCell(score.energyScore, 'energy', score.energyScore, 'Energy fit score') : emptyMetricCell()) +
        '</tr>';
    });
    html += '</tbody></table></div></section>';
    return html;
  }
  function renderCurrentTransitionScore() {
    if (!els.currentTransitionScore) return;
    const from = transitionFromIdx === null ? null : byIdx.get(transitionFromIdx);
    const to = transitionToIdx === null ? null : byIdx.get(transitionToIdx);
    if (!from || !to) {
      const ctx = recommendationContext();
      const source = ctx.sourceIdx === null ? null : byIdx.get(Number(ctx.sourceIdx));
      els.currentTransitionScore.innerHTML = source
        ? '<div class="diagnostic-subhead">No explicit transition selected. Current recommendation source:</div>' + trackSummaryHtml(source, { size: 'large', showArt: true })
        : '<div class="muted">Assign Track 1 and Track 2 to inspect a selected transition. With no track selected, recommendations fall back to the previous filled sequence slot.</div>';
      return;
    }
    const slot = Math.max(0, targetSlot());
    const score = scoreTransition(transitionFromIdx, transitionToIdx, slot);
    els.currentTransitionScore.innerHTML = transitionFocusHtml(from, to, score, slot);
  }
  function renderTransitionDiagnostics() {
    if (!els.transitionDiagnostics) return;
    let html = '<div class="diagnostics-grid">';
    html += sequenceTransitionTableHtml();
    html += '</div>';
    els.transitionDiagnostics.innerHTML = html;
    updatePlayButtons();
    updateRowScrubbers();
  }
  function valueRange(rows, getter) {
    const vals = rows.map(getter).map(Number).filter(Number.isFinite);
    if (!vals.length) return { min: NaN, max: NaN };
    return { min: Math.min(...vals), max: Math.max(...vals) };
  }
  function relativeStrength(value, range, fallback=null) {
    value = Number(value);
    if (!Number.isFinite(value)) return NaN;
    if (range && Number.isFinite(range.min) && Number.isFinite(range.max) && Math.abs(range.max - range.min) > 1e-9) {
      return clamp((value - range.min) / (range.max - range.min), 0, 1);
    }
    if (fallback !== null) return clamp(Number(fallback), 0, 1);
    return clamp(value, 0, 1);
  }
  function recommendationMetricCell(value, cls, strength, digits=2, label='') {
    return '<td class="num recommendation-score-cell' + (showScoreValues() ? '' : ' bars-only') + '">' +
      scoreMeterHtml(value, cls, strength, digits, label) +
      '</td>';
  }
  function scoreMeterHtml(value, cls, strength, digits=2, label='', options={}) {
    const numeric = Number(value);
    const finite = Number.isFinite(numeric);
    const s = Number(strength);
    const pct = finite && Number.isFinite(s) ? clamp(s, 0, 1) * 100 : 0;
    const title = label ? label + ': ' + (finite ? fmt(numeric, digits) : 'n/a') : '';
    const valuesVisible = Boolean(options && options.forceValues) || showScoreValues();
    return '<div class="score-meter ' + esc(cls) + (finite ? '' : ' empty') + (valuesVisible ? '' : ' bars-only') + '"' + (title ? ' title="' + esc(title) + '"' : '') + '>' +
      '<span class="score-meter-value">' + (finite ? fmt(numeric, digits) : 'n/a') + '</span>' +
      '<span class="score-meter-track"><i style="width:' + pct.toFixed(1) + '%"></i></span>' +
      '</div>';
  }
  function rankMeterHtml(rankInfo, options={}) {
    const rank = Number(rankInfo && rankInfo.rank);
    const total = Number(rankInfo && rankInfo.total);
    const finite = Number.isFinite(rank) && rank > 0;
    const strength = finite && Number.isFinite(total) && total > 1
      ? 1 - ((rank - 1) / Math.max(1, total - 1))
      : (finite ? 1 : 0);
    const title = finite
      ? 'Recommendation rank: #' + rank + (Number.isFinite(total) && total > 0 ? ' of ' + total : '')
      : 'Recommendation rank: n/a';
    const valuesVisible = Boolean(options && options.forceValues) || showScoreValues();
    return '<div class="score-meter rank' + (finite ? '' : ' empty') + (valuesVisible ? '' : ' bars-only') + '" title="' + esc(title) + '">' +
      '<span class="score-meter-value">' + (finite ? ('#' + rank) : 'n/a') + '</span>' +
      '<span class="score-meter-track"><i style="width:' + (clamp(strength, 0, 1) * 100).toFixed(1) + '%"></i></span>' +
      '</div>';
  }
  function renderRecommendations() {
    const restoreQueryFocus = document.activeElement
      && document.activeElement.getAttribute
      && document.activeElement.getAttribute('data-rec-filter') === 'query';
    const filterbar = recommendationFilterHtml();
    const slot = targetSlot();
    if (slot < 0) {
      els.recommendationPanel.innerHTML = filterbar + '<div class="muted" style="padding:10px;">Sequence is full. Select a slot to replace a track.</div>';
      return;
    }
    const ctx = recommendationContext(slot);
    const prepared = preparedRecommendationRows(25);
    const rows = prepared.rows;
    const actionLabel = sequence[slot] === null ? 'Append' : 'Replace';
    const actionSymbol = sequence[slot] === null ? '+' : '⇄';
    let html = filterbar +
      '<div class="muted" style="padding:8px 8px 0;">' +
      esc(recommendationContextLabel(ctx)) +
      ' for slot <b>' + (slot + 1) + '</b> (' + actionLabel.toLowerCase() + '). ' +
      (prepared.query
        ? 'Showing <b>' + prepared.matchedCount + '</b> matches from <b>' + prepared.allRows.length + '</b> scored candidates' +
          (prepared.pinnedOutsideQueryCount ? ', plus <b>' + prepared.pinnedOutsideQueryCount + '</b> pinned outside the search.' : '.')
        : 'Showing top <b>' + Math.min(25, prepared.matchedCount) + '</b> of <b>' + prepared.allRows.length + '</b> scored candidates.') +
      '</div>' +
      '<table><thead><tr><th class="num">#</th><th>Actions</th><th>Waveform</th><th>Track</th><th class="num">Final</th><th class="num">Mix</th><th class="num">Loss</th><th class="num">Style</th><th class="num">Rhythm</th><th class="num">Harmony</th><th class="num">Energy fit</th></tr></thead><tbody>';
    rows.forEach((r, i) => {
      const isPinned = pinnedRecommendationIdxs.has(Number(r.idx));
      const rowClass = [isPinned ? 'pinned-row' : '', Number(r.idx) === selectedIdx ? 'selected-slot' : ''].filter(Boolean).join(' ');
      html += '<tr' + (rowClass ? ' class="' + rowClass + '"' : '') + '><td class="num">' + (isPinned ? '★ ' : '') + (r.globalRank || (i + 1)) + '</td>' +
        '<td><div class="table-actions">' +
        actionSymbolButton('data-pin-rec-idx="' + r.idx + '"', isPinned ? '★' : '☆', isPinned ? 'Unpin candidate' : 'Pin candidate', isPinned ? 'pin-active' : 'pin-button') +
        actionSymbolButton('data-append-idx="' + r.idx + '"', actionSymbol, actionLabel + ' to sequence', 'primary') +
        actionSymbolButton('data-library-outgoing="' + r.idx + '"', '1', 'Set as Track 1', 'track-one') +
        actionSymbolButton('data-library-incoming="' + r.idx + '"', '2', 'Set as Track 2', 'track-two') +
        '<button class="play-button" data-play-idx="' + r.idx + '" data-play-mode="recommendation" aria-label="Play and highlight">▶</button>' +
        '</div></td>' +
        '<td class="waveform-cell">' + rowWaveformScrubberHtml(r, { showTime: false }) + '</td>' +
        '<td>' + trackSummaryHtml(r, { size: 'compact', showArt: true, slot: r.slot ?? slot }) + '</td>' +
        recommendationMetricCell(r.finalScore, 'final', r.finalScore, 2, 'Final score') +
        recommendationMetricCell(r.baseline, 'mix', r.baseline, 2, 'Weighted mix score') +
        recommendationMetricCell(r.penalty, 'penalty', r.penalty, 2, 'Energy score loss') +
        recommendationMetricCell(r.styleScore, 'style', r.styleScore, 2, 'Style score') +
        recommendationMetricCell(r.rhythmScore, 'rhythm', r.rhythmScore, 2, 'Rhythm score') +
        recommendationMetricCell(r.harmonyScore, 'harmonic', r.harmonyScore, 2, 'Harmony score') +
        recommendationMetricCell(r.hasEnergyTarget ? r.energyScore : NaN, 'energy', r.energyScore, 2, r.hasEnergyTarget ? 'Energy fit score' : 'No energy target for this slot') +
        '</tr>';
    });
    if (!rows.length) {
      html += '<tr><td colspan="11" class="muted" style="padding:10px;">No recommendations match the current filters.</td></tr>';
    }
    html += '</tbody></table>';
    els.recommendationPanel.innerHTML = html;
    if (restoreQueryFocus) {
      const queryInput = els.recommendationPanel.querySelector('[data-rec-filter="query"]');
      if (queryInput) {
        queryInput.focus();
        const n = String(queryInput.value || '').length;
        try { queryInput.setSelectionRange(n, n); } catch (err) {}
      }
    }
    updatePlayButtons();
    updateRowScrubbers();
  }
  function renderEnergyCurve() {
    const targets = targetCurve();
    const guide = targetGuide();
    const x = Array.from({ length: sequence.length }, (_, i) => i + 1);
    const actual = sequence.map(idx => idx === null ? null : energyOf(byIdx.get(idx)));
    const theme = mapTheme();
    const traces = [
      { x, y: guide, type: 'scatter', mode: 'lines', name: 'Energy guide', connectgaps: false, line: { color: theme.muted, width: 2, dash: 'dash' }, hovertemplate: 'Guide %{y:.1f}<extra></extra>' },
      { x, y: targets, type: 'scatter', mode: 'markers', name: 'Explicit target', marker: { size: 10, color: theme.ink, line: { color: theme.panel, width: 1 } }, hovertemplate: 'Target %{y:.1f}<extra></extra>' },
      { x, y: actual, type: 'scatter', mode: 'lines+markers', name: 'Selected energy', line: { color: '#14b8a6', width: 2 }, marker: { size: 9 }, hovertemplate: 'Track %{y:.1f}<extra></extra>' }
    ];
    const slot = targetSlot();
    if (slot >= 0 && Number.isFinite(targets[slot])) traces.push({ x: [slot + 1], y: [targets[slot]], type: 'scatter', mode: 'markers', name: 'Active target', marker: { size: 14, color: '#f59e0b', symbol: 'x' }, hoverinfo: 'skip' });
    Plotly.react('energy-curve', traces, {
      title: { text: 'Drag to set targets · dashed guide is not scored until applied', font: { size: 13, color: theme.ink } },
      margin: { t: 48, r: 20, b: 46, l: 48 },
      template: figureLightMode() ? 'plotly_white' : 'plotly_dark',
      paper_bgcolor: theme.panel,
      plot_bgcolor: theme.panel,
      font: { color: theme.ink },
      dragmode: false,
      xaxis: { title: 'Sequence slot', dtick: 1, range: [0.5, sequence.length + 0.5], fixedrange: true, gridcolor: theme.grid, zerolinecolor: theme.grid },
      yaxis: { title: 'Energy', range: [0.5, 9.5], fixedrange: true, tickmode: 'array', tickvals: ENERGY_AXIS_TICKS, ticktext: ENERGY_AXIS_TICKS.map(String), gridcolor: theme.grid, zerolinecolor: theme.grid },
      legend: { orientation: 'h', x: 0, y: -0.24, xanchor: 'left', yanchor: 'top' }
    }, { displayModeBar: false, scrollZoom: false, doubleClick: false, responsive: true });
  }
  function renderTransitionBadges() {
    if (!els.transitionBadges) return;
    const from = transitionFromIdx === null ? null : byIdx.get(transitionFromIdx);
    const to = transitionToIdx === null ? null : byIdx.get(transitionToIdx);
    let scoreHtml = '<div class="transition-mini-score"><span class="muted">Assign Track 1 and Track 2 to preview the transition score.</span></div>';
    if (from && to) {
      const slot = Math.max(0, targetSlot());
      const score = scoreTransition(transitionFromIdx, transitionToIdx, slot);
      const rankInfo = transitionRecommendationRank(transitionFromIdx, transitionToIdx, slot);
      const meterRows = [
        ['Final', score.finalScore, 'final', score.finalScore, 'Final score'],
        ['Mix', score.baseline, 'mix', score.baseline, 'Weighted mix score'],
        ['Rank', rankInfo, 'rank', NaN, 'Recommendation rank'],
        ['Energy Fit', score.energyScore, 'energy', score.energyScore, 'Energy fit score'],
        ['Style', score.styleScore, 'style', score.styleScore, 'Style score'],
        ['Rhythm', score.rhythmScore, 'rhythm', score.rhythmScore, 'Rhythm score'],
        ['Harmony', score.harmonyScore, 'harmonic', score.harmonyScore, 'Harmony score'],
        ['Loss', score.penalty, 'penalty', score.penalty, 'Energy score loss'],
      ];
      scoreHtml =
        '<div class="transition-mini-score">' +
        '<div class="transition-mini-meter-list">' +
        meterRows.map(row =>
          '<div class="transition-mini-meter-row"><span>' + esc(row[0]) + '</span>' +
          (row[2] === 'rank'
            ? rankMeterHtml(row[1], { forceValues: true })
            : scoreMeterHtml(row[1], row[2], row[3], 2, row[4], { forceValues: true })) +
          '</div>'
        ).join('') +
        '</div>' +
        '</div>';
    }
    els.transitionBadges.innerHTML =
      '<div class="transition-badge out"><b>Track 1</b>' + shortTrack(from) + '</div>' +
      '<div class="transition-badge in"><b>Track 2</b>' + shortTrack(to) + '</div>' +
      scoreHtml;
  }
  function renderTransitionEditorHtml(from, to) {
    if (!from || !to) {
      return '<div class="transition-editor"><div class="transition-editor-empty">Assign Track 1 and Track 2 to align cue points and preview windows.</div></div>';
    }
    const pitchValues = [-3, -2, -1, 0, 1, 2, 3];
    const compactMetaHtml = deck => {
      const info = transitionWindowInfo(deck);
      if (!info) return '';
      const direction = deck === 'from' ? 'Outgoing' : 'Incoming';
      const cfg = info.cfg;
      const pitchField = deck === 'from' ? 'from_pitch_shift' : 'to_pitch_shift';
      const pitchShift = Number(previewFormState[pitchField]) || 0;
      const locked = deck === 'from' && transitionReferenceLocked;
      return '<div class="transition-track-meta transition-track-meta-compact">' +
        '<div class="transition-track-label">' + esc(info.cfg.label) + ' · ' + direction + '</div>' +
        '<div class="transition-track-artwork">' + artworkHtml(info.record) + '</div>' +
        '<div class="transition-track-title">' + esc(info.record.title || 'Untitled') + '</div>' +
        '<div class="transition-track-artist muted">' + esc(info.record.artists || '') + '</div>' +
        '<div class="transition-track-facts">' + transitionDeckMetaHtml(info.record, pitchShift) + '</div>' +
        '<div class="transition-track-facts">' + loudnessMetaHtml(info.record, 'transition') + '</div>' +
        '<div class="transition-meta-controls' + (locked ? ' locked' : '') + '">' +
        selectInput(pitchField, 'Repitch', pitchValues, previewFormState[pitchField], v => (Number(v) > 0 ? '+' : '') + v + ' st') +
        '</div>' +
        '</div>';
    };
    const beatJumpLabel = value => value < 1 ? (value === .5 ? '½' : value === .25 ? '¼' : '⅛') : String(value);
    const beatJumpHtml = deck => {
      const selected = transitionBeatJumpSizes[deck] || 4;
      const index = Math.max(0, BEAT_JUMP_SIZES.indexOf(selected));
      const start = clamp(index - 1, 0, Math.max(0, BEAT_JUMP_SIZES.length - 4));
      const visible = BEAT_JUMP_SIZES.slice(start, start + 4);
      return '<div class="transition-beat-jump" aria-label="Beat jump controls">' +
        '<div class="transition-beat-size-strip"><button type="button" data-transition-beat-size-step="' + deck + ':-1" aria-label="Smaller beat jump">‹</button>' +
        visible.map(value => '<button type="button" class="' + (value === selected ? 'active' : '') + '" data-transition-beat-size="' + deck + ':' + value + '" aria-pressed="' + (value === selected ? 'true' : 'false') + '">' + beatJumpLabel(value) + '</button>').join('') +
        '<button type="button" data-transition-beat-size-step="' + deck + ':1" aria-label="Larger beat jump">›</button></div>' +
        '<div class="transition-beat-jump-actions"><button type="button" data-transition-beat-jump="' + deck + ':-1" aria-label="Jump backward ' + beatJumpLabel(selected) + ' beats">‹</button><button type="button" data-transition-beat-jump="' + deck + ':1" aria-label="Jump forward ' + beatJumpLabel(selected) + ' beats">›</button></div>' +
        '</div>';
    };
    const laneToolsHtml = deck => {
      const info = transitionWindowInfo(deck);
      if (!info) return '';
      const cfg = info.cfg;
      const bounds = transitionNudgeBounds(info);
      const locked = deck === 'from' && transitionReferenceLocked;
      const disabled = locked ? ' disabled' : '';
      const nudgeLabel = locked ? 'Reference locked' : ('<span data-transition-nudge-label="' + esc(deck) + '">' + fmt(info.nudgeBeats, 3) + ' beats</span>');
      return '<div class="transition-lane-tools' + (locked ? ' locked' : '') + '" data-transition-tools="' + esc(deck) + '">' +
        '<div class="transition-adjustment-readout">' + nudgeLabel + ' · <span data-transition-beatgrid-label="' + esc(deck) + '">' + fmt(info.beatgridMs, 0) + ' ms</span></div>' +
        cueSelectHtml(info.record, cfg.role, cfg.cueField, 'Cue section') +
        beatJumpHtml(deck) +
        '<details class="transition-fine-tune"><summary>Fine tune <span>ⓘ</span></summary><div class="transition-fine-controls">' +
        numericInput(cfg.nudgeField, 'Nudge beats', previewFormState[cfg.nudgeField], bounds.min, bounds.max, .125) +
        numericInput(cfg.beatgridField, 'Adjust Beatgrid', previewFormState[cfg.beatgridField], -250, 250, 5) +
        '<div class="transition-beatgrid-nudge"><button type="button" data-transition-beatgrid-adjust="' + esc(deck) + ':-5"' + disabled + '>−5 ms</button><button type="button" data-transition-beatgrid-adjust="' + esc(deck) + ':5"' + disabled + '>+5 ms</button><button type="button" class="transition-align-reset" data-transition-reset-align="' + esc(deck) + '"' + disabled + '>Reset</button></div></div></details>' +
        '</div>';
    };
    const overviewHtml = deck => {
      const info = transitionWindowInfo(deck);
      const duration = Math.max(1, audioDuration(info && info.record));
      const initial = info && info.record && rowScrubPositions.has(Number(info.record.idx))
        ? rowScrubValue(info.record)
        : Math.max(0, Number(info && info.transitionStart) || 0);
      return '<div class="transition-overview-strip ' + esc(deck) + '">' +
        '<button class="transition-track-play" type="button" data-transition-track-play="' + esc(deck) + '" aria-label="Play ' + esc(info && info.cfg ? info.cfg.label : 'track') + '">▶</button>' +
        '<div class="transition-overview-track">' +
        '<canvas class="transition-overview-canvas" data-transition-surface="overview" data-transition-deck="' + esc(deck) + '" height="58"></canvas>' +
        '<input class="transition-track-scrub scrub-range" data-transition-track-scrub="' + esc(deck) + '" type="range" min="0" max="' + esc(duration) + '" step="0.01" value="' + esc(initial) + '" aria-label="' + esc(info && info.cfg ? info.cfg.label : 'Track') + ' playback position">' +
        '</div>' +
        '</div>';
    };
    const seamPlayHtml = () => '<div class="transition-seam-play">' +
      '<button id="transition-play-toggle" class="play-button" type="button" aria-label="Play"' + (transportCanPlay() ? '' : ' disabled') + '>▶</button>' +
      '</div>';
    const seamActionsHtml = () => '<div class="transition-seam-actions">' +
      '<span id="transition-scrub-time" class="scrub-time">0:00 / ' + esc(timeText(Number(transitionRender && transitionRender.transition && transitionRender.transition.duration_seconds) || 0)) + '</span>' +
      '<button id="transition-loop-toggle" class="transition-seam-action transition-loop-toggle' + (transitionLoopEnabled ? ' active' : '') + '" type="button" aria-pressed="' + (transitionLoopEnabled ? 'true' : 'false') + '" title="' + (transitionLoopEnabled ? 'Disable transition loop' : 'Enable transition loop') + '" aria-label="' + (transitionLoopEnabled ? 'Disable transition loop' : 'Enable transition loop') + '">↻</button>' +
      (config.app_mode
        ? '<button id="transition-render-button" class="transition-seam-action transition-render-action" type="button" data-preview-action="render" title="Render transition" aria-label="Render transition"' + (transitionRenderPending ? ' disabled' : '') + '>↻</button>'
        : '') +
      '<button class="transition-seam-action" type="button" data-preview-action="swap" title="Swap Track 1 and Track 2" aria-label="Swap Track 1 and Track 2">⇅</button>' +
      '</div>';
    const viewportHtml = () => '<div class="transition-viewport" aria-label="Transition view range"><div class="transition-viewport-controls"><div class="transition-viewport-track" data-transition-viewport-track><i data-transition-viewport-selection></i><button type="button" data-transition-viewport-handle="start" aria-label="Move start of view"></button><button type="button" data-transition-viewport-handle="end" aria-label="Move end of view"></button></div></div></div>';
    return '<div class="transition-editor">' +
      '<div class="transition-editor-head">' +
      '<h2>Transition Alignment</h2>' +
      '<div class="transition-editor-tools">' +
      '<label><input id="transition-snap-to-beat" type="checkbox"' + (transitionSnapToBeat ? ' checked' : '') + '> Snap to beat</label>' +
      '<label><input id="transition-lock-track-one" type="checkbox"' + (transitionReferenceLocked ? ' checked' : '') + '> Lock Track 1 reference</label>' +
      '<span id="transition-editor-dirty" class="transition-editor-hint"></span>' +
      '</div>' +
      '</div>' +
      '<div class="transition-editor-help"><button type="button" class="transition-help-button" aria-label="Transition alignment help">ⓘ<span role="tooltip">Drag either waveform to move its section; click a cue marker to select it. The connected beat and bar lines show alignment. Beatgrid offsets are local preview adjustments.</span></button></div>' +
      '<div class="transition-connected-stage">' +
      '<aside class="transition-from-meta">' + compactMetaHtml('from') + '</aside>' +
      '<section class="transition-from-section transition-track-section">' +
      overviewHtml('from') +
      '<canvas class="transition-connected-canvas transition-connected-from" data-transition-surface="connected" data-transition-deck="from" height="176"></canvas>' +
      '</section>' +
      '<aside class="transition-from-tools">' + laneToolsHtml('from') + '</aside>' +
      seamPlayHtml() +
      '<div class="transition-seam">' +
      '<canvas class="transition-seam-grid" data-transition-surface="seam-grid" height="26"></canvas>' +
      '<div class="transition-seam-transport">' + transitionTransportHtml() + '</div>' +
      '</div>' +
      seamActionsHtml() +
      '<aside class="transition-to-tools">' + laneToolsHtml('to') + '</aside>' +
      '<section class="transition-to-section transition-track-section">' +
      '<canvas class="transition-connected-canvas transition-connected-to" data-transition-surface="connected" data-transition-deck="to" height="176"></canvas>' +
      overviewHtml('to') +
      '</section>' +
      '<aside class="transition-to-meta">' + compactMetaHtml('to') + '</aside>' +
      '<div class="transition-viewport-left"><button type="button" data-transition-viewport-zoom="out" title="Zoom out" aria-label="Zoom out">−</button></div>' +
      viewportHtml() +
      '<div class="transition-viewport-actions"><button type="button" data-transition-viewport-zoom="in" title="Zoom in" aria-label="Zoom in">+</button><button type="button" data-transition-viewport-reset>Fit</button></div>' +
      '</div>' +
      liveMixerHtml() +
      '</div>';
  }
  function cueMarkerColor(role) {
    const r = String(role || '').toLowerCase();
    if (r === 'in') return '#10b981';
    if (r === 'out') return '#f59e0b';
    if (r === 'drop') return '#ec4899';
    if (r === 'break') return '#38bdf8';
    if (r === 'start') return '#94a3b8';
    return '#cbd5e1';
  }
  function peakAtTime(record, timeSec) {
    const duration = Math.max(1, Number(record.duration_seconds) || audioDuration(record) || 1);
    if (Number(timeSec) < 0 || Number(timeSec) > duration) return 0;
    const peaks = waveformSeries(waveformPayload(record), 'detail');
    if (!peaks.length) return 0.12;
    const pos = clamp((Number(timeSec) / duration) * (peaks.length - 1), 0, peaks.length - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(peaks.length - 1, lo + 1);
    const frac = pos - lo;
    const a = Number(peaks[lo]) || 0.05;
    const b = Number(peaks[hi]) || a;
    return clamp(a + ((b - a) * frac), 0.035, 1);
  }
  function profileValueAtTime(record, profile, timeSec) {
    const values = Array.isArray(profile) ? profile : [];
    if (!values.length) return peakAtTime(record, timeSec);
    const duration = Math.max(1, Number(record.duration_seconds) || audioDuration(record) || 1);
    if (Number(timeSec) < 0 || Number(timeSec) > duration) return 0;
    const pos = clamp((Number(timeSec) / duration) * (values.length - 1), 0, values.length - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(values.length - 1, lo + 1);
    const frac = pos - lo;
    const a = Number(values[lo]) || 0.05;
    const b = Number(values[hi]) || a;
    return clamp(a + ((b - a) * frac), 0.035, 1);
  }
  function signedProfileValueAtTime(record, values, timeSec) {
    const profile = Array.isArray(values) ? values : [];
    if (!profile.length) return 0;
    const duration = Math.max(1, Number(record.duration_seconds) || audioDuration(record) || 1);
    if (Number(timeSec) < 0 || Number(timeSec) > duration) return 0;
    const pos = clamp((Number(timeSec) / duration) * (profile.length - 1), 0, profile.length - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(profile.length - 1, lo + 1);
    const frac = pos - lo;
    const a = Number(profile[lo]) || 0;
    const b = Number(profile[hi]) || a;
    return clamp(a + ((b - a) * frac), -1, 1);
  }
  function transitionLaneRect(width, height, surface) {
    if (surface === 'overview') return { x: 0, y: 7, w: Math.max(1, width), h: Math.max(1, height - 14) };
    if (surface === 'connected') return { x: 0, y: 0, w: Math.max(1, width), h: Math.max(1, height) };
    return { x: 10, y: 10, w: Math.max(1, width - 20), h: Math.max(1, height - 20) };
  }
  function transitionVisibleRange(info, surface) {
    const trackDuration = Math.max(info.beatSec, info.trackDuration || info.duration);
    if (surface === 'overview') return { start: 0, end: trackDuration, duration: trackDuration };
    const start = info.contextStart;
    const end = info.contextEnd;
    return { start, end, duration: Math.max(info.beatSec, end - start) };
  }
  function transitionViewportDefaults(reference) {
    const totalBeats = Math.max(8, reference.totalBars * reference.beatsPerBar);
    return { start: -reference.frontBars * reference.beatsPerBar, end: totalBeats - reference.frontBars * reference.beatsPerBar };
  }
  function transitionViewportState(reference) {
    const defaults = transitionViewportDefaults(reference);
    const minSpan = 8;
    const start = Number(transitionViewport && transitionViewport.start);
    const end = Number(transitionViewport && transitionViewport.end);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end - start < minSpan) {
      transitionViewport = Object.assign({}, defaults);
    }
    transitionViewport.start = clamp(Number(transitionViewport.start), defaults.start, defaults.end - minSpan);
    transitionViewport.end = clamp(Number(transitionViewport.end), transitionViewport.start + minSpan, defaults.end);
    return transitionViewport;
  }
  function transitionSharedView(info, reference) {
    const viewport = transitionViewportState(reference);
    const start = info.transitionStart + viewport.start * info.beatSec;
    const end = info.transitionStart + viewport.end * info.beatSec;
    return { start, end, duration: Math.max(info.beatSec, end - start) };
  }
  function xForTransitionTime(time, view, lane) {
    return lane.x + (clamp((Number(time) - view.start) / Math.max(1e-9, view.duration), 0, 1) * lane.w);
  }
  function transitionWaveSamples(record, view, count) {
    const n = Math.max(8, Math.round(count || 160));
    const samples = [];
    for (let i = 0; i < n; i += 1) {
      const t = view.start + ((i + 0.5) / n) * view.duration;
      let peak = peakAtTime(record, t);
      if (!Number.isFinite(peak)) peak = 0.08 + 0.05 * Math.sin(i * 0.43);
      samples.push(peak <= 0 ? 0 : clamp(peak, 0.035, 1));
    }
    return samples;
  }
  function transitionProfileSamples(record, profile, view, count) {
    const n = Math.max(8, Math.round(count || 160));
    const values = normalizeWaveformValues(profile);
    const samples = [];
    for (let i = 0; i < n; i += 1) {
      const t = view.start + ((i + 0.5) / n) * view.duration;
      samples.push(profileValueAtTime(record, values, t));
    }
    return samples;
  }
  function transitionBandSamples(record, view, count) {
    const payload = waveformPayload(record);
    if (!payload || !payload.bands) return null;
    const low = payload.bands.low && payload.bands.low.length ? transitionProfileSamples(record, payload.bands.low, view, count) : null;
    const mid = payload.bands.mid && payload.bands.mid.length ? transitionProfileSamples(record, payload.bands.mid, view, count) : null;
    const high = payload.bands.high && payload.bands.high.length ? transitionProfileSamples(record, payload.bands.high, view, count) : null;
    return (low || mid || high) ? { low, mid, high } : null;
  }
  function transitionEnvelopeSamples(record, view, count) {
    const payload = waveformPayload(record);
    const envelope = waveformEnvelopeSeries(payload, 'detail');
    if (!envelope || !envelope.min || !envelope.max || !envelope.min.length || !envelope.max.length) return null;
    const minValues = normalizeSignedWaveformValues(envelope.min);
    const maxValues = normalizeSignedWaveformValues(envelope.max);
    const n = Math.max(8, Math.round(count || 160));
    const low = [];
    const high = [];
    for (let i = 0; i < n; i += 1) {
      const t = view.start + ((i + 0.5) / n) * view.duration;
      low.push(signedProfileValueAtTime(record, minValues, t));
      high.push(signedProfileValueAtTime(record, maxValues, t));
    }
    return { min: low, max: high };
  }
  function smoothTransitionSample(samples, i, pow=1) {
    const a = samples[Math.max(0, i - 2)] || 0;
    const b = samples[Math.max(0, i - 1)] || 0;
    const c = samples[i] || 0;
    const d = samples[Math.min(samples.length - 1, i + 1)] || 0;
    const e = samples[Math.min(samples.length - 1, i + 2)] || 0;
    return Math.pow(clamp((a + 2 * b + 3 * c + 2 * d + e) / 9, 0, 1), pow);
  }
  function drawTransitionEnvelope(ctx, samples, lane, color, scale, pow=1) {
    if (!samples.length) return;
    const mid = lane.y + lane.h * 0.5;
    const amp = lane.h * 0.46 * scale;
    const denom = Math.max(1, samples.length - 1);
    ctx.save();
    ctx.fillStyle = color;
    ctx.beginPath();
    samples.forEach((_, i) => {
      const x = lane.x + (i / denom) * lane.w;
      const y = mid - smoothTransitionSample(samples, i, pow) * amp;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    for (let i = samples.length - 1; i >= 0; i -= 1) {
      const x = lane.x + (i / denom) * lane.w;
      const y = mid + smoothTransitionSample(samples, i, pow) * amp;
      ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }
  function drawTransitionSignedEnvelope(ctx, envelope, lane, color, strokeColor=null) {
    if (!envelope || !envelope.min || !envelope.max || !envelope.min.length || !envelope.max.length) return;
    const n = Math.min(envelope.min.length, envelope.max.length);
    if (n < 2) return;
    const mid = lane.y + lane.h * 0.5;
    const amp = lane.h * 0.48;
    const denom = Math.max(1, n - 1);
    ctx.save();
    ctx.fillStyle = color;
    ctx.beginPath();
    for (let i = 0; i < n; i += 1) {
      const x = lane.x + (i / denom) * lane.w;
      const y = mid - (clamp(Number(envelope.max[i]) || 0, -1, 1) * amp);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    for (let i = n - 1; i >= 0; i -= 1) {
      const x = lane.x + (i / denom) * lane.w;
      const y = mid - (clamp(Number(envelope.min[i]) || 0, -1, 1) * amp);
      ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
    if (strokeColor) {
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    ctx.restore();
  }
  function automationCurveProgress(t, curve) {
    if (curve === 'smooth') return t * t * (3 - 2 * t);
    if (curve === 'exponential') return (Math.exp(4 * t) - 1) / (Math.exp(4) - 1);
    return t;
  }
  function automationValueAt(points, beat, laneName='volume') {
    const list = Array.isArray(points) ? points : [];
    if (!list.length) return 0;
    let index = -1;
    for (let i = 0; i < list.length; i += 1) {
      if (Number(list[i].beat) <= beat + .000001) index = i;
      else break;
    }
    if (index < 0) return Number(list[0].value) || 0;
    if (index >= list.length - 1) return Number(list[list.length - 1].value) || 0;
    const a = list[index], b = list[index + 1];
    const span = Number(b.beat) - Number(a.beat);
    if (span <= .000001) return Number(b.value) || 0;
    const t = clamp((beat - Number(a.beat)) / span, 0, 1);
    const progress = automationCurveProgress(t, a.curve);
    if (usesDawAutomationScale(laneName)) {
      const start = automationScalePosition(laneName, Number(a.value));
      const end = automationScalePosition(laneName, Number(b.value));
      return automationValueFromScalePosition(laneName, start + ((end - start) * progress));
    }
    return Number(a.value) + ((Number(b.value) - Number(a.value)) * progress);
  }
  function automationValueLabel(laneName, value) {
    const n = Number(value) || 0;
    if (laneName === 'filter') {
      return filterCutoffLabel(n);
    }
    if (laneName === 'delay' || laneName === 'reverb') return Math.round(clamp(n, 0, 1) * 100) + '%';
    if (n <= -95) return laneName === 'volume' ? 'MUTE' : 'KILL';
    return n.toFixed(1) + ' dB';
  }
  function formatFilterFrequency(hz) {
    const value = Math.max(1, Number(hz) || 0);
    return value >= 1000 ? (value / 1000).toFixed(value >= 10000 ? 0 : 1) + ' kHz' : Math.round(value) + ' Hz';
  }
  function filterCutoffForAmount(value) {
    const amount = clamp(Number(value) || 0, -1, 1);
    const wet = Math.abs(amount);
    if (wet < .001) return null;
    if (amount < 0) return 20000 * Math.pow(20 / 20000, wet);
    return 20 * Math.pow(20000 / 20, wet);
  }
  function filterCutoffLabel(value) {
    const amount = clamp(Number(value) || 0, -1, 1);
    const cutoff = filterCutoffForAmount(amount);
    if (cutoff === null) return 'Bypass';
    return (amount < 0 ? 'LPF ' : 'HPF ') + formatFilterFrequency(cutoff) + (Math.abs(amount) >= .995 ? ' · KILL' : '');
  }
  function automationValueUnit(laneName) {
    return laneName === 'filter' ? 'filter' : (laneName === 'delay' || laneName === 'reverb' ? 'depth' : 'dB');
  }
  function selectedAutomationSegmentParts() {
    const selected = transitionAutomationSelectedSegment;
    if (!selected) return null;
    const points = automationLane(selected.deck, selected.lane);
    if (!points || selected.index < 0 || selected.index >= points.length - 1) return null;
    return { ...selected, points, start: points[selected.index], end: points[selected.index + 1] };
  }
  function setAutomationPointSelection(deck, lane, index) {
    const points = automationLane(deck, lane);
    if (!points || index < 0 || index >= points.length) return;
    transitionAutomationSelectedPoint = { deck, lane, index };
    const segmentIndex = index > 0 && sameAutomationBeat(points[index - 1].beat, points[index].beat)
      ? index - 1
      : (index < points.length - 1 ? index : points.length - 2);
    transitionAutomationSelectedSegment = points.length < 2 ? null : { deck, lane, index: segmentIndex };
  }
  function setAutomationSegmentSelection(deck, lane, index) {
    const points = automationLane(deck, lane);
    if (!points || index < 0 || index >= points.length - 1) return;
    transitionAutomationSelectedPoint = null;
    transitionAutomationSelectedSegment = { deck, lane, index };
  }
  function syncAutomationEditorControls() {
    const active = transitionAutomationActiveLane;
    document.querySelectorAll('[data-automation-lane]').forEach(button => {
      button.classList.toggle('active', button.getAttribute('data-automation-lane') === active);
    });
    const segment = selectedAutomationSegmentParts();
    const isJump = !!(segment && Math.abs(Number(segment.start.beat) - Number(segment.end.beat)) < .000001);
    const curveSelect = document.querySelector('[data-automation-curve]');
    if (curveSelect) {
      curveSelect.disabled = !segment || isJump;
      curveSelect.value = segment ? (segment.start.curve || 'linear') : 'linear';
    }
    const unit = segment ? automationValueUnit(segment.lane) : 'value';
    document.querySelectorAll('[data-automation-segment-label]').forEach(label => {
      const [endpoint, field] = String(label.getAttribute('data-automation-segment-label')).split(':');
      const valueLabel = segment && segment.lane === 'filter' ? 'position' : unit;
      label.textContent = field === 'beat' && isJump ? 'Jump beat' : (endpoint === 'start' ? 'Start ' : 'End ') + (field === 'beat' ? 'beat' : valueLabel);
    });
    document.querySelectorAll('[data-automation-segment-field]').forEach(input => {
      const [endpoint, field] = String(input.getAttribute('data-automation-segment-field')).split(':');
      const point = segment && (endpoint === 'start' ? segment.start : segment.end);
      const pointIndex = segment ? segment.index + (endpoint === 'end' ? 1 : 0) : -1;
      const fixedBeat = field === 'beat' && segment && (pointIndex === 0 || pointIndex === segment.points.length - 1);
      const hiddenDuplicateJumpBeat = isJump && field === 'beat' && endpoint === 'end';
      const wrapper = input.closest('label');
      if (wrapper) wrapper.hidden = hiddenDuplicateJumpBeat;
      input.disabled = !point || fixedBeat || hiddenDuplicateJumpBeat;
      input.value = point ? String(field === 'beat' ? Number(point.beat).toFixed(2) : Number(point.value).toFixed(segment.lane === 'filter' || segment.lane === 'delay' || segment.lane === 'reverb' ? 2 : 1)) : '';
      input.min = field === 'beat' ? '0' : String(automationLaneLimits(segment ? segment.lane : 'volume')[0]);
      input.max = field === 'beat' ? String(transitionAutomationBeats()) : String(automationLaneLimits(segment ? segment.lane : 'volume')[1]);
      input.step = field === 'beat' ? '.01' : (segment && (segment.lane === 'filter' || segment.lane === 'delay' || segment.lane === 'reverb') ? '.01' : '.1');
    });
    document.querySelectorAll('[data-automation-filter-readout]').forEach(readout => {
      const endpoint = readout.getAttribute('data-automation-filter-readout');
      const point = segment && (endpoint === 'start' ? segment.start : segment.end);
      readout.textContent = segment && segment.lane === 'filter' && point ? filterCutoffLabel(point.value) : '';
      readout.hidden = !(segment && segment.lane === 'filter' && point);
    });
    const deleteButton = document.querySelector('[data-automation-action="delete-segment-keyframe"]');
    if (deleteButton) deleteButton.disabled = !segment || segment.index + 1 >= segment.points.length - 1;
  }
  function automationValueY(laneName, value, lane) {
    return lane.y + ((1 - automationScalePosition(laneName, value)) * lane.h);
  }
  function automationPointXY(info, view, lane, laneName, point) {
    const time = info.transitionStart + Number(point.beat) * info.beatSec;
    return { x: xForTransitionTime(time, view, lane), y: automationValueY(laneName, point.value, lane) };
  }
  function automationLaneColor(laneName) {
    if (laneName === 'volume') return '#48c8f2';
    if (laneName === 'filter') return '#e58cff';
    if (laneName === 'delay') return '#60a5fa';
    if (laneName === 'reverb') return '#a78bfa';
    if (laneName === 'eq_low') return '#ffca3a';
    if (laneName === 'eq_mid') return '#f59e0b';
    return '#fb7185';
  }
  function automationLaneButtonHtml(deck, lane) {
    const label = lane === 'eq_low' ? 'Low EQ'
      : lane === 'eq_mid' ? 'Mid EQ'
      : lane === 'eq_high' ? 'High EQ'
      : lane[0].toUpperCase() + lane.slice(1);
    const icon = lane === 'volume' ? 'VOL'
      : lane === 'filter' ? 'FLT'
      : lane === 'delay' ? 'DLY'
      : lane === 'reverb' ? 'REV'
      : lane.replace('eq_', '').toUpperCase();
    return '<button type="button" class="fx-item lane-' + lane + (transitionAutomationActiveLane === deck + ':' + lane ? ' active' : '') + '" style="--automation-lane-color:' + automationLaneColor(lane) + '" data-automation-lane="' + deck + ':' + lane + '"><span class="fx-icon lane-' + lane + '">' + icon + '</span><span class="fx-line lane-' + lane + '"></span>' + label + '</button>';
  }
  function automationDeckLegendHtml(deck) {
    const trackLabel = deck === 'from' ? 'Track 1' : 'Track 2';
    return '<div class="automation-deck-legend"><b>' + trackLabel + '</b><div class="automation-lane-groups"><div class="automation-lane-group">' + LIVE_CHANNEL_LANES.map(lane => automationLaneButtonHtml(deck, lane)).join('') + '</div><div class="automation-lane-group automation-fx-lane-group"><span>FX</span>' + LIVE_FX_LANES.map(lane => automationLaneButtonHtml(deck, lane)).join('') + '</div></div></div>';
  }
  function drawAutomationScaleTicks(ctx, laneName, lane) {
    if (!usesDawAutomationScale(laneName)) return;
    const labels = [[laneName === 'volume' ? 'MUTE' : 'KILL', -96], ['−30', -30], ['−13', -13], ['0', 0], ['+6', 6]];
    ctx.save();
    ctx.font = '9px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    ctx.textAlign = 'left';
    labels.forEach(([label, value]) => {
      const y = automationValueY(laneName, value, lane);
      ctx.strokeStyle = 'rgba(226,232,240,.15)';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(lane.x, y); ctx.lineTo(lane.x + lane.w, y); ctx.stroke();
      ctx.fillStyle = 'rgba(226,232,240,.62)';
      ctx.fillText(label, lane.x + 4, clamp(y - 3, lane.y + 9, lane.y + lane.h - 3));
    });
    ctx.restore();
  }
  function drawFilterScaleTicks(ctx, lane) {
    const labels = [-1, -.5, 0, .5, 1].map(value => [filterCutoffLabel(value), value]);
    ctx.save();
    ctx.font = '9px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    ctx.textAlign = 'left';
    labels.forEach(([label, value]) => {
      const y = automationValueY('filter', value, lane);
      ctx.strokeStyle = value === 0 ? 'rgba(229,140,255,.34)' : 'rgba(226,232,240,.15)';
      ctx.lineWidth = value === 0 ? 1.3 : 1;
      ctx.beginPath(); ctx.moveTo(lane.x, y); ctx.lineTo(lane.x + lane.w, y); ctx.stroke();
      ctx.fillStyle = value === 0 ? 'rgba(245,208,254,.84)' : 'rgba(226,232,240,.62)';
      ctx.fillText(label, lane.x + 4, clamp(y - 3, lane.y + 9, lane.y + lane.h - 3));
    });
    ctx.restore();
  }
  function drawTransitionAutomationCurves(ctx, info, view, lane) {
    if (!info || !view || !lane) return;
    ensureTransitionAutomation();
    const deck = info.cfg.deck;
    const lanes = ['volume', 'eq_low', 'eq_mid', 'eq_high', 'filter'];
    const transitionStart = Math.max(view.start, info.transitionStart);
    const transitionEnd = Math.min(view.end, info.transitionEnd);
    if (transitionEnd <= transitionStart) return;
    ctx.save();
    if (transitionAutomationActiveLane && transitionAutomationActiveLane.endsWith(':filter')) {
      const y = automationValueY('filter', 0, lane);
      ctx.strokeStyle = 'rgba(229,140,255,.48)';
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(lane.x, y); ctx.lineTo(lane.x + lane.w, y); ctx.stroke();
      ctx.setLineDash([]);
    }
    lanes.forEach(laneName => {
      const points = automationLane(deck, laneName);
      if (!points) return;
      const active = transitionAutomationActiveLane === deck + ':' + laneName;
      const color = automationLaneColor(laneName);
      if (active) {
        if (laneName === 'filter') drawFilterScaleTicks(ctx, lane);
        else drawAutomationScaleTicks(ctx, laneName, lane);
      }
      const samples = Math.max(32, Math.floor(lane.w / 8));
      ctx.strokeStyle = color;
      ctx.globalAlpha = active ? 1 : .30;
      ctx.lineWidth = active ? 2.8 : 1.2;
      ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      ctx.beginPath();
      let previousBeat = null;
      for (let i = 0; i <= samples; i += 1) {
        const time = transitionStart + (i / samples) * (transitionEnd - transitionStart);
        const beat = (time - info.transitionStart) / info.beatSec;
        const x = xForTransitionTime(time, view, lane);
        const y = automationValueY(laneName, automationValueAt(points, beat, laneName), lane);
        const crossesJump = previousBeat !== null && points.some((point, index) => index > 0 && sameAutomationBeat(point.beat, points[index - 1].beat) && Number(point.beat) > previousBeat && Number(point.beat) <= beat);
        if (i === 0 || crossesJump) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        previousBeat = beat;
      }
      ctx.stroke();
      points.forEach((point, index) => {
        if (index === 0 || Math.abs(Number(point.beat) - Number(points[index - 1].beat)) >= .000001) return;
        const x = automationPointXY(info, view, lane, laneName, point).x;
        const y0 = automationValueY(laneName, points[index - 1].value, lane);
        const y1 = automationValueY(laneName, point.value, lane);
        ctx.save();
        ctx.strokeStyle = color;
        ctx.globalAlpha = active ? 1 : .30;
        ctx.lineWidth = active ? 2.8 : 1.2;
        ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke();
        ctx.restore();
      });
      if (active) {
        const selectedSegment = selectedAutomationSegmentParts();
        if (selectedSegment && selectedSegment.deck === deck && selectedSegment.lane === laneName) {
          const startBeat = Number(selectedSegment.start.beat);
          const endBeat = Number(selectedSegment.end.beat);
          const segmentSamples = 24;
          ctx.save();
          ctx.strokeStyle = 'rgba(255,255,255,.92)';
          ctx.lineWidth = 4.4;
          ctx.globalAlpha = .72;
          ctx.beginPath();
          if (Math.abs(endBeat - startBeat) < .000001) {
            const x = automationPointXY(info, view, lane, laneName, selectedSegment.start).x;
            ctx.moveTo(x, automationValueY(laneName, selectedSegment.start.value, lane));
            ctx.lineTo(x, automationValueY(laneName, selectedSegment.end.value, lane));
          } else {
            for (let i = 0; i <= segmentSamples; i += 1) {
              const beat = startBeat + ((i / segmentSamples) * (endBeat - startBeat));
              const time = info.transitionStart + beat * info.beatSec;
              const x = xForTransitionTime(time, view, lane);
              const y = automationValueY(laneName, automationValueAt(points, beat, laneName), lane);
              if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
          }
          ctx.stroke();
          ctx.restore();
        }
        points.forEach((point, index) => {
          const xy = automationPointXY(info, view, lane, laneName, point);
          if (xy.x < lane.x - 8 || xy.x > lane.x + lane.w + 8) return;
          const selected = transitionAutomationSelectedPoint && transitionAutomationSelectedPoint.deck === deck && transitionAutomationSelectedPoint.lane === laneName && transitionAutomationSelectedPoint.index === index;
          ctx.fillStyle = selected ? '#fff' : color;
          ctx.strokeStyle = '#0f172a'; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.arc(xy.x, xy.y, selected ? 5.5 : 4.2, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
          if (selected) {
            const label = Number(point.beat).toFixed(2) + ' · ' + automationValueLabel(laneName, point.value);
            ctx.save();
            ctx.font = '11px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
            const width = Math.ceil(ctx.measureText(label).width) + 10;
            const lx = clamp(xy.x + 9, lane.x + 3, lane.x + lane.w - width - 3);
            const ly = clamp(xy.y - 25, lane.y + 3, lane.y + lane.h - 20);
            ctx.fillStyle = 'rgba(15,23,42,.94)';
            ctx.strokeStyle = color; ctx.lineWidth = 1;
            if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(lx, ly, width, 19, 4); ctx.fill(); ctx.stroke(); }
            else { ctx.fillRect(lx, ly, width, 19); ctx.strokeRect(lx, ly, width, 19); }
            ctx.fillStyle = '#f8fafc'; ctx.fillText(label, lx + 5, ly + 13);
            ctx.restore();
          }
        });
      }
    });
    ctx.restore();
  }
  function drawTransitionWaveform(ctx, info, view, lane, surface) {
    const sampleCount = surface === 'overview' ? Math.max(260, Math.floor(lane.w * 1.25)) : Math.max(900, Math.floor(lane.w * 2.4));
    const samples = transitionWaveSamples(info.record, view, sampleCount);
    const signedEnvelope = transitionEnvelopeSamples(info.record, view, sampleCount);
    const bands = transitionBandSamples(info.record, view, sampleCount);
    if (signedEnvelope) {
      drawTransitionSignedEnvelope(ctx, signedEnvelope, lane, 'rgba(238,242,255,.22)', 'rgba(255,255,255,.12)');
    }
    drawTransitionEnvelope(ctx, bands && bands.low ? bands.low : samples, lane, 'rgba(24,75,216,.86)', 1.0, 1.08);
    drawTransitionEnvelope(ctx, bands && bands.mid ? bands.mid : samples, lane, 'rgba(211,128,39,.78)', 0.70, 0.90);
    drawTransitionEnvelope(ctx, bands && bands.high ? bands.high : samples, lane, 'rgba(248,242,226,.90)', 0.34, 0.55);
    const mid = lane.y + lane.h * 0.5;
    const amp = lane.h * 0.34;
    const denom = Math.max(1, samples.length - 1);
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,.13)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    samples.forEach((_, i) => {
      const x = lane.x + (i / denom) * lane.w;
      const y = mid - smoothTransitionSample(samples, i, 1.1) * amp;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.restore();
  }
  function drawTransitionGrid(ctx, info, view, lane, surface) {
    const firstBeat = Math.floor((view.start - info.transitionStart) / info.beatSec) - 1;
    const lastBeat = Math.ceil((view.end - info.transitionStart) / info.beatSec) + 1;
    const beatSpan = Math.max(1, lastBeat - firstBeat);
    const step = surface === 'overview'
      ? Math.max(info.beatsPerBar * 4, Math.ceil(beatSpan / 80 / info.beatsPerBar) * info.beatsPerBar)
      : 1;
    for (let b = firstBeat; b <= lastBeat; b += step) {
      const t = info.transitionStart + b * info.beatSec;
      if (t < view.start || t > view.end) continue;
      const x = xForTransitionTime(t, view, lane);
      const isBar = b % info.beatsPerBar === 0;
      const isPhrase = b % (info.beatsPerBar * 4) === 0;
      ctx.strokeStyle = isPhrase ? '#555' : (isBar ? '#3e3e3e' : '#2d2d2d');
      ctx.lineWidth = isPhrase ? 1.25 : 1;
      ctx.beginPath();
      ctx.moveTo(x, lane.y);
      ctx.lineTo(x, lane.y + lane.h);
      ctx.stroke();
      if (surface !== 'overview' && isBar && lane.w / Math.max(1, beatSpan / info.beatsPerBar) > 24) {
        ctx.fillStyle = isPhrase ? '#9f9f9f' : '#666';
        ctx.font = '10px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
        ctx.fillText(String(Math.round(b / info.beatsPerBar)), x + 4, lane.y + 13);
      }
    }
  }
  function drawTransitionSelection(ctx, info, view, lane) {
    const selectedStart = Math.max(view.start, info.transitionStart);
    const selectedEnd = Math.min(view.end, info.transitionEnd);
    ctx.save();
    ctx.fillStyle = 'rgba(0,0,0,.36)';
    const visibleStart = Math.max(view.start, info.contextStart);
    const visibleEnd = Math.min(view.end, info.contextEnd);
    if (visibleEnd > visibleStart) {
      const x0 = xForTransitionTime(visibleStart, view, lane);
      const x1 = xForTransitionTime(Math.max(visibleStart, selectedStart), view, lane);
      const x2 = xForTransitionTime(Math.min(visibleEnd, selectedEnd), view, lane);
      const x3 = xForTransitionTime(visibleEnd, view, lane);
      if (x1 > x0) ctx.fillRect(x0, lane.y, x1 - x0, lane.h);
      if (x3 > x2) ctx.fillRect(x2, lane.y, x3 - x2, lane.h);
    }
    if (selectedEnd <= selectedStart) { ctx.restore(); return; }
    const x0 = xForTransitionTime(selectedStart, view, lane);
    const x1 = xForTransitionTime(selectedEnd, view, lane);
    ctx.fillStyle = info.cfg.deck === 'from' ? roleColors.track1Fill : roleColors.track2Fill;
    ctx.fillRect(x0, lane.y, Math.max(2, x1 - x0), lane.h);
    ctx.strokeStyle = info.cfg.color;
    ctx.lineWidth = 2;
    [x0, x1].forEach(x => {
      ctx.beginPath();
      ctx.moveTo(x, lane.y);
      ctx.lineTo(x, lane.y + lane.h);
      ctx.stroke();
    });
    ctx.restore();
  }
  function drawTransitionCueMarkers(ctx, info, view, lane, surface) {
    cuesForRecord(info.record).forEach(cue => {
      const start = Number(cue.start_seconds);
      if (!Number.isFinite(start) || start < view.start || start > view.end) return;
      const x = xForTransitionTime(start, view, lane);
      ctx.strokeStyle = cueMarkerColor(cue.role);
      ctx.lineWidth = surface === 'overview' ? 1 : 1.4;
      ctx.beginPath();
      ctx.moveTo(x, lane.y + 4);
      ctx.lineTo(x, lane.y + lane.h - 4);
      ctx.stroke();
      const label = String(cue.name || cue.role || '').slice(0, surface === 'overview' ? 9 : 16);
      if (label && (surface !== 'overview' || x > lane.x + 8 && x < lane.x + lane.w - 42)) {
        ctx.fillStyle = cueMarkerColor(cue.role);
        ctx.font = (surface === 'overview' ? '10px' : '11px') + ' ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
        ctx.fillText(label, clamp(x + 4, lane.x + 4, lane.x + lane.w - 84), surface === 'overview' ? lane.y + 12 : lane.y + lane.h - 8);
      }
    });
  }
  function transitionPlaybackProgress() {
    const duration = activeTransportDuration();
    const position = activeTransportPosition();
    if (!Number.isFinite(duration) || duration <= 0 || !Number.isFinite(position)) return null;
    return clamp(position / duration, 0, 1);
  }
  function transitionDeckPlaybackTime(info) {
    if (activeTransportMode() === 'live' && livePlayback && livePlayback.sourceContextStarts) {
      const sourceStart = Number(livePlayback.sourceContextStarts[info.cfg.deck]);
      if (Number.isFinite(sourceStart)) return sourceStart + activeTransportPosition();
    }
    const progress = transitionPlaybackProgress();
    return progress == null ? null : info.contextStart + (progress * info.duration);
  }
  function livePlaybackProgress() {
    if (!livePlayback || !liveAudioContext || !Number.isFinite(livePlayback.duration) || livePlayback.duration <= 0) return null;
    const elapsed = Math.max(0, liveAudioContext.currentTime - livePlayback.startAt);
    return ((livePlayback.startPosition + elapsed) % livePlayback.duration) / livePlayback.duration;
  }
  function drawTransitionPlaybackHead(ctx, info, view, lane, surface) {
    if (surface === 'overview' && previewAudioIdx === Number(info.record.idx) && previewAudioContext === 'transition-' + info.cfg.deck) return;
    const trackTime = transitionDeckPlaybackTime(info);
    if (trackTime == null) return;
    if (trackTime < view.start || trackTime > view.end) return;
    const x = xForTransitionTime(trackTime, view, lane);
    ctx.save();
    ctx.strokeStyle = surface === 'overview' ? 'rgba(255,255,255,.72)' : 'rgba(255,255,255,.94)';
    ctx.lineWidth = surface === 'overview' ? 1.4 : 2.1;
    ctx.shadowColor = 'rgba(103,232,249,.62)';
    ctx.shadowBlur = surface === 'overview' ? 4 : 8;
    ctx.beginPath();
    ctx.moveTo(x, lane.y + 2);
    ctx.lineTo(x, lane.y + lane.h - 2);
    ctx.stroke();
    if (surface === 'section') {
      ctx.fillStyle = 'rgba(255,255,255,.95)';
      ctx.beginPath();
      ctx.moveTo(x, lane.y + 2);
      ctx.lineTo(x - 5, lane.y + 11);
      ctx.lineTo(x + 5, lane.y + 11);
      ctx.closePath();
      ctx.fill();
    }
    ctx.restore();
  }
  function drawTransitionDeckPlaybackHead(ctx, info, view, lane, surface) {
    if (surface !== 'overview' || previewAudioIdx !== Number(info.record.idx) || previewAudioContext !== 'transition-' + info.cfg.deck || !els.audio) return;
    const trackTime = Number(els.audio.currentTime);
    if (!Number.isFinite(trackTime) || trackTime < view.start || trackTime > view.end) return;
    const x = xForTransitionTime(trackTime, view, lane);
    ctx.save();
    ctx.strokeStyle = 'rgba(103,232,249,.98)';
    ctx.lineWidth = 2;
    ctx.shadowColor = 'rgba(34,211,238,.72)';
    ctx.shadowBlur = 7;
    ctx.beginPath(); ctx.moveTo(x, lane.y + 1); ctx.lineTo(x, lane.y + lane.h - 1); ctx.stroke();
    ctx.restore();
  }
  function transitionPlaybackBounds(reference) {
    const duration = activeTransportDuration();
    if (!reference || !Number.isFinite(duration) || duration <= 0) return null;
    return { start: 0, end: duration, duration };
  }
  function transitionVisiblePlaybackProgress(reference) {
    const view = transitionSharedView(reference, reference);
    const trackTime = transitionDeckPlaybackTime(reference);
    if (!Number.isFinite(trackTime)) return null;
    return clamp((trackTime - view.start) / view.duration, 0, 1);
  }
  function drawConnectedTransitionPlaybackHead(ctx, lane, reference) {
    const progress = transitionVisiblePlaybackProgress(reference);
    if (progress == null) return;
    const x = lane.x + progress * lane.w;
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,.96)';
    ctx.lineWidth = 2.1;
    ctx.shadowColor = 'rgba(103,232,249,.65)';
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.moveTo(x, lane.y);
    ctx.lineTo(x, lane.y + lane.h);
    ctx.stroke();
    ctx.restore();
  }
  function drawConnectedTransitionSurface(canvas, info, reference) {
    const rect = canvas.getBoundingClientRect();
    const widthCss = Math.max(1, rect.width || canvas.clientWidth || 420);
    const heightCss = Math.max(1, rect.height || 160);
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(widthCss * dpr);
    canvas.height = Math.floor(heightCss * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, widthCss, heightCss);
    ctx.fillStyle = '#141414';
    ctx.fillRect(0, 0, widthCss, heightCss);
    const lane = transitionLaneRect(widthCss, heightCss, 'connected');
    const view = transitionSharedView(info, reference);
    const gridView = transitionSharedView(reference, reference);
    drawTransitionGrid(ctx, reference, gridView, lane, 'connected');
    drawTransitionWaveform(ctx, info, view, lane, 'connected');
    drawTransitionSelection(ctx, info, view, lane);
    drawTransitionAutomationCurves(ctx, info, view, lane);
    drawTransitionCueMarkers(ctx, info, view, lane, 'connected');
    drawConnectedTransitionPlaybackHead(ctx, lane, reference);
  }
  function drawTransitionSeamGrid(canvas, reference) {
    if (!canvas || !reference) return;
    const rect = canvas.getBoundingClientRect();
    const widthCss = Math.max(1, rect.width || canvas.clientWidth || 420);
    const heightCss = Math.max(1, rect.height || 24);
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(widthCss * dpr);
    canvas.height = Math.floor(heightCss * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, widthCss, heightCss);
    ctx.fillStyle = '#141414';
    ctx.fillRect(0, 0, widthCss, heightCss);
    const lane = { x: 0, y: 0, w: widthCss, h: heightCss };
    const view = transitionSharedView(reference, reference);
    const firstBeat = Math.floor((view.start - reference.transitionStart) / reference.beatSec) - 1;
    const lastBeat = Math.ceil((view.end - reference.transitionStart) / reference.beatSec) + 1;
    for (let beat = firstBeat; beat <= lastBeat; beat += 1) {
      const time = reference.transitionStart + beat * reference.beatSec;
      if (time < view.start || time > view.end) continue;
      const x = xForTransitionTime(time, view, lane);
      const isBar = beat % reference.beatsPerBar === 0;
      const isPhrase = beat % (reference.beatsPerBar * 4) === 0;
      ctx.strokeStyle = isPhrase ? '#555' : (isBar ? '#3e3e3e' : '#2d2d2d');
      ctx.lineWidth = isPhrase ? 1.25 : 1;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, heightCss);
      ctx.stroke();
    }
  }
  function drawTransitionSurface(canvas, info, surface) {
    const rect = canvas.getBoundingClientRect();
    const widthCss = Math.max(1, rect.width || canvas.clientWidth || (surface === 'overview' ? 320 : 420));
    const heightCss = Math.max(1, rect.height || (surface === 'overview' ? 54 : 148));
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(widthCss * dpr);
    canvas.height = Math.floor(heightCss * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, widthCss, heightCss);
    ctx.fillStyle = '#121212';
    ctx.fillRect(0, 0, widthCss, heightCss);
    const lane = transitionLaneRect(widthCss, heightCss, surface);
    ctx.fillStyle = '#1d1d1d';
    if (ctx.roundRect) {
      ctx.beginPath();
      ctx.roundRect(lane.x, lane.y, lane.w, lane.h, surface === 'overview' ? 5 : 7);
      ctx.fill();
    } else {
      ctx.fillRect(lane.x, lane.y, lane.w, lane.h);
    }
    const view = transitionVisibleRange(info, surface);
    drawTransitionGrid(ctx, info, view, lane, surface);
    drawTransitionWaveform(ctx, info, view, lane, surface);
    drawTransitionSelection(ctx, info, view, lane);
    if (surface === 'section') drawTransitionAutomationCurves(ctx, info, view, lane);
    drawTransitionCueMarkers(ctx, info, view, lane, surface);
    drawTransitionPlaybackHead(ctx, info, view, lane, surface);
    drawTransitionDeckPlaybackHead(ctx, info, view, lane, surface);
    if (surface === 'section') {
      ctx.fillStyle = 'rgba(242,242,242,.84)';
      ctx.font = '11px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.fillText('drag timing / click cue', lane.x + 8, lane.y + lane.h - 10);
    }
  }
  function drawTransitionOverview(canvas, info) {
    drawTransitionSurface(canvas, info, 'overview');
  }
  function drawTransitionSection(canvas, info) {
    drawTransitionSurface(canvas, info, 'section');
  }
  function transitionCueAtPoint(canvas, info, ev) {
    const rect = canvas.getBoundingClientRect();
    const surface = canvas.getAttribute('data-transition-surface') || 'section';
    const lane = transitionLaneRect(Math.max(1, rect.width), Math.max(1, rect.height), surface);
    const x = Number(ev.clientX) - rect.left;
    const y = Number(ev.clientY) - rect.top;
    if (x < lane.x || x > lane.x + lane.w || y < lane.y - 6 || y > lane.y + lane.h + 6) return null;
    const reference = transitionWindowInfo('from') || info;
    const view = surface === 'connected' ? transitionSharedView(info, reference) : transitionVisibleRange(info, surface);
    const threshold = surface === 'overview' ? 8 : 10;
    let best = null;
    let bestDx = Infinity;
    cuesForRecord(info.record).forEach(cue => {
      const start = Number(cue.start_seconds);
      if (!Number.isFinite(start) || start < view.start || start > view.end) return;
      const cueX = xForTransitionTime(start, view, lane);
      const dx = Math.abs(cueX - x);
      if (dx < bestDx && dx <= threshold) {
        best = cue;
        bestDx = dx;
      }
    });
    return best;
  }
  function drawTransitionEditor() {
    safeUi('transition editor draw', () => {
      const reference = transitionWindowInfo('from');
      const seamGrid = document.querySelector('[data-transition-surface="seam-grid"]');
      if (seamGrid && reference) drawTransitionSeamGrid(seamGrid, reference);
      ['from', 'to'].forEach(deck => {
        const info = transitionWindowInfo(deck);
        if (!info) return;
        ensureDetailedWaveform(info.record, { redrawTransition: true });
        const overview = document.querySelector('[data-transition-surface="overview"][data-transition-deck="' + deck + '"]');
        const section = document.querySelector('[data-transition-surface="section"][data-transition-deck="' + deck + '"]');
        const connected = document.querySelector('[data-transition-surface="connected"][data-transition-deck="' + deck + '"]');
        if (overview) drawTransitionOverview(overview, info);
        if (section) drawTransitionSection(section, info);
        if (connected && reference) drawConnectedTransitionSurface(connected, info, reference);
        const label = document.querySelector('[data-transition-nudge-label="' + deck + '"]');
        if (label) label.textContent = fmt(info.nudgeBeats, 3) + ' beats';
        const beatgridLabel = document.querySelector('[data-transition-beatgrid-label="' + deck + '"]');
        if (beatgridLabel) beatgridLabel.textContent = fmt(info.beatgridMs, 0) + ' ms';
      });
      if (reference) syncTransitionViewportControls(reference);
      updateTransitionTrackPlayButtons();
      updateTransitionDirtyUi();
    });
  }
  function renderTransitionPreview() {
    if (!els.transitionPreview) return;
    const playbackState = captureTransitionPlaybackState();
    const from = transitionFromIdx === null ? null : byIdx.get(transitionFromIdx);
    const to = transitionToIdx === null ? null : byIdx.get(transitionToIdx);
    if (config.app_mode && from && to) scheduleLivePreparation();
    let scoreHtml = '';
    if (from && to) {
      const slot = Math.max(0, targetSlot());
      const score = scoreTransition(transitionFromIdx, transitionToIdx, slot);
      const rankInfo = transitionRecommendationRank(transitionFromIdx, transitionToIdx, slot);
      scoreHtml =
        '<section class="transition-preview-score transition-score-compact"><div><span>Match</span><b>' + esc(fmt(score.finalScore, 0)) + '</b></div>' +
        '<button type="button" class="transition-help-button" aria-label="Transition score details">ⓘ<span role="tooltip">Energy ' + esc(fmt(score.energyScore, 0)) + ' · Style ' + esc(fmt(score.styleScore, 0)) + ' · Rhythm ' + esc(fmt(score.rhythmScore, 0)) + ' · Harmony ' + esc(fmt(score.harmonicScore, 0)) + ' · Rank ' + esc(fmt(rankInfo, 0)) + '</span></button></section>';
    }
    let controlsHtml = '';
    const topStatus = transitionPreviewMode === 'live'
      ? liveStatusText()
      : transitionRenderStatusInfo().text;
    const nowActionsHtml = config.app_mode ?
      '<div class="transition-now-actions">' +
      '<label class="transition-now-mode">Mode<select data-preview-mode><option value="verified"' + (transitionPreviewMode === 'verified' ? ' selected' : '') + '>Verified</option><option value="live"' + (transitionPreviewMode === 'live' ? ' selected' : '') + '>Live</option></select></label>' +
      '<span class="transition-now-status" data-live-status title="' + esc(topStatus) + '">' + esc(topStatus) + '</span>' +
      '<button type="button" class="transition-now-icon' + (transitionLoopEnabled ? ' active' : '') + '" data-transition-loop-top aria-pressed="' + (transitionLoopEnabled ? 'true' : 'false') + '" title="' + (transitionLoopEnabled ? 'Disable transition loop' : 'Enable transition loop') + '" aria-label="' + (transitionLoopEnabled ? 'Disable transition loop' : 'Enable transition loop') + '">↻</button>' +
      '<button type="button" class="transition-now-action" data-preview-action="render">Render</button>' +
      '<button type="button" class="transition-now-icon" data-preview-action="swap" title="Swap Track 1 and Track 2" aria-label="Swap Track 1 and Track 2">⇅</button>' +
      '</div>' : '';
    const nowPlayingHtml = from && to ?
      '<section class="transition-now-playing">' +
      '<div class="transition-now-track from">' + artworkHtml(from) + '<div><span>Track 1 · Out</span><b>' + esc(from.title || 'Untitled') + '</b><small>' + esc(from.artists || '') + '</small></div></div>' +
      '<span class="transition-now-arrow">→</span>' +
      '<div class="transition-now-track to">' + artworkHtml(to) + '<div><span>Track 2 · In</span><b>' + esc(to.title || 'Untitled') + '</b><small>' + esc(to.artists || '') + '</small></div></div>' +
      scoreHtml + nowActionsHtml + '</section>' : scoreHtml;
    if (config.app_mode) {
      if (!appLoadStarted) loadAppRuntime();
      if (appOptions && appOptions.error) {
        controlsHtml = '<div class="warn" style="margin-top:14px;">' + esc(appOptions.error) + '</div>';
      } else if (!appLoadDone) {
        controlsHtml = '<div class="muted" style="margin-top:14px;">Loading render controls...</div>';
      } else if (!from || !to) {
        controlsHtml = '<div class="' + (transitionRenderWarn ? 'warn' : 'muted') + '" style="margin-top:14px;">' + esc(transitionRenderMessage || 'Assign Track 1 and Track 2 to render a transition preview.') + '</div>';
      } else {
        const presets = (appOptions.presets || ['auto']);
        const volumeModes = appOptions.volume_modes || [];
        const eqModes = appOptions.eq_modes || [];
        const filterModes = appOptions.filter_modes || [];
        const qualityOptions = appOptions.transition_qualities || [
          { id: 'preview', label: 'Preview — faster R2, 44.1 kHz' },
          { id: 'final', label: 'Final — higher-quality R3, 44.1 kHz' },
        ];
        const qualityLabels = new Map(qualityOptions.map(option => [String(option.id), String(option.label || option.id)]));
        const renderStatus = transitionRenderStatusInfo();
        controlsHtml =
          '<div class="transition-settings-panel">' +
          '<section class="preview-render-section preview-live-card"><div class="preview-card-title"><h3>Preview</h3><button type="button" class="transition-help-button" aria-label="Preview mode help">ⓘ<span role="tooltip">Live uses prepared Web Audio decks. Verified render is the export-quality preview.</span></button></div><div class="preview-live-toolbar">' +
          selectInput('preset', 'Preset', presets, previewFormState.preset, titleCaseOption) +
          '<div class="preview-status" data-live-status>' + esc(liveStatusText()) + '</div></div>' +
          '<div class="preview-render-grid preview-timing-grid">' +
          steppedTimingInput('overlap_bars', 'Overlap', previewFormState.overlap_bars) +
          steppedTimingInput('padding_bars', 'Padding', previewFormState.padding_bars) +
          '</div><details class="preview-drawer"><summary>Render quality <span>ⓘ</span></summary><div>' + selectInput('quality', 'Quality', qualityOptions.map(option => String(option.id)), previewFormState.quality, value => qualityLabels.get(String(value)) || String(value)) + '</div></details></section>' +
          '<section class="preview-render-section preview-automation-card"><details class="preview-drawer"><summary>Automation <span>ⓘ</span></summary><div class="preview-mode-controls">' +
          selectInput('volume_mode', 'Volume', volumeModes, previewFormState.volume_mode, titleCaseOption) +
          selectInput('eq_mode', 'Low EQ', eqModes, previewFormState.eq_mode, titleCaseOption) +
          selectInput('filter_mode', 'Filter', filterModes, previewFormState.filter_mode, titleCaseOption) +
          '<label>Time snap<select data-automation-snap="time"><option value="off"' + (transitionAutomationSnapTime === 'off' ? ' selected' : '') + '>Off</option><option value="quarter"' + (transitionAutomationSnapTime === 'quarter' ? ' selected' : '') + '>¼ beat</option><option value="half"' + (transitionAutomationSnapTime === 'half' ? ' selected' : '') + '>½ beat</option><option value="beat"' + (transitionAutomationSnapTime === 'beat' ? ' selected' : '') + '>Beat</option><option value="bar"' + (transitionAutomationSnapTime === 'bar' ? ' selected' : '') + '>Bar</option></select></label>' +
          '<label>Value snap<select data-automation-snap="value"><option value="off"' + (transitionAutomationSnapValue === 'off' ? ' selected' : '') + '>Off</option><option value="1"' + (transitionAutomationSnapValue === '1' ? ' selected' : '') + '>1 dB</option><option value="3"' + (transitionAutomationSnapValue === '3' ? ' selected' : '') + '>3 dB</option></select></label>' +
          '<button type="button" data-automation-action="restore">Restore preset</button>' +
          '</div>' +
          '<div class="fx-legend automation-legend" aria-label="Select an automation lane to edit">' +
          ['from', 'to'].map(automationDeckLegendHtml).join('') +
          '</div>' +
          '<div class="automation-actions"><button type="button" data-automation-action="reset-lane">Reset lane</button><button type="button" data-automation-action="copy-lane">Copy to other deck</button><button type="button" data-automation-action="mirror-lane">Time mirror</button><label>Segment<select data-automation-curve disabled><option value="linear">Linear</option><option value="smooth">Smooth</option><option value="exponential">Exponential</option></select></label><label><span data-automation-segment-label="start:beat">Start beat</span><input data-automation-segment-field="start:beat" type="number" step="0.01" disabled></label><label><span data-automation-segment-label="start:value">Start value</span><input data-automation-segment-field="start:value" type="number" step="0.1" disabled><small data-automation-filter-readout="start"></small></label><label><span data-automation-segment-label="end:beat">End beat</span><input data-automation-segment-field="end:beat" type="number" step="0.01" disabled></label><label><span data-automation-segment-label="end:value">End value</span><input data-automation-segment-field="end:value" type="number" step="0.1" disabled><small data-automation-filter-readout="end"></small></label><button type="button" data-automation-action="delete-segment-keyframe" disabled>Delete keyframe</button></div>' +
          '<div class="automation-editor-hint">Select a lane, then double-click to add points or drag points. Filter: LPF ↓ · bypass — · HPF ↑.</div></details></section>' +
          '<div class="preview-render-actions">' +
          automationSummaryHtml() +
          '<div class="preview-actions">' +
          '<button type="button" data-preview-action="reset-alignments">Reset both alignments</button>' +
          '<button type="button" data-preview-action="clear">Clear transition</button>' +
          '</div>' +
          transitionLoadingHtml(transitionRenderPending) +
          '<div id="transition-render-status" class="preview-status' + (renderStatus.warn ? ' warn' : '') + '">' + esc(renderStatus.text) + '</div>' +
          '</div>' +
          '</div>';
      }
    } else {
      controlsHtml =
        '<div class="preview-actions">' +
        '<button type="button" data-preview-action="clear">Clear transition</button>' +
        '</div>';
    }
    const editorHtml = safeUi(
      'transition editor render',
      () => renderTransitionEditorHtml(from, to),
      '<div class="transition-editor"><div class="transition-editor-empty warn">Transition editor failed to render. Other app controls remain available.</div></div>'
    );
    const t = transitionRender && transitionRender.transition ? transitionRender.transition : null;
    const metaText = t ? (String(t.from_track_number || '') + ' ' + (t.from_title || '') + ' → ' + String(t.to_track_number || '') + ' ' + (t.to_title || '') + (t.duration_seconds ? ' / ' + Number(t.duration_seconds).toFixed(2) + 's' : '')) : 'No render yet';
    const linksHtml = transitionRender && transitionRender.urls ?
      '<a href="' + esc(transitionRender.urls.preview) + '" target="_blank">WAV</a>' : '';
    els.transitionPreview.innerHTML =
      '<div class="transition-workbench">' +
      '<section class="transition-main">' +
      nowPlayingHtml +
      '<section class="transition-controls">' + controlsHtml + '</section>' +
      editorHtml +
      '<div class="transition-topbar"><div class="transition-meta">' + esc(metaText) + '</div><div class="transition-links">' + linksHtml + '</div></div>' +
      '</section>' +
      '</div>';
    safeUi('transition preview bind', bindTransitionPreviewOverlay);
    restoreTransitionPlaybackState(playbackState);
  }
  function liveMixerLaneLabel(lane) {
    return lane === 'volume' ? 'Volume' : lane === 'eq_low' ? 'Low' : lane === 'eq_mid' ? 'Mid' : lane === 'eq_high' ? 'High' : lane === 'delay' ? 'Delay' : lane === 'reverb' ? 'Reverb' : 'Filter';
  }
  function liveMixerValueText(lane, value) {
    return automationValueLabel(lane, value);
  }
  function liveManualValue(deck, lane) {
    const value = liveManualOverrides[deck] && liveManualOverrides[deck][lane];
    return Number.isFinite(value) ? value : null;
  }
  function liveMixerKnobPosition(lane, value) {
    const numeric = Number(value) || 0;
    if (lane === 'filter') return clamp((numeric + 1) * .5, 0, 1);
    if (lane === 'delay' || lane === 'reverb') return clamp(numeric, 0, 1);
    return numeric <= 0
      ? clamp(.5 * ((numeric + 96) / 96), 0, .5)
      : clamp(.5 + (.5 * (numeric / 6)), .5, 1);
  }
  function liveMixerControlHtml(deck, lane) {
    const override = liveManualValue(deck, lane);
    const value = override === null ? liveLaneValueAt(deck, lane, liveTransportPosition) : override;
    const limits = automationLaneLimits(lane);
    const percent = lane === 'volume'
      ? clamp((value - limits[0]) / Math.max(.001, limits[1] - limits[0]), 0, 1)
      : liveMixerKnobPosition(lane, value);
    const automated = override === null;
    const control = lane === 'volume'
      ? '<input class="live-mixer-fader" data-live-mixer-control="' + deck + ':' + lane + '" type="range" min="' + limits[0] + '" max="' + limits[1] + '" step="0.1" value="' + value + '" title="Double-click to reset to 0 dB" aria-label="' + (deck === 'from' ? 'Track 1' : 'Track 2') + ' volume">'
      : '<div class="live-mixer-knob" style="--live-knob-fill:' + (percent * 270).toFixed(2) + 'deg;--live-knob-angle:' + (-135 + percent * 270).toFixed(2) + 'deg"><input data-live-mixer-control="' + deck + ':' + lane + '" type="range" min="' + limits[0] + '" max="' + limits[1] + '" step="' + (lane === 'filter' || lane === 'delay' || lane === 'reverb' ? '.01' : '.1') + '" value="' + value + '" title="Double-click to reset" aria-label="' + (deck === 'from' ? 'Track 1' : 'Track 2') + ' ' + liveMixerLaneLabel(lane) + '"><i></i></div>';
    return '<label class="live-mixer-control ' + (lane === 'volume' ? 'live-mixer-volume' : '') + '" data-live-mixer-lane="' + deck + ':' + lane + '">' +
      '<span>' + liveMixerLaneLabel(lane) + '</span>' + control +
      '<output data-live-mixer-value="' + deck + ':' + lane + '">' + esc(liveMixerValueText(lane, value)) + '</output>' +
      '<button type="button" class="live-mixer-auto' + (automated ? ' active' : '') + '" data-live-mixer-auto="' + deck + ':' + lane + '" aria-pressed="' + (automated ? 'true' : 'false') + '">AUTO</button>' +
      '</label>';
  }
  function liveAllAutomationEnabled() {
    return ['from', 'to'].every(deck => LIVE_MIXER_LANES.every(lane => liveManualValue(deck, lane) === null));
  }
  function liveFxSettingValueText(setting, value) {
    const numeric = Number(value) || 0;
    return setting === 'reverb_decay' ? numeric.toFixed(2) + ' s' : Math.round(numeric * 100) + '%';
  }
  function liveFxSettingKnobHtml(deck, effect, setting, label, value, min, max, step) {
    const numeric = clamp(Number(value) || 0, min, max);
    const percent = clamp((numeric - min) / Math.max(.001, max - min), 0, 1);
    const track = deck === 'from' ? 'Track 1' : 'Track 2';
    return '<label class="live-fx-setting"><span>' + label + '</span>' +
      '<div class="live-mixer-knob live-fx-setting-knob" data-live-fx-setting-knob="' + deck + ':' + setting + '" style="--live-knob-fill:' + (percent * 270).toFixed(2) + 'deg;--live-knob-angle:' + (-135 + percent * 270).toFixed(2) + 'deg">' +
      '<input data-live-fx-setting="' + deck + ':' + setting + '" type="range" min="' + min + '" max="' + max + '" step="' + step + '" value="' + numeric + '" title="Double-click to reset" aria-label="' + track + ' ' + effect + ' ' + label.toLowerCase() + '"><i></i></div>' +
      '<output data-live-fx-setting-value="' + deck + ':' + setting + '">' + esc(liveFxSettingValueText(setting, numeric)) + '</output></label>';
  }
  function liveMixerHtml() {
    if (!config.app_mode || transitionPreviewMode !== 'live') return '';
    const fxUnit = (deck, effect) => {
      const settings = (previewFormState.effects && previewFormState.effects[deck]) || {};
      const isDelay = effect === 'delay';
      const setting = isDelay ? 'delay_beats' : 'reverb_decay';
      const tone = isDelay ? 'delay_tone' : 'reverb_tone';
      const value = Number(settings[setting]) || (isDelay ? .5 : 1.2);
      const toneValue = Number(settings[tone]);
      const timeControl = isDelay
        ? '<label class="live-fx-select"><span>Beats</span><select data-live-fx-setting="' + deck + ':' + setting + '" aria-label="' + (deck === 'from' ? 'Track 1' : 'Track 2') + ' delay beats">' + [.125,.25,.5,.75,1,2,4].map(beats => '<option value="' + beats + '"' + (beats === value ? ' selected' : '') + '>' + (beats === .125 ? '⅛' : beats === .25 ? '¼' : beats === .5 ? '½' : beats === .75 ? '¾' : beats) + '</option>').join('') + '</select></label>'
        : liveFxSettingKnobHtml(deck, effect, setting, 'Decay', value, .25, 4, .05);
      return '<div class="live-fx-unit ' + effect + '"><b>' + (isDelay ? 'Echo' : 'Reverb') + '</b>' + liveMixerControlHtml(deck, effect) + timeControl + liveFxSettingKnobHtml(deck, effect, tone, 'Tone', Number.isFinite(toneValue) ? toneValue : .5, 0, 1, .01) + '</div>';
    };
    const deckHtml = deck => '<section class="live-mixer-deck ' + deck + '"><h3>' + (deck === 'from' ? 'Track 1 · outgoing' : 'Track 2 · incoming') + '</h3><div class="live-mixer-deck-controls">' + LIVE_CHANNEL_LANES.map(lane => liveMixerControlHtml(deck, lane)).join('') + '</div><div class="live-mixer-fx">' + fxUnit(deck, 'delay') + fxUnit(deck, 'reverb') + '</div></section>';
    return '<section class="live-mixer-dock' + (liveMixerOpen ? '' : ' collapsed') + '">' +
      '<div class="live-mixer-head"><button type="button" data-live-mixer-toggle aria-expanded="' + (liveMixerOpen ? 'true' : 'false') + '">Live mixer ' + (liveMixerOpen ? '▾' : '▸') + '</button><span>Manual moves suspend only that lane’s automation. AUTO resumes it; double-click a control for neutral.</span></div>' +
      (liveMixerOpen ? '<div class="live-mixer-body">' + deckHtml('from') + '<section class="live-crossfader"><label>Crossfader<input data-live-crossfader type="range" min="-1" max="1" step="0.01" value="' + liveCrossfader + '" aria-label="Live crossfader"></label><button type="button" class="live-mixer-auto-all' + (liveAllAutomationEnabled() ? ' active' : '') + '" data-live-mixer-auto-all aria-pressed="' + (liveAllAutomationEnabled() ? 'true' : 'false') + '" title="Toggle all deck automation">AUTO</button><div><span>1</span><button type="button" data-live-crossfader-reset title="Reset crossfader">Center</button><span>2</span></div></section>' + deckHtml('to') + '</div>' : '') +
      '</section>';
  }
  function transitionTransportHtml() {
    const mode = activeTransportMode();
    const duration = activeTransportDuration();
    const ready = transportCanPlay();
    const status = mode === 'live'
      ? liveStatusText()
      : (transitionRenderPending ? 'Rendering transition audio…' : (ready ? 'Verified render ready' : 'Render transition to enable verified playback.'));
    const body = '<div class="transition-scrubbar' + (ready ? '' : ' transition-scrubbar-empty') + '">' +
      '<input id="transition-scrub" class="scrub-range" aria-label="' + esc(status) + '" type="range" min="0" max="' + esc(Math.max(.01, duration || 1)) + '" step="0.01" value="' + esc(clamp(activeTransportPosition(), 0, Math.max(.01, duration || 1))) + '"' + (ready ? '' : ' disabled') + '>' +
      renderResultHtml() + '</div>';
    return '<div class="transition-transport-row transition-seam-transport-row">' +
      '<div class="transition-transport-main">' + body + '</div>' +
      '</div>';
  }
  function renderResultHtml() {
    if (!transitionRender || !transitionRender.ok || !transitionRender.urls) return '';
    const stamp = esc(transitionRender.rendered_at || '');
    return '<audio id="transition-preview-audio" preload="metadata" src="' + esc(transitionRender.urls.preview) + '?t=' + stamp + '"></audio>';
  }
  function activeTransportMode() {
    return transitionPreviewMode === 'live' ? 'live' : 'verified';
  }
  function liveTimingFromPayload(payload) {
    if (!payload) return null;
    const start = Math.max(0, Number(payload.transport_start_seconds != null ? payload.transport_start_seconds : payload.loop_start_seconds) || 0);
    const end = Math.max(start + .01, Number(payload.transport_end_seconds != null ? payload.transport_end_seconds : payload.loop_end_seconds) || 0);
    const transitionStart = clamp(Number(payload.transition_start_seconds), start, end);
    const transitionEnd = clamp(Number(payload.transition_end_seconds), transitionStart, end);
    return { start, end, duration: end - start, transitionStart, transitionEnd, fromBpm: Number(payload.from_bpm) || 120, beatsPerBar: Number(payload.beats_per_bar) || 4 };
  }
  function liveTransportTiming() {
    // A newer HQ preparation must never redefine an active source's loop or
    // automation clock. Keep the previous timing snapshot until the user
    // pauses/restarts, then the new prepared context becomes authoritative.
    if (livePlayback && livePlayback.timing) return livePlayback.timing;
    return liveTimingFromPayload(liveTransition && liveTransition.live);
  }
  function activeTransportDuration() {
    if (activeTransportMode() === 'live') {
      const timing = liveTransportTiming();
      return timing ? timing.duration : 0;
    }
    return transitionDuration(transitionAudioElement());
  }
  function activeTransportPosition() {
    if (activeTransportMode() === 'live') {
      if (liveSeekPending) return liveSeekPending.position;
      if (!livePlayback || !liveAudioContext || !Number.isFinite(livePlayback.duration)) return liveTransportPosition;
      const elapsed = Math.max(0, liveAudioContext.currentTime - livePlayback.startAt);
      const position = livePlayback.startPosition + elapsed;
      return transitionLoopEnabled ? position % livePlayback.duration : Math.min(position, livePlayback.duration);
    }
    const audio = transitionAudioElement();
    return audio && Number.isFinite(Number(audio.currentTime)) ? Number(audio.currentTime) : 0;
  }
  function transportCanPlay() {
    if (activeTransportMode() === 'live') {
      // A background HQ preparation must not disable the central pause control
      // for an already-running guarded rough preview.
      if (livePlayback || liveSeekInFlight) return !!liveTransportTiming();
      return !!(liveTransition && liveTransition.ok && !liveTransitionPending && liveTransportTiming());
    }
    return !!(transitionRender && transitionRender.ok && transitionRender.urls && transitionDuration(transitionAudioElement()) >= 0);
  }
  function transportIsPlaying() {
    if (activeTransportMode() === 'live') return !!(livePlayback || liveSeekInFlight);
    const audio = transitionAudioElement();
    return !!(audio && !audio.paused && !audio.ended);
  }
  function livePreparationKey() {
    const from = transitionFromIdx === null ? null : byIdx.get(transitionFromIdx);
    const to = transitionToIdx === null ? null : byIdx.get(transitionToIdx);
    if (!from || !to) return '';
    const s = previewFormState;
    return JSON.stringify({
      from: trackId(from), to: trackId(to), from_cue: s.from_cue || '', to_cue: s.to_cue || '',
      overlap_bars: Number(s.overlap_bars) || 0, front_padding_bars: Number(s.front_padding_bars) || 0,
      back_padding_bars: Number(s.back_padding_bars) || 0, from_nudge_beats: Number(s.from_nudge_beats) || 0,
      to_nudge_beats: Number(s.to_nudge_beats) || 0, from_beatgrid_ms: Number(s.from_beatgrid_ms) || 0,
      to_beatgrid_ms: Number(s.to_beatgrid_ms) || 0, from_pitch_shift: Number(s.from_pitch_shift) || 0,
      to_pitch_shift: Number(s.to_pitch_shift) || 0,
    });
  }
  function livePreparationBody() {
    const from = transitionFromIdx === null ? null : byIdx.get(transitionFromIdx);
    const to = transitionToIdx === null ? null : byIdx.get(transitionToIdx);
    if (!from || !to) return null;
    syncPreviewFormStateFromDom();
    return Object.assign({}, previewFormState, {
      from_track: trackId(from), to_track: trackId(to), live_guard_bars: 4,
      loudness_match_mode: loudnessMatchMode(),
    });
  }
  function livePreparedOffsetSeconds(deck) {
    const prepared = liveTransition && liveTransition._preparation;
    const timing = liveTransportTiming();
    if (!prepared || !timing) return 0;
    const nudgeField = deck === 'from' ? 'from_nudge_beats' : 'to_nudge_beats';
    const gridField = deck === 'from' ? 'from_beatgrid_ms' : 'to_beatgrid_ms';
    const beatSeconds = 60 / Math.max(1, Number(timing.fromBpm) || 120);
    return ((Number(previewFormState[nudgeField]) || 0) - (Number(prepared[nudgeField]) || 0)) * beatSeconds
      + (((Number(previewFormState[gridField]) || 0) - (Number(prepared[gridField]) || 0)) / 1000);
  }
  function refreshLiveRoughOffsets() {
    ['from', 'to'].forEach(deck => { liveRoughOffsets[deck] = livePreparedOffsetSeconds(deck); });
    return liveRoughOffsets;
  }
  function liveRoughOffsetsAreSafe() {
    const timing = liveTransportTiming();
    const payload = liveTransition && liveTransition.live;
    const prepared = liveTransition && liveTransition._preparation;
    const current = livePreparationBody();
    if (!timing || !payload || !prepared || !current) return false;
    const stableFields = ['from_track', 'to_track', 'from_cue', 'to_cue', 'overlap_bars', 'front_padding_bars', 'back_padding_bars', 'from_pitch_shift', 'to_pitch_shift'];
    if (stableFields.some(field => String(prepared[field] || '') !== String(current[field] || ''))) return false;
    const guardSeconds = Math.max(0, Number(payload.guard_bars) || 0) * Math.max(1, Number(payload.beats_per_bar) || 4) * 60 / Math.max(1, Number(payload.from_bpm) || 120);
    return ['from', 'to'].every(deck => Math.abs(Number(liveRoughOffsets[deck]) || 0) <= Math.max(0, guardSeconds - .02));
  }
  function liveDeckRoughOffset(deck) {
    const key = livePreparationKey();
    if (liveTransition && liveTransition._key === key) return 0;
    return Number(liveRoughOffsets[deck]) || 0;
  }
  function liveStatusText() {
    if (livePlayback) {
      if (liveTransitionPending) return 'Preparing updated HQ audio · current live transport continues';
      if (liveTransitionError) return 'Updated HQ preparation failed · current live transport continues';
      if (liveTransition && livePlayback.key !== liveTransition._key) return 'Updated HQ ready · pause and play the central transport to use it';
      return transitionLoopEnabled ? 'Live loop playing · live mixer automation' : 'Live transport playing · live mixer automation';
    }
    if (liveTransitionPending) return 'Preparing high-quality live deck audio…';
    if (liveTransitionMessage) return liveTransitionMessage;
    if (liveTransition && liveTransition.ok) return 'HQ live deck audio ready';
    return 'Select two tracks to prepare live audition.';
  }
  function updateLiveAuditionUi() {
    document.querySelectorAll('[data-live-status]').forEach(el => {
      el.textContent = liveStatusText();
      el.classList.toggle('warn', liveTransitionError);
    });
    syncLiveMixerUi();
    updateTransitionScrubber();
  }
  function syncLiveMixerUi() {
    document.querySelectorAll('[data-live-mixer-control]').forEach(input => {
      const [deck, lane] = String(input.getAttribute('data-live-mixer-control') || '').split(':');
      if (!deck || !lane) return;
      const override = liveManualValue(deck, lane);
      const value = override === null ? liveLaneValueAt(deck, lane, liveTransportPosition) : override;
      input.value = String(value);
      const limits = automationLaneLimits(lane);
      const knob = input.closest('.live-mixer-knob');
      if (knob) {
        const percent = liveMixerKnobPosition(lane, value);
        knob.style.setProperty('--live-knob-fill', (percent * 270).toFixed(2) + 'deg');
        knob.style.setProperty('--live-knob-angle', (-135 + percent * 270).toFixed(2) + 'deg');
      }
      const wrapper = input.closest('[data-live-mixer-lane]');
      if (wrapper) wrapper.classList.toggle('manual', override !== null);
      const readout = document.querySelector('[data-live-mixer-value="' + deck + ':' + lane + '"]');
      if (readout) readout.textContent = liveMixerValueText(lane, value);
      const auto = document.querySelector('[data-live-mixer-auto="' + deck + ':' + lane + '"]');
      if (auto) {
        auto.classList.toggle('active', override === null);
        auto.setAttribute('aria-pressed', override === null ? 'true' : 'false');
      }
    });
    document.querySelectorAll('[data-live-crossfader]').forEach(input => { input.value = String(liveCrossfader); });
    document.querySelectorAll('[data-live-mixer-auto-all]').forEach(button => {
      const active = liveAllAutomationEnabled();
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', active ? 'true' : 'false');
      button.setAttribute('title', active ? 'Suspend all deck automation' : 'Resume all deck automation');
    });
    syncLiveFxSettingUi();
  }
  function syncLiveFxSettingUi() {
    document.querySelectorAll('[data-live-fx-setting]').forEach(input => {
      const [deck, setting] = String(input.getAttribute('data-live-fx-setting') || '').split(':');
      const settings = previewFormState.effects && previewFormState.effects[deck];
      if (!settings || !Object.prototype.hasOwnProperty.call(settings, setting)) return;
      const value = Number(settings[setting]);
      if (!Number.isFinite(value)) return;
      input.value = String(value);
      const knob = input.closest('.live-fx-setting-knob');
      if (knob) {
        const min = Number(input.min);
        const max = Number(input.max);
        const percent = clamp((value - min) / Math.max(.001, max - min), 0, 1);
        knob.style.setProperty('--live-knob-fill', (percent * 270).toFixed(2) + 'deg');
        knob.style.setProperty('--live-knob-angle', (-135 + percent * 270).toFixed(2) + 'deg');
      }
      const output = document.querySelector('[data-live-fx-setting-value="' + deck + ':' + setting + '"]');
      if (output) output.textContent = liveFxSettingValueText(setting, value);
    });
  }
  function setLiveManualOverride(deck, lane, rawValue) {
    if (!LIVE_MIXER_LANES.includes(lane) || !liveManualOverrides[deck]) return;
    const limits = automationLaneLimits(lane);
    liveManualOverrides[deck][lane] = clamp(Number(rawValue) || 0, limits[0], limits[1]);
    requestLiveAutomationReschedule();
    syncLiveMixerUi();
  }
  function setLiveFxSetting(deck, setting, rawValue) {
    if (!['from', 'to'].includes(deck) || !['delay_beats', 'delay_tone', 'reverb_decay', 'reverb_tone'].includes(setting)) return;
    if (!previewFormState.effects) previewFormState.effects = { from: {}, to: {} };
    if (!previewFormState.effects[deck]) previewFormState.effects[deck] = {};
    const raw = Number(rawValue);
    if (!Number.isFinite(raw)) return;
    const value = setting === 'delay_beats'
      ? [.125, .25, .5, .75, 1, 2, 4].reduce((best, candidate) => Math.abs(candidate - raw) < Math.abs(best - raw) ? candidate : best, .5)
      : setting === 'reverb_decay' ? clamp(raw, .25, 4) : clamp(raw, 0, 1);
    previewFormState.effects[deck][setting] = value;
    if (livePlayback && livePlayback.decks[deck]) applyLiveFxSettings(livePlayback.decks[deck], deck);
    syncLiveFxSettingUi();
    updateTransitionDirtyUi();
  }
  function restoreLiveLaneAutomation(deck, lane) {
    if (!liveManualOverrides[deck]) return;
    delete liveManualOverrides[deck][lane];
    requestLiveAutomationReschedule();
    syncLiveMixerUi();
  }
  function toggleAllLiveAutomation() {
    if (liveAllAutomationEnabled()) {
      const position = activeTransportPosition();
      ['from', 'to'].forEach(deck => {
        LIVE_MIXER_LANES.forEach(lane => {
          liveManualOverrides[deck][lane] = liveAutomationLaneValueAt(deck, lane, position);
        });
      });
    } else {
      liveManualOverrides.from = {};
      liveManualOverrides.to = {};
    }
    requestLiveAutomationReschedule();
    syncLiveMixerUi();
  }
  function setTransitionLoopEnabled(enabled) {
    const next = !!enabled;
    if (transitionLoopEnabled === next) return;
    const currentLivePosition = livePlayback && liveAudioContext
      ? (livePlayback.startPosition + Math.max(0, liveAudioContext.currentTime - livePlayback.startAt)) % livePlayback.duration
      : activeTransportPosition();
    transitionLoopEnabled = next;
    const rendered = transitionAudioElement();
    if (rendered) rendered.loop = next;
    if (activeTransportMode() === 'live' && (livePlayback || liveSeekInFlight)) seekLiveTransport(currentLivePosition, { restart: true });
    updateTransitionScrubber();
  }
  async function prepareLiveTransition({ playWhenReady=false } = {}) {
    if (!config.app_mode || liveTransitionPending) return null;
    const body = livePreparationBody();
    if (!body) return null;
    const key = livePreparationKey();
    if (liveTransition && liveTransition._key === key) {
      if (playWhenReady) await startLiveAudition();
      return liveTransition;
    }
    const requestId = liveTransitionRequestId + 1;
    const previousLiveTransition = liveTransition;
    liveTransitionRequestId = requestId;
    liveTransitionPending = true;
    liveTransitionMessage = '';
    liveTransitionError = false;
    updateLiveAuditionUi();
    try {
      const res = await fetch('/api/prepare-live-transition', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      });
      const data = await res.json();
      if (requestId !== liveTransitionRequestId) return null;
      if (!data.ok) throw new Error(data.error || 'Live preparation failed');
      data._key = key;
      data._preparation = { ...body };
      liveTransition = data;
      liveTransitionMessage = 'HQ ready — restart to use prepared audio.';
      liveTransitionError = false;
      if (playWhenReady) await startLiveAudition();
      return data;
    } catch (err) {
      if (requestId === liveTransitionRequestId) {
        liveTransition = previousLiveTransition && previousLiveTransition.ok ? previousLiveTransition : null;
        liveTransitionMessage = err && err.message ? err.message : 'Live preparation failed';
        liveTransitionError = true;
      }
      return null;
    } finally {
      if (requestId === liveTransitionRequestId) {
        liveTransitionPending = false;
        updateLiveAuditionUi();
      }
    }
  }
  function scheduleLivePreparation() {
    if (!config.app_mode || transitionFromIdx === null || transitionToIdx === null) return;
    if (livePreparationTimer !== null) clearTimeout(livePreparationTimer);
    livePreparationTimer = setTimeout(() => {
      livePreparationTimer = null;
      const key = livePreparationKey();
      if (key && (!liveTransition || liveTransition._key !== key)) prepareLiveTransition();
    }, 350);
  }
  function liveAudioEngine() {
    if (liveAudioContext) return liveAudioContext;
    const Constructor = window.AudioContext || window.webkitAudioContext;
    if (!Constructor) throw new Error('Web Audio is unavailable in this browser.');
    liveAudioContext = new Constructor();
    liveMasterGain = liveAudioContext.createGain();
    liveMasterGain.gain.value = Math.pow(10, -3 / 20);
    liveCompressor = liveAudioContext.createDynamicsCompressor();
    liveCompressor.threshold.value = -12;
    liveCompressor.knee.value = 12;
    liveCompressor.ratio.value = 12;
    liveCompressor.attack.value = 0.003;
    liveCompressor.release.value = 0.08;
    liveMasterGain.connect(liveCompressor);
    liveCompressor.connect(liveAudioContext.destination);
    return liveAudioContext;
  }
  async function liveBuffer(url) {
    const key = String(url || '');
    if (!key) throw new Error('Prepared deck audio is unavailable.');
    if (liveBufferCache.has(key)) {
      const value = liveBufferCache.get(key);
      liveBufferCache.delete(key); liveBufferCache.set(key, value);
      return value;
    }
    const context = liveAudioEngine();
    const response = await fetch(key);
    if (!response.ok) throw new Error('Could not load prepared deck audio.');
    const value = await context.decodeAudioData(await response.arrayBuffer());
    liveBufferCache.set(key, value);
    while (liveBufferCache.size > 6) liveBufferCache.delete(liveBufferCache.keys().next().value);
    return value;
  }
  const LIVE_ISOLATOR_LOW_HZ = 275;
  const LIVE_ISOLATOR_HIGH_HZ = 3000;
  const LIVE_FILTER_MIN_HZ = 20;
  function livePositionInTransport(position, timing) {
    const raw = Number(position) || 0;
    if (!timing || !transitionLoopEnabled) return clamp(raw, 0, Math.max(0, timing ? timing.duration : 0));
    return ((raw % timing.duration) + timing.duration) % timing.duration;
  }
  function liveContextGatedValue(deck, lane, position, value) {
    if (lane !== 'volume') return value;
    const timing = liveTransportTiming();
    if (!timing) return value;
    const local = livePositionInTransport(position, timing);
    const overlapStart = timing.transitionStart - timing.start;
    const overlapEnd = timing.transitionEnd - timing.start;
    if (local < overlapStart) return deck === 'from' ? 0 : -96;
    if (local >= overlapEnd) return deck === 'from' ? -96 : 0;
    return value;
  }
  function liveAutomationLaneValueAt(deck, lane, position) {
    ensureTransitionAutomation();
    const timing = liveTransportTiming();
    if (!timing) return 0;
    const points = automationLane(deck, lane) || [];
    const totalBeats = Math.max(.001, transitionAutomationBeats());
    const overlapStart = timing.transitionStart - timing.start;
    const overlapDuration = Math.max(.001, timing.transitionEnd - timing.transitionStart);
    const local = livePositionInTransport(position, timing);
    const beat = local <= overlapStart ? 0 : (local >= overlapStart + overlapDuration
      ? totalBeats
      : totalBeats * ((local - overlapStart) / overlapDuration));
    const automatic = Number(automationValueAt(points, beat, lane));
    const limits = automationLaneLimits(lane);
    return clamp(Number.isFinite(automatic) ? automatic : 0, limits[0], limits[1]);
  }
  function liveLaneValueAt(deck, lane, position) {
    const automatic = liveAutomationLaneValueAt(deck, lane, position);
    const override = liveManualValue(deck, lane);
    const value = override === null ? automatic : override;
    const limits = automationLaneLimits(lane);
    const normalized = clamp(Number.isFinite(value) ? value : 0, limits[0], limits[1]);
    return liveContextGatedValue(deck, lane, position, normalized);
  }
  function liveDbToGain(value) {
    const db = Number(value);
    return db <= -95.5 ? 0 : Math.pow(10, (Number.isFinite(db) ? db : 0) / 20);
  }
  function liveLaneEnvelope(deck, lane, startPosition, duration, transform=value => value, samples=1025) {
    const values = new Float32Array(samples);
    for (let i = 0; i < samples; i += 1) {
      values[i] = transform(liveLaneValueAt(deck, lane, startPosition + (duration * (i / (samples - 1)))));
    }
    return values;
  }
  function liveCrossfaderGain(deck) {
    const value = clamp(Number(liveCrossfader) || 0, -1, 1);
    const travel = deck === 'from' ? Math.max(0, value) : Math.max(0, -value);
    return Math.cos(travel * Math.PI * .5);
  }
  function applyLiveCrossfader({ smooth=true } = {}) {
    if (!livePlayback || !liveAudioContext) return;
    const now = liveAudioContext.currentTime;
    ['from', 'to'].forEach(deck => {
      const param = livePlayback.decks[deck] && livePlayback.decks[deck].performance && livePlayback.decks[deck].performance.gain;
      if (!param) return;
      if (typeof param.cancelAndHoldAtTime === 'function') param.cancelAndHoldAtTime(now);
      else param.cancelScheduledValues(now);
      param.setTargetAtTime(liveCrossfaderGain(deck), now, smooth ? .012 : .001);
    });
  }
  function liveAutomationWindow(startAt, startPosition, smooth) {
    const timing = liveTransportTiming();
    if (!timing) return null;
    const horizon = Math.min(LIVE_AUTOMATION_HORIZON_SECONDS, Math.max(2, timing.duration));
    const rampSeconds = smooth ? .02 : .008;
    const curveStart = startAt + rampSeconds + .001;
    const curvePosition = startPosition + (curveStart - startAt);
    return { startAt, curveStart, curvePosition, horizon, smooth };
  }
  function scheduleLiveParameter(param, values, window) {
    if (!param || !window) return window ? window.startAt : 0;
    const { startAt, curveStart, horizon } = window;
    if (typeof param.cancelAndHoldAtTime === 'function') param.cancelAndHoldAtTime(startAt);
    else {
      param.cancelScheduledValues(startAt);
      param.setValueAtTime(param.value, startAt);
    }
    param.linearRampToValueAtTime(values[0], curveStart - .001);
    param.setValueCurveAtTime(values, curveStart, horizon);
    return curveStart + horizon;
  }
  function liveFilterValues(deck, window) {
    const amount = liveLaneEnvelope(deck, 'filter', window.curvePosition, window.horizon);
    const dry = new Float32Array(amount.length);
    const lowWet = new Float32Array(amount.length);
    const highWet = new Float32Array(amount.length);
    const lowFrequency = new Float32Array(amount.length);
    const highFrequency = new Float32Array(amount.length);
    const kill = new Float32Array(amount.length);
    const maxFrequency = Math.min(20000, ((liveAudioContext && liveAudioContext.sampleRate) || 44100) * .5 - 100);
    for (let i = 0; i < amount.length; i += 1) {
      const value = amount[i];
      const wet = Math.abs(value);
      dry[i] = 1 - wet;
      lowWet[i] = value < 0 ? wet : 0;
      highWet[i] = value > 0 ? wet : 0;
      lowFrequency[i] = value < 0 ? maxFrequency * Math.pow(LIVE_FILTER_MIN_HZ / maxFrequency, wet) : maxFrequency;
      highFrequency[i] = value > 0 ? LIVE_FILTER_MIN_HZ * Math.pow(maxFrequency / LIVE_FILTER_MIN_HZ, wet) : LIVE_FILTER_MIN_HZ;
      const edge = clamp((wet - .975) / .02, 0, 1);
      kill[i] = wet >= .995 ? 0 : 1 - smoothstep(edge);
    }
    return { dry, lowWet, highWet, lowFrequency, highFrequency, kill };
  }
  function liveFxSettings(deck) {
    const settings = previewFormState.effects && previewFormState.effects[deck] || {};
    const delayBeats = [.125, .25, .5, .75, 1, 2, 4].reduce((best, value) => Math.abs(value - Number(settings.delay_beats || .5)) < Math.abs(best - Number(settings.delay_beats || .5)) ? value : best, .5);
    return {
      delayBeats,
      delayTone: clamp(Number.isFinite(Number(settings.delay_tone)) ? Number(settings.delay_tone) : .5, 0, 1),
      reverbDecay: clamp(Number(settings.reverb_decay) || 1.2, .25, 4),
      reverbTone: clamp(Number.isFinite(Number(settings.reverb_tone)) ? Number(settings.reverb_tone) : .5, 0, 1),
    };
  }
  function applyLiveFxSettings(graph, deck, { smooth=true } = {}) {
    if (!graph || !liveAudioContext) return;
    const settings = liveFxSettings(deck);
    const now = liveAudioContext.currentTime;
    const target = (param, value) => {
      if (!param) return;
      param.cancelScheduledValues(now);
      param.setTargetAtTime(value, now, smooth ? .018 : .001);
    };
    target(graph.delay.delayTime, (60 / Math.max(1, Number(liveTransportTiming() && liveTransportTiming().fromBpm) || 120)) * settings.delayBeats);
    target(graph.delayTone.frequency, 1800 * Math.pow(6, settings.delayTone));
    const feedback = clamp(.32 + (settings.reverbDecay / 4) * .48, .32, .8);
    graph.reverbFeedbacks.forEach(param => target(param.gain, feedback));
    graph.reverbTones.forEach(node => target(node.frequency, 1400 * Math.pow(7.5, settings.reverbTone)));
  }
  function applyLiveLoudnessGain(graph, deck, { smooth=true } = {}) {
    if (!graph || !graph.loudness || !liveAudioContext) return;
    const record = transitionDeckConfig(deck).record;
    const target = loudnessGainLinear(record, 'transition');
    const now = liveAudioContext.currentTime;
    graph.loudness.gain.cancelScheduledValues(now);
    graph.loudness.gain.setTargetAtTime(target, now, smooth ? .018 : .001);
  }
  function applyLiveLoudnessGains({ smooth=true } = {}) {
    if (!livePlayback) return;
    ['from', 'to'].forEach(deck => applyLiveLoudnessGain(livePlayback.decks[deck], deck, { smooth }));
  }
  function createLiveDeckGraph(context, source) {
    const loudness = context.createGain();
    const low = context.createBiquadFilter(); low.type = 'lowpass'; low.frequency.value = LIVE_ISOLATOR_LOW_HZ; low.Q.value = Math.SQRT1_2;
    const high = context.createBiquadFilter(); high.type = 'highpass'; high.frequency.value = LIVE_ISOLATOR_HIGH_HZ; high.Q.value = Math.SQRT1_2;
    const midBase = context.createGain();
    const lowDelta = context.createGain();
    const highDelta = context.createGain();
    const eqOutput = context.createGain();
    source.connect(loudness);
    loudness.connect(midBase); midBase.connect(eqOutput);
    loudness.connect(low); low.connect(lowDelta); lowDelta.connect(eqOutput);
    loudness.connect(high); high.connect(highDelta); highDelta.connect(eqOutput);

    const dry = context.createGain();
    const filterLow = context.createBiquadFilter(); filterLow.type = 'lowpass'; filterLow.Q.value = Math.SQRT1_2;
    const filterHigh = context.createBiquadFilter(); filterHigh.type = 'highpass'; filterHigh.Q.value = Math.SQRT1_2;
    const lowWet = context.createGain();
    const highWet = context.createGain();
    const filterOutput = context.createGain();
    const filterKill = context.createGain();
    const volume = context.createGain();
    const performance = context.createGain();
    const fxOutput = context.createGain();
    const delay = context.createDelay(4);
    const delayTone = context.createBiquadFilter(); delayTone.type = 'lowpass';
    const delayHighPass = context.createBiquadFilter(); delayHighPass.type = 'highpass'; delayHighPass.frequency.value = 180;
    const delayFeedback = context.createGain();
    const delayWet = context.createGain();
    const reverbWet = context.createGain();
    const reverbFeedbacks = [];
    const reverbTones = [];
    eqOutput.connect(dry); dry.connect(filterOutput);
    eqOutput.connect(filterLow); filterLow.connect(lowWet); lowWet.connect(filterOutput);
    eqOutput.connect(filterHigh); filterHigh.connect(highWet); highWet.connect(filterOutput);
    filterOutput.connect(filterKill); filterKill.connect(volume);
    volume.connect(fxOutput);
    volume.connect(delay); delay.connect(delayTone); delayTone.connect(delayHighPass); delayHighPass.connect(delayWet); delayWet.connect(fxOutput);
    delayHighPass.connect(delayFeedback); delayFeedback.connect(delay);
    [0.0297, 0.0371, 0.0411, 0.0437].forEach(seconds => {
      const comb = context.createDelay(.1); comb.delayTime.value = seconds;
      const tone = context.createBiquadFilter(); tone.type = 'lowpass'; tone.frequency.value = 4000;
      const feedback = context.createGain(); feedback.gain.value = .45;
      const returnGain = context.createGain(); returnGain.gain.value = .25;
      volume.connect(comb); comb.connect(tone); tone.connect(returnGain); returnGain.connect(reverbWet); tone.connect(feedback); feedback.connect(comb);
      reverbFeedbacks.push(feedback); reverbTones.push(tone);
    });
    // The reverb comb returns share one wet bus, just like the delay return.
    // Without this connection the macro was scheduled but inaudible.
    reverbWet.connect(fxOutput);
    fxOutput.connect(performance); performance.connect(liveMasterGain);
    const graph = { loudness, volume, performance, midBase, lowDelta, highDelta, dry, lowWet, highWet, filterLow, filterHigh, filterKill, delay, delayTone, delayFeedback, delayWet, reverbWet, reverbFeedbacks, reverbTones };
    return graph;
  }
  function scheduleLiveDeckAutomation(graph, deck, startAt, startPosition, { smooth=false } = {}) {
    const window = liveAutomationWindow(startAt, startPosition, smooth);
    if (!window) return startAt;
    const volume = liveLaneEnvelope(deck, 'volume', window.curvePosition, window.horizon, liveDbToGain);
    const low = liveLaneEnvelope(deck, 'eq_low', window.curvePosition, window.horizon, liveDbToGain);
    const mid = liveLaneEnvelope(deck, 'eq_mid', window.curvePosition, window.horizon, liveDbToGain);
    const high = liveLaneEnvelope(deck, 'eq_high', window.curvePosition, window.horizon, liveDbToGain);
    const lowDelta = new Float32Array(low.length);
    const highDelta = new Float32Array(high.length);
    for (let i = 0; i < low.length; i += 1) {
      lowDelta[i] = low[i] - mid[i];
      highDelta[i] = high[i] - mid[i];
    }
    const filter = liveFilterValues(deck, window);
    const delay = liveLaneEnvelope(deck, 'delay', window.curvePosition, window.horizon);
    const reverb = liveLaneEnvelope(deck, 'reverb', window.curvePosition, window.horizon);
    const until = [
      scheduleLiveParameter(graph.volume.gain, volume, window),
      scheduleLiveParameter(graph.midBase.gain, mid, window),
      scheduleLiveParameter(graph.lowDelta.gain, lowDelta, window),
      scheduleLiveParameter(graph.highDelta.gain, highDelta, window),
      scheduleLiveParameter(graph.dry.gain, filter.dry, window),
      scheduleLiveParameter(graph.lowWet.gain, filter.lowWet, window),
      scheduleLiveParameter(graph.highWet.gain, filter.highWet, window),
      scheduleLiveParameter(graph.filterLow.frequency, filter.lowFrequency, window),
      scheduleLiveParameter(graph.filterHigh.frequency, filter.highFrequency, window),
      scheduleLiveParameter(graph.filterKill.gain, filter.kill, window),
      scheduleLiveParameter(graph.delayWet.gain, delay.map(value => value * .52), window),
      scheduleLiveParameter(graph.delayFeedback.gain, delay.map(value => .18 + (value * .38)), window),
      scheduleLiveParameter(graph.reverbWet.gain, reverb.map(value => value * .46), window),
    ];
    return Math.min(...until);
  }
  function rescheduleLiveAutomation({ smooth=true } = {}) {
    if (!livePlayback || !liveAudioContext) return;
    const startAt = liveAudioContext.currentTime + .005;
    const position = activeTransportPosition();
    const until = ['from', 'to'].map(deck => scheduleLiveDeckAutomation(livePlayback.decks[deck], deck, startAt, position, { smooth }));
    livePlayback.volumeScheduledUntil = Math.min(...until);
  }
  function requestLiveAutomationReschedule() {
    if (!livePlayback || liveAutomationRescheduleFrame !== null) return;
    const update = () => {
      liveAutomationRescheduleFrame = null;
      rescheduleLiveAutomation({ smooth: true });
    };
    liveAutomationRescheduleFrame = window.requestAnimationFrame ? window.requestAnimationFrame(update) : setTimeout(update, 0);
  }
  function stopLiveAudition({ clear=false, invalidateStart=true } = {}) {
    if (clear) cancelLiveSeekQueue();
    if (invalidateStart) livePlaybackStartRequestId += 1;
    if (livePlayback) {
      if (!liveSeekPending) liveTransportPosition = activeTransportPosition();
      livePlayback.sources.forEach(source => { try { source.stop(); } catch (err) {} try { source.disconnect(); } catch (err) {} });
      livePlayback.gains.forEach(gain => { try { gain.disconnect(); } catch (err) {} });
    }
    livePlayback = null;
    if (clear) {
      liveTransition = null;
      liveTransitionMessage = '';
      liveTransitionError = false;
      liveTransportPosition = 0;
      liveTransitionRequestId += 1;
      if (livePreparationTimer !== null) { clearTimeout(livePreparationTimer); livePreparationTimer = null; }
      liveManualOverrides.from = {};
      liveManualOverrides.to = {};
      liveCrossfader = 0;
      liveRoughOffsets.from = 0;
      liveRoughOffsets.to = 0;
    }
    updateLiveAuditionUi();
  }
  async function startLiveAudition({ position=liveTransportPosition, restartToken=null } = {}) {
    if (!liveTransition || !liveTransition.ok) {
      await prepareLiveTransition({ playWhenReady: true });
      return;
    }
    const startRequestId = livePlaybackStartRequestId + 1;
    livePlaybackStartRequestId = startRequestId;
    let startedSources = [];
    let startedGains = [];
    try {
      const context = liveAudioEngine();
      await context.resume();
      liveTransitionMessage = 'Loading prepared deck audio…';
      liveTransitionError = false;
      updateLiveAuditionUi();
      const timing = liveTransportTiming();
      if (!timing) throw new Error('Prepared live timing is unavailable. Prepare the deck audio again.');
      const [fromBuffer, toBuffer] = await Promise.all([liveBuffer(liveTransition.urls.from), liveBuffer(liveTransition.urls.to)]);
      if (startRequestId !== livePlaybackStartRequestId || (restartToken !== null && (!liveSeekPending || restartToken !== liveSeekPending.token))) {
        startedSources.forEach(source => { try { source.stop(); } catch (err) {} try { source.disconnect(); } catch (err) {} });
        return;
      }
      const loopStart = timing.start;
      const loopEnd = timing.end;
      if (!Number.isFinite(loopEnd) || loopEnd <= loopStart + .005) {
        throw new Error('Prepared deck window is too short for the requested loop. Restart HQ preparation.');
      }
      const loopDuration = loopEnd - loopStart;
      const startPosition = clamp(Number(position) || 0, 0, Math.max(0, loopDuration - .001));
      stopLiveAudition({ clear: false, invalidateStart: false });
      const startAt = context.currentTime + .045;
      const makeDeck = (deck, buffer) => {
        const roughOffset = liveDeckRoughOffset(deck);
        const deckLoopStart = loopStart + roughOffset;
        const deckLoopEnd = loopEnd + roughOffset;
        if (deckLoopStart < 0 || deckLoopEnd > buffer.duration - .005 || deckLoopEnd <= deckLoopStart + .005) {
          throw new Error('The requested beat jump is outside prepared guard audio. Wait for HQ preparation.');
        }
        const source = context.createBufferSource();
        source.buffer = buffer;
        source.loop = transitionLoopEnabled;
        source.loopStart = deckLoopStart;
        source.loopEnd = deckLoopEnd;
        const graph = createLiveDeckGraph(context, source);
        applyLiveFxSettings(graph, deck, { smooth: false });
        applyLiveLoudnessGain(graph, deck, { smooth: false });
        const sourceWindow = transitionWindowInfo(deck);
        return { source, graph, loopStart: deckLoopStart, sourceContextStart: sourceWindow ? sourceWindow.contextStart : 0 };
      };
      const from = makeDeck('from', fromBuffer);
      const to = makeDeck('to', toBuffer);
      startedSources = [from.source, to.source];
      startedGains = [from.graph.volume, from.graph.performance, to.graph.volume, to.graph.performance];
      if (els.audio && !els.audio.paused) pauseSharedAudio();
      const rendered = transitionAudioElement();
      if (rendered && !rendered.paused) rendered.pause();
      const volumeScheduledUntil = Math.min(
        scheduleLiveDeckAutomation(from.graph, 'from', startAt, startPosition),
        scheduleLiveDeckAutomation(to.graph, 'to', startAt, startPosition),
      );
      from.source.start(startAt, from.loopStart + startPosition);
      to.source.start(startAt, to.loopStart + startPosition);
      if (!transitionLoopEnabled) {
        const remaining = Math.max(.001, loopDuration - startPosition);
        from.source.stop(startAt + remaining);
        to.source.stop(startAt + remaining);
      }
      if (startRequestId !== livePlaybackStartRequestId || (restartToken !== null && (!liveSeekPending || restartToken !== liveSeekPending.token))) {
        startedSources.forEach(source => { try { source.stop(); } catch (err) {} try { source.disconnect(); } catch (err) {} });
        return;
      }
      liveTransportPosition = startPosition;
      livePlayback = {
        sources: [from.source, to.source],
        gains: [from.graph.volume, from.graph.performance, to.graph.volume, to.graph.performance],
        decks: { from: from.graph, to: to.graph },
        sourceContextStarts: { from: from.sourceContextStart, to: to.sourceContextStart },
        timing: Object.assign({}, timing),
        startAt, startPosition, duration: loopDuration, volumeScheduledUntil, key: liveTransition._key,
      };
      if (liveTransition._key === livePreparationKey()) {
        liveRoughOffsets.from = 0;
        liveRoughOffsets.to = 0;
      }
      applyLiveCrossfader({ smooth: false });
      from.source.onended = () => {
        if (livePlayback && livePlayback.sources.includes(from.source) && !transitionLoopEnabled) {
          liveTransportPosition = loopDuration;
          stopLiveAudition({ invalidateStart: false });
          updateTransitionScrubber();
        }
      };
      liveTransitionMessage = transitionLoopEnabled ? 'Live loop playing · live mixer automation' : 'Live transport playing · live mixer automation';
      liveTransitionError = false;
      updateLiveAuditionUi();
      requestLivePlaybackFrame();
    } catch (err) {
      startedSources.forEach(source => { try { source.stop(); } catch (stopErr) {} try { source.disconnect(); } catch (disconnectErr) {} });
      startedGains.forEach(gain => { try { gain.disconnect(); } catch (disconnectErr) {} });
      if (startRequestId !== livePlaybackStartRequestId) return;
      liveTransitionMessage = 'Live audition failed: ' + (err && err.message ? err.message : 'Could not start prepared audio.');
      liveTransitionError = true;
      updateLiveAuditionUi();
    }
  }
  function requestLivePlaybackFrame() {
    if (!livePlayback || !window.requestAnimationFrame) return;
    window.requestAnimationFrame(() => {
      if (!livePlayback || !liveAudioContext) return;
      liveTransportPosition = activeTransportPosition();
      if (livePlayback.volumeScheduledUntil - liveAudioContext.currentTime < LIVE_AUTOMATION_REFRESH_SECONDS) rescheduleLiveAutomation({ smooth: false });
      requestTransitionEditorDraw();
      updateTransitionScrubber();
      updateLiveAuditionUi();
      requestLivePlaybackFrame();
    });
  }
  function transitionAudioElement() {
    return document.getElementById('transition-preview-audio');
  }
  function captureTransitionPlaybackState() {
    const audio = transitionAudioElement();
    if (!audio || !Number.isFinite(Number(audio.currentTime))) return null;
    return { time: Number(audio.currentTime), playing: !audio.paused && !audio.ended };
  }
  function restoreTransitionPlaybackState(state) {
    if (!state) return;
    const audio = transitionAudioElement();
    if (!audio) return;
    const restore = () => {
      const duration = transitionDuration(audio);
      try { audio.currentTime = clamp(state.time, 0, Number.isFinite(duration) && duration > 0 ? duration : state.time); } catch (err) {}
      if (state.playing) {
        const promise = audio.play();
        if (promise && promise.catch) promise.catch(() => {});
      }
      updateTransitionScrubber();
    };
    if (audio.readyState >= 1) restore();
    else audio.addEventListener('loadedmetadata', restore, { once: true });
  }
  function transitionDuration(audio) {
    const audioDuration = Number(audio && audio.duration);
    if (Number.isFinite(audioDuration) && audioDuration > 0) return audioDuration;
    const renderDuration = Number(transitionRender && transitionRender.transition && transitionRender.transition.duration_seconds);
    return Number.isFinite(renderDuration) && renderDuration > 0 ? renderDuration : 0;
  }
  function scheduleTransitionScrubber() {
    if (transitionScrubFrame !== null || !window.requestAnimationFrame) return;
    transitionScrubFrame = window.requestAnimationFrame(() => {
      transitionScrubFrame = null;
      updateTransitionScrubber();
    });
  }
  function requestTransitionEditorDraw() {
    if (activePane !== 'preview') return;
    if (!window.requestAnimationFrame) {
      drawTransitionEditor();
      return;
    }
    if (transitionEditorFrame !== null) return;
    transitionEditorFrame = window.requestAnimationFrame(() => {
      transitionEditorFrame = null;
      drawTransitionEditor();
    });
  }
  function updateTransitionScrubber() {
    const button = document.getElementById('transition-play-toggle');
    const scrub = document.getElementById('transition-scrub');
    const time = document.getElementById('transition-scrub-time');
    if (!button || !scrub || !time) return;
    const duration = activeTransportDuration();
    const current = activeTransportPosition();
    const ready = transportCanPlay();
    button.disabled = !ready;
    button.textContent = transportIsPlaying() ? '⏸' : '▶';
    button.setAttribute('aria-label', transportIsPlaying() ? 'Pause' : 'Play');
    const loop = document.getElementById('transition-loop-toggle');
    if (loop) {
      loop.classList.toggle('active', transitionLoopEnabled);
      loop.setAttribute('aria-pressed', transitionLoopEnabled ? 'true' : 'false');
      loop.setAttribute('title', transitionLoopEnabled ? 'Disable transition loop' : 'Enable transition loop');
      loop.setAttribute('aria-label', transitionLoopEnabled ? 'Disable transition loop' : 'Enable transition loop');
    }
    scrub.disabled = !ready;
    scrub.min = '0';
    scrub.max = String(Math.max(.01, duration || 1));
    if (!transitionScrubDragging) scrub.value = String(clamp(current, 0, Number(scrub.max) || 1));
    setRangeProgress(scrub, scrub.value, scrub.max);
    time.textContent = timeText(current) + ' / ' + timeText(duration);
    requestTransitionEditorDraw();
    if (transportIsPlaying()) scheduleTransitionScrubber();
  }
  function seekLiveTransport(position, { restart=true } = {}) {
    const duration = activeTransportDuration();
    liveTransportPosition = clamp(Number(position) || 0, 0, Math.max(0, duration - .001));
    if (!restart || (!livePlayback && !liveSeekInFlight)) return;
    const token = (liveSeekPending ? liveSeekPending.token : 0) + 1;
    liveSeekPending = { position: liveTransportPosition, token };
    if (liveSeekInFlight || liveSeekFrame !== null) return;
    const queue = () => {
      liveSeekFrame = null;
      processLiveSeekQueue();
    };
    liveSeekFrame = window.requestAnimationFrame ? window.requestAnimationFrame(queue) : setTimeout(queue, 0);
  }
  async function processLiveSeekQueue() {
    if (liveSeekInFlight) return;
    liveSeekInFlight = true;
    try {
      while (liveSeekPending) {
        const request = liveSeekPending;
        stopLiveAudition({ clear: false, invalidateStart: true });
        await startLiveAudition({ position: request.position, restartToken: request.token });
        if (liveSeekPending && liveSeekPending.token === request.token) liveSeekPending = null;
      }
    } finally {
      liveSeekInFlight = false;
      updateTransitionScrubber();
    }
  }
  function cancelLiveSeekQueue() {
    liveSeekPending = null;
    livePlaybackStartRequestId += 1;
    if (liveSeekFrame !== null) {
      if (window.cancelAnimationFrame) window.cancelAnimationFrame(liveSeekFrame);
      else clearTimeout(liveSeekFrame);
      liveSeekFrame = null;
    }
  }
  async function toggleCentralTransport() {
    if (!transportCanPlay()) return;
    if (activeTransportMode() === 'live') {
      if (livePlayback || liveSeekInFlight) {
        cancelLiveSeekQueue();
        stopLiveAudition();
      } else {
        const duration = activeTransportDuration();
        if (!transitionLoopEnabled && liveTransportPosition >= duration - .002) liveTransportPosition = 0;
        await startLiveAudition({ position: liveTransportPosition });
      }
      updateTransitionScrubber();
      return;
    }
    const audio = transitionAudioElement();
    if (!audio) return;
    audio.loop = transitionLoopEnabled;
    if (audio.paused || audio.ended) {
      if (els.audio && !els.audio.paused) pauseSharedAudio();
      stopLiveAudition();
      if (!transitionLoopEnabled && audio.ended) audio.currentTime = 0;
      const promise = audio.play();
      if (promise && promise.catch) promise.catch(() => {});
    } else {
      audio.pause();
    }
    updateTransitionScrubber();
  }
  function bindTransitionScrubber() {
    const audio = transitionAudioElement();
    const button = document.getElementById('transition-play-toggle');
    const scrub = document.getElementById('transition-scrub');
    if (!button || !scrub) return;
    applyMasterVolume();
    if (audio) audio.loop = transitionLoopEnabled;
    if (!button.__seqBound) {
      button.__seqBound = true;
      button.addEventListener('click', () => { toggleCentralTransport(); });
    }
    const loop = document.getElementById('transition-loop-toggle');
    if (loop && !loop.__seqBound) {
      loop.__seqBound = true;
      loop.addEventListener('click', () => setTransitionLoopEnabled(!transitionLoopEnabled));
    }
    if (!scrub.__seqBound) {
      scrub.__seqBound = true;
      scrub.addEventListener('input', () => {
        transitionScrubDragging = true;
        const value = Number(scrub.value || 0);
        if (activeTransportMode() === 'live') {
          seekLiveTransport(value, { restart: true });
        } else {
          const activeAudio = transitionAudioElement();
          try { activeAudio.currentTime = value; } catch (err) {}
        }
        setRangeProgress(scrub, value, scrub.max);
        updateTransitionScrubber();
      });
      scrub.addEventListener('change', () => {
        transitionScrubDragging = false;
        const value = Number(scrub.value || 0);
        if (activeTransportMode() === 'live') {
          seekLiveTransport(value, { restart: true });
        } else {
          const activeAudio = transitionAudioElement();
          try { activeAudio.currentTime = value; } catch (err) {}
        }
        updateTransitionScrubber();
      });
      scrub.addEventListener('pointerup', () => { transitionScrubDragging = false; updateTransitionScrubber(); });
      scrub.addEventListener('pointercancel', () => { transitionScrubDragging = false; updateTransitionScrubber(); });
    }
    if (audio && !audio.__seqParentBound) {
      audio.__seqParentBound = true;
      ['play', 'pause', 'ended', 'timeupdate', 'loadedmetadata', 'seeked'].forEach(name => {
        audio.addEventListener(name, updateTransitionScrubber);
      });
    }
    updateTransitionScrubber();
  }
  function applyTransitionEditorDrag(ev) {
    if (!transitionEditorDrag) return;
    const info = transitionWindowInfo(transitionEditorDrag.deck);
    if (!info) return;
    if (transitionEditorDrag.deck === 'from' && transitionReferenceLocked) return;
    const dx = Number(ev.clientX) - transitionEditorDrag.startX;
    const secondsPerPixel = transitionEditorDrag.secondsPerPixel;
    const deltaSeconds = -dx * secondsPerPixel;
    let nudge = transitionEditorDrag.startNudge + (deltaSeconds / info.beatSec);
    nudge = Math.round(nudge * 8) / 8;
    nudge = clampTransitionNudge(transitionEditorDrag.deck, nudge);
    previewFormState[info.cfg.nudgeField] = nudge;
    syncPreviewFieldElements(info.cfg.nudgeField, nudge);
    drawTransitionEditor();
    updatePreviewEffectDisplay({ dirty: true });
  }
  function syncTransitionViewportControls(reference) {
    if (!reference) return;
    const viewport = transitionViewportState(reference);
    const bounds = transitionViewportDefaults(reference);
    const span = Math.max(1, bounds.end - bounds.start);
    const startPct = clamp((viewport.start - bounds.start) / span, 0, 1) * 100;
    const endPct = clamp((viewport.end - bounds.start) / span, 0, 1) * 100;
    document.querySelectorAll('[data-transition-viewport-track]').forEach(el => {
      el.style.setProperty('--viewport-start', startPct.toFixed(3) + '%');
      el.style.setProperty('--viewport-end', endPct.toFixed(3) + '%');
    });
  }
  function updateTransitionAlignment(deck, beats=0, beatgridMs=0, reset=false) {
    const info = transitionWindowInfo(deck);
    if (!info || (deck === 'from' && transitionReferenceLocked)) return;
    if (reset) {
      previewFormState[info.cfg.nudgeField] = 0;
      previewFormState[info.cfg.beatgridField] = 0;
    } else {
      previewFormState[info.cfg.nudgeField] = clampTransitionNudge(deck, (Number(previewFormState[info.cfg.nudgeField]) || 0) + Number(beats || 0));
      previewFormState[info.cfg.beatgridField] = clamp(Math.round(((Number(previewFormState[info.cfg.beatgridField]) || 0) + Number(beatgridMs || 0)) / 5) * 5, -250, 250);
    }
    syncPreviewFieldElements(info.cfg.nudgeField, previewFormState[info.cfg.nudgeField]);
    syncPreviewFieldElements(info.cfg.beatgridField, previewFormState[info.cfg.beatgridField]);
    updatePreviewEffectDisplay({ dirty: true });
  }
  function setTransitionBeatJumpSize(deck, value) {
    if (!transitionBeatJumpSizes[deck]) return;
    const target = Number(value);
    if (!BEAT_JUMP_SIZES.includes(target)) return;
    transitionBeatJumpSizes[deck] = target;
    renderTransitionPreview();
  }
  function stepTransitionBeatJumpSize(deck, direction) {
    const current = transitionBeatJumpSizes[deck] || 4;
    const index = Math.max(0, BEAT_JUMP_SIZES.indexOf(current));
    const next = clamp(index + (Number(direction) < 0 ? -1 : 1), 0, BEAT_JUMP_SIZES.length - 1);
    setTransitionBeatJumpSize(deck, BEAT_JUMP_SIZES[next]);
  }
  function applyTransitionBeatJump(deck, direction) {
    const info = transitionWindowInfo(deck);
    if (!info || (deck === 'from' && transitionReferenceLocked)) return;
    const beats = (transitionBeatJumpSizes[deck] || 4) * (Number(direction) < 0 ? -1 : 1);
    updateTransitionAlignment(deck, beats, 0);
    refreshLiveRoughOffsets();
    if (activeTransportMode() !== 'live' || (!livePlayback && !liveSeekInFlight)) return;
    if (liveRoughOffsetsAreSafe()) {
      liveTransitionMessage = 'Rough beat jump active · preparing HQ audio.';
      liveTransitionError = false;
      seekLiveTransport(activeTransportPosition(), { restart: true });
    } else {
      liveTransitionMessage = 'Beat jump exceeds prepared guard · preparing HQ audio.';
      liveTransitionError = false;
      updateLiveAuditionUi();
    }
  }
  function activeAutomationParts() {
    if (!transitionAutomationActiveLane) return null;
    const [deck, lane] = transitionAutomationActiveLane.split(':');
    return deck && lane ? { deck, lane } : null;
  }
  function snapAutomationBeat(value, info, bypass=false) {
    if (bypass || transitionAutomationSnapTime === 'off') return clamp(value, 0, transitionAutomationBeats());
    const grid = transitionAutomationSnapTime === 'quarter' ? .25 : transitionAutomationSnapTime === 'half' ? .5 : transitionAutomationSnapTime === 'bar' ? info.beatsPerBar : 1;
    return clamp(Math.round(value / grid) * grid, 0, transitionAutomationBeats());
  }
  function snapAutomationValue(value, lane, bypass=false) {
    const limits = automationLaneLimits(lane);
    if (bypass || transitionAutomationSnapValue === 'off') return clamp(value, limits[0], limits[1]);
    const step = lane === 'filter' ? .1 : (lane === 'delay' || lane === 'reverb' ? .05 : Number(transitionAutomationSnapValue) || 1);
    let snapped = Math.round(value / step) * step;
    if (lane !== 'filter' && lane !== 'delay' && lane !== 'reverb' && snapped < -93) snapped = -96;
    return clamp(snapped, limits[0], limits[1]);
  }
  function automationValueFromEvent(canvas, info, ev, laneName) {
    const rect = canvas.getBoundingClientRect();
    const surface = canvas.getAttribute('data-transition-surface') || 'connected';
    const lane = transitionLaneRect(Math.max(1, rect.width), Math.max(1, rect.height), surface);
    const reference = transitionWindowInfo('from') || info;
    const view = surface === 'connected' ? transitionSharedView(info, reference) : transitionVisibleRange(info, surface);
    const x = clamp(Number(ev.clientX) - rect.left, lane.x, lane.x + lane.w);
    const y = clamp(Number(ev.clientY) - rect.top, lane.y, lane.y + lane.h);
    const time = view.start + ((x - lane.x) / Math.max(1, lane.w)) * view.duration;
    return {
      beat: snapAutomationBeat((time - info.transitionStart) / info.beatSec, info, !!ev.altKey),
      value: snapAutomationValue(automationValueFromScalePosition(laneName, 1 - ((y - lane.y) / Math.max(1, lane.h))), laneName, !!ev.altKey),
      lane,
      view,
    };
  }
  function findAutomationPointAtEvent(canvas, info, ev, deck, laneName) {
    const points = automationLane(deck, laneName) || [];
    const value = automationValueFromEvent(canvas, info, ev, laneName);
    let match = null;
    let distance = Infinity;
    points.forEach((point, index) => {
      const xy = automationPointXY(info, value.view, value.lane, laneName, point);
      const d = Math.hypot(xy.x - (Number(ev.clientX) - canvas.getBoundingClientRect().left), xy.y - (Number(ev.clientY) - canvas.getBoundingClientRect().top));
      if (d < distance) { distance = d; match = index; }
    });
    return distance <= 12 ? match : null;
  }
  function findAutomationSegmentAtEvent(canvas, info, ev, deck, laneName) {
    const points = automationLane(deck, laneName) || [];
    if (points.length < 2) return null;
    const value = automationValueFromEvent(canvas, info, ev, laneName);
    const pointerX = Number(ev.clientX) - canvas.getBoundingClientRect().left;
    const pointerY = Number(ev.clientY) - canvas.getBoundingClientRect().top;
    let match = null;
    let distance = Infinity;
    for (let index = 0; index < points.length - 1; index += 1) {
      const startBeat = Number(points[index].beat);
      const endBeat = Number(points[index + 1].beat);
      if (Math.abs(endBeat - startBeat) < .000001) {
        const x = automationPointXY(info, value.view, value.lane, laneName, points[index]).x;
        const y0 = automationValueY(laneName, points[index].value, value.lane);
        const y1 = automationValueY(laneName, points[index + 1].value, value.lane);
        const y = clamp(pointerY, Math.min(y0, y1), Math.max(y0, y1));
        const d = Math.hypot(pointerX - x, pointerY - y);
        if (d < distance) { distance = d; match = index; }
        continue;
      }
      let previous = null;
      for (let step = 0; step <= 20; step += 1) {
        const beat = startBeat + ((step / 20) * (endBeat - startBeat));
          const xy = automationPointXY(info, value.view, value.lane, laneName, { beat, value: automationValueAt(points, beat, laneName) });
        if (previous) {
          const dx = xy.x - previous.x;
          const dy = xy.y - previous.y;
          const denom = Math.max(.000001, (dx * dx) + (dy * dy));
          const t = clamp((((pointerX - previous.x) * dx) + ((pointerY - previous.y) * dy)) / denom, 0, 1);
          const d = Math.hypot(pointerX - (previous.x + t * dx), pointerY - (previous.y + t * dy));
          if (d < distance) { distance = d; match = index; }
        }
        previous = xy;
      }
    }
    return distance <= 9 ? match : null;
  }
  function deleteSelectedAutomationPoint() {
    const selected = transitionAutomationSelectedPoint;
    if (!selected) return false;
    const points = automationLane(selected.deck, selected.lane);
    if (!points || selected.index <= 0 || selected.index >= points.length - 1) return false;
    points.splice(selected.index, 1);
    transitionAutomationSelectedPoint = null;
    transitionAutomationSelectedSegment = null;
    markTransitionAutomationEdited();
    return true;
  }
  function addAutomationPointFromEvent(canvas, info, ev) {
    const deck = canvas.getAttribute('data-transition-deck');
    const active = activeAutomationParts();
    if (!active || active.deck !== deck || canvas.getAttribute('data-transition-surface') !== 'connected') return false;
    const points = automationLane(deck, active.lane);
    if (!points || findAutomationPointAtEvent(canvas, info, ev, deck, active.lane) !== null) return false;
    const next = automationValueFromEvent(canvas, info, ev, active.lane);
    const total = transitionAutomationBeats();
    if (next.beat <= .001 || next.beat >= total - .001) return false;
    points.push({ beat: next.beat, value: next.value, curve: 'linear' });
    transitionAutomation[deck][active.lane] = normalizeAutomationPoints(points, active.lane);
    const normalized = transitionAutomation[deck][active.lane];
    let index = 0;
    let distance = Infinity;
    normalized.forEach((point, candidate) => {
      const candidateDistance = Math.abs(Number(point.beat) - next.beat) + Math.abs(Number(point.value) - next.value);
      if (candidateDistance < distance) { distance = candidateDistance; index = candidate; }
    });
    setAutomationPointSelection(deck, active.lane, index);
    markTransitionAutomationEdited();
    return true;
  }
  function sameAutomationBeat(a, b) {
    return Math.abs(Number(a) - Number(b)) < .000001;
  }
  function dragAutomationPoint(points, drag, next) {
    let index = drag.index;
    const point = points[index];
    if (!point) return index;
    point.value = next.value;
    if (index <= 0 || index >= points.length - 1) return index;
    const currentBeat = Number(point.beat);
    const pairedBefore = sameAutomationBeat(currentBeat, points[index - 1].beat);
    const pairedAfter = sameAutomationBeat(currentBeat, points[index + 1].beat);
    if (pairedAfter) {
      // The dragged point is the before side of a jump. Moving it left resolves
      // the pair; moving it right remains clamped against its after value.
      if (next.beat < currentBeat - .000001) {
        const min = Number(points[index - 1].beat) + .001;
        point.beat = clamp(next.beat, min, currentBeat - .001);
      } else point.beat = currentBeat;
      return index;
    }
    if (pairedBefore) {
      // The dragged point is the after side of a jump. Moving it right resolves
      // the pair; moving it left remains clamped against its before value.
      if (next.beat > currentBeat + .000001) {
        const max = Number(points[index + 1].beat) - .001;
        point.beat = clamp(next.beat, currentBeat + .001, max);
      } else point.beat = currentBeat;
      return index;
    }
    const left = points[index - 1];
    const right = points[index + 1];
    if (next.beat <= Number(left.beat) + .000001) {
      if (index - 1 === 0) point.beat = Number(left.beat) + .001;
      else {
        point.beat = Number(left.beat);
        // A point arriving from the right replaces a jump's after side.
        if (index - 2 >= 0 && sameAutomationBeat(points[index - 2].beat, left.beat)) {
          points.splice(index - 1, 1);
          index -= 1;
        }
      }
    } else if (next.beat >= Number(right.beat) - .000001) {
      if (index + 1 === points.length - 1) point.beat = Number(right.beat) - .001;
      else {
        point.beat = Number(right.beat);
        // A point arriving from the left replaces a jump's before side.
        if (index + 2 < points.length && sameAutomationBeat(right.beat, points[index + 2].beat)) points.splice(index + 1, 1);
      }
    } else {
      point.beat = next.beat;
    }
    return index;
  }
  function bindTransitionEditor() {
    document.querySelectorAll('[data-automation-lane]').forEach(button => {
      if (button.__seqBound) return;
      button.__seqBound = true;
      button.addEventListener('click', () => {
        const lane = String(button.getAttribute('data-automation-lane') || '');
        transitionAutomationActiveLane = transitionAutomationActiveLane === lane ? null : lane;
        transitionAutomationSelectedPoint = null;
        transitionAutomationSelectedSegment = null;
        updatePreviewEffectDisplay({ dirty: false });
      });
    });
    document.querySelectorAll('[data-automation-snap]').forEach(select => {
      if (select.__seqBound) return;
      select.__seqBound = true;
      select.addEventListener('change', () => {
        if (select.getAttribute('data-automation-snap') === 'time') transitionAutomationSnapTime = select.value;
        else transitionAutomationSnapValue = select.value;
      });
    });
    const curveSelect = document.querySelector('[data-automation-curve]');
    syncAutomationEditorControls();
    if (curveSelect && !curveSelect.__seqBound) {
      curveSelect.__seqBound = true;
      curveSelect.addEventListener('change', () => {
        const selected = selectedAutomationSegmentParts();
        if (!selected) return;
        selected.start.curve = curveSelect.value;
        markTransitionAutomationEdited();
      });
    }
    document.querySelectorAll('[data-automation-segment-field]').forEach(input => {
      if (input.__seqBound) return;
      input.__seqBound = true;
      input.addEventListener('change', () => {
        const selected = selectedAutomationSegmentParts();
        if (!selected) return;
        const [endpoint, field] = String(input.getAttribute('data-automation-segment-field')).split(':');
        const point = endpoint === 'start' ? selected.start : selected.end;
        const pointIndex = selected.index + (endpoint === 'end' ? 1 : 0);
        const raw = Number(input.value);
        if (!Number.isFinite(raw)) { syncAutomationEditorControls(); return; }
        if (field === 'value') point.value = snapAutomationValue(raw, selected.lane);
        else if (Math.abs(Number(selected.start.beat) - Number(selected.end.beat)) < .000001) {
          const info = transitionWindowInfo(selected.deck);
          const min = Number(selected.points[selected.index - 1].beat) + .001;
          const max = Number(selected.points[selected.index + 2].beat) - .001;
          const beat = clamp(snapAutomationBeat(raw, info), min, max);
          selected.start.beat = beat;
          selected.end.beat = beat;
        } else if (pointIndex > 0 && pointIndex < selected.points.length - 1) {
          const info = transitionWindowInfo(selected.deck);
          const min = Number(selected.points[pointIndex - 1].beat) + .001;
          const max = Number(selected.points[pointIndex + 1].beat) - .001;
          point.beat = clamp(snapAutomationBeat(raw, info), min, max);
        }
        transitionAutomation[selected.deck][selected.lane] = normalizeAutomationPoints(selected.points, selected.lane);
        markTransitionAutomationEdited();
      });
    });
    document.querySelectorAll('[data-automation-action]').forEach(button => {
      if (button.__seqBound) return;
      button.__seqBound = true;
      button.addEventListener('click', () => {
        const action = button.getAttribute('data-automation-action');
        const active = activeAutomationParts();
        if (action === 'restore') {
          previewFormState.preset = transitionAutomationPreset || 'auto';
          transitionAutomationSelectedPoint = null;
          transitionAutomationSelectedSegment = null;
          ensureTransitionAutomation({ reset: true });
          snapshotTransitionAutomation({ resetHistory: true });
          renderTransitionPreview();
          return;
        }
        if (action === 'delete-segment-keyframe') {
          const selected = selectedAutomationSegmentParts();
          if (!selected || selected.index + 1 >= selected.points.length - 1) return;
          selected.points.splice(selected.index + 1, 1);
          transitionAutomation[selected.deck][selected.lane] = normalizeAutomationPoints(selected.points, selected.lane);
          transitionAutomationSelectedPoint = null;
          transitionAutomationSelectedSegment = { deck: selected.deck, lane: selected.lane, index: Math.min(selected.index, selected.points.length - 2) };
          markTransitionAutomationEdited();
          return;
        }
        if (!active) return;
        const points = automationLane(active.deck, active.lane);
        if (!points) return;
        if (action === 'reset-lane') {
          const template = presetAutomationTemplate(transitionAutomationPreset || 'auto');
          transitionAutomation[active.deck][active.lane] = cloneAutomation(template[active.deck][active.lane]);
        } else if (action === 'copy-lane') {
          const other = active.deck === 'from' ? 'to' : 'from';
          transitionAutomation[other][active.lane] = cloneAutomation(points);
        } else if (action === 'mirror-lane') {
          const total = transitionAutomationBeats();
          transitionAutomation[active.deck][active.lane] = points.map(point => ({ ...point, beat: total - Number(point.beat) })).reverse().map((point, index, list) => ({ ...point, curve: index < list.length - 1 ? point.curve : 'linear' }));
        }
        transitionAutomationSelectedPoint = null;
        transitionAutomationSelectedSegment = null;
        markTransitionAutomationEdited();
      });
    });
    const snap = document.getElementById('transition-snap-to-beat');
    if (snap && !snap.__seqBound) {
      snap.__seqBound = true;
      snap.addEventListener('change', () => {
        transitionSnapToBeat = !!snap.checked;
        drawTransitionEditor();
      });
    }
    const lock = document.getElementById('transition-lock-track-one');
    if (lock && !lock.__seqBound) {
      lock.__seqBound = true;
      lock.addEventListener('change', () => {
        transitionReferenceLocked = !!lock.checked;
        renderTransitionPreview();
      });
    }
    document.querySelectorAll('[data-transition-adjust]').forEach(button => {
      if (button.__seqBound) return;
      button.__seqBound = true;
      button.addEventListener('click', () => {
        const [deck, beats] = String(button.getAttribute('data-transition-adjust') || '').split(':');
        updateTransitionAlignment(deck, Number(beats) || 0, 0);
      });
    });
    document.querySelectorAll('[data-transition-beat-size]').forEach(button => {
      if (button.__seqBound) return;
      button.__seqBound = true;
      button.addEventListener('click', () => {
        const [deck, size] = String(button.getAttribute('data-transition-beat-size') || '').split(':');
        setTransitionBeatJumpSize(deck, Number(size));
      });
    });
    document.querySelectorAll('[data-transition-beat-size-step]').forEach(button => {
      if (button.__seqBound) return;
      button.__seqBound = true;
      button.addEventListener('click', () => {
        const [deck, direction] = String(button.getAttribute('data-transition-beat-size-step') || '').split(':');
        stepTransitionBeatJumpSize(deck, Number(direction));
      });
    });
    document.querySelectorAll('[data-transition-beat-jump]').forEach(button => {
      if (button.__seqBound) return;
      button.__seqBound = true;
      button.addEventListener('click', () => {
        const [deck, direction] = String(button.getAttribute('data-transition-beat-jump') || '').split(':');
        applyTransitionBeatJump(deck, Number(direction));
      });
    });
    document.querySelectorAll('[data-transition-beatgrid-adjust]').forEach(button => {
      if (button.__seqBound) return;
      button.__seqBound = true;
      button.addEventListener('click', () => {
        const [deck, milliseconds] = String(button.getAttribute('data-transition-beatgrid-adjust') || '').split(':');
        updateTransitionAlignment(deck, 0, Number(milliseconds) || 0);
      });
    });
    document.querySelectorAll('[data-transition-reset-align]').forEach(button => {
      if (button.__seqBound) return;
      button.__seqBound = true;
      button.addEventListener('click', () => updateTransitionAlignment(button.getAttribute('data-transition-reset-align'), 0, 0, true));
    });
    const reference = transitionWindowInfo('from');
    syncTransitionViewportControls(reference);
    document.querySelectorAll('[data-transition-viewport-track]').forEach(track => {
      if (track.__seqBound) return;
      track.__seqBound = true;
      let drag = null;
      const update = ev => {
        if (!drag) return;
        const currentReference = transitionWindowInfo('from');
        if (!currentReference) return;
        const bounds = transitionViewportDefaults(currentReference);
        const minSpan = 8;
        const delta = ((Number(ev.clientX) - drag.startX) / Math.max(1, drag.width)) * (bounds.end - bounds.start);
        const next = { start: drag.start, end: drag.end };
        if (drag.mode === 'start') next.start = clamp(drag.start + delta, bounds.start, drag.end - minSpan);
        else if (drag.mode === 'end') next.end = clamp(drag.end + delta, drag.start + minSpan, bounds.end);
        else {
          const span = drag.end - drag.start;
          next.start = clamp(drag.start + delta, bounds.start, bounds.end - span);
          next.end = next.start + span;
        }
        transitionViewport = next;
        syncTransitionViewportControls(currentReference);
        drawTransitionEditor();
      };
      track.addEventListener('pointerdown', ev => {
        if (ev.button !== 0) return;
        const currentReference = transitionWindowInfo('from');
        if (!currentReference) return;
        const viewport = transitionViewportState(currentReference);
        const rect = track.getBoundingClientRect();
        const handle = closestAttr(ev.target, 'data-transition-viewport-handle');
        const xPct = clamp((Number(ev.clientX) - rect.left) / Math.max(1, rect.width), 0, 1);
        const bounds = transitionViewportDefaults(currentReference);
        const point = bounds.start + xPct * (bounds.end - bounds.start);
        const mode = handle || (point >= viewport.start && point <= viewport.end ? 'pan' : (Math.abs(point - viewport.start) <= Math.abs(point - viewport.end) ? 'start' : 'end'));
        drag = { mode, startX: Number(ev.clientX), start: viewport.start, end: viewport.end, width: rect.width };
        try { track.setPointerCapture(ev.pointerId); } catch (err) {}
        ev.preventDefault();
      });
      track.addEventListener('pointermove', update);
      const end = ev => {
        if (!drag) return;
        update(ev);
        drag = null;
        try { track.releasePointerCapture(ev.pointerId); } catch (err) {}
      };
      track.addEventListener('pointerup', end);
      track.addEventListener('pointercancel', () => { drag = null; });
    });
    document.querySelectorAll('[data-transition-viewport-reset]').forEach(button => {
      if (button.__seqBound) return;
      button.__seqBound = true;
      button.addEventListener('click', () => {
        transitionViewport = null;
        drawTransitionEditor();
      });
    });
    document.querySelectorAll('[data-transition-viewport-zoom]').forEach(button => {
      if (button.__seqBound) return;
      button.__seqBound = true;
      button.addEventListener('click', () => {
        const currentReference = transitionWindowInfo('from');
        if (!currentReference) return;
        const bounds = transitionViewportDefaults(currentReference);
        const current = transitionViewportState(currentReference);
        const center = (current.start + current.end) / 2;
        const minSpan = 8;
        const fullSpan = bounds.end - bounds.start;
        const factor = button.getAttribute('data-transition-viewport-zoom') === 'in' ? 0.72 : 1.38;
        const span = clamp((current.end - current.start) * factor, minSpan, fullSpan);
        const start = clamp(center - span / 2, bounds.start, bounds.end - span);
        transitionViewport = { start, end: start + span };
        syncTransitionViewportControls(currentReference);
        drawTransitionEditor();
      });
    });
    document.querySelectorAll('[data-transition-surface][data-transition-deck]').forEach(canvas => {
      if (canvas.__seqBound) return;
      canvas.__seqBound = true;
      canvas.addEventListener('pointerdown', ev => {
        if (ev.button !== 0) return;
        const deck = canvas.getAttribute('data-transition-deck');
        const info = transitionWindowInfo(deck);
        if (!info) return;
        const active = activeAutomationParts();
        const surface = canvas.getAttribute('data-transition-surface');
        if (active && active.deck === deck && surface === 'connected') {
          const points = automationLane(deck, active.lane);
          if (!points) return;
          const index = findAutomationPointAtEvent(canvas, info, ev, deck, active.lane);
          if (index !== null) {
            setAutomationPointSelection(deck, active.lane, index);
            transitionAutomationDrag = { deck, lane: active.lane, index };
            try { canvas.setPointerCapture(ev.pointerId); } catch (err) {}
            updatePreviewEffectDisplay({ dirty: false });
            ev.preventDefault();
            ev.stopPropagation();
            return;
          }
          const segment = findAutomationSegmentAtEvent(canvas, info, ev, deck, active.lane);
          if (segment !== null) {
            setAutomationSegmentSelection(deck, active.lane, segment);
            updatePreviewEffectDisplay({ dirty: false });
            ev.preventDefault();
            ev.stopPropagation();
            return;
          }
          // Leave empty automation-lane clicks alone so the browser can emit
          // the subsequent dblclick event that inserts a new automation point.
          return;
        }
        if (deck === 'from' && transitionReferenceLocked) return;
        const cue = transitionCueAtPoint(canvas, info, ev);
        if (cue) {
          previewFormState[info.cfg.cueField] = cue.name || '';
          syncPreviewFieldElements(info.cfg.cueField, previewFormState[info.cfg.cueField]);
          drawTransitionEditor();
          updatePreviewEffectDisplay({ dirty: true });
          ev.preventDefault();
          ev.stopPropagation();
          return;
        }
        const rect = canvas.getBoundingClientRect();
        const canvasSurface = canvas.getAttribute('data-transition-surface');
        const currentReference = transitionWindowInfo('from') || info;
        const view = canvasSurface === 'connected' ? transitionSharedView(info, currentReference) : transitionVisibleRange(info, canvasSurface);
        const seconds = view.duration;
        transitionEditorDrag = {
          deck,
          startX: Number(ev.clientX),
          startNudge: Number(previewFormState[info.cfg.nudgeField]) || 0,
          secondsPerPixel: seconds / Math.max(1, rect.width),
        };
        try { canvas.setPointerCapture(ev.pointerId); } catch (err) {}
        ev.preventDefault();
        ev.stopPropagation();
      });
      canvas.addEventListener('dblclick', ev => {
        const deck = canvas.getAttribute('data-transition-deck');
        const info = transitionWindowInfo(deck);
        if (!info || !addAutomationPointFromEvent(canvas, info, ev)) return;
        ev.preventDefault();
        ev.stopPropagation();
      });
      canvas.addEventListener('pointermove', ev => {
        if (transitionAutomationDrag && transitionAutomationDrag.deck === canvas.getAttribute('data-transition-deck')) {
          const info = transitionWindowInfo(transitionAutomationDrag.deck);
          const points = info && automationLane(transitionAutomationDrag.deck, transitionAutomationDrag.lane);
          if (!info || !points) return;
          const next = automationValueFromEvent(canvas, info, ev, transitionAutomationDrag.lane);
          transitionAutomationDrag.index = dragAutomationPoint(points, transitionAutomationDrag, next);
          setAutomationPointSelection(transitionAutomationDrag.deck, transitionAutomationDrag.lane, transitionAutomationDrag.index);
          previewFormState.preset = 'custom';
          const presetEl = document.getElementById('preset');
          if (presetEl) presetEl.value = 'custom';
          previewFormState.automation = transitionAutomation;
          requestLiveAutomationReschedule();
          syncAutomationEditorControls();
          drawTransitionEditor();
          updateTransitionDirtyUi();
          ev.preventDefault();
          return;
        }
        if (!transitionEditorDrag || transitionEditorDrag.deck !== canvas.getAttribute('data-transition-deck')) return;
        applyTransitionEditorDrag(ev);
        ev.preventDefault();
      });
      const endDrag = ev => {
        if (transitionAutomationDrag && transitionAutomationDrag.deck === canvas.getAttribute('data-transition-deck')) {
          transitionAutomationDrag = null;
          snapshotTransitionAutomation();
          updatePreviewEffectDisplay({ dirty: true });
          try { canvas.releasePointerCapture(ev.pointerId); } catch (err) {}
          return;
        }
        if (!transitionEditorDrag) return;
        applyTransitionEditorDrag(ev);
        transitionEditorDrag = null;
        try { canvas.releasePointerCapture(ev.pointerId); } catch (err) {}
        updateTransitionDirtyUi();
      };
      canvas.addEventListener('pointerup', endDrag);
      canvas.addEventListener('pointercancel', () => { transitionEditorDrag = null; transitionAutomationDrag = null; });
    });
    drawTransitionEditor();
  }
  function bindTransitionPreviewOverlay() {
    const renderButton = document.getElementById('transition-render-button');
    if (renderButton) {
      renderButton.onclick = window.__djRenderTransition;
    }
    setTimeout(() => safeUi('transition editor bind', bindTransitionEditor), 0);
    setTimeout(() => safeUi('transition scrubber bind', bindTransitionScrubber), 0);
    setTimeout(() => safeUi('live mixer bind', bindLiveMixer), 0);
    setTimeout(() => safeUi('transition editor draw', drawTransitionEditor), 0);
    setTimeout(() => safeUi('transition dirty ui update', updateTransitionDirtyUi), 0);
  }
  function bindLiveMixer() {
    document.querySelectorAll('[data-live-mixer-control]').forEach(input => {
      if (input.__seqBound) return;
      input.__seqBound = true;
      input.addEventListener('input', () => {
        const [deck, lane] = String(input.getAttribute('data-live-mixer-control') || '').split(':');
        setLiveManualOverride(deck, lane, input.value);
      });
      input.addEventListener('dblclick', ev => {
        ev.preventDefault();
        const [deck, lane] = String(input.getAttribute('data-live-mixer-control') || '').split(':');
        setLiveManualOverride(deck, lane, 0);
      });
    });
    document.querySelectorAll('[data-live-mixer-auto]').forEach(button => {
      if (button.__seqBound) return;
      button.__seqBound = true;
      button.addEventListener('click', () => {
        const [deck, lane] = String(button.getAttribute('data-live-mixer-auto') || '').split(':');
        restoreLiveLaneAutomation(deck, lane);
      });
    });
    document.querySelectorAll('[data-live-fx-setting]').forEach(input => {
      if (input.__seqBound) return;
      input.__seqBound = true;
      const update = () => {
        const [deck, setting] = String(input.getAttribute('data-live-fx-setting') || '').split(':');
        setLiveFxSetting(deck, setting, input.value);
      };
      input.addEventListener('input', update);
      input.addEventListener('change', update);
      input.addEventListener('dblclick', ev => {
        const [deck, setting] = String(input.getAttribute('data-live-fx-setting') || '').split(':');
        const defaults = { delay_tone: .5, reverb_decay: 1.2, reverb_tone: .5 };
        if (!Object.prototype.hasOwnProperty.call(defaults, setting)) return;
        ev.preventDefault();
        setLiveFxSetting(deck, setting, defaults[setting]);
      });
    });
    document.querySelectorAll('[data-live-crossfader]').forEach(input => {
      if (input.__seqBound) return;
      input.__seqBound = true;
      input.addEventListener('input', () => {
        liveCrossfader = clamp(Number(input.value) || 0, -1, 1);
        applyLiveCrossfader();
      });
      input.addEventListener('dblclick', ev => {
        ev.preventDefault();
        liveCrossfader = 0;
        applyLiveCrossfader();
        syncLiveMixerUi();
      });
    });
    document.querySelectorAll('[data-live-crossfader-reset]').forEach(button => {
      if (button.__seqBound) return;
      button.__seqBound = true;
      button.addEventListener('click', () => {
        liveCrossfader = 0;
        applyLiveCrossfader();
        syncLiveMixerUi();
      });
    });
    document.querySelectorAll('[data-live-mixer-auto-all]').forEach(button => {
      if (button.__seqBound) return;
      button.__seqBound = true;
      button.addEventListener('click', () => toggleAllLiveAutomation());
    });
    document.querySelectorAll('[data-live-mixer-toggle]').forEach(button => {
      if (button.__seqBound) return;
      button.__seqBound = true;
      button.addEventListener('click', () => {
        liveMixerOpen = !liveMixerOpen;
        renderTransitionPreview();
      });
    });
    bindLiveRotaryControls();
    syncLiveMixerUi();
  }
  function bindLiveRotaryControls() {
    document.querySelectorAll('.live-mixer-knob input[type="range"]').forEach(input => {
      const knob = input.closest('.live-mixer-knob');
      if (!knob || knob.__seqRotaryBound) return;
      knob.__seqRotaryBound = true;
      let drag = null;
      const finish = ev => {
        if (!drag) return;
        drag = null;
        try { knob.releasePointerCapture(ev.pointerId); } catch (err) {}
      };
      knob.addEventListener('pointerdown', ev => {
        if (ev.button !== 0) return;
        const min = Number(input.min);
        const max = Number(input.max);
        const value = Number(input.value);
        if (!Number.isFinite(min) || !Number.isFinite(max) || !Number.isFinite(value) || max <= min) return;
        ev.preventDefault();
        try { input.focus({ preventScroll: true }); } catch (err) { input.focus(); }
        drag = { x: Number(ev.clientX), y: Number(ev.clientY), value, min, max, step: Math.max(.0001, Number(input.step) || .01) };
        try { knob.setPointerCapture(ev.pointerId); } catch (err) {}
      }, { capture: true });
      knob.addEventListener('pointermove', ev => {
        if (!drag) return;
        // Up and right both increase, while a short movement remains precise.
        const travel = ((Number(ev.clientX) - drag.x) * .5) - (Number(ev.clientY) - drag.y);
        const raw = drag.value + ((travel / LIVE_KNOB_TRAVEL_PX) * (drag.max - drag.min));
        const snapped = drag.min + (Math.round((raw - drag.min) / drag.step) * drag.step);
        input.value = String(clamp(snapped, drag.min, drag.max));
        input.dispatchEvent(new Event('input', { bubbles: true }));
        ev.preventDefault();
      });
      knob.addEventListener('pointerup', finish);
      knob.addEventListener('pointercancel', finish);
      knob.addEventListener('dblclick', ev => {
        const mixerControl = String(input.getAttribute('data-live-mixer-control') || '');
        const fxControl = String(input.getAttribute('data-live-fx-setting') || '');
        if (mixerControl) {
          const [deck, lane] = mixerControl.split(':');
          setLiveManualOverride(deck, lane, 0);
        } else if (fxControl) {
          const [deck, setting] = fxControl.split(':');
          const defaults = { delay_tone: .5, reverb_decay: 1.2, reverb_tone: .5 };
          if (Object.prototype.hasOwnProperty.call(defaults, setting)) setLiveFxSetting(deck, setting, defaults[setting]);
        }
        ev.preventDefault();
      });
    });
  }
  function updatePreviewFormStateFromElement(el) {
    const field = el && el.getAttribute && el.getAttribute('data-preview-field');
    if (!field) return;
    if (field === 'preset') {
      if (String(el.value) !== String(previewFormState.preset) || !transitionAutomation) applyPresetToPreviewState(el.value);
      updateTransitionDirtyUi();
      return;
    }
    const previousValue = previewFormState[field];
    if (el.type === 'number') previewFormState[field] = Number(el.value);
    else previewFormState[field] = el.value;
    if (field === 'padding_bars') {
      setTransitionPadding(previewFormState[field]);
      if (el) el.value = String(previewFormState.padding_bars);
      return;
    }
    if (TRANSITION_TIMING_STEPS[field]) previewFormState[field] = nearestTransitionTimingValue(field, previewFormState[field]);
    if (field === 'from_nudge_beats') previewFormState[field] = clampTransitionNudge('from', previewFormState[field]);
    if (field === 'to_nudge_beats') previewFormState[field] = clampTransitionNudge('to', previewFormState[field]);
    if (field === 'from_beatgrid_ms' || field === 'to_beatgrid_ms') previewFormState[field] = clamp(Math.round((Number(previewFormState[field]) || 0) / 5) * 5, -250, 250);
    if ((field === 'from_nudge_beats' || field === 'to_nudge_beats') && el) el.value = String(previewFormState[field]);
    if ((field === 'from_beatgrid_ms' || field === 'to_beatgrid_ms') && el) el.value = String(previewFormState[field]);
    if (TRANSITION_TIMING_STEPS[field] && el) el.value = String(previewFormState[field]);
    syncPreviewFieldElements(field, previewFormState[field], el);
    if (['volume_mode', 'eq_mode', 'filter_mode'].includes(field)) {
      if (String(previousValue) !== String(previewFormState[field])) applyLegacyModeAutomation(field);
      return;
    }
    drawTransitionEditor();
  }
  function syncPreviewFormStateFromDom() {
    document.querySelectorAll('[data-preview-field]').forEach(updatePreviewFormStateFromElement);
  }
  async function renderBackendTransition() {
    if (transitionRenderPending) return;
    stopLiveAudition();
    const from = transitionFromIdx === null ? null : byIdx.get(transitionFromIdx);
    const to = transitionToIdx === null ? null : byIdx.get(transitionToIdx);
    if (!from || !to) {
      transitionRenderMessage = 'Assign Track 1 and Track 2 before rendering.';
      transitionRenderWarn = true;
      renderTransitionPreview();
      return;
    }
    syncPreviewFormStateFromDom();
    ensureTransitionAutomation();
    const body = Object.assign({}, previewFormState, {
      from_track: trackId(from),
      to_track: trackId(to),
      loudness_match_mode: loudnessMatchMode(),
      automation: cloneAutomation(transitionAutomation),
      effects: cloneAutomation(previewFormState.effects),
      overwrite: false,
    });
    const requestId = transitionRenderRequestId + 1;
    transitionRenderRequestId = requestId;
    transitionRenderPending = true;
    transitionRenderMessage = 'Rendering transition...';
    transitionRenderWarn = false;
    showTransitionPendingUi();
    try {
      const res = await fetch('/api/render-transition', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (requestId !== transitionRenderRequestId) return;
      if (!data.ok) throw new Error(data.error || 'Render failed');
      data.rendered_at = Date.now();
      transitionRender = data;
      transitionRenderedState = Object.assign({}, body, { effects: cloneAutomation(body.effects) });
      transitionRenderMessage = 'Rendered preview is current.';
      transitionRenderWarn = false;
    } catch (err) {
      if (requestId !== transitionRenderRequestId) return;
      transitionRenderMessage = err.message || String(err);
      transitionRenderWarn = true;
    } finally {
      if (requestId === transitionRenderRequestId) {
        transitionRenderPending = false;
        renderTransitionPreview();
      }
    }
  }
  window.__djRenderTransition = function(ev) {
    if (ev) {
      ev.preventDefault();
      ev.stopPropagation();
    }
    renderBackendTransition();
    return false;
  };
  function energyValueFromMouse(ev) {
    const gd = els.energyCurve;
    if (!gd || !gd._fullLayout || !gd._fullLayout._size) return null;
    const rect = gd.getBoundingClientRect();
    const size = gd._fullLayout._size;
    const xPixel = ev.clientX - rect.left - size.l;
    const yPixel = ev.clientY - rect.top - size.t;
    if (xPixel < 0 || yPixel < 0 || xPixel > size.w || yPixel > size.h) return null;
    const xValue = gd._fullLayout.xaxis.p2d(xPixel);
    const yValue = gd._fullLayout.yaxis.p2d(yPixel);
    const slot = clamp(Math.round(Number(xValue)) - 1, 0, sequence.length - 1);
    const value = Math.round(clamp(Number(yValue), 1, 9) * 10) / 10;
    if (!Number.isFinite(slot) || !Number.isFinite(value)) return null;
    return { slot, value };
  }
  function beginEnergyDrag(ev) {
    if (ev.button !== 0) return;
    const point = energyValueFromMouse(ev);
    if (!point) return;
    planningEditSnapshot = planningSnapshot();
    draggingEnergySlot = point.slot;
    setTargetSlotValue(point.slot, point.value, true);
    ev.preventDefault();
  }
  function updateEnergyDrag(ev) {
    if (draggingEnergySlot === null) return;
    const point = energyValueFromMouse(ev);
    if (!point) return;
    setTargetSlotValue(draggingEnergySlot, point.value, true);
  }
  function endEnergyDrag() {
    if (draggingEnergySlot !== null && planningEditSnapshot) commitPlanningChange(planningEditSnapshot);
    draggingEnergySlot = null;
    planningEditSnapshot = null;
  }
  function updateAppendControls() {
    const slot = targetSlot();
    const replacing = slot >= 0 && sequence[slot] !== null;
    setActionButton(els.appendSelected, replacing ? '⇄' : '+', replacing ? 'Replace selected slot' : 'Append selected track', 'primary');
    els.appendSelected.disabled = selectedIdx === null || !canPlaceTrack(selectedIdx, slot);
  }
  function renderSelectionOnly() {
    safeUi('map overlays', updatePacmapOverlays);
    safeUi('map static overlays', ensureMapEffectsLoop);
    safeUi('append controls', updateAppendControls);
    if (activePane === 'explore') {
      safeUi('sequence render', renderSequence);
      safeUi('recommendations render', renderRecommendations);
    }
    if (activePane === 'library') safeUi('library render', renderLibrary);
    if (activePane === 'diagnostics') {
      safeUi('transition score render', renderCurrentTransitionScore);
      safeUi('transition diagnostics render', renderTransitionDiagnostics);
    }
    safeUi('transition badges render', renderTransitionBadges);
    safeUi('audio ui sync', syncAudioUi);
    if (els.settingsPopover && !els.settingsPopover.classList.contains('hidden')) {
      safeUi('settings render', renderSettingsPanel);
    }
  }
  function renderAll() {
    safeUi('weight labels', updateWeightLabels);
    safeUi('map coordinates', () => {
      syncMapRenderer({ render: false });
      const previousPoints = currentPoints;
      currentPoints = layoutInterpolatedPoints(weights());
      recordMapTrails(previousPoints, currentPoints);
      updatePointCoordinates();
      applyPointColorMode();
      updatePacmapOverlays();
      ensureMapEffectsLoop();
      drawMiniMap();
    });
    safeUi('append controls', updateAppendControls);
    if (activePane === 'explore') {
      safeUi('sequence render', renderSequence);
      safeUi('recommendations render', renderRecommendations);
      safeUi('energy curve render', renderEnergyCurve);
    }
    if (activePane === 'library') safeUi('library render', renderLibrary);
    if (activePane === 'diagnostics') {
      safeUi('transition score render', renderCurrentTransitionScore);
      safeUi('transition diagnostics render', renderTransitionDiagnostics);
    }
    if (activePane === 'preview') safeUi('transition preview render', renderTransitionPreview);
    safeUi('transition badges render', renderTransitionBadges);
    safeUi('audio ui sync', syncAudioUi);
    if (els.settingsPopover && !els.settingsPopover.classList.contains('hidden')) {
      safeUi('settings render', renderSettingsPanel);
    }
  }
  function downloadCsv() {
    const targets = targetCurve();
    const w = weights();
    const header = [
      'slot','target_energy','actual_energy','final_score','baseline_score','energy_fit',
      'style_score','rhythm_score','harmony_score','style_weight','rhythm_weight',
      'harmony_weight','rhythm_tempo_weight','mix','track_number','title','artists',
      'genre','key','bpm','filename'
    ];
    const lines = [header.join(',')];
    for (let i = 0; i < sequence.length; i += 1) {
      const r = sequence[i] === null ? null : byIdx.get(sequence[i]);
      let score = null;
      if (r && i > 0 && sequence[i - 1] !== null) {
        score = scoreTransition(Number(sequence[i - 1]), Number(sequence[i]), i);
      }
      const vals = [
        i + 1, fmt(targets[i], 3), r ? fmt(energyOf(r), 3) : '',
        score ? fmt(score.finalScore, 4) : '', score ? fmt(score.baseline, 4) : '',
        score ? fmt(score.energyScore, 4) : '', score ? fmt(score.styleScore, 4) : '',
        score ? fmt(score.rhythmScore, 4) : '', score ? fmt(score.harmonyScore, 4) : '',
        fmt(w.style, 4), fmt(w.rhythm, 4), fmt(w.harmony, 4), fmt(RHYTHM_TEMPO_WEIGHT, 4),
        r ? r.mix_slug : '', r ? r.track_number : '',
        r ? r.title : '', r ? r.artists : '', r ? r.genre : '', r ? r.key : '', r ? fmt(r.est_bpm, 3) : '', r ? r.filename : ''
      ];
      lines.push(vals.map(v => '"' + String(v).replace(/"/g, '""') + '"').join(','));
    }
    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'energy_sequence.csv';
    a.click();
    URL.revokeObjectURL(url);
  }

  els.sequenceLength.value = String(sequenceLength);
  applyAppSettingsToControls();
  setComponentSliders({
    style: (config.weights && config.weights.style_weight) ?? 0.50,
    rhythm: (config.weights && config.weights.rhythm_weight) ?? 0.30,
    harmony: (config.weights && config.weights.harmony_weight) ?? 0.20,
  });
  setActionButton(els.setOutgoing, '1', 'Set selected track as Track 1', 'track-one');
  setActionButton(els.setIncoming, '2', 'Set selected track as Track 2', 'track-two');
  setActionButton(els.clearTransition, '×', 'Clear transition pair', 'danger');
  setActionButton(els.clearLast, '−', 'Clear last sequence track', 'danger');
  setActionButton(els.resetSequence, '⌧', 'Reset sequence', 'danger');
  setActionButton(els.downloadSequence, '↓', 'Download sequence CSV');
  if (els.fullscreenToggle) {
    els.fullscreenToggle.addEventListener('click', ev => {
      ev.stopPropagation();
      toggleFullscreen();
    });
  }
  document.addEventListener('fullscreenchange', syncFullscreenToggle);
  document.addEventListener('webkitfullscreenchange', syncFullscreenToggle);
  syncFullscreenToggle();
  if (els.settingsToggle) {
    els.settingsToggle.addEventListener('click', ev => {
      ev.stopPropagation();
      setHelpPanelOpen(false);
      toggleSettingsPanel();
    });
  }
  if (els.helpToggle) {
    els.helpToggle.addEventListener('click', ev => {
      ev.stopPropagation();
      setSettingsPanelOpen(false);
      toggleHelpPanel();
    });
  }
  if (els.helpClose) {
    els.helpClose.addEventListener('click', ev => {
      ev.stopPropagation();
      setHelpPanelOpen(false);
    });
  }
  if (els.settingsClose) {
    els.settingsClose.addEventListener('click', ev => {
      ev.stopPropagation();
      setSettingsPanelOpen(false);
    });
  }
  document.addEventListener('click', ev => {
    if (els.settingsPopover && !els.settingsPopover.classList.contains('hidden')) {
      if (els.settingsPopover.contains(ev.target) || (els.settingsToggle && els.settingsToggle.contains(ev.target))) return;
      setSettingsPanelOpen(false);
    }
    if (els.helpPopover && !els.helpPopover.classList.contains('hidden')) {
      if (els.helpPopover.contains(ev.target) || (els.helpToggle && els.helpToggle.contains(ev.target))) return;
      setHelpPanelOpen(false);
    }
  });
  document.addEventListener('keydown', ev => {
    if (ev.key === 'Escape') {
      setSettingsPanelOpen(false);
      setHelpPanelOpen(false);
    }
    const tag = ev.target && ev.target.tagName ? String(ev.target.tagName).toLowerCase() : '';
    const typing = tag === 'input' || tag === 'textarea' || tag === 'select' || (ev.target && ev.target.isContentEditable);
    const key = String(ev.key || '').toLowerCase();
    if (!typing && (ev.metaKey || ev.ctrlKey) && !ev.altKey) {
      if (key === 'z') {
        if (activePane === 'preview' && transitionAutomationActiveLane) restoreTransitionAutomationHistory(ev.shiftKey ? 1 : -1);
        else if (ev.shiftKey) redoPlanning();
        else undoPlanning();
        ev.preventDefault();
        return;
      }
      if (key === 'y') {
        if (activePane === 'preview' && transitionAutomationActiveLane) restoreTransitionAutomationHistory(1);
        else redoPlanning();
        ev.preventDefault();
        return;
      }
    }
    if (!typing && activePane === 'preview' && (key === 'delete' || key === 'backspace') && deleteSelectedAutomationPoint()) {
      ev.preventDefault();
      return;
    }
    if (typing || ev.metaKey || ev.ctrlKey || ev.altKey) return;
    if (key === 'f' || key === 'f11') {
      toggleFullscreen();
      ev.preventDefault();
    } else if (key === '+' || key === '=' || key === 'i') {
      handleMapAction('zoom-in');
      ev.preventDefault();
    } else if (key === '?') {
      setSettingsPanelOpen(false);
      toggleHelpPanel();
      ev.preventDefault();
    } else if (key === '1') {
      if (selectedIdx !== null) setTransitionEndpoint('out', selectedIdx, { toggle: false });
      ev.preventDefault();
    } else if (key === '2') {
      if (selectedIdx !== null) setCandidateTrack(selectedIdx, { autoplay: false, showPopover: true });
      ev.preventDefault();
    } else if (key === '-' || key === '_') {
      handleMapAction('zoom-out');
      ev.preventDefault();
    } else if (key === 'arrowleft') {
      handleMapAction('pan-left');
      ev.preventDefault();
    } else if (key === 'arrowright') {
      handleMapAction('pan-right');
      ev.preventDefault();
    } else if (key === 'arrowup') {
      handleMapAction('pan-up');
      ev.preventDefault();
    } else if (key === 'arrowdown') {
      handleMapAction('pan-down');
      ev.preventDefault();
    } else if (key === 'r') {
      handleMapAction('reset');
      ev.preventDefault();
    }
  });
  els.tabButtons.forEach(btn => btn.addEventListener('click', () => setActivePane(btn.getAttribute('data-pane-tab') || 'explore')));
  els.sequenceLength.addEventListener('change', () => mutatePlanning(() => setSequenceLength(els.sequenceLength.value, false)));
  els.energySource.addEventListener('change', renderAll);
  els.colorMode.addEventListener('change', () => {
    setAppSetting('point_color', els.colorMode.value);
    renderAll();
  });
  if (els.mapEffectsEnabled) els.mapEffectsEnabled.addEventListener('change', () => {
    setAppSetting('map_fx', Boolean(els.mapEffectsEnabled.checked));
    if (!els.mapEffectsEnabled.checked) mapTrailSegments = [];
    renderAll();
  });
  if (els.showScoreValues) els.showScoreValues.addEventListener('change', () => {
    setAppSetting('show_score_values', Boolean(els.showScoreValues.checked));
    safeUi('recommendations render', renderRecommendations);
    if (els.settingsPopover && !els.settingsPopover.classList.contains('hidden')) {
      safeUi('settings render', renderSettingsPanel);
    }
  });
  if (els.figureLightMode) els.figureLightMode.addEventListener('change', () => {
    setAppSetting('figure_light_mode', Boolean(els.figureLightMode.checked));
    applyFigureTheme({ renderEnergy: true });
    if (els.settingsPopover && !els.settingsPopover.classList.contains('hidden')) {
      safeUi('settings render', renderSettingsPanel);
    }
  });
  if (els.loudnessMatchMode) els.loudnessMatchMode.addEventListener('change', () => {
    const mode = LOUDNESS_MATCH_MODES.includes(els.loudnessMatchMode.value) ? els.loudnessMatchMode.value : 'auditions_transitions';
    setAppSetting('loudness_match_mode', mode);
    applySharedAudioLoudness({ smooth: true });
    applyLiveLoudnessGains({ smooth: true });
    renderSelectionOnly();
    updateTransitionDirtyUi();
    if (els.settingsPopover && !els.settingsPopover.classList.contains('hidden')) {
      safeUi('settings render', renderSettingsPanel);
    }
  });
  if (els.latentLinksPerTrack) els.latentLinksPerTrack.addEventListener('input', () => {
    setAppSetting('latent_links_per_track', clamp(Math.round(Number(els.latentLinksPerTrack.value)), 0, 8));
    updateMapLinkSettingLabels();
    renderAll();
  });
  if (els.recommendedLinksHighlight) els.recommendedLinksHighlight.addEventListener('input', () => {
    setAppSetting('recommended_links_highlight', clamp(Math.round(Number(els.recommendedLinksHighlight.value)), 1, 25));
    updateMapLinkSettingLabels();
    renderAll();
  });
  els.penaltyScale.addEventListener('input', renderAll);
  [els.weightStyle, els.weightRhythm, els.weightHarmony].forEach(el => {
    if (!el) return;
    el.addEventListener('input', () => {
      setComponentSliders(componentWeightsRaw());
      renderAll();
    });
  });
  if (els.simplex) {
    let draggingSimplex = false;
    els.simplex.addEventListener('pointerdown', ev => { draggingSimplex = true; els.simplex.setPointerCapture(ev.pointerId); setWeightsFromSimplexEvent(ev); });
    els.simplex.addEventListener('pointermove', ev => { if (draggingSimplex) setWeightsFromSimplexEvent(ev); });
    els.simplex.addEventListener('pointerup', ev => { draggingSimplex = false; try { els.simplex.releasePointerCapture(ev.pointerId); } catch (err) {} });
    els.simplex.addEventListener('pointercancel', () => { draggingSimplex = false; });
  }
  if (els.setOutgoing) els.setOutgoing.addEventListener('click', () => { if (selectedIdx !== null) setTransitionEndpoint('out', selectedIdx); });
  if (els.setIncoming) els.setIncoming.addEventListener('click', () => { if (selectedIdx !== null) setCandidateTrack(selectedIdx, { autoplay: false, showPopover: true }); });
  els.appendSelected.addEventListener('click', () => { if (selectedIdx !== null) appendTrack(selectedIdx); });
  if (els.clearTransition) els.clearTransition.addEventListener('click', clearTransitionPair);
  if (els.undoPlanning) els.undoPlanning.addEventListener('click', undoPlanning);
  if (els.redoPlanning) els.redoPlanning.addEventListener('click', redoPlanning);
  if (els.applyEnergyGuides) els.applyEnergyGuides.addEventListener('click', applyTargetGuides);
  if (els.clearEnergyTargets) els.clearEnergyTargets.addEventListener('click', () => mutatePlanning(() => {
    targetValues = Array.from({ length: sequenceLength }, () => null);
  }));
  els.clearLast.addEventListener('click', () => {
    mutatePlanning(() => {
      for (let i = sequence.length - 1; i >= 0; i -= 1) { if (sequence[i] !== null) { sequence[i] = null; break; } }
    });
  });
  els.resetSequence.addEventListener('click', () => {
    mutatePlanning(() => {
      sequence = Array.from({ length: sequenceLength }, () => null);
      targetValues = Array.from({ length: sequenceLength }, () => null);
      selectedSlot = null;
    });
  });
  els.downloadSequence.addEventListener('click', downloadCsv);
  if (els.librarySearch) els.librarySearch.addEventListener('input', () => {
    libraryQuery = els.librarySearch.value || '';
    renderLibrary();
  });
  if (els.audio) {
    applyMasterVolume();
    els.audio.addEventListener('play', () => {
      previewAudioPlaying = true;
      syncAudioUi();
      requestAudioProgressFrame();
    });
    els.audio.addEventListener('pause', () => {
      previewAudioPlaying = false;
      syncAudioUi();
    });
    els.audio.addEventListener('ended', () => {
      previewAudioPlaying = false;
      syncAudioUi();
    });
    els.audio.addEventListener('timeupdate', () => {
      if (previewAudioIdx !== null) rowScrubPositions.set(Number(previewAudioIdx), Number(els.audio.currentTime || 0));
      syncAudioUi();
    });
    els.audio.addEventListener('loadedmetadata', syncAudioUi);
    els.audio.addEventListener('seeked', syncAudioUi);
  }
  if (els.songPlayerToggle) {
    els.songPlayerToggle.addEventListener('click', () => {
      if (selectedIdx === null || !byIdx.has(selectedIdx)) return;
      if (previewAudioIdx === selectedIdx && previewAudioContext === 'main' && previewAudioPlaying) pauseSharedAudio();
      else {
        const record = byIdx.get(selectedIdx);
        const start = previewAudioIdx === selectedIdx && els.audio ? Number(els.audio.currentTime || 0) : previewStart(record);
        playRecordAt(record, start, 'main');
      }
    });
  }
  if (els.songWaveform) {
    els.songWaveform.addEventListener('pointerdown', ev => {
      waveformPointerActive = true;
      try { els.songWaveform.setPointerCapture(ev.pointerId); } catch (err) {}
      seekMainWaveformFromEvent(ev, true);
    });
    els.songWaveform.addEventListener('pointermove', ev => {
      if (waveformPointerActive) seekMainWaveformFromEvent(ev, true);
    });
    els.songWaveform.addEventListener('pointerup', ev => {
      waveformPointerActive = false;
      try { els.songWaveform.releasePointerCapture(ev.pointerId); } catch (err) {}
      seekMainWaveformFromEvent(ev, true);
    });
    els.songWaveform.addEventListener('pointercancel', () => {
      waveformPointerActive = false;
    });
  }
  if (els.globalToggle) {
    els.globalToggle.addEventListener('click', () => {
      const record = currentAudioRecord();
      if (!record) return;
      if (previewAudioIdx === Number(record.idx) && previewAudioPlaying) {
        pauseSharedAudio();
        return;
      }
      const start = previewAudioIdx === Number(record.idx) && els.audio
        ? Number(els.audio.currentTime || 0)
        : (rowScrubPositions.has(Number(record.idx)) ? rowScrubValue(record) : previewStart(record));
      playRecordAt(record, start, 'global');
    });
  }
  if (els.globalScrub) {
    const handleGlobalScrub = () => {
      const record = currentAudioRecord();
      if (!record) return;
      const idx = Number(record.idx);
      const value = Number(els.globalScrub.value || 0);
      if (!Number.isFinite(idx) || !Number.isFinite(value)) return;
      rowScrubPositions.set(idx, value);
      if (previewAudioIdx === idx && els.audio) seekSharedAudio(value);
      updateRowScrubbers();
      updateGlobalPlayer();
      drawMainWaveform();
    };
    els.globalScrub.addEventListener('input', handleGlobalScrub);
    els.globalScrub.addEventListener('change', handleGlobalScrub);
  }
  if (els.globalVolume) {
    masterVolume = clamp(Number(els.globalVolume.value || masterVolume), 0, 1);
    applyMasterVolume();
    els.globalVolume.addEventListener('input', () => {
      masterVolume = clamp(Number(els.globalVolume.value), 0, 1);
      applyMasterVolume();
    });
    els.globalVolume.addEventListener('change', () => {
      masterVolume = clamp(Number(els.globalVolume.value), 0, 1);
      applyMasterVolume();
    });
  }
  if (els.globalVolumeButton) {
    els.globalVolumeButton.addEventListener('click', ev => {
      ev.preventDefault();
      ev.stopPropagation();
      masterVolume = masterVolume <= 0.001 ? Math.max(0.1, lastNonzeroVolume || 0.85) : 0;
      applyMasterVolume();
    });
  }
  if (els.globalTrack1) els.globalTrack1.addEventListener('click', () => {
    const record = currentAudioRecord();
    if (record) setTransitionEndpoint('out', Number(record.idx), { toggle: false });
  });
  if (els.globalTrack2) els.globalTrack2.addEventListener('click', () => {
    const record = currentAudioRecord();
    if (record) setCandidateTrack(Number(record.idx), { autoplay: false, showPopover: true });
  });
  document.body.addEventListener('pointerdown', ev => {
    const wave = closestEl(ev.target, '[data-diagnostic-waveform-idx]');
    if (!wave) return;
    diagnosticWaveformPointerIdx = Number(wave.getAttribute('data-diagnostic-waveform-idx'));
    try { wave.setPointerCapture(ev.pointerId); } catch (err) {}
    seekDiagnosticWaveformFromEvent(ev, diagnosticWaveformPointerIdx, { play: true });
    ev.preventDefault();
    ev.stopPropagation();
  });
  document.body.addEventListener('pointermove', ev => {
    if (diagnosticWaveformPointerIdx === null) return;
    seekDiagnosticWaveformFromEvent(ev, diagnosticWaveformPointerIdx, { play: false });
    ev.preventDefault();
  });
  document.body.addEventListener('pointerup', ev => {
    if (diagnosticWaveformPointerIdx === null) return;
    seekDiagnosticWaveformFromEvent(ev, diagnosticWaveformPointerIdx, { play: false });
    diagnosticWaveformPointerIdx = null;
    ev.preventDefault();
  });
  document.body.addEventListener('pointercancel', () => {
    diagnosticWaveformPointerIdx = null;
  });
  document.body.addEventListener('pointerdown', ev => {
    const wave = closestEl(ev.target, '[data-library-waveform-idx]');
    if (!wave) return;
    rowWaveformPointerIdx = Number(wave.getAttribute('data-library-waveform-idx'));
    try { wave.setPointerCapture(ev.pointerId); } catch (err) {}
    seekLibraryWaveformFromEvent(ev, rowWaveformPointerIdx, { play: true });
    ev.preventDefault();
    ev.stopPropagation();
  });
  document.body.addEventListener('pointermove', ev => {
    if (rowWaveformPointerIdx === null) return;
    seekLibraryWaveformFromEvent(ev, rowWaveformPointerIdx, { play: false });
    ev.preventDefault();
  });
  document.body.addEventListener('pointerup', ev => {
    if (rowWaveformPointerIdx === null) return;
    seekLibraryWaveformFromEvent(ev, rowWaveformPointerIdx, { play: false });
    rowWaveformPointerIdx = null;
    ev.preventDefault();
  });
  document.body.addEventListener('pointercancel', () => {
    rowWaveformPointerIdx = null;
  });
  document.body.addEventListener('dragstart', ev => {
    const slotValue = closestAttr(ev.target, 'data-sequence-drag-slot');
    if (slotValue === null) return;
    const slot = Number(slotValue);
    if (!Number.isFinite(slot) || slot < 0 || slot >= sequence.length) return;
    sequenceDragSlot = slot;
    if (ev.dataTransfer) {
      ev.dataTransfer.effectAllowed = 'move';
      ev.dataTransfer.setData('text/plain', String(slot));
    }
    const row = closestEl(ev.target, '[data-sequence-drop-slot]');
    if (row) row.classList.add('drag-source');
  });
  document.body.addEventListener('dragover', ev => {
    const dropValue = closestAttr(ev.target, 'data-sequence-drop-slot');
    if (dropValue === null || sequenceDragSlot === null) return;
    ev.preventDefault();
    if (ev.dataTransfer) ev.dataTransfer.dropEffect = 'move';
  });
  document.body.addEventListener('drop', ev => {
    const dropValue = closestAttr(ev.target, 'data-sequence-drop-slot');
    if (dropValue === null || sequenceDragSlot === null) return;
    ev.preventDefault();
    moveSequenceSlot(sequenceDragSlot, Number(dropValue));
  });
  document.body.addEventListener('dragend', () => {
    if (sequenceDragSlot === null) return;
    sequenceDragSlot = null;
    document.querySelectorAll('.sequence-list .drag-source').forEach(row => row.classList.remove('drag-source'));
  });
  window.addEventListener('resize', () => setTimeout(() => safeUi('main waveform draw', drawMainWaveform), 30));
  window.addEventListener('resize', () => setTimeout(() => safeUi('library waveform draw', drawLibraryWaveforms), 30));
  window.addEventListener('resize', () => setTimeout(() => safeUi('transition editor draw', drawTransitionEditor), 30));
  window.addEventListener('resize', () => setTimeout(() => safeUi('mini map draw', drawMiniMap), 30));
  window.addEventListener('resize', () => setTimeout(() => safeUi('webgl map resize', () => { if (webglMap) webglMap.resize(); }), 30));

  document.body.addEventListener('click', ev => {
    const appendIdx = closestAttr(ev.target, 'data-append-idx');
    const removeSlotValue = closestAttr(ev.target, 'data-remove-slot');
    const seqSlotValue = closestAttr(ev.target, 'data-seq-slot');
    const clearTargetSlotAttr = closestAttr(ev.target, 'data-clear-target-slot');
    const placeSlotValue = closestAttr(ev.target, 'data-place-slot');
    const focusTrackIdx = closestAttr(ev.target, 'data-focus-track');
    const playButton = closestEl(ev.target, '[data-play-idx]');
    const playIdx = playButton ? playButton.getAttribute('data-play-idx') : null;
    const playMode = playButton ? (playButton.getAttribute('data-play-mode') || 'point') : 'point';
    const libraryCurrent = closestAttr(ev.target, 'data-library-current');
    const libraryOutgoing = closestAttr(ev.target, 'data-library-outgoing');
    const libraryIncoming = closestAttr(ev.target, 'data-library-incoming');
    const libraryPlace = closestAttr(ev.target, 'data-library-place');
    const librarySortKey = closestAttr(ev.target, 'data-library-sort');
    const previewAction = closestAttr(ev.target, 'data-preview-action');
    const previewStep = closestAttr(ev.target, 'data-preview-step');
    const topLoopToggle = closestAttr(ev.target, 'data-transition-loop-top');
    const transitionTrackPlay = closestAttr(ev.target, 'data-transition-track-play');
    const mapAction = closestAttr(ev.target, 'data-map-action');
    const pinRecIdx = closestAttr(ev.target, 'data-pin-rec-idx');
    if (mapAction !== null) {
      handleMapAction(mapAction);
      ev.preventDefault();
      return;
    }
    if (pinRecIdx !== null) {
      const idx = Number(pinRecIdx);
      if (pinnedRecommendationIdxs.has(idx)) pinnedRecommendationIdxs.delete(idx);
      else pinnedRecommendationIdxs.add(idx);
      safeUi('recommendations render', renderRecommendations);
      safeUi('map overlays', updatePacmapOverlays);
      safeUi('map effects', ensureMapEffectsLoop);
      ev.preventDefault();
      return;
    }
    if (librarySortKey !== null) {
      if (librarySort !== librarySortKey) {
        librarySort = librarySortKey;
        libraryDir = 'asc';
      } else if (libraryDir === 'asc') {
        libraryDir = 'desc';
      } else {
        librarySort = null;
        libraryDir = null;
      }
      renderLibrary();
    }
    if (appendIdx !== null) appendTrack(Number(appendIdx));
    if (removeSlotValue !== null) removeSlot(Number(removeSlotValue));
    if (clearTargetSlotAttr !== null) mutatePlanning(() => clearTargetSlotValue(Number(clearTargetSlotAttr), false));
    if (seqSlotValue !== null) mutatePlanning(() => {
      const slot = Number(seqSlotValue);
      selectedSlot = selectedSlot === slot ? null : slot;
    });
    if (placeSlotValue !== null && selectedIdx !== null) { selectedSlot = Number(placeSlotValue); appendTrack(selectedIdx); }
    if (playIdx !== null) toggleTrackPreview(Number(playIdx), { mode: playMode });
    if (focusTrackIdx !== null) focusTrack(Number(focusTrackIdx), { autoplay: false, showPopover: true });
    if (libraryCurrent !== null) setCurrentTrack(Number(libraryCurrent));
    if (libraryOutgoing !== null) setTransitionEndpoint('out', Number(libraryOutgoing), { toggle: false });
    if (libraryIncoming !== null) setCandidateTrack(Number(libraryIncoming), { autoplay: false, showPopover: true });
    if (libraryPlace !== null) appendTrack(Number(libraryPlace));
    if (previewStep !== null) {
      const [field, direction] = String(previewStep).split(':');
      stepTransitionTimingField(field, Number(direction));
      ev.preventDefault();
      return;
    }
    if (topLoopToggle !== null) {
      setTransitionLoopEnabled(!transitionLoopEnabled);
      ev.preventDefault();
      return;
    }
    if (previewAction === 'swap') swapTransitionPair();
    if (previewAction === 'reset-alignments') resetBothDeckAlignments();
    if (previewAction === 'clear') clearTransitionPair();
    if (previewAction === 'render') renderBackendTransition();
    if (transitionTrackPlay !== null) toggleTransitionTrackPreview(transitionTrackPlay);
  });
  document.body.addEventListener('change', ev => {
    if (updateRecFiltersFromElement(ev.target)) {
      renderRecommendations();
      updatePacmapOverlays();
      ensureMapEffectsLoop();
      return;
    }
    if (ev.target && ev.target.getAttribute && ev.target.getAttribute('data-preview-field')) {
      updatePreviewFormStateFromElement(ev.target);
      updatePreviewEffectDisplay({ dirty: true });
      return;
    }
    if (ev.target && ev.target.getAttribute && ev.target.getAttribute('data-preview-mode') !== null) {
      transitionPreviewMode = String(ev.target.value || 'verified') === 'live' ? 'live' : 'verified';
      if (transitionPreviewMode === 'live') {
        const rendered = transitionAudioElement();
        if (rendered && !rendered.paused) rendered.pause();
        scheduleLivePreparation();
      } else stopLiveAudition();
      renderTransitionPreview();
      return;
    }
    const slotValue = ev.target && ev.target.getAttribute && ev.target.getAttribute('data-target-slot');
    if (slotValue === null) return;
    const slot = Number(slotValue);
    const rawValue = String(ev.target.value || '').trim();
    mutatePlanning(() => {
      if (!rawValue) clearTargetSlotValue(slot, false);
      else setTargetSlotValue(slot, rawValue, false);
    });
  });
  document.body.addEventListener('input', ev => {
    if (updateRecFiltersFromElement(ev.target)) {
      renderRecommendations();
      updatePacmapOverlays();
      ensureMapEffectsLoop();
      return;
    }
    if (ev.target && ev.target.getAttribute && ev.target.getAttribute('data-preview-field')) {
      updatePreviewFormStateFromElement(ev.target);
      updatePreviewEffectDisplay({ dirty: true });
      return;
    }
    const transitionDeck = ev.target && ev.target.getAttribute && ev.target.getAttribute('data-transition-track-scrub');
    if (transitionDeck !== null) {
      const info = transitionWindowInfo(transitionDeck);
      const value = Number(ev.target.value || 0);
      if (info && info.record && Number.isFinite(value)) {
        const idx = Number(info.record.idx);
        rowScrubPositions.set(idx, clamp(value, 0, info.trackDuration));
        if (previewAudioIdx === idx && previewAudioContext === 'transition-' + transitionDeck && els.audio) seekSharedAudio(value);
        updateTransitionTrackPlayButtons();
        requestTransitionEditorDraw();
      }
      return;
    }
    const scrubIdx = ev.target && ev.target.getAttribute && ev.target.getAttribute('data-library-scrub-idx');
    if (scrubIdx !== null) {
      const idx = Number(scrubIdx);
      const value = Number(ev.target.value || 0);
      if (Number.isFinite(idx) && Number.isFinite(value)) {
        rowScrubPositions.set(idx, value);
        if (previewAudioIdx === idx && els.audio) seekSharedAudio(value);
        updateRowScrubbers();
      }
    }
  });

  if (els.energyCurve) {
    els.energyCurve.addEventListener('mousedown', beginEnergyDrag);
    document.addEventListener('mousemove', updateEnergyDrag);
    document.addEventListener('mouseup', endEnergyDrag);
  }
  if (els.mapMiniMap) {
    els.mapMiniMap.addEventListener('pointerdown', ev => {
      recenterMapFromMiniMap(ev);
      ev.preventDefault();
      ev.stopPropagation();
    });
  }

  if (plot && plot.on) {
    plot.on('plotly_click', ev => {
      if (usingWebglMap()) return;
      if (!ev || !ev.points || !ev.points.length) return;
      const c = ev.points[0].customdata || [];
      const idx = Number(c[0]);
      if (Number.isFinite(idx)) handleMapTrackClick(idx, ev.event);
    });
    plot.on('plotly_hover', ev => {
      if (usingWebglMap()) return;
      if (!ev || !ev.points || !ev.points.length) return;
      const c = ev.points[0].customdata || [];
      const idx = Number(c[0]);
      if (Number.isFinite(idx) && byIdx.has(idx)) showSongHover(byIdx.get(idx), ev.event);
    });
    plot.on('plotly_unhover', () => { if (!usingWebglMap()) hideSongHover(); });
    plot.on('plotly_relayout', () => {
      if (usingWebglMap()) return;
      const view = mapViewRanges();
      const next = view ? constrainMapRanges(view.x0, view.x1, view.y0, view.y1) : null;
      if (!mapRelayoutClampActive && next && mapRangesDiffer(view, next) && window.Plotly && plot) {
        mapRelayoutClampActive = true;
        const done = () => {
          mapRelayoutClampActive = false;
          safeUi('map overlays after zoom clamp', updatePacmapOverlays);
          safeUi('map effects after zoom clamp', ensureMapEffectsLoop);
          safeUi('mini map after zoom clamp', drawMiniMap);
        };
        const relayout = Plotly.relayout(plot, {
          'xaxis.range': [next.x0, next.x1],
          'yaxis.range': [next.y0, next.y1],
        });
        if (relayout && typeof relayout.finally === 'function') relayout.finally(done);
        else done();
        return;
      }
      safeUi('map overlays after zoom', updatePacmapOverlays);
      safeUi('map effects after zoom', ensureMapEffectsLoop);
      safeUi('mini map after zoom', drawMiniMap);
    });
    plot.on('plotly_doubleclick', () => {
      if (usingWebglMap()) return false;
      if ((Date.now() - lastPointDoubleClickMs) < 900) return false;
      if (lastClickedIdx !== null && (Date.now() - lastClickMs) < 720) {
        assignTransitionFromDoubleClick(lastClickedIdx);
        return false;
      }
      if (transitionFromIdx !== null || transitionToIdx !== null) clearTransitionPair();
      return false;
    });
    plot.addEventListener('dblclick', ev => {
      if (usingWebglMap()) return;
      const onPoint = isPlotPointTarget(ev.target);
      handlePlotDoubleClick(onPoint);
      ev.preventDefault();
      ev.stopPropagation();
    });
    plot.addEventListener('click', ev => {
      if (usingWebglMap()) return;
      if (Number(ev.detail || 0) > 1) return;
      const onPoint = isPlotPointTarget(ev.target);
      if (onPoint) return;
      const nearTrackPoint = plotClickNearTrackPoint(ev);
      if (nearTrackPoint) return;
      handleMapBlankClick();
    });
  }
  const restoredPlanningWorkspace = restorePlanningWorkspace();
  if (restoredPlanningWorkspace && els.planningSaveStatus) els.planningSaveStatus.textContent = 'Restored local plan';
  updatePlanningControls();
  setActivePane('explore', { render: false });
  renderAll();
})();
