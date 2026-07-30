(function() {
  const records = JSON.parse(document.getElementById('seq-records-json').textContent);
  const simMap = JSON.parse(document.getElementById('seq-sim-json').textContent);
  const idxToPoint = JSON.parse(document.getElementById('seq-points-json').textContent);
  const layoutEntries = JSON.parse(document.getElementById('seq-layout-entries-json').textContent);
  const config = JSON.parse(document.getElementById('seq-config-json').textContent);
  const plot = document.getElementById('__PLOT_ID__');
  const appSettings = config.app_settings || {};
  const artworkUriCount = records.filter(r => r && r.artwork_uri).length;
  appSettings.defaults = Object.assign({
    latent_links_per_track: 3,
    recommended_links_highlight: 25,
    point_color: 'genre',
    map_renderer: 'plotly',
    map_fx: true,
    map_artwork: artworkUriCount * 2 >= Math.max(1, records.length),
    show_score_values: true,
    click_to_play: true,
    mini_map_collapsed: false,
    side_settings_open: false,
    build_segment: 'sequence',
  }, appSettings.defaults || {});
  appSettings.current = Object.assign({}, appSettings.defaults, appSettings.current || {});
  appSettings.behavior = Object.assign({
    latent_links: 'Suggested links: the strongest weighted-similarity lines drawn from each track, using the current weight sliders.',
    recommended_links: "The selected track's ranked next-track recommendations highlighted on the map.",
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
  };
  const DEFAULT_TARGET_ENERGY = 5;
  const ENERGY_AXIS_TICKS = [1, 3, 5, 7, 9];
  const MAP_MIN_ZOOM_FRACTION = 0.045;
  const MAP_MAX_ZOOM_FRACTION = 1.35;
  const MAP_WHEEL_ZOOM_IN_FACTOR = 0.945;
  const MAP_WHEEL_ZOOM_OUT_FACTOR = 1.06;
  const MAP_BUTTON_ZOOM_IN_FACTOR = 0.92;
  const MAP_BUTTON_ZOOM_OUT_FACTOR = 1.09;
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
  let hoveredIdx = null;
  let selectedSlot = null;
  let sequenceLength = Number(config.default_length || 10);
  let sequence = Array.from({ length: sequenceLength }, () => null);
  let targetValues = Array.from({ length: sequenceLength }, () => DEFAULT_TARGET_ENERGY);
  let sequenceDragSlot = null;
  let draggingEnergySlot = null;
  let currentPoints = [];
  let activePane = 'explore';
  const BUILD_SEGMENTS = ['sequence', 'recommend', 'transition'];
  let activeBuildSegment = 'sequence';
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
    from_nudge_beats: 0,
    to_nudge_beats: 0,
    from_pitch_shift: 0,
    to_pitch_shift: 0,
    preset: 'auto',
    volume_mode: 'smooth-crossfade',
    eq_mode: 'center-bass-swap',
    filter_mode: 'none',
  };

  const els = {
    sequenceLength: document.getElementById('sequence-length'),
    energySource: document.getElementById('energy-source'),
    colorMode: document.getElementById('color-mode'),
    mapRenderer: document.getElementById('map-renderer'),
    mapEffectsEnabled: document.getElementById('map-effects-enabled'),
    mapArtworkEnabled: document.getElementById('map-artwork-enabled'),
    showScoreValues: document.getElementById('show-score-values'),
    clickToPlay: document.getElementById('click-to-play'),
    latentLinksPerTrack: document.getElementById('latent-links-per-track'),
    latentLinksPerTrackVal: document.getElementById('latent-links-per-track-val'),
    recommendedLinksHighlight: document.getElementById('recommended-links-highlight'),
    recommendedLinksHighlightVal: document.getElementById('recommended-links-highlight-val'),
    penaltyScale: document.getElementById('energy-penalty-scale'),
    penaltyScaleVal: document.getElementById('energy-penalty-scale-val'),
    weightStyle: document.getElementById('weight-style'),
    weightMaest: document.getElementById('weight-maest'),
    weightChroma: document.getElementById('weight-chroma'),
    weightTempo: document.getElementById('weight-tempo'),
    weightGroove: document.getElementById('weight-groove'),
    weightStyleVal: document.getElementById('weight-style-val'),
    weightMaestVal: document.getElementById('weight-maest-val'),
    weightChromaVal: document.getElementById('weight-chroma-val'),
    weightTempoVal: document.getElementById('weight-tempo-val'),
    weightGrooveVal: document.getElementById('weight-groove-val'),
    simplex: document.getElementById('simplex-control'),
    simplexHandle: document.getElementById('simplex-handle'),
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
    },
    transitionBadges: document.getElementById('transition-badges'),
    transitionScore: document.getElementById('transition-score'),
    buildSegButtons: Array.from(document.querySelectorAll('[data-build-seg]')),
    buildSegPanels: Array.from(document.querySelectorAll('[data-build-seg-panel]')),
    transitionPreview: document.getElementById('transition-preview-panel'),
    setOutgoing: document.getElementById('set-outgoing'),
    setIncoming: document.getElementById('set-incoming'),
    songPopover: document.getElementById('song-popover'),
    audio: document.getElementById('track-audio'),
    songPlayerStatus: document.getElementById('song-player-status'),
    songWaveform: document.getElementById('song-waveform'),
    appendSelected: document.getElementById('append-selected'),
    clearTransition: document.getElementById('clear-transition'),
    clearLast: document.getElementById('clear-last'),
    resetSequence: document.getElementById('reset-sequence'),
    downloadSequence: document.getElementById('download-sequence'),
    energyCurve: document.getElementById('energy-curve'),
    sequenceList: document.getElementById('sequence-list'),
    recommendationPanel: document.getElementById('recommendation-panel'),
    transitionDiagnostics: document.getElementById('transition-diagnostics'),
    currentTransitionScore: document.getElementById('current-transition-score'),
    mapEffects: document.getElementById('map-effects-canvas'),
    mapArtwork: document.getElementById('map-artwork-canvas'),
    webglMap: document.getElementById('webgl-map-canvas'),
    mapMiniMap: document.getElementById('map-mini-map'),
    miniMapToggle: document.getElementById('mini-map-toggle'),
    sideSettings: document.getElementById('side-settings'),
    mapZoomControls: document.getElementById('map-zoom-controls'),
    mapColorLegend: document.getElementById('map-color-legend'),
    mapLegendRestore: document.getElementById('map-legend-restore'),
    songHoverCard: document.getElementById('song-hover-card'),
    librarySearch: document.getElementById('library-search'),
    libraryTable: document.getElementById('library-table'),
    globalArt: document.getElementById('global-art'),
    globalTitle: document.getElementById('global-title'),
    globalArtist: document.getElementById('global-artist'),
    globalToggle: document.getElementById('global-toggle'),
    globalScrub: document.getElementById('global-scrub'),
    globalVolume: document.getElementById('global-volume'),
    globalVolumeButton: document.getElementById('global-volume-button'),
    globalTime: document.getElementById('global-time'),
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
  function fmt(v, d=3) { return Number.isFinite(Number(v)) ? Number(v).toFixed(d) : ''; }
  function setRangeProgress(el, value, maxValue) {
    if (!el) return;
    const max = Number(maxValue || el.max || 0);
    const pct = max > 0 ? clamp((Number(value) || 0) / max, 0, 1) * 100 : 0;
    el.style.setProperty('--progress', pct.toFixed(2) + '%');
  }
  function applyMasterVolume() {
    masterVolume = clamp(Number(masterVolume), 0, 1);
    if (masterVolume > 0.001) lastNonzeroVolume = masterVolume;
    if (els.audio) els.audio.volume = masterVolume;
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
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  }
  function roundSettingNumbers(key, value) {
    return typeof value === 'number' && Number.isFinite(value) && !Number.isInteger(value) ? Number(value.toFixed(4)) : value;
  }
  function formatSettingValue(value) {
    if (value === null || value === undefined || value === '') return '<span class="muted">None</span>';
    if (typeof value === 'boolean') return value ? 'true' : 'false';
    if (typeof value === 'number') return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(6)));
    if (Array.isArray(value) || typeof value === 'object') {
      return '<code>' + esc(JSON.stringify(value, roundSettingNumbers, 2)) + '</code>';
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
    scheduleSessionSave();
  }
  function currentUiSettingsRows() {
    const row = (label, value) => [label, value];
    return [
      row('Active pane', activePane),
      row('Track count', records.length),
      row('Sequence length', sequenceLength),
      row('Energy source', els.energySource ? els.energySource.value : ''),
      row('Point color', appSetting('point_color', 'genre')),
      row('Map renderer', selectedMapRenderer()),
      row('Map effects', Boolean(appSetting('map_fx', true))),
      row('Album art on map', artworkMarkersEnabled()),
      row('Suggested links per track', appSetting('latent_links_per_track', 3)),
      row('Recommended links highlighted', appSetting('recommended_links_highlight', 25)),
      row('Show recommendation scores', Boolean(appSetting('show_score_values', true))),
      row('Play on select', Boolean(appSetting('click_to_play', true))),
      row('Energy penalty scale', els.penaltyScale ? Number(els.penaltyScale.value) : null),
      row('Current weights', weights()),
      row('Preview preset', previewFormState.preset),
      row('Preview volume mode', previewFormState.volume_mode),
      row('Preview EQ mode', previewFormState.eq_mode),
      row('Preview filter mode', previewFormState.filter_mode),
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
        ['Suggested links', appSettings.behavior && appSettings.behavior.latent_links],
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
    const savedSession = savedSessionInfo();
    const savedSessionSection = '<section class="settings-section"><h3>Saved session</h3>' +
      settingsRowsHtml([
        ['Saved session', savedSession ? 'Available' : 'None'],
        ['Last saved', savedSession && savedSession.savedAt ? new Date(Number(savedSession.savedAt)).toLocaleString() : null],
      ]) +
      '<div class="settings-session-actions"><button type="button" data-clear-saved-session' + (savedSession ? '' : ' disabled') + '>Clear saved session</button></div>' +
      '</section>';
    const intro = '<div class="muted" style="font-size:11px;">Read-only snapshot of the current configuration. Interactive settings live in the Explore side panel.</div>';
    els.settingsContent.innerHTML = intro + sections.join('') + savedSessionSection;
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
        helpKeyHtml('click', 'Select a track and show its detail panel. Selecting never changes Track 1 or Track 2.'),
        helpKeyHtml('double click', 'Set or clear Track 1 as the recommendation source.'),
        helpKeyHtml('1', 'Set the selected track as Track 1.'),
        helpKeyHtml('2', 'Set the selected track as Track 2 (assign Track 1 first).'),
        helpKeyHtml('hover', 'Show preview without changing selection.'),
        helpKeyHtml('Esc', 'Close open panels.'),
      ]) +
      helpCardHtml('Sequence workflow', [
        helpKeyHtml('+ / ⇄', 'Add the selected track to the target slot (⇄ replaces a filled slot) and make it Track 1; a message shows what changed, with undo.'),
        helpKeyHtml('○ / ●', 'Select or deselect a sequence slot; placement and recommendations target the selected slot.'),
        helpKeyHtml('Track 1', 'Locked source for recommendations and transition scoring.'),
        helpKeyHtml('Track 2', 'Destination track for the transition preview.'),
        helpKeyHtml('undo', 'Adding, removing, or resetting tracks shows a message with an undo button.'),
      ]) +
      helpCardHtml('Display modes', [
        helpKeyHtml('Genre', 'Rolled-up genre categories.'),
        helpKeyHtml('Energy', 'Tagged or auto energy, depending on Energy source.'),
        helpKeyHtml('Target', 'Selected energy minus the target energy for the active slot.'),
        helpKeyHtml('Key wheel', 'Camelot key color; gray means missing or unavailable.'),
        helpKeyHtml('Tempo', 'Estimated BPM.'),
        helpKeyHtml('6.0 (+1.0)', 'Track energy, and its offset from the slot target energy.'),
        helpKeyHtml('Album art', 'Side-panel option: show album covers as map markers (Plotly renderer only). Ring colors follow the key under the map.'),
        helpKeyHtml('legend', 'Click a genre in the map legend to hide it; "Show all genres" under the map brings everything back.'),
      ]) +
      helpCardHtml('Scoring', [
        helpKeyHtml('Final', 'Final = Mix × Fit. Overall recommendation score. On Explore, hover a Final value for its Mix, Fit, and Loss parts.'),
        helpKeyHtml('Mix', 'Weighted blend of Style, Tempo, Groove, and Key similarity (set by the weight controls). Full column on Diagnostics.'),
        helpKeyHtml('Fit', 'How well the track energy matches the slot target (1 = perfect).'),
        helpKeyHtml('Loss', 'Loss = Mix − Final: score lost to energy mismatch. Shown as a negative amount with a right-filled bar; closer to 0 is better.'),
        helpKeyHtml('Rank', 'Position among all scored candidates for the slot (#1 is best).'),
      ]) +
      helpCardHtml('Panes and tables', [
        helpKeyHtml('← / →', 'Switch panes while a pane tab is focused.'),
        helpKeyHtml('↑ / ↓', 'Row buttons in the Sequence table move a slot up or down.'),
        helpKeyHtml('headers', 'Library column headers sort — click, or press Enter or Space on a focused header.'),
        helpKeyHtml('settings', 'Weight sliders and map options live under "Map and scoring settings" in the Explore side panel.'),
      ]) +
      helpCardHtml('Audio', [
        helpKeyHtml('play', 'Preview the selected or row track.'),
        helpKeyHtml('Play on select', 'Side-panel option: when off, selecting a track only selects it; use the play button to listen.'),
        helpKeyHtml('volume icon', 'Mute or unmute.'),
        helpKeyHtml('volume bar', 'Adjust playback volume.'),
        helpKeyHtml('waveform', 'Click or drag to seek. Green shows the listened region; the dashed amber line is the preview start cue.'),
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
  function toastStackEl() {
    let stack = document.getElementById('toast-stack');
    if (!stack) {
      stack = document.createElement('div');
      stack.id = 'toast-stack';
      stack.className = 'toast-stack';
      stack.setAttribute('aria-live', 'polite');
      document.body.appendChild(stack);
    }
    return stack;
  }
  function dismissToast(toast) {
    if (!toast) return;
    if (toast.__seqToastTimer) window.clearTimeout(toast.__seqToastTimer);
    if (toast.parentNode) toast.parentNode.removeChild(toast);
  }
  function showToast(message, opts={}) {
    const stack = toastStackEl();
    const tone = opts.tone === 'warn' || opts.tone === 'error' ? opts.tone : 'info';
    // Single-level undo per group: a new grouped toast supersedes any older undo
    // toast in the same group so its now-stale snapshot can no longer be applied.
    if (opts.undoGroup) {
      Array.from(stack.children).forEach(child => {
        if (child.__seqToastUndoGroup === opts.undoGroup) dismissToast(child);
      });
    }
    const toast = document.createElement('div');
    toast.className = 'toast toast-' + tone;
    toast.__seqToastUndoGroup = opts.undoGroup || null;
    const text = document.createElement('span');
    text.className = 'toast-message';
    text.textContent = String(message == null ? '' : message);
    toast.appendChild(text);
    if (typeof opts.onUndo === 'function') {
      const undo = document.createElement('button');
      undo.type = 'button';
      undo.className = 'toast-undo';
      undo.textContent = opts.undoLabel || 'Undo';
      undo.addEventListener('click', ev => {
        ev.stopPropagation();
        dismissToast(toast);
        safeUi('toast undo', opts.onUndo);
        showToast('Restored.', { duration: 2500 });
      });
      toast.appendChild(undo);
    }
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'toast-close';
    close.setAttribute('aria-label', 'Dismiss notification');
    close.textContent = '✕';
    close.addEventListener('click', ev => {
      ev.stopPropagation();
      dismissToast(toast);
    });
    toast.appendChild(close);
    stack.appendChild(toast);
    while (stack.children.length > 3) dismissToast(stack.firstElementChild);
    const duration = Number(opts.duration);
    toast.__seqToastTimer = window.setTimeout(() => dismissToast(toast), Number.isFinite(duration) && duration > 0 ? duration : 6000);
    return toast;
  }
  function sequenceSnapshot() {
    return {
      sequence: sequence.slice(),
      targetValues: targetValues.slice(),
      sequenceLength: sequenceLength,
      selectedSlot: selectedSlot,
    };
  }
  function restoreSequenceSnapshot(snap) {
    if (!snap) return;
    sequenceLength = snap.sequenceLength;
    sequence = snap.sequence.slice();
    targetValues = snap.targetValues.slice();
    selectedSlot = snap.selectedSlot;
    if (els.sequenceLength) els.sequenceLength.value = String(sequenceLength);
    scheduleSessionSave();
    renderAll();
  }
  function sequenceTrackLabel(idx) {
    const record = idx === null ? null : byIdx.get(Number(idx));
    return record && record.title ? String(record.title) : 'track';
  }
  const SESSION_STORAGE_KEY = 'djSequenceBuilder.session.v1';
  let sessionSaveTimer = null;
  let sessionSaveSuspended = false;
  let datasetSignatureCache = null;
  function datasetSignature() {
    if (datasetSignatureCache) return datasetSignatureCache;
    let hash = 5381;
    for (let i = 0; i < records.length; i += 1) {
      const record = records[i];
      const part = i + ':' + (trackId(record) || String((record && record.title) || '')) + ';';
      for (let j = 0; j < part.length; j += 1) hash = (hash * 33 + part.charCodeAt(j)) >>> 0;
    }
    datasetSignatureCache = records.length + '.' + hash.toString(36);
    return datasetSignatureCache;
  }
  function collectSessionState() {
    return {
      v: 1,
      sig: datasetSignature(),
      savedAt: Date.now(),
      sequence: sequence.slice(),
      targetValues: targetValues.slice(),
      sequenceLength: sequenceLength,
      selectedSlot: selectedSlot,
      transitionFromIdx: transitionFromIdx,
      transitionToIdx: transitionToIdx,
      pinned: Array.from(pinnedRecommendationIdxs),
      appSettings: Object.assign({}, appSettings.current || {}),
      previewFormState: Object.assign({}, previewFormState),
    };
  }
  function saveSessionNow() {
    if (sessionSaveSuspended) return;
    try {
      window.localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(collectSessionState()));
    } catch (err) {}
  }
  function scheduleSessionSave() {
    if (sessionSaveSuspended) return;
    if (sessionSaveTimer) window.clearTimeout(sessionSaveTimer);
    sessionSaveTimer = window.setTimeout(() => {
      sessionSaveTimer = null;
      saveSessionNow();
    }, 300);
  }
  function flushSessionSave() {
    if (sessionSaveTimer) {
      window.clearTimeout(sessionSaveTimer);
      sessionSaveTimer = null;
      saveSessionNow();
    }
  }
  function savedSessionInfo() {
    try {
      const raw = window.localStorage.getItem(SESSION_STORAGE_KEY);
      if (!raw) return null;
      const data = JSON.parse(raw);
      return data && data.v === 1 ? data : null;
    } catch (err) {
      return null;
    }
  }
  function clearSavedSession() {
    if (sessionSaveTimer) {
      window.clearTimeout(sessionSaveTimer);
      sessionSaveTimer = null;
    }
    // Suspend auto-save so the clear actually sticks for this page session
    // (otherwise the next edit would immediately re-create the saved snapshot).
    sessionSaveSuspended = true;
    try { window.localStorage.removeItem(SESSION_STORAGE_KEY); } catch (err) {}
  }
  function restoreSession() {
    try {
      const data = savedSessionInfo();
      if (!data || data.sig !== datasetSignature()) return false;
      const validIdx = v => {
        if (v === null || v === undefined) return null;
        const idx = Number(v);
        return Number.isFinite(idx) && byIdx.has(idx) ? idx : null;
      };
      const n = clamp(Math.round(Number(data.sequenceLength || sequenceLength)), 2, 40);
      sequenceLength = n;
      const seq = Array.isArray(data.sequence) ? data.sequence : [];
      sequence = Array.from({ length: n }, (_, i) => validIdx(seq[i]));
      const targets = Array.isArray(data.targetValues) ? data.targetValues : [];
      targetValues = Array.from({ length: n }, (_, i) => normalizeTargetEnergy(targets[i] === null ? undefined : targets[i]));
      const slot = data.selectedSlot === null || data.selectedSlot === undefined ? NaN : Math.round(Number(data.selectedSlot));
      selectedSlot = Number.isFinite(slot) && slot >= 0 && slot < n ? slot : null;
      transitionFromIdx = validIdx(data.transitionFromIdx);
      transitionToIdx = validIdx(data.transitionToIdx);
      if (transitionToIdx !== null && transitionToIdx === transitionFromIdx) transitionToIdx = null;
      pinnedRecommendationIdxs.clear();
      (Array.isArray(data.pinned) ? data.pinned : []).forEach(v => {
        const idx = validIdx(v);
        if (idx !== null) pinnedRecommendationIdxs.add(idx);
      });
      if (data.appSettings && typeof data.appSettings === 'object') {
        appSettings.current = Object.assign({}, appSettings.defaults, data.appSettings);
      }
      if (data.previewFormState && typeof data.previewFormState === 'object') {
        Object.keys(previewFormState).forEach(key => {
          if (data.previewFormState[key] !== undefined) previewFormState[key] = data.previewFormState[key];
        });
      }
      if (els.sequenceLength) els.sequenceLength.value = String(sequenceLength);
      applyAppSettingsToControls();
      return true;
    } catch (err) {
      clearSavedSession();
      return false;
    }
  }
  function selectedMapRenderer() {
    const raw = String(appSetting('map_renderer', els.mapRenderer ? els.mapRenderer.value : 'plotly')).toLowerCase();
    return raw === 'webgl' ? 'webgl' : 'plotly';
  }
  function wantsWebglMap() {
    return selectedMapRenderer() === 'webgl';
  }
  function usingWebglMap() {
    return wantsWebglMap() && webglMap && webglMap.available();
  }
  function fallbackToPlotlyMap(reason) {
    if (reason) reportUiError('webgl map fallback', reason);
    setAppSetting('map_renderer', 'plotly');
    if (els.mapRenderer) els.mapRenderer.value = 'plotly';
    const pane = plotPaneEl();
    if (pane) pane.classList.remove('map-renderer-webgl');
    webglMap = null;
    if (window.Plotly && plot) {
      setTimeout(() => Plotly.Plots.resize(plot), 20);
    }
  }
  function plotPaneEl() {
    return els.webglMap ? els.webglMap.closest('.plot-pane') : document.querySelector('.plot-pane');
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
    function hitTest(ev, radius=18) {
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
        const rgba = [0.9725, 0.9804, 0.9882, alpha];
        lineColors.set(rgba, colorOffset);
        lineColors.set(rgba, colorOffset + 4);
        lineCount += 1;
      });
      gl.useProgram(program);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.clearColor(0.066, 0.094, 0.145, 1);
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
    const mode = selectedMapRenderer();
    if (els.mapRenderer && els.mapRenderer.value !== mode) els.mapRenderer.value = mode;
    const pane = plotPaneEl();
    if (mode === 'webgl') {
      if (!webglMap) {
        try {
          webglMap = createWebglMapRenderer(els.webglMap);
        } catch (err) {
          reportUiError('webgl map init', err);
          webglMap = null;
        }
      }
      if (webglMap && webglMap.available()) {
        if (pane) pane.classList.add('map-renderer-webgl');
        try {
          if (!webglMap.viewRanges()) webglMap.fitBounds();
          if (render) webglMap.render();
          return 'webgl';
        } catch (err) {
          fallbackToPlotlyMap(err);
          return 'plotly';
        }
      }
      fallbackToPlotlyMap('WebGL is unavailable in this browser.');
      return 'plotly';
    }
    if (pane) pane.classList.remove('map-renderer-webgl');
    return 'plotly';
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
    const target = targetCurve()[slotNumber] ?? 5;
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
    const energy = '<span class="track-energy" title="Track energy (1-9)" style="color:' + esc(energyTextColor(value)) + '">' + esc(fmt(value, 2)) + '</span>';
    if (!Number.isFinite(residual)) return energy;
    const extent = targetEnergyResidualExtentForSlot(resolvedSlot);
    const maxAbs = Math.max(Math.abs(Number(extent.min) || 0), Math.abs(Number(extent.max) || 0), 1);
    const residualTitle = 'Offset vs slot ' + (Number(resolvedSlot) + 1) + ' target ' + fmt(value - residual, 1);
    return energy + ' <span class="track-energy-residual" title="' + esc(residualTitle) + '" style="color:' + esc(twoSidedResidualColor(residual, maxAbs)) + '">(' + esc(signedFmt(residual, 2)) + ')</span>';
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
      '<div class="track-meta-line muted">' + trackMetaHtml(record, { slot, showEnergy, showResidual }) + '</div>';
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
    transitionRender = null;
    transitionRenderedState = null;
    transitionRenderPending = false;
    transitionRenderMessage = '';
    transitionRenderWarn = false;
    transitionRenderRequestId += 1;
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
  function numericInput(id, label, value, min, max, step, title='') {
    return '<label' + (title ? ' title="' + esc(title) + '"' : '') + '>' + esc(label) + '<input id="' + id + '" data-preview-field="' + id + '" type="number" min="' + min + '" max="' + max + '" step="' + step + '" value="' + esc(value) + '"></label>';
  }
  function selectInput(id, label, values, selectedValue, labelFn) {
    const labels = labelFn || (v => v);
    return '<label>' + esc(label) + '<select id="' + id + '" data-preview-field="' + id + '">' +
      values.map(v => optionHtml(v, labels(v), selectedValue)).join('') +
      '</select></label>';
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
    const transitionStart = anchor + (nudgeBeats * beatSec);
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
    return clamp(n, bounds.min, bounds.max);
  }
  function presetDetails(name) {
    return (appOptions && appOptions.preset_details && appOptions.preset_details[name]) || {};
  }
  function applyPresetToPreviewState(name) {
    previewFormState.preset = name || 'auto';
    const preset = presetDetails(previewFormState.preset);
    previewFormState.volume_mode = preset.volume_mode || previewFormState.volume_mode || 'crossfade';
    previewFormState.eq_mode = preset.eq_mode || previewFormState.eq_mode || 'none';
    previewFormState.filter_mode = preset.filter_mode || previewFormState.filter_mode || 'none';
    ['volume_mode', 'eq_mode', 'filter_mode'].forEach(id => {
      const el = document.getElementById(id);
      if (el && previewFormState[id] != null) el.value = previewFormState[id];
    });
    const presetEl = document.getElementById('preset');
    if (presetEl) presetEl.value = previewFormState.preset;
  }
  function resolvedPreviewModes() {
    return {
      volume: previewFormState.volume_mode || 'crossfade',
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
      from_pitch_shift: Number(s.from_pitch_shift) || 0,
      to_pitch_shift: Number(s.to_pitch_shift) || 0,
      volume_mode: s.volume_mode || 'crossfade',
      eq_mode: s.eq_mode || 'none',
      filter_mode: s.filter_mode || 'none',
    });
  }
  function previewSettingsDirty() {
    return !!(transitionRender && transitionRenderedState && previewSignature(previewFormState) !== previewSignature(transitionRenderedState));
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
      button.textContent = transitionRenderPending ? 'Rendering...' : (transitionRender && dirty ? 'Render again' : 'Render transition');
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
  }
  function showTransitionPendingUi() {
    updateTransitionDirtyUi();
    const transport = document.querySelector('.transition-transport-row');
    if (transport && !transitionRender) transport.outerHTML = transitionTransportHtml();
  }
  function updatePreviewEffectDisplay({ dirty=true } = {}) {
    safeUi('transition editor draw', drawTransitionEditor);
    if (dirty) safeUi('transition dirty ui update', updateTransitionDirtyUi);
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
      appOptions = { error: 'Server data unavailable: ' + (err.message || String(err)) };
    }
    renderTransitionPreview();
  }
  function setActivePane(pane, { render=true } = {}) {
    activePane = pane || 'explore';
    els.tabButtons.forEach(btn => {
      const active = btn.getAttribute('data-pane-tab') === activePane;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-selected', active ? 'true' : 'false');
      btn.setAttribute('tabindex', active ? '0' : '-1');
    });
    Object.keys(els.panes).forEach(key => {
      if (els.panes[key]) els.panes[key].classList.toggle('active', key === activePane);
    });
    if (activePane === 'explore') {
      setTimeout(() => safeUi('map renderer sync', () => syncMapRenderer()), 20);
    }
    if (activePane === 'explore' && selectedMapRenderer() === 'plotly' && window.Plotly && plot) {
      setTimeout(() => Plotly.Plots.resize(plot), 30);
      setTimeout(ensureMapEffectsLoop, 40);
    }
    // Energy curve and transition editor live in the always-visible right column,
    // so keep them sized/drawn whenever the analysis view changes width.
    if (window.Plotly && els.energyCurve) {
      setTimeout(() => Plotly.Plots.resize(els.energyCurve), 30);
    }
    setTimeout(() => safeUi('transition editor draw', requestTransitionEditorDraw), 40);
    if (render) setTimeout(() => safeUi('pane render', renderAll), 0);
  }
  function setBuildSegment(seg, { persist=true } = {}) {
    activeBuildSegment = BUILD_SEGMENTS.indexOf(seg) >= 0 ? seg : 'sequence';
    els.buildSegButtons.forEach(btn => {
      const active = btn.getAttribute('data-build-seg') === activeBuildSegment;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-selected', active ? 'true' : 'false');
      btn.setAttribute('tabindex', active ? '0' : '-1');
    });
    els.buildSegPanels.forEach(panel => {
      panel.classList.toggle('hidden', panel.getAttribute('data-build-seg-panel') !== activeBuildSegment);
    });
    if (persist) setAppSetting('build_segment', activeBuildSegment);
    // Canvas/Plotly content renders at zero size while its panel is hidden, so
    // size/redraw the active segment's visuals once it becomes visible.
    if (activeBuildSegment === 'sequence' && window.Plotly && els.energyCurve) {
      setTimeout(() => safeUi('energy curve resize', () => Plotly.Plots.resize(els.energyCurve)), 20);
    }
    if (activeBuildSegment === 'recommend') {
      setTimeout(() => safeUi('recommendation scrubbers', updateRowScrubbers), 20);
    }
    if (activeBuildSegment === 'transition') {
      setTimeout(() => safeUi('transition editor draw', requestTransitionEditorDraw), 20);
    }
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
      '<canvas class="row-waveform" data-library-waveform-idx="' + record.idx + '" data-duration="' + esc(max) + '" height="30" aria-label="Playback scrubber" title="Preview scrubber — green shows the listened region (click or drag to seek); the dashed amber line is the preview start cue"></canvas>' +
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
  }
  function updateGlobalPlayer() {
    // Empty until the user picks a track: no auto-loaded default. Clicking a track
    // (map dot, table row, deck play) loads it here.
    const record = currentAudioRecord();
    const hasRecord = !!(record && audioUri(record));
    const activeRecord = record && previewAudioIdx === Number(record.idx);
    const playing = activeRecord && previewAudioPlaying;
    const duration = audioDuration(record);
    const current = activeRecord && els.audio && Number.isFinite(Number(els.audio.currentTime))
      ? Number(els.audio.currentTime)
      : rowScrubValue(record);
    if (els.globalArt) {
      if (record && record.artwork_uri) {
        els.globalArt.className = 'global-art';
        els.globalArt.innerHTML = '<img src="' + esc(record.artwork_uri) + '" alt="">';
      } else {
        els.globalArt.className = 'global-art art-placeholder';
        els.globalArt.textContent = '';
      }
    }
    if (els.globalTitle) {
      els.globalTitle.textContent = record ? String(record.title || 'Untitled') : '';
    }
    if (els.globalArtist) {
      els.globalArtist.innerHTML = record
        ? '<span>' + esc(record.artists || '') + '</span><span class="global-meta-line">' + trackMetaHtml(record, { slot: relevantResidualSlot(record) }) + '</span>'
        : '';
    }
    if (els.globalToggle) {
      els.globalToggle.disabled = !hasRecord;
      els.globalToggle.textContent = playing ? '⏸' : '▶';
      els.globalToggle.setAttribute('aria-label', playing ? 'Pause' : 'Play');
      els.globalToggle.setAttribute('title', playing ? 'Pause' : 'Play');
    }
    if (els.globalScrub) {
      const max = Number.isFinite(duration) && duration > 0 ? duration : 1;
      els.globalScrub.disabled = !hasRecord;
      els.globalScrub.max = String(max);
      els.globalScrub.value = String(clamp(current || 0, 0, max));
      setRangeProgress(els.globalScrub, current || 0, max);
    }
    if (els.globalTime) els.globalTime.textContent = timeText(current || 0) + ' / ' + timeText(duration);
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
  }
  function syncAudioUi() {
    previewAudioPlaying = !!(els.audio && !els.audio.paused && !els.audio.ended);
    updatePlayButtons();
    updateRowScrubbers();
    updateTransitionTrackPlayButtons();
    updateGlobalPlayer();
    drawMainWaveform();
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
    const url = audioUri(record);
    if (!url) return;
    const tAudio = transitionAudioElement();
    if (tAudio && !tAudio.paused) { try { tAudio.pause(); } catch (err) {} }
    const start = Math.max(0, Number(startSeconds) || 0);
    const idx = Number(record.idx);
    const sourceChanged = previewAudioUrl !== url;
    previewAudioIdx = idx;
    previewAudioContext = context;
    previewAudioStart = start;
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
      if (promise && promise.catch) promise.catch(err => {
        if (!err || err.name !== 'NotAllowedError') return;
        previewAudioPlaying = false;
        showToast('Playback was blocked by the browser. Press play again.', { tone: 'warn' });
        syncAudioUi();
      });
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
    renderSelectionOnly();
  }
  function setCandidateTrack(idx, { autoplay=null, showPopover=true, render=true } = {}) {
    idx = Number(idx);
    if (!byIdx.has(idx)) return;
    selectedIdx = idx;
    scheduleSessionSave();
    const record = byIdx.get(idx);
    const doAutoplay = autoplay === null ? Boolean(appSetting('click_to_play', true)) : Boolean(autoplay);
    if (showPopover) showSongPopover(record, { autoplay: doAutoplay });
    else hideSongPopover({ stopAudio: false });
    if (render) renderSelectionOnly();
  }
  function showSongPopover(record, { autoplay=true } = {}) {
    if (!els.songPopover || !record) return;
    els.songPopover.classList.remove('hidden');
    // Track identity (art/title/artist/meta) is shown by the bottom player bar.
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
        safeUi('map effects hover', ensureMapEffectsLoop);
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
      safeUi('map effects hover', ensureMapEffectsLoop);
    }
    if (els.songHoverCard) els.songHoverCard.classList.add('hidden');
  }
  function toggleTrackPreview(idx, { mode='point' } = {}) {
    idx = Number(idx);
    if (!byIdx.has(idx)) return;
    const record = byIdx.get(idx);
    const context = mode === 'library' || mode === 'diagnostics' ? mode : 'point';
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
    playRecordAt(info.record, Math.max(0, Number(info.transitionStart) || 0), context);
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
    if (mode === 'eq:center-bass-swap') return deck === 'a' ? (half ? 1 : 0.15) : (half ? 0.15 : 1);
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
  function updateMapLegendRestore() {
    const btn = els.mapLegendRestore;
    if (!btn) return;
    let hiddenCount = 0;
    if (!usingWebglMap() && plot && Array.isArray(plot.data)) {
      baseTraceIndices.forEach(traceIdx => {
        const trace = plot.data[traceIdx];
        if (trace && trace.visible === 'legendonly') hiddenCount += 1;
      });
    }
    btn.classList.toggle('hidden', hiddenCount === 0);
    btn.textContent = hiddenCount ? 'Show all genres (' + hiddenCount + ' hidden)' : 'Show all genres';
  }
  function restoreHiddenGenreTraces() {
    if (window.Plotly && plot && baseTraceIndices.length) {
      Plotly.restyle(plot, { visible: true }, baseTraceIndices);
    }
    updateMapLegendRestore();
    safeUi('map overlays after legend restore', updatePacmapOverlays);
    safeUi('map effects after legend restore', ensureMapEffectsLoop);
    safeUi('mini map after legend restore', drawMiniMap);
  }
  const simplexVertices = {
    tempo: { x: 180, y: 34 },
    groove: { x: 44, y: 270 },
    chroma: { x: 316, y: 270 },
  };
  function mixWeightsRaw() {
    const te = Math.max(0, Number(els.weightTempo ? els.weightTempo.value : 0));
    const gr = Math.max(0, Number(els.weightGroove ? els.weightGroove.value : 0));
    const ch = Math.max(0, Number(els.weightChroma ? els.weightChroma.value : 0));
    const s = te + gr + ch;
    if (s <= 1e-12) return { tempo: 1/3, groove: 1/3, chroma: 1/3 };
    return { tempo: te / s, groove: gr / s, chroma: ch / s };
  }
  function setMixSliders(mix) {
    if (els.weightTempo) els.weightTempo.value = String(clamp(mix.tempo, 0, 1));
    if (els.weightGroove) els.weightGroove.value = String(clamp(mix.groove, 0, 1));
    if (els.weightChroma) els.weightChroma.value = String(clamp(mix.chroma, 0, 1));
  }
  function simplexPoint(mix) {
    return {
      x: mix.tempo * simplexVertices.tempo.x + mix.groove * simplexVertices.groove.x + mix.chroma * simplexVertices.chroma.x,
      y: mix.tempo * simplexVertices.tempo.y + mix.groove * simplexVertices.groove.y + mix.chroma * simplexVertices.chroma.y,
    };
  }
  function updateSimplexHandle(mix) {
    if (!els.simplexHandle) return;
    const p = simplexPoint(mix);
    els.simplexHandle.setAttribute('cx', String(p.x));
    els.simplexHandle.setAttribute('cy', String(p.y));
  }
  function simplexWeightsFromPoint(px, py) {
    const a = simplexVertices.tempo;
    const b = simplexVertices.groove;
    const c = simplexVertices.chroma;
    const denom = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
    let te = ((b.y - c.y) * (px - c.x) + (c.x - b.x) * (py - c.y)) / denom;
    let gr = ((c.y - a.y) * (px - c.x) + (a.x - c.x) * (py - c.y)) / denom;
    let ch = 1 - te - gr;
    te = clamp(te, 0, 1); gr = clamp(gr, 0, 1); ch = clamp(ch, 0, 1);
    const s = Math.max(1e-12, te + gr + ch);
    return { tempo: te / s, groove: gr / s, chroma: ch / s };
  }
  function eventToSvgPoint(ev) {
    const rect = els.simplex.getBoundingClientRect();
    return { x: (ev.clientX - rect.left) * 360 / Math.max(1, rect.width), y: (ev.clientY - rect.top) * 320 / Math.max(1, rect.height) };
  }
  function setWeightsFromSimplexEvent(ev) {
    const p = eventToSvgPoint(ev);
    setMixSliders(simplexWeightsFromPoint(p.x, p.y));
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
    if (config.control_mode === 'genre-mixability') {
      const style = clamp(Number(els.weightStyle ? els.weightStyle.value : 0.45), 0, 1);
      const mix = mixWeightsRaw();
      const m = 1 - style;
      return { maest: style, tempo: m * mix.tempo, groove: m * mix.groove, chroma: m * mix.chroma, mix };
    }
    const ma = Math.max(0, Number(els.weightMaest ? els.weightMaest.value : 0));
    const ch = Math.max(0, Number(els.weightChroma ? els.weightChroma.value : 0));
    const te = Math.max(0, Number(els.weightTempo ? els.weightTempo.value : 0));
    const s = ma + ch + te;
    if (s <= 1e-12) return { maest: 1/3, chroma: 1/3, tempo: 1/3, groove: 0 };
    return { maest: ma/s, chroma: ch/s, tempo: te/s, groove: 0 };
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
  function artworkMarkersEnabled() {
    return Boolean(appSetting('map_artwork', els.mapArtworkEnabled ? els.mapArtworkEnabled.checked : true));
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
    if (els.mapRenderer) els.mapRenderer.value = selectedMapRenderer();
    if (els.mapEffectsEnabled) els.mapEffectsEnabled.checked = Boolean(appSetting('map_fx', true));
    if (els.mapArtworkEnabled) els.mapArtworkEnabled.checked = artworkMarkersEnabled();
    if (els.showScoreValues) els.showScoreValues.checked = showScoreValues();
    if (els.clickToPlay) els.clickToPlay.checked = Boolean(appSetting('click_to_play', true));
    if (els.latentLinksPerTrack) els.latentLinksPerTrack.value = String(latentLinksPerTrack());
    if (els.recommendedLinksHighlight) els.recommendedLinksHighlight.value = String(recommendedLinksHighlight());
    if (els.sideSettings) els.sideSettings.open = Boolean(appSetting('side_settings_open', false));
    applyMiniMapCollapsedState();
    updateMapLinkSettingLabels();
  }
  function weightedCandidateScore(c) {
    const w = weights();
    const useNorm = config.control_mode === 'genre-mixability';
    const styleScore = useNorm ? Number(c.maest_score_norm ?? c.maest_similarity ?? 0) : Number(c.maest_similarity || 0);
    const tempoScore = useNorm ? Number(c.tempo_score_norm ?? c.tempo_similarity ?? 0) : Number(c.tempo_similarity || 0);
    const grooveScore = useNorm ? Number(c.groove_score_norm ?? c.groove_similarity ?? 0) : 0;
    const keyScore = useNorm ? Number(c.chroma_score_norm ?? c.chroma_similarity ?? 0) : Number(c.chroma_similarity || 0);
    const score = w.maest * styleScore + w.tempo * tempoScore + w.groove * grooveScore + w.chroma * keyScore;
    return Number.isFinite(score) ? score : 0;
  }
  function layoutInterpolatedPoints(w) {
    if (!Array.isArray(layoutEntries) || !layoutEntries.length) {
      return records.map(r => idxToPoint[String(r.idx)] || [0, 0]);
    }
    const target = [w.maest, w.tempo, w.groove, w.chroma];
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
  function hiddenGenreNames() {
    const hidden = new Set();
    if (!usingWebglMap() && plot && Array.isArray(plot.data)) {
      baseTraceIndices.forEach(traceIdx => {
        const trace = plot.data[traceIdx];
        if (trace && trace.visible === 'legendonly' && trace.name != null) hidden.add(String(trace.name));
      });
    }
    return hidden;
  }
  function recordGenreHidden(record, hiddenSet) {
    if (!record || !hiddenSet || !hiddenSet.size) return false;
    return hiddenSet.has(String(record.genre)) || hiddenSet.has(String(record.raw_genre));
  }
  function nearestTrackPointIdx(ev, radius=18) {
    if (!ev || !els.mapEffects) return null;
    const canvasRect = els.mapEffects.getBoundingClientRect();
    const x = Number(ev.clientX) - canvasRect.left;
    const y = Number(ev.clientY) - canvasRect.top;
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    const hiddenGenres = hiddenGenreNames();
    let bestIdx = null;
    let bestD2 = radius * radius;
    for (let i = 0; i < records.length; i += 1) {
      if (recordGenreHidden(records[i], hiddenGenres)) continue;
      const idx = Number(records[i].idx);
      const p = plotPointPx(currentPoint(idx));
      if (!p) continue;
      const dx = p.x - x;
      const dy = p.y - y;
      const d2 = dx * dx + dy * dy;
      if (d2 <= bestD2) {
        bestD2 = d2;
        bestIdx = idx;
      }
    }
    return bestIdx;
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
  function applyMiniMapCollapsedState() {
    const collapsed = Boolean(appSetting('mini_map_collapsed', false));
    if (els.mapMiniMap) els.mapMiniMap.classList.toggle('collapsed', collapsed);
    const pane = plotPaneEl();
    if (pane) pane.classList.toggle('mini-map-collapsed', collapsed);
    if (els.miniMapToggle) {
      els.miniMapToggle.textContent = collapsed ? '▴' : '▾';
      const label = collapsed ? 'Show mini map' : 'Hide mini map';
      els.miniMapToggle.setAttribute('aria-label', label);
      els.miniMapToggle.setAttribute('title', label);
      els.miniMapToggle.setAttribute('aria-pressed', collapsed ? 'false' : 'true');
    }
  }
  function drawMiniMap() {
    if (appSetting('mini_map_collapsed', false)) return;
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
    ctx.fillStyle = 'rgba(15, 23, 42, .92)';
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
    drawRing(selectedIdx, roleColors.selected, 4.5);
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
      ctx.strokeStyle = 'rgba(248, 250, 252, .88)';
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
      ctx.save();
      ctx.strokeStyle = 'rgba(248, 250, 252, ' + alpha.toFixed(3) + ')';
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
    ctx.save();
    ctx.fillStyle = 'rgba(15, 23, 42, .82)';
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
      ctx.fillStyle = '#f8fafc';
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
      drawStaticMapBadge(ctx, plotPointPx(currentPoint(hoveredIdx)), '', '#f8fafc', 13);
    }
    if (selectedIdx !== null) drawStaticMapBadge(ctx, plotPointPx(currentPoint(selectedIdx)), '', roleColors.selected, 13);
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
  const ARTWORK_MARKER_SIZE = 26;
  const artworkMarkerImages = new Map();
  function artworkMarkerThumb(uri) {
    if (!uri) return null;
    let entry = artworkMarkerImages.get(uri);
    if (!entry) {
      const img = new Image();
      entry = { img, thumb: null, failed: false };
      img.decoding = 'async';
      img.onload = () => { ensureMapEffectsLoop(); };
      img.onerror = () => { entry.failed = true; };
      img.src = uri;
      artworkMarkerImages.set(uri, entry);
    }
    if (entry.failed) return null;
    if (!entry.thumb) {
      if (!entry.img.complete || !entry.img.naturalWidth) return null;
      const scale = window.devicePixelRatio || 1;
      const size = Math.max(1, Math.round(ARTWORK_MARKER_SIZE * scale));
      const thumb = document.createElement('canvas');
      thumb.width = size;
      thumb.height = size;
      const tctx = thumb.getContext('2d');
      if (!tctx) return null;
      tctx.beginPath();
      tctx.arc(size / 2, size / 2, size / 2, 0, Math.PI * 2);
      tctx.clip();
      tctx.drawImage(entry.img, 0, 0, size, size);
      entry.thumb = thumb;
    }
    return entry.thumb;
  }
  function drawArtworkMarkers(recRows) {
    const canvas = els.mapArtwork;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 20 || rect.height < 20) return;
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.floor(rect.width * dpr));
    const height = Math.max(1, Math.floor(rect.height * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    if (!artworkMarkersEnabled() || usingWebglMap() || !plot) return;
    const radius = ARTWORK_MARKER_SIZE / 2 - 1;
    const margin = radius + 2;
    const seqIdxSet = new Set(sequence.filter(v => v !== null));
    const recIdxSet = new Set((recRows || []).map(row => Number(row.idx)));
    const hiddenGenres = hiddenGenreNames();
    ctx.lineWidth = 2;
    records.forEach(record => {
      if (!record || !record.artwork_uri) return;
      if (recordGenreHidden(record, hiddenGenres)) return;
      const idx = Number(record.idx);
      const p = plotPointPx(currentPoint(idx));
      if (!p || p.x < -margin || p.y < -margin || p.x > rect.width + margin || p.y > rect.height + margin) return;
      const thumb = artworkMarkerThumb(record.artwork_uri);
      if (!thumb) return;
      ctx.drawImage(thumb, p.x - radius, p.y - radius, radius * 2, radius * 2);
      let ring;
      if (idx === transitionFromIdx) ring = roleColors.track1;
      else if (idx === transitionToIdx) ring = roleColors.track2;
      else if (idx === selectedIdx || idx === hoveredIdx) ring = roleColors.selected;
      else if (seqIdxSet.has(idx)) ring = roleColors.sequence;
      else if (recIdxSet.has(idx)) ring = roleColors.recommendation;
      else ring = genreMarkerColor(record.genre || record.raw_genre);
      ctx.strokeStyle = ring;
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      ctx.stroke();
    });
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
    safeUi('artwork markers', () => drawArtworkMarkers(recContext.sourceIdx === null ? [] : rows));

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
    if (els.weightStyleVal) els.weightStyleVal.textContent = fmt(w.maest, 2);
    if (els.weightMaestVal) els.weightMaestVal.textContent = fmt(w.maest, 2);
    if (els.weightChromaVal) els.weightChromaVal.textContent = fmt(w.chroma, 2);
    if (els.weightTempoVal) els.weightTempoVal.textContent = fmt(w.tempo, 2);
    if (els.weightGrooveVal) els.weightGrooveVal.textContent = fmt(w.groove, 2);
    if (w.mix) updateSimplexHandle(w.mix);
    els.penaltyScaleVal.textContent = fmt(Number(els.penaltyScale.value || 0), 2);
  }
  function energyPenaltyScale() {
    return clamp(Number(els.penaltyScale ? els.penaltyScale.value : 0), 0, 1);
  }
  function energyAdjustment(rawError, baseline) {
    const scale = energyPenaltyScale();
    const errorMagnitude = Math.abs(Number(rawError) || 0);
    const energyErrorPower = errorMagnitude * errorMagnitude;
    const energyScore = 1 / (1 + scale * energyErrorPower);
    const numericBaseline = Number(baseline);
    const hasBaseline = Number.isFinite(numericBaseline);
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
      const knownIdxs = new Set(allRows.map(row => Number(row.idx)));
      const stubSlot = targetSlot();
      const usedIdxs = selectedIndicesExcept(stubSlot);
      pinnedRecommendationIdxs.forEach(pinnedIdx => {
        const idx = Number(pinnedIdx);
        if (knownIdxs.has(idx) || !byIdx.has(idx)) return;
        if (recFilters.excludeUsed && usedIdxs.has(idx)) return;
        pinnedRows.push({
          idx,
          pinned: true,
          pinnedOutsideQuery: false,
          globalRank: null,
          slot: stubSlot,
          finalScore: NaN,
          baseline: NaN,
          energyScore: NaN,
          penalty: NaN,
          styleScore: NaN,
          tempoScore: NaN,
          grooveScore: NaN,
          keyScore: NaN,
        });
      });
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
    if (selectedIdx !== null && byIdx.has(Number(selectedIdx))) {
      return { mode: 'selected', slot, sourceIdx: Number(selectedIdx), sourceSlot: null };
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
    let nextFrom = transitionFromIdx;
    let nextTo = transitionToIdx;
    if (kind === 'out') {
      nextFrom = toggle && transitionFromIdx === idx ? null : idx;
      if (nextTo === idx) nextTo = null;
    } else {
      nextTo = toggle && transitionToIdx === idx ? null : idx;
      if (nextFrom === idx) nextFrom = null;
    }
    if (nextFrom === transitionFromIdx && nextTo === transitionToIdx) return;
    resetTransitionRenderState();
    previewFormState.from_cue = '';
    previewFormState.to_cue = '';
    transitionFromIdx = nextFrom;
    transitionToIdx = nextTo;
    scheduleSessionSave();
    renderAll();
  }
  function assignTrackTwo(idx) {
    idx = Number(idx);
    if (!Number.isFinite(idx) || !byIdx.has(idx)) return;
    if (transitionFromIdx === null) {
      showToast('Assign Track 1 first — Track 2 is the destination of a transition.', { tone: 'warn' });
      return;
    }
    if (Number(transitionFromIdx) === idx) {
      showToast('Track 2 must be a different track than Track 1.', { tone: 'warn' });
      return;
    }
    setTransitionEndpoint('in', idx, { toggle: false });
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
    resetTransitionRenderState();
    previewFormState.from_cue = '';
    previewFormState.to_cue = '';
    scheduleSessionSave();
    renderAll();
  }
  function clearTransitionPair() {
    transitionFromIdx = null;
    transitionToIdx = null;
    lastClickedIdx = null;
    lastClickMs = 0;
    lastPointDoubleClickMs = 0;
    lastPointDoubleClickIdx = null;
    resetTransitionRenderState();
    previewFormState.from_cue = '';
    previewFormState.to_cue = '';
    scheduleSessionSave();
    renderAll();
  }
  function setSequenceLength(n) {
    n = clamp(Math.round(Number(n || 10)), 2, 40);
    const snap = sequenceSnapshot();
    const old = sequence.slice();
    const oldTargets = targetValues.slice();
    const dropped = old.slice(n).filter(v => v !== null).length;
    sequenceLength = n;
    sequence = Array.from({ length: n }, (_, i) => i < old.length ? old[i] : null);
    targetValues = Array.from({ length: n }, (_, i) => i < oldTargets.length
      ? normalizeTargetEnergy(oldTargets[i])
      : DEFAULT_TARGET_ENERGY);
    selectedSlot = selectedSlot !== null && selectedSlot < n ? selectedSlot : null;
    scheduleSessionSave();
    renderAll();
    if (dropped) {
      showToast('Sequence length set to ' + n + ' (' + dropped + ' track' + (dropped === 1 ? '' : 's') + ' dropped).', {
        undoGroup: 'sequence',
        onUndo: () => restoreSequenceSnapshot(snap),
      });
    }
  }
  function normalizeTargetEnergy(value, fallback=DEFAULT_TARGET_ENERGY) {
    const n = Number(value);
    if (!Number.isFinite(n)) return fallback;
    return Math.round(clamp(n, 1, 9) * 2) / 2;
  }
  function setTargetSlotValue(slot, value, shouldRender=true) {
    slot = Math.round(Number(slot));
    value = normalizeTargetEnergy(value, NaN);
    if (!Number.isFinite(slot) || !Number.isFinite(value)) return;
    targetValues[slot] = value;
    scheduleSessionSave();
    if (shouldRender) renderAll();
  }
  function targetAnchors() {
    const anchors = [];
    for (let slot = 0; slot < targetValues.length; slot += 1) {
      const val = Number(targetValues[slot]);
      if (Number.isFinite(val)) anchors.push({ slot, value: clamp(val, 1, 9) });
    }
    return anchors.sort((a,b) => a.slot - b.slot);
  }
  function targetCurve() {
    const anchors = targetAnchors();
    const out = [];
    if (!anchors.length) {
      for (let i = 0; i < sequenceLength; i += 1) out.push(5);
      return out;
    }
    if (anchors.length === 1) return Array.from({ length: sequenceLength }, () => anchors[0].value);
    for (let i = 0; i < sequenceLength; i += 1) {
      if (i <= anchors[0].slot) { out.push(anchors[0].value); continue; }
      if (i >= anchors[anchors.length - 1].slot) { out.push(anchors[anchors.length - 1].value); continue; }
      let left = anchors[0], right = anchors[anchors.length - 1];
      for (let j = 0; j < anchors.length - 1; j += 1) {
        if (anchors[j].slot <= i && i <= anchors[j+1].slot) { left = anchors[j]; right = anchors[j+1]; break; }
      }
      const t = (i - left.slot) / Math.max(1, right.slot - left.slot);
      out.push(left.value + t * (right.value - left.value));
    }
    return out;
  }
  function scoreCandidate(c, slot) {
    const w = weights();
    const useNorm = config.control_mode === 'genre-mixability';
    const styleScore = useNorm ? Number(c.maest_score_norm ?? c.maest_similarity ?? 0) : Number(c.maest_similarity || 0);
    const tempoScore = useNorm ? Number(c.tempo_score_norm ?? c.tempo_similarity ?? 0) : Number(c.tempo_similarity || 0);
    const grooveScore = useNorm ? Number(c.groove_score_norm ?? c.groove_similarity ?? 0) : 0;
    const keyScore = useNorm ? Number(c.chroma_score_norm ?? c.chroma_similarity ?? 0) : Number(c.chroma_similarity || 0);
    const baseline = w.maest * styleScore + w.tempo * tempoScore + w.groove * grooveScore + w.chroma * keyScore;
    const record = byIdx.get(Number(c.idx));
    const e = energyOf(record);
    const target = targetCurve()[slot] ?? 5;
    const rawError = Number.isFinite(e) ? e - target : 0;
    const adjustment = energyAdjustment(rawError, baseline);
    return { baseline, styleScore, tempoScore, grooveScore, keyScore, energy: e, target, rawError, ...adjustment };
  }
  function scoreInitialCandidate(idx, slot) {
    const record = byIdx.get(Number(idx));
    const e = energyOf(record);
    const target = targetCurve()[slot] ?? 5;
    const rawError = Number.isFinite(e) ? e - target : 0;
    const adjustment = energyAdjustment(rawError, 1);
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
      tempoScore: NaN,
      grooveScore: NaN,
      keyScore: NaN,
      energy: e,
      target,
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
    const target = targetCurve()[slot] ?? 5;
    const rawError = Number.isFinite(e) ? e - target : 0;
    const adjustment = energyAdjustment(rawError, NaN);
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
      tempoScore: NaN,
      grooveScore: NaN,
      keyScore: NaN,
      energy: e,
      target,
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
        const targetEnergy = targetCurve()[slot] ?? 5;
        const energyRange = Number(recFilters.energyRange);
        const hasEnergyRange = String(recFilters.energyRange || '').trim() !== '' && Number.isFinite(energyRange) && energyRange >= 0;
        if (hasEnergyRange && Math.abs(energyOf(record) - targetEnergy) > energyRange) continue;
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
    const targetEnergy = targetCurve()[slot] ?? 5;
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
      if (hasEnergyRange && Math.abs(energyOf(record) - targetEnergy) > energyRange) continue;
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
    setCandidateTrack(idx, { showPopover: true, render: true });
  }
  function setCurrentTrack(idx) {
    idx = Number(idx);
    if (!byIdx.has(idx)) return;
    selectedIdx = idx;
    hideSongPopover({ stopAudio: false });
    scheduleSessionSave();
    renderSelectionOnly();
  }
  function appendTrack(idx, opts={}) {
    // Snapshot before deriving the slot so an explicit place-in-slot request does
    // not have to mutate the global selectedSlot first (undo then restores the
    // user's real prior selection, not the transient place target).
    const snap = sequenceSnapshot();
    const slot = opts.slot != null ? Math.round(Number(opts.slot)) : targetSlot();
    if (!(slot >= 0 && slot < sequence.length)) return;
    idx = Number(idx);
    if (!canPlaceTrack(idx, slot)) return;
    const pairSnap = {
      transitionFromIdx: transitionFromIdx,
      transitionToIdx: transitionToIdx,
      transitionRender: transitionRender,
      transitionRenderedState: transitionRenderedState,
      fromCue: previewFormState.from_cue,
      toCue: previewFormState.to_cue,
    };
    const replaced = sequence[slot] !== null;
    const promoted = transitionFromIdx !== idx;
    const clearedTrackTwo = transitionToIdx !== null;
    const discardedRender = transitionRender !== null;
    sequence[slot] = idx;
    selectedSlot = null;
    transitionFromIdx = idx;
    transitionToIdx = null;
    resetTransitionRenderState();
    previewFormState.from_cue = '';
    previewFormState.to_cue = '';
    scheduleSessionSave();
    setCandidateTrack(idx, { showPopover: true, render: false });
    renderAll();
    const parts = [(replaced ? 'Replaced slot ' : 'Added to slot ') + (slot + 1)];
    if (promoted) parts.push('Track 1 → ' + sequenceTrackLabel(idx));
    if (clearedTrackTwo) parts.push('Track 2 cleared');
    if (discardedRender) parts.push('rendered preview discarded');
    showToast(parts.join(' · '), {
      undoGroup: 'sequence',
      onUndo: () => {
        // Invalidate any render still in flight for the post-append pair before
        // restoring, so its result can't attach against the wrong pair's audio.
        resetTransitionRenderState();
        transitionFromIdx = pairSnap.transitionFromIdx;
        transitionToIdx = pairSnap.transitionToIdx;
        transitionRender = pairSnap.transitionRender;
        transitionRenderedState = pairSnap.transitionRenderedState;
        previewFormState.from_cue = pairSnap.fromCue;
        previewFormState.to_cue = pairSnap.toCue;
        restoreSequenceSnapshot(snap);
      },
    });
  }
  function removeSlot(slot) {
    slot = Number(slot);
    if (slot >= 0 && slot < sequence.length && sequence[slot] !== null) {
      const snap = sequenceSnapshot();
      const label = sequenceTrackLabel(sequence[slot]);
      sequence[slot] = null;
      scheduleSessionSave();
      showToast('Removed "' + label + '" from slot ' + (slot + 1) + '.', {
        undoGroup: 'sequence',
        onUndo: () => restoreSequenceSnapshot(snap),
      });
    }
    renderAll();
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
    scheduleSessionSave();
    renderAll();
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
    const sortTh = (key, label, cls='', title='') => {
      const active = librarySort === key;
      const indicator = active ? (libraryDir === 'asc' ? '▲' : '▼') : '↕';
      const ariaSort = active ? 'aria-sort="' + (libraryDir === 'asc' ? 'ascending' : 'descending') + '" ' : '';
      return '<th ' + (cls ? 'class="' + cls + '" ' : '') + (title ? 'title="' + esc(title) + '" ' : '') + ariaSort + 'data-library-sort="' + esc(key) + '" data-sort-indicator="' + indicator + '">' +
        '<button type="button" class="th-sort" data-library-sort="' + esc(key) + '">' + label + '</button></th>';
    };
    let html = '<table><thead><tr>' +
      sortTh('title', 'Track') +
      sortTh('artists', 'Artist') +
      sortTh('key', 'Key', 'short center') +
      sortTh('est_bpm', 'BPM', 'short center') +
      sortTh('human_energy', 'Tagged<br>energy', 'short center wrap-head', 'Energy rating tagged in the library metadata (1-9). Used for scoring when Energy source is Tagged energy.') +
      sortTh('glm_energy', 'Auto<br>energy', 'short center wrap-head', 'Energy estimated by a model. Continuous value that may not match the tag. Used for scoring when Energy source is Auto energy; falls back to the tag when missing.') +
      sortTh('duration_seconds', 'Duration', 'short center') +
      sortTh('raw_genre', 'Genre') +
      '<th>Actions</th></tr></thead><tbody>';
    rows.forEach(r => {
      const rowClasses = [];
      if (Number(r.idx) === selectedIdx) rowClasses.push('selected-slot');
      if (transitionFromIdx !== null && Number(r.idx) === Number(transitionFromIdx)) rowClasses.push('is-track1');
      if (transitionToIdx !== null && Number(r.idx) === Number(transitionToIdx)) rowClasses.push('is-track2');
      if (Number(r.idx) === previewAudioIdx) rowClasses.push(previewAudioPlaying ? 'now-playing' : 'now-playing-paused');
      const rowClass = rowClasses.length ? ' class="' + rowClasses.join(' ') + '"' : '';
      const titleAttr = [r.title, r.artists].filter(Boolean).join(' - ');
      html += '<tr data-library-row-idx="' + r.idx + '"' + rowClass + '>' +
        '<td class="library-title-cell" title="' + esc(titleAttr) + '"><div class="library-title-summary">' + artworkHtml(r) + '<span>' + esc(r.title) + '</span></div></td>' +
        '<td>' + esc(r.artists) + '</td>' +
        '<td class="center">' + keyHtml(r.key) + '</td>' +
        '<td class="center">' + roundedBpm(r) + '</td>' +
        '<td class="center">' + (Number.isFinite(Number(r.human_energy)) ? fmt(r.human_energy, 2) : '') + '</td>' +
        '<td class="center">' + (Number.isFinite(Number(r.glm_energy)) ? fmt(r.glm_energy, 2) : '') + '</td>' +
        '<td class="center">' + esc(durationText(r)) + '</td>' +
        '<td>' + esc(rawGenre(r)) + '</td>' +
        '<td><div class="library-actions">' +
        rowPlayerHtml(r) +
        '<button data-library-current="' + r.idx + '" title="Select this track (shows details and next-track recommendations)" aria-label="Select this track">Select</button>' +
        actionSymbolButton('data-library-place="' + r.idx + '"', '+', 'Add to sequence', 'primary') +
        actionSymbolButton('data-library-outgoing="' + r.idx + '"', 'T1', 'Set as Track 1', 'track-one') +
        actionSymbolButton('data-library-incoming="' + r.idx + '"', 'T2', 'Set as Track 2', 'track-two') +
        '</div></td></tr>';
    });
    html += '</tbody></table>';
    els.libraryTable.innerHTML = html;
    updatePlayButtons();
    updateRowScrubbers();
  }
  function renderSequence() {
    const targets = targetCurve();
    const lastFilled = sequence.reduce((acc, v, j) => v !== null ? j : acc, -1);
    const gapSlots = sequence.map((v, j) => v === null && j < lastFilled ? j + 1 : null).filter(v => v !== null);
    let html = gapSlots.length
      ? '<div class="sequence-gap-warning warn">Gap at slot' + (gapSlots.length > 1 ? 's' : '') + ' ' + gapSlots.join(', ') + ' — scores bridge across ' + (gapSlots.length > 1 ? 'them' : 'it') + '.</div>'
      : '';
    html += '<table><thead><tr><th class="num slot-col">Slot</th><th>Track</th><th class="target-col">Target</th><th class="num actual-col" title="Energy of the track placed in this slot — compare with the Target column">Track<br>energy</th><th class="actions-col"></th></tr></thead><tbody>';
    for (let i = 0; i < sequence.length; i += 1) {
      const idx = sequence[i];
      const r = idx === null ? null : byIdx.get(idx);
      const classes = [];
      if (selectedSlot === i) classes.push('selected-slot');
      if (sequenceDragSlot === i) classes.push('drag-source');
      const cls = classes.length ? ' class="' + classes.join(' ') + '"' : '';
      html += '<tr' + cls + ' data-sequence-drop-slot="' + i + '"><td class="num"><span class="sequence-drag-handle" draggable="true" data-sequence-drag-slot="' + i + '" title="Drag to reorder">↕</span> ' + (i + 1) + '</td><td>';
      // Energy is its own column in this table — keep it out of the meta line.
      if (r) html += trackSummaryHtml(r, { size: 'compact', showArt: true, slot: i, showEnergy: false });
      else if (i < lastFilled) html += '<span class="warn" title="Gap: transition scores and CSV export bridge across this slot">empty — gap</span>';
      else html += '<span class="muted">empty</span>';
      const targetValue = Number(targetValues[i]);
      const targetInput = '<input class="target-edit" data-target-slot="' + i + '" type="number" min="1" max="9" step="0.5" placeholder="' + fmt(targets[i], 2) + '" value="' + (Number.isFinite(targetValue) ? fmt(targetValue, 2) : '') + '" aria-label="Target energy for slot ' + (i + 1) + '">';
      html += '</td><td class="target-col">' + targetInput + '</td><td class="num actual-col">' + (r ? fmt(energyOf(r), 2) : '') + '</td><td class="actions-col"><div class="table-actions">';
      if (r) html += '<button class="play-button" data-play-idx="' + r.idx + '" aria-label="Play">▶</button> ';
      html += actionSymbolButton('data-seq-slot="' + i + '"', selectedSlot === i ? '●' : '○', selectedSlot === i ? 'Deselect slot' : 'Select slot', selectedSlot === i ? 'is-active' : '') + ' ';
      html += actionSymbolButton('data-move-slot-up="' + i + '"' + (i === 0 ? ' disabled' : ''), '↑', 'Move slot ' + (i + 1) + ' up') + ' ';
      html += actionSymbolButton('data-move-slot-down="' + i + '"' + (i === sequence.length - 1 ? ' disabled' : ''), '↓', 'Move slot ' + (i + 1) + ' down') + ' ';
      if (selectedIdx !== null && canPlaceTrack(selectedIdx, i)) {
        html += actionSymbolButton('data-place-slot="' + i + '"', r ? '⇄' : '+', r ? 'Replace with selected track' : 'Add selected track here', 'primary') + ' ';
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
  function scoreFeatureRows(score) {
    return [
      { key: 'style', label: 'Style / MAEST', value: Number(score && score.styleScore), weight: weights().maest },
      { key: 'tempo', label: 'Tempo', value: Number(score && score.tempoScore), weight: weights().tempo },
      { key: 'groove', label: 'Groove', value: Number(score && score.grooveScore), weight: weights().groove },
      { key: 'harmonic', label: 'Harmonic / chroma', value: Number(score && score.keyScore), weight: weights().chroma },
      { key: 'energy', label: 'Energy fit', value: Number(score && score.energyScore), weight: energyPenaltyScale() },
    ];
  }
  function featureBarsHtml(score, keys=null) {
    const rows = scoreFeatureRows(score).filter(row => !keys || keys.includes(row.key));
    return '<div class="diagnostic-feature-bars">' + rows.map(row => {
      const finite = Number.isFinite(row.value);
      const pct = finite ? clamp(row.value, 0, 1) * 100 : 0;
      return '<div class="diagnostic-feature-bar ' + esc(row.key) + '">' +
        '<span>' + esc(row.label) + '</span>' +
        '<div class="diagnostic-bar-track"><i style="width:' + pct.toFixed(1) + '%"></i></div>' +
        '<b>' + (finite ? fmt(row.value, 3) : 'n/a') + '</b>' +
        '<em>w ' + metricValue(row.weight, 2) + '</em>' +
        '</div>';
    }).join('') + '</div>';
  }
  function scoreSummaryBarsHtml(score) {
    const rows = [
      { key: 'final', label: 'Final', value: Number(score && score.finalScore), strength: Number(score && score.finalScore) },
      { key: 'mix', label: 'Mix', value: Number(score && score.baseline), strength: Number(score && score.baseline) },
      { key: 'penalty', label: 'Loss', value: Number.isFinite(Number(score && score.penalty)) ? -Number(score.penalty) : NaN, strength: Number(score && score.penalty) },
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
      '<div class="diagnostic-embedding-section"><h4>Chroma</h4>' +
      diagnosticHeatmapLegendHtml(extents.chroma, DIAGNOSTIC_CHROMA_HEATMAP_STOPS) +
      '<div class="diagnostic-embedding-stack">' +
      diagnosticChromaHtml(from, extents.chroma, 'Track 1') +
      diagnosticChromaHtml(to, extents.chroma, 'Track 2') +
      '</div></div>' +
      '<div class="diagnostic-embedding-section"><h4>Groove</h4>' +
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
    // The pair itself is always on screen in the right column's Track 1 / Track 2
    // cards, so this deep-dive view only labels the slot and goes straight to detail.
    return '<div class="diagnostic-subhead">Slot ' + (slot + 1) + (score && score.missing ? ' <span class="warn">missing direct similarity row</span>' : '') + '</div>' +
      scoreSummaryBarsHtml(score) +
      featureBarsHtml(score) +
      '<div class="focused-transition-section">' + diagnosticEmbeddingComparisonHtml(from, to) + '</div>' +
      '<div class="focused-transition-waveforms"><h3>Full track waveforms</h3>' +
      '<div class="diagnostic-subhead">Click or drag either waveform to play and scrub the full track.</div>' +
      diagnosticWaveformComparisonHtml(from, to, slot) + '</div>';
  }
  const SCORE_TIPS = {
    final: 'Final = Mix × Fit. Overall recommendation score.',
    mix: 'Mix: weighted blend of Style, Tempo, Groove, and Key similarity (set by the weight controls).',
    fit: 'Fit: energy-target match for the slot = 1 / (1 + penalty scale × error²). 1 is a perfect match.',
    loss: 'Loss = Mix − Final: score lost to energy mismatch. Shown as a negative amount with a right-filled bar; closer to 0 is better.',
    style: 'Style: overall sound similarity (audio-embedding distance).',
    tempo: 'Tempo: BPM compatibility.',
    groove: 'Groove: rhythm-pattern similarity.',
    key: 'Key: harmonic (chroma) compatibility.',
    rank: 'Rank: position among all scored candidates for this slot (#1 is best).',
  };
  function scoreTh(label, key) {
    return '<th class="num" title="' + esc(SCORE_TIPS[key] || '') + '">' + label + '</th>';
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
      '<th class="num">Slot</th><th class="num">Actions</th><th title="Green shows the listened region; the dashed amber line is the preview start cue. Click to seek.">Waveform</th><th>Track</th>' +
      scoreTh('Rank', 'rank') + scoreTh('Final', 'final') + scoreTh('Mix', 'mix') + scoreTh('Fit', 'fit') + scoreTh('Loss', 'loss') +
      scoreTh('Style', 'style') + scoreTh('Tempo', 'tempo') + scoreTh('Groove', 'groove') + scoreTh('Key', 'key') +
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
          (previous ? '' : '<div class="muted">First track — no incoming transition</div>') +
          (score && score.missing ? '<div class="warn">missing similarity row</div>' : '') +
          (previous && current.slot - previous.slot > 1 ? '<div class="warn">bridges gap — scored from slot ' + (previous.slot + 1) + '</div>' : '') + '</td>' +
        rankCell(rankInfo) +
        (previous ? metricCell(score.finalScore, 'final', score.finalScore, 'Final score') : emptyMetricCell()) +
        (previous ? metricCell(score.baseline, 'mix', score.baseline, 'Weighted mix score') : emptyMetricCell()) +
        (previous ? metricCell(score.energyScore, 'energy', score.energyScore, 'Energy fit score') : emptyMetricCell()) +
        (previous ? metricCell(Number.isFinite(Number(score.penalty)) ? -Number(score.penalty) : NaN, 'penalty', score.penalty, 'Loss (Mix − Final): score lost to energy mismatch; the bar fills from the right, closer to 0 is better') : emptyMetricCell()) +
        (previous ? metricCell(score.styleScore, 'style', score.styleScore, 'Style score') : emptyMetricCell()) +
        (previous ? metricCell(score.tempoScore, 'tempo', score.tempoScore, 'Tempo score') : emptyMetricCell()) +
        (previous ? metricCell(score.grooveScore, 'groove', score.grooveScore, 'Groove score') : emptyMetricCell()) +
        (previous ? metricCell(score.keyScore, 'harmonic', score.keyScore, 'Harmonic score') : emptyMetricCell()) +
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
      // Pair state is always shown by the right column's Track 1 / Track 2 cards.
      els.currentTransitionScore.innerHTML =
        '<div class="diagnostic-subhead">Assign both tracks (press 1 or 2 on a selected track, or use the T1 / T2 row buttons) to see the full transition breakdown.</div>' +
        (source
          ? '<div class="diagnostic-subhead">Recommendations are scored from:</div>' + trackSummaryHtml(source, { size: 'compact', showArt: true })
          : '');
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
    const actionLabel = sequence[slot] === null ? 'Add' : 'Replace';
    const actionSymbol = sequence[slot] === null ? '+' : '⇄';
    const sourceRecord = ctx.sourceIdx === null ? null : byIdx.get(Number(ctx.sourceIdx));
    const sourceTitle = sourceRecord && sourceRecord.title ? ' ("' + esc(sourceRecord.title) + '")' : '';
    let anchorHtml = '';
    if (ctx.mode === 'track1') {
      anchorHtml = ' · scored from Track 1' + sourceTitle;
      const anchorSlot = sequence.indexOf(Number(ctx.sourceIdx));
      if (anchorSlot >= 0 && anchorSlot > slot) anchorHtml += ' <span class="warn">(Track 1 sits later in the sequence than this slot)</span>';
    } else if (ctx.mode === 'selected') {
      anchorHtml = ' · scored from the selected track' + sourceTitle;
    } else if (ctx.mode === 'sequence') {
      anchorHtml = ' · scored from slot ' + (Number(ctx.sourceSlot) + 1) + sourceTitle;
    } else {
      anchorHtml = ' · scored from the target energy curve only (no source track)';
    }
    let html = filterbar +
      '<div class="muted" style="padding:8px 8px 0;">' +
      (sequence[slot] === null ? 'Adding to slot <b>' : 'Replacing slot <b>') + (slot + 1) + '</b>' +
      anchorHtml + '. ' +
      (prepared.query
        ? 'Showing <b>' + prepared.matchedCount + '</b> matches from <b>' + prepared.allRows.length + '</b> scored candidates' +
          (prepared.pinnedOutsideQueryCount ? ', plus <b>' + prepared.pinnedOutsideQueryCount + '</b> pinned outside the search.' : '.')
        : 'Showing top <b>' + Math.min(25, prepared.matchedCount) + '</b> of <b>' + prepared.allRows.length + '</b> scored candidates.') +
      '</div>' +
      '<table><thead><tr>' + scoreTh('#', 'rank') +
      '<th title="☆ pin candidate · + add to sequence (⇄ replace) · T1 / T2 set Track 1 / Track 2 · ▶ play preview">Actions</th>' +
      '<th title="Green shows the listened region; the dashed amber line is the preview start cue. Click to seek.">Waveform</th><th>Track</th>' +
      '<th class="num" title="Final = Mix × Fit. Overall recommendation score. Hover a value for its Mix, Fit, and Loss parts.">Final</th>' + scoreTh('Fit', 'fit') +
      scoreTh('Style', 'style') + scoreTh('Tempo', 'tempo') + scoreTh('Groove', 'groove') + scoreTh('Key', 'key') +
      '</tr></thead><tbody>';
    rows.forEach((r, i) => {
      const isPinned = pinnedRecommendationIdxs.has(Number(r.idx));
      html += '<tr' + (isPinned ? ' class="pinned-row"' : '') + '><td class="num">' + (isPinned ? '★ ' : '') + (r.globalRank === null ? '—' : (r.globalRank || (i + 1))) + '</td>' +
        '<td><div class="table-actions">' +
        actionSymbolButton('data-pin-rec-idx="' + r.idx + '"', isPinned ? '★' : '☆', isPinned ? 'Unpin candidate' : 'Pin candidate', isPinned ? 'pin-active' : 'pin-button') +
        actionSymbolButton('data-append-idx="' + r.idx + '"', actionSymbol, actionLabel === 'Add' ? 'Add to sequence' : 'Replace slot ' + (slot + 1), 'primary') +
        actionSymbolButton('data-library-outgoing="' + r.idx + '"', 'T1', 'Set as Track 1', 'track-one') +
        actionSymbolButton('data-library-incoming="' + r.idx + '"', 'T2', 'Set as Track 2', 'track-two') +
        '<button class="play-button" data-play-idx="' + r.idx + '" data-play-mode="library" aria-label="Play">▶</button>' +
        '</div></td>' +
        '<td class="waveform-cell">' + rowWaveformScrubberHtml(r, { showTime: false }) + '</td>' +
        '<td>' + trackSummaryHtml(r, { size: 'compact', showArt: true, slot: r.slot ?? slot }) + '</td>' +
        recommendationMetricCell(r.finalScore, 'final', r.finalScore, 2, 'Final (Mix ' + (Number.isFinite(Number(r.baseline)) ? fmt(r.baseline, 2) : 'n/a') + ' × Fit ' + (Number.isFinite(Number(r.energyScore)) ? fmt(r.energyScore, 2) : 'n/a') + ', Loss ' + (Number.isFinite(Number(r.penalty)) ? fmt(r.penalty, 2) : 'n/a') + ')') +
        recommendationMetricCell(r.energyScore, 'energy', r.energyScore, 2, 'Energy fit score') +
        recommendationMetricCell(r.styleScore, 'style', r.styleScore, 2, 'Style score') +
        recommendationMetricCell(r.tempoScore, 'tempo', r.tempoScore, 2, 'Tempo score') +
        recommendationMetricCell(r.grooveScore, 'groove', r.grooveScore, 2, 'Groove score') +
        recommendationMetricCell(r.keyScore, 'harmonic', r.keyScore, 2, 'Harmonic score') +
        '</tr>';
    });
    if (!rows.length) {
      html += '<tr><td colspan="10" class="muted" style="padding:10px;">No recommendations match the current filters.</td></tr>';
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
    const x = Array.from({ length: sequence.length }, (_, i) => i + 1);
    const actual = sequence.map(idx => idx === null ? null : energyOf(byIdx.get(idx)));
    const traces = [
      { x, y: targets, type: 'scatter', mode: 'lines+markers', name: 'Target energy', line: { color: '#e5e7eb', width: 2 }, marker: { size: 10 } },
      { x, y: actual, type: 'scatter', mode: 'lines+markers', name: 'Track energy', line: { color: '#14b8a6', width: 2 }, marker: { size: 9 } }
    ];
    const slot = targetSlot();
    if (slot >= 0) traces.push({ x: [slot + 1], y: [targets[slot]], type: 'scatter', mode: 'markers', name: 'Next slot', marker: { size: 14, color: '#f59e0b', symbol: 'x' } });
    Plotly.react('energy-curve', traces, {
      title: { text: 'Drag target points to edit the energy curve', font: { size: 13, color: '#e5e7eb' } },
      margin: { t: 48, r: 20, b: 46, l: 48 },
      template: 'plotly_dark',
      paper_bgcolor: '#111827',
      plot_bgcolor: '#111827',
      font: { color: '#e5e7eb' },
      dragmode: false,
      xaxis: { title: 'Sequence slot', dtick: 1, range: [0.5, sequence.length + 0.5], fixedrange: true, gridcolor: '#263244', zerolinecolor: '#263244' },
      yaxis: { title: 'Energy', range: [0.5, 9.5], fixedrange: true, tickmode: 'array', tickvals: ENERGY_AXIS_TICKS, ticktext: ENERGY_AXIS_TICKS.map(String), gridcolor: '#263244', zerolinecolor: '#263244' },
      legend: { orientation: 'h', x: 0, y: -0.24, xanchor: 'left', yanchor: 'top' }
    }, { displayModeBar: false, scrollZoom: false, doubleClick: false, responsive: true });
  }
  function renderTransitionBadges() {
    if (!els.transitionBadges) return;
    const from = transitionFromIdx === null ? null : byIdx.get(transitionFromIdx);
    const to = transitionToIdx === null ? null : byIdx.get(transitionToIdx);
    // No pair: stay empty. The transition editor already shows one progressive
    // "assign Track 1/2" message in the same panel — do not repeat it here.
    let scoreHtml = '';
    if (from && to) {
      const slot = Math.max(0, targetSlot());
      const score = scoreTransition(transitionFromIdx, transitionToIdx, slot);
      const rankInfo = transitionRecommendationRank(transitionFromIdx, transitionToIdx, slot);
      const meterRows = [
        ['Final', score.finalScore, 'final', score.finalScore, 'Final score', 'final'],
        ['Mix', score.baseline, 'mix', score.baseline, 'Weighted mix score', 'mix'],
        ['Style', score.styleScore, 'style', score.styleScore, 'Style score', 'style'],
        ['Tempo', score.tempoScore, 'tempo', score.tempoScore, 'Tempo score', 'tempo'],
        ['Groove', score.grooveScore, 'groove', score.grooveScore, 'Groove score', 'groove'],
        ['Key', score.keyScore, 'harmonic', score.keyScore, 'Harmonic score', 'key'],
        ['Loss', Number.isFinite(Number(score.penalty)) ? -Number(score.penalty) : NaN, 'penalty', score.penalty, 'Energy score loss', 'loss'],
        ['Rank #', rankInfo, 'rank', NaN, 'Recommendation rank', 'rank'],
      ];
      scoreHtml =
        '<div class="transition-mini-score">' +
        '<div class="transition-mini-meter-list">' +
        meterRows.map(row =>
          '<div class="transition-mini-meter-row"><span title="' + esc(SCORE_TIPS[row[5]] || '') + '">' + esc(row[0]) + '</span>' +
          (row[2] === 'rank'
            ? rankMeterHtml(row[1], { forceValues: true })
            : scoreMeterHtml(row[1], row[2], row[3], 2, row[4], { forceValues: true })) +
          '</div>'
        ).join('') +
        '</div>' +
        '</div>';
    }
    const lastFilledIdx = sequence.reduce((acc, v) => v !== null ? v : acc, null);
    const followsTail = from && lastFilledIdx !== null && Number(transitionFromIdx) === Number(lastFilledIdx);
    // The two deck cards are the single place the transition pair is shown: album art,
    // title, artist, meta and a per-deck play button (reuses the editor's deck-play handler).
    const deckHtml = (cls, label, deck, record, noteHtml) =>
      '<div class="transition-badge ' + cls + '">' +
      '<div class="deck-card-head"><b>' + label + '</b>' +
      (record ? '<button type="button" class="deck-play play-button" data-transition-track-play="' + deck + '" aria-label="Play ' + label + '">▶</button>' : '') +
      '</div>' +
      (record ? trackSummaryHtml(record, { size: 'compact', showArt: true }) : '<span class="muted">None selected</span>') +
      (noteHtml || '') +
      '</div>';
    els.transitionBadges.innerHTML =
      deckHtml('out', 'Track 1', 'from', from,
        followsTail ? '<div class="muted transition-badge-note">Follows the newest sequence track</div>' : '') +
      deckHtml('in', 'Track 2', 'to', to, '');
    // The detailed transition score lives in the Transition segment, not the deck cards.
    if (els.transitionScore) els.transitionScore.innerHTML = scoreHtml;
    safeUi('deck play buttons', updateTransitionTrackPlayButtons);
  }
  function renderTransitionEditorHtml(from, to) {
    if (!from || !to) {
      let msg = 'Assign Track 1 and Track 2 to render a transition preview.';
      if (from && !to) msg = 'Track 1 is set — now assign Track 2.';
      else if (!from && to) msg = 'Track 2 is set — now assign Track 1.';
      return '<div class="transition-editor"><div class="transition-editor-empty">' + esc(msg) + '</div></div>';
    }
    const pitchValues = [-3, -2, -1, 0, 1, 2, 3];
    const laneInfoHtml = deck => {
      const info = transitionWindowInfo(deck);
      if (!info) return '';
      const cfg = info.cfg;
      // Everything this lane used to print is on screen already: title/artist/BPM on
      // the deck cards, cue in the Cue select, nudge in the Nudge beats input. Only the
      // deck label is needed, to say which overview lane this is.
      return '<div class="transition-lane-head"><b>' + esc(cfg.label) + '</b></div>';
    };
    const laneToolsHtml = deck => {
      const info = transitionWindowInfo(deck);
      if (!info) return '';
      const cfg = info.cfg;
      const pitchField = deck === 'from' ? 'from_pitch_shift' : 'to_pitch_shift';
      const bounds = transitionNudgeBounds(info);
      return '<div class="transition-lane-tools">' +
        cueSelectHtml(info.record, cfg.role, cfg.cueField, 'Cue') +
        numericInput(cfg.nudgeField, 'Nudge beats', previewFormState[cfg.nudgeField], bounds.min, bounds.max, 0.05) +
        selectInput(pitchField, 'Repitch', pitchValues, previewFormState[pitchField], v => (Number(v) > 0 ? '+' : '') + v + ' st') +
        '</div>';
    };
    const overviewHtml = deck => {
      const info = transitionWindowInfo(deck);
      if (!info) return '';
      const cues = cuesForRecord(info.record).filter(cue => Number.isFinite(Number(cue.start_seconds)));
      const cueListHtml = cues.length
        ? '<div class="transition-cue-list">' + cues.map(cue =>
            '<span class="transition-cue-chip" style="color:' + esc(cueMarkerColor(cue.role)) + '">' +
            esc(cue.name || cue.role || 'cue') + ' @ ' + esc(timeText(Number(cue.start_seconds))) + '</span>'
          ).join('') + '</div>'
        : '';
      return '<div class="transition-overview-card ' + esc(deck) + '" data-transition-card="' + esc(deck) + '-overview">' +
        laneInfoHtml(deck) +
        '<div class="transition-overview-row">' +
        '<button type="button" class="transition-track-play" data-transition-track-play="' + esc(deck) + '" aria-label="Play ' + esc(info.cfg.label) + '">▶</button>' +
        '<canvas class="transition-overview-canvas" data-transition-surface="overview" data-transition-deck="' + esc(deck) + '" height="76"></canvas>' +
        '</div>' +
        cueListHtml +
        '</div>';
    };
    const sectionHtml = deck => {
      const info = transitionWindowInfo(deck);
      if (!info) return '';
      const cfg = info.cfg;
      return '<div class="transition-window-card ' + esc(deck) + '" data-transition-card="' + esc(deck) + '-section">' +
        '<div class="transition-window-main">' +
        '<canvas class="transition-section-canvas" data-transition-surface="section" data-transition-deck="' + esc(deck) + '" height="245"></canvas>' +
        '</div>' +
        '<aside class="transition-track-side">' +
        '<div class="transition-track-meta"><div class="transition-track-label">' + esc(cfg.label) + '</div></div>' +
        laneToolsHtml(deck) +
        '</aside>' +
        '</div>';
    };
    return '<div class="transition-editor">' +
      '<div class="transition-editor-head">' +
      '<div class="transition-editor-tools">' +
      '<label><input id="transition-snap-to-beat" type="checkbox"' + (transitionSnapToBeat ? ' checked' : '') + '> Snap to beat</label>' +
      '</div>' +
      '</div>' +
      '<div class="transition-editor-hint">Drag a transition lane or overview window to shift timing. Cue markers can be clicked directly; snap locks movement to beats.</div>' +
      '<div class="transition-native-stage">' +
      overviewHtml('from') +
      '<div class="transition-window-stack">' + sectionHtml('from') + transitionTransportHtml() + sectionHtml('to') + '</div>' +
      overviewHtml('to') +
      '</div>' +
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
    if (surface === 'overview') return { x: 8, y: 7, w: Math.max(1, width - 16), h: Math.max(1, height - 14) };
    return { x: 10, y: 10, w: Math.max(1, width - 20), h: Math.max(1, height - 20) };
  }
  function transitionVisibleRange(info, surface) {
    const trackDuration = Math.max(info.beatSec, info.trackDuration || info.duration);
    if (surface === 'overview') return { start: 0, end: trackDuration, duration: trackDuration };
    const start = info.contextStart;
    const end = info.contextEnd;
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
  function transitionAutomationValue(info, family, timeSec) {
    const modes = resolvedPreviewModes();
    const mode = family + ':' + (modes[family] || 'none');
    const deck = info.cfg.deck === 'from' ? 'a' : 'b';
    const duration = Math.max(1e-9, info.transitionEnd - info.transitionStart);
    const x = clamp((Number(timeSec) - info.transitionStart) / duration, 0, 1);
    if (timeSec < info.transitionStart) {
      if (family === 'volume') return deck === 'a' ? 1 : 0;
      if (family === 'eq') return 1;
      return null;
    }
    if (timeSec > info.transitionEnd) {
      if (family === 'volume') return deck === 'a' ? 0 : 1;
      if (family === 'eq') return 1;
      return null;
    }
    return previewValueFor(mode, deck, x);
  }
  function drawTransitionAutomationCurve(ctx, info, view, lane, family, color) {
    const samples = Math.max(90, Math.floor(lane.w / 5));
    let started = false;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = family === 'filter' ? 2.0 : 2.4;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let i = 0; i <= samples; i += 1) {
      const pct = i / samples;
      const timeSec = view.start + (pct * view.duration);
      const value = transitionAutomationValue(info, family, timeSec);
      if (value == null || !Number.isFinite(Number(value))) {
        started = false;
        continue;
      }
      const x = lane.x + (pct * lane.w);
      const y = lane.y + ((1 - clamp(Number(value), 0, 1)) * lane.h);
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
  function drawTransitionAutomationCurves(ctx, info, view, lane) {
    if (!info || !view || !lane) return;
    drawTransitionAutomationCurve(ctx, info, view, lane, 'volume', 'rgba(72,200,242,.94)');
    drawTransitionAutomationCurve(ctx, info, view, lane, 'eq', 'rgba(255,209,61,.90)');
    drawTransitionAutomationCurve(ctx, info, view, lane, 'filter', 'rgba(229,140,255,.88)');
    ctx.save();
    ctx.fillStyle = 'rgba(242,242,242,.70)';
    ctx.font = '10px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    ctx.fillText('volume / EQ / filter', lane.x + 8, lane.y + 14);
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
    if (selectedEnd <= selectedStart) return;
    const x0 = xForTransitionTime(selectedStart, view, lane);
    const x1 = xForTransitionTime(selectedEnd, view, lane);
    ctx.save();
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
      const label = String(cue.name || cue.role || '').slice(0, surface === 'overview' ? 12 : 20);
      if (label && (surface !== 'overview' || x > lane.x + 8 && x < lane.x + lane.w - 42)) {
        ctx.font = '700 ' + (surface === 'overview' ? '11px' : '12px') + ' ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
        const textX = clamp(x + 4, lane.x + 4, lane.x + lane.w - 96);
        const textY = surface === 'overview' ? lane.y + 13 : lane.y + lane.h - 8;
        const textWidth = ctx.measureText(label).width;
        ctx.fillStyle = 'rgba(10,10,10,.72)';
        ctx.fillRect(textX - 3, textY - 10, textWidth + 6, 14);
        ctx.fillStyle = cueMarkerColor(cue.role);
        ctx.fillText(label, textX, textY);
      }
    });
  }
  function transitionPlaybackProgress() {
    if (!transitionRender || !transitionRender.ok) return null;
    const audio = transitionAudioElement();
    const scrub = document.getElementById('transition-scrub');
    const duration = transitionDuration(audio);
    if (!Number.isFinite(duration) || duration <= 0) return null;
    let current = audio && Number.isFinite(Number(audio.currentTime)) ? Number(audio.currentTime) : NaN;
    if (!Number.isFinite(current) && scrub) current = Number(scrub.value);
    if (!Number.isFinite(current)) return null;
    return clamp(current / duration, 0, 1);
  }
  function drawTransitionPlaybackHead(ctx, info, view, lane, surface) {
    const progress = transitionPlaybackProgress();
    if (progress == null) return;
    const trackTime = info.contextStart + (progress * info.duration);
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
    const view = transitionVisibleRange(info, surface);
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
      ['from', 'to'].forEach(deck => {
        const info = transitionWindowInfo(deck);
        if (!info) return;
        ensureDetailedWaveform(info.record, { redrawTransition: true });
        const overview = document.querySelector('[data-transition-surface="overview"][data-transition-deck="' + deck + '"]');
        const section = document.querySelector('[data-transition-surface="section"][data-transition-deck="' + deck + '"]');
        if (overview) drawTransitionOverview(overview, info);
        if (section) drawTransitionSection(section, info);
      });
      updateTransitionTrackPlayButtons();
      updateTransitionDirtyUi();
    });
  }
  function renderTransitionPreview() {
    if (!els.transitionPreview) return;
    const from = transitionFromIdx === null ? null : byIdx.get(transitionFromIdx);
    const to = transitionToIdx === null ? null : byIdx.get(transitionToIdx);
    let controlsHtml = '';
    if (config.app_mode) {
      if (!appLoadStarted) loadAppRuntime();
      if (appOptions && appOptions.error) {
        controlsHtml = '<div class="warn" style="margin-top:14px;">Could not load render controls from the server.<br>' +
          '<span class="muted">' + esc(appOptions.error) + '</span><br>' +
          '<button type="button" data-preview-action="retry-load" style="margin-top:8px;">Retry</button></div>';
      } else if (!appLoadDone) {
        controlsHtml = '<div class="muted" style="margin-top:14px;">Loading render controls...</div>';
      } else if (!from || !to) {
        controlsHtml = transitionRenderWarn && transitionRenderMessage
          ? '<div class="warn" style="margin-top:14px;">' + esc(transitionRenderMessage) + '</div>'
          : '';
      } else {
        const presets = (appOptions.presets || ['auto']);
        const volumeModes = appOptions.volume_modes || [];
        const eqModes = appOptions.eq_modes || [];
        const filterModes = appOptions.filter_modes || [];
        const renderStatus = transitionRenderStatusInfo();
        controlsHtml =
          '<div class="transition-settings-panel">' +
          '<section class="preview-render-section"><h3>Render Controls</h3><div class="preview-render-grid preview-timing-grid">' +
          numericInput('overlap_bars', 'Overlap (bars)', previewFormState.overlap_bars, 1, 64, 1, 'Length of the blended region where both tracks play, in bars') +
          numericInput('front_padding_bars', 'Front pad (bars)', previewFormState.front_padding_bars, 0, 16, 1, 'Bars of Track 1 audio kept before the overlap starts') +
          numericInput('back_padding_bars', 'Back pad (bars)', previewFormState.back_padding_bars, 0, 16, 1, 'Bars of Track 2 audio kept after the overlap ends') +
          '</div></section>' +
          '<section class="preview-render-section"><h3>Transition FX</h3><div class="preview-mode-controls">' +
          selectInput('preset', 'Preset', presets, previewFormState.preset) +
          selectInput('volume_mode', 'Volume', volumeModes, previewFormState.volume_mode) +
          selectInput('eq_mode', 'EQ', eqModes, previewFormState.eq_mode) +
          selectInput('filter_mode', 'Filter', filterModes, previewFormState.filter_mode) +
          '</div>' +
          '<div class="fx-legend" aria-label="Automation curve legend">' +
          '<span class="fx-item"><span class="fx-icon" aria-hidden="true">VOL</span><span class="fx-line volume"></span>Volume line</span>' +
          '<span class="fx-item"><span class="fx-icon" aria-hidden="true">EQ</span><span class="fx-line eq"></span>EQ line</span>' +
          '<span class="fx-item"><span class="fx-icon" aria-hidden="true">FLT</span><span class="fx-line filter"></span>Filter line</span>' +
          '</div></section>' +
          '<div class="preview-render-actions">' +
          '<div class="preview-actions">' +
          '<button id="transition-render-button" type="button" class="primary" data-preview-action="render" onclick="return window.__djRenderTransition ? window.__djRenderTransition(event) : false;"' + (transitionRenderPending ? ' disabled' : '') + '>' +
          (transitionRenderPending ? 'Rendering...' : (transitionRender && previewSettingsDirty() ? 'Render again' : 'Render transition')) +
          '</button>' +
          '<button type="button" data-preview-action="swap">Swap</button>' +
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
        '<button type="button" data-preview-action="swap">Swap</button>' +
        '<button type="button" data-preview-action="clear">Clear transition</button>' +
        '</div>';
    }
    const editorHtml = safeUi(
      'transition editor render',
      () => renderTransitionEditorHtml(from, to),
      '<div class="transition-editor"><div class="transition-editor-empty warn">Transition editor failed to render. Other app controls remain available.</div></div>'
    );
    // The pair, the render state and the rendered duration are already on screen
    // (deck cards, status line, transport time) — the topbar only carries the WAV link.
    const linksHtml = transitionRender && transitionRender.urls ?
      '<a href="' + esc(transitionRender.urls.preview) + '" target="_blank">WAV</a>' : '';
    els.transitionPreview.innerHTML =
      '<div class="transition-workbench">' +
      '<section class="transition-main">' +
      (controlsHtml ? '<section class="transition-controls">' + controlsHtml + '</section>' : '') +
      editorHtml +
      (linksHtml ? '<div class="transition-topbar"><div class="transition-links">' + linksHtml + '</div></div>' : '') +
      '</section>' +
      '</div>';
    safeUi('transition preview bind', bindTransitionPreviewOverlay);
  }
  function transitionTransportHtml() {
    const result = renderResultHtml();
    // Render progress and readiness are already stated by the status line and the
    // render button above — don't repeat a spinner and the same sentence here.
    const body = result || '<div class="transition-scrubbar transition-scrubbar-empty"></div>';
    return '<div class="transition-transport-row">' +
      '<div class="transition-transport-main">' + body + '</div>' +
      '</div>';
  }
  function renderResultHtml() {
    if (!transitionRender || !transitionRender.ok || !transitionRender.urls) return '';
    const t = transitionRender.transition || {};
    const duration = Math.max(1, Number(t.duration_seconds) || 1);
    const stamp = esc(transitionRender.rendered_at || '');
    return '<div class="transition-scrubbar">' +
      '<div class="transition-transport-controls">' +
      '<button id="transition-play-toggle" class="play-button" type="button" aria-label="Play">▶</button>' +
      '<div class="transition-transport-title">Rendered transition</div>' +
      '<span id="transition-scrub-time" class="scrub-time">0:00 / ' + esc(timeText(duration)) + '</span>' +
      '</div>' +
      '<input id="transition-scrub" class="scrub-range" type="range" min="0" max="' + esc(duration) + '" step="0.01" value="0" aria-label="Rendered transition playback position">' +
      '<audio id="transition-preview-audio" preload="metadata" src="' + esc(transitionRender.urls.preview) + '?t=' + stamp + '"></audio>' +
      '</div>';
  }
  function transitionAudioElement() {
    return document.getElementById('transition-preview-audio');
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
    if (!els.transitionPreview) return;
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
    const audio = transitionAudioElement();
    const button = document.getElementById('transition-play-toggle');
    const scrub = document.getElementById('transition-scrub');
    const time = document.getElementById('transition-scrub-time');
    if (!button || !scrub || !time) return;
    const duration = transitionDuration(audio);
    const current = audio && Number.isFinite(Number(audio.currentTime)) ? Number(audio.currentTime) : 0;
    const transitionPlaying = !!(audio && !audio.paused && !audio.ended);
    button.textContent = transitionPlaying ? '⏸' : '▶';
    button.setAttribute('aria-label', transitionPlaying ? 'Pause' : 'Play');
    const title = document.querySelector('.transition-transport-title');
    if (title) title.textContent = (transitionPlaying ? '♪ ' : '') + 'Rendered transition';
    if (Number.isFinite(duration) && duration > 0) scrub.max = String(duration);
    if (!transitionScrubDragging) scrub.value = String(clamp(current, 0, Number(scrub.max) || duration || 1));
    setRangeProgress(scrub, scrub.value, scrub.max);
    time.textContent = timeText(current) + ' / ' + timeText(duration);
    requestTransitionEditorDraw();
    if (audio && !audio.paused && !audio.ended) scheduleTransitionScrubber();
  }
  function bindTransitionScrubber() {
    const audio = transitionAudioElement();
    const button = document.getElementById('transition-play-toggle');
    const scrub = document.getElementById('transition-scrub');
    if (!button || !scrub) return;
    applyMasterVolume();
    if (!button.__seqBound) {
      button.__seqBound = true;
      button.addEventListener('click', () => {
        const activeAudio = transitionAudioElement();
        if (!activeAudio) return;
        if (activeAudio.paused || activeAudio.ended) {
          pauseSharedAudio();
          const promise = activeAudio.play();
          if (promise && promise.catch) promise.catch(() => {});
        } else {
          activeAudio.pause();
        }
        updateTransitionScrubber();
      });
    }
    if (!scrub.__seqBound) {
      scrub.__seqBound = true;
      scrub.addEventListener('input', () => {
        transitionScrubDragging = true;
        const activeAudio = transitionAudioElement();
        const value = Number(scrub.value || 0);
        if (activeAudio && Number.isFinite(value)) {
          try { activeAudio.currentTime = value; } catch (err) {}
        }
        setRangeProgress(scrub, value, scrub.max);
        updateTransitionScrubber();
      });
      scrub.addEventListener('change', () => {
        transitionScrubDragging = false;
        const activeAudio = transitionAudioElement();
        const value = Number(scrub.value || 0);
        if (activeAudio && Number.isFinite(value)) {
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
      audio.addEventListener('error', () => {
        showToast('Couldn\'t load the rendered transition audio.', { tone: 'error' });
        updateTransitionScrubber();
      });
    }
    updateTransitionScrubber();
  }
  function applyTransitionEditorDrag(ev) {
    if (!transitionEditorDrag) return;
    const info = transitionWindowInfo(transitionEditorDrag.deck);
    if (!info) return;
    const dx = Number(ev.clientX) - transitionEditorDrag.startX;
    const secondsPerPixel = transitionEditorDrag.secondsPerPixel;
    const deltaSeconds = dx * secondsPerPixel;
    let nudge = transitionEditorDrag.startNudge + (deltaSeconds / info.beatSec);
    nudge = transitionSnapToBeat ? Math.round(nudge) : Math.round(nudge * 100) / 100;
    nudge = clampTransitionNudge(transitionEditorDrag.deck, nudge);
    previewFormState[info.cfg.nudgeField] = nudge;
    syncPreviewFieldElements(info.cfg.nudgeField, nudge);
    scheduleSessionSave();
    drawTransitionEditor();
    updatePreviewEffectDisplay({ dirty: true });
  }
  function bindTransitionEditor() {
    const snap = document.getElementById('transition-snap-to-beat');
    if (snap && !snap.__seqBound) {
      snap.__seqBound = true;
      snap.addEventListener('change', () => {
        transitionSnapToBeat = !!snap.checked;
        drawTransitionEditor();
      });
    }
    document.querySelectorAll('[data-transition-surface][data-transition-deck]').forEach(canvas => {
      if (canvas.__seqBound) return;
      canvas.__seqBound = true;
      canvas.addEventListener('pointerdown', ev => {
        if (ev.button !== 0) return;
        const deck = canvas.getAttribute('data-transition-deck');
        const info = transitionWindowInfo(deck);
        if (!info) return;
        const cue = transitionCueAtPoint(canvas, info, ev);
        if (cue) {
          previewFormState[info.cfg.cueField] = cue.name || '';
          syncPreviewFieldElements(info.cfg.cueField, previewFormState[info.cfg.cueField]);
          scheduleSessionSave();
          drawTransitionEditor();
          updatePreviewEffectDisplay({ dirty: true });
          ev.preventDefault();
          ev.stopPropagation();
          return;
        }
        const rect = canvas.getBoundingClientRect();
        const surface = canvas.getAttribute('data-transition-surface');
        const view = transitionVisibleRange(info, surface);
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
      canvas.addEventListener('pointermove', ev => {
        if (!transitionEditorDrag || transitionEditorDrag.deck !== canvas.getAttribute('data-transition-deck')) return;
        applyTransitionEditorDrag(ev);
        ev.preventDefault();
      });
      const endDrag = ev => {
        if (!transitionEditorDrag) return;
        applyTransitionEditorDrag(ev);
        transitionEditorDrag = null;
        try { canvas.releasePointerCapture(ev.pointerId); } catch (err) {}
        updateTransitionDirtyUi();
      };
      canvas.addEventListener('pointerup', endDrag);
      canvas.addEventListener('pointercancel', () => { transitionEditorDrag = null; });
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
    setTimeout(() => safeUi('transition editor draw', drawTransitionEditor), 0);
    setTimeout(() => safeUi('transition dirty ui update', updateTransitionDirtyUi), 0);
  }
  function updatePreviewFormStateFromElement(el) {
    const field = el && el.getAttribute && el.getAttribute('data-preview-field');
    if (!field) return;
    if (field === 'preset') {
      applyPresetToPreviewState(el.value);
      updateTransitionDirtyUi();
      scheduleSessionSave();
      return;
    }
    if (el.type === 'number') previewFormState[field] = Number(el.value);
    else previewFormState[field] = el.value;
    if (field === 'from_nudge_beats') previewFormState[field] = clampTransitionNudge('from', previewFormState[field]);
    if (field === 'to_nudge_beats') previewFormState[field] = clampTransitionNudge('to', previewFormState[field]);
    if ((field === 'from_nudge_beats' || field === 'to_nudge_beats') && el) el.value = String(previewFormState[field]);
    syncPreviewFieldElements(field, previewFormState[field], el);
    if (['volume_mode', 'eq_mode', 'filter_mode'].includes(field) && previewFormState.preset !== 'custom') {
      previewFormState.preset = 'custom';
      const presetEl = document.getElementById('preset');
      if (presetEl) presetEl.value = 'custom';
    }
    scheduleSessionSave();
    drawTransitionEditor();
  }
  function syncPreviewFormStateFromDom() {
    document.querySelectorAll('[data-preview-field]').forEach(updatePreviewFormStateFromElement);
  }
  async function renderBackendTransition() {
    if (transitionRenderPending) return;
    const from = transitionFromIdx === null ? null : byIdx.get(transitionFromIdx);
    const to = transitionToIdx === null ? null : byIdx.get(transitionToIdx);
    if (!from || !to) {
      transitionRenderMessage = 'Assign Track 1 and Track 2 before rendering.';
      transitionRenderWarn = true;
      renderTransitionPreview();
      return;
    }
    syncPreviewFormStateFromDom();
    const body = Object.assign({}, previewFormState, {
      from_track: trackId(from),
      to_track: trackId(to),
      overwrite: true,
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
      let data = null;
      try {
        data = await res.json();
      } catch (parseErr) {
        throw new Error('Render failed (HTTP ' + res.status + '): the server response was not valid JSON.');
      }
      if (requestId !== transitionRenderRequestId) return;
      if (!data.ok) throw new Error(data.error || ('Render failed (HTTP ' + res.status + ')'));
      data.rendered_at = Date.now();
      transitionRender = data;
      transitionRenderedState = Object.assign({}, body);
      transitionRenderMessage = 'Rendered preview is current.';
      transitionRenderWarn = false;
    } catch (err) {
      if (requestId !== transitionRenderRequestId) return;
      const raw = (err && err.message) || String(err);
      transitionRenderMessage = err && err.name === 'TypeError'
        ? 'Could not reach the render server. Check that the app is still running, then try again.'
        : raw;
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
    const value = Math.round(clamp(Number(yValue), 1, 9) * 2) / 2;
    if (!Number.isFinite(slot) || !Number.isFinite(value)) return null;
    return { slot, value };
  }
  function beginEnergyDrag(ev) {
    if (ev.button !== 0) return;
    const point = energyValueFromMouse(ev);
    if (!point) return;
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
    if (draggingEnergySlot !== null) scheduleSessionSave();
    draggingEnergySlot = null;
  }
  function updateAppendControls() {
    const slot = targetSlot();
    const replacing = slot >= 0 && sequence[slot] !== null;
    setActionButton(els.appendSelected, replacing ? '⇄' : '+', replacing ? 'Replace track in selected slot' : 'Add selected track to sequence', 'primary');
    els.appendSelected.disabled = selectedIdx === null || !canPlaceTrack(selectedIdx, slot);
    const hasFilled = sequence.some(v => v !== null);
    const curveEdited = targetValues.some(v => Number(v) !== DEFAULT_TARGET_ENERGY);
    if (els.clearLast) els.clearLast.disabled = !hasFilled;
    if (els.resetSequence) els.resetSequence.disabled = !hasFilled && !curveEdited;
  }
  function renderSelectionOnly() {
    safeUi('map overlays', updatePacmapOverlays);
    safeUi('map static overlays', ensureMapEffectsLoop);
    safeUi('append controls', updateAppendControls);
    // Right build column is always visible — render its pieces regardless of the active analysis tab.
    safeUi('sequence render', renderSequence);
    safeUi('recommendations render', renderRecommendations);
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
    // Right build column is always visible — always render sequence, recommendations,
    // energy curve, and the transition preview regardless of the active analysis tab.
    safeUi('sequence render', renderSequence);
    safeUi('recommendations render', renderRecommendations);
    safeUi('energy curve render', renderEnergyCurve);
    safeUi('transition preview render', renderTransitionPreview);
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
  function downloadCsv() {
    const targets = targetCurve();
    const header = ['slot','target_energy','actual_energy','mix','track_number','title','artists','genre','key','bpm','filename'];
    const lines = [header.join(',')];
    for (let i = 0; i < sequence.length; i += 1) {
      const r = sequence[i] === null ? null : byIdx.get(sequence[i]);
      const vals = [
        i + 1, fmt(targets[i], 3), r ? fmt(energyOf(r), 3) : '', r ? r.mix_slug : '', r ? r.track_number : '',
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
    const lastFilled = sequence.reduce((acc, v, j) => v !== null ? j : acc, -1);
    const gapCount = sequence.filter((v, j) => v === null && j < lastFilled).length;
    if (gapCount) {
      showToast('Exported with ' + gapCount + ' empty slot' + (gapCount === 1 ? '' : 's') + ' — scores bridge gaps.', { tone: 'warn' });
    }
  }

  els.sequenceLength.value = String(sequenceLength);
  applyAppSettingsToControls();
  if (config.control_mode === 'genre-mixability') {
    if (els.weightStyle) els.weightStyle.value = String((config.weights && config.weights.maest) || 0.45);
    setMixSliders({
      tempo: (config.weights && config.weights.mix_tempo) || 0.34,
      groove: (config.weights && config.weights.mix_groove) || 0.33,
      chroma: (config.weights && config.weights.mix_chroma) || 0.33,
    });
  } else {
    if (els.weightMaest) els.weightMaest.value = String((config.weights && config.weights.maest) || 0.6);
    if (els.weightChroma) els.weightChroma.value = String((config.weights && config.weights.chroma) || 0.25);
    if (els.weightTempo) els.weightTempo.value = String((config.weights && config.weights.tempo) || 0.15);
  }
  setActionButton(els.setOutgoing, 'T1', 'Set selected track as Track 1', 'track-one');
  setActionButton(els.setIncoming, 'T2', 'Set selected track as Track 2', 'track-two');
  setActionButton(els.clearTransition, '×', 'Clear transition pair', 'danger');
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
    if (typing || ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const key = String(ev.key || '').toLowerCase();
    if (key === '+' || key === '=' || key === 'i') {
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
      if (selectedIdx !== null) assignTrackTwo(selectedIdx);
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
  const paneTabsNav = document.querySelector('.pane-tabs');
  if (paneTabsNav) paneTabsNav.addEventListener('keydown', ev => {
    if (ev.key !== 'ArrowLeft' && ev.key !== 'ArrowRight') return;
    const current = els.tabButtons.indexOf(document.activeElement);
    if (current < 0) return;
    const step = ev.key === 'ArrowRight' ? 1 : els.tabButtons.length - 1;
    const next = els.tabButtons[(current + step) % els.tabButtons.length];
    if (next) {
      next.focus();
      setActivePane(next.getAttribute('data-pane-tab') || 'explore');
    }
    ev.preventDefault();
    ev.stopPropagation();
  });
  els.buildSegButtons.forEach(btn => btn.addEventListener('click', () => setBuildSegment(btn.getAttribute('data-build-seg'))));
  const buildSegNav = document.querySelector('.build-segments');
  if (buildSegNav) buildSegNav.addEventListener('keydown', ev => {
    if (ev.key !== 'ArrowLeft' && ev.key !== 'ArrowRight') return;
    const current = els.buildSegButtons.indexOf(document.activeElement);
    if (current < 0) return;
    const step = ev.key === 'ArrowRight' ? 1 : els.buildSegButtons.length - 1;
    const next = els.buildSegButtons[(current + step) % els.buildSegButtons.length];
    if (next) {
      next.focus();
      setBuildSegment(next.getAttribute('data-build-seg'));
    }
    ev.preventDefault();
    ev.stopPropagation();
  });
  els.sequenceLength.addEventListener('change', () => setSequenceLength(els.sequenceLength.value));
  els.energySource.addEventListener('change', renderAll);
  els.colorMode.addEventListener('change', () => {
    setAppSetting('point_color', els.colorMode.value);
    renderAll();
  });
  if (els.mapRenderer) els.mapRenderer.addEventListener('change', () => {
    setAppSetting('map_renderer', els.mapRenderer.value === 'webgl' ? 'webgl' : 'plotly');
    syncMapRenderer({ render: false });
    renderAll();
  });
  if (els.mapArtworkEnabled) els.mapArtworkEnabled.addEventListener('change', () => {
    setAppSetting('map_artwork', Boolean(els.mapArtworkEnabled.checked));
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
  if (els.clickToPlay) els.clickToPlay.addEventListener('change', () => {
    setAppSetting('click_to_play', Boolean(els.clickToPlay.checked));
  });
  if (els.sideSettings) els.sideSettings.addEventListener('toggle', () => {
    setAppSetting('side_settings_open', Boolean(els.sideSettings.open));
  });
  if (els.miniMapToggle) els.miniMapToggle.addEventListener('click', ev => {
    ev.preventDefault();
    ev.stopPropagation();
    setAppSetting('mini_map_collapsed', !appSetting('mini_map_collapsed', false));
    applyMiniMapCollapsedState();
    safeUi('mini map draw', drawMiniMap);
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
  if (els.weightStyle) els.weightStyle.addEventListener('input', renderAll);
  if (els.weightMaest) els.weightMaest.addEventListener('input', renderAll);
  [els.weightChroma, els.weightTempo, els.weightGroove].forEach(el => {
    if (!el) return;
    el.addEventListener('input', () => {
      if (config.control_mode === 'genre-mixability') setMixSliders(mixWeightsRaw());
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
  if (els.setIncoming) els.setIncoming.addEventListener('click', () => { if (selectedIdx !== null) assignTrackTwo(selectedIdx); });
  els.appendSelected.addEventListener('click', () => { if (selectedIdx !== null) appendTrack(selectedIdx); });
  if (els.clearTransition) els.clearTransition.addEventListener('click', clearTransitionPair);
  els.clearLast.addEventListener('click', () => {
    let cleared = null;
    for (let i = sequence.length - 1; i >= 0; i -= 1) { if (sequence[i] !== null) { cleared = i; break; } }
    if (cleared === null) return;
    const snap = sequenceSnapshot();
    const label = sequenceTrackLabel(sequence[cleared]);
    sequence[cleared] = null;
    scheduleSessionSave();
    renderAll();
    showToast('Removed "' + label + '" from slot ' + (cleared + 1) + '.', {
      undoGroup: 'sequence',
      onUndo: () => restoreSequenceSnapshot(snap),
    });
  });
  els.resetSequence.addEventListener('click', () => {
    const removed = sequence.filter(v => v !== null).length;
    const curveEdited = targetValues.some(v => Number(v) !== DEFAULT_TARGET_ENERGY);
    const snap = sequenceSnapshot();
    sequence = Array.from({ length: sequenceLength }, () => null);
    targetValues = Array.from({ length: sequenceLength }, () => DEFAULT_TARGET_ENERGY);
    selectedSlot = null;
    scheduleSessionSave();
    renderAll();
    if (!removed && !curveEdited) return;
    const detail = removed
      ? removed + ' track' + (removed === 1 ? '' : 's') + ' removed'
      : 'energy curve cleared';
    showToast('Sequence reset (' + detail + ').', {
      undoGroup: 'sequence',
      onUndo: () => restoreSequenceSnapshot(snap),
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
    els.audio.addEventListener('error', () => {
      if (!previewAudioUrl) return;
      const record = currentAudioRecord();
      previewAudioPlaying = false;
      showToast('Couldn\'t load audio for ' + (record && record.title ? '"' + record.title + '"' : 'this track') + '.', { tone: 'error' });
      syncAudioUi();
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
  window.addEventListener('resize', () => setTimeout(() => safeUi('energy curve resize', () => { if (window.Plotly && els.energyCurve) Plotly.Plots.resize(els.energyCurve); }), 30));
  window.addEventListener('resize', () => setTimeout(() => safeUi('main waveform draw', drawMainWaveform), 30));
  window.addEventListener('resize', () => setTimeout(() => safeUi('library waveform draw', drawLibraryWaveforms), 30));
  window.addEventListener('resize', () => setTimeout(() => safeUi('transition editor draw', drawTransitionEditor), 30));
  window.addEventListener('resize', () => setTimeout(() => safeUi('mini map draw', drawMiniMap), 30));
  window.addEventListener('resize', () => setTimeout(() => safeUi('webgl map resize', () => { if (webglMap) webglMap.resize(); }), 30));
  // Flush any debounced session write before the page goes away so the last edit is not lost.
  window.addEventListener('pagehide', () => safeUi('session flush', flushSessionSave));
  document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'hidden') safeUi('session flush', flushSessionSave); });

  document.body.addEventListener('click', ev => {
    const appendIdx = closestAttr(ev.target, 'data-append-idx');
    const removeSlotValue = closestAttr(ev.target, 'data-remove-slot');
    const seqSlotValue = closestAttr(ev.target, 'data-seq-slot');
    const placeSlotValue = closestAttr(ev.target, 'data-place-slot');
    const moveSlotUp = closestAttr(ev.target, 'data-move-slot-up');
    const moveSlotDown = closestAttr(ev.target, 'data-move-slot-down');
    const playButton = closestEl(ev.target, '[data-play-idx]');
    const playIdx = playButton ? playButton.getAttribute('data-play-idx') : null;
    const playMode = playButton ? (playButton.getAttribute('data-play-mode') || 'point') : 'point';
    const libraryCurrent = closestAttr(ev.target, 'data-library-current');
    const libraryOutgoing = closestAttr(ev.target, 'data-library-outgoing');
    const libraryIncoming = closestAttr(ev.target, 'data-library-incoming');
    const libraryPlace = closestAttr(ev.target, 'data-library-place');
    const librarySortKey = closestAttr(ev.target, 'data-library-sort');
    const previewAction = closestAttr(ev.target, 'data-preview-action');
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
      scheduleSessionSave();
      safeUi('recommendations render', renderRecommendations);
      safeUi('map overlays', updatePacmapOverlays);
      safeUi('map effects', ensureMapEffectsLoop);
      ev.preventDefault();
      return;
    }
    if (closestAttr(ev.target, 'data-clear-saved-session') !== null) {
      ev.stopPropagation();
      clearSavedSession();
      showToast('Saved session cleared — auto-save paused until you reload.', { tone: 'warn' });
      safeUi('settings render', renderSettingsPanel);
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
      const sortButton = document.querySelector('.th-sort[data-library-sort="' + librarySortKey + '"]');
      if (sortButton) sortButton.focus();
    }
    if (moveSlotUp !== null) {
      const slot = Number(moveSlotUp);
      moveSequenceSlot(slot, slot - 1);
      const focusTarget = document.querySelector('[data-move-slot-up="' + (slot - 1) + '"]:not([disabled])')
        || document.querySelector('[data-move-slot-down="' + (slot - 1) + '"]:not([disabled])');
      if (focusTarget) focusTarget.focus();
      return;
    }
    if (moveSlotDown !== null) {
      const slot = Number(moveSlotDown);
      moveSequenceSlot(slot, slot + 1);
      const focusTarget = document.querySelector('[data-move-slot-down="' + (slot + 1) + '"]:not([disabled])')
        || document.querySelector('[data-move-slot-up="' + (slot + 1) + '"]:not([disabled])');
      if (focusTarget) focusTarget.focus();
      return;
    }
    if (appendIdx !== null) appendTrack(Number(appendIdx));
    if (removeSlotValue !== null) removeSlot(Number(removeSlotValue));
    if (seqSlotValue !== null) {
      const slot = Number(seqSlotValue);
      selectedSlot = selectedSlot === slot ? null : slot;
      scheduleSessionSave();
      renderAll();
    }
    if (placeSlotValue !== null && selectedIdx !== null) { appendTrack(selectedIdx, { slot: Number(placeSlotValue) }); }
    if (playIdx !== null) toggleTrackPreview(Number(playIdx), { mode: playMode });
    if (libraryCurrent !== null) setCurrentTrack(Number(libraryCurrent));
    if (libraryOutgoing !== null) setTransitionEndpoint('out', Number(libraryOutgoing), { toggle: false });
    if (libraryIncoming !== null) assignTrackTwo(Number(libraryIncoming));
    if (libraryPlace !== null) appendTrack(Number(libraryPlace));
    if (previewAction === 'swap') swapTransitionPair();
    if (previewAction === 'clear') clearTransitionPair();
    if (previewAction === 'render') renderBackendTransition();
    if (previewAction === 'retry-load') {
      appLoadStarted = false;
      appLoadDone = false;
      appOptions = null;
      loadAppRuntime();
      safeUi('transition preview render', renderTransitionPreview);
    }
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
    const slotValue = ev.target && ev.target.getAttribute && ev.target.getAttribute('data-target-slot');
    if (slotValue === null) return;
    const slot = Number(slotValue);
    const val = Number(ev.target.value);
    targetValues[slot] = Number.isFinite(val) ? normalizeTargetEnergy(val) : DEFAULT_TARGET_ENERGY;
    scheduleSessionSave();
    renderAll();
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
  if (els.mapLegendRestore) {
    els.mapLegendRestore.addEventListener('click', () => safeUi('legend restore', restoreHiddenGenreTraces));
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
    plot.on('plotly_legendclick', () => {
      setTimeout(() => safeUi('legend restore cue', updateMapLegendRestore), 0);
    });
    plot.on('plotly_legenddoubleclick', () => {
      setTimeout(() => safeUi('legend restore cue', updateMapLegendRestore), 0);
    });
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
      const nearestIdx = nearestTrackPointIdx(ev, 18);
      if (nearestIdx !== null) {
        if ((Date.now() - lastPlotPointClickMs) > 80) handleMapTrackClick(nearestIdx, ev);
        return;
      }
      handleMapBlankClick();
    });
  }
  restoreSession();
  setActivePane('explore', { render: false });
  setBuildSegment(appSetting('build_segment', 'sequence'), { persist: false });
  renderAll();
})();
