/**
 * AirWrite Learn English — Frontend Application
 * Modules: Camera, Translate, Review (Spell + Cloze), Collections, Progress
 */

/* ═══════════════════════ CORE UTILITIES ═══════════════════════ */

const API = 'http://127.0.0.1:8765/api';
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => document.querySelectorAll(selector);

async function api(path, options = {}) {
  const response = await fetch(API + path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || 'Lỗi kết nối backend');
  return body;
}

// Highlight the matched word in an example sentence
function highlightExample(sentence, word) {
  if (!sentence || !word) return sentence || '';
  const regex = new RegExp(`(${word})`, 'gi');
  return sentence.replace(regex, '<strong>$1</strong>');
}

// Generate cloze template: hide ~40% of letters at non-consecutive positions
function makeClozeTemplate(word) {
  const letters = word.split('');
  const count = Math.max(1, Math.round(letters.length * 0.4));
  // Pick positions to hide (avoid first and last for readability)
  const pool = letters.map((_, i) => i).filter(i => i > 0 && i < letters.length - 1);
  const shuffled = pool.sort(() => Math.random() - 0.5).slice(0, count);
  const hidden = new Set(shuffled);
  return letters.map((ch, i) => ({ ch, hidden: hidden.has(i) }));
}

/* ═══════════════════════ CAMERA MODULE ════════════════════════ */

let stream = null;
let frameTimer = null;
let frameCanvas = document.createElement('canvas');
let frameCtx = frameCanvas.getContext('2d');
let cameraContract = { width: 1280, height: 720, fps: 30, mirror: true };

function buildFrameSender({ sessionGetter, onFrame, onGesture, onError }) {
  return setInterval(async () => {
    const session = sessionGetter();
    const cam = $('#camera');
    if (!session || !cam || cam.readyState < 2) return;

    frameCtx.drawImage(cam, 0, 0, frameCanvas.width, frameCanvas.height);
    try {
      const frame = await api(`/recognition/sessions/${session.session_id}/frames`, {
        method: 'POST',
        body: JSON.stringify({ image: frameCanvas.toDataURL('image/jpeg', 0.7) }),
      });
      if (onFrame) onFrame(frame);
      if (onGesture) onGesture(frame.state, frame.did_draw);
    } catch (err) {
      if (onError) onError(err.message);
    }
  }, 100);
}

async function startCamera(videoEl) {
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      width:  { ideal: cameraContract.width },
      height: { ideal: cameraContract.height },
      frameRate: { ideal: cameraContract.fps, max: cameraContract.fps },
      facingMode: 'user',
    },
  });
  videoEl.srcObject = stream;
  const actual = stream.getVideoTracks()[0]?.getSettings() || {};
  frameCanvas.width  = actual.width  || cameraContract.width;
  frameCanvas.height = actual.height || cameraContract.height;
  return actual;
}

function stopCamera() {
  clearInterval(frameTimer);
  stream?.getTracks().forEach(t => t.stop());
  stream = null;
  const cam = $('#camera');
  if (cam) cam.srcObject = null;
}

/* ═════════════════════ TRANSLATE MODULE ═══════════════════════ */

let translateSession = null;
let lastWord = '';

// Gesture → visual indicator
function updateGestureUI(state, didDraw) {
  const dot   = $('#gesture-dot');
  const label = $('#gesture-label');
  if (!dot || !label) return;

  const MAP = {
    NO_HAND:    ['', 'Chưa phát hiện tay'],
    UNKNOWN:    ['', 'Cử chỉ không xác định'],
    INDEX_ONLY: [didDraw ? 'drawing' : 'drawing', 'Đang vẽ… (ngón trỏ)'],
    TWO_FINGERS:['paused', '✌️ Pause — chuyển chữ tiếp theo'],
    OPEN_PALM:  ['', 'Xòe bàn tay (chưa sử dụng)'],
    FIST:       ['done', '✊ Nắm tay → Tự động Done!'],
  };
  const [dotClass, text] = MAP[state] || ['', state];
  dot.className = 'gesture-dot' + (dotClass ? ` ${dotClass}` : '');
  label.textContent = text;

  // FIST gesture auto-triggers Done
  if (state === 'FIST') {
    triggerDone();
  }
}

async function triggerDone() {
  const draftEl       = $('#draft');
  const draftPhEl     = $('#draft-placeholder');
  const candidatesEl  = $('#candidates');
  const commitBtn     = $('#commit');

  if (!translateSession) return;
  try {
    draftEl.textContent = '';
    if (draftPhEl) draftPhEl.textContent = 'Đang nhận diện chữ bằng AI…';
    const result = await api(`/recognition/sessions/${translateSession.session_id}/finish`, {
      method: 'POST',
    });
    translateSession = result;

    const word = result.draft || '';
    if (draftEl) draftEl.textContent = word;
    if (draftPhEl) draftPhEl.style.display = word ? 'none' : 'block';

    renderCandidateChips(result.segments || [], candidatesEl);

    const hasUncertain = (result.segments || []).some(s => s.status === 'uncertain');
    const showCommit = !!word && !hasUncertain;
    commitBtn?.classList.toggle('hidden', !showCommit);
  } catch (err) {
    if (draftPhEl) draftPhEl.textContent = err.message;
  }
}

function renderCandidateChips(segments, container) {
  if (!container) return;
  container.innerHTML = segments.map(seg => {
    const pct  = Math.round(seg.confidence * 100);
    const cls  = seg.status === 'uncertain' ? 'uncertain' : 'accepted';
    return `
      <div class="candidate-chip ${cls}">
        <span class="letter">${seg.prediction}</span>
        <span class="conf">${pct}%</span>
        <div class="conf-bar">
          <div class="conf-bar-fill" style="width:${pct}%"></div>
        </div>
      </div>`;
  }).join('');
}

function renderTranslation(entry) {
  const panel = $('#translation');
  if (!panel) return;
  panel.classList.remove('hidden');

  if (!entry) {
    panel.innerHTML = `
      <div class="translation-card">
        <div class="missing-word-card">
          <h3>${lastWord}</h3>
          <p>Từ "<strong>${lastWord}</strong>" chưa có trong từ điển nội bộ.</p>
          <p style="margin-top:8px; font-size:13px;">Thêm từ vào dataset rồi khởi động lại backend. Kết quả nhận diện vẫn hợp lệ.</p>
        </div>
      </div>`;
    return;
  }

  const pos       = entry.part_of_speech || '';
  const synonyms  = (entry.synonyms  || []).map(w => `<span class="word-tag synonym">${w}</span>`).join('');
  const antonyms  = (entry.antonyms  || []).map(w => `<span class="word-tag antonym">${w}</span>`).join('');
  const example   = entry.examples?.[0] || '';
  const exHtml    = example ? highlightExample(example, entry.word) : '<em>Chưa có ví dụ.</em>';

  panel.innerHTML = `
    <div class="translation-card" style="animation: slideUp 0.4s ease;">
      <div class="translation-header">
        <div class="translation-word">${entry.word}</div>
        <div class="word-meta">
          ${pos ? `<span class="pos-badge">${pos}</span>` : ''}
          <span class="vi-meaning">${entry.vietnamese_meaning}</span>
        </div>
      </div>

      <div class="translation-body">
        <div class="trans-section full-width">
          <div class="trans-label">📖 Định nghĩa</div>
          <div class="trans-content">${entry.definition}</div>
        </div>

        <div class="trans-section">
          <div class="trans-label">🟢 Từ đồng nghĩa</div>
          <div class="word-tags">
            ${synonyms || '<span style="color:var(--text-muted);font-size:13px;">Chưa có</span>'}
          </div>
        </div>

        <div class="trans-section">
          <div class="trans-label">🔴 Từ trái nghĩa</div>
          <div class="word-tags">
            ${antonyms || '<span style="color:var(--text-muted);font-size:13px;">Chưa có</span>'}
          </div>
        </div>

        <div class="trans-section full-width">
          <div class="trans-label">💬 Ví dụ</div>
          <div class="example-text">${exHtml}</div>
        </div>
      </div>

      <div class="translation-footer">
        <button id="save-word" class="btn-success">💾 Lưu vào từ điển cá nhân</button>
        <button id="try-again" class="btn-quiet">↩ Viết lại</button>
      </div>
    </div>`;

  $('#save-word')?.addEventListener('click', async () => {
    try {
      await api(`/vocabulary/${lastWord}/save`, { method: 'POST' });
      const btn = $('#save-word');
      if (btn) { btn.textContent = '✅ Đã lưu!'; btn.disabled = true; }
    } catch (err) {
      alert(err.message);
    }
  });

  $('#try-again')?.addEventListener('click', () => {
    panel.classList.add('hidden');
    const draftEl = $('#draft');
    if (draftEl) draftEl.textContent = '';
    const ph = $('#draft-placeholder');
    if (ph) { ph.textContent = 'Chưa có chữ nào…'; ph.style.display = ''; }
    $('#candidates').innerHTML = '';
    $('#commit')?.classList.add('hidden');
  });
}

/* ═══════════════ REVIEW MODULE (Spell + Cloze) ════════════════ */

let reviewSession = null;
let reviewStream  = null;
let reviewTimer   = null;
let reviewMode    = 'spell';   // 'spell' | 'cloze'
let currentReviewWord = null;  // full vocabulary entry
let clozeTemplate = [];        // [{ch, hidden}]
let clozeFilled   = [];        // filled characters for blank positions
let clozeBlankIdx = 0;         // which blank we're filling next
let spellLetters  = [];        // accumulated letters for spell mode

// Switch review mode
$$('.mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    $$('.mode-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    reviewMode = btn.dataset.mode;
    updateReviewModeUI();
    if (currentReviewWord) startReviewSession(currentReviewWord);
  });
});

function updateReviewModeUI() {
  const isCloze = reviewMode === 'cloze';
  const panelTitle  = $('#review-panel-title');
  const modeBadge   = $('#review-mode-badge');
  const clozeDiv    = $('#cloze-display');
  const spellHint   = $('#spell-hint');

  if (panelTitle)  panelTitle.textContent = isCloze ? '🧩 Cloze Mode' : '✍️ AirWrite Spell';
  if (modeBadge)   modeBadge.textContent  = isCloze ? 'CLOZE' : 'SPELL';
  if (clozeDiv)    clozeDiv.classList.toggle('hidden', !isCloze);
  if (spellHint)   spellHint.classList.toggle('hidden', isCloze);
}

async function loadReview() {
  const listEl = $('#review-word-list');
  if (!listEl) return;
  try {
    const words = await api('/vocabulary');
    listEl.innerHTML = words.length
      ? words.map(word => `
          <article class="card" data-word="${word.normalized_word}">
            <span class="eyebrow">${word.part_of_speech || 'word'}</span>
            <h2>${word.word}</h2>
            <p>${word.vietnamese_meaning}</p>
            <div class="card-footer">
              <button class="btn-quiet review-start-btn" data-word="${word.normalized_word}">
                ✍️ Ôn tập
              </button>
            </div>
          </article>`).join('')
      : `<div class="empty-state">
           <span class="icon">📚</span>
           <p>Hãy lưu một từ đã dịch để bắt đầu ôn tập.</p>
         </div>`;

    $$('.review-start-btn').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const word = await api(`/dictionary/${btn.dataset.word}`);
        openReviewSession(word);
      });
    });

  } catch (err) {
    listEl.innerHTML = `<p class="status-text error">${err.message}</p>`;
  }
}

function openReviewSession(wordEntry) {
  currentReviewWord = wordEntry;
  $('#review-word-list').classList.add('hidden');
  $('#review-session').classList.remove('hidden');
  startReviewSession(wordEntry);
}

function startReviewSession(wordEntry) {
  spellLetters  = [];
  clozeTemplate = makeClozeTemplate(wordEntry.word);
  clozeFilled   = [];
  clozeBlankIdx = 0;

  // Show Vietnamese prompt
  const vnEl = $('#vn-prompt-word');
  if (vnEl) vnEl.textContent = wordEntry.vietnamese_meaning;

  // Show/hide English word (only in cloze)
  const enEl = $('#review-word-english');
  if (enEl) {
    enEl.textContent = reviewMode === 'cloze' ? wordEntry.word : '';
    enEl.classList.toggle('hidden', reviewMode !== 'cloze');
  }

  // Clear live letters
  const liveEl = $('#live-letters-overlay');
  if (liveEl) liveEl.innerHTML = '';

  // Render cloze word
  if (reviewMode === 'cloze') renderClozeDisplay();

  // Show done button when camera starts
  updateReviewModeUI();

  const feedback = $('#review-feedback');
  if (feedback) { feedback.classList.add('hidden'); feedback.innerHTML = ''; }
  $('#review-done')?.classList.add('hidden');
  $('#review-clear')?.classList.add('hidden');
  $('#review-next')?.classList.add('hidden');
}

function renderClozeDisplay() {
  const container = $('#cloze-word');
  if (!container) return;
  container.innerHTML = clozeTemplate.map((item, i) => {
    const blanks = clozeTemplate.filter(x => x.hidden);
    const blankIdx = blanks.findIndex((_, bi) => clozeTemplate.indexOf(blanks[bi]) === i);
    if (!item.hidden) {
      return `<div class="cloze-letter shown">${item.ch}</div>`;
    }
    const filled = clozeFilled[blankIdx];
    if (filled !== undefined) {
      return `<div class="cloze-letter filled">${filled}</div>`;
    }
    return `<div class="cloze-letter blank"></div>`;
  }).join('');
}

// Review camera start
$('#review-start-camera')?.addEventListener('click', async () => {
  const statusEl = $('#review-status');
  try {
    const actual = await startCamera($('#camera'));
    statusEl.textContent = `Camera: ${actual.width || '?'}×${actual.height || '?'} @ ${(actual.frameRate || 0).toFixed(0)} FPS`;
    if ($('#camera').style) $('#camera').style.transform = cameraContract.mirror ? 'scaleX(-1)' : 'none';

    // Start recognition session
    reviewSession = await api('/recognition/sessions', { method: 'POST' });

    // Start frame loop
    reviewTimer = buildFrameSender({
      sessionGetter: () => reviewSession,
      onFrame: (frame) => {
        const img = $('#review-airwrite-frame');
        if (img && frame.frame) {
          img.src = frame.frame;
          img.classList.remove('hidden');
          $('#review-cam-placeholder')?.classList.add('hidden');
        }
        statusEl.textContent = `AirWrite: ${frame.state}`;
      },
      onGesture: (state, didDraw) => {
        // OPEN_PALM auto-trigger done in review
        if (state === 'OPEN_PALM') triggerReviewDone();
      },
      onError: (msg) => { statusEl.textContent = msg; },
    });

    $('#review-done')?.classList.remove('hidden');
    $('#review-clear')?.classList.remove('hidden');
    $('#review-start-camera')?.classList.add('hidden');

  } catch (err) {
    if (statusEl) statusEl.textContent = `Camera lỗi: ${err.message}`;
  }
});

$('#review-clear')?.addEventListener('click', async () => {
  if (!reviewSession) return;
  try {
    await api(`/recognition/sessions/${reviewSession.session_id}/clear`, { method: 'POST' });
    // In spell mode, remove last letter from overlay
    if (reviewMode === 'spell' && spellLetters.length > 0) {
      spellLetters.pop();
      renderSpellOverlay();
    }
    // In cloze mode, undo last fill
    if (reviewMode === 'cloze' && clozeFilled.length > 0) {
      clozeFilled.pop();
      if (clozeBlankIdx > 0) clozeBlankIdx--;
      renderClozeDisplay();
    }
  } catch (err) {
    console.warn('Clear error:', err);
  }
});

$('#review-done')?.addEventListener('click', triggerReviewDone);

async function triggerReviewDone() {
  if (!reviewSession) return;
  try {
    const result = await api(`/recognition/sessions/${reviewSession.session_id}/finish`, {
      method: 'POST',
    });
    reviewSession = result;
    const letter = result.draft?.slice(-1) || '';

    if (reviewMode === 'spell') {
      if (letter) {
        spellLetters.push(letter);
        renderSpellOverlay();
      }
      // Auto-evaluate when word length matches
      if (currentReviewWord && spellLetters.length >= currentReviewWord.word.length) {
        evaluateSpell();
      }
    } else if (reviewMode === 'cloze') {
      // Fill next blank
      const blanks = clozeTemplate.filter(x => x.hidden);
      if (letter && clozeBlankIdx < blanks.length) {
        clozeFilled[clozeBlankIdx] = letter;
        clozeBlankIdx++;
        renderClozeDisplay();
      }
      // Auto-evaluate when all blanks filled
      if (clozeBlankIdx >= blanks.length) {
        evaluateCloze();
      }
    }
  } catch (err) {
    const s = $('#review-status');
    if (s) s.textContent = err.message;
  }
}

function renderSpellOverlay() {
  const overlay = $('#live-letters-overlay');
  if (!overlay) return;
  overlay.innerHTML = spellLetters.map(ch =>
    `<div class="live-letter">${ch}</div>`
  ).join('');
}

function evaluateSpell() {
  if (!currentReviewWord) return;
  const written   = spellLetters.join('').toLowerCase();
  const expected  = currentReviewWord.word.toLowerCase();
  const correct   = written === expected;
  showReviewFeedback(correct, expected, written);
}

function evaluateCloze() {
  if (!currentReviewWord) return;
  const blanks  = clozeTemplate.filter(x => x.hidden);
  let allCorrect = true;

  // Color each filled blank
  const container = $('#cloze-word');
  const blankEls  = container?.querySelectorAll('.cloze-letter.filled') || [];
  blankEls.forEach((el, i) => {
    const expected = blanks[i]?.ch.toLowerCase() || '';
    const filled   = (clozeFilled[i] || '').toLowerCase();
    const ok = filled === expected;
    el.classList.remove('filled');
    el.classList.add(ok ? 'correct' : 'wrong');
    if (!ok) allCorrect = false;
  });

  showReviewFeedback(allCorrect, currentReviewWord.word, clozeFilled.join(''));
}

function showReviewFeedback(correct, expected, written) {
  const feedback = $('#review-feedback');
  if (!feedback) return;
  feedback.classList.remove('hidden');
  feedback.innerHTML = `
    <div class="result-banner ${correct ? 'correct' : 'wrong'}">
      <span class="icon">${correct ? '🎉' : '❌'}</span>
      <div>
        <div class="text">${correct ? 'Chính xác!' : 'Chưa đúng'}</div>
        ${!correct ? `<div class="detail">Bạn viết: <strong>${written}</strong> — Đúng là: <strong>${expected}</strong></div>` : ''}
      </div>
    </div>`;
  $('#review-next')?.classList.remove('hidden');
}

$('#review-next')?.addEventListener('click', () => {
  // Reset for next attempt of same word
  startReviewSession(currentReviewWord);
  spellLetters = [];
  renderSpellOverlay();
  $('#review-feedback').classList.add('hidden');
  $('#review-next')?.classList.add('hidden');
});

$('#review-back')?.addEventListener('click', () => {
  // Stop review camera
  clearInterval(reviewTimer);
  reviewTimer = null;
  stopCamera();
  // Reset session UI
  reviewSession = null;
  currentReviewWord = null;
  spellLetters = [];
  $('#review-session')?.classList.add('hidden');
  $('#review-word-list')?.classList.remove('hidden');
  $('#review-airwrite-frame')?.classList.add('hidden');
  $('#review-cam-placeholder')?.classList.remove('hidden');
  $('#review-start-camera')?.classList.remove('hidden');
  $('#review-done')?.classList.add('hidden');
  $('#review-clear')?.classList.add('hidden');
  $('#review-next')?.classList.add('hidden');
  loadReview();
});

/* \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550 COLLECTIONS MODULE \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550 */

let currentCollection = null; // the opened collection object

async function loadCollections() {
  const container = $('#collections');
  if (!container) return;
  // Always show grid view, hide detail
  $('#collections-view')?.classList.remove('hidden');
  $('#collection-detail')?.classList.add('hidden');

  try {
    const sets = await api('/collections');
    container.innerHTML = sets.length ? sets.map(set => `
      <article class="card collection-card" data-id="${set.id}" tabindex="0" role="button"
               aria-label="M\u1edf ch\u1ee7 \u0111\u1ec1 ${set.name}">
        <span class="eyebrow">${set.category}</span>
        <h2>${set.name}</h2>
        <p>${set.description}</p>
        <div class="card-footer">
          <span class="mode-badge">${set.total_words ?? (set.vocabulary?.length ?? 0)} t\u1eeb</span>
          <span style="font-size:12px; color:var(--text-muted); margin-left:auto;">Nh\u1ea5n \u0111\u1ec3 xem \u2192</span>
        </div>
      </article>`).join('')
    : `<div class="empty-state" style="grid-column:1/-1;">
         <span class="icon">📚</span>
         <p>Ch\u01b0a c\u00f3 b\u1ed9 t\u1eeb n\u00e0o. Th\u00eam collection v\u00e0o backend.</p>
       </div>`;

    $$('.collection-card').forEach(card => {
      card.addEventListener('click', () => openCollection(card.dataset.id));
      card.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') openCollection(card.dataset.id); });
    });
  } catch (err) {
    container.innerHTML = `<p class="status-text error">${err.message}</p>`;
  }
}

async function openCollection(id) {
  try {
    const set = await api(`/collections/${id}`);
    currentCollection = set;

    // Populate header
    const catEl = $('#detail-category');
    const nameEl = $('#detail-name');
    const descEl = $('#detail-description');
    if (catEl) catEl.textContent = set.category;
    if (nameEl) nameEl.textContent = set.name;
    if (descEl) descEl.textContent = set.description;

    // Populate vocabulary list
    const vocabEl = $('#detail-vocab-list');
    if (vocabEl) {
      vocabEl.innerHTML = (set.vocabulary || []).map((word, idx) => `
        <div class="vocab-card">
          <div class="vocab-card-word">${idx + 1}. ${word.word}</div>
          <div class="vocab-card-vi">${word.vietnamese_meaning}</div>
          ${word.part_of_speech ? `<span class="vocab-card-pos">${word.part_of_speech}</span>` : '<span></span>'}
          <div class="vocab-card-def">${word.definition || '<em style="color:var(--text-muted)">Ch\u01b0a c\u00f3 \u0111\u1ecbnh ngh\u0129a.</em>'}</div>
        </div>`).join('');
    }

    // Switch view
    $('#collections-view')?.classList.add('hidden');
    $('#collection-detail')?.classList.remove('hidden');

  } catch (err) {
    const container = $('#collections');
    if (container) container.innerHTML = `<p class="status-text error">${err.message}</p>`;
  }
}

// Back button
$('#back-to-collections-btn')?.addEventListener('click', loadCollections);

// Learn buttons in detail view
$$('.collection-detail-learn-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    if (!currentCollection) return;
    const mode = btn.dataset.mode; // 'spell' | 'cloze'
    reviewMode = mode === 'cloze' ? 'cloze' : 'spell';

    // Update mode selector tabs in review screen
    $$('.mode-btn').forEach(b => b.classList.remove('active'));
    const targetId = mode === 'cloze' ? 'mode-btn-cloze' : 'mode-btn-spell';
    $(`#${targetId}`)?.classList.add('active');
    updateReviewModeUI();

    // Navigate to review tab and open with first word
    const firstWord = currentCollection.vocabulary?.[0];
    if (firstWord) {
      location.hash = '#review';
      setTimeout(() => openReviewSession(firstWord), 120);
    }
  });
});

/* ══════════════════ PROGRESS MODULE ═══════════════════════════ */

async function loadProgress() {
  const listEl = $('#progress-list');
  if (!listEl) return;
  try {
    const items = await api('/progress');
    listEl.innerHTML = items.length
      ? items.map(item => `
          <div class="progress-card">
            <h2>${item.word}</h2>
            <div class="vi" id="pvi-${item.vocabulary_id}">—</div>
            <div class="mastery-bar-wrap">
              <div class="mastery-label">
                <span>Mức độ thuần thục</span>
                <span>${Math.round(item.mastery * 100)}%</span>
              </div>
              <div class="mastery-bar">
                <div class="mastery-fill" style="width:${item.mastery * 100}%"></div>
              </div>
            </div>
            <span class="state-badge ${item.state}">${item.state === 'mastered' ? '🏆 Thành thạo' : item.state === 'learning' ? '📈 Đang học' : '🆕 Mới'}</span>
          </div>`)
        .join('')
      : `<div class="empty-state" style="grid-column:1/-1;">
           <span class="icon">📊</span>
           <p>Chưa có lần luyện tập nào. Hãy dịch và lưu từ, sau đó ôn tập!</p>
         </div>`;

    // Load Vietnamese meaning for each progress item
    items.forEach(async item => {
      try {
        const entry = await api(`/dictionary/${item.word}`);
        const el = $(`#pvi-${item.vocabulary_id}`);
        if (el) el.textContent = entry.vietnamese_meaning;
      } catch {}
    });

  } catch (err) {
    listEl.innerHTML = `<p class="status-text error">${err.message}</p>`;
  }
}

/* ══════════════════ TRANSLATE WIRING ══════════════════════════ */

// Load camera settings + health check
api('/camera/settings').then(settings => {
  cameraContract = settings;
  $('#camera-contract').textContent = `${settings.width}×${settings.height} · ${settings.fps} FPS`;
}).catch(() => {
  $('#camera-status').textContent = 'Camera contract không khả dụng, dùng mặc định AirWrite.';
});

api('/health').then(health => {
  // silently update airwrite status (not shown in UI)
  const el = $('#airwrite-status');
  if (!el) return;
  el.classList.remove('loading-pulse');
  if (health.airwrite === 'ready') {
    el.classList.add('active');
  }
}).catch(() => {
  const el = $('#airwrite-status');
  if (el) { el.classList.remove('loading-pulse'); }
});

// Start camera (translate page)
$('#start-camera')?.addEventListener('click', async () => {
  const statusEl = $('#camera-status');
  try {
    const actual = await startCamera($('#camera'));
    const cam = $('#camera');
    if (cam) cam.style.transform = cameraContract.mirror ? 'scaleX(-1)' : 'none';
    statusEl.textContent = `Camera: ${actual.width || '?'}×${actual.height || '?'} @ ${(actual.frameRate || 0).toFixed(1)} FPS`;

    translateSession = await api('/recognition/sessions', { method: 'POST' });

    frameTimer = buildFrameSender({
      sessionGetter: () => translateSession,
      onFrame: (frame) => {
        const img = $('#airwrite-frame');
        if (img && frame.frame) {
          img.src = frame.frame;
          img.classList.remove('hidden');
          $('#camera-placeholder')?.classList.add('hidden');
        }
        statusEl.textContent = `Cử chỉ: ${frame.state}`;
        updateGestureUI(frame.state, frame.did_draw);
      },
      onError: (msg) => { statusEl.textContent = msg; },
    });

    $('#start-camera')?.classList.add('hidden');
    $('#stop-camera')?.classList.remove('hidden');
  } catch (err) {
    statusEl.textContent = `Camera lỗi: ${err.message}`;
  }
});

// Stop camera
$('#stop-camera')?.addEventListener('click', () => {
  clearInterval(frameTimer);
  frameTimer = null;
  stopCamera();
  translateSession = null;
  $('#start-camera')?.classList.remove('hidden');
  $('#stop-camera')?.classList.add('hidden');
  $('#airwrite-frame')?.classList.add('hidden');
  $('#camera-placeholder')?.classList.remove('hidden');
  $('#camera-status').textContent = 'Camera đã tắt.';
  const dot = $('#gesture-dot');
  if (dot) dot.className = 'gesture-dot';
  const label = $('#gesture-label');
  if (label) label.textContent = 'Chưa phát hiện tay';
});

// Clear — 1st click: xóa nét trước lần pause cuối; 2nd click liên tiếp: xóa toàn bộ canvas
let clearClickTimer = null;
let clearClickCount = 0;
$('#clear-canvas')?.addEventListener('click', async () => {
  if (!translateSession) return;
  clearClickCount++;
  clearTimeout(clearClickTimer);

  if (clearClickCount >= 2) {
    // Nhấn 2 lần: xóa toàn bộ
    clearClickCount = 0;
    try {
      await api(`/recognition/sessions/${translateSession.session_id}/clear_all`, { method: 'POST' });
    } catch {}
    // fallback: also try standard clear twice
    try {
      await api(`/recognition/sessions/${translateSession.session_id}/clear`, { method: 'POST' });
    } catch {}
    $('#draft').textContent = '';
    const ph = $('#draft-placeholder');
    if (ph) { ph.textContent = 'Đã xóa toàn bộ canvas'; ph.style.display = ''; }
    $('#candidates').innerHTML = '';
    $('#commit')?.classList.add('hidden');
    const btn = $('#clear-canvas');
    if (btn) { btn.textContent = 'Đã xóa toàn bộ'; setTimeout(() => { btn.textContent = 'Xóa'; }, 1500); }
  } else {
    // Nhấn 1 lần: xóa nét trước lần pause cuối
    try {
      await api(`/recognition/sessions/${translateSession.session_id}/clear`, { method: 'POST' });
      $('#draft').textContent = '';
      const ph = $('#draft-placeholder');
      if (ph) { ph.textContent = 'Đã xóa nét trước pause cuối — nhấn lần nữa để xóa tất cả'; ph.style.display = ''; }
      $('#candidates').innerHTML = '';
      $('#commit')?.classList.add('hidden');
    } catch (err) {
      $('#camera-status').textContent = err.message;
    }
    // Reset count sau 1.5s nếu không nhấn lần 2
    clearClickTimer = setTimeout(() => { clearClickCount = 0; }, 1500);
  }
});

// Done button
$('#done')?.addEventListener('click', triggerDone);

// Commit (translate)
$('#commit')?.addEventListener('click', async () => {
  if (!translateSession) return;
  try {
    const result = await api(`/recognition/sessions/${translateSession.session_id}/commit`, {
      method: 'POST',
    });
    lastWord = result.word;
    renderTranslation(result.entry);
  } catch (err) {
    const ph = $('#draft-placeholder');
    if (ph) ph.textContent = err.message;
  }
});

/* ══════════════════ ROUTER ════════════════════════════════════ */

function onHashChange() {
  const screen = location.hash.slice(1) || 'translate';
  $$('.screen').forEach(el => el.classList.toggle('active', el.id === screen));
  $$('nav a').forEach(a => a.classList.toggle('active', a.getAttribute('href') === `#${screen}`));

  if (screen === 'review')   loadReview();
  if (screen === 'learn')    loadCollections();
  if (screen === 'progress') loadProgress();
}

window.addEventListener('hashchange', onHashChange);
window.dispatchEvent(new HashChangeEvent('hashchange'));
