/**
 * AirWrite Learn English — Frontend Application
 * Modules: Camera, Translate, Review (Spell + Cloze), Collections, Progress
 */

/* ═══════════════════════ CORE UTILITIES ═══════════════════════ */

const API = '/api';
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

// Generate cloze template: hide 1-3 letters depending on word length
function makeClozeTemplate(word) {
  const letters = (word || '').split('');
  const n = letters.length;
  let blankCount = 1;
  if (n >= 4 && n <= 6) blankCount = 2;
  else if (n > 6) blankCount = Math.min(4, Math.max(2, Math.round(n * 0.35)));

  let pool = [];
  if (n <= 2) {
    pool = [n - 1];
  } else if (n === 3) {
    pool = [1];
  } else {
    for (let i = 1; i < n - 1; i++) pool.push(i);
  }
  const shuffled = [...pool].sort(() => Math.random() - 0.5);
  const hiddenIndices = new Set(shuffled.slice(0, blankCount).sort((a, b) => a - b));

  let blankCounter = 0;
  return letters.map((ch, i) => {
    const isHidden = hiddenIndices.has(i);
    return {
      ch,
      hidden: isHidden,
      blankIndex: isHidden ? blankCounter++ : null,
    };
  });
}

/* ═══════════════════════ CAMERA MODULE ════════════════════════ */

let stream = null;
let frameTimer = null;
let reviewTimer = null;
let isReviewEvaluating = false;
let reviewEvaluated = false;
let lastReviewGesture = null;
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
      width: { ideal: cameraContract.width },
      height: { ideal: cameraContract.height },
      frameRate: { ideal: cameraContract.fps, max: cameraContract.fps },
      facingMode: 'user',
    },
  });
  videoEl.srcObject = stream;
  const actual = stream.getVideoTracks()[0]?.getSettings() || {};
  frameCanvas.width = actual.width || cameraContract.width;
  frameCanvas.height = actual.height || cameraContract.height;
  return actual;
}

function stopCamera() {
  if (frameTimer) {
    clearInterval(frameTimer);
    frameTimer = null;
  }
  if (reviewTimer) {
    clearInterval(reviewTimer);
    reviewTimer = null;
  }
  stream?.getTracks().forEach(t => t.stop());
  stream = null;
  const cam = $('#camera');
  if (cam) cam.srcObject = null;
}

function stopTranslateCamera() {
  if (frameTimer) {
    clearInterval(frameTimer);
    frameTimer = null;
  }
  stopCamera();
  translateSession = null;
  $('#start-camera')?.classList.remove('hidden');
  $('#stop-camera')?.classList.add('hidden');
  $('#airwrite-frame')?.classList.add('hidden');
  $('#camera-placeholder')?.classList.remove('hidden');
  const statusEl = $('#camera-status');
  if (statusEl) statusEl.textContent = 'Camera đã tắt.';
  const dot = $('#gesture-dot');
  if (dot) dot.className = 'gesture-dot';
  const label = $('#gesture-label');
  if (label) label.textContent = 'Chưa phát hiện tay';
}

function stopReviewCamera() {
  if (reviewTimer) {
    clearInterval(reviewTimer);
    reviewTimer = null;
  }
  isReviewEvaluating = false;
  reviewEvaluated = false;
  lastReviewGesture = null;
  stopCamera();
  if (reviewSession) {
    api(`/recognition/sessions/${reviewSession.session_id}/clear_all`, { method: 'POST' }).catch(() => {});
  }
  reviewSession = null;
  $('#review-airwrite-frame')?.classList.add('hidden');
  $('#review-cam-placeholder')?.classList.remove('hidden');
  $('#review-start-camera')?.classList.remove('hidden');
  $('#review-done')?.classList.add('hidden');
  $('#review-clear')?.classList.add('hidden');
  const statusEl = $('#review-status');
  if (statusEl) statusEl.textContent = 'Camera chưa bật.';
}

/* ═════════════════════ TRANSLATE MODULE ═══════════════════════ */

let translateSession = null;
let lastWord = '';
let isTranslateEvaluating = false;
let lastTranslateGesture = null;

// Gesture → visual indicator
function updateGestureUI(state, didDraw) {
  const dot = $('#gesture-dot');
  const label = $('#gesture-label');
  if (!dot || !label) return;

  const MAP = {
    NO_HAND: ['', 'Chưa phát hiện tay'],
    UNKNOWN: ['', 'Cử chỉ không xác định'],
    INDEX_ONLY: [didDraw ? 'drawing' : 'drawing', 'Đang vẽ… (ngón trỏ)'],
    TWO_FINGERS: ['paused', '✌️ Pause — chuyển chữ tiếp theo'],
    OPEN_PALM: ['', 'Xòe bàn tay (chưa sử dụng)'],
    FIST: ['done', '✊ Nắm tay → Tự động Done!'],
  };
  const [dotClass, text] = MAP[state] || ['', state];
  dot.className = 'gesture-dot' + (dotClass ? ` ${dotClass}` : '');
  label.textContent = text;

  // FIST gesture auto-triggers Done only once on edge transition
  if (state === 'FIST' && lastTranslateGesture !== 'FIST') {
    if (!isTranslateEvaluating) {
      triggerDone();
    }
  }
  lastTranslateGesture = state;
}

async function triggerDone() {
  const draftEl = $('#draft');
  const draftPhEl = $('#draft-placeholder');
  const candidatesEl = $('#candidates');
  const commitBtn = $('#commit');

  if (!translateSession || isTranslateEvaluating) return;
  isTranslateEvaluating = true;
  try {
    if (draftPhEl) {
      draftPhEl.textContent = 'Đang nhận diện chữ bằng AI…';
      draftPhEl.style.display = 'block';
    }
    const result = await api(`/recognition/sessions/${translateSession.session_id}/finish`, {
      method: 'POST',
    });
    translateSession = result;

    const word = result.draft || '';
    if (draftEl) draftEl.textContent = word;
    if (draftPhEl) draftPhEl.style.display = word ? 'none' : 'block';

    renderCandidateChips(result.segments || [], candidatesEl);

    // Show commit button whenever a word is recognized
    commitBtn?.classList.toggle('hidden', !word);
  } catch (err) {
    if (draftPhEl) draftPhEl.textContent = err.message;
  } finally {
    isTranslateEvaluating = false;
  }
}

function renderCandidateChips(segments, container) {
  if (!container) return;
  container.innerHTML = segments.map((seg, segIdx) => {
    const pct = Math.round(seg.confidence * 100);
    const cls = seg.status === 'uncertain' ? 'uncertain' : 'accepted';
    const otherCandidates = (seg.candidates || []).filter(c => c.label !== seg.prediction);
    const candBtns = otherCandidates.map(c => `
      <button class="candidate-alt-btn" data-seg-idx="${segIdx}" data-letter="${c.label}" title="Đổi thành '${c.label}' (${Math.round(c.confidence * 100)}%)">${c.label}</button>
    `).join('');
    return `
      <div class="candidate-chip ${cls}">
        <span class="letter">${seg.prediction}</span>
        <span class="conf">${pct}%</span>
        <div class="conf-bar">
          <div class="conf-bar-fill" style="width:${pct}%"></div>
        </div>
        ${candBtns ? `<div class="candidate-alts">${candBtns}</div>` : ''}
      </div>`;
  }).join('');

  container.querySelectorAll('.candidate-alt-btn').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const segIdx = parseInt(btn.dataset.segIdx, 10);
      const newChar = btn.dataset.letter;
      if (!translateSession || !translateSession.segments) return;
      if (translateSession.segments[segIdx]) {
        translateSession.segments[segIdx].prediction = newChar;
        translateSession.segments[segIdx].status = 'accepted';
      }
      const newDraft = translateSession.segments.map(s => s.prediction).join('');
      translateSession.draft = newDraft;
      const draftEl = $('#draft');
      if (draftEl) draftEl.textContent = newDraft;
      renderCandidateChips(translateSession.segments, container);
      try {
        await api(`/recognition/sessions/${translateSession.session_id}`, {
          method: 'PATCH',
          body: JSON.stringify({ draft: newDraft }),
        });
      } catch { }
      $('#commit')?.classList.remove('hidden');
    });
  });
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

  const pos = entry.part_of_speech || '';
  const synonyms = (entry.synonyms || []).map(w => `<span class="word-tag synonym">${w}</span>`).join('');
  const antonyms = (entry.antonyms || []).map(w => `<span class="word-tag antonym">${w}</span>`).join('');
  const example = entry.examples?.[0] || '';
  const exHtml = example ? highlightExample(example, entry.word) : '<em>Chưa có ví dụ.</em>';

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
let reviewStream = null;
let reviewMode = 'spell';   // 'spell' | 'cloze'
let currentReviewWord = null;  // full vocabulary entry
let clozeTemplate = [];        // [{ch, hidden, blankIndex}]
let clozeFilled = [];          // filled characters for blank positions
let spellLetters = [];         // recognized letters for spell mode
let reviewWordList = [];       // loaded review vocabulary list

// Switch review mode with complete canvas & state reset
async function switchReviewMode(newMode) {
  if (reviewMode === newMode && currentReviewWord && !$('#review-session')?.classList.contains('hidden')) {
    return;
  }
  reviewMode = newMode;
  $$('.mode-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.mode === reviewMode));
  updateReviewModeUI();

  // Reset backend canvas completely to wipe out any drawing
  if (reviewSession) {
    try {
      await api(`/recognition/sessions/${reviewSession.session_id}/clear_all`, { method: 'POST' });
    } catch (e) {
      console.warn('Canvas clear on mode switch failed:', e);
    }
  }

  // Clear internal state & overlays
  isReviewEvaluating = false;
  reviewEvaluated = false;
  lastReviewGesture = null;
  spellLetters = [];
  clozeFilled = [];

  const liveEl = $('#live-letters-overlay');
  if (liveEl) liveEl.innerHTML = '';

  const feedback = $('#review-feedback');
  if (feedback) { feedback.classList.add('hidden'); feedback.innerHTML = ''; }
  $('#review-next')?.classList.add('hidden');

  if (currentReviewWord) {
    if (reviewMode === 'cloze') {
      clozeTemplate = makeClozeTemplate(currentReviewWord.word);
      renderClozeDisplay(false);
    } else {
      const clozeContainer = $('#cloze-word');
      if (clozeContainer) clozeContainer.innerHTML = '';
    }
    const statusEl = $('#review-status');
    if (statusEl && reviewSession) {
      statusEl.textContent = `Chế độ: ${reviewMode === 'cloze' ? 'Cloze (Đục lỗ)' : 'AirWrite Spell'} — Canvas đã làm mới.`;
    }
  }
}

$$('.mode-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    await switchReviewMode(btn.dataset.mode);
  });
});

function updateReviewModeUI() {
  const isCloze = reviewMode === 'cloze';
  const panelTitle = $('#review-panel-title');
  const modeBadge = $('#review-mode-badge');
  const clozeDiv = $('#cloze-display');
  const spellHint = $('#spell-hint');

  if (panelTitle) panelTitle.textContent = isCloze ? '🧩 Cloze Mode (Đục lỗ)' : '✍️ AirWrite Spell';
  if (modeBadge) modeBadge.textContent = isCloze ? 'CLOZE' : 'SPELL';
  if (clozeDiv) clozeDiv.classList.toggle('hidden', !isCloze);
  if (spellHint) {
    spellHint.classList.toggle('hidden', isCloze);
    spellHint.textContent = 'Viết toàn bộ chữ cái tiếng Anh vào không khí bằng ngón trỏ. Khi viết xong, nhấn nút ✔ Done (hoặc nắm tay ✊) để kiểm tra kết quả.';
  }
}

async function loadReview() {
  const listEl = $('#review-word-list');
  if (!listEl) return;
  try {
    const words = await api('/vocabulary');
    reviewWordList = words;
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

async function startReviewSession(wordEntry) {
  currentReviewWord = wordEntry;
  isReviewEvaluating = false;
  reviewEvaluated = false;
  lastReviewGesture = null;
  spellLetters = [];
  clozeTemplate = makeClozeTemplate(wordEntry.word);
  clozeFilled = [];

  // Reset backend canvas if session is already running
  if (reviewSession) {
    try {
      await api(`/recognition/sessions/${reviewSession.session_id}/clear_all`, { method: 'POST' });
    } catch { }
  }

  // Show Vietnamese prompt
  const vnEl = $('#vn-prompt-word');
  if (vnEl) vnEl.textContent = wordEntry.vietnamese_meaning;

  // Always keep English word hidden during testing
  const enEl = $('#review-word-english');
  if (enEl) enEl.classList.add('hidden');

  // Clear live overlay
  const liveEl = $('#live-letters-overlay');
  if (liveEl) liveEl.innerHTML = '';

  // Render cloze if in cloze mode
  if (reviewMode === 'cloze') {
    renderClozeDisplay(false);
  } else {
    const clozeContainer = $('#cloze-word');
    if (clozeContainer) clozeContainer.innerHTML = '';
  }

  updateReviewModeUI();

  const feedback = $('#review-feedback');
  if (feedback) { feedback.classList.add('hidden'); feedback.innerHTML = ''; }
  $('#review-next')?.classList.add('hidden');

  if (reviewSession) {
    $('#review-done')?.classList.remove('hidden');
    $('#review-clear')?.classList.remove('hidden');
  }
}

function renderClozeDisplay(evaluated = false) {
  const container = $('#cloze-word');
  if (!container) return;
  const blanks = clozeTemplate.filter(x => x.hidden);

  const hintEl = $('#cloze-display .hint-text');
  if (hintEl) {
    hintEl.innerHTML = `Từ bị đục lỗ — hãy viết <strong>${blanks.length} chữ cái còn thiếu</strong> vào không khí rồi nhấn <strong>✔ Done</strong>:`;
  }

  container.innerHTML = clozeTemplate.map(item => {
    if (!item.hidden) {
      return `<div class="cloze-letter shown">${item.ch}</div>`;
    }
    const filledChar = clozeFilled[item.blankIndex] || '';
    if (!evaluated) {
      if (filledChar) {
        return `<div class="cloze-letter filled">${filledChar}</div>`;
      }
      return `<div class="cloze-letter blank">_</div>`;
    }
    // Evaluated: strictly check match
    const expectedChar = item.ch.toLowerCase();
    const isMatch = filledChar.toLowerCase() === expectedChar;
    if (isMatch) {
      return `<div class="cloze-letter correct" title="Chính xác">${filledChar}</div>`;
    } else {
      return `<div class="cloze-letter wrong" title="Đúng là: ${item.ch}">${filledChar || '∅'}</div>`;
    }
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
        // FIST gesture triggers Done only once on edge transition and only if not yet evaluated
        if (state === 'FIST' && lastReviewGesture !== 'FIST') {
          if (!isReviewEvaluating && !reviewEvaluated) {
            triggerReviewDone();
          }
        }
        lastReviewGesture = state;
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
    await api(`/recognition/sessions/${reviewSession.session_id}/clear_all`, { method: 'POST' });
  } catch (err) {
    console.warn('Clear error:', err);
  }
  isReviewEvaluating = false;
  reviewEvaluated = false;
  lastReviewGesture = null;
  spellLetters = [];
  clozeFilled = [];

  const liveEl = $('#live-letters-overlay');
  if (liveEl) liveEl.innerHTML = '';

  if (reviewMode === 'cloze') {
    renderClozeDisplay(false);
  }

  const feedback = $('#review-feedback');
  if (feedback) { feedback.classList.add('hidden'); feedback.innerHTML = ''; }
  $('#review-next')?.classList.add('hidden');

  const s = $('#review-status');
  if (s) s.textContent = 'Đã xóa toàn bộ nét vẽ. Bạn có thể viết lại.';
});

$('#review-done')?.addEventListener('click', triggerReviewDone);

async function triggerReviewDone() {
  if (!reviewSession || isReviewEvaluating || reviewEvaluated) return;
  isReviewEvaluating = true;
  const doneBtn = $('#review-done');
  if (doneBtn) doneBtn.disabled = true;
  const statusEl = $('#review-status');
  try {
    if (statusEl) statusEl.textContent = 'Đang nhận diện chữ viết qua AI…';
    const result = await api(`/recognition/sessions/${reviewSession.session_id}/finish`, {
      method: 'POST',
    });
    reviewSession = result;
    const recognized = (result.draft || '').trim().toLowerCase();

    reviewEvaluated = true;

    if (reviewMode === 'spell') {
      evaluateSpell(recognized);
    } else if (reviewMode === 'cloze') {
      evaluateCloze(recognized, result.segments || []);
    }
    if (statusEl) statusEl.textContent = `Nhận diện: "${recognized || '(trống)'}"`;
  } catch (err) {
    if (statusEl) statusEl.textContent = `Lỗi nhận diện: ${err.message}`;
  } finally {
    isReviewEvaluating = false;
    if (doneBtn) doneBtn.disabled = false;
  }
}

function evaluateSpell(recognized) {
  if (!currentReviewWord) return;
  const expected = (currentReviewWord.word || '').trim().toLowerCase();
  const correct = recognized.length > 0 && recognized === expected;

  // Build letter-by-letter diff comparison boxes
  const expChars = expected.split('');
  const recChars = (recognized || '').split('');
  const maxLen = Math.max(expChars.length, recChars.length);
  const diffBoxes = [];

  for (let i = 0; i < maxLen; i++) {
    const exp = expChars[i] || '';
    const rec = recChars[i] || '';
    const isMatch = exp.toLowerCase() === rec.toLowerCase();
    diffBoxes.push(`
      <div class="spell-compare-box ${isMatch ? 'correct' : 'wrong'}">
        <span class="expected-char">${exp || '—'}</span>
        ${!isMatch ? `<span class="actual-char">${rec || '∅'}</span>` : ''}
      </div>
    `);
  }

  const diffHtml = `<div class="spell-compare-row">${diffBoxes.join('')}</div>`;

  // Render bottom overlay on camera
  const overlay = $('#live-letters-overlay');
  if (overlay) {
    overlay.innerHTML = (recognized ? recognized.split('') : []).map((ch, i) => {
      const match = (expected[i] || '').toLowerCase() === ch.toLowerCase();
      return `<div class="live-letter ${match ? 'correct' : 'wrong'}">${ch}</div>`;
    }).join('');
  }

  showReviewFeedback({
    correct,
    expected: currentReviewWord.word,
    written: recognized,
    mode: 'spell',
    diffHtml,
  });
}

function evaluateCloze(recognized, segments = []) {
  if (!currentReviewWord) return;
  const blanks = clozeTemplate.filter(x => x.hidden);

  // Extract written characters from draft or segment predictions
  let writtenChars = (recognized || '').split('').filter(c => c.trim().length > 0);
  if (writtenChars.length === 0 && segments.length > 0) {
    writtenChars = segments.map(s => (s.prediction || '').toLowerCase()).filter(Boolean);
  }

  // Map to blank positions
  clozeFilled = blanks.map((_, i) => writtenChars[i] || '');

  // Render blanks with evaluated colors (green for correct, red for wrong)
  renderClozeDisplay(true);

  // STRICT check: EVERY single blank must be non-empty and match expected character
  const hasEmpty = clozeFilled.length < blanks.length || clozeFilled.some(c => !c);
  const allCorrect = !hasEmpty && blanks.every((b, idx) => (clozeFilled[idx] || '').toLowerCase() === b.ch.toLowerCase());

  showReviewFeedback({
    correct: allCorrect,
    expected: currentReviewWord.word,
    written: clozeFilled.filter(Boolean).join(''),
    mode: 'cloze',
  });
}

function showReviewFeedback({ correct, expected, written, mode, diffHtml = '' }) {
  const feedback = $('#review-feedback');
  if (!feedback) return;
  feedback.classList.remove('hidden');

  feedback.innerHTML = `
    <div class="result-banner ${correct ? 'correct' : 'wrong'}">
      <span class="icon">${correct ? '🎉' : '❌'}</span>
      <div style="flex:1;">
        <div class="text">${correct ? 'Chính xác!' : 'Chưa đúng'}</div>
        ${!correct
          ? `<div class="detail">Bạn ${mode === 'cloze' ? 'điền' : 'viết'}: <strong style="color:var(--rose)">${written || '(chưa có)'}</strong> — Từ đúng là: <strong style="color:var(--emerald)">${expected}</strong></div>`
          : `<div class="detail">Từ chính xác: <strong style="color:var(--emerald)">${expected}</strong></div>`
        }
        ${diffHtml ? `<div style="margin-top:10px;">${diffHtml}</div>` : ''}
      </div>
    </div>`;
  $('#review-next')?.classList.remove('hidden');
}

$('#review-next')?.addEventListener('click', () => {
  isReviewEvaluating = false;
  reviewEvaluated = false;
  lastReviewGesture = null;
  // Move to next word in collection or review list, or re-run
  let nextWord = null;
  if (currentCollection?.vocabulary?.length > 0 && currentReviewWord) {
    const idx = currentCollection.vocabulary.findIndex(w => w.normalized_word === currentReviewWord.normalized_word);
    if (idx !== -1) {
      nextWord = currentCollection.vocabulary[(idx + 1) % currentCollection.vocabulary.length];
    }
  }
  if (!nextWord && reviewWordList?.length > 0 && currentReviewWord) {
    const idx = reviewWordList.findIndex(w => w.normalized_word === currentReviewWord.normalized_word);
    if (idx !== -1) {
      nextWord = reviewWordList[(idx + 1) % reviewWordList.length];
    }
  }
  if (nextWord) {
    openReviewSession(nextWord);
  } else if (currentReviewWord) {
    startReviewSession(currentReviewWord);
  }
});

$('#review-back')?.addEventListener('click', async () => {
  stopReviewCamera();
  currentReviewWord = null;
  spellLetters = [];
  clozeFilled = [];
  $('#review-session')?.classList.add('hidden');
  $('#review-word-list')?.classList.remove('hidden');
  $('#review-next')?.classList.add('hidden');
  loadReview();
});

/* ════════════════════════════════════ COLLECTIONS MODULE ══════════════════════════════════════ */

let currentCollection = null; // the opened collection object
let selectedCollectionWord = null; // word selected in the topic detail view

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
               aria-label="Mở chủ đề ${set.name}">
        <span class="eyebrow">${set.category}</span>
        <h2>${set.name}</h2>
        <p>${set.description}</p>
        <div class="card-footer">
          <span class="mode-badge">${set.total_words ?? (set.vocabulary?.length ?? 0)} từ</span>
          <span style="font-size:12px; color:var(--text-muted); margin-left:auto;">Nhấn để xem →</span>
        </div>
      </article>`).join('')
      : `<div class="empty-state" style="grid-column:1/-1;">
         <span class="icon">📚</span>
         <p>Chưa có bộ từ nào. Thêm collection vào backend.</p>
       </div>`;

    $$('.collection-card').forEach(card => {
      card.addEventListener('click', () => openCollection(card.dataset.id));
      card.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') openCollection(card.dataset.id); });
    });
  } catch (err) {
    container.innerHTML = `<p class="status-text error">${err.message}</p>`;
  }
}

async function startPracticeFromCollection(word, mode) {
  reviewMode = mode;
  $$('.mode-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === reviewMode));
  updateReviewModeUI();

  if (reviewSession) {
    try {
      await api(`/recognition/sessions/${reviewSession.session_id}/clear_all`, { method: 'POST' });
    } catch { }
  }

  location.hash = '#review';
  setTimeout(() => {
    openReviewSession(word);
  }, 120);
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

    // Populate vocabulary list with Spell and Cloze buttons on each word
    const vocabEl = $('#detail-vocab-list');
    if (vocabEl) {
      vocabEl.innerHTML = (set.vocabulary || []).map((word, idx) => `
        <div class="vocab-card collection-vocab-item" data-word="${word.normalized_word}" tabindex="0" role="region" aria-label="Từ ${word.word}">
          <div class="vocab-card-header">
            <div class="vocab-card-word">${idx + 1}. ${word.word}</div>
            ${word.part_of_speech ? `<span class="vocab-card-pos">${word.part_of_speech}</span>` : ''}
            <div class="vocab-card-vi">${word.vietnamese_meaning}</div>
          </div>
          <div class="vocab-card-def">${word.definition || '<em style="color:var(--text-muted)">Chưa có định nghĩa.</em>'}</div>
          <div class="vocab-card-actions">
            <button class="vocab-btn-spell" data-word="${word.normalized_word}" data-mode="spell" title="Ôn tập chế độ Spell (viết toàn bộ từ)">✍️ Spell</button>
            <button class="vocab-btn-cloze" data-word="${word.normalized_word}" data-mode="cloze" title="Ôn tập chế độ Cloze (điền chữ cái đục lỗ)">🧩 Cloze</button>
          </div>
        </div>`).join('');

      $$('.vocab-btn-spell').forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          const word = set.vocabulary.find(w => w.normalized_word === btn.dataset.word);
          if (word) startPracticeFromCollection(word, 'spell');
        });
      });

      $$('.vocab-btn-cloze').forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          const word = set.vocabulary.find(w => w.normalized_word === btn.dataset.word);
          if (word) startPracticeFromCollection(word, 'cloze');
        });
      });

      $$('.collection-vocab-item').forEach(card => {
        card.addEventListener('click', (e) => {
          if (e.target.closest('button')) return;
          const word = set.vocabulary.find(w => w.normalized_word === card.dataset.word);
          if (word) startPracticeFromCollection(word, reviewMode || 'spell');
        });
      });
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

// Learn buttons in detail view header
$$('.collection-detail-learn-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    if (!currentCollection || !currentCollection.vocabulary?.length) return;
    const mode = btn.dataset.mode === 'cloze' ? 'cloze' : 'spell';
    const targetWord = currentCollection.vocabulary[0];
    startPracticeFromCollection(targetWord, mode);
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
      } catch { }
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

// Stop camera (translate page)
$('#stop-camera')?.addEventListener('click', () => {
  stopTranslateCamera();
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
    } catch { }
    // fallback: also try standard clear twice
    try {
      await api(`/recognition/sessions/${translateSession.session_id}/clear`, { method: 'POST' });
    } catch { }
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

// Manual editing of draft
$('#draft')?.addEventListener('input', async () => {
  const newDraft = ($('#draft').textContent || '').trim();
  if (!translateSession) return;
  translateSession.draft = newDraft;
  $('#commit')?.classList.toggle('hidden', !newDraft);
  try {
    await api(`/recognition/sessions/${translateSession.session_id}`, {
      method: 'PATCH',
      body: JSON.stringify({ draft: newDraft }),
    });
  } catch { }
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

let currentActiveScreen = null;

function onHashChange() {
  const newScreen = location.hash.slice(1) || 'translate';

  // Tự động tắt camera của trang trước đó khi người dùng chuyển sang trang khác
  if (currentActiveScreen && currentActiveScreen !== newScreen) {
    if (currentActiveScreen === 'translate') {
      stopTranslateCamera();
    } else if (currentActiveScreen === 'review') {
      stopReviewCamera();
    }
  }

  currentActiveScreen = newScreen;

  $$('.screen').forEach(el => el.classList.toggle('active', el.id === newScreen));
  $$('nav a').forEach(a => a.classList.toggle('active', a.getAttribute('href') === `#${newScreen}`));

  if (newScreen === 'review') loadReview();
  if (newScreen === 'learn') loadCollections();
  if (newScreen === 'progress') loadProgress();
}

window.addEventListener('hashchange', onHashChange);
window.dispatchEvent(new HashChangeEvent('hashchange'));
