const dropZone    = document.getElementById('dropZone');
const dropInner   = document.getElementById('dropInner');
const dropPreview = document.getElementById('dropPreview');
const previewImg  = document.getElementById('previewImg');
const fileInput   = document.getElementById('fileInput');
const clearBtn    = document.getElementById('clearBtn');
const predictBtn  = document.getElementById('predictBtn');
const btnText     = document.getElementById('btnText');

const resultIdle    = document.getElementById('resultIdle');
const resultLoading = document.getElementById('resultLoading');
const resultContent = document.getElementById('resultContent');
const resultError   = document.getElementById('resultError');
const resultDigit   = document.getElementById('resultDigit');
const confidenceVal = document.getElementById('confidenceVal');
const confidenceBar = document.getElementById('confidenceBar');
const probGrid      = document.getElementById('probGrid');
const errorMsg      = document.getElementById('errorMsg');

let selectedFile = null;

/* ── file select ── */
fileInput.addEventListener('change', e => {
    if (e.target.files[0]) loadFile(e.target.files[0]);
});

/* ── drag and drop ── */
dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
});

/* ── click zone to open file dialog ── */
dropZone.addEventListener('click', e => {
    if (e.target === clearBtn || e.target.closest('.clear-btn')) return;
    if (selectedFile) return;
    fileInput.click();
});

/* ── clear ── */
clearBtn.addEventListener('click', e => {
    e.stopPropagation();
    resetUpload();
    showState('idle');
});

/* ── predict ── */
predictBtn.addEventListener('click', () => {
    if (!selectedFile) return;
    predict(selectedFile);
});

function loadFile(file) {
    selectedFile = file;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    dropInner.classList.add('hidden');
    dropPreview.classList.remove('hidden');
    predictBtn.disabled = false;
    showState('idle');
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    previewImg.src = '';
    dropInner.classList.remove('hidden');
    dropPreview.classList.add('hidden');
    predictBtn.disabled = true;
}

async function predict(file) {
    showState('loading');
    btnText.textContent = 'Classifying…';
    predictBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ error: 'Server error' }));
            throw new Error(err.error || 'Server error');
        }

        const data = await res.json();
        // data = { predicted_digit: 7, confidence: 0.983, probabilities: [0.001, ...] }
        showResult(data);

    } catch (err) {
        showState('error');
        errorMsg.textContent = err.message || 'Something went wrong.';
    } finally {
        btnText.textContent = 'Classify digit';
        predictBtn.disabled = false;
    }
}

function showResult(data) {
    const digit = data.predicted_digit;
    const conf  = data.confidence;
    const probs = data.probabilities;

    resultDigit.textContent = digit;
    confidenceVal.textContent = (conf * 100).toFixed(1) + '%';

    // animate bar after paint
    requestAnimationFrame(() => {
        setTimeout(() => {
            confidenceBar.style.width = (conf * 100) + '%';
        }, 50);
    });

    // build prob grid
    probGrid.innerHTML = '';
    probs.forEach((p, i) => {
        const cell = document.createElement('div');
        cell.className = 'prob-cell' + (i === digit ? ' is-top' : '');
        cell.innerHTML = `
            <span class="prob-cell-digit">${i}</span>
            <span class="prob-cell-val">${(p * 100).toFixed(1)}%</span>
        `;
        probGrid.appendChild(cell);
    });

    showState('result');
}

function showState(state) {
    resultIdle.classList.add('hidden');
    resultLoading.classList.add('hidden');
    resultContent.classList.add('hidden');
    resultError.classList.add('hidden');

    if (state === 'idle')    resultIdle.classList.remove('hidden');
    if (state === 'loading') resultLoading.classList.remove('hidden');
    if (state === 'result')  resultContent.classList.remove('hidden');
    if (state === 'error')   resultError.classList.remove('hidden');
}