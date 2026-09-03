const API_BASE = 'http://127.0.0.1:8000';
const MAX_HISTORY = 10;

// DOM elements
const urlInput = document.getElementById('url-input');
const scanBtn = document.getElementById('scan-btn');
const scanPageBtn = document.getElementById('scan-page-btn');
const resultDiv = document.getElementById('result');
const resultIcon = document.getElementById('result-icon');
const resultStatus = document.getElementById('result-status');
const resultUrl = document.getElementById('result-url');
const historyList = document.getElementById('history-list');
const clearHistoryBtn = document.getElementById('clear-history');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');

// ── Backend health check ──────────────────────────────────────────
async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      statusDot.className = 'status online';
      statusText.textContent = 'Online';
      return true;
    }
  } catch { /* offline */ }
  statusDot.className = 'status offline';
  statusText.textContent = 'Offline';
  return false;
}

// ── Scan URL ──────────────────────────────────────────────────────
async function scanUrl(url) {
  if (!url || !url.trim()) return;

  // Normalize URL
  url = url.trim();
  if (!/^https?:\/\//i.test(url)) {
    url = 'http://' + url;
  }

  // Show loading
  setLoading(true);
  hideResult();

  try {
    const res = await fetch(`${API_BASE}/api/scan-url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
      signal: AbortSignal.timeout(8000),
    });

    if (!res.ok) throw new Error(`Server error: ${res.status}`);

    const data = await res.json();
    const isThreat = data.status === 'Threat Detected';

    showResult(isThreat, url);
    addToHistory(url, isThreat);
  } catch (err) {
    showError(url, err.message);
  } finally {
    setLoading(false);
  }
}

// ── UI helpers ────────────────────────────────────────────────────
function setLoading(loading) {
  scanBtn.disabled = loading;
  scanPageBtn.disabled = loading;
  if (loading) {
    scanBtn.innerHTML = '<div class="spinner"></div>';
  } else {
    scanBtn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
      <circle cx="11" cy="11" r="8"/>
      <path d="M21 21l-4.35-4.35"/>
    </svg>`;
  }
}

function showResult(isThreat, url) {
  resultDiv.hidden = false;
  resultDiv.className = `result ${isThreat ? 'threat' : 'safe'}`;
  resultIcon.textContent = isThreat ? '🚨' : '✅';
  resultStatus.textContent = isThreat ? 'Threat Detected' : 'Safe — No threats found';
  resultUrl.textContent = url;
}

function showError(url, message) {
  resultDiv.hidden = false;
  resultDiv.className = 'result error';
  resultIcon.textContent = '⚠️';
  resultStatus.textContent = 'Scan Failed';
  resultUrl.textContent = message.includes('Failed to fetch') || message.includes('timeout')
    ? 'Backend is offline. Start CyberShield AI with: python main.py'
    : message;
}

function hideResult() {
  resultDiv.hidden = true;
}

// ── History ───────────────────────────────────────────────────────
async function getHistory() {
  return new Promise(resolve => {
    chrome.storage.local.get('scanHistory', data => {
      resolve(data.scanHistory || []);
    });
  });
}

async function addToHistory(url, isThreat) {
  const history = await getHistory();
  // Remove duplicate
  const filtered = history.filter(h => h.url !== url);
  // Add to front
  filtered.unshift({ url, isThreat, time: Date.now() });
  // Trim
  const trimmed = filtered.slice(0, MAX_HISTORY);
  chrome.storage.local.set({ scanHistory: trimmed });
  renderHistory(trimmed);
}

async function renderHistory(history) {
  if (!history) history = await getHistory();

  if (history.length === 0) {
    historyList.innerHTML = '<div class="history-empty">No scans yet</div>';
    return;
  }

  historyList.innerHTML = history.map(h => {
    const icon = h.isThreat ? '🚨' : '✅';
    const tagClass = h.isThreat ? 'threat-tag' : 'safe-tag';
    const tagText = h.isThreat ? 'THREAT' : 'SAFE';
    // Truncate URL for display
    let displayUrl = h.url;
    try { displayUrl = new URL(h.url).hostname; } catch { /* keep full */ }
    return `
      <div class="history-item" data-url="${escapeHtml(h.url)}">
        <span class="h-icon">${icon}</span>
        <span class="h-url" title="${escapeHtml(h.url)}">${escapeHtml(displayUrl)}</span>
        <span class="h-status ${tagClass}">${tagText}</span>
      </div>
    `;
  }).join('');

  // Click to re-scan
  historyList.querySelectorAll('.history-item').forEach(item => {
    item.addEventListener('click', () => {
      const url = item.dataset.url;
      urlInput.value = url;
      scanUrl(url);
    });
  });
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ── Event listeners ───────────────────────────────────────────────
scanBtn.addEventListener('click', () => {
  scanUrl(urlInput.value);
});

urlInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') scanUrl(urlInput.value);
});

scanPageBtn.addEventListener('click', async () => {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab?.url) {
      urlInput.value = tab.url;
      scanUrl(tab.url);
    }
  } catch {
    // Could not get current tab
  }
});

clearHistoryBtn.addEventListener('click', () => {
  chrome.storage.local.set({ scanHistory: [] });
  renderHistory([]);
});

// ── Init ──────────────────────────────────────────────────────────
(async function init() {
  await checkBackendHealth();
  await renderHistory();

  // Check if we received a URL from the context menu
  chrome.storage.local.get('contextScanUrl', data => {
    if (data.contextScanUrl) {
      urlInput.value = data.contextScanUrl;
      scanUrl(data.contextScanUrl);
      chrome.storage.local.remove('contextScanUrl');
    }
  });
})();
