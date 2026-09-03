const API_BASE = 'http://127.0.0.1:8000';

// ── Context menu setup ────────────────────────────────────────────
chrome.runtime.onInstalled.addListener(() => {
  // Right-click on a link
  chrome.contextMenus.create({
    id: 'scan-link',
    title: '🛡️ CyberShield — Scan this link',
    contexts: ['link'],
  });

  // Right-click on selected text (might be a URL)
  chrome.contextMenus.create({
    id: 'scan-selection',
    title: '🛡️ CyberShield — Scan selected URL',
    contexts: ['selection'],
  });

  // Right-click on the page itself
  chrome.contextMenus.create({
    id: 'scan-page',
    title: '🛡️ CyberShield — Scan this page',
    contexts: ['page'],
  });
});

// ── Context menu click handler ────────────────────────────────────
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  let url = '';

  if (info.menuItemId === 'scan-link' && info.linkUrl) {
    url = info.linkUrl;
  } else if (info.menuItemId === 'scan-selection' && info.selectionText) {
    url = info.selectionText.trim();
  } else if (info.menuItemId === 'scan-page' && tab?.url) {
    url = tab.url;
  }

  if (!url) return;

  // Normalize
  if (!/^https?:\/\//i.test(url)) {
    url = 'http://' + url;
  }

  // Perform the scan in background
  const result = await scanInBackground(url);

  // Show notification-style badge
  if (result.isThreat) {
    chrome.action.setBadgeText({ text: '!' });
    chrome.action.setBadgeBackgroundColor({ color: '#EF4444' });
    // Auto-clear badge after 5 seconds
    setTimeout(() => chrome.action.setBadgeText({ text: '' }), 5000);
  } else {
    chrome.action.setBadgeText({ text: '✓' });
    chrome.action.setBadgeBackgroundColor({ color: '#22C55E' });
    setTimeout(() => chrome.action.setBadgeText({ text: '' }), 3000);
  }

  // Store URL for popup to pick up
  chrome.storage.local.set({ contextScanUrl: url });
});

// ── Background scan function ──────────────────────────────────────
async function scanInBackground(url) {
  try {
    const res = await fetch(`${API_BASE}/api/scan-url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
      signal: AbortSignal.timeout(8000),
    });

    if (!res.ok) return { isThreat: false, error: `HTTP ${res.status}` };

    const data = await res.json();
    const isThreat = data.status === 'Threat Detected';

    // Save to history
    const stored = await chrome.storage.local.get('scanHistory');
    const history = stored.scanHistory || [];
    const filtered = history.filter(h => h.url !== url);
    filtered.unshift({ url, isThreat, time: Date.now() });
    chrome.storage.local.set({ scanHistory: filtered.slice(0, 10) });

    return { isThreat, error: null };
  } catch (err) {
    return { isThreat: false, error: err.message };
  }
}
