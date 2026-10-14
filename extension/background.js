const TARGET = "https://coindcx.com/trade/BTCINR";
const API = "http://127.0.0.1:8765/api/v1/prediction";
const POLL_MS = 5000;
const lastByTab = new Map();

async function pollTab(tab) {
  if (!tab?.id || !tab.url || !tab.url.startsWith(TARGET)) return;
  try {
    const response = await fetch(API, { cache: "no-store" });
    if (!response.ok) throw new Error("Prediction service unavailable");
    const payload = await response.json();
    lastByTab.set(tab.id, payload);
    chrome.tabs.sendMessage(tab.id, { type: "BTCINR_FORECAST", payload }).catch(() => {});
  } catch (error) {
    chrome.tabs.sendMessage(tab.id, {
      type: "BTCINR_FORECAST",
      payload: { status: "UNAVAILABLE", data_status: "STALE", reason: String(error) }
    }).catch(() => {});
  }
}

async function pollActiveTab() {
  const tabs = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  await pollTab(tabs[0]);
}

chrome.runtime.onInstalled.addListener(() => {
  chrome.alarms.create("btc-inr-overlay-poll", { periodInMinutes: 0.5 });
});
chrome.runtime.onStartup.addListener(() => {
  chrome.alarms.create("btc-inr-overlay-poll", { periodInMinutes: 0.5 });
});
chrome.alarms.onAlarm.addListener(alarm => {
  if (alarm.name === "btc-inr-overlay-poll") pollActiveTab();
});
chrome.tabs.onActivated.addListener(pollActiveTab);
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete") pollTab(tab);
});
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === "BTCINR_GET_LAST") {
    sendResponse(lastByTab.get(sender.tab?.id) || { status: "UNAVAILABLE" });
  }
});
