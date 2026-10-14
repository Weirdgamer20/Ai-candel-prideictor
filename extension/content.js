(() => {
  "use strict";
  const OVERLAY_ID = "btc-inr-future-overlay";
  let latestPayload = null;
  let observer = null;
  let renderTimer = null;

  function findChartRegion() {
    const candidates = [...document.querySelectorAll("canvas")]
      .map(el => ({ el, rect: el.getBoundingClientRect() }))
      .filter(x => x.rect.width >= 400 && x.rect.height >= 180 && x.rect.bottom > 0 && x.rect.top < innerHeight)
      .sort((a, b) => (b.rect.width * b.rect.height) - (a.rect.width * a.rect.height));
    if (!candidates.length) return null;
    const canvas = candidates[0].el;
    // Use the nearest stable chart container when possible; never cover the whole page.
    let node = canvas;
    for (let i = 0; i < 5 && node.parentElement; i++) {
      const r = node.getBoundingClientRect();
      if (r.width >= 400 && r.height >= 180 && r.width < innerWidth * 0.95 && r.height < innerHeight * 0.95) {
        if (node !== canvas && node.querySelectorAll("canvas").length >= 1) return node;
      }
      node = node.parentElement;
    }
    return canvas;
  }

  function getOverlay(region) {
    let overlay = document.getElementById(OVERLAY_ID);
    if (!overlay) {
      overlay = document.createElement("div");
      overlay.id = OVERLAY_ID;
      overlay.setAttribute("aria-hidden", "true");
      const canvas = document.createElement("canvas");
      overlay.appendChild(canvas);
      document.documentElement.appendChild(overlay);
    }
    const rect = region.getBoundingClientRect();
    Object.assign(overlay.style, {
      left: (rect.left + scrollX) + "px",
      top: (rect.top + scrollY) + "px",
      width: rect.width + "px",
      height: rect.height + "px",
      display: "block"
    });
    const canvas = overlay.firstElementChild;
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    return { overlay, canvas, width: rect.width, height: rect.height, dpr };
  }

  function draw(payload) {
    const region = findChartRegion();
    const existing = document.getElementById(OVERLAY_ID);
    if (!region || payload?.status !== "RESEARCH_FORECAST" || payload?.forecast_status !== "READY" ||
        payload?.data_status !== "FRESH" || !Array.isArray(payload?.forecast?.candles)) {
      if (existing) existing.remove();
      return;
    }

    const { overlay, canvas, width, height, dpr } = getOverlay(region);
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    // Fail closed until a reliable chart price-scale/time-scale mapper is available.
    // Drawing at guessed coordinates would create false precision and mislead the user.
    const candles = payload.forecast.candles;
    if (!candles.length || !candles.every(c => [c.open, c.high, c.low, c.close].every(v => Number.isFinite(v) && v > 0))) {
      overlay.remove();
      return;
    }

    // Initial renderer deliberately draws only when a price scale can be identified
    // from visible DOM labels. Unknown chart geometry means no forecast is painted.
    const labels = [...region.querySelectorAll("*")]
      .filter(el => el.children.length === 0)
      .map(el => (el.textContent || "").trim())
      .filter(t => /\d[\d,]*\.\d{1,2}/.test(t))
      .map(t => Number(t.replace(/,/g, "")))
      .filter(v => Number.isFinite(v) && v > 10000);
    if (labels.length < 2) {
      overlay.remove();
      return;
    }

    // Price-label positions are only valid when DOM labels share the chart container.
    // Until their coordinates can be associated unambiguously with the price axis,
    // suppress the forecast rather than draw candles on an unverified scale.
    overlay.remove();
  }

  function scheduleDraw() {
    clearTimeout(renderTimer);
    renderTimer = setTimeout(() => draw(latestPayload), 150);
  }

  chrome.runtime.onMessage.addListener(message => {
    if (message?.type === "BTCINR_FORECAST") {
      latestPayload = message.payload;
      scheduleDraw();
    }
  });
  window.addEventListener("resize", scheduleDraw, { passive: true });
  window.addEventListener("scroll", scheduleDraw, { passive: true });
  observer = new MutationObserver(scheduleDraw);
  observer.observe(document.documentElement, { childList: true, subtree: true });
  chrome.runtime.sendMessage({ type: "BTCINR_GET_LAST" }, payload => {
    if (payload) { latestPayload = payload; scheduleDraw(); }
  });
})();
