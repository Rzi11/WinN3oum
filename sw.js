/* Win N3oum : garde l'app sur le téléphone pour qu'elle s'ouvre tout de suite.
   La météo (Open-Meteo) et les retours (Supabase) ne passent JAMAIS par ce cache : toujours en direct. */
const V = "wn-v3";
const SHELL = ["./", "index.html", "plages.json", "manifest.webmanifest",
  "fonts/baloo-2-latin-600-normal.woff2", "fonts/baloo-2-latin-800-normal.woff2",
  "fonts/atkinson-hyperlegible-latin-400-normal.woff2", "fonts/atkinson-hyperlegible-latin-700-normal.woff2",
  "icons/icon-192.png", "icons/apple-touch-icon.png"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(V).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== V).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});

const sig = r => r && (r.headers.get("etag") || r.headers.get("last-modified") || "");

self.addEventListener("fetch", e => {
  const req = e.request, url = new URL(req.url);
  if (req.method !== "GET" || url.origin !== location.origin) return;        // météo, Supabase : en direct
  const isPage = req.mode === "navigate" || url.pathname.endsWith("/index.html") || url.pathname.endsWith("/");
  const isData = url.pathname.includes("/data/") || url.pathname.endsWith("plages.json");

  if (isPage) {
    // L'app s'ouvre tout de suite depuis le téléphone, et la nouvelle version se télécharge en douce.
    e.respondWith((async () => {
      const c = await caches.open(V), key = "index.html";
      const old = await c.match(key);
      const fresh = fetch(req.url, { cache: "no-cache", credentials: "same-origin" }).then(async r => {
        if (r.ok) {
          const changed = old && sig(r) && sig(r) !== sig(old);
          await c.put(key, r.clone());
          if (changed) (await self.clients.matchAll()).forEach(cl => cl.postMessage({ type: "update" }));
        }
        return r;
      }).catch(() => null);
      if (old) { e.waitUntil(fresh); return old; }
      return (await fresh) || new Response("Hors connexion", { status: 503 });
    })());
    return;
  }
  if (isData) {
    // Les prévisions du modèle : toujours les plus récentes, la dernière copie si pas de réseau.
    e.respondWith(fetch(req.url, { cache: "no-cache", credentials: "same-origin" }).then(r => { if (r.ok) { const cp = r.clone(); caches.open(V).then(c => c.put(req, cp)); } return r; })
      .catch(() => caches.match(req).then(r => r || Response.error())));
    return;
  }
  // Polices, icônes : gardées une fois pour toutes.
  e.respondWith(caches.match(req).then(r => r || fetch(req).then(res => { if (res.ok) { const cp = res.clone(); caches.open(V).then(c => c.put(req, cp)); } return res; })));
});
