"""
Win N3oum ? — modèle algues v1.0 (lancé chaque nuit par GitHub Actions, après les méduses).

  python drift/algues.py           prévision normale (7 jours, heure par heure) + apprentissage sur les signalements
  python drift/algues.py --test    test hors ligne avec des vagues, du vent et des signalements inventés

Ce qu'on prévoit : les feuilles mortes d'herbiers (posidonie, cymodocée) qui arrivent sur les plages.
Pas les « algues » microscopiques (eau verte, mousse) : trop compliqué, on laisse ça aux signalements.

Deux choses différentes :
  1. DANS L'EAU à l'entrée : des feuilles qui flottent près du bord. Ça vient avec le vent et les vagues
     vers la plage des dernières heures, et ça part vite (demi-vie ~18 h) dès que le vent tourne.
  2. SUR LE SABLE : un STOCK qui se remplit à chaque coup de mer vers la plage (vagues assez fortes),
     et qui ne se vide que lentement (demi-vie ~10 jours), sauf si la plage est nettoyée (hôtels).

  arrivée  = saison (les herbiers perdent leurs feuilles en automne) × exposition de la plage × herbier devant
           × (vagues qui arrivent de face)² + (vent fort vers la plage)²
  stock    = stock d'hier × oubli + arrivée
  probabilité affichée = courbe logistique du stock (réglée avec les signalements)

Les signalements servent deux fois :
  - CORRECTION : un retour crédible « rien sur le sable » vide le stock à partir de ce moment ;
    « algues sur le sable » le remplit. La prévision des jours suivants part de là.
  - APPRENTISSAGE : chaque retour devient un exemple (data/algues_exemples.csv). Quand il y en a assez,
    on règle la courbe, la vitesse d'oubli et un écart propre à chaque plage (une plage « plus propre
    que prévu » est sans doute nettoyée). Les réglages ne sont adoptés que s'ils prévoient mieux.
"""
import os, sys, csv, json, math, datetime as dt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VERSION = "algues-v1.0"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
P_OUT = os.path.join(DATA, "algues.json")
P_EX = os.path.join(DATA, "algues_exemples.csv")
P_SET = os.path.join(DATA, "algues_reglages.json")
TZ = "Africa/Tunis"
UTC_OFFSET = 1                       # Tunisie : UTC+1 toute l'année

PAST_DAYS = 40                       # passé utilisé pour remplir le stock (≥ 3 × la demi-vie la plus longue testée)
FORECAST_DAYS = 7

# ------------------------------------------------------------------ réglages de départ (À RÉGLER par l'apprentissage)
# Feuilles disponibles selon le mois (chute des feuilles à l'automne, tempêtes d'hiver). Indicatif.
SUPPLY = {1: .6, 2: .55, 3: .5, 4: .45, 5: .4, 6: .35, 7: .35, 8: .4, 9: .7, 10: 1.0, 11: 1.0, 12: .8}
H0 = 0.4                             # m : en dessous, les vagues ne déposent presque rien sur le sable
WIND_REF = 20.0                      # km/h : vent vers la plage « qui compte »
SHELTER_F = 0.7                      # plage abritée : moins d'arrivées
SHALLOW_F = 1.3                      # plage très peu profonde : herbiers tout près
CLEAN_MONTHS = range(5, 11)          # mai → octobre : saison où les hôtels ratissent
CLEAN_HALF_LIFE_H = 30               # plage nettoyée : le stock part en ~1 jour
TAUS_SAND_DAYS = (3, 6, 10, 20)      # demi-vies testées pour le stock sur le sable
TAUS_WATER_H = (8, 18, 36)           # demi-vies testées pour les feuilles dans l'eau
DEFAULT = dict(tau_s=10, tau_w=18,
               sand=dict(a=-2.2, b=2.0, ref=4.0, off={}),
               water=dict(a=-2.6, b=2.2, ref=1.5, off={}))
ASSIM_DAYS = 7                       # signalements des 7 derniers jours utilisés pour corriger le stock
CALIB_MIN = 30                       # exemples minimum avant d'apprendre
CALIB_MIN_CLASS = 6                  # dont au moins 6 « oui » et 6 « non »
CALIB_MIN_GAIN = 0.01                # il faut battre les réglages actuels (log-loss) d'au moins ça
OFF_SHRINK = 5.0                     # prudence sur les écarts par plage (plus c'est grand, plus on reste près de 0)
LEVEL_YES, LEVEL_MAYBE = 0.55, 0.35


# ====================================================================== outils
def jload(p, default):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def jsave(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def km_between(lat1, lon1, lat2, lon2):
    dy = (lat2 - lat1) * 111.32
    dx = (lon2 - lon1) * 111.32 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dx, dy)


def load_beaches():
    with open(os.path.join(ROOT, "plages.json"), encoding="utf-8") as f:
        raw = json.load(f)["plages"]
    out = []
    for b in raw:
        if b.get("region", "djerba") != "djerba":
            continue
        e = 1.0
        if int(b.get("shelter", 0)):
            e *= SHELTER_F
        herb = b.get("herbier")                       # facultatif : 0 = sable nu devant, 1 = normal, 2 = beaucoup d'herbiers
        if herb is not None:
            e *= (0.5, 1.0, 1.4)[int(herb)]
        elif int(b.get("shallow", 0)):
            e *= SHALLOW_F
        out.append(dict(id=b["id"], name=b["name"], lat=float(b["lat"]), lon=float(b["lon"]), face=float(b["face"]),
                        expo=e, cleaned=int(b.get("nettoyee", 0))))
    return out


# ====================================================================== données
def fetch_series(beaches, past_days=PAST_DAYS, forecast_days=FORECAST_DAYS):
    """Vagues (hauteur, direction d'où elles viennent) et vent (vitesse km/h, direction d'où il vient), heure par heure,
    heure de Tunisie, pour chaque plage. Open-Meteo, comme l'app."""
    from sources import http_get
    lat = ",".join(f"{b['lat']:.4f}" for b in beaches)
    lon = ",".join(f"{b['lon']:.4f}" for b in beaches)
    common = dict(latitude=lat, longitude=lon, timezone=TZ, past_days=past_days, forecast_days=forecast_days)
    w = http_get("https://api.open-meteo.com/v1/forecast",
                 params=dict(common, hourly="wind_speed_10m,wind_direction_10m", wind_speed_unit="kmh")).json()
    status = {"vent": "ok"}
    try:
        m = http_get("https://marine-api.open-meteo.com/v1/marine", params=dict(common, hourly="wave_height,wave_direction")).json()
        status["vagues"] = "ok"
    except Exception as e:
        print(f"[algues] vagues indisponibles ({str(e)[:100]}) : vent seul")
        m, status["vagues"] = None, "indisponible"
    w = w if isinstance(w, list) else [w]
    m = (m if isinstance(m, list) else [m]) if m else [None] * len(beaches)
    times = w[0]["hourly"]["time"]
    series = []
    for k, b in enumerate(beaches):
        hw = w[k]["hourly"]
        s = dict(time=times, wind=np.array([x if x is not None else 0 for x in hw["wind_speed_10m"]], float),
                 wdir=np.array([x if x is not None else 0 for x in hw["wind_direction_10m"]], float))
        if m[k] is not None:
            hm = m[k]["hourly"]
            idx = {t: i for i, t in enumerate(hm["time"])}
            H = np.full(len(times), np.nan); D = np.full(len(times), np.nan)
            for i, t in enumerate(times):
                j = idx.get(t)
                if j is not None:
                    H[i] = hm["wave_height"][j] if hm["wave_height"][j] is not None else np.nan
                    D[i] = hm["wave_direction"][j] if hm["wave_direction"][j] is not None else np.nan
            s["wave"], s["vdir"] = H, D
        else:
            s["wave"] = np.full(len(times), np.nan); s["vdir"] = np.full(len(times), np.nan)
        series.append(s)
    # Plage collée à la côte sans donnée de vagues (case « terre » du modèle) : on prend la plage voisine la plus proche
    for k, s in enumerate(series):
        if np.isfinite(s["wave"]).mean() < 0.5:
            near = sorted((km_between(beaches[k]["lat"], beaches[k]["lon"], b["lat"], b["lon"]), j)
                          for j, b in enumerate(beaches) if j != k and np.isfinite(series[j]["wave"]).mean() >= 0.5)
            if near and near[0][0] < 25:
                j = near[0][1]
                s["wave"], s["vdir"], s["wave_from"] = series[j]["wave"].copy(), series[j]["vdir"].copy(), beaches[j]["id"]
                print(f"[algues] {beaches[k]['name']} : vagues reprises de {beaches[j]['name']}")
    return times, series, status


# ====================================================================== le modèle
def forcing(b, s, times):
    """Arrivées heure par heure : vers le sable (dépôt) et dans l'eau (feuilles qui flottent)."""
    face = math.radians(b["face"])
    month = np.array([int(t[5:7]) for t in times])
    sup = np.array([SUPPLY[mo] for mo in month]) * b["expo"]
    cv = np.cos(np.radians(s["wdir"]) - face)                 # vent qui vient de la mer, droit sur la plage = 1
    won = np.maximum(0, s["wind"] * cv) / WIND_REF
    woff = np.maximum(0, -s["wind"] * cv) / WIND_REF
    H = np.nan_to_num(s["wave"], nan=0.0)
    cw = np.nan_to_num(np.cos(np.radians(s["vdir"]) - face), nan=0.0)
    cw = np.where(np.isfinite(s["vdir"]), np.maximum(0, cw), np.maximum(0, cv))   # sans direction de vagues : celle du vent
    dep_sand = sup * (np.maximum(0, H - H0) ** 2 * cw + 0.35 * np.maximum(0, won - 1) ** 2)
    dep_water = sup * (0.6 * H ** 2 * cw + 0.25 * won ** 2)
    return dep_sand, dep_water, woff


def run_stock(dep, half_life_h, clean_mask=None, extra_loss=None, resets=None):
    """Stock qui se remplit avec « dep » et s'oublie avec la demi-vie donnée. resets = {heure: valeur} imposées par les signalements."""
    k = math.log(2) / half_life_h
    kc = math.log(2) / CLEAN_HALF_LIFE_H
    S = np.zeros(len(dep)); x = 0.0
    for i in range(len(dep)):
        kk = k + (kc if clean_mask is not None and clean_mask[i] else 0.0)
        if extra_loss is not None:
            kk += 0.08 * extra_loss[i] ** 2              # vent de terre : pousse les feuilles flottantes vers le large
        x = x * math.exp(-kk) + dep[i]
        if resets and i in resets:
            x = resets[i]
        S[i] = x
    return S


def indices(b, s, times, tau_s, tau_w, resets_s=None, resets_w=None):
    dep_s, dep_w, woff = forcing(b, s, times)
    clean = None
    if b["cleaned"]:
        clean = np.array([int(t[5:7]) in CLEAN_MONTHS and 6 <= int(t[11:13]) <= 9 for t in times])   # ratissage le matin
    S = run_stock(dep_s, tau_s * 24, clean_mask=clean, resets=resets_s)
    W = run_stock(dep_w, tau_w, extra_loss=woff, resets=resets_w)
    return S, W


def prob(x, ref, a, b, off=0.0):
    return sigmoid(a + b * np.log1p(np.asarray(x) / ref) + off)


# ====================================================================== signalements
def utc_to_key(t_utc):
    """'2026-10-13T08:12:00+00:00' → '2026-10-13T09:00' (heure de Tunisie, arrondie à l'heure)."""
    d = dt.datetime.strptime(t_utc[:16], "%Y-%m-%dT%H:%M") + dt.timedelta(hours=UTC_OFFSET)
    return d.strftime("%Y-%m-%dT%H:00")


def report_weight(rep, b):
    """Crédibilité d'un retour algues (même logique que les méduses, sans la partie espèce)."""
    import credibility as K
    w = K.trust(rep.get("dev") or {})
    if rep.get("user_lat") is not None:
        d = km_between(rep["user_lat"], rep["user_lon"], b["lat"], b["lon"])
        w *= 1.0 if d <= 1.5 else 0.7 if d <= 10 else 0.3
    else:
        w *= 0.6
    return w


def fetch_algae_reports(beaches, days=PAST_DAYS):
    import credibility as K
    if not K.enabled():
        print("[algues] pas de clé Supabase : pas de signalements")
        return []
    try:
        reps = K.fetch_reports(days=days)
    except Exception as e:
        print(f"[algues] signalements illisibles ({str(e)[:120]})")
        return []
    out = []
    byid = {b["id"]: b for b in beaches}
    for r in reps:
        if r.get("algae") is None:
            continue
        b = byid.get(r.get("beach_id"))
        if b is None and r.get("lat") is not None:
            d = sorted((km_between(r["lat"], r["lon"], x["lat"], x["lon"]), x["id"]) for x in beaches)
            b = byid[d[0][1]] if d and d[0][0] <= 5 else None
        if b is None:
            continue
        a = int(r["algae"])
        out.append(dict(id=str(r["id"]), beach=b["id"], key=utc_to_key(r.get("observed_at") or r["created_at"]),
                        sand=int(a in (1, 3)), water=int(a in (2, 3)), w=round(report_weight(r, b), 3)))
    print(f"[algues] {len(out)} retours algues sur {days} jours")
    return out


def resets_from_reports(reps, beach_id, times, settings, now_key):
    """Correction du stock par les retours récents et crédibles (≥ 0,4). Plusieurs retours la même heure : majorité pondérée."""
    pos = {t: i for i, t in enumerate(times)}
    lim = (dt.datetime.strptime(now_key, "%Y-%m-%dT%H:00") - dt.timedelta(days=ASSIM_DAYS)).strftime("%Y-%m-%dT%H:00")
    votes_s, votes_w = {}, {}
    for r in reps:
        if r["beach"] != beach_id or r["w"] < 0.4 or r["key"] < lim or r["key"] > now_key or r["key"] not in pos:
            continue
        i = pos[r["key"]]
        votes_s.setdefault(i, []).append((r["sand"], r["w"]))
        votes_w.setdefault(i, []).append((r["water"], r["w"]))

    def to_resets(votes, ref, a, b):
        out = {}
        for i, v in votes.items():
            yes = sum(w for y, w in v if y) / sum(w for _, w in v)
            # niveau de stock qui correspond à « on en voit » (p ≈ 0,75) ou « on n'en voit pas » (p ≈ 0,1)
            target = 0.75 if yes >= 0.5 else 0.10
            x = (math.log(target / (1 - target)) - a) / b
            out[i] = max(0.0, ref * (math.exp(x) - 1))
        return out
    s, w = settings["sand"], settings["water"]
    return to_resets(votes_s, s["ref"], s["a"], s["b"]), to_resets(votes_w, w["ref"], w["a"], w["b"])


# ====================================================================== apprentissage
EX_FIELDS = ["id", "key", "beach", "sand", "water", "w"] + [f"s{t}" for t in TAUS_SAND_DAYS] + [f"w{t}" for t in TAUS_WATER_H]


def update_examples(reps, beaches, times, series):
    """Ajoute les nouveaux retours comme exemples, avec l'état du modèle SANS correction (sinon il tricherait)."""
    have = {}
    if os.path.exists(P_EX):
        with open(P_EX, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                have[row["id"]] = row
    pos = {t: i for i, t in enumerate(times)}
    cache = {}
    new = 0
    for r in reps:
        if r["id"] in have or r["key"] not in pos:
            continue
        k = next(j for j, b in enumerate(beaches) if b["id"] == r["beach"])
        if k not in cache:
            cache[k] = {("s", t): indices(beaches[k], series[k], times, t, 18)[0] for t in TAUS_SAND_DAYS}
            cache[k].update({("w", t): indices(beaches[k], series[k], times, 10, t)[1] for t in TAUS_WATER_H})
        i = pos[r["key"]]
        row = dict(id=r["id"], key=r["key"], beach=r["beach"], sand=r["sand"], water=r["water"], w=r["w"])
        row.update({f"s{t}": round(float(cache[k][("s", t)][i]), 4) for t in TAUS_SAND_DAYS})
        row.update({f"w{t}": round(float(cache[k][("w", t)][i]), 4) for t in TAUS_WATER_H})
        have[r["id"]] = row
        new += 1
    rows = sorted(have.values(), key=lambda x: x["key"])
    os.makedirs(DATA, exist_ok=True)
    with open(P_EX, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=EX_FIELDS)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: row.get(k, "") for k in EX_FIELDS})
    print(f"[algues] {new} nouveaux exemples, {len(rows)} au total")
    return rows


def _logloss(y, p, w):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))


def _auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float((np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg)))


def _fit(x, y, w, beach, ids):
    """Logistique : a + b·x + écart par plage (pénalisé pour rester prudent). Descente de gradient simple, pas de dépendance."""
    nb = len(ids)
    a, b, off = 0.0, 1.0, np.zeros(nb)
    bi = np.array([ids.index(z) for z in beach])
    lr = 0.5
    for _ in range(3000):
        p = sigmoid(a + b * x + off[bi])
        g = (p - y) * w / w.sum()
        ga, gb = g.sum(), (g * x).sum()
        go = np.bincount(bi, weights=g, minlength=nb) + OFF_SHRINK * off / w.sum()
        a -= lr * ga; b -= lr * gb; off -= lr * go
    return a, max(b, 0.05), off


def learn(rows, settings, beaches):
    """Choisit la meilleure demi-vie et règle la courbe, avec validation croisée (5 morceaux, dans l'ordre du temps)."""
    report = {"n": len(rows)}
    if len(rows) < CALIB_MIN:
        report["note"] = f"{len(rows)} exemples : il en faut {CALIB_MIN} avant d'apprendre. Réglages de départ."
        return settings, report
    ids = [b["id"] for b in beaches]
    beach = np.array([r["beach"] for r in rows])
    keep = np.isin(beach, ids)
    rows = [r for r, k in zip(rows, keep) if k]; beach = beach[keep]
    w = np.array([float(r["w"]) for r in rows])
    new = json.loads(json.dumps(settings))
    for part, taus, tkey, ykey in (("sand", TAUS_SAND_DAYS, "tau_s", "sand"), ("water", TAUS_WATER_H, "tau_w", "water")):
        y = np.array([int(r[ykey]) for r in rows], float)
        if y.sum() < CALIB_MIN_CLASS or (1 - y).sum() < CALIB_MIN_CLASS:
            report[part] = dict(note="pas assez de « oui » et de « non » pour apprendre")
            continue
        cur = settings[part]
        xcur = np.log1p(np.array([float(r[f"{part[0]}{settings[tkey]}"]) for r in rows]) / cur["ref"])
        p_cur = sigmoid(cur["a"] + cur["b"] * xcur + np.array([cur["off"].get(z, 0.0) for z in beach]))
        # 5 morceaux par jours alternés (un jour entier reste dans le même morceau : pas de triche entre retours voisins)
        dnum = np.array([dt.date.fromisoformat(r["key"][:10]).toordinal() for r in rows])
        folds = [np.where(dnum % 5 == k)[0] for k in range(5)]
        folds = [f for f in folds if len(f)]
        best = None
        for tau in taus:
            raw = np.array([float(r[f"{part[0]}{tau}"]) for r in rows])
            ref = float(np.median(raw[raw > 0])) if (raw > 0).any() else 1.0
            x = np.log1p(raw / ref)
            p_cv = np.zeros(len(rows))
            for f in folds:
                tr = np.setdiff1d(np.arange(len(rows)), f)
                a, b, off = _fit(x[tr], y[tr], w[tr], beach[tr], ids)
                p_cv[f] = sigmoid(a + b * x[f] + off[[ids.index(z) for z in beach[f]]])
            ll = _logloss(y, p_cv, w)
            if best is None or ll < best[0]:
                best = (ll, tau, ref, x, p_cv)
        ll, tau, ref, x, p_cv = best
        ll_cur = _logloss(y, p_cur, w)
        a, b, off = _fit(x, y, w, beach, ids)
        info = dict(exemples=int(len(y)), oui=int(y.sum()), demi_vie=tau, logloss_actuel=round(ll_cur, 3),
                    logloss_appris=round(ll, 3), auc_actuel=_auc(y, p_cur), auc_appris=_auc(y, p_cv))
        if ll < ll_cur - CALIB_MIN_GAIN:
            new[tkey] = tau
            new[part] = dict(a=round(float(a), 3), b=round(float(b), 3), ref=round(ref, 4),
                             off={z: round(float(o), 3) for z, o in zip(ids, off) if abs(o) >= 0.05})
            info["adopte"] = True
        else:
            info["adopte"] = False
            info["note"] = "les réglages appris ne prévoient pas mieux : on garde les actuels"
        report[part] = info
    report["par_plage"] = {z: int((beach == z).sum()) for z in ids}
    return new, report


def explain_offsets(settings, beaches, rep):
    """Phrases simples sur les plages où les retours diffèrent nettement du modèle (souvent : nettoyage). 25 retours minimum."""
    names = {b["id"]: b["name"] for b in beaches}
    n = rep.get("par_plage", {})
    out = []
    for z, o in settings["sand"]["off"].items():
        if n.get(z, 0) < 25:
            continue
        if o <= -0.6:
            out.append(f"{names.get(z, z)} : moins d'algues sur le sable que prévu (plage sans doute nettoyée)")
        elif o >= 0.6:
            out.append(f"{names.get(z, z)} : plus d'algues sur le sable que prévu (herbiers tout près ?)")
    return out


# ====================================================================== sortie
def level(ps, pw):
    if pw >= LEVEL_YES:
        return 3                         # dans l'eau
    if ps >= LEVEL_YES:
        return 2                         # sur le sable
    if max(ps, pw) >= LEVEL_MAYBE:
        return 1                         # possibles
    return 0                             # sans doute aucune


def why(b, s, times, i0, i1):
    """Explication courte : le dernier coup de mer vers la plage dans les 5 jours avant."""
    face = math.radians(b["face"])
    lo = max(0, i0 - 120)
    H = np.nan_to_num(s["wave"][lo:i1], nan=0)
    cw = np.maximum(0, np.cos(np.radians(np.nan_to_num(s["vdir"][lo:i1])) - face))
    on = H * cw
    if on.size and on.max() >= H0 + 0.2:
        j = int(np.argmax(on))
        d = dt.datetime.strptime(times[lo + j][:10], "%Y-%m-%d")
        return f"vagues vers la plage le {d.day}/{d.month} ({on.max():.1f} m)"
    return None


def build(beaches, times, series, reps, settings, now_key):
    days = sorted({t[:10] for t in times if t >= now_key[:10]})[:FORECAST_DAYS]
    i_start = times.index(days[0] + "T00:00") if days and days[0] + "T00:00" in times else 0
    out_b = {}
    for b, s in zip(beaches, series):
        rs, rw = resets_from_reports(reps, b["id"], times, settings, now_key)
        S, W = indices(b, s, times, settings["tau_s"], settings["tau_w"], rs, rw)
        sp, wp = settings["sand"], settings["water"]
        ps = prob(S, sp["ref"], sp["a"], sp["b"], sp["off"].get(b["id"], 0.0))
        pw = prob(W, wp["ref"], wp["a"], wp["b"], wp["off"].get(b["id"], 0.0))
        seg = slice(i_start, i_start + 24 * len(days))
        lv = [level(a, c) for a, c in zip(ps[seg], pw[seg])]
        out_b[b["id"]] = dict(
            sand=[int(round(100 * v)) for v in ps[seg]], water=[int(round(100 * v)) for v in pw[seg]], level=lv,
            why=why(b, s, times, i_start, i_start + 24),
            corrected=bool(rs or rw), nettoyee=bool(b["cleaned"]))
    return dict(generated=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).replace(microsecond=0).isoformat() + "Z", version=VERSION,
                start=times[i_start], hours=24 * len(days),
                levels=["Sans doute aucune", "Possibles", "Sur le sable", "Dans l'eau"],
                method="Stock de feuilles d'herbiers : vagues et vent vers la plage × saison × exposition, oubli lent, nettoyage ; "
                       "corrigé par les retours des 7 derniers jours. Indicatif.",
                beaches=out_b)


def run(test=False):
    import credibility as K  # noqa: F401  (vérifie que le module est là)
    beaches = load_beaches()
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) + dt.timedelta(hours=UTC_OFFSET)
    now_key = now.strftime("%Y-%m-%dT%H:00")
    if test:
        import random
        times, series, status = synthetic_series(beaches, now)
        reps = synthetic_reports(beaches, times, series, now_key, random.Random(3))
    else:
        times, series, status = fetch_series(beaches)
        reps = fetch_algae_reports(beaches)
    settings = jload(P_SET, {}).get("settings") or json.loads(json.dumps(DEFAULT))
    rows = update_examples(reps, beaches, times, series)
    settings, rep = learn(rows, settings, beaches)
    jsave(P_SET, dict(updated=now_key, version=VERSION, settings=settings, apprentissage=rep,
                      plages=explain_offsets(settings, beaches, rep)))
    out = build(beaches, times, series, reps, settings, now_key)
    out["sources"] = dict(status, signalements=len(reps))
    jsave(P_OUT, out)
    day0 = slice(0, 24)
    for b in beaches:
        x = out["beaches"][b["id"]]
        print(f"  {b['name']:<14} sable {max(x['sand'][day0]):>3}%  eau {max(x['water'][day0]):>3}%  "
              f"niveaux 7 j : {''.join(str(max(x['level'][d*24:(d+1)*24])) for d in range(out['hours']//24))}"
              f"{'  (corrigé)' if x['corrected'] else ''}  {x['why'] or ''}")
    print(f"[algues] apprentissage : {json.dumps(rep, ensure_ascii=False)[:400]}")


# ====================================================================== test hors ligne
def synthetic_series(beaches, now):
    """Un mois d'automne inventé : mer calme, puis coup de vent d'est (2 m de vagues) il y a 3 jours, puis vent de terre."""
    t0 = (now - dt.timedelta(days=PAST_DAYS)).replace(minute=0, second=0, microsecond=0, hour=0)
    n = 24 * (PAST_DAYS + FORECAST_DAYS)
    times = [(t0 + dt.timedelta(hours=i)).strftime("%Y-%m-%dT%H:00") for i in range(n)]
    hrs = np.arange(n)
    storm = sum(np.exp(-((hrs - 24 * (PAST_DAYS - d)) / 18.0) ** 2) * k for d, k in ((3, 1.0), (11, 0.7), (19, 1.1), (30, 0.6)))
    storm2 = np.exp(-((hrs - 24 * (PAST_DAYS + 3)) / 12.0) ** 2)   # un autre coup de mer prévu dans 3 jours, du nord
    series = []
    for b in beaches:
        wind = 12 + 35 * storm + 30 * storm2 + 4 * np.sin(hrs / 24 * 2 * np.pi)
        wdir = np.where(storm2 > 0.2, 10.0, np.where(storm > 0.2, 80.0, 250.0))
        wave = 0.3 + 1.8 * storm + 1.4 * storm2
        vdir = np.where(storm2 > 0.2, 10.0, np.where(storm > 0.2, 80.0, 120.0))
        series.append(dict(time=times, wind=wind, wdir=wdir, wave=wave, vdir=vdir))
    return times, series, {"vent": "synthétique", "vagues": "synthétique"}


def synthetic_reports(beaches, times, series, now_key, rnd):
    """Des retours inventés qui suivent une « vérité » (stock à 6 jours, Bowling nettoyée) pour voir si l'apprentissage la retrouve."""
    reps = []
    truth = dict(beaches=[dict(b, cleaned=1 if b["id"] == "bowling" else b["cleaned"]) for b in beaches])
    for k, b in enumerate(truth["beaches"]):
        S, W = indices(b, series[k], times, 6, 18)
        ps = prob(S, 3.0, -2.0, 2.4); pw = prob(W, 1.5, -2.6, 2.2)
        for i, t in enumerate(times):
            if t > now_key or t[11:13] not in ("10", "16") or rnd.random() > 0.45:
                continue
            reps.append(dict(id=f"t{k}-{i}", beach=b["id"], key=t, sand=int(rnd.random() < ps[i]),
                             water=int(rnd.random() < pw[i]), w=round(0.4 + 0.6 * rnd.random(), 2)))
    return reps


if __name__ == "__main__":
    if "--test" in sys.argv:
        P_OUT = os.path.join(DATA, "_test_algues.json")
        P_EX = os.path.join(DATA, "_test_algues_exemples.csv")
        P_SET = os.path.join(DATA, "_test_algues_reglages.json")
        for p in (P_EX, P_SET):
            if os.path.exists(p):
                os.remove(p)
        run(test=True)
    else:
        run()
