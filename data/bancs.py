"""
Registre des zones de présence suivies (v0.7, vocabulaire v0.8) — « on garde un œil sur chaque banc ».

NB (relecture externe) : on ne peut pas prouver qu'une nouvelle observation vient du MÊME groupe de méduses ;
on parle donc de « zone de présence suivie ». Les parts de scénarios sont des parts de SIMULATIONS, pas des probabilités.

Chaque nuit :
  1. Naissance  : chaque observation crédible (iNaturalist, GBIF, OBIS, signalement fiable) qui n'appartient à aucun
                  banc connu crée un banc : 160 méduses virtuelles autour du point, chacune avec son propre scénario
                  (prise au vent de 50 % à 150 %, en surface ou à 15 m le jour comme la pélagie).
  2. Recalage   : une observation à moins de 12 km d'un banc suivi = le même banc. Les méduses virtuelles proches de
                  l'observation sont gardées (et multipliées), les autres sont écartées : la position est resserrée.
  3. Avancée    : le banc avance avec les courants RÉELS des dernières heures (Copernicus recalcule chaque jour le passé
                  récent avec les mesures satellites et bouées), pas avec la prévision de la veille. Ça améliore les
                  courants utilisés, mais ne corrige pas une erreur sur la position de départ ou le comportement.
  4. Projection : chaque banc est projeté sur 8 jours avec les courants, le vent et les vagues prévus.
                  Si des scénarios arrivent sur une plage de Djerba → alerte lointaine, avec la part des scénarios simulés.
  5. Contrôle   : des « rien vu » sur une plage où il devait arriver l'affaiblissent ; sans nouvelle observation il
                  s'estompe (demi-vie 12 jours). Il est arrêté s'il est trop vieux, trop étalé, échoué ou contredit.
  6. Carnet     : chaque prévision d'arrivée est notée puis vérifiée les jours suivants (data/carnet.json) → on saura
                  à combien de jours le suivi est fiable.

Fichiers : data/bancs_etat.json (état complet, méduses virtuelles), data/bancs.json (résumé lu par l'app),
           data/carnet.json (carnet de bord des prévisions).
"""
import os, sys, json, math, datetime as dt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import sources as S
import model as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("WN_DATA") or os.path.join(ROOT, "data")
EPOCH = np.datetime64("1970-01-01T00:00", "m")
DAY = np.timedelta64(1, "D")


def paths():
    return (os.path.join(DATA, "bancs_etat.json"), os.path.join(DATA, "bancs.json"), os.path.join(DATA, "carnet.json"))


def jload(p, default):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def jsave(p, obj, indent=1):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def t2i(t):
    return int((t.astype("datetime64[m]") - EPOCH) / np.timedelta64(1, "m"))


def i2t(i):
    return EPOCH + np.timedelta64(int(i), "m")


def fr_date(d):
    """'2026-10-14' → '14/10'"""
    return f"{d[8:10]}/{d[5:7]}"


# ====================================================================== noms
def place_name(lat, lon):
    best, bd = None, 1e9
    for n, (la, lo) in C.PLACES.items():
        d = float(M.km_between(lat, lon, la, lo))
        if d < bd:
            best, bd = n, d
    de = "d'" if best[0].lower() in "aeiouéèêàâîôûh" else "de "
    if bd <= 25:
        return f"près {de}{best}"
    if bd <= 90:
        return f"au large {de}{best}"
    return "en pleine mer"


SPECIES_FR = [("pelagia", "pélagies", "pélagie"), ("pélagie", "pélagies", "pélagie"),
              ("rhizostoma", "méduses chou-fleur", "méduse chou-fleur"), ("chou-fleur", "méduses chou-fleur", "méduse chou-fleur"),
              ("cotylorhiza", "méduses œuf au plat", "méduse œuf au plat"), ("œuf", "méduses œuf au plat", "méduse œuf au plat"),
              ("rhopilema", "méduses nomades", "méduse nomade"), ("physalia", "galions portugais", "galion portugais"),
              ("galion", "galions portugais", "galion portugais"), ("velella", "vélelles", "vélelle"),
              ("aurelia", "aurélies", "aurélie"), ("chrysaora", "chrysaores", "chrysaore"), ("carybdea", "cuboméduses", "cuboméduse")]


def species_fr(taxon, plural=True):
    t = (taxon or "").lower()
    for key, pl, sg in SPECIES_FR:
        if key in t:
            return pl if plural else sg
    return "méduses" if plural else "méduse"


def is_sail(taxon):
    t = (taxon or "").lower()
    return any(k in t for k in C.BANC_SAIL_TAXA)


# ====================================================================== méduses virtuelles
def new_particles(drift, lat, lon, t, spread_km, n, taxon, rng):
    dy = rng.normal(0, spread_km, n) / 111.32
    dx = rng.normal(0, spread_km, n) / (111.32 * math.cos(math.radians(lat)))
    ok = drift.is_sea(t, lat + dy, lon + dx)
    if ok.sum() < n / 2:                          # observation faite depuis la plage : on repart de la mer la plus proche
        ns = drift.nearest_sea(t, lat, lon)
        if ns is not None:
            lat, lon = ns
            r = min(spread_km, 2.0)
            dy = rng.normal(0, r, n) / 111.32
            dx = rng.normal(0, r, n) / (111.32 * math.cos(math.radians(lat)))
            ok = drift.is_sea(t, lat + dy, lon + dx)
    if not ok.any():
        return None
    py, px = (lat + dy)[ok], (lon + dx)[ok]
    k = len(py)
    sail = is_sail(taxon)
    tl = (taxon or "").lower()
    share_mig = 0.0 if sail else (0.7 if ("pelagia" in tl or "pélagie" in tl) else C.TYPES["migrante"]["share"])
    kind = (rng.random(k) < share_mig).astype(int)
    wm = rng.uniform(1 - C.BANC_WINDAGE_SPREAD, 1 + C.BANC_WINDAGE_SPREAD, k) * (C.BANC_SAIL_WINDAGE if sail else 1.0)
    return dict(lat=py.round(4).tolist(), lon=px.round(4).tolist(), kind=kind.tolist(), wm=wm.round(3).tolist(),
                pt=[t2i(t)] * k)


def centroid(lat, lon):
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    if not len(lat):
        return None, None, None
    cy, cx = float(lat.mean()), float(lon.mean())
    sp = float(np.sqrt(np.mean(M.km_between(cy, cx, lat, lon) ** 2)))
    return cy, cx, sp


# ====================================================================== moteur de dérive (tous les bancs ensemble)
def advect(drift, P, t_start, t_end, snaps=None, nb=0):
    """
    P : dict de tableaux numpy lat, lon, kind, wm, pt (minutes), bi (n° de banc), alive (bool).
    Avance chaque méduse virtuelle de max(pt, t_start) à t_end. Modifie P.
    Retourne arrivals[(bi, beach_id, date)] = nb, landed[bi] = liste de (lat, lon), snapshots[date][bi] = (lat, lon, étalement).
    """
    beaches = drift.beaches
    bids = [b["id"] for b in beaches]
    arrivals, landed, snapshots = {}, {}, {}
    step = np.timedelta64(C.DT // 60, "m")
    t = t_start
    snaps = sorted(snaps or [])
    si = 0
    pt = np.array([i2t(x) for x in P["pt"]]) if len(P["pt"]) else np.array([], "datetime64[m]")
    rb = drift.region

    def hit_beach(idx, radius, t):
        if not len(idx):
            return np.zeros(0, bool)
        d = M.km_between(P["lat"][idx, None], P["lon"][idx, None], drift.blat[None, :], drift.blon[None, :])
        nbr = np.argmin(d, axis=1)
        hit = d[np.arange(len(idx)), nbr] <= radius
        day = M.local_date(t)
        for n, i in zip(nbr[hit], idx[hit]):
            key = (int(P["bi"][i]), bids[n], day)
            arrivals[key] = arrivals.get(key, 0) + 1
        return hit

    while t < t_end:
        act = np.where(P["alive"] & (pt <= t))[0]
        if len(act):
            y, x = P["lat"][act], P["lon"][act]
            kd, wm = P["kind"][act], P["wm"][act]
            u1, v1 = drift.velocity(t, y, x, kd, wm)
            cos = np.cos(np.radians(y))
            ym = y + np.nan_to_num(v1) * C.DT / 2 / M.M_PER_DEG
            xm = x + np.nan_to_num(u1) * C.DT / 2 / (M.M_PER_DEG * cos)
            u2, v2 = drift.velocity(t + step // 2, ym, xm, kd, wm)
            u2 = np.where(np.isnan(u2), u1, u2); v2 = np.where(np.isnan(v2), v1, v2)
            ny = y + np.nan_to_num(v2) * C.DT / M.M_PER_DEG
            nx = x + np.nan_to_num(u2) * C.DT / (M.M_PER_DEG * cos)
            land = ~drift.is_sea(t, ny, nx)
            P["lat"][act] = np.where(land, y, ny); P["lon"][act] = np.where(land, x, nx)
            # à la côte : arrivée sur une plage (< 6 km) ou échouage ailleurs
            li = act[land]
            if len(li):
                hit_beach(li, C.R_BEACHED_KM, t)
                for i in li:
                    landed.setdefault(int(P["bi"][i]), []).append((float(P["lat"][i]), float(P["lon"][i])))
                P["alive"][li] = False
            # près d'une plage (< 3 km) : arrivée
            fi = act[~land]
            near = fi[drift.near_beach(P["lat"][fi], P["lon"][fi])]
            hit = hit_beach(near, C.R_NEAR_KM, t)
            if len(near):
                P["alive"][near[hit]] = False
            out = (P["lat"][act] < rb["lat_min"] - .3) | (P["lat"][act] > rb["lat_max"] + .3) | \
                  (P["lon"][act] < rb["lon_min"] - .3) | (P["lon"][act] > rb["lon_max"] + .3)
            P["alive"][act[out]] = False
        t = t + step
        while si < len(snaps) and t >= snaps[si]:
            day = M.local_date(snaps[si])
            snapshots[day] = {}
            for b in range(nb):
                sel = P["alive"] & (P["bi"] == b) & (pt <= t)
                if sel.any():
                    snapshots[day][b] = centroid(P["lat"][sel], P["lon"][sel])
            si += 1
    return arrivals, landed, snapshots


def to_arrays(bancs):
    keys = ("lat", "lon", "kind", "wm", "pt")
    P = {k: [] for k in keys}; P["bi"] = []
    for i, b in enumerate(bancs):
        for k in keys:
            P[k] += b["p"][k]
        P["bi"] += [i] * len(b["p"]["lat"])
    out = dict(lat=np.array(P["lat"], float), lon=np.array(P["lon"], float), kind=np.array(P["kind"], int),
               wm=np.array(P["wm"], float), pt=np.array(P["pt"], np.int64), bi=np.array(P["bi"], int))
    out["alive"] = np.ones(len(out["lat"]), bool)
    return out


def from_arrays(bancs, P, t_now):
    """Réécrit les méduses virtuelles encore en mer dans chaque banc (positions à t_now)."""
    tn = t2i(t_now)
    for i, b in enumerate(bancs):
        sel = P["alive"] & (P["bi"] == i)
        b["p"] = dict(lat=P["lat"][sel].round(4).tolist(), lon=P["lon"][sel].round(4).tolist(),
                      kind=P["kind"][sel].tolist(), wm=P["wm"][sel].round(3).tolist(), pt=[max(tn, int(x)) for x in P["pt"][sel]])


# ====================================================================== recalage par une observation
def confirm(b, o, drift, rng):
    lat, lon = np.array(b["p"]["lat"]), np.array(b["p"]["lon"])
    if not len(lat):
        return False
    d = M.km_between(o["lat"], o["lon"], lat, lon)
    _, _, sp = centroid(lat, lon)
    R = max(4.0, (sp or 0) / 2)
    w = np.exp(-0.5 * (d / R) ** 2)
    w = w / w.sum() if w.sum() > 1e-9 else np.full(len(lat), 1 / len(lat))
    n = C.BANC_PARTICLES
    keep = int(0.6 * n)
    idx = rng.choice(len(lat), size=keep, p=w)
    jit = 1.0 / 111.32
    kept = dict(lat=(lat[idx] + rng.normal(0, jit, keep)).round(4).tolist(),
                lon=(lon[idx] + rng.normal(0, jit, keep)).round(4).tolist(),
                kind=[b["p"]["kind"][i] for i in idx], wm=[b["p"]["wm"][i] for i in idx], pt=[b["p"]["pt"][i] for i in idx])
    fresh = new_particles(drift, o["lat"], o["lon"], o["time"], o.get("spread_km", 2), n - keep, o.get("taxon"), rng)
    if fresh:
        for k in kept:
            kept[k] += fresh[k]
    b["p"] = kept
    b["force"] = round(min(1.0, b["force"] + 0.35 * min(1.0, o["weight"])), 3)
    b["confirmations"] += 1
    b["refs"].append(o["ref"])
    b["dernier_vu"] = str(o["time"])[:16]
    b["dernier_vu_lieu"] = place_name(o["lat"], o["lon"])
    if b.get("espece") in (None, "méduses") and species_fr(o.get("taxon")) != "méduses":
        b["espece"] = species_fr(o.get("taxon"))
    b["n0"] = max(b["n0"], len(b["p"]["lat"]))
    return True


def birth(st, o, drift, rng):
    p = new_particles(drift, o["lat"], o["lon"], o["time"], o.get("spread_km", 2), C.BANC_PARTICLES, o.get("taxon"), rng)
    if p is None:
        return None
    b = dict(id=st["next_id"], ne=str(o["time"])[:16], lieu=place_name(o["lat"], o["lon"]), lat0=round(o["lat"], 3),
             lon0=round(o["lon"], 3), source=o["source"], taxon=o.get("taxon") or "?", espece=species_fr(o.get("taxon")),
             force=round(min(1.0, 0.5 + 0.3 * min(1.0, o["weight"])), 3), confirmations=0, contradictions=0,
             refs=[o["ref"]], dernier_vu=str(o["time"])[:16], dernier_vu_lieu=place_name(o["lat"], o["lon"]),
             maj=str(o["time"])[:16], n0=len(p["lat"]), p=p, trace=[], dist=[])
    st["next_id"] += 1
    return b


def merge_close(bancs, rng):
    """Deux bancs qui se retrouvent au même endroit sont fusionnés (c'est sans doute le même)."""
    out = []
    for b in sorted(bancs, key=lambda x: x["ne"]):
        cy, cx, sp = centroid(b["p"]["lat"], b["p"]["lon"])
        if cy is None:
            out.append(b); continue
        target = None
        for a in out:
            ay, ax, asp = centroid(a["p"]["lat"], a["p"]["lon"])
            if ay is not None and float(M.km_between(cy, cx, ay, ax)) <= max(C.BANC_MERGE_KM, 0.5 * (sp + asp)):
                target = a; break
        if target is None:
            out.append(b); continue
        for k in target["p"]:
            target["p"][k] += b["p"][k]
        m = len(target["p"]["lat"])
        if m > C.BANC_PARTICLES:
            idx = sorted(rng.choice(m, C.BANC_PARTICLES, replace=False).tolist())
            target["p"] = {k: [v[i] for i in idx] for k, v in target["p"].items()}
        target["force"] = round(min(1.0, max(target["force"], b["force"]) + 0.1), 3)
        target["confirmations"] += b["confirmations"] + 1
        target["refs"] += b["refs"]
        target["dernier_vu"] = max(target["dernier_vu"], b["dernier_vu"])
        target["n0"] = max(target["n0"], len(target["p"]["lat"]))
        print(f"[bancs] zone n°{b['id']} fusionnée avec la n°{target['id']}")
    return out


# ====================================================================== preuves sur les plages (pour le carnet)
def evidence(obs, reports, beaches):
    """(date, plage) → {'vu': n, 'rien': n} à partir des observations scientifiques et des signalements fiables."""
    ev = {}
    def add(day, bid, key):
        e = ev.setdefault((day, bid), {"vu": 0, "rien": 0}); e[key] += 1
    for o in obs:
        if o["source"] not in ("iNaturalist", "GBIF", "OBIS"):
            continue
        d = [float(M.km_between(o["lat"], o["lon"], b["lat"], b["lon"])) for b in beaches]
        k = int(np.argmin(d))
        if d[k] <= 8:
            add(M.local_date(o["time"]), beaches[k]["id"], "rien" if o.get("none_seen") else "vu")
    if reports:
        import credibility as CR
        for r in reports:
            if r.get("dev", {}).get("banned"):
                continue
            if CR.trust(r["dev"]) < 0.4 or r.get("seen") is None:
                continue
            add(M.local_date(r["time"]), r["beach_id"], "vu" if r["seen"] else "rien")
    return ev


def ev_around(ev, day, bid):
    d0 = dt.date.fromisoformat(day)
    vu = rien = 0
    for k in (-1, 0, 1):
        e = ev.get(((d0 + dt.timedelta(days=k)).isoformat(), bid))
        if e:
            vu += e["vu"]; rien += e["rien"]
    return vu, rien


def update_carnet(carnet, new_entries, ev, today, beaches, bancs_by_id):
    """Ajoute les nouvelles prévisions, vérifie les anciennes, compte les « surprises », met à jour les stats."""
    known = {(e["banc"], e["faite"], e["plage"], e["date"]) for e in carnet["entrees"]}
    for e in new_entries:
        if (e["banc"], e["faite"], e["plage"], e["date"]) not in known:
            carnet["entrees"].append(e)
    t0 = dt.date.fromisoformat(today)
    for e in carnet["entrees"]:
        if e.get("resultat"):
            continue
        d = dt.date.fromisoformat(e["date"])
        if d >= t0:
            continue
        vu, rien = ev_around(ev, e["date"], e["plage"])
        if vu:
            e["resultat"] = "vu"
        elif rien:
            e["resultat"] = "rien vu"
            b = bancs_by_id.get(e["banc"])
            if b is not None and not e.get("contredit"):
                b["force"] = round(b["force"] * C.BANC_CONTRADICT, 3)
                b["contradictions"] += 1
                e["contredit"] = True
        elif (t0 - d).days >= 4:
            e["resultat"] = "pas d'info"
    # surprises : méduses vues sur une plage de Djerba sans qu'aucun banc ne l'ait annoncé
    seen = set(tuple(x) for x in carnet.get("_surprises_vues", []))
    bids = {b["id"] for b in beaches}
    for (day, bid), e in ev.items():
        if not e["vu"] or bid not in bids or (day, bid) in seen:
            continue
        if dt.date.fromisoformat(day) >= t0:
            continue
        seen.add((day, bid))
        d0 = dt.date.fromisoformat(day)
        annonce = any(x["plage"] == bid and abs((dt.date.fromisoformat(x["date"]) - d0).days) <= 1 and x["faite"] < day
                      for x in carnet["entrees"])
        carnet["surprises" if not annonce else "annonces_confirmees"].append(dict(date=day, plage=bid))
    carnet["_surprises_vues"] = sorted(seen)[-2000:]
    # garde 400 jours
    lim = (t0 - dt.timedelta(days=400)).isoformat()
    carnet["entrees"] = [e for e in carnet["entrees"] if e["date"] >= lim][-6000:]
    carnet["surprises"] = [x for x in carnet["surprises"] if x["date"] >= lim]
    carnet["annonces_confirmees"] = [x for x in carnet["annonces_confirmees"] if x["date"] >= lim]
    # statistiques par délai (1 à 8 jours)
    by = {}
    for e in carnet["entrees"]:
        s = by.setdefault(str(e["delai"]), {"prevues": 0, "verifiables": 0, "vu": 0, "rien_vu": 0})
        s["prevues"] += 1
        if e.get("resultat") == "vu":
            s["verifiables"] += 1; s["vu"] += 1
        elif e.get("resultat") == "rien vu":
            s["verifiables"] += 1; s["rien_vu"] += 1
    for s in by.values():
        s["taux_reussite"] = round(s["vu"] / s["verifiables"], 2) if s["verifiables"] else None
    carnet["stats"] = dict(par_delai=dict(sorted(by.items(), key=lambda kv: int(kv[0]))),
                           surprises=len(carnet["surprises"]), annonces_confirmees=len(carnet["annonces_confirmees"]),
                           note="Chaque prévision d'arrivée d'un banc est vérifiée avec les observations et signalements ±1 jour. "
                                "taux_reussite = « vu » / (« vu » + « rien vu »). Surprises = méduses vues sans annonce d'un banc.")
    return carnet


# ====================================================================== nuit
def run(t_now, obs, reports=None, status=None, loaders=None):
    """Met à jour le registre et retourne (résumé pour l'app, alertes par plage)."""
    status = status if status is not None else {}
    L = loaders or S
    P_STATE, P_OUT, P_CARNET = paths()
    rng = np.random.default_rng(int(t2i(t_now)) % 100000)
    st = jload(P_STATE, {"next_id": 1, "bancs": [], "termines": [], "refs": [], "maj": None})
    region = C.REGION_BOX
    today = M.local_date(t_now + np.timedelta64(2, "h"))
    used = set(st["refs"])

    # --- observations candidates
    cands = []
    for o in obs:
        if o.get("none_seen") or o["weight"] < 0.3 or o["source"] == "presse" or o["ref"] in used:
            continue
        if not (region["lat_min"] <= o["lat"] <= region["lat_max"] and region["lon_min"] <= o["lon"] <= region["lon_max"]):
            continue
        age = (t_now - o["time"]) / DAY
        if age < -1 or age > C.LOOKBACK_DAYS:
            continue
        cands.append(o)
    cands.sort(key=lambda o: o["time"])

    bancs = st["bancs"]
    pts = [x for b in bancs for x in b["p"]["pt"]]
    t_first = min([i2t(min(pts))] if pts else [t_now])
    t_first = min([t_first] + [o["time"] for o in cands]) - np.timedelta64(3, "h")
    t_end = t_now + np.timedelta64(C.BANC_HORIZON_DAYS, "D")
    if not bancs and not cands:
        status["bancs"] = "aucun banc à suivre"
        return write_outputs(st, [], {}, today, t_now, obs, reports)

    # --- données : vent, courants surface + 15 m, vagues, sur toute la région, du passé à +8 jours
    s_dt = t_first.astype("datetime64[s]").astype(dt.datetime)
    e_dt = (t_end + np.timedelta64(1, "h")).astype("datetime64[s]").astype(dt.datetime)
    past = int((t_now - t_first) / DAY) + 2
    try:
        wind = L.load_wind(region, 4, past_days=min(92, past), forecast_days=C.BANC_HORIZON_DAYS + 1)
        cur_s = M.ensure_ascending(L.load_currents_surface(region, s_dt, e_dt))
    except Exception as e:
        status["bancs"] = f"indisponible : {str(e)[:120]}"
        print(f"[bancs] données indisponibles : {e}")
        return None, {}
    cur_d = stokes = None
    try:
        cur_d = M.ensure_ascending(L.load_currents_depth(region, s_dt, e_dt))
    except Exception as e:
        print(f"[bancs] courants 15 m indisponibles : {str(e)[:120]}")
    try:
        stokes = M.ensure_ascending(L.load_stokes(region, s_dt, e_dt))
    except Exception as e:
        print(f"[bancs] vagues indisponibles : {str(e)[:120]}")
    cmax = cur_s.times.max()
    if stokes is not None and stokes.times.max() < cmax:
        cmax = stokes.times.max()
    if cmax < t_end:
        print(f"[bancs] courants prévus jusqu'au {str(cmax)[:10]} seulement : projection raccourcie")
        t_end = min(t_end, cmax)
    drift = M.Drift(cur_s, cur_d, stokes, wind, None, C.BEACHES, region)

    # --- 1-2. recalage des bancs connus, naissance des nouveaux
    n_new = n_conf = 0
    for o in cands:
        best, bd = None, 1e9
        for b in bancs:
            if not b["p"]["lat"]:
                continue
            d = float(np.min(M.km_between(o["lat"], o["lon"], np.array(b["p"]["lat"]), np.array(b["p"]["lon"]))))
            if d < bd:
                best, bd = b, d
        if best is not None and bd <= C.BANC_MERGE_KM:
            if confirm(best, o, drift, rng):
                n_conf += 1
        else:
            nb = birth(st, o, drift, rng)
            if nb:
                bancs.append(nb); n_new += 1
        used.add(o["ref"])

    # --- 3. avancée jusqu'à maintenant avec les courants du passé récent (analyse)
    P = to_arrays(bancs)
    _, landed, _ = advect(drift, P, t_first, t_now, nb=len(bancs))
    landed = {bancs[i]["id"]: v for i, v in landed.items()}
    from_arrays(bancs, P, t_now)
    bancs = merge_close(bancs, rng)

    # --- force, trace, fin de suivi
    alive_b, ended = [], []
    for b in bancs:
        last = np.datetime64(b["maj"], "m")
        days = max(0.0, (t_now - last) / DAY)
        if days > 0:
            b["force"] = round(b["force"] * 0.5 ** (days / C.BANC_HALF_LIFE_DAYS), 3)
        b["maj"] = str(t_now)[:16]
        cy, cx, sp = centroid(b["p"]["lat"], b["p"]["lon"])
        age = (t_now - np.datetime64(b["ne"], "m")) / DAY
        alive_frac = len(b["p"]["lat"]) / max(1, b["n0"])
        why = None
        if age > C.BANC_MAX_AGE_DAYS:
            why = f"suivi pendant {C.BANC_MAX_AGE_DAYS} jours sans être revu : arrêté"
        elif alive_frac < C.BANC_MIN_ALIVE or cy is None:
            lp = landed.get(b["id"], [])
            where = place_name(*np.mean(np.array(lp), axis=0)) if lp else "hors de la zone suivie"
            why = f"arrivé à la côte ou sorti de la zone ({where})"
        elif sp > C.BANC_MAX_SPREAD_KM:
            why = "trop dispersé : on ne peut plus dire où il est"
        elif b["force"] < C.BANC_MIN_STRENGTH:
            why = "contredit par des « rien vu »" if b["contradictions"] else "pas revu depuis longtemps : il s'est sans doute dispersé"
        if why:
            ended.append(dict(id=b["id"], nom=nom(b), fin=today, raison=why))
            print(f"[bancs] fin de la zone n°{b['id']} : {why}")
            continue
        dist = float(M.km_between(cy, cx, *C.DJERBA_CENTER))
        b["trace"] = (b["trace"] + [dict(date=today, lat=round(cy, 3), lon=round(cx, 3), etalement=round(sp, 1))])[-35:]
        b["dist"] = (b["dist"] + [round(dist, 1)])[-35:]
        alive_b.append(b)
    bancs = alive_b

    # --- 4. projection sur 8 jours
    snaps = [np.datetime64(M.local_date(t_now + np.timedelta64(2, "h")) + "T11:00", "m") + k * DAY
             for k in range(C.BANC_HORIZON_DAYS + 1)]
    snaps = [s for s in snaps if t_now <= s <= t_end]
    P = to_arrays(bancs)
    n_alive = np.bincount(P["bi"], minlength=len(bancs)) if len(bancs) else np.zeros(0)
    arrivals, flanded, snapshots = advect(drift, P, t_now, t_end, snaps=snaps, nb=len(bancs))
    forecasts = {}
    for i, b in enumerate(bancs):
        per = {}
        for (bi, bid, day), n in arrivals.items():
            if bi == i:
                per.setdefault(bid, {})[day] = n
        traj = [dict(date=d, lat=round(v[i][0], 3), lon=round(v[i][1], 3), etalement=round(v[i][2], 1))
                for d, v in sorted(snapshots.items()) if i in v]
        sel = P["alive"] & (P["bi"] == i)
        end = centroid(P["lat"][sel], P["lon"][sel]) if sel.any() else None
        forecasts[b["id"]] = dict(per=per, traj=traj, n=int(n_alive[i]) if len(n_alive) else 0, end=end,
                                  landed=flanded.get(i, []))

    # --- 5-6. carnet + contrôle par les « rien vu »
    ev = evidence(obs, reports, C.BEACHES)
    new_entries = []
    for b in bancs:
        f = forecasts[b["id"]]
        for bid, days in f["per"].items():
            for day, n in days.items():
                frac = n / max(1, f["n"])
                if frac >= C.BANC_ALERT_FRAC / 2:
                    lead = (dt.date.fromisoformat(day) - dt.date.fromisoformat(today)).days
                    new_entries.append(dict(banc=b["id"], faite=today, date=day, delai=max(0, lead), plage=bid,
                                            part=round(frac, 3), force=b["force"], resultat=None))
    carnet = jload(P_CARNET, {"entrees": [], "surprises": [], "annonces_confirmees": [], "stats": {}})
    carnet = update_carnet(carnet, new_entries, ev, today, C.BEACHES, {b["id"]: b for b in bancs})
    jsave(P_CARNET, carnet)

    st["bancs"] = bancs
    st["termines"] = [x for x in st["termines"] + ended
                      if (dt.date.fromisoformat(today) - dt.date.fromisoformat(x["fin"])).days <= C.BANC_ARCHIVE_DAYS]
    st["refs"] = sorted(used)[-5000:]
    st["maj"] = str(t_now)[:16]
    status["bancs"] = f"ok ({len(bancs)} zones suivies, {n_new} nouvelles, {n_conf} recalées, {len(ended)} terminées)"
    print(f"[bancs] {status['bancs']}")
    return write_outputs(st, bancs, forecasts, today, t_now, obs, reports, carnet)


def nom(b):
    return f"Zone n°{b['id']} · {b['espece']} repérées {b['lieu']} le {fr_date(b['ne'][:10])}".replace(
        "galions portugais repérées", "galions portugais repérés")


def confidence(score):
    lo, mid, hi = C.BANC_CONF
    return None if score < lo else "faible" if score < mid else "moyenne" if score < hi else "forte"


def write_outputs(st, bancs, forecasts, today, t_now, obs, reports, carnet=None):
    P_STATE, P_OUT, P_CARNET = paths()
    names = {b["id"]: b.get("short", b["name"]) for b in C.BEACHES}
    actifs, alerts = [], {}
    for b in bancs:
        f = forecasts.get(b["id"], {})
        arr = []
        for bid, days in (f.get("per") or {}).items():
            tot = sum(days.values())
            frac = tot / max(1, f.get("n", 0))
            score = frac * (0.5 + 0.5 * b["force"])
            conf = confidence(score)
            if frac < C.BANC_ALERT_FRAC or conf is None:
                continue
            ds = sorted(days)
            cum, dmin, dmax = 0, ds[0], ds[-1]
            for d in ds:                                         # 80 % central des arrivées
                cum += days[d]
                if cum >= 0.1 * tot and dmin == ds[0] and cum - days[d] < 0.1 * tot:
                    dmin = d
                if cum >= 0.9 * tot:
                    dmax = d; break
            arr.append(dict(plage=bid, nom=names.get(bid, bid), du=dmin, au=dmax, part=round(frac, 3), confiance=conf))
        arr.sort(key=lambda a: (a["du"], -a["part"]))
        dist_now = b["dist"][-1] if b["dist"] else None
        dist_prev = b["dist"][-2] if len(b["dist"]) > 1 else None
        end = f.get("end")
        dist_end = float(M.km_between(end[0], end[1], *C.DJERBA_CENTER)) if end and end[0] is not None else None
        if dist_prev is None:
            tend = "nouveau"
        elif dist_now < dist_prev - 5:
            tend = "se rapproche"
        elif dist_now > dist_prev + 5:
            tend = "s'éloigne"
        else:
            tend = "stable"
        if arr:
            a = arr[0]
            when = f"le {fr_date(a['du'])}" if a["du"] == a["au"] else f"entre le {fr_date(a['du'])} et le {fr_date(a['au'])}"
            msg = f"Pourrait approcher {a['nom']} {when} : {round(a['part'] * 100)} % des scénarios simulés y arrivent."; cap = "approche"
        elif dist_end is not None and dist_now is not None and dist_end < dist_now - 10:
            msg = f"Se dirige vers Djerba, sans arriver sur nos plages dans les simulations à {C.BANC_HORIZON_DAYS} jours."; cap = "rapproche"
        elif dist_end is not None and dist_now is not None and dist_end > dist_now + 10:
            msg = f"S'éloigne de Djerba (direction : {place_name(end[0], end[1])})."; cap = "eloigne"
        else:
            msg = "Reste dans sa zone, pas de menace pour Djerba d'ici " + f"{C.BANC_HORIZON_DAYS} jours."; cap = "reste"
        cy, cx, sp = centroid(b["p"]["lat"], b["p"]["lon"])
        actifs.append(dict(id=b["id"], nom=nom(b), espece=b["espece"], ne=b["ne"][:10], vu_pres=b["lieu"], source=b["source"],
                           age_jours=int((t_now - np.datetime64(b["ne"], "m")) / DAY), force=b["force"],
                           confirmations=b["confirmations"], contradictions=b["contradictions"],
                           dernier_vu=b["dernier_vu"][:10], position=dict(lat=round(cy, 3), lon=round(cx, 3), lieu=place_name(cy, cx)),
                           etalement_km=round(sp, 1), distance_djerba_km=None if dist_now is None else round(dist_now),
                           tendance=tend, cap=cap, message=msg, arrivees=arr, trace=b["trace"], prevision=f.get("traj", [])))
        for a in arr:
            alerts.setdefault(a["plage"], []).append(dict(banc=b["id"], nom=nom(b), du=a["du"], au=a["au"],
                                                          confiance=a["confiance"], part=a["part"], force=b["force"]))
    actifs.sort(key=lambda x: (not x["arrivees"], x["distance_djerba_km"] if x["distance_djerba_km"] is not None else 1e9))
    out = dict(generated=dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z", version=C.VERSION,
               horizon_jours=C.BANC_HORIZON_DAYS, actifs=actifs, termines=st["termines"][-15:],
               carnet=(carnet or jload(P_CARNET, {})).get("stats"),
               methode="Chaque observation crédible devient une zone de présence : 160 méduses virtuelles (autant de scénarios). "
                       "Chaque nuit elle avance avec les courants recalculés par Copernicus, puis elle est projetée sur "
                       f"{C.BANC_HORIZON_DAYS} jours. Une observation proche la recale (sans prouver que c'est le même groupe), "
                       "des « rien vu » l'affaiblissent. Les pourcentages sont des parts de scénarios simulés, pas des probabilités.")
    jsave(P_OUT, out)
    jsave(P_STATE, st, indent=None)
    return out, alerts


if __name__ == "__main__":
    # test hors ligne : deux nuits de suite sur données synthétiques
    import synthetic as SY
    import tempfile
    DATA = tempfile.mkdtemp()
    now = np.datetime64(dt.datetime.utcnow().strftime("%Y-%m-%dT00:00"), "m")
    for k in (0, 1, 2):
        t = now + k * DAY
        print(f"===== nuit {k + 1} ({str(t)[:10]})")
        out, alerts = run(t, SY.test_obs(now), SY.test_reports(now), {}, loaders=SY)
        for a in out["actifs"]:
            print(f"  {a['nom']} | {a['position']['lieu']} | étalement {a['etalement_km']} km | {a['tendance']} | {a['message']}")
        for x in out["termines"]:
            print(f"  terminé : {x['nom']} — {x['raison']}")
    print(json.dumps(jload(paths()[2], {}).get("stats"), ensure_ascii=False, indent=1)[:800])
