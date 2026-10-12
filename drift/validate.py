"""
Validation régionale : le modèle met-il les endroits où des méduses ont vraiment été vues
plus haut que le reste de la côte, la veille ?

Méthode :
  1. On place des « points témoins » tous les ~12 km le long de toutes les côtes de la zone
     (côte est tunisienne, Kerkennah, Djerba, Malte, sud de la Sicile, nord-ouest libyen).
  2. On récupère les observations réelles de méduses des 12 derniers mois (iNaturalist, GBIF)
     et on rattache chacune au point de côte le plus proche.
  3. Pour chaque jour observé, on lance le modèle la veille (sans connaître l'observation) et on
     regarde le rang du point observé parmi tous les points de côte ce jour-là (percentile).
  4. On fait pareil pour chaque facteur pris seul → on voit lesquels aident vraiment.

Au hasard, le percentile moyen serait 50 %. Plus il est haut, plus le modèle distingue les bons endroits.
Biais à garder en tête : les gens observent là où il y a des gens (plages touristiques).
"""
import math, datetime as dt
import numpy as np
import config as C
import sources as S
import model as M


def coastal_points(mask_lats, mask_lons, sea):
    """Cases de mer en bord de côte, espacées d'environ VALID_POINT_SPACING_KM, avec leur orientation."""
    ny, nx = sea.shape
    cands = []
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            if not sea[y, x]:
                continue
            nb = ~sea[y - 1:y + 2, x - 1:x + 2]
            if not nb.any():
                continue
            ly, lx = np.nonzero(nb)
            # direction « vers la mer » = depuis le centre des cases de terre voisines vers la case
            vy = -(ly.mean() - 1); vx = -(lx.mean() - 1)
            if vy == 0 and vx == 0:
                continue
            face = (math.degrees(math.atan2(vx, vy)) + 360) % 360
            cands.append((float(mask_lats[y]), float(mask_lons[x]), face))
    rb = C.REGION_BOX
    cands = [c for c in cands if rb["lat_min"] + .15 < c[0] < rb["lat_max"] - .15 and rb["lon_min"] + .15 < c[1] < rb["lon_max"] - .15]
    picked = []
    for la, lo, f in sorted(cands):
        if all(float(M.km_between(la, lo, p[0], p[1])) >= C.VALID_POINT_SPACING_KM for p in picked[-400:]):
            picked.append((la, lo, f))
    return [dict(id=f"p{k}", name=f"point {k}", lat=la, lon=lo, face=round(f), shelter=0, wadi=0)
            for k, (la, lo, f) in enumerate(picked)]


def snap(points, o):
    d = np.array([float(M.km_between(o["lat"], o["lon"], p["lat"], p["lon"])) for p in points])
    k = int(np.argmin(d))
    return (k, float(d[k])) if d[k] <= C.VALID_SNAP_KM else (None, None)


def percentile(values, v):
    values = np.asarray(values, float)
    return float(((values < v).sum() + 0.5 * (values == v).sum() - 0.5) / max(1, len(values) - 1))


def run_validation(predict, gather_obs, jsave, out_path, loaders=None):
    L = loaders or S
    now = dt.datetime.utcnow()
    t_to = np.datetime64(now.strftime("%Y-%m-%dT00:00"), "m") - np.timedelta64(3, "D")
    t_from = t_to - np.timedelta64(C.VALID_DAYS_BACK, "D")
    status = {}
    if loaders is not None and hasattr(loaders, "validation_obs"):
        obs = loaders.validation_obs(t_to)
    else:
        obs, _ = gather_obs(t_from, t_to, status, with_news=False)
    obs = [o for o in obs if not o.get("none_seen")]
    print(f"[validation] {len(obs)} observations sur la zone régionale")

    # --- carte terre/mer de la zone
    try:
        g = M.ensure_ascending(L.load_bathymetry(C.REGION_BOX))
        sea = np.isfinite(g.data["deptho"]) if g.data["deptho"].ndim == 2 else np.isfinite(g.data["deptho"][0])
        lats, lons = g.lats, g.lons
    except Exception as e:
        print(f"[validation] fonds indisponibles ({e}), carte terre/mer depuis les courants")
        t0 = t_to - np.timedelta64(1, "D")
        c = M.ensure_ascending(L.load_currents_surface(C.REGION_BOX, t0.astype("datetime64[s]").astype(dt.datetime),
                                                       t_to.astype("datetime64[s]").astype(dt.datetime)))
        sea = np.isfinite(c.data["uo"][0]); lats, lons = c.lats, c.lons
    points = coastal_points(lats, lons, sea)
    print(f"[validation] {len(points)} points témoins le long des côtes")

    # --- rattacher chaque observation à un point de côte
    events = []
    for o in obs:
        k, d = snap(points, o)
        if k is None:
            continue
        events.append(dict(date=M.local_date(o["time"]), point=points[k]["id"], km=round(d, 1), lat=o["lat"], lon=o["lon"],
                           taxon=o.get("taxon", "?"), source=o["source"], ref=o.get("ref", "")))
    # un seul événement par point et par jour
    uniq = {}
    for e in events:
        uniq.setdefault((e["date"], e["point"]), e)
    events = sorted(uniq.values(), key=lambda e: e["date"], reverse=True)
    print(f"[validation] {len(events)} événements (jour × point) utilisables")

    # --- regrouper les jours : une simulation lancée la veille couvre le jour J et J+1
    dates = sorted({e["date"] for e in events}, reverse=True)
    runs, covered = [], set()
    for d in dates:
        if d in covered:
            continue
        D = dt.date.fromisoformat(d)
        start = D - dt.timedelta(days=1)              # on regarde J-1 et J (prévision à 1–2 jours)
        covers = {(start + dt.timedelta(days=1)).isoformat(), D.isoformat()}
        covers.add((D - dt.timedelta(days=1)).isoformat())
        runs.append((start, sorted(covers)))
        covered |= covers
        if len(runs) >= C.VALID_MAX_RUNS:
            break

    rows, all_risks = [], []
    by_point = {}          # (point, facteur) -> {jour: valeur} pour la comparaison dans le temps
    factors = ("risk", "drift", "obs", "wind_past", "habitat", "exposure", "arrivals")
    C.VALID_MAX_RUNS = max(C.VALID_MAX_RUNS, 30)
    for start, covers in runs:
        t_now = np.datetime64(start.isoformat() + "T00:00", "m") - np.timedelta64(1, "D")
        past = [o for o in obs if o["time"] < t_now]
        st = {}
        try:
            res, _ = predict(t_now, past, st, historical=True, loaders=loaders, beaches=points, box=C.REGION_BOX,
                             spacing=C.VALID_SPACING, lookback_days=C.VALID_LOOKBACK_DAYS)
        except Exception as e:
            print(f"[validation] simulation du {start} échouée : {str(e)[:150]}")
            continue
        days = res["days"]
        for di, day in enumerate(days):
            if di == 0:
                continue                               # jour 0 = le jour où l'on lance : pas une prévision
            vals = {f: [] for f in factors}
            ids = []
            for p in points:
                r = res["beaches"][p["id"]]["days"][di]
                ids.append(p["id"])
                vals["risk"].append(r["risk"])
                for f in factors[1:]:
                    vals[f].append(r["factors"].get(f, np.nan) if f != "habitat" else r["factors"].get("habitat", np.nan))
            all_risks += vals["risk"]
            ev_points = {e["point"] for e in events}
            for i, pid in enumerate(ids):
                if pid in ev_points:
                    for f in factors:
                        by_point.setdefault((pid, f), {})[day] = vals[f][i]
            for e in [e for e in events if e["date"] == day]:
                i = ids.index(e["point"])
                row = dict(e, lead_days=di, level=res["beaches"][e["point"]]["days"][di]["level"], mode=res["mode"])
                for f in factors:
                    arr = np.array(vals[f], float)
                    if np.isnan(arr[i]):
                        continue
                    row["pct_" + f] = round(percentile(arr[~np.isnan(arr)], arr[i]), 3)
                rows.append(row)
        print(f"[validation] simulation du {start} : {sum(1 for r in rows if r['date'] in covers)} événements notés")

    # --- comparaison dans le temps : le jour observé ressort-il par rapport aux autres jours simulés au même endroit ?
    for r in rows:
        for f in factors:
            series = by_point.get((r["point"], f), {})
            others = [v for d, v in series.items() if d != r["date"] and v == v]
            if len(others) >= 5 and r["date"] in series and series[r["date"]] == series[r["date"]]:
                r["tpct_" + f] = round(percentile(others + [series[r["date"]]], series[r["date"]]), 3)

    # --- résumé
    def mean(key):
        v = [r[key] for r in rows if key in r]
        return round(float(np.mean(v)), 3) if v else None
    summary = dict(
        evenements=len(rows),
        percentile_moyen_risque=mean("pct_risk"),
        part_dans_le_quart_superieur=round(float(np.mean([r["pct_risk"] >= 0.75 for r in rows])), 3) if rows else None,
        percentile_moyen_par_facteur={f: mean("pct_" + f) for f in factors[1:]},
        dans_le_temps=dict(
            evenements=sum(1 for r in rows if "tpct_risk" in r),
            percentile_moyen_risque=mean("tpct_risk"),
            par_facteur={f: mean("tpct_" + f) for f in factors[1:]},
            lecture="Même endroit, autres jours : le jour où des méduses ont été vues avait-il un risque plus haut que d'habitude ?"),
        niveaux_aux_points_observes={lv: sum(r["level"] == lv for r in rows) for lv in ("Faible", "Possible", "Élevé", "Signalé")},
        lecture="50 % = pas mieux que le hasard. 60–65 % = utile. Au-delà de 70 % = très bon. "
                "Un facteur proche de 50 % n'aide pas : baisser son poids dans config.py.",
    )
    if all_risks:
        q = np.quantile(all_risks, [0.5, 0.8, 0.95])
        summary["seuils_suggeres"] = dict(
            possible=round(float(q[1]), 3), eleve=round(float(q[2]), 3),
            explication="Seuils pour que ~20 % des côtes soient « Possible ou plus » et ~5 % « Élevé » (à reporter dans LEVELS de config.py).")
    jsave(out_path, dict(generated=now.replace(microsecond=0).isoformat() + "Z", version=C.VERSION,
                         points=len(points), summary=summary, events=rows[:400]))
    print("[validation] résumé :")
    for k, v in summary.items():
        print(f"    {k} : {v}")
