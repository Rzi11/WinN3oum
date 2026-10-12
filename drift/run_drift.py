"""
Win N3oum ? — modèle méduses v0.2 (lancé chaque nuit par GitHub Actions).

  python drift/run_drift.py            prévision normale (aujourd'hui + 2 jours)
  python drift/run_drift.py --replay   rejoue le passé sur les observations réelles de Djerba (vérification)
  python drift/run_drift.py --validate validation sur ~100 observations de toute la région (points témoins)
  python drift/run_drift.py --test     test hors ligne (ajouter --replay ou --validate pour tester ces modes)

Le risque combine :
  dérive physique (courants surface + 15 m, vent, vagues, cycle jour/nuit de la pélagie, marée)
  × eau favorable (température, salinité, chlorophylle, couche de surface)
  × exposition de la plage (ouverture, profondeur, abri, oued)
  + observations récentes (iNaturalist, GBIF, signalements, presse) comme points de départ et comme mémoire
  + vent vers la plage des 3 derniers jours, saison, vent fort.
C'est un indicateur, pas une prévision validée : les poids sont dans config.py et se règlent avec le rejeu.
"""
import os, sys, json, math, datetime as dt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import sources as S
import model as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
P_OUT = os.path.join(DATA, "meduses.json")
P_HIST = os.path.join(DATA, "historique.json")
P_VERIF = os.path.join(DATA, "verification.json")
P_REPLAY = os.path.join(DATA, "rejeu.json")
P_REPORTS = os.path.join(DATA, "signalements.json")


def jload(p, default):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def jsave(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def to_dt(t64):
    return t64.astype("datetime64[s]").astype(dt.datetime)


# ====================================================================== observations
def gather_obs(t_from, t_to, status, with_news=True, box=None):
    box = box or C.REGION_BOX
    d1, d2 = str(t_from)[:10], str(t_to)[:10]
    obs = []
    for name, fn in (("iNaturalist", lambda: S.load_inaturalist(box, d1, d2)),
                     ("GBIF", lambda: S.load_gbif(box, d1, d2)),
                     ("OBIS", lambda: S.load_obis(box, d1, d2)),
                     ("signalements", lambda: S.load_reports(P_REPORTS, t_from))):
        try:
            got = fn(); obs += got; status[name] = f"ok ({len(got)})"
        except Exception as e:
            status[name] = f"indisponible : {str(e)[:120]}"
    articles = []
    if with_news:
        try:
            got, articles = S.load_news(C.NEWS_PLACES, C.NEWS_QUERIES)
            obs += got; status["presse"] = f"ok ({len(got)})"
        except Exception as e:
            status["presse"] = f"indisponible : {str(e)[:120]}"
    obs = [o for o in obs if t_from <= o["time"] <= t_to]
    return S.dedupe(obs), articles


# ====================================================================== prévision
def predict(t_now, obs, status, history=None, historical=False, loaders=None,
            beaches=None, box=None, spacing=None, lookback_days=None, region=None):
    """Calcule le risque par plage pour 3 jours à partir de t_now (00:00 UTC).
    box = zone des plages et des méduses de fond ; region = zone de dérive (courants, vent, observations)."""
    L = loaders or S
    region = region or C.REGION_BOX
    beaches = beaches or C.BEACHES
    box = box or C.LOCAL_BOX
    lookback_days = lookback_days or C.LOOKBACK_DAYS
    t_end = t_now + np.timedelta64(C.FORECAST_HOURS, "h")
    seeds_obs = [o for o in obs if not o.get("none_seen") and o["weight"] > 0 and
                 t_now - np.timedelta64(lookback_days, "D") <= o["time"] < t_now]
    t_start = min([o["time"] for o in seeds_obs] + [t_now]) - np.timedelta64(3, "h")
    s_dt, e_dt = to_dt(t_start), to_dt(t_end + np.timedelta64(1, "h"))
    days = [M.local_date(t_now + np.timedelta64(h, "h")) for h in (2, 26, 50)]

    # --- vent (obligatoire)
    if historical:
        wind = L.load_wind(region, 4, start_date=str(t_start - np.timedelta64(3, "D"))[:10], end_date=str(t_end)[:10])
    else:
        past = int((t_now - t_start) / np.timedelta64(1, "D")) + 4
        wind = L.load_wind(region, 4, past_days=min(92, past), forecast_days=4)
    status["vent"] = "ok"

    # --- courants de surface (obligatoire pour la dérive, sinon mode secours)
    mode = "derive"
    try:
        cur_s = M.ensure_ascending(L.load_currents_surface(region, s_dt, e_dt)); status["courants surface"] = cur_s.name
    except Exception as e:
        cur_s = None; mode = "vent"; status["courants surface"] = f"indisponible : {str(e)[:150]}"

    def opt(key, fn):
        try:
            g = fn(); status[key] = getattr(g, "name", "ok"); return g
        except Exception as e:
            status[key] = f"indisponible : {str(e)[:120]}"; return None

    cur_d = stokes = None
    if cur_s is not None:
        cur_d = opt("courants 15 m", lambda: M.ensure_ascending(L.load_currents_depth(region, s_dt, e_dt)))
        stokes = opt("vagues", lambda: M.ensure_ascending(L.load_stokes(region, s_dt, e_dt)))
    sea_level = opt("marée", lambda: M.ensure_ascending(L.load_sea_level(box, to_dt(t_now - np.timedelta64(1, "D")), e_dt)))
    hab = {}
    try:
        hab = {k: M.ensure_ascending(g) for k, g in L.load_habitat(box, to_dt(t_now - np.timedelta64(1, "D")), e_dt).items()}
    except Exception as e:
        print(f"[habitat] {e}")
    status["eau favorable"] = ", ".join(hab) if hab else "indisponible"
    bathy = opt("fonds marins", lambda: M.ensure_ascending(L.load_bathymetry(box)))

    # --- dérive
    arrivals, n_bg, n_obs = {}, 0, 0
    exposure = {}
    if cur_s is not None:
        drift = M.Drift(cur_s, cur_d, stokes, wind, sea_level, beaches, region)
        seeds, n_bg, n_obs = M.make_seeds(drift, hab, obs, t_now, box, spacing, lookback_days)
        arrivals = drift.run(t_now, t_end, seeds)
        k = cur_s.tindex(t_now)
        mask = (cur_s.lats, cur_s.lons, ~np.isnan(cur_s.data["uo"][k]))
        for b in beaches:
            exposure[b["id"]] = M.beach_exposure(mask, bathy, b)
    else:
        for b in beaches:
            exposure[b["id"]] = (1.0 + 0.15 * b.get("shelter", 0) - 0.15 * b.get("wadi", 0), {})

    # --- combinaison
    out_beaches, raw_today = {}, {}
    for bi, b in enumerate(beaches):
        rows = []
        hist = (history or {}).get(b["id"], {})
        hvals = [v for d, v in sorted(hist.items()) if d < days[0]][-C.HISTORY_DAYS:]
        for d in days:
            t_day = np.datetime64(d + "T00:00", "m") - np.timedelta64(C.UTC_OFFSET, "h")
            a = arrivals.get((b["id"], d), {"all": 0.0, "obs": 0.0})
            raw = a["all"]
            if mode == "vent":
                push_day, _ = M.onshore_wind_past(wind, b, t_day + np.timedelta64(24, "h"), hours=24)
                raw = push_day * 6
            if d == days[0]:
                raw_today[b["id"]] = raw
            if len(hvals) >= C.HISTORY_MIN_RUNS:
                med = float(np.median(hvals))
                drift_f = 1 - math.exp(-(raw / (med + 0.5)) / 2)
                drift_ref = f"comparé à sa médiane ({med:.1f})"
            else:
                drift_f = 1 - math.exp(-raw / C.K_ARRIVALS)
                drift_ref = "échelle fixe (historique en construction)"
            obs_f, signal = M.obs_persistence(obs, b, t_day)
            push, _ = M.onshore_wind_past(wind, b, t_day, 72)
            wind_f = float(np.clip(push / 5, 0, 1))
            news_f = M.news_factor(obs, b, t_day)
            hab_f, hab_det = M.habitat_scores(hab, b["lat"], b["lon"], t_day + np.timedelta64(12, "h")) if hab else (None, {})
            fac = dict(drift=drift_f, obs=obs_f, wind_past=wind_f, calm=1 - wind_f, news=news_f)
            if hab_f is not None:
                fac["habitat"] = hab_f
            ws = {k: v for k, v in C.W.items() if k in fac}
            P = sum(ws[k] * fac[k] for k in ws) / sum(ws.values())
            # un banc qui arrive depuis une observation réelle pèse plus lourd que la seule physique
            if a["obs"] > 0:
                P = max(P, min(1.0, 0.35 + 0.5 * (1 - math.exp(-a["obs"] / 5))))
            season = C.SEASON[int(d[5:7])]
            E, E_det = exposure[b["id"]]
            wspd = M.day_wind_speed(wind, b, d)
            strong = 0.8 if wspd > C.STRONG_WIND_KMH else 1.0
            moon = M.moon_age_days(t_day + np.timedelta64(12, "h"))
            lunar = 1 + C.LUNAR_WEIGHT * math.cos(2 * math.pi * (moon - 23) / 29.53)
            risk = float(np.clip(season * (C.BASE_PRESENCE + (1 - C.BASE_PRESENCE) * P) * E * strong * lunar, 0, 1))
            cal = calibration()
            if cal:
                # poids appris sur toute la Méditerranée (calibrate.py) — l'effort d'observation est neutralisé
                x = dict(drift=drift_f, obs=obs_f, calm=1 - wind_f, habitat=hab_f if hab_f is not None else 0.5,
                         exposure=E, log_arrivals=math.log1p(max(0.0, raw)), from_obs=math.log1p(max(0.0, a["obs"])),
                         season=season, effort=cal["effort_mean"])
                vals = [x[k] for k in cal["features"]]
                if cal.get("feat_min"):                          # pas d'extrapolation hors des données apprises
                    vals = [min(max(v, a_), b_) for v, a_, b_ in zip(vals, cal["feat_min"], cal["feat_max"])]
                z = cal["intercept"] + sum(c * v for c, v in zip(cal["coef"], vals))
                pcal = 1 / (1 + math.exp(-max(-30, min(30, z))))
                # rang parmi toutes les situations apprises → même échelle que le risque manuel
                # (80e percentile = seuil « Possible », 95e = « Élevé »)
                pct = float(np.interp(pcal, cal["quantiles"], np.linspace(0, 1, len(cal["quantiles"])))) if cal.get("quantiles") else 0.5
                risk = float(np.interp(pct, [0, 0.8, 0.95, 1.0], [0.03, C.LEVELS[0], C.LEVELS[1], 0.9]))
            lv = levels()
            level = "Faible" if risk < lv[0] else "Possible" if risk < lv[1] else "Élevé"
            if signal and d in days[:2]:
                level = "Signalé"
            why = explain(fac, a, E, strong, season, mode)
            # v0.8 : on sépare ce qui a été OBSERVÉ, ce que la physique SUGGÈRE, et la CONFIANCE qu'on peut avoir
            seen_txt = observed_text(obs, b, t_day)
            conf = "observé" if seen_txt else confidence(obs, b, t_now, mode)
            rows.append(dict(date=d, level=level, risk=round(risk, 2), why=why, observe=seen_txt,
                             transport=transport_parts(fac, a, E, strong, season, mode), confiance=conf,
                             factors=dict({k: round(v, 2) for k, v in fac.items()}, arrivals=round(raw, 1),
                                          from_obs=round(a["obs"], 1), drift_ref=drift_ref, exposure=round(E, 2),
                                          season=season, strong_wind=strong < 1, wind_kmh=round(wspd),
                                          moon_age=round(moon, 1), habitat_detail=hab_det)))
        sst = rows[0]["factors"]["habitat_detail"].get("sst") if rows else None
        out_beaches[b["id"]] = dict(name=b["name"], exposure=exposure[b["id"]][1], days=rows,
                                    species=M.likely_species(int(days[0][5:7]), sst))
    return dict(mode=mode, days=days, beaches=out_beaches, particles=dict(fond=n_bg, observations=n_obs)), raw_today


_LEVELS = None
_CAL = "?"


def calibration():
    """Poids appris (data/calibration.json) s'ils ont été adoptés, sinon None."""
    global _CAL
    if _CAL == "?":
        c = jload(os.path.join(DATA, "calibration.json"), {})
        _CAL = c if c.get("adopted") and c.get("coef") else None
        if _CAL:
            print(f"[calibrage] poids appris utilisés (AUC {c.get('auc_cv')} vs manuel {c.get('auc_manual')}, {c.get('n_pos')} observations)")
    return _CAL


def levels():
    """Seuils issus de la dernière validation de la même version du modèle, sinon ceux de config.py."""
    global _LEVELS
    if _LEVELS is None and calibration():
        _LEVELS = C.LEVELS                                   # le risque calibré est déjà ramené sur cette échelle
    if _LEVELS is None:
        v = jload(os.path.join(DATA, "validation.json"), {})
        sg = (v.get("summary") or {}).get("seuils_suggeres")
        if v.get("version") == C.VERSION and sg and 0 < sg["possible"] < sg["eleve"] < 1:
            _LEVELS = (sg["possible"], sg["eleve"])
            print(f"[seuils] repris de la validation : {_LEVELS}")
        else:
            _LEVELS = C.LEVELS
    return _LEVELS


def observed_text(obs, b, t_day):
    """Observation scientifique (iNaturalist, GBIF, OBIS) à < 10 km dans les 72 h avant la fin du jour, en clair."""
    import bancs as BA
    best = None
    t_end = t_day + np.timedelta64(24, "h")
    for o in obs:
        if o["source"] not in ("iNaturalist", "GBIF", "OBIS") or o.get("none_seen"):
            continue
        age_h = (t_end - o["time"]) / np.timedelta64(1, "h")
        if age_h < 0 or age_h > 72:
            continue
        d = float(M.km_between(b["lat"], b["lon"], o["lat"], o["lon"]))
        if d > 10:
            continue
        if best is None or o["time"] > best[0]["time"]:
            best = (o, d)
    if best is None:
        return None
    o, d = best
    day = M.local_date(o["time"])
    return f"{BA.species_fr(o.get('taxon'))} vues à {d:.0f} km le {day[8:10]}/{day[5:7]} ({o['source']})"


def confidence(obs, b, t_now, mode):
    """« insuffisant » s'il n'y a ni courants ni aucune information de terrain (vu ou « rien vu ») à < 60 km en 14 jours."""
    if mode == "vent":
        return "insuffisant"
    for o in obs:
        if o["source"] == "presse":
            continue
        age_d = (t_now - o["time"]) / np.timedelta64(1, "D")
        if -1 <= age_d <= 14 and float(M.km_between(b["lat"], b["lon"], o["lat"], o["lon"])) <= 60:
            return "estimation"
    return "insuffisant"


def transport_parts(f, a, E, strong, season, mode):
    """Ce que suggèrent les courants, le vent et l'eau (sans les observations)."""
    p = []
    if a["obs"] > 0:
        p.append("des méduses vues en mer pourraient dériver jusqu'ici")
    if mode == "vent":
        p.append("estimation au vent seul (courants indisponibles)")
    elif f["drift"] >= 0.5:
        p.append("courant et vent poussent vers la plage")
    elif f["drift"] >= 0.2:
        p.append("un peu de dérive vers la plage")
    else:
        p.append("peu de dérive vers la plage")
    if f.get("habitat", 0.5) >= 0.8:
        p.append("eau favorable aux méduses")
    if E >= 1.15:
        p.append("plage qui retient ce qui flotte")
    if strong < 1:
        p.append("vent fort : méduses brassées en profondeur")
    if season <= 0.3:
        p.append("hors saison")
    return p[:3]


def explain(f, a, E, strong, season, mode):
    parts = []
    if a["obs"] > 0:
        parts.append("un banc observé récemment dérive vers la plage")
    if f["obs"] > 0.4:
        parts.append("méduses signalées tout près ces derniers jours")
    if mode == "vent":
        parts.append("estimation au vent seul")
    elif f["drift"] >= 0.5:
        parts.append("courant et vent poussent vers la plage")
    elif f["drift"] >= 0.2:
        parts.append("un peu de dérive vers la plage")
    else:
        parts.append("peu de dérive vers la plage")
    if f["wind_past"] <= 0.1:
        parts.append("mer calme ces derniers jours")
    if f["news"] > 0:
        parts.append("méduses dans la presse locale")
    if f.get("habitat", 0.5) >= 0.8:
        parts.append("eau favorable aux méduses")
    if E >= 1.15:
        parts.append("plage qui piège les méduses")
    if strong < 1:
        parts.append("vent fort : méduses brassées en profondeur")
    if season <= 0.3:
        parts.append("hors saison")
    return parts[:3]


# ====================================================================== historique & vérification
def update_history(raw_today, day):
    h = jload(P_HIST, {})
    for bid, v in raw_today.items():
        h.setdefault(bid, {})[day] = round(v, 2)
        keep = sorted(h[bid])[-C.HISTORY_DAYS:]
        h[bid] = {k: h[bid][k] for k in keep}
    jsave(P_HIST, h)
    return h


def update_verification(out, obs, t_now):
    v = jload(P_VERIF, {"predictions": {}, "observations": [], "stats": {}})
    run = str(t_now)[:10]
    for bid, b in out["beaches"].items():
        for r in b["days"][:2]:
            v["predictions"][f"{r['date']}|{bid}"] = dict(level=r["level"], risk=r["risk"], run=run)
    refs = {o["ref"] for o in v["observations"]}
    lb = C.LOCAL_BOX
    for o in obs:
        if o["source"] == "presse" or o["ref"] in refs:
            continue
        if not (lb["lat_min"] - .05 <= o["lat"] <= lb["lat_max"] + .05 and lb["lon_min"] - .05 <= o["lon"] <= lb["lon_max"] + .05):
            continue
        d = [float(M.km_between(o["lat"], o["lon"], b["lat"], b["lon"])) for b in C.BEACHES]
        k = int(np.argmin(d))
        if d[k] > 8:
            continue
        v["observations"].append(dict(date=M.local_date(o["time"]), beach=C.BEACHES[k]["id"], km=round(d[k], 1),
                                      source=o["source"], taxon=o.get("taxon", "?"), none_seen=bool(o.get("none_seen")), ref=o["ref"]))
    st = {"Faible": 0, "Possible": 0, "Élevé": 0, "Signalé": 0, "sans prévision": 0}
    for ob in v["observations"]:
        if ob["none_seen"]:
            continue
        p = v["predictions"].get(f"{ob['date']}|{ob['beach']}")
        st[p["level"] if p else "sans prévision"] += 1
    v["stats"] = dict(observations_avec_meduses=st, note="Observations réelles comparées au niveau prévu ce jour-là sur la plage la plus proche.")
    v["predictions"] = dict(sorted(v["predictions"].items())[-2000:])
    jsave(P_VERIF, v)


# ====================================================================== modes
def run_normal(loaders=None, test=False):
    now = dt.datetime.utcnow()
    t_now = np.datetime64(now.strftime("%Y-%m-%dT00:00"), "m")
    status = {}
    if test:
        obs, articles = loaders.test_obs(t_now), []
        status["observations"] = "synthétiques"
    else:
        obs, articles = gather_obs(t_now - np.timedelta64(C.LOOKBACK_DAYS, "D"), t_now + np.timedelta64(1, "D"), status)
    # --- signalements des utilisateurs (Supabase) : points de départ + fusion pondérée par la crédibilité
    import credibility as CR
    reports = []
    if test and hasattr(loaders, "test_reports"):
        reports = loaders.test_reports(t_now)
    elif CR.enabled():
        try:
            reports = CR.fetch_reports(C.LOOKBACK_DAYS)
        except Exception as e:
            status["signalements app"] = f"indisponible : {str(e)[:120]}"
    if reports:
        status["signalements app"] = f"ok ({len(reports)})"
        obs = obs + CR.as_seed_obs(reports, C.BEACHES)
    elif "signalements app" not in status:
        status["signalements app"] = "non configuré" if not CR.enabled() else "ok (0)"
    history = jload(P_HIST, {})
    res, raw_today = predict(t_now, obs, status, history=history, loaders=loaders)
    science = [o for o in obs if o["source"] in ("iNaturalist", "GBIF", "OBIS")]
    if reports:
        res = CR.fuse(res, reports, science, C.BEACHES, levels())
    rep_stats = None
    if reports and CR.enabled() and not test:
        try:
            rep_stats = CR.update_reputation(reports, science, res, C.BEACHES)
        except Exception as e:
            print(f"[réputation] échec : {e}")
    # --- registre des bancs suivis (v0.7) : avancée réelle + projection 8 jours + carnet de bord
    n_bancs = None
    try:
        import bancs as BA
        bout, alerts = BA.run(t_now, obs, reports, status, loaders)
        if bout is not None:
            n_bancs = len(bout["actifs"])
            apply_banc_alerts(res, alerts)
    except Exception as e:
        status["bancs"] = f"erreur : {str(e)[:120]}"
        print(f"[bancs] erreur : {e}")
    cal = calibration()
    out = dict(generated=now.replace(microsecond=0).isoformat() + "Z", version=C.VERSION, mode=res["mode"],
               calibration=(dict(auc=cal["auc_cv"], manual=cal["auc_manual"], n_pos=cal["n_pos"]) if cal else None),
               method="Dérive (courants surface + 15 m, vent, vagues, jour/nuit, marée) × eau favorable × exposition + observations, presse, vent passé, saison. Indicateur non validé.",
               sources=status, particles=res["particles"], days=res["days"], beaches=res["beaches"],
               observations=summarize_obs(obs), news=articles[:10], reputation=rep_stats, bancs_suivis=n_bancs)
    jsave(P_OUT, out)
    if not test:
        update_history(raw_today, res["days"][0])
        update_verification(out, obs, t_now)
    print_table(out)


def apply_banc_alerts(res, alerts):
    """Alertes lointaines des bancs suivis → ajoutées aux plages ; pour les 3 jours affichés, un banc en approche
    (confiance moyenne ou forte, banc encore « fort ») fait passer au moins à « Possible »."""
    for bid, lst in alerts.items():
        b = res["beaches"].get(bid)
        if b is None:
            continue
        b["bancs"] = lst
        for r in b["days"]:
            for a in lst:
                if a["du"] <= r["date"] <= a["au"] and a["confiance"] in ("moyenne", "forte") and a["force"] >= 0.4:
                    r["factors"]["banc"] = a["banc"]
                    if r["level"] == "Faible":
                        r["level"] = "Possible"
                        r["risk"] = max(r["risk"], levels()[0])
                    msg = f"zone de présence suivie n°{a['banc']} : des scénarios arrivent ici"
                    if msg not in r["why"]:
                        r["why"] = [msg] + r["why"][:2]
                    break


def run_replay(loaders=None, max_events=8):
    """Rejoue la veille de chaque observation réelle de méduses près de Djerba et regarde ce que le modèle aurait dit."""
    now = dt.datetime.utcnow()
    t_to = np.datetime64(now.strftime("%Y-%m-%dT00:00"), "m") - np.timedelta64(3, "D")
    t_from = t_to - np.timedelta64(540, "D")
    status = {}
    lb = C.LOCAL_BOX
    if loaders is not None and hasattr(loaders, "replay_obs"):
        all_obs = loaders.replay_obs(t_to)
    else:
        all_obs, _ = gather_obs(t_from, t_to, status, with_news=False)
    local = [o for o in all_obs if lb["lat_min"] - .05 <= o["lat"] <= lb["lat_max"] + .05 and lb["lon_min"] - .05 <= o["lon"] <= lb["lon_max"] + .05]
    by_day = {}
    for o in sorted(local, key=lambda x: x["time"], reverse=True):
        by_day.setdefault(M.local_date(o["time"]), o)
    events = list(by_day.items())[:max_events]
    print(f"[rejeu] {len(local)} observations près de Djerba, {len(events)} jours rejoués")
    results = []
    for day, o in events:
        t_now = np.datetime64(day + "T00:00", "m") - np.timedelta64(1, "D")
        past_obs = [x for x in all_obs if x["time"] < t_now]
        st = {}
        try:
            res, _ = predict(t_now, past_obs, st, historical=True, loaders=loaders)
        except Exception as e:
            print(f"  {day} : échec {e}")
            results.append(dict(date=day, error=str(e)[:200])); continue
        d = [float(M.km_between(o["lat"], o["lon"], b["lat"], b["lon"])) for b in C.BEACHES]
        k = int(np.argmin(d)); bid = C.BEACHES[k]["id"]
        levels = {x: res["beaches"][x]["days"][1] for x in res["beaches"]}
        ranking = sorted(levels, key=lambda x: -levels[x]["risk"])
        results.append(dict(date=day, observed_beach=bid, km=round(d[k], 1), taxon=o.get("taxon"), source=o["source"],
                            predicted=levels[bid]["level"], risk=levels[bid]["risk"], rank=ranking.index(bid) + 1,
                            of=len(ranking), all={x: levels[x]["level"] for x in ranking}, mode=res["mode"], sources=st))
        print(f"  {day} {bid:<11} prévu {levels[bid]['level']:<8} rang {ranking.index(bid) + 1}/{len(ranking)} ({res['mode']})")
    ok = [r for r in results if "predicted" in r]
    summary = dict(events=len(ok),
                   possible_ou_plus=sum(r["predicted"] != "Faible" for r in ok),
                   rang_moyen=round(float(np.mean([r["rank"] for r in ok])), 2) if ok else None,
                   note=f"Rang 1 = la plage observée était celle que le modèle jugeait la plus à risque. Au hasard, le rang moyen serait {(len(C.BEACHES) + 1) / 2:g} sur {len(C.BEACHES)}.")
    jsave(P_REPLAY, dict(generated=now.replace(microsecond=0).isoformat() + "Z", version=C.VERSION, summary=summary, results=results))
    print(f"[rejeu] résumé : {json.dumps(summary, ensure_ascii=False)}")


def summarize_obs(obs):
    by = {}
    for o in obs:
        by[o["source"]] = by.get(o["source"], 0) + 1
    recent = sorted(obs, key=lambda o: o["time"], reverse=True)[:15]
    return dict(count=by, recent=[dict(date=str(o["time"])[:10], lat=round(o["lat"], 3), lon=round(o["lon"], 3),
                                       source=o["source"], taxon=o.get("taxon", "?")) for o in recent])


def print_table(out):
    print(f"[sortie] mode {out['mode']} — sources :")
    for k, v in out["sources"].items():
        print(f"    {k:<18} {v}")
    for bid, b in out["beaches"].items():
        print(f"  {b['name']:<15} " + "  ".join(
            f"{r['date'][5:]}:{r['level']:<8}({r['risk']:.2f})" for r in b["days"]) + f"  | {', '.join(b['days'][1]['why'])}")


P_VALID = os.path.join(DATA, "validation.json")


if __name__ == "__main__":
    test = "--test" in sys.argv
    loaders = None
    if test:
        import synthetic as loaders
    if "--validate" in sys.argv:
        import validate
        validate.run_validation(predict, gather_obs, jsave, P_VALID, loaders=loaders)
    elif "--replay" in sys.argv:
        run_replay(loaders=loaders, max_events=2 if test else 8)
    else:
        run_normal(loaders=loaders, test=test)
