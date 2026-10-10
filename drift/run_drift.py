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
def gather_obs(t_from, t_to, status, with_news=True):
    d1, d2 = str(t_from)[:10], str(t_to)[:10]
    obs = []
    for name, fn in (("iNaturalist", lambda: S.load_inaturalist(C.REGION_BOX, d1, d2)),
                     ("GBIF", lambda: S.load_gbif(C.REGION_BOX, d1, d2)),
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
            beaches=None, box=None, spacing=None, lookback_days=None):
    """Calcule le risque par plage pour 3 jours à partir de t_now (00:00 UTC)."""
    L = loaders or S
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
        wind = L.load_wind(C.REGION_BOX, 4, start_date=str(t_start - np.timedelta64(3, "D"))[:10], end_date=str(t_end)[:10])
    else:
        past = int((t_now - t_start) / np.timedelta64(1, "D")) + 4
        wind = L.load_wind(C.REGION_BOX, 4, past_days=min(92, past), forecast_days=4)
    status["vent"] = "ok"

    # --- courants de surface (obligatoire pour la dérive, sinon mode secours)
    mode = "derive"
    try:
        cur_s = M.ensure_ascending(L.load_currents_surface(C.REGION_BOX, s_dt, e_dt)); status["courants surface"] = cur_s.name
    except Exception as e:
        cur_s = None; mode = "vent"; status["courants surface"] = f"indisponible : {str(e)[:150]}"

    def opt(key, fn):
        try:
            g = fn(); status[key] = getattr(g, "name", "ok"); return g
        except Exception as e:
            status[key] = f"indisponible : {str(e)[:120]}"; return None

    cur_d = stokes = None
    if cur_s is not None:
        cur_d = opt("courants 15 m", lambda: M.ensure_ascending(L.load_currents_depth(C.REGION_BOX, s_dt, e_dt)))
        stokes = opt("vagues", lambda: M.ensure_ascending(L.load_stokes(C.REGION_BOX, s_dt, e_dt)))
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
        drift = M.Drift(cur_s, cur_d, stokes, wind, sea_level, beaches)
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
            ws = dict(C.W)
            fac = dict(drift=drift_f, obs=obs_f, wind_past=wind_f, news=news_f)
            P = sum(ws[k] * fac[k] for k in ws) / sum(ws.values())
            hm = 1.0
            if hab_f is not None:
                fac["habitat"] = hab_f
                hm = C.HABITAT_MULT[0] + (C.HABITAT_MULT[1] - C.HABITAT_MULT[0]) * hab_f
            # un banc qui arrive depuis une observation réelle pèse plus lourd que la seule physique
            if a["obs"] > 0:
                P = max(P, min(1.0, 0.35 + 0.5 * (1 - math.exp(-a["obs"] / 5))))
            season = C.SEASON[int(d[5:7])]
            E, E_det = exposure[b["id"]]
            wspd = M.day_wind_speed(wind, b, d)
            strong = 0.8 if wspd > C.STRONG_WIND_KMH else 1.0
            moon = M.moon_age_days(t_day + np.timedelta64(12, "h"))
            lunar = 1 + C.LUNAR_WEIGHT * math.cos(2 * math.pi * (moon - 23) / 29.53)
            risk = float(np.clip(season * (C.BASE_PRESENCE + (1 - C.BASE_PRESENCE) * P) * E * hm * strong * lunar, 0, 1))
            level = "Faible" if risk < C.LEVELS[0] else "Possible" if risk < C.LEVELS[1] else "Élevé"
            if signal and d in days[:2]:
                level = "Signalé"
            why = explain(fac, a, E, strong, season, mode)
            rows.append(dict(date=d, level=level, risk=round(risk, 2), why=why,
                             factors=dict({k: round(v, 2) for k, v in fac.items()}, arrivals=round(raw, 1),
                                          from_obs=round(a["obs"], 1), drift_ref=drift_ref, exposure=round(E, 2),
                                          season=season, strong_wind=strong < 1, wind_kmh=round(wspd),
                                          moon_age=round(moon, 1), habitat_detail=hab_det)))
        sst = rows[0]["factors"]["habitat_detail"].get("sst") if rows else None
        out_beaches[b["id"]] = dict(name=b["name"], exposure=exposure[b["id"]][1], days=rows,
                                    species=M.likely_species(int(days[0][5:7]), sst))
    return dict(mode=mode, days=days, beaches=out_beaches, particles=dict(fond=n_bg, observations=n_obs)), raw_today


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
    if f["wind_past"] >= 0.5:
        parts.append("vent de mer ces 3 derniers jours")
    if f["news"] > 0:
        parts.append("méduses dans la presse locale")
    if f.get("habitat", 0.5) >= 0.85 and f["drift"] >= 0.2:
        parts.append("eau très favorable")
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
    history = jload(P_HIST, {})
    res, raw_today = predict(t_now, obs, status, history=history, loaders=loaders)
    out = dict(generated=now.replace(microsecond=0).isoformat() + "Z", version=C.VERSION, mode=res["mode"],
               method="Dérive (courants surface + 15 m, vent, vagues, jour/nuit, marée) × eau favorable × exposition + observations, presse, vent passé, saison. Indicateur non validé.",
               sources=status, particles=res["particles"], days=res["days"], beaches=res["beaches"],
               observations=summarize_obs(obs), news=articles[:10])
    jsave(P_OUT, out)
    if not test:
        update_history(raw_today, res["days"][0])
        update_verification(out, obs, t_now)
    print_table(out)


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
                   note="Rang 1 = la plage observée était celle que le modèle jugeait la plus à risque. Au hasard, le rang moyen serait 4 sur 7.")
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
