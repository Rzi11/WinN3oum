"""
Apprentissage méditerranéen : accumule chaque nuit des exemples « méduses vues / pas vues »
sur toute la Méditerranée, avec les facteurs que le modèle calcule la veille.

  python drift/learn.py --budget 25      (minutes)
  python drift/learn.py --test           (hors ligne, données synthétiques)

Pourquoi : autour de la Tunisie, il n'y a qu'une cinquantaine d'observations par an — trop peu pour régler
le modèle. La physique est la même partout : une pélagie dérive pareil à Valence et à Djerba. On apprend
donc sur des milliers d'observations (Espagne, Italie, Grèce, Croatie, Levant…) puis on applique à la Tunisie.

Chaque ligne de data/training.csv = un point de côte × un jour, avec :
  - les facteurs calculés la VEILLE (sans connaître l'observation) : dérive, observations, mer calme,
    eau favorable, exposition, arrivées, saison…
  - l'effort d'observation autour du point (nb d'observations à < 20 km sur l'année) : les gens voient des
    méduses là où il y a des gens ; ce facteur permet au calibrage de corriger ce biais
  - label = 1 si des méduses ont été observées à < 10 km ce jour-là, sinon 0
"""
import os, sys, csv, json, math, time, random, datetime as dt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import sources as S
import model as M
import validate as V

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P_TRAIN = os.path.join(ROOT, "data", "training.csv")
P_STATE = os.path.join(ROOT, "data", "learn_state.json")
FIELDS = ["region", "run", "date", "lead", "point", "lat", "lon", "label", "weight",
          "drift", "obs", "calm", "habitat", "exposure", "log_arrivals", "from_obs", "season", "effort", "mode"]


def load_state():
    try:
        with open(P_STATE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"next": 0, "done": {}}


def save_state(st):
    os.makedirs(os.path.dirname(P_STATE), exist_ok=True)
    with open(P_STATE, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=1)


def append_rows(rows):
    new = not os.path.exists(P_TRAIN)
    os.makedirs(os.path.dirname(P_TRAIN), exist_ok=True)
    with open(P_TRAIN, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def effort_map(points, obs):
    out = {}
    for p in points:
        n = sum(1 for o in obs if float(M.km_between(p["lat"], p["lon"], o["lat"], o["lon"])) <= C.LEARN_EFFORT_KM)
        out[p["id"]] = math.log1p(n)
    return out


def learn_region(name, box, predict, gather_obs, state, loaders=None, deadline=None):
    L = loaders or S
    now = dt.datetime.utcnow()
    t_to = np.datetime64(now.strftime("%Y-%m-%dT00:00"), "m") - np.timedelta64(3, "D")
    t_from = t_to - np.timedelta64(C.VALID_DAYS_BACK, "D")
    status = {}
    if loaders is not None and hasattr(loaders, "validation_obs"):
        obs = loaders.validation_obs(t_to)
    else:
        obs, _ = gather_obs(t_from, t_to, status, with_news=False, box=box)
    obs = [o for o in obs if not o.get("none_seen")]
    print(f"[apprentissage] {name} : {len(obs)} observations sur 12 mois")
    if len(obs) < 3:
        return 0
    try:
        g = M.ensure_ascending(L.load_bathymetry(box))
        sea = np.isfinite(g.data["deptho"]) if g.data["deptho"].ndim == 2 else np.isfinite(g.data["deptho"][0])
        points = V.coastal_points(g.lats, g.lons, sea, box)
    except Exception as e:
        print(f"[apprentissage] {name} : carte terre/mer indisponible ({e})")
        return 0
    if len(points) > 450:                          # régions très découpées (Égée) : on allège
        random.Random(1).shuffle(points); points = points[:450]
    print(f"[apprentissage] {name} : {len(points)} points de côte")
    events = set()
    for o in obs:
        k, d = V.snap(points, o)
        if k is not None:
            events.add((M.local_date(o["time"]), points[k]["id"]))
    effort = effort_map(points, obs)
    done = set(state["done"].get(name, []))
    dates = sorted({d for d, _ in events if d not in done}, reverse=True)
    n_rows = 0
    rng = random.Random(len(done) + 7)
    for d in dates[:C.LEARN_RUNS_PER_REGION]:
        if deadline and time.time() > deadline:
            break
        D = dt.date.fromisoformat(d)
        t_now = np.datetime64((D - dt.timedelta(days=2)).isoformat() + "T00:00", "m")   # lancé 1–2 jours avant
        past = [o for o in obs if o["time"] < t_now]
        st = {}
        try:
            res, _ = predict(t_now, past, st, historical=True, loaders=loaders, beaches=points, box=box, region=box,
                             spacing=C.LEARN_SPACING, lookback_days=C.VALID_LOOKBACK_DAYS)
        except Exception as e:
            print(f"[apprentissage] {name} {d} : échec {str(e)[:150]}")
            state["done"].setdefault(name, []).append(d)
            continue
        rows = []
        for lead in (1, 2):
            for p in points:
                r = res["beaches"][p["id"]]["days"][lead]
                label = 1 if (r["date"], p["id"]) in events else 0
                if label == 0 and rng.random() > C.LEARN_NEG_SAMPLE:
                    continue
                f = r["factors"]
                rows.append(dict(region=name, run=str(t_now)[:10], date=r["date"], lead=lead, point=p["id"],
                                 lat=round(p["lat"], 3), lon=round(p["lon"], 3), label=label,
                                 weight=1.0 if label else round(1 / C.LEARN_NEG_SAMPLE, 3),
                                 drift=f.get("drift", 0), obs=f.get("obs", 0), calm=f.get("calm", 1 - f.get("wind_past", 0)),
                                 habitat=f.get("habitat", ""), exposure=f.get("exposure", 1),
                                 log_arrivals=round(math.log1p(max(0.0, f.get("arrivals", 0))), 3),
                                 from_obs=round(math.log1p(max(0.0, f.get("from_obs", 0))), 3),
                                 season=f.get("season", ""), effort=round(effort[p["id"]], 3), mode=res["mode"]))
        append_rows(rows)
        n_pos = sum(r["label"] for r in rows)
        n_rows += len(rows)
        state["done"].setdefault(name, []).append(d)
        print(f"[apprentissage] {name} {d} : {len(rows)} exemples dont {n_pos} avec méduses")
    return n_rows


def main():
    import run_drift as R
    test = "--test" in sys.argv
    loaders = None
    if test:
        import synthetic as loaders
    budget = C.LEARN_BUDGET_MIN
    if "--budget" in sys.argv:
        budget = float(sys.argv[sys.argv.index("--budget") + 1])
    deadline = time.time() + budget * 60
    state = load_state()
    names = list(C.LEARN_REGIONS)
    if test:
        names = ["tunisie_sicile"]
    total, visited = 0, 0
    while time.time() < deadline and visited < len(names):
        name = names[state["next"] % len(names)]
        state["next"] += 1
        visited += 1
        try:
            total += learn_region(name, C.LEARN_REGIONS[name], R.predict, R.gather_obs, state, loaders, deadline)
        except Exception as e:
            print(f"[apprentissage] {name} : erreur {str(e)[:200]}")
        save_state(state)
    n = 0
    if os.path.exists(P_TRAIN):
        with open(P_TRAIN, encoding="utf-8") as f:
            n = sum(1 for _ in f) - 1
    print(f"[apprentissage] {total} nouveaux exemples ce soir, {n} au total dans training.csv")


if __name__ == "__main__":
    main()
