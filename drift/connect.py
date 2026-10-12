"""
Carte « d'où viennent les méduses de Djerba ? » (v0.7) — calculée une fois, puis rafraîchie de temps en temps.

  python drift/connect.py --budget 120     (minutes ; mode « connectivite » du workflow)
  python drift/connect.py --test           (hors ligne, données synthétiques)

Méthode (« rétro-dérive ») : tous les 3 jours de mars à novembre, sur les 4 dernières années, on place des méduses
virtuelles devant chaque plage de Djerba et on remonte le temps avec les courants réels de ces jours-là (réanalyse
Copernicus). On note où était cette eau 1, 2 … 21 jours avant. Résultat, par saison :
  - « passage » : part de l'eau arrivée à Djerba qui est passée par chaque zone (Zarzis, Libye, Kerkennah, Lampedusa,
                  Malte, Sicile…) dans les 21 jours avant, et le délai typique ;
  - « origine » : où était l'eau 3, 7, 14 et 21 jours avant ;
  - une grille (cases de ~10 km) pour une future carte.
Ça dit où surveiller en priorité, et ça répond à « un banc vu en Sicile peut-il arriver à Djerba ? » avec des chiffres.

Limite : courants de surface seuls (sans le vent ni la descente des pélagies à 15 m le jour) : c'est la route de
l'eau, pas exactement celle de chaque méduse.
"""
import os, sys, json, math, time, datetime as dt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import sources as S
import model as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("WN_DATA") or os.path.join(ROOT, "data")
SEASONS = {"printemps": (3, 4, 5), "ete": (6, 7, 8), "automne": (9, 10, 11)}
LAGS_SHOWN = (3, 7, 14, 21)
ZONE_NAMES = list(C.CONNECT_ZONES)
OTHER = "Ailleurs (large, hors zones)"


def zone_of(lat, lon):
    z = np.full(len(lat), -1, int)
    for k, name in enumerate(ZONE_NAMES):
        a, b, c, d = C.CONNECT_ZONES[name]
        sel = (z < 0) & (lat >= a) & (lat <= b) & (lon >= c) & (lon <= d)
        z[sel] = k
    return z


def release_points(g, rng):
    """Points de mer à 3–8 km devant chaque plage (dans un cône de ±50° autour de son orientation)."""
    sea = np.isfinite(g.data["uo"][0])
    lat, lon = [], []
    for b in C.BEACHES:
        n, tries = 0, 0
        while n < C.CONNECT_PER_BEACH and tries < 2000:
            tries += 1
            a = math.radians(b["face"] + rng.uniform(-50, 50))
            r = rng.uniform(3, 8)
            la = b["lat"] + r * math.cos(a) / 111.32
            lo = b["lon"] + r * math.sin(a) / (111.32 * math.cos(math.radians(b["lat"])))
            iy = int(np.argmin(np.abs(g.lats - la))); ix = int(np.argmin(np.abs(g.lons - lo)))
            if sea[iy, ix]:
                lat.append(la); lon.append(lo); n += 1
    return np.array(lat), np.array(lon)


def backtrack(g, lat0, lon0, t_rel, days, dt_h=3):
    """Remonte le temps : positions (jours 0..days) de l'eau qui se trouve en (lat0, lon0) à t_rel."""
    U, V = g.data["uo"], g.data["vo"]
    Uz, Vz = np.nan_to_num(U), np.nan_to_num(V)
    t0 = g.times[0]
    nt = len(g.times)
    py, px = lat0.copy(), lon0.copy()
    out = np.full((days + 1, len(py), 2), np.nan)
    out[0, :, 0], out[0, :, 1] = py, px
    alive = np.ones(len(py), bool)
    dts = -dt_h * 3600.0
    steps_per_day = int(24 / dt_h)

    def vel(tf, y, x):
        k = min(max(tf, 0.0), nt - 1.0001)
        i = int(k); w = k - i
        u = (1 - w) * M.bilinear(U[i], g.lats, g.lons, y, x) + w * M.bilinear(U[i + 1], g.lats, g.lons, y, x)
        v = (1 - w) * M.bilinear(V[i], g.lats, g.lons, y, x) + w * M.bilinear(V[i + 1], g.lats, g.lons, y, x)
        # près de la côte (case « terre ») : vitesse de la mer voisine plutôt que rien
        bad = np.isnan(u)
        if bad.any():
            u[bad] = (1 - w) * M.bilinear(Uz[i], g.lats, g.lons, y[bad], x[bad]) + w * M.bilinear(Uz[i + 1], g.lats, g.lons, y[bad], x[bad])
            v[bad] = (1 - w) * M.bilinear(Vz[i], g.lats, g.lons, y[bad], x[bad]) + w * M.bilinear(Vz[i + 1], g.lats, g.lons, y[bad], x[bad])
        return np.nan_to_num(u), np.nan_to_num(v)

    tf = (t_rel - t0) / np.timedelta64(1, "D")
    for d in range(1, days + 1):
        for _ in range(steps_per_day):
            if tf < 0:
                alive[:] = False
                break
            y, x = py[alive], px[alive]
            cos = np.cos(np.radians(y))
            u1, v1 = vel(tf, y, x)
            ym = y + v1 * dts / 2 / M.M_PER_DEG; xm = x + u1 * dts / 2 / (M.M_PER_DEG * cos)
            u2, v2 = vel(tf + dt_h / 48, ym, xm)
            ny = y + v2 * dts / M.M_PER_DEG; nx = x + u2 * dts / (M.M_PER_DEG * cos)
            inside = (ny > g.lats[0]) & (ny < g.lats[-1]) & (nx > g.lons[0]) & (nx < g.lons[-1])
            py[alive] = np.where(inside, ny, y); px[alive] = np.where(inside, nx, x)
            idx = np.where(alive)[0]
            alive[idx[~inside]] = False            # sorti de la carte : on perd sa trace (venu de très loin)
            tf -= dt_h / 24
        out[d, alive, 0], out[d, alive, 1] = py[alive], px[alive]
    return out


def season_of(month):
    for s, ms in SEASONS.items():
        if month in ms:
            return s
    return None


def empty_acc():
    return dict(n=0, visits={k: 0 for k in ZONE_NAMES}, first_lag={k: [] for k in ZONE_NAMES},
                origin={str(l): {} for l in LAGS_SHOWN}, grid={})


def accumulate(acc, traj):
    """traj : (jours+1, n, 2)."""
    days, n = traj.shape[0] - 1, traj.shape[1]
    acc["n"] += n
    zones = np.full((days + 1, n), -2, int)                 # -2 = perdu (sorti de la carte)
    for d in range(days + 1):
        ok = np.isfinite(traj[d, :, 0])
        zones[d, ok] = zone_of(traj[d, ok, 0], traj[d, ok, 1])
    for k, name in enumerate(ZONE_NAMES):
        if name == "Autour de Djerba":
            continue
        hit = zones[1:] == k
        anyh = hit.any(axis=0)
        acc["visits"][name] += int(anyh.sum())
        first = np.argmax(hit, axis=0)[anyh] + 1
        acc["first_lag"][name] += first.tolist()
    for l in LAGS_SHOWN:
        if l > days:
            continue
        o = acc["origin"][str(l)]
        for z in zones[l]:
            key = ZONE_NAMES[z] if z >= 0 else (OTHER if z == -1 else "Perdue (venue de plus loin)")
            o[key] = o.get(key, 0) + 1
    for d in range(1, days + 1):
        ok = np.isfinite(traj[d, :, 0])
        for la, lo in zip(np.round(traj[d, ok, 0] / 0.1) * 0.1, np.round(traj[d, ok, 1] / 0.1) * 0.1):
            key = f"{la:.1f},{lo:.1f}"
            acc["grid"][key] = acc["grid"].get(key, 0) + 1


def summarize(acc):
    n = max(1, acc["n"])
    passage = []
    for name in ZONE_NAMES:
        if name == "Autour de Djerba":
            continue
        part = acc["visits"][name] / n
        lags = acc["first_lag"][name]
        passage.append(dict(zone=name, part=round(part, 3), delai_median=int(np.median(lags)) if lags else None,
                            delai_rapide=int(np.percentile(lags, 10)) if lags else None))
    passage.sort(key=lambda x: -x["part"])
    origine = {l: dict(sorted(((k, round(v / n, 3)) for k, v in o.items()), key=lambda kv: -kv[1]))
               for l, o in acc["origin"].items()}
    tot = sum(acc["grid"].values()) or 1
    grid = sorted(([float(k.split(",")[0]), float(k.split(",")[1]), round(v / tot, 5)] for k, v in acc["grid"].items()),
                  key=lambda x: -x[2])[:600]
    return dict(particules=acc["n"], passage=passage, origine=origine, grille=grid)


def main():
    test = "--test" in sys.argv
    L = S
    if test:
        import synthetic as L
    budget = 120.0
    if "--budget" in sys.argv:
        budget = float(sys.argv[sys.argv.index("--budget") + 1])
    deadline = time.time() + budget * 60
    rng = np.random.default_rng(3)
    this_year = dt.date.today().year
    years = [this_year - k for k in range(1, C.CONNECT_YEARS_BACK + 1)]
    if test:
        years = [this_year - 1]
    accs = {s: empty_acc() for s in SEASONS}
    done_years, sources = [], set()
    for y in years:
        if time.time() > deadline:
            print("[connectivité] budget de temps atteint"); break
        start = dt.datetime(y, 3, 1) - dt.timedelta(days=C.CONNECT_DAYS + 2)
        end = dt.datetime(y, 11, 30, 23)
        if test:
            start, end = dt.datetime(y, 6, 1), dt.datetime(y, 7, 15)
        try:
            g = M.ensure_ascending(L.load_currents_daily(C.CONNECT_BOX, start, end))
        except Exception as e:
            print(f"[connectivité] {y} : courants indisponibles ({str(e)[:150]})"); continue
        sources.add(g.name)
        lat0, lon0 = release_points(g, rng)
        print(f"[connectivité] {y} : {len(lat0)} méduses virtuelles par lâcher, données {g.name}")
        d = start + dt.timedelta(days=C.CONNECT_DAYS + 2)
        n_rel = 0
        while d <= end and time.time() < deadline:
            s = season_of(d.month)
            if s:
                t_rel = np.datetime64(d.strftime("%Y-%m-%dT12:00"), "m")
                traj = backtrack(g, lat0, lon0, t_rel, C.CONNECT_DAYS)
                accumulate(accs[s], traj)
                n_rel += 1
            d += dt.timedelta(days=C.CONNECT_EVERY_DAYS)
        print(f"[connectivité] {y} : {n_rel} lâchers")
        done_years.append(y)
    if not done_years:
        print("[connectivité] aucune année calculée"); return
    out = dict(generated=dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z", version=C.VERSION,
               annees=done_years, donnees=sorted(sources), jours=C.CONNECT_DAYS,
               saisons={s: summarize(a) for s, a in accs.items() if a["n"]},
               note="Rétro-dérive : où était l'eau qui arrive devant les plages de Djerba, 1 à 21 jours avant. "
                    "Courants de surface seuls (sans vent ni plongée des pélagies) : c'est la route de l'eau.")
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "connectivite.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for s, r in out["saisons"].items():
        print(f"[connectivité] {s} ({r['particules']} trajectoires) — passées par… :")
        for p in r["passage"][:8]:
            print(f"    {p['zone']:<36} {p['part'] * 100:5.1f} %   délai typique {p['delai_median']} j (le plus rapide ~{p['delai_rapide']} j)")


if __name__ == "__main__":
    main()
