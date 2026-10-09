"""
Win N3oum ? — modèle de dérive des méduses (lancé chaque nuit par GitHub Actions).

Principe :
  1. On récupère les prévisions de courants de surface (Copernicus Marine, Méditerranée, ~4 km)
     et de vent (Open-Meteo) pour les 3 prochains jours autour de Djerba.
  2. On lâche des milliers de « méduses virtuelles » en mer, partout dans la zone.
  3. On les fait dériver heure par heure : courant + une petite part du vent (les méduses
     flottent près de la surface et « prennent » un peu le vent).
  4. On compte combien arrivent sur chaque plage, jour par jour.
  5. On combine avec la saison (les méduses ne sont pas là toute l'année) pour donner
     un niveau de risque : Faible / Possible / Élevé.

Ce n'est PAS une prévision validée : c'est un indicateur de « poussée vers la plage ».
Il faudra le régler avec les signalements de l'été (paramètres WINDAGE, K_ARRIVALS, SEASON).

Si Copernicus ne répond pas, le script passe en mode secours « vent seul »
(vent vers la plage sur 24 h), pour que l'app ait toujours quelque chose à afficher.

Lancer en local pour tester sans connexion : python drift/run_drift.py --test
"""
import json, math, os, sys, datetime as dt
import numpy as np

# ------------------------------------------------------------------ réglages
BOX = dict(lon_min=10.45, lon_max=11.45, lat_min=33.40, lat_max=34.15)   # Djerba + golfe de Gabès proche
HOURS = 72                       # horizon de prévision
DT = 1800                        # pas de temps : 30 min
SEED_SPACING = 0.012             # ~1,2 km entre deux méduses virtuelles au départ
WINDAGE = 0.012                  # part du vent ajoutée au courant (1,2 %) — à régler
R_NEAR_KM = 3.0                  # une méduse à moins de 3 km d'une plage compte comme « arrivée »
R_BEACHED_KM = 6.0               # méduse échouée sur la côte : rattachée à la plage la plus proche (< 6 km)
K_ARRIVALS = 15.0                # nb d'arrivées/jour qui donne une « poussée » d'environ 63 % — à régler
UTC_OFFSET = 1                   # Tunisie : UTC+1, sans heure d'été

# Présence saisonnière dans le sud tunisien (0 = absentes, 1 = pleine saison).
# Hypothèse de départ tirée de la recherche (pélagie, méduse chou-fleur, Rhopilema) — à affiner.
SEASON = {1: .15, 2: .15, 3: .2, 4: .3, 5: .5, 6: .7, 7: .85, 8: .9, 9: .8, 10: .55, 11: .3, 12: .2}

# face = direction vers laquelle la plage regarde la mer (0 = Nord) — utilisé par le mode secours
BEACHES = [
    dict(id="rmel",       name="Ras R'mel",      lat=33.895, lon=10.925, face=0),
    dict(id="seguia",     name="La Seguia",      lat=33.735, lon=11.005, face=125),
    dict(id="mahrez",     name="Sidi Mahrez",    lat=33.858, lon=10.985, face=20),
    dict(id="aghir",      name="Aghir",          lat=33.772, lon=11.025, face=100),
    dict(id="taguermess", name="Ras Taguermess", lat=33.825, lon=11.055, face=65),
    dict(id="kantara",    name="El Kantara",     lat=33.665, lon=10.955, face=155),
    dict(id="jmour",      name="Sidi Jmour",     lat=33.848, lon=10.745, face=295),
]

# Jeux de données Copernicus essayés dans l'ordre (courants de surface, Méditerranée, prévision)
# (le programme cherche aussi lui-même les jeux horaires / 15 min du produit, voir find_datasets)
CMEMS_PRODUCT = "MEDSEA_ANALYSISFORECAST_PHY_006_013"
CMEMS_DATASETS = [
    "cmems_mod_med_phy-cur_anfc_4.2km-2D_PT1H-m",
    "cmems_mod_med_phy_anfc_4.2km-2D_PT1H-m",
    "cmems_mod_med_phy-cur_anfc_4.2km_PT15M-i",
    "cmems_mod_med_phy-cur_anfc_4.2km_PT1H-m",
    "cmems_mod_med_phy-cur_anfc_4.2km_P1D-m",       # secours : moyenne journalière (fonctionne)
]

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "meduses.json")
M_PER_DEG = 111320.0


# ------------------------------------------------------------------ données
def find_datasets(copernicusmarine):
    """Liste les jeux du produit Méditerranée et met en tête ceux qui sont horaires ou au quart d'heure."""
    found = []
    try:
        cat = copernicusmarine.describe(product_id=CMEMS_PRODUCT)
        for prod in getattr(cat, "products", []) or []:
            for ds in getattr(prod, "datasets", []) or []:
                found.append(ds.dataset_id)
    except Exception as e:
        print(f"[courants] catalogue indisponible : {e}")
    print(f"[courants] jeux du produit : {found}")
    def rank(d):
        fast = "PT15M" in d or "PT1H" in d
        return (0 if fast and "cur" in d else 1 if fast else 2)
    auto = sorted([d for d in found if ("PT1H" in d or "PT15M" in d) and ("cur" in d or "2D" in d)], key=rank)
    seen, order = set(), []
    for d in auto + CMEMS_DATASETS:
        if d not in seen:
            seen.add(d); order.append(d)
    return order


def load_currents(start, end):
    """Retourne (times[datetime64], lats, lons, u[t,y,x], v[t,y,x]) en m/s, NaN sur la terre."""
    import copernicusmarine
    user = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
    pwd = os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")
    if not user or not pwd:
        raise RuntimeError("Identifiants Copernicus absents (secrets COPERNICUS_USER / COPERNICUS_PASSWORD)")
    last = None
    for ds_id in find_datasets(copernicusmarine):
        try:
            print(f"[courants] essai {ds_id} …")
            ds = copernicusmarine.open_dataset(
                dataset_id=ds_id, username=user, password=pwd,
                variables=["uo", "vo"],
                minimum_longitude=BOX["lon_min"] - 0.1, maximum_longitude=BOX["lon_max"] + 0.1,
                minimum_latitude=BOX["lat_min"] - 0.1, maximum_latitude=BOX["lat_max"] + 0.1,
                minimum_depth=0, maximum_depth=1.1,
                start_datetime=start.strftime("%Y-%m-%dT%H:%M:%S"),
                end_datetime=end.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            if "depth" in ds.dims:
                ds = ds.isel(depth=0)
            ds = ds.load()
            times = ds["time"].values
            lats = ds["latitude"].values.astype(float)
            lons = ds["longitude"].values.astype(float)
            u = ds["uo"].values.astype(float)
            v = ds["vo"].values.astype(float)
            if "uo" not in ds or "vo" not in ds:
                raise RuntimeError("pas de variables uo/vo dans ce jeu")
            if len(times) < 2:
                raise RuntimeError("moins de 2 pas de temps")
            print(f"[courants] OK : {len(times)} pas de temps, grille {len(lats)}×{len(lons)}")
            return times, lats, lons, u, v, ds_id
        except Exception as e:  # on tente le jeu suivant
            print(f"[courants] échec {ds_id} : {e}")
            last = e
    raise RuntimeError(f"aucun jeu de courants disponible ({last})")


def load_wind(start):
    """Vent à 10 m sur une grille 3×3 (Open-Meteo). Retourne (times, glats, glons, u[t,i,j], v[t,i,j]) en m/s."""
    import requests
    glats = np.linspace(BOX["lat_min"], BOX["lat_max"], 3)
    glons = np.linspace(BOX["lon_min"], BOX["lon_max"], 3)
    LA, LO = np.meshgrid(glats, glons, indexing="ij")
    params = dict(latitude=",".join(f"{x:.3f}" for x in LA.ravel()),
                  longitude=",".join(f"{x:.3f}" for x in LO.ravel()),
                  hourly="wind_speed_10m,wind_direction_10m", wind_speed_unit="ms",
                  timezone="GMT", past_days=1, forecast_days=4)
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    js = js if isinstance(js, list) else [js]
    times = np.array(js[0]["hourly"]["time"], dtype="datetime64[m]")
    nt = len(times)
    U = np.zeros((nt, 3, 3)); V = np.zeros((nt, 3, 3))
    for k, pt in enumerate(js):
        i, j = divmod(k, 3)
        s = np.array([x if x is not None else 0 for x in pt["hourly"]["wind_speed_10m"]], float)
        d = np.radians(np.array([x if x is not None else 0 for x in pt["hourly"]["wind_direction_10m"]], float))
        U[:, i, j] = -s * np.sin(d)   # direction = d'où vient le vent
        V[:, i, j] = -s * np.cos(d)
    print(f"[vent] OK : {nt} heures")
    return times, glats, glons, U, V


# ------------------------------------------------------------------ outils
def km_between(lat1, lon1, lat2, lon2):
    dy = (lat2 - lat1) * 111.32
    dx = (lon2 - lon1) * 111.32 * np.cos(np.radians((lat1 + lat2) / 2))
    return np.hypot(dx, dy)


def bilinear(field, lats, lons, py, px):
    """Interpolation bilinéaire d'un champ [y,x] ; NaN si un coin est sur la terre ou hors grille."""
    fy = (py - lats[0]) / (lats[1] - lats[0])
    fx = (px - lons[0]) / (lons[1] - lons[0])
    iy = np.floor(fy).astype(int); ix = np.floor(fx).astype(int)
    out = np.full(py.shape, np.nan)
    ok = (iy >= 0) & (ix >= 0) & (iy < len(lats) - 1) & (ix < len(lons) - 1)
    iy0, ix0 = iy[ok], ix[ok]
    wy, wx = fy[ok] - iy0, fx[ok] - ix0
    f00 = field[iy0, ix0]; f01 = field[iy0, ix0 + 1]; f10 = field[iy0 + 1, ix0]; f11 = field[iy0 + 1, ix0 + 1]
    out[ok] = (f00 * (1 - wy) * (1 - wx) + f01 * (1 - wy) * wx + f10 * wy * (1 - wx) + f11 * wy * wx)
    return out


def local_date(t64):
    t = t64.astype("datetime64[s]").astype(dt.datetime) + dt.timedelta(hours=UTC_OFFSET)
    return t.date().isoformat()


# ------------------------------------------------------------------ simulation
def simulate(t0, cur, wind):
    ctimes, clats, clons, cu, cv = cur
    wtimes, wlats, wlons, wu, wv = wind
    ctimes_h = ctimes.astype("datetime64[m]")

    def cur_index(t):
        # pas de temps courant le plus proche (données horaires ou journalières)
        return int(np.argmin(np.abs(ctimes_h - t)))

    def wind_at(t, py, px):
        k = int(np.argmin(np.abs(wtimes - t)))
        i = np.clip(np.rint((py - wlats[0]) / (wlats[1] - wlats[0])).astype(int), 0, 2)
        j = np.clip(np.rint((px - wlons[0]) / (wlons[1] - wlons[0])).astype(int), 0, 2)
        return wu[k, i, j], wv[k, i, j]

    def velocity(t, py, px):
        k = cur_index(t)
        u = bilinear(cu[k], clats, clons, py, px)
        v = bilinear(cv[k], clats, clons, py, px)
        a, b = wind_at(t, py, px)
        return u + WINDAGE * a, v + WINDAGE * b

    # semis : toutes les cases de mer de la zone
    sy = np.arange(BOX["lat_min"], BOX["lat_max"], SEED_SPACING)
    sx = np.arange(BOX["lon_min"], BOX["lon_max"], SEED_SPACING)
    PY, PX = np.meshgrid(sy, sx, indexing="ij")
    py, px = PY.ravel(), PX.ravel()
    u0, _ = velocity(t0, py, px)
    sea = ~np.isnan(u0)
    py, px = py[sea], px[sea]
    alive = np.ones(py.shape, bool)
    print(f"[dérive] {len(py)} méduses virtuelles lâchées")

    blat = np.array([b["lat"] for b in BEACHES]); blon = np.array([b["lon"] for b in BEACHES])
    arrivals = {}   # (beach_id, date) -> nb

    def record(idx, t, radius):
        if not len(idx):
            return np.zeros(0, bool)
        d = km_between(py[idx, None], px[idx, None], blat[None, :], blon[None, :])
        nearest = np.argmin(d, axis=1)
        hit = d[np.arange(len(idx)), nearest] <= radius
        day = local_date(t)
        for n in nearest[hit]:
            key = (BEACHES[n]["id"], day)
            arrivals[key] = arrivals.get(key, 0) + 1
        return hit

    steps = int(HOURS * 3600 / DT)
    for s in range(steps):
        t = t0 + np.timedelta64(int(s * DT / 60), "m")
        tm = t + np.timedelta64(int(DT / 120), "m")
        idx = np.where(alive)[0]
        if not len(idx):
            break
        y, x = py[idx], px[idx]
        u1, v1 = velocity(t, y, x)
        coslat = np.cos(np.radians(y))
        ym = y + v1 * DT / 2 / M_PER_DEG
        xm = x + u1 * DT / 2 / (M_PER_DEG * coslat)
        u2, v2 = velocity(tm, ym, xm)
        u2 = np.where(np.isnan(u2), u1, u2); v2 = np.where(np.isnan(v2), v1, v2)
        ny = y + v2 * DT / M_PER_DEG
        nx = x + u2 * DT / (M_PER_DEG * coslat)
        landed = np.isnan(u1) | np.isnan(bilinear(cu[cur_index(t)], clats, clons, ny, nx))
        py[idx] = np.where(landed, y, ny)
        px[idx] = np.where(landed, x, nx)
        out_box = (py[idx] < BOX["lat_min"] - .2) | (py[idx] > BOX["lat_max"] + .2) | \
                  (px[idx] < BOX["lon_min"] - .2) | (px[idx] > BOX["lon_max"] + .2)
        # 1) échouées sur la côte → plage la plus proche
        li = idx[landed]
        record(li, t, R_BEACHED_KM)
        alive[li] = False
        # 2) passées tout près d'une plage → arrivée
        fi = idx[~landed & ~out_box]
        hit = record(fi, t, R_NEAR_KM)
        alive[fi[hit]] = False
        alive[idx[out_box]] = False
    return arrivals, int(sea.sum())


def wind_only(t0, wind):
    """Mode secours : poussée du vent vers chaque plage (moyenne journalière de la composante « vers la plage »)."""
    wtimes, wlats, wlons, wu, wv = wind
    arrivals = {}
    for b in BEACHES:
        i = int(np.argmin(np.abs(wlats - b["lat"]))); j = int(np.argmin(np.abs(wlons - b["lon"])))
        f = math.radians(b["face"])
        # vecteur « de la mer vers la plage » = opposé de la direction où regarde la plage
        ox, oy = -math.sin(f), -math.cos(f)
        for k, t in enumerate(wtimes):
            if t < t0 or t >= t0 + np.timedelta64(HOURS, "h"):
                continue
            push = max(0.0, wu[k, i, j] * ox + wv[k, i, j] * oy)       # m/s vers la plage
            key = (b["id"], local_date(t))
            arrivals[key] = arrivals.get(key, 0) + push * 0.6          # échelle grossière ≈ arrivées
    return arrivals


# ------------------------------------------------------------------ sortie
def build_output(arrivals, t0, mode, detail):
    days = sorted({local_date(t0 + np.timedelta64(h, "h")) for h in range(0, HOURS, 6)})[:3]
    out = dict(generated=dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
               mode=mode, detail=detail, method="v0.1 — dérive courant + 1,2 % du vent, non validée",
               days=days, beaches={})
    for b in BEACHES:
        rows = []
        for d in days:
            n = float(arrivals.get((b["id"], d), 0))
            drift = 1 - math.exp(-n / K_ARRIVALS)
            base = SEASON[int(d[5:7])]
            risk = base * (0.25 + 0.75 * drift)
            level = "Faible" if risk < 0.2 else "Possible" if risk < 0.45 else "Élevé"
            rows.append(dict(date=d, arrivals=round(n, 1), drift=round(drift, 2),
                             season=base, risk=round(risk, 2), level=level))
        out["beaches"][b["id"]] = dict(name=b["name"], days=rows)
    return out


def main():
    test = "--test" in sys.argv
    now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    t0 = np.datetime64(now.replace(hour=0), "m")
    end = now.replace(hour=0) + dt.timedelta(hours=HOURS + 1)

    if test:
        cur, wind = synthetic(t0)
        arrivals, n = simulate(t0, cur, wind)
        out = build_output(arrivals, t0, "test", f"données synthétiques, {n} méduses")
    else:
        wind = load_wind(t0)
        try:
            times, lats, lons, u, v, ds_id = load_currents(now.replace(hour=0), end)
            arrivals, n = simulate(t0, (times, lats, lons, u, v), wind)
            out = build_output(arrivals, t0, "derive", f"Copernicus {ds_id} + Open-Meteo, {n} méduses virtuelles")
        except Exception as e:
            print(f"[secours] dérive impossible ({e}) → mode vent seul")
            out = build_output(wind_only(t0, wind), t0, "vent", f"vent seul (Copernicus indisponible : {e})"[:300])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[sortie] {OUT}")
    for bid, b in out["beaches"].items():
        print(f"  {b['name']:<15} " + "  ".join(f"{r['date'][5:]}:{r['level']:<8}({r['arrivals']})" for r in b["days"]))


def synthetic(t0):
    """Données factices pour tester sans réseau : île ronde au centre, courant vers l'est, vent d'est."""
    lats = np.arange(BOX["lat_min"] - .1, BOX["lat_max"] + .1, 0.042)
    lons = np.arange(BOX["lon_min"] - .1, BOX["lon_max"] + .1, 0.042)
    times = np.array([t0 + np.timedelta64(h, "h") for h in range(HOURS + 2)])
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    land = ((LA - 33.78) / 0.13) ** 2 + ((LO - 10.90) / 0.16) ** 2 < 1
    u = np.where(land, np.nan, 0.08) * np.ones((len(times), 1, 1))
    v = np.where(land, np.nan, -0.03) * np.ones((len(times), 1, 1))
    wtimes = times.astype("datetime64[m]")
    wl, wo = np.linspace(BOX["lat_min"], BOX["lat_max"], 3), np.linspace(BOX["lon_min"], BOX["lon_max"], 3)
    wu = -6.0 * np.ones((len(times), 3, 3)); wv = np.zeros((len(times), 3, 3))
    return (times, lats, lons, u, v), (wtimes, wl, wo, wu, wv)


if __name__ == "__main__":
    main()
