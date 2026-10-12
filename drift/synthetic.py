"""Données factices pour tester tout le modèle sans réseau (python drift/run_drift.py --test)."""
import datetime as dt
import numpy as np
from sources import Grid
import config as C


def _grid(box, res=0.042):
    lats = np.arange(box["lat_min"] - .1, box["lat_max"] + .1, res)
    lons = np.arange(box["lon_min"] - .1, box["lon_max"] + .1, res)
    return lats, lons


def _land(lats, lons):
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    djerba = ((LA - 33.78) / 0.13) ** 2 + ((LO - 10.90) / 0.16) ** 2 < 1
    mainland = LO < 10.35 + (LA - 32.4) * 0.12          # côte tunisienne grossière à l'ouest
    sicily = ((LA - 37.5) / 0.5) ** 2 + ((LO - 14.2) / 1.2) ** 2 < 1
    malta = ((LA - 35.9) / 0.06) ** 2 + ((LO - 14.4) / 0.1) ** 2 < 1
    return djerba | mainland | sicily | malta


def _times(start, end):
    t0 = np.datetime64(start.strftime("%Y-%m-%dT%H:00"), "m")
    n = int((end - start).total_seconds() // 3600) + 2
    return t0 + np.arange(n) * np.timedelta64(60, "m")


def load_wind(box, n=3, start_date=None, end_date=None, past_days=3, forecast_days=4):
    now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    if start_date:
        s = dt.datetime.fromisoformat(start_date); e = dt.datetime.fromisoformat(end_date) + dt.timedelta(days=1)
    else:
        s = now.replace(hour=0) - dt.timedelta(days=past_days); e = now.replace(hour=0) + dt.timedelta(days=forecast_days)
    times = _times(s, e)
    glats = np.linspace(box["lat_min"], box["lat_max"], n); glons = np.linspace(box["lon_min"], box["lon_max"], n)
    h = np.arange(len(times))[:, None, None]
    U = (-5.0 + 2 * np.sin(h / 12)) * np.ones((len(times), n, n), "float32")   # vent d'est qui varie
    V = (1.0 * np.cos(h / 20)) * np.ones((len(times), n, n), "float32")
    return Grid(times, glats, glons, {"u": U, "v": V}, "synthétique")


def _currents(box, start, end, speed=1.0):
    lats, lons = _grid(box); times = _times(start, end)
    land = _land(lats, lons)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    # courant qui descend le long de la côte est tunisienne vers le golfe de Gabès, + tourbillon
    u = (0.05 - 0.04 * np.sin((LA - 33) * 3)) * speed
    v = (-0.12 + 0.03 * np.cos((LO - 11) * 3)) * speed
    U = np.where(land, np.nan, u)[None].repeat(len(times), 0).astype("float32")
    V = np.where(land, np.nan, v)[None].repeat(len(times), 0).astype("float32")
    tide = 0.15 * np.sin(2 * np.pi * np.arange(len(times)) / 12.42)[:, None, None]
    U = U + tide.astype("float32") * 0.3
    return Grid(times, lats, lons, {"uo": U, "vo": V}, "synthétique")


def load_currents_surface(box, start, end):
    return _currents(box, start, end, 1.0)


def load_currents_depth(box, start, end):
    return _currents(box, start, end, 0.6)


def load_currents_daily(box, start, end):
    g = _currents(box, start, end, 1.0)
    g.times = g.times[::24]
    g.data = {k: v[::24] for k, v in g.data.items()}
    return g


def load_stokes(box, start, end):
    lats, lons = _grid(box); times = _times(start, end)
    U = np.full((len(times), len(lats), len(lons)), -0.03, "float32")
    V = np.full_like(U, 0.0)
    return Grid(times, lats, lons, {"VSDX": U, "VSDY": V}, "synthétique")


def load_sea_level(box, start, end):
    lats, lons = _grid(box); times = _times(start, end)
    z = 0.5 * np.sin(2 * np.pi * np.arange(len(times)) / 12.42)
    Z = (z[:, None, None] * np.ones((len(times), len(lats), len(lons)))).astype("float32")
    Z[:, _land(lats, lons)] = np.nan
    return Grid(times, lats, lons, {"zos": Z}, "synthétique")


def load_habitat(box, start, end):
    lats, lons = _grid(box); times = _times(start, end)[::24]
    land = _land(lats, lons)
    out = {}
    for key, val in (("sst", 24.5), ("sal", 38.2), ("mld", 18.0), ("chl", 0.4)):
        f = np.full((len(times), len(lats), len(lons)), val, "float32"); f[:, land] = np.nan
        out[key] = Grid(times, lats, lons, {key: f}, "synthétique")
    return out


def load_bathymetry(box):
    lats, lons = _grid(box)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    d = np.clip(np.hypot(LA - 33.78, LO - 10.9) * 60 - 8, 1, 80).astype("float32")
    d[_land(lats, lons)] = np.nan
    return Grid(None, lats, lons, {"deptho": d}, "synthétique")


def test_obs(t_now):
    """Un banc vu à Malte il y a 6 jours, des méduses vues au large de Zarzis hier, un article de presse, un « rien vu »."""
    h = np.timedelta64(1, "h")
    return [
        dict(lat=35.85, lon=14.55, time=t_now - 144 * h, taxon="Pelagia noctiluca", source="iNaturalist", weight=1.0, spread_km=3, ref="t1"),
        dict(lat=33.62, lon=11.20, time=t_now - 20 * h, taxon="Rhizostoma pulmo", source="iNaturalist", weight=1.0, spread_km=2, ref="t2"),
        dict(lat=33.86, lon=10.99, time=t_now - 10 * h, taxon="?", source="signalement", weight=1.0, spread_km=1.5, ref="t3"),
        dict(lat=33.74, lon=11.01, time=t_now - 8 * h, taxon="?", source="signalement", weight=0.0, spread_km=1.5, ref="t4", none_seen=True),
        dict(lat=33.80, lon=10.95, time=t_now - 30 * h, taxon="?", source="presse", weight=0.5, spread_km=12, ref="méduses à Djerba"),
    ]


def replay_obs(t_to):
    h = np.timedelta64(1, "h")
    return [
        dict(lat=33.88, lon=10.87, time=t_to - 30 * 24 * h, taxon="Rhizostoma pulmo", source="iNaturalist", weight=1.0, spread_km=2, ref="r1"),
        dict(lat=33.86, lon=10.99, time=t_to - 60 * 24 * h, taxon="Rhizostoma pulmo", source="iNaturalist", weight=1.0, spread_km=2, ref="r2"),
    ]


def validation_obs(t_to):
    """Observations factices réparties sur la région (côte est tunisienne, Djerba, Malte)."""
    rng = np.random.default_rng(1)
    h = np.timedelta64(1, "h")
    spots = [(35.77, 10.86), (35.50, 11.10), (33.86, 10.99), (35.92, 14.50), (34.72, 11.25), (33.55, 11.15)]
    out = []
    for k in range(18):
        la, lo = spots[k % len(spots)]
        out.append(dict(lat=la + rng.normal(0, .02), lon=lo + rng.normal(0, .02), time=t_to - int(rng.integers(5, 200)) * 24 * h,
                        taxon="Pelagia noctiluca", source="iNaturalist", weight=1.0, spread_km=2, ref=f"v{k}"))
    return out


def test_reports(t_now):
    """Signalements factices : 2 baigneurs indépendants à Sidi Mahrez, un « menteur » isolé à El Kantara, une sentinelle à Aghir."""
    h = np.timedelta64(1, "h")
    new = dict(alpha=2, beta=2, role="public", banned=False, n_reports=1)
    good = dict(alpha=9, beta=2, role="public", banned=False, n_reports=12)
    liar = dict(alpha=2, beta=7, role="public", banned=False, n_reports=9)
    sent = dict(alpha=8, beta=2, role="sentinelle", banned=False, n_reports=20)
    return [
        dict(id=1, time=t_now + 8 * h, device_id="a", beach_id="mahrez", lat=33.858, lon=10.985, user_lat=33.859, user_lon=10.984, user_acc=20, seen=True, amount=2, species="Pélagie", stung=False, ip_hash="n1", status="new", dev=new),
        dict(id=2, time=t_now + 9 * h, device_id="b", beach_id="mahrez", lat=33.858, lon=10.985, user_lat=None, user_lon=None, user_acc=None, seen=True, amount=1, species=None, stung=True, ip_hash="n2", status="new", dev=good),
        dict(id=3, time=t_now + 9 * h, device_id="c", beach_id="kantara", lat=33.665, lon=10.955, user_lat=36.8, user_lon=10.2, user_acc=30, seen=True, amount=3, species="Galion", stung=False, ip_hash="n3", status="new", dev=liar),
        dict(id=4, time=t_now + 9 * h, device_id="d", beach_id="kantara", lat=33.665, lon=10.955, user_lat=None, user_lon=None, user_acc=None, seen=True, amount=3, species=None, stung=False, ip_hash="n3", status="new", dev=new),
        dict(id=5, time=t_now + 7 * h, device_id="e", beach_id="aghir", lat=33.772, lon=11.025, user_lat=33.772, user_lon=11.024, user_acc=10, seen=False, amount=0, species=None, stung=False, ip_hash="n5", status="new", dev=sent),
    ]
