"""
Récupération des données. Chaque source est facultative : si elle échoue,
le modèle continue sans elle et le note dans « sources » du fichier de sortie.
"""
import os, re, json, math, datetime as dt
import numpy as np
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

UA = {"User-Agent": "WinN3oum/0.3 (app baignade Tunisie; contact via github.com/Rzi11)"}


def http_get(url, params=None, timeout=90, tries=4):
    """GET avec réessais (le service de vent historique est parfois lent)."""
    import requests, time
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=UA)
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            print(f"[réseau] essai {k + 1}/{tries} échoué ({str(e)[:80]}), nouvel essai…")
            time.sleep(5 * (k + 1))
    raise last


class Grid:
    """Champ sur grille régulière. data[var] = tableau [t, y, x] (ou [y, x] si times est None)."""
    def __init__(self, times, lats, lons, data, name=""):
        self.times, self.lats, self.lons, self.data, self.name = times, lats, lons, data, name

    def tindex(self, t):
        return int(np.argmin(np.abs(self.times - t)))


# ====================================================================== Copernicus Marine
PHY = "MEDSEA_ANALYSISFORECAST_PHY_006_013"
WAV = "MEDSEA_ANALYSISFORECAST_WAV_006_017"
BGC = "MEDSEA_ANALYSISFORECAST_BGC_006_014"
_CATALOG = {}


def _cm():
    import copernicusmarine
    return copernicusmarine


def cm_datasets(product):
    if product in _CATALOG:
        return _CATALOG[product]
    ids = []
    try:
        cat = _cm().describe(product_id=product)
        for prod in getattr(cat, "products", []) or []:
            for ds in getattr(prod, "datasets", []) or []:
                ids.append(ds.dataset_id)
    except Exception as e:
        print(f"[copernicus] catalogue {product} indisponible : {e}")
    _CATALOG[product] = ids
    print(f"[copernicus] {product} : {ids}")
    return ids


def cm_open(product, patterns, variables, box, start=None, end=None, depth=None, label="", parts=(None,)):
    """
    Ouvre le premier jeu du produit dont l'identifiant contient tous les morceaux d'un motif.
    patterns : liste de tuples de morceaux, par ordre de préférence. depth : (min, max) en m ou None.
    Retourne un Grid (variables demandées) ou lève une erreur.
    """
    user = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
    pwd = os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")
    if not user or not pwd:
        raise RuntimeError("identifiants Copernicus absents")
    ids = cm_datasets(product)
    cands = []
    for pat in patterns:
        for d in ids:
            if all(p in d for p in pat) and d not in cands:
                cands.append(d)
    if not cands:
        cands = [p[0] for p in patterns if len(p) == 1 and p[0].startswith("cmems_")]
    last = None
    for ds_id in cands:
        for part in parts:
            try:
                kw = dict(dataset_id=ds_id, username=user, password=pwd, variables=variables,
                          minimum_longitude=box["lon_min"] - .1, maximum_longitude=box["lon_max"] + .1,
                          minimum_latitude=box["lat_min"] - .1, maximum_latitude=box["lat_max"] + .1)
                if start is not None:
                    kw["start_datetime"] = start.strftime("%Y-%m-%dT%H:%M:%S")
                    kw["end_datetime"] = end.strftime("%Y-%m-%dT%H:%M:%S")
                if depth is not None:
                    kw["minimum_depth"], kw["maximum_depth"] = depth
                if part:
                    kw["dataset_part"] = part
                ds = _cm().open_dataset(**kw)
                if "depth" in ds.dims:
                    ds = ds.isel(depth=0)
                ds = ds.load()
                data = {v: ds[v].values.astype("float32") for v in variables}
                times = ds["time"].values.astype("datetime64[m]") if "time" in ds.dims else None
                g = Grid(times, ds["latitude"].values.astype(float), ds["longitude"].values.astype(float), data, ds_id)
                nt = 1 if times is None else len(times)
                print(f"[{label}] OK {ds_id}{' part=' + part if part else ''} : {nt} pas de temps, {len(g.lats)}×{len(g.lons)}")
                return g
            except Exception as e:
                print(f"[{label}] échec {ds_id}{' part=' + part if part else ''} : {str(e)[:200]}")
                last = e
    raise RuntimeError(f"{label} indisponible ({last})")


def load_currents_surface(box, start, end):
    return cm_open(PHY, [("cur", "2D", "PT1H"), ("cur", "PT15M"), ("cur", "3D", "PT1H"), ("cur", "P1D")],
                   ["uo", "vo"], box, start, end, depth=(0, 1.1), label="courants surface")


def load_currents_depth(box, start, end):
    return cm_open(PHY, [("cur", "3D", "PT1H"), ("cur", "P1D")], ["uo", "vo"], box, start, end,
                   depth=(12, 18), label="courants 15 m")


def load_stokes(box, start, end):
    return cm_open(WAV, [("wav", "PT1H"), ("wav",)], ["VSDX", "VSDY"], box, start, end, label="vagues (Stokes)")


def load_sea_level(box, start, end):
    return cm_open(PHY, [("ssh", "PT15M"), ("ssh", "2D", "PT1H"), ("ssh", "PT1H"), ("ssh", "P1D")],
                   ["zos"], box, start, end, label="hauteur de mer")


def load_habitat(box, start, end):
    """Température, salinité, couche de surface, chlorophylle (moyennes journalières). Dictionnaire de Grid."""
    out = {}
    specs = [
        ("sst", PHY, [("tem", "P1D"), ("tem", "2D", "PT1H")], "thetao", (0, 1.1)),
        ("sal", PHY, [("sal", "P1D"), ("sal", "2D", "PT1H")], "so", (0, 1.1)),
        ("mld", PHY, [("mld", "P1D"), ("mld", "2D", "PT1H")], "mlotst", None),
        ("chl", BGC, [("pft", "P1D"), ("bgc", "P1D")], "chl", (0, 1.1)),
    ]
    for key, prod, pats, var, depth in specs:
        try:
            g = cm_open(prod, pats, [var], box, start, end, depth=depth, label=f"habitat {key}")
            g.data = {key: g.data[var]}
            out[key] = g
        except Exception as e:
            print(f"[habitat] {key} ignoré : {str(e)[:150]}")
    return out


PHY_MY = "MEDSEA_MULTIYEAR_PHY_006_004"


def load_currents_daily(box, start, end):
    """Courants de surface journaliers sur plusieurs années (réanalyse, puis « interim », puis analyse récente)."""
    try:
        return cm_open(PHY_MY, [("cur", "rean", "-d"), ("cur", "my", "P1D"), ("cur", "myint", "P1D"), ("cur", "int", "-d")],
                       ["uo", "vo"], box, start, end, depth=(0, 1.1), label="courants journaliers (réanalyse)")
    except Exception as e:
        print(f"[connectivité] réanalyse indisponible pour cette période ({str(e)[:120]}), essai avec l'analyse récente")
        return cm_open(PHY, [("cur", "P1D")], ["uo", "vo"], box, start, end, depth=(0, 1.1), label="courants journaliers (analyse)")


def load_bathymetry(box):
    return cm_open(PHY, [("static",)], ["deptho"], box, label="fonds marins", parts=(None, "bathy"))


# ====================================================================== Open-Meteo (vent)
def load_wind(box, n=3, start_date=None, end_date=None, past_days=3, forecast_days=4):
    """Vent à 10 m sur une grille n×n. Prévision normale, ou passé (start_date/end_date) pour le rejeu."""
    import requests
    glats = np.linspace(box["lat_min"], box["lat_max"], n)
    glons = np.linspace(box["lon_min"], box["lon_max"], n)
    LA, LO = np.meshgrid(glats, glons, indexing="ij")
    params = dict(latitude=",".join(f"{x:.3f}" for x in LA.ravel()), longitude=",".join(f"{x:.3f}" for x in LO.ravel()),
                  hourly="wind_speed_10m,wind_direction_10m", wind_speed_unit="ms", timezone="GMT")
    if start_date:
        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        params.update(start_date=start_date, end_date=end_date)
    else:
        url = "https://api.open-meteo.com/v1/forecast"
        params.update(past_days=past_days, forecast_days=forecast_days)
    r = http_get(url, params=params, timeout=120)
    js = r.json()
    js = js if isinstance(js, list) else [js]
    times = np.array(js[0]["hourly"]["time"], dtype="datetime64[m]")
    U = np.zeros((len(times), n, n), "float32"); V = np.zeros_like(U)
    for k, pt in enumerate(js):
        i, j = divmod(k, n)
        s = np.array([x if x is not None else 0 for x in pt["hourly"]["wind_speed_10m"]], float)
        d = np.radians([x if x is not None else 0 for x in pt["hourly"]["wind_direction_10m"]])
        U[:, i, j] = -s * np.sin(d); V[:, i, j] = -s * np.cos(d)
    print(f"[vent] OK : {len(times)} heures, grille {n}×{n}")
    return Grid(times, glats, glons, {"u": U, "v": V}, "open-meteo")


# ====================================================================== observations
def _inat_taxon_id(name, rank):
    import requests
    r = http_get("https://api.inaturalist.org/v1/taxa", params=dict(q=name, rank=rank, per_page=5), timeout=30)
    for t in r.json().get("results", []):
        if t.get("name", "").lower() == name.lower():
            return t["id"]
    return None


def load_inaturalist(box, d1, d2):
    import requests
    taxa = [("Scyphozoa", "class"), ("Cubozoa", "class"), ("Physalia physalis", "species"),
            ("Velella velella", "species"), ("Porpita porpita", "species"), ("Olindias muelleri", "species")]
    ids = [i for i in (_inat_taxon_id(n, r) for n, r in taxa) if i]
    if not ids:
        raise RuntimeError("taxons iNaturalist introuvables")
    obs, page = [], 1
    while page <= 5:
        r = http_get("https://api.inaturalist.org/v1/observations", timeout=60, params=dict(
            taxon_id=",".join(map(str, ids)), swlat=box["lat_min"], swlng=box["lon_min"], nelat=box["lat_max"],
            nelng=box["lon_max"], d1=d1, d2=d2, per_page=200, page=page, order_by="observed_on"))
        res = r.json().get("results", [])
        for o in res:
            loc = o.get("location")
            when = o.get("time_observed_at") or o.get("observed_on")
            if not loc or not when:
                continue
            la, lo = map(float, loc.split(","))
            obs.append(dict(lat=la, lon=lo, time=np.datetime64(when[:16].replace(" ", "T"), "m"),
                            taxon=(o.get("taxon") or {}).get("name", "?"), source="iNaturalist", weight=1.0,
                            spread_km=2.0, ref=f"inat:{o.get('id')}"))
        if len(res) < 100:
            break
        page += 1
    print(f"[iNaturalist] {len(obs)} observations")
    return obs


# Toutes les « méduses » utiles au baigneur, pas seulement les Scyphozoa (relecture externe) :
# Cubozoa (cuboméduses), et des Hydrozoa : galion portugais, vélelle, porpite, Olindias.
GBIF_NAMES = ["Cubozoa", "Physalia physalis", "Velella velella", "Porpita porpita", "Olindias muelleri"]


def gbif_keys():
    keys = [352]                                  # 352 = Scyphozoa (méduses vraies)
    for name in GBIF_NAMES:
        try:
            m = http_get("https://api.gbif.org/v1/species/match", params=dict(name=name), timeout=30, tries=2).json()
            if m.get("usageKey") and m.get("matchType") in ("EXACT", "FUZZY"):
                keys.append(int(m["usageKey"]))
        except Exception as e:
            print(f"[GBIF] taxon {name} introuvable ({str(e)[:60]})")
    return keys


def load_gbif(box, d1, d2):
    import requests
    obs = []
    r = http_get("https://api.gbif.org/v1/occurrence/search", timeout=60, params=dict(
        taxonKey=gbif_keys(), decimalLatitude=f"{box['lat_min']},{box['lat_max']}",
        decimalLongitude=f"{box['lon_min']},{box['lon_max']}", eventDate=f"{d1},{d2}",
        hasCoordinate="true", limit=300))
    for o in r.json().get("results", []):
        if "decimalLatitude" not in o or not o.get("eventDate"):
            continue
        if "inaturalist" in str(o.get("datasetName", "")).lower():
            continue                              # déjà pris via iNaturalist
        obs.append(dict(lat=o["decimalLatitude"], lon=o["decimalLongitude"],
                        time=np.datetime64(o["eventDate"][:10] + "T12:00", "m"), taxon=o.get("species") or o.get("scientificName", "?"),
                        source="GBIF", weight=1.0, spread_km=3.0, ref=f"gbif:{o.get('key')}"))
    print(f"[GBIF] {len(obs)} observations")
    return obs


def load_obis(box, d1, d2):
    """OBIS (base mondiale de biodiversité marine) : campagnes scientifiques, JellyWatch, etc. Sans compte."""
    poly = (f"POLYGON(({box['lon_min']} {box['lat_min']},{box['lon_max']} {box['lat_min']},"
            f"{box['lon_max']} {box['lat_max']},{box['lon_min']} {box['lat_max']},{box['lon_min']} {box['lat_min']}))")
    obs = []
    for taxon in ("Scyphozoa", "Cubozoa", "Physalia physalis", "Velella velella", "Porpita porpita", "Olindias"):
        r = http_get("https://api.obis.org/v3/occurrence", timeout=60, params=dict(
            scientificname=taxon, geometry=poly, startdate=d1, enddate=d2, size=1000))
        for o in r.json().get("results", []):
            if "inaturalist" in str(o.get("datasetName", "")).lower():
                continue                          # déjà pris via iNaturalist
            if o.get("decimalLatitude") is None or not o.get("eventDate"):
                continue
            ed = str(o["eventDate"])[:16].replace(" ", "T")
            if len(ed) == 10:
                ed += "T12:00"
            try:
                t = np.datetime64(ed[:16], "m")
            except Exception:
                continue
            obs.append(dict(lat=float(o["decimalLatitude"]), lon=float(o["decimalLongitude"]), time=t,
                            taxon=o.get("scientificName", "?"), source="OBIS", weight=1.0, spread_km=3.0,
                            ref=f"obis:{o.get('id')}"))
    print(f"[OBIS] {len(obs)} observations")
    return obs


def load_reports(path, since):
    """Signalements des utilisateurs de l'app (fichier facultatif data/signalements.json)."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    obs = []
    for r in raw:
        t = np.datetime64(r["time"][:16], "m")
        if t < since:
            continue
        n = r.get("count", 1)                     # 0 = « rien vu »
        obs.append(dict(lat=r["lat"], lon=r["lon"], time=t, taxon=r.get("species", "?"), source="signalement",
                        weight=0.0 if n == 0 else min(2.0, 0.6 + 0.2 * n), spread_km=1.5, ref=r.get("id", ""), none_seen=(n == 0)))
    print(f"[signalements] {len(obs)}")
    return obs


def load_news(places, queries, since_days=7):
    """Google Actualités (RSS) : articles récents parlant de méduses + un lieu connu → observation approximative."""
    import requests
    now = dt.datetime.now(dt.timezone.utc)
    obs, seen, articles = [], set(), []
    jelly = re.compile(r"m[ée]duse|قنديل|حريق|jellyfish|medus", re.I)
    for q in queries:
        lang = "ar" if re.search(r"[؀-ۿ]", q) else "fr"
        url = "https://news.google.com/rss/search"
        try:
            r = requests.get(url, params=dict(q=q, hl=lang, gl="TN", ceid=f"TN:{lang}"), headers=UA, timeout=30)
            r.raise_for_status()
            root = ET.fromstring(r.content)
        except Exception as e:
            print(f"[presse] « {q} » : {e}")
            continue
        for it in root.iter("item"):
            title = (it.findtext("title") or "").strip()
            link = it.findtext("link") or ""
            try:
                pub = parsedate_to_datetime(it.findtext("pubDate"))
            except Exception:
                continue
            if (now - pub).days > since_days or link in seen or not jelly.search(title):
                continue
            seen.add(link)
            low = title.lower()
            for place, (la, lo) in places.items():
                if place in low:
                    obs.append(dict(lat=la, lon=lo, time=np.datetime64(pub.strftime("%Y-%m-%dT%H:%M"), "m"),
                                    taxon="?", source="presse", weight=0.5, spread_km=12.0, ref=title[:120]))
                    articles.append(dict(title=title[:160], date=pub.strftime("%Y-%m-%d"), place=place))
                    break
    print(f"[presse] {len(obs)} articles localisés")
    return obs, articles


def dedupe(obs, km=2.0, hours=24):
    out = []
    for o in sorted(obs, key=lambda x: x["time"]):
        dup = False
        for p in out[-50:]:
            if abs((o["time"] - p["time"]) / np.timedelta64(1, "h")) < hours and \
               math.hypot((o["lat"] - p["lat"]) * 111, (o["lon"] - p["lon"]) * 92) < km and o["source"] != "presse":
                dup = True; break
        if not dup:
            out.append(o)
    return out
