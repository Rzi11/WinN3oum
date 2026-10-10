"""
Cœur du modèle méduses : dérive physique + facteurs.
"""
import math, datetime as dt
import numpy as np
import config as C

M_PER_DEG = 111320.0


# ====================================================================== outils
def km_between(lat1, lon1, lat2, lon2):
    dy = (lat2 - lat1) * 111.32
    dx = (lon2 - lon1) * 111.32 * np.cos(np.radians((lat1 + lat2) / 2))
    return np.hypot(dx, dy)


def bilinear(field, lats, lons, py, px):
    """Interpolation bilinéaire sur [y,x] ; NaN si un coin est sur la terre ou hors grille."""
    fy = (py - lats[0]) / (lats[1] - lats[0])
    fx = (px - lons[0]) / (lons[1] - lons[0])
    iy = np.floor(fy).astype(int); ix = np.floor(fx).astype(int)
    out = np.full(np.shape(py), np.nan)
    ok = (iy >= 0) & (ix >= 0) & (iy < len(lats) - 1) & (ix < len(lons) - 1)
    iy0, ix0 = iy[ok], ix[ok]
    wy, wx = fy[ok] - iy0, fx[ok] - ix0
    out[ok] = (field[iy0, ix0] * (1 - wy) * (1 - wx) + field[iy0, ix0 + 1] * (1 - wy) * wx +
               field[iy0 + 1, ix0] * wy * (1 - wx) + field[iy0 + 1, ix0 + 1] * wy * wx)
    return out


def nearest(grid_lats, grid_lons, field2d, lat, lon, max_km=15):
    """Valeur de la case de mer la plus proche (pour les plages, souvent juste à côté d'une case « terre »)."""
    iy = np.argmin(np.abs(grid_lats - lat)); ix = np.argmin(np.abs(grid_lons - lon))
    best, bd = None, 1e9
    r = 4
    for y in range(max(0, iy - r), min(len(grid_lats), iy + r + 1)):
        for x in range(max(0, ix - r), min(len(grid_lons), ix + r + 1)):
            v = field2d[y, x]
            if np.isfinite(v):
                d = float(km_between(lat, lon, grid_lats[y], grid_lons[x]))
                if d < bd and d <= max_km:
                    best, bd = float(v), d
    return best


def ensure_ascending(g):
    """Remet latitudes/longitudes dans l'ordre croissant si besoin."""
    if g.lats[0] > g.lats[-1]:
        g.lats = g.lats[::-1]
        g.data = {k: v[..., ::-1, :] for k, v in g.data.items()}
    if g.lons[0] > g.lons[-1]:
        g.lons = g.lons[::-1]
        g.data = {k: v[..., ::-1] for k, v in g.data.items()}
    return g


def local_date(t64):
    t = t64.astype("datetime64[s]").astype(dt.datetime) + dt.timedelta(hours=C.UTC_OFFSET)
    return t.date().isoformat()


def is_daytime(t64, lon):
    """Jour approximatif : heure solaire locale entre 6h30 et 18h."""
    h = (t64.astype("datetime64[m]").astype(int) % 1440) / 60.0 + lon / 15.0
    h = h % 24
    return (h > 6.5) & (h < 18.0)


def moon_age_days(t64):
    """Âge de la lune en jours (0 = nouvelle lune, ~14,8 = pleine lune)."""
    days = (t64.astype("datetime64[m]") - np.datetime64("2000-01-06T18:14", "m")) / np.timedelta64(1, "D")
    return float(days % 29.530588853)


# ====================================================================== habitat (eau favorable)
def habitat_scores(hab, lat, lon, t):
    """Retourne (score 0-1, détail) pour un point, à partir des grilles disponibles."""
    parts, det = [], {}
    def val(key):
        g = hab.get(key)
        if g is None:
            return None
        k = g.tindex(t) if g.times is not None else None
        f = g.data[key][k] if k is not None else g.data[key]
        return nearest(g.lats, g.lons, f, lat, lon)
    sst = val("sst")
    if sst is not None:
        s = np.interp(sst, [13, 17, 20, 28, 31], [0.1, 0.5, 1.0, 1.0, 0.6]); parts.append((0.5, s)); det["sst"] = round(sst, 1)
    chl = val("chl")
    if chl is not None and chl > 0:
        s = float(np.clip(0.3 + 0.35 * math.log10(chl / 0.05), 0, 1)); parts.append((0.2, s)); det["chl"] = round(chl, 3)
    sal = val("sal")
    if sal is not None:
        g = hab["sal"]; f = g.data["sal"][g.tindex(t)] if g.times is not None else g.data["sal"]
        mean = float(np.nanmean(f))
        s = float(np.clip(0.5 + (mean - sal) * 0.5, 0, 1)); parts.append((0.15, s)); det["sal"] = round(sal, 2)
    mld = val("mld")
    if mld is not None:
        s = float(np.clip(1 - (mld - 10) / 40, 0, 1)); parts.append((0.15, s)); det["mld"] = round(mld, 1)
    if not parts:
        return None, det
    wsum = sum(w for w, _ in parts)
    return float(sum(w * s for w, s in parts) / wsum), det


def habitat_field(hab, lats, lons, t):
    """Score d'habitat sur une liste de points (pour pondérer les méduses « de fond »)."""
    g = hab.get("sst")
    if g is None:
        return np.ones(len(lats))
    f = g.data["sst"][g.tindex(t)] if g.times is not None else g.data["sst"]
    sst = bilinear(np.nan_to_num(f, nan=np.nanmean(f)), g.lats, g.lons, lats, lons)
    w = np.interp(np.nan_to_num(sst, nan=22), [13, 17, 20, 28, 31], [0.1, 0.5, 1.0, 1.0, 0.6])
    gc = hab.get("chl")
    if gc is not None:
        fc = gc.data["chl"][gc.tindex(t)] if gc.times is not None else gc.data["chl"]
        chl = bilinear(np.nan_to_num(fc, nan=np.nanmedian(fc)), gc.lats, gc.lons, lats, lons)
        w = w * np.clip(0.7 + 0.3 * np.log10(np.clip(np.nan_to_num(chl, nan=0.1), 0.01, 10) / 0.1), 0.5, 1.5)
    return w / max(1e-6, np.mean(w))


# ====================================================================== exposition des plages (statique)
def beach_exposure(mask_grid, bathy, b):
    """
    Ouverture sur le large (rayons de 20 km sans toucher la terre), profondeur à ~5 km, abri, oued.
    Étude Costa del Sol : plages abritées / peu ouvertes / profondes = plus d'arrivées (effet piège).
    """
    lats, lons, sea = mask_grid
    open_rays = 0
    face = math.radians(b["face"])
    for k in range(36):
        a = 2 * math.pi * k / 36
        hit = False
        for d in np.arange(1.5, 20.1, 1.5):
            la = b["lat"] + d * math.cos(a) / 111.32
            lo = b["lon"] + d * math.sin(a) / (111.32 * math.cos(math.radians(b["lat"])))
            iy = int(np.argmin(np.abs(lats - la))); ix = int(np.argmin(np.abs(lons - lo)))
            if not sea[iy, ix]:
                if d > 3:      # la première case peut être « terre » à cause de la grille
                    hit = True
                    break
        open_rays += 0 if hit else 1
    opening = open_rays / 36
    depth5 = None
    if bathy is not None:
        la = b["lat"] + 5 * math.cos(face) / 111.32
        lo = b["lon"] + 5 * math.sin(face) / (111.32 * math.cos(math.radians(b["lat"])))
        depth5 = nearest(bathy.lats, bathy.lons, bathy.data["deptho"], la, lo, max_km=8)
    dnorm = 0.5 if depth5 is None else float(np.clip(depth5 / 30, 0, 1))
    e = 1 + 0.15 * b.get("shelter", 0) + 0.15 * (1 - opening) + 0.10 * (dnorm - 0.5) - 0.15 * b.get("wadi", 0)
    return float(np.clip(e, 0.75, 1.35)), dict(opening=round(opening, 2), depth5km=None if depth5 is None else round(depth5, 1))


# ====================================================================== dérive
class Drift:
    def __init__(self, cur_s, cur_d, stokes, wind, sea_level, beaches=None):
        self.cs, self.cd, self.st, self.w, self.sl = cur_s, cur_d, stokes, wind, sea_level
        self.beaches = beaches or C.BEACHES
        self.blat = np.array([b["lat"] for b in self.beaches]); self.blon = np.array([b["lon"] for b in self.beaches])
        self.tide_cache = {}
        # carte des cases « proches d'une plage » : on ne teste la distance aux plages que là (rapide)
        g = self.cs
        LA, LO = np.meshgrid(g.lats, g.lons, indexing="ij")
        near = np.zeros(LA.shape, bool)
        rad = (C.R_NEAR_KM + 6) / 111.32
        for la, lo in zip(self.blat, self.blon):
            near |= (np.abs(LA - la) < rad) & (np.abs(LO - lo) < rad / math.cos(math.radians(la)))
        self.near_raster = near

    def near_beach(self, py, px):
        g = self.cs
        iy = np.clip(np.rint((py - g.lats[0]) / (g.lats[1] - g.lats[0])).astype(int), 0, len(g.lats) - 1)
        ix = np.clip(np.rint((px - g.lons[0]) / (g.lons[1] - g.lons[0])).astype(int), 0, len(g.lons) - 1)
        return self.near_raster[iy, ix]

    def nearest_sea(self, t, lat, lon, max_km=15):
        """Case de mer la plus proche d'un point (pour décaler une observation faite trop près du rivage)."""
        g = self.cs; k = g.tindex(t)
        sea = ~np.isnan(g.data["uo"][k])
        iy = int(np.argmin(np.abs(g.lats - lat))); ix = int(np.argmin(np.abs(g.lons - lon)))
        best, bd = None, 1e9
        r = int(max_km / 4) + 2
        for y in range(max(1, iy - r), min(len(g.lats) - 1, iy + r + 1)):
            for x in range(max(1, ix - r), min(len(g.lons) - 1, ix + r + 1)):
                if sea[y, x] and sea[y - 1:y + 2, x - 1:x + 2].all():      # case de mer « pleine » (pas en bord de côte)
                    d = float(km_between(lat, lon, g.lats[y], g.lons[x]))
                    if d < bd and d <= max_km:
                        best, bd = (float(g.lats[y]), float(g.lons[x])), d
        return best

    # ---- champs
    def _cur(self, g, t, py, px):
        k = g.tindex(t)
        return bilinear(g.data["uo"][k], g.lats, g.lons, py, px), bilinear(g.data["vo"][k], g.lats, g.lons, py, px)

    def _wind(self, t, py, px):
        g = self.w; k = g.tindex(t)
        i = np.clip(np.rint((py - g.lats[0]) / (g.lats[1] - g.lats[0])).astype(int), 0, len(g.lats) - 1)
        j = np.clip(np.rint((px - g.lons[0]) / (g.lons[1] - g.lons[0])).astype(int), 0, len(g.lons) - 1)
        return g.data["u"][k, i, j], g.data["v"][k, i, j]

    def velocity(self, t, py, px, kind):
        """kind : tableau de 0 (surface) / 1 (migrante)."""
        u, v = self._cur(self.cs, t, py, px)
        wu, wv = self._wind(t, py, px)
        windage = np.where(kind == 0, C.TYPES["surface"]["windage"], C.TYPES["migrante"]["windage"])
        stk = np.where(kind == 0, C.TYPES["surface"]["stokes"], C.TYPES["migrante"]["stokes"])
        deep = (kind == 1) & is_daytime(t, px)          # pélagie le jour : plus profond, sans vent ni vagues
        if self.cd is not None and deep.any():
            ud, vd = self._cur(self.cd, t, py[deep], px[deep])
            u = u.copy(); v = v.copy()
            u[deep] = np.where(np.isnan(ud), u[deep], ud)
            v[deep] = np.where(np.isnan(vd), v[deep], vd)
        windage = np.where(deep, 0.0, windage); stk = np.where(deep, 0.0, stk)
        su = sv = 0.0
        if self.st is not None:
            k = self.st.tindex(t)
            su = np.nan_to_num(bilinear(self.st.data["VSDX"][k], self.st.lats, self.st.lons, py, px))
            sv = np.nan_to_num(bilinear(self.st.data["VSDY"][k], self.st.lats, self.st.lons, py, px))
        return u + windage * wu + stk * su, v + windage * wv + stk * sv

    def is_sea(self, t, py, px):
        k = self.cs.tindex(t)
        return ~np.isnan(bilinear(self.cs.data["uo"][k], self.cs.lats, self.cs.lons, py, px))

    def _tide_cell(self, bi):
        """Case de mer la plus proche de la plage sur la grille de hauteur de mer (calculée une fois)."""
        key = ("cell", bi)
        if key not in self.tide_cache:
            g = self.sl
            z = g.data["zos"]
            f = np.where(np.isfinite(z).any(axis=0), 0.0, np.nan) if z.ndim == 3 else z
            iy = int(np.argmin(np.abs(g.lats - self.blat[bi]))); ix = int(np.argmin(np.abs(g.lons - self.blon[bi])))
            best, bd = None, 1e9
            for y in range(max(0, iy - 3), min(len(g.lats), iy + 4)):
                for x in range(max(0, ix - 3), min(len(g.lons), ix + 4)):
                    if np.isfinite(f[y, x]):
                        d = float(km_between(self.blat[bi], self.blon[bi], g.lats[y], g.lons[x]))
                        if d < bd and d <= 12:
                            best, bd = (y, x), d
            self.tide_cache[key] = best
        return self.tide_cache[key]

    def tide_mult(self, bi, t):
        """Arrivée à marée haute = plus d'échouages. 0,85 (basse) → 1,15 (haute)."""
        if self.sl is None:
            return 1.0
        cell = self._tide_cell(bi)
        if cell is None:
            return 1.0
        day = local_date(t)
        key = (bi, day)
        g = self.sl
        if key not in self.tide_cache:
            idx = np.array([k for k, tt in enumerate(g.times) if local_date(tt) == day], int)
            vals = g.data["zos"][idx, cell[0], cell[1]] if len(idx) else np.array([])
            ok = np.isfinite(vals)
            self.tide_cache[key] = (g.times[idx][ok], vals[ok])
        times, vals = self.tide_cache[key]
        if len(vals) < 3:
            return 1.0
        lo, hi = vals.min(), vals.max()
        if hi - lo < 0.05:
            return 1.0
        k = int(np.argmin(np.abs((times - t) / np.timedelta64(1, "m"))))
        return float(0.85 + 0.30 * (vals[k] - lo) / (hi - lo))

    # ---- simulation
    def run(self, t_now, t_end, seeds):
        """
        seeds : dict avec lat, lon, t0 (datetime64), w (poids), kind (0/1), origin (0 = fond, 1 = observation).
        Retourne arrivals[(beach_id, date)] = {"all": poids, "obs": poids venant d'observations}.
        """
        py, px = seeds["lat"].astype(float).copy(), seeds["lon"].astype(float).copy()
        t0, w, kind, origin = seeds["t0"], seeds["w"], seeds["kind"], seeds["origin"]
        alive = np.ones(len(py), bool)
        arrivals = {}
        t = t0.min() if len(t0) else t_now
        step = np.timedelta64(C.DT // 60, "m")
        bids = [b["id"] for b in self.beaches]

        def record(idx, t, radius):
            if not len(idx):
                return np.zeros(0, bool)
            d = km_between(py[idx, None], px[idx, None], self.blat[None, :], self.blon[None, :])
            nb = np.argmin(d, axis=1)
            hit = d[np.arange(len(idx)), nb] <= radius
            if t >= t_now:
                day = local_date(t)
                for n, i in zip(nb[hit], idx[hit]):
                    m = self.tide_mult(n, t)
                    a = arrivals.setdefault((bids[n], day), {"all": 0.0, "obs": 0.0})
                    a["all"] += w[i] * m
                    if origin[i] == 1:
                        a["obs"] += w[i] * m
            return hit

        while t < t_end:
            act = np.where(alive & (t0 <= t))[0]
            if len(act):
                y, x, kd = py[act], px[act], kind[act]
                u1, v1 = self.velocity(t, y, x, kd)
                cos = np.cos(np.radians(y))
                ym = y + np.nan_to_num(v1) * C.DT / 2 / M_PER_DEG
                xm = x + np.nan_to_num(u1) * C.DT / 2 / (M_PER_DEG * cos)
                u2, v2 = self.velocity(t + step // 2, ym, xm, kd)
                u2 = np.where(np.isnan(u2), u1, u2); v2 = np.where(np.isnan(v2), v1, v2)
                ny = y + v2 * C.DT / M_PER_DEG
                nx = x + u2 * C.DT / (M_PER_DEG * cos)
                landed = ~self.is_sea(t, y, x) | ~self.is_sea(t, ny, nx) | np.isnan(ny)
                py[act] = np.where(landed, y, ny); px[act] = np.where(landed, x, nx)
                rb = C.REGION_BOX
                out = (py[act] < rb["lat_min"] - .3) | (py[act] > rb["lat_max"] + .3) | \
                      (px[act] < rb["lon_min"] - .3) | (px[act] > rb["lon_max"] + .3)
                li = act[landed]
                record(li, t, C.R_BEACHED_KM); alive[li] = False
                fi = act[~landed & ~out]
                near = fi[self.near_beach(py[fi], px[fi])]
                hit = record(near, t, C.R_NEAR_KM)
                alive[near[hit]] = False
                alive[act[out]] = False
            t = t + step
        return arrivals


# ====================================================================== semis (points de départ)
def make_seeds(drift, hab, obs, t_now, box=None, spacing=None, lookback_days=None):
    rng = np.random.default_rng(42)
    lb = box or C.LOCAL_BOX
    spacing = spacing or C.SEED_SPACING
    lookback_days = lookback_days or C.LOOKBACK_DAYS
    sy = np.arange(lb["lat_min"], lb["lat_max"], spacing)
    sx = np.arange(lb["lon_min"], lb["lon_max"], spacing)
    PY, PX = np.meshgrid(sy, sx, indexing="ij")
    py, px = PY.ravel(), PX.ravel()
    sea = drift.is_sea(t_now, py, px)
    py, px = py[sea], px[sea]
    wb = habitat_field(hab, py, px, t_now)            # plus de méduses de fond là où l'eau leur convient
    n_bg = len(py)
    lat, lon, t0, w, origin = [py], [px], [np.full(n_bg, t_now)], [wb], [np.zeros(n_bg, int)]
    n_obs = 0
    for o in obs:
        if o.get("none_seen") or o["weight"] <= 0:
            continue
        if not (C.REGION_BOX["lat_min"] <= o["lat"] <= C.REGION_BOX["lat_max"] and C.REGION_BOX["lon_min"] <= o["lon"] <= C.REGION_BOX["lon_max"]):
            continue
        age_d = (t_now - o["time"]) / np.timedelta64(1, "D")
        if age_d < 0 or age_d > lookback_days:
            continue
        n = C.PARTICLES_PER_OBS
        r = o.get("spread_km", C.OBS_SPREAD_KM)
        cy, cx = o["lat"], o["lon"]
        dy = rng.normal(0, r, n) / 111.32
        dx = rng.normal(0, r, n) / (111.32 * math.cos(math.radians(cy)))
        ok = drift.is_sea(o["time"], cy + dy, cx + dx)
        if ok.sum() < n / 2:
            # observation faite depuis la plage : la grille (4 km) la voit « à terre » → on repart de la mer la plus proche
            ns = drift.nearest_sea(o["time"], cy, cx)
            if ns is not None:
                cy, cx = ns
                dy = rng.normal(0, min(r, 2.0), n) / 111.32
                dx = rng.normal(0, min(r, 2.0), n) / (111.32 * math.cos(math.radians(cy)))
                ok = drift.is_sea(o["time"], cy + dy, cx + dx)
        ly, lx = cy + dy, cx + dx
        if not ok.any():
            continue
        ly, lx = ly[ok], lx[ok]
        # un banc observé = beaucoup de méduses : chaque observation pèse comme ~40 méduses de fond, puis s'estompe
        wt = 40.0 * o["weight"] * math.exp(-age_d / 6) / len(ly)
        lat.append(ly); lon.append(lx); t0.append(np.full(len(ly), o["time"])); w.append(np.full(len(ly), wt))
        origin.append(np.ones(len(ly), int)); n_obs += 1
    lat = np.concatenate(lat); lon = np.concatenate(lon)
    seeds = dict(lat=lat, lon=lon, t0=np.concatenate(t0).astype("datetime64[m]"), w=np.concatenate(w),
                 origin=np.concatenate(origin))
    kind = (rng.random(len(lat)) < C.TYPES["migrante"]["share"]).astype(int)
    seeds["kind"] = kind
    print(f"[semis] {n_bg} méduses de fond + {len(lat) - n_bg} issues de {n_obs} observations")
    return seeds, n_bg, n_obs


# ====================================================================== facteurs par plage et par jour
def onshore_wind_past(wind, b, t_day_start, hours=72):
    """Vent moyen vers la plage (m/s) sur les heures précédant le début du jour."""
    g = wind
    i = int(np.argmin(np.abs(g.lats - b["lat"]))); j = int(np.argmin(np.abs(g.lons - b["lon"])))
    f = math.radians(b["face"]); ox, oy = -math.sin(f), -math.cos(f)
    sel = (g.times >= t_day_start - np.timedelta64(hours, "h")) & (g.times < t_day_start)
    if not sel.any():
        return 0.0, 0.0
    u, v = g.data["u"][sel, i, j], g.data["v"][sel, i, j]
    push = np.clip(u * ox + v * oy, 0, None)
    return float(push.mean()), float(np.hypot(u, v).mean() * 3.6)


def day_wind_speed(wind, b, day):
    g = wind
    i = int(np.argmin(np.abs(g.lats - b["lat"]))); j = int(np.argmin(np.abs(g.lons - b["lon"])))
    sel = np.array([local_date(t) == day for t in g.times])
    if not sel.any():
        return 0.0
    return float(np.hypot(g.data["u"][sel, i, j], g.data["v"][sel, i, j]).mean() * 3.6)


def obs_persistence(obs, b, t_day_start):
    """Observations proches (< 10 km) dans les 3 jours avant : présence probable. « Rien vu » fait baisser."""
    s, neg, signal = 0.0, 0.0, False
    for o in obs:
        age_h = (t_day_start - o["time"]) / np.timedelta64(1, "h")
        if age_h < -24 or age_h > 72:
            continue
        d = float(km_between(b["lat"], b["lon"], o["lat"], o["lon"]))
        if o["source"] == "presse":
            continue
        if d > 10:
            continue
        decay = math.exp(-max(0, age_h) / 36) * (1 - d / 12)
        if o.get("none_seen"):
            neg += 0.5 * decay
        else:
            s += o["weight"] * decay
            if d <= 5 and age_h <= 48:
                signal = True
    return float(np.clip(1 - math.exp(-s) - 0.3 * neg, 0, 1)), signal


def news_factor(obs, b, t_day_start):
    n = sum(1 for o in obs if o["source"] == "presse" and
            float(km_between(b["lat"], b["lon"], o["lat"], o["lon"])) < 60 and
            0 <= (t_day_start - o["time"]) / np.timedelta64(1, "D") <= 7)
    return float(min(1.0, 0.5 * n))


def likely_species(month, sst):
    out = []
    for s in C.SPECIES:
        if month in s["months"] and (sst is None or s["t_min"] <= sst <= s["t_max"]):
            out.append(dict(name=s["name"], sting=s["sting"]))
    return out
