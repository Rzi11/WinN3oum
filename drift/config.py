"""
Win N3oum ? — réglages du modèle méduses (v0.2).

Tous les chiffres marqués « À RÉGLER » sont des hypothèses de départ :
ils seront ajustés avec le rejeu du passé et les signalements de l'été.
"""

VERSION = "v0.8"
UTC_OFFSET = 1                     # Tunisie : UTC+1, pas d'heure d'été

# ------------------------------------------------------------------ zones
# Zone locale : là où on compte les arrivées sur les plages (grille fine, méduses « de fond »).
LOCAL_BOX = dict(lon_min=10.45, lon_max=11.45, lat_min=33.40, lat_max=34.15)
# Zone régionale : d'où peuvent venir les bancs (détroit de Sicile, Malte, Lampedusa, Libye, côte est tunisienne).
REGION_BOX = dict(lon_min=10.0, lon_max=15.6, lat_min=32.4, lat_max=38.2)

FORECAST_HOURS = 72                # prévision : aujourd'hui + 2 jours
LOOKBACK_DAYS = 14                 # observations récentes utilisées comme points de départ
DT = 1800                          # pas de temps : 30 min

# ------------------------------------------------------------------ méduses virtuelles
SEED_SPACING = 0.012               # ~1,2 km entre méduses « de fond » dans la zone locale
PARTICLES_PER_OBS = 60             # méduses virtuelles lâchées autour de chaque observation
OBS_SPREAD_KM = 4.0                # rayon de dispersion autour d'une observation

# Deux comportements :
#  - « surface » : flotte en surface (méduse chou-fleur, galion, vélelle…) → courant de surface + vent + vagues
#  - « migrante » : pélagie → en surface la nuit, vers ~15 m le jour (moins de vent, courant plus profond)
TYPES = {
    "surface":  dict(share=0.5, windage=0.015, stokes=1.0, day_depth=False),   # À RÉGLER
    "migrante": dict(share=0.5, windage=0.008, stokes=0.6, day_depth=True),    # À RÉGLER
}
DAY_DEPTH_M = 15                   # profondeur de jour des pélagies

# ------------------------------------------------------------------ arrivées
R_NEAR_KM = 3.0                    # méduse à moins de 3 km d'une plage = « arrivée »
R_BEACHED_KM = 6.0                 # méduse échouée : rattachée à la plage la plus proche (< 6 km)
K_ARRIVALS = 15.0                  # échelle absolue tant qu'on n'a pas d'historique — À RÉGLER
HISTORY_DAYS = 60                  # historique par plage pour la comparaison « à elle-même »
HISTORY_MIN_RUNS = 7               # nb de nuits avant d'utiliser la comparaison à soi-même

# ------------------------------------------------------------------ poids des facteurs (À RÉGLER)
# Poids réglés d'après la 1re validation régionale (26 observations, oct. 2026) — petit échantillon, à revoir :
#   eau favorable 71 % (utile) · exposition 68 % (utile) · dérive 45 % (pas de signal à cette échelle)
#   vent de mer passé 33 % (signal INVERSE : on voit plus de méduses par mer calme → facteur « calme »)
W = dict(
    drift=0.30,        # poussée physique vers la plage (gardée : c'est elle qui fait varier le risque d'un jour à l'autre)
    habitat=0.30,      # eau favorable (température, salinité, chlorophylle, couche de surface)
    obs=0.20,          # observations récentes proches (persistance)
    calm=0.10,         # peu de vent de mer ces 3 derniers jours (eau calme = méduses visibles et baigneurs présents)
    news=0.10,         # articles de presse récents sur la région
)
BASE_PRESENCE = 0.15               # part du risque qui ne dépend que de la saison
LEVELS = (0.2, 0.45)               # seuils par défaut ; remplacés automatiquement par ceux de data/validation.json (même version)
STRONG_WIND_KMH = 35               # au-delà : méduses brassées en profondeur, moins visibles
LUNAR_WEIGHT = 0.0                 # cycle lunaire (cuboméduses) : calculé mais pas utilisé tant que non prouvé ici

# ------------------------------------------------------------------ validation régionale
VALID_DAYS_BACK = 365              # observations des 12 derniers mois
VALID_MAX_RUNS = 20                # nb max de simulations (≈ 2–3 min chacune)
VALID_LOOKBACK_DAYS = 7            # observations antérieures utilisées comme points de départ
VALID_SPACING = 0.03               # ~3 km entre méduses de fond (zone régionale, plus grande)
VALID_POINT_SPACING_KM = 12        # un point témoin tous les ~12 km de côte
VALID_SNAP_KM = 10                 # observation rattachée au point de côte le plus proche (< 10 km)

# ------------------------------------------------------------------ apprentissage méditerranéen
# Chaque nuit, quelques simulations dans une région (à tour de rôle) pour accumuler des exemples d'apprentissage.
LEARN_REGIONS = {
    "tunisie_sicile":  dict(lon_min=10.0, lon_max=15.6, lat_min=32.4, lat_max=38.2),
    "espagne_est":     dict(lon_min=-1.0, lon_max=4.6,  lat_min=37.0, lat_max=42.6),
    "espagne_sud":     dict(lon_min=-5.6, lon_max=-0.4, lat_min=35.6, lat_max=38.4),
    "ligurie_azur":    dict(lon_min=5.0,  lon_max=10.5, lat_min=41.8, lat_max=44.6),
    "tyrrhenienne":    dict(lon_min=10.5, lon_max=16.2, lat_min=38.0, lat_max=42.8),
    "adriatique":      dict(lon_min=12.2, lon_max=19.8, lat_min=40.0, lat_max=45.8),
    "grece_ionienne":  dict(lon_min=19.0, lon_max=24.0, lat_min=36.0, lat_max=40.6),
    "egee":            dict(lon_min=22.5, lon_max=28.5, lat_min=35.0, lat_max=41.0),
    "levant":          dict(lon_min=31.5, lon_max=36.2, lat_min=31.0, lat_max=37.0),
}
LEARN_BUDGET_MIN = 25              # minutes d'apprentissage par nuit (après la prévision)
LEARN_RUNS_PER_REGION = 2          # simulations par région et par passage
LEARN_SPACING = 0.04               # ~4 km entre méduses de fond (plus rapide)
LEARN_NEG_SAMPLE = 0.15            # on garde 15 % des exemples « pas d'observation » (pondérés ×1/0,15)
LEARN_EFFORT_KM = 20               # « effort d'observation » = nb d'observations à moins de 20 km (corrige le biais touristique)
CALIB_MIN_POS = 30                 # nb minimum d'observations avant d'adopter des poids appris
CALIB_MIN_GAIN = 0.02              # les poids appris doivent battre les poids manuels d'au moins 0,02 d'AUC

# ------------------------------------------------------------------ registre des bancs suivis (v0.7)
# Chaque observation crédible devient un « banc » suivi nuit après nuit : il avance avec les courants RÉELS
# des dernières 24 h (recalcul Copernicus), puis il est projeté sur quelques jours avec les courants prévus.
BANC_HORIZON_DAYS = 8              # projection : 8 jours (les courants Copernicus sont prévus sur 10 jours)
BANC_PARTICLES = 160               # méduses virtuelles par banc (= autant de scénarios : vent ±50 %, surface / 15 m)
BANC_WINDAGE_SPREAD = 0.5          # chaque méduse virtuelle subit entre 50 % et 150 % de l'effet du vent
BANC_SAIL_TAXA = ("physalia", "velella", "galion", "vélelle")   # « voiliers » : poussés beaucoup plus par le vent
BANC_SAIL_WINDAGE = 2.5            # multiplicateur de prise au vent des voiliers
BANC_MERGE_KM = 12                 # une observation à < 12 km d'un banc suivi = le même banc (il est renforcé)
BANC_HALF_LIFE_DAYS = 12           # sans nouvelle observation, la « force » d'un banc diminue de moitié en 12 jours
BANC_MAX_AGE_DAYS = 30             # au-delà, on arrête de le suivre
BANC_MAX_SPREAD_KM = 60            # trop étalé : on ne sait plus où il est
BANC_MIN_ALIVE = 0.10              # moins de 10 % des méduses virtuelles encore en mer : banc échoué / sorti
BANC_MIN_STRENGTH = 0.15
BANC_CONTRADICT = 0.7              # « rien vu » là où il devait arriver : force × 0,7
BANC_ALERT_FRAC = 0.03             # au moins 3 % des scénarios arrivent sur une plage = alerte
BANC_CONF = (0.03, 0.10, 0.30)     # part des scénarios qui arrivent : faible / moyenne / forte
BANC_ARCHIVE_DAYS = 20             # bancs terminés : encore affichés 20 jours (avec la raison)

# Lieux pour nommer un banc (« vu près de … »)
PLACES = {
    "Djerba": (33.80, 10.92), "Zarzis": (33.50, 11.12), "El Bibane": (33.25, 11.30), "Gabès": (33.90, 10.12),
    "Skhira": (34.30, 10.10), "Sfax": (34.74, 10.80), "Kerkennah": (34.70, 11.20), "Mahdia": (35.50, 11.10),
    "Monastir": (35.77, 10.86), "Sousse": (35.83, 10.66), "Hammamet": (36.38, 10.62), "Kélibia": (36.85, 11.12),
    "Pantelleria": (36.80, 11.95), "Lampedusa": (35.50, 12.60), "Linosa": (35.86, 12.86), "Malte": (35.90, 14.45),
    "sud de la Sicile": (36.90, 14.00), "Zouara (Libye)": (32.93, 12.08), "Tripoli (Libye)": (32.90, 13.18),
    "Sabratha (Libye)": (32.80, 12.48),
}

# ------------------------------------------------------------------ carte « d'où viennent les méduses de Djerba »
CONNECT_BOX = dict(lon_min=9.8, lon_max=16.5, lat_min=31.5, lat_max=38.6)
CONNECT_YEARS_BACK = 4             # 4 dernières années de courants (réanalyse Copernicus)
CONNECT_DAYS = 21                  # on remonte 21 jours en arrière
CONNECT_EVERY_DAYS = 3             # un lâcher tous les 3 jours, de mars à novembre
CONNECT_PER_BEACH = 30             # méduses virtuelles lâchées devant chaque plage
CONNECT_ZONES = {                  # zones « sources » (boîtes lat/lon) — l'ordre compte : la première qui contient le point
    "Autour de Djerba":                 (33.40, 34.05, 10.45, 11.35),
    "Zarzis – El Bibane":               (32.95, 33.40, 10.90, 11.80),
    "Golfe de Gabès (Gabès – Skhira)":  (33.60, 34.45, 9.90, 10.45),
    "Sfax – Kerkennah":                 (34.45, 35.05, 10.40, 11.70),
    "Mahdia – Monastir – Sousse":       (35.05, 36.20, 10.40, 11.40),
    "Large du golfe de Gabès":          (33.40, 35.05, 11.35, 12.60),
    "Côte libyenne (Zouara – Tripoli)": (32.30, 33.20, 11.80, 14.00),
    "Lampedusa – Linosa":               (35.05, 36.20, 12.20, 13.20),
    "Pantelleria – canal de Sicile":    (36.20, 37.40, 10.90, 12.40),
    "Malte":                            (35.50, 36.30, 13.80, 14.90),
    "Sud de la Sicile":                 (36.30, 37.60, 12.40, 15.60),
    "Large libyen et maltais":          (32.30, 36.30, 12.60, 16.50),
}

# Présence saisonnière dans le sud tunisien (0 = absentes, 1 = pleine saison) — À AFFINER avec les données Gueroun
SEASON = {1: .15, 2: .15, 3: .2, 4: .3, 5: .5, 6: .7, 7: .85, 8: .9, 9: .8, 10: .55, 11: .3, 12: .2}

# Espèces et températures où on les rencontre (indicatif, d'après la littérature)
SPECIES = [
    dict(name="Pélagie (Pelagia noctiluca)", sting="forte", t_min=13, t_max=27, months=range(1, 13)),
    dict(name="Méduse chou-fleur (Rhizostoma pulmo)", sting="faible", t_min=16, t_max=29, months=range(4, 11)),
    dict(name="Méduse œuf au plat (Cotylorhiza tuberculata)", sting="très faible", t_min=22, t_max=30, months=range(7, 11)),
    dict(name="Rhopilema nomadica (méduse nomade)", sting="forte", t_min=25, t_max=31, months=range(7, 11)),
    dict(name="Galion portugais (Physalia physalis)", sting="très forte", t_min=14, t_max=24, months=range(3, 7)),
]

# ------------------------------------------------------------------ plages
# La liste vient de plages.json (à la racine), partagée avec l'app. La liste ci-dessous ne sert que de secours.
# face : direction vers laquelle la plage regarde la mer (0 = Nord)
# shelter : 1 si la plage est encadrée par un cap/digue (piège à méduses, cf. étude Costa del Sol) — À VÉRIFIER
# wadi : 1 si arrivée d'eau douce proche (moins de méduses)
_BEACHES_SECOURS = [
    dict(id="rmel",       name="Ras R'mel",      zone="djerba", lat=33.895, lon=10.925, face=0,   shelter=1, wadi=0),
    dict(id="seguia",     name="La Seguia",      zone="djerba", lat=33.735, lon=11.005, face=125, shelter=0, wadi=0),
    dict(id="mahrez",     name="Sidi Mahrez",    zone="djerba", lat=33.858, lon=10.985, face=20,  shelter=0, wadi=0),
    dict(id="aghir",      name="Aghir",          zone="djerba", lat=33.772, lon=11.025, face=100, shelter=0, wadi=0),
    dict(id="taguermess", name="Ras Taguermess", zone="djerba", lat=33.825, lon=11.055, face=65,  shelter=0, wadi=0),
    dict(id="kantara",    name="El Kantara",     zone="djerba", lat=33.665, lon=10.955, face=155, shelter=1, wadi=0),
    dict(id="jmour",      name="Sidi Jmour",     zone="djerba", lat=33.848, lon=10.745, face=295, shelter=0, wadi=0),
]


def _load_beaches():
    import os, json
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plages.json")
    try:
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)["plages"]
        out = []
        for b in raw:
            if b.get("region", "djerba") != "djerba":
                continue                      # le modèle méduses ne couvre que Djerba pour l'instant
            out.append(dict(id=b["id"], name=b["name"], zone=b.get("region", "djerba"), lat=float(b["lat"]), lon=float(b["lon"]),
                            face=float(b["face"]), shelter=int(b.get("shelter", 0)), wadi=int(b.get("wadi", 0))))
        if out:
            return out
    except Exception as e:
        print(f"[plages] plages.json illisible ({e}) : liste de secours")
    return _BEACHES_SECOURS


BEACHES = _load_beaches()
DJERBA_CENTER = (33.80, 10.92)

# Régions citées dans la presse → coordonnées (pour transformer un article en « observation » approximative)
NEWS_PLACES = {
    "djerba": (33.80, 10.95), "جربة": (33.80, 10.95), "zarzis": (33.50, 11.12), "جرجيس": (33.50, 11.12),
    "gabès": (33.88, 10.10), "gabes": (33.88, 10.10), "قابس": (33.88, 10.10),
    "sfax": (34.74, 10.76), "صفاقس": (34.74, 10.76), "kerkennah": (34.70, 11.20), "قرقنة": (34.70, 11.20),
    "mahdia": (35.50, 11.07), "المهدية": (35.50, 11.07), "monastir": (35.77, 10.83), "المنستير": (35.77, 10.83),
    "sousse": (35.83, 10.64), "سوسة": (35.83, 10.64), "hammamet": (36.40, 10.62), "الحمامات": (36.40, 10.62),
    "nabeul": (36.45, 10.74), "نابل": (36.45, 10.74), "kélibia": (36.85, 11.10), "kelibia": (36.85, 11.10),
    "malte": (35.90, 14.45), "malta": (35.90, 14.45), "lampedusa": (35.50, 12.60),
}
NEWS_QUERIES = [
    "méduses Tunisie", "méduses Djerba", "méduses plage Tunisie",
    "قنديل البحر تونس", "الحريقة البحر تونس",
]
