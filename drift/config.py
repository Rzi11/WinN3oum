"""
Win N3oum ? — réglages du modèle méduses (v0.2).

Tous les chiffres marqués « À RÉGLER » sont des hypothèses de départ :
ils seront ajustés avec le rejeu du passé et les signalements de l'été.
"""

VERSION = "v0.3"
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
W = dict(
    drift=0.55,        # poussée physique vers la plage
    obs=0.25,          # observations récentes proches (persistance)
    wind_past=0.10,    # vent vers la plage sur les 72 dernières heures
    news=0.10,         # articles de presse récents sur la région
)
# L'eau favorable ne fait plus partie de la somme (elle était « très favorable » partout et rendait tout « Possible ») :
# elle module le risque de ×0,8 (eau défavorable) à ×1,1 (eau idéale).
HABITAT_MULT = (0.8, 1.1)
BASE_PRESENCE = 0.15               # part du risque qui ne dépend que de la saison
LEVELS = (0.2, 0.45)               # seuils Faible / Possible / Élevé — À RÉGLER avec validation.json
STRONG_WIND_KMH = 35               # au-delà : méduses brassées en profondeur, moins visibles
LUNAR_WEIGHT = 0.0                 # cycle lunaire (cuboméduses) : calculé mais pas utilisé tant que non prouvé ici

# ------------------------------------------------------------------ validation régionale
VALID_DAYS_BACK = 365              # observations des 12 derniers mois
VALID_MAX_RUNS = 20                # nb max de simulations (≈ 2–3 min chacune)
VALID_LOOKBACK_DAYS = 7            # observations antérieures utilisées comme points de départ
VALID_SPACING = 0.03               # ~3 km entre méduses de fond (zone régionale, plus grande)
VALID_POINT_SPACING_KM = 12        # un point témoin tous les ~12 km de côte
VALID_SNAP_KM = 10                 # observation rattachée au point de côte le plus proche (< 10 km)

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
# face : direction vers laquelle la plage regarde la mer (0 = Nord)
# shelter : 1 si la plage est encadrée par un cap/digue (piège à méduses, cf. étude Costa del Sol) — À VÉRIFIER
# wadi : 1 si arrivée d'eau douce proche (moins de méduses)
BEACHES = [
    dict(id="rmel",       name="Ras R'mel",      zone="djerba", lat=33.895, lon=10.925, face=0,   shelter=1, wadi=0),
    dict(id="seguia",     name="La Seguia",      zone="djerba", lat=33.735, lon=11.005, face=125, shelter=0, wadi=0),
    dict(id="mahrez",     name="Sidi Mahrez",    zone="djerba", lat=33.858, lon=10.985, face=20,  shelter=0, wadi=0),
    dict(id="aghir",      name="Aghir",          zone="djerba", lat=33.772, lon=11.025, face=100, shelter=0, wadi=0),
    dict(id="taguermess", name="Ras Taguermess", zone="djerba", lat=33.825, lon=11.055, face=65,  shelter=0, wadi=0),
    dict(id="kantara",    name="El Kantara",     zone="djerba", lat=33.665, lon=10.955, face=155, shelter=1, wadi=0),
    dict(id="jmour",      name="Sidi Jmour",     zone="djerba", lat=33.848, lon=10.745, face=295, shelter=0, wadi=0),
]

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
