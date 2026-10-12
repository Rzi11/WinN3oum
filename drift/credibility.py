"""
Signalements des utilisateurs : les faire compter sans se faire piéger.

Chaque signalement pèse dans le calcul, mais son poids dépend de 5 vérifications croisées :
  1. Réputation de l'appareil  : signalements passés confirmés / contredits (modèle bêta, part de 50 %).
                                 Appareil neuf plafonné ; sentinelles (maîtres-nageurs, hôtels…) très fiables.
  2. Présence sur place        : position du téléphone (si acceptée) à moins de 1,5 km de la plage.
  3. Plausibilité              : le signalement colle-t-il au modèle (dérive, eau, saison) et à l'espèce possible ?
  4. Recoupement               : d'autres personnes indépendantes (autres réseaux), iNaturalist/OBIS/GBIF disent pareil ?
                                 Plusieurs appareils sur le même réseau = une seule voix.
  5. Contradiction             : les « rien vu » des autres font baisser.

Fusion : on part du risque du modèle et on ajoute le poids de chaque signalement en « log-cote »
(méthode bayésienne). Un signalement seul et non vérifié peut faire monter d'un cran, jamais créer
une alerte « Signalé » : il faut un recoupement.

Chaque nuit, la réputation des appareils est mise à jour : confirmé → plus de poids la fois suivante,
contredit → moins ; un appareil qui ment souvent est banni sans le savoir (ses envois sont ignorés).
"""
import os, math, json, datetime as dt
import numpy as np
import config as C
import model as M

SUPA_URL = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
for _suffix in ("/rest/v1", "/rest/v1/rpc"):        # tolère l'adresse copiée depuis la page « Data API »
    if SUPA_URL.endswith(_suffix):
        SUPA_URL = SUPA_URL[: -len(_suffix)]
SUPA_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

MAX_LLR_PUBLIC = 1.0        # poids maximum d'une source non vérifiée (≈ fait passer 20 % → 40 %)
MAX_LLR_SENTINEL = 2.5
NEW_DEVICE_CAP = 0.45       # confiance max tant qu'un appareil a moins de 3 signalements
ONSITE_KM = 1.5
BAN_MARGIN = 6              # banni quand contredit 6 fois de plus que confirmé


def enabled():
    return bool(SUPA_URL and SUPA_KEY)


def _headers():
    return {"apikey": SUPA_KEY, "Authorization": f"Bearer {SUPA_KEY}", "Content-Type": "application/json"}


def fetch_reports(days=14):
    import requests
    since = (dt.datetime.utcnow() - dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = requests.get(f"{SUPA_URL}/rest/v1/reports", headers=_headers(), timeout=60, params={
        "select": "id,created_at,device_id,beach_id,lat,lon,user_lat,user_lon,user_acc,seen,amount,species,stung,ip_hash,status,devices(alpha,beta,role,banned,n_reports)",
        "created_at": f"gte.{since}", "status": "neq.rejected", "order": "created_at.asc", "limit": "5000"})
    r.raise_for_status()
    out = []
    for x in r.json():
        d = x.get("devices") or {}
        if d.get("banned"):
            continue
        x["time"] = np.datetime64(x["created_at"][:16], "m")
        x["dev"] = d
        out.append(x)
    print(f"[signalements] {len(out)} signalements sur {days} jours")
    return out


# ---------------------------------------------------------------- briques
def trust(dev):
    a, b = float(dev.get("alpha", 2)), float(dev.get("beta", 2))
    t = a / (a + b)
    if dev.get("role") == "sentinelle":
        return max(t, 0.85)
    if int(dev.get("n_reports", 0)) < 3:
        t = min(t, NEW_DEVICE_CAP)
    return t


def onsite_factor(rep, b):
    if rep.get("user_lat") is None:
        return 0.6, None                         # position non partagée : crédit partiel
    d = float(M.km_between(rep["user_lat"], rep["user_lon"], b["lat"], b["lon"]))
    acc = (rep.get("user_acc") or 50) / 1000
    if d - acc <= ONSITE_KM:
        return 1.0, round(d, 1)
    if d <= 10:
        return 0.7, round(d, 1)
    return 0.3, round(d, 1)                       # signale une plage où il n'est pas


def plausibility(rep, prior, species_names):
    """Le signalement colle-t-il au reste ? (le modèle n'est pas validé : effet volontairement modéré)."""
    if rep["seen"]:
        f = 0.6 + 0.8 * prior                     # 0,6 (modèle dit « très peu probable ») → 1,4
    else:
        f = 0.6 + 0.8 * (1 - prior)
    sp = (rep.get("species") or "").lower()
    if rep["seen"] and sp and species_names and not any(sp[:6] in s.lower() for s in species_names):
        f *= 0.75                                 # espèce hors saison / hors température
    return float(np.clip(f, 0.4, 1.4))


def base_llr(t):
    """Poids d'un « j'ai vu » selon la fiabilité t de la source (rapport de vraisemblance)."""
    p_true = t * 0.9 + (1 - t) * 0.3              # dit « vu » quand il y en a
    p_false = t * 0.1 + (1 - t) * 0.3             # dit « vu » quand il n'y en a pas
    return math.log(p_true / p_false)


def logit(p):
    p = min(max(p, 0.02), 0.98)
    return math.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def match_beach(rep, beaches):
    for b in beaches:
        if b["id"] == rep["beach_id"]:
            return b
    d = [float(M.km_between(rep["lat"], rep["lon"], b["lat"], b["lon"])) for b in beaches]
    k = int(np.argmin(d))
    return beaches[k] if d[k] <= 5 else None


def score_report(rep, b, prior, species_names, ref_time):
    """Poids signé (log-cote) d'un signalement pour un moment donné, avec le détail pour l'explication."""
    t = trust(rep["dev"])
    on, dist = onsite_factor(rep, b)
    pl = plausibility(rep, prior, species_names)
    age_h = max(0.0, (ref_time - rep["time"]) / np.timedelta64(1, "h"))
    decay = math.exp(-age_h / 12)
    llr = base_llr(t) * on * pl * decay
    if rep["seen"]:
        llr *= 1 + 0.15 * (int(rep.get("amount", 1)) - 1)     # « beaucoup » pèse un peu plus
        if rep.get("stung"):
            llr *= 1.1
    else:
        llr = -0.7 * llr                                        # « rien vu » : un peu moins informatif
    cap = MAX_LLR_SENTINEL if rep["dev"].get("role") == "sentinelle" else MAX_LLR_PUBLIC
    llr = float(np.clip(llr, -cap, cap))
    return llr, dict(trust=round(t, 2), onsite=on, dist_km=dist, plaus=round(pl, 2), age_h=round(age_h, 1))


# ---------------------------------------------------------------- fusion avec le modèle
def fuse(out, reports, science_obs, beaches, levels):
    """Modifie le risque de chaque plage (jour 0 et 1) avec les signalements ; ajoute un résumé par plage."""
    for b in beaches:
        bo = out["beaches"].get(b["id"])
        if bo is None:
            continue
        species_names = [s["name"] for s in bo.get("species", [])]
        mine = [r for r in reports if (mb := match_beach(r, beaches)) is not None and mb["id"] == b["id"]]
        for di, row in enumerate(bo["days"][:2]):
            ref = np.datetime64(row["date"] + "T12:00", "m") - np.timedelta64(C.UTC_OFFSET, "h")
            prior = row["risk"]
            recent = [r for r in mine if np.timedelta64(0, "h") <= ref - r["time"] <= np.timedelta64(36, "h")
                      or (di == 0 and abs((ref - r["time"]) / np.timedelta64(1, "h")) <= 12)]
            if not recent:
                continue
            # une voix par appareil (le plus récent), puis une voix par réseau (la plus forte)
            latest = {}
            for r in sorted(recent, key=lambda x: x["time"]):
                latest[r["device_id"]] = r
            by_net = {}
            details = []
            for r in latest.values():
                llr, det = score_report(r, b, prior, species_names, ref)
                det.update(seen=r["seen"], amount=r.get("amount", 0))
                details.append(det)
                fresh, _ = score_report(r, b, prior, species_names, r["time"])   # poids sans l'effet du temps
                k = r.get("ip_hash") or r["device_id"]
                if k not in by_net or abs(llr) > abs(by_net[k][0]):
                    by_net[k] = (llr, r, fresh)
            total = sum(v[0] for v in by_net.values())
            post = sigmoid(logit(max(prior, 0.03)) + total)
            # recoupement : ≥ 2 réseaux indépendants qui ont vu, ou une sentinelle, ou 1 + observation scientifique
            pos = [v for v in by_net.values() if v[1]["seen"] and v[2] > 0.2]
            sci = any(o["source"] in ("iNaturalist", "GBIF", "OBIS") and
                      float(M.km_between(o["lat"], o["lon"], b["lat"], b["lon"])) <= 5 and
                      abs((ref - o["time"]) / np.timedelta64(1, "h")) <= 48 for o in science_obs)
            sentinel = any(v[1]["dev"].get("role") == "sentinelle" and v[1]["seen"] for v in by_net.values())
            corroborated = len(pos) >= 2 or sentinel or (len(pos) >= 1 and sci)
            row["risk_model"] = row["risk"]
            row["risk"] = round(float(post), 2)
            lv = "Faible" if post < levels[0] else "Possible" if post < levels[1] else "Élevé"
            if corroborated and post >= 0.5:
                lv = "Signalé"
            elif row["level"] == "Signalé":
                lv = "Signalé"
            row["level"] = lv
            n_seen = sum(1 for r in latest.values() if r["seen"]); n_none = len(latest) - n_seen
            row["reports"] = dict(seen=n_seen, none_seen=n_none, networks=len(by_net), corroborated=corroborated,
                                  shift=round(float(post - prior), 2))
            msg = (f"{n_seen} baigneur{'s' if n_seen > 1 else ''} en {'ont' if n_seen > 1 else 'a'} vu" if n_seen else "")
            if n_none:
                msg += (" · " if msg else "") + f"{n_none} « rien vu »"
            has_sent = any(v[1]["dev"].get("role") == "sentinelle" for v in by_net.values())
            tag = " (sentinelle)" if has_sent else " (recoupé)" if corroborated else " (non vérifié)"
            if msg:
                row["why"] = ([msg + tag] + row.get("why", []))[:3]
    return out


# ---------------------------------------------------------------- réputation (chaque nuit)
def update_reputation(reports, science_obs, out, beaches):
    """
    Pour chaque signalement de plus de 18 h pas encore évalué : est-il confirmé par les AUTRES sources ?
    Confirmé → +1 confiance. Contredit → +1 méfiance. Le modèle seul ne fait que des petits ajustements.
    """
    import requests
    now = np.datetime64(dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M"), "m")
    todo = [r for r in reports if r.get("status") == "new" and (now - r["time"]) > np.timedelta64(18, "h")]
    delta = {}
    scored = []
    for r in todo:
        b = match_beach(r, beaches)
        if b is None:
            scored.append(r["id"]); continue
        others = [o for o in reports if o["device_id"] != r["device_id"] and (o.get("ip_hash") != r.get("ip_hash") or not r.get("ip_hash"))
                  and match_beach(o, beaches) is not None and match_beach(o, beaches)["id"] == b["id"]
                  and abs((o["time"] - r["time"]) / np.timedelta64(1, "h")) <= 4]
        agree = sum(1 for o in others if o["seen"] == r["seen"])
        disagree = sum(1 for o in others if o["seen"] != r["seen"])
        sci = any(float(M.km_between(o["lat"], o["lon"], b["lat"], b["lon"])) <= 5 and
                  abs((o["time"] - r["time"]) / np.timedelta64(1, "h")) <= 24 for o in science_obs)
        da = db = 0.0
        if r["seen"] and sci:
            da += 1
        da += min(agree, 3) * 0.7
        db += min(disagree, 3) * 0.7
        # contradiction forte avec le modèle seulement si personne d'autre ne s'est prononcé
        if not others and not sci and r["seen"]:
            prior = None
            for row in out["beaches"].get(b["id"], {}).get("days", []):
                if row["date"] == M.local_date(r["time"]):
                    prior = row.get("risk_model", row["risk"])
            if prior is not None and prior < 0.1:
                db += 0.25
        d = delta.setdefault(r["device_id"], [0.0, 0.0, r["dev"]])
        d[0] += da; d[1] += db
        scored.append(r["id"])
    n_ban = 0
    for dev_id, (da, db, dev) in delta.items():
        a = float(dev.get("alpha", 2)) + da
        bb = float(dev.get("beta", 2)) + db
        body = {"alpha": round(a, 2), "beta": round(bb, 2)}
        if bb - a >= BAN_MARGIN and dev.get("role") != "sentinelle":
            body["banned"] = True; n_ban += 1
        requests.patch(f"{SUPA_URL}/rest/v1/devices", headers=_headers(), params={"id": f"eq.{dev_id}"},
                       data=json.dumps(body), timeout=30).raise_for_status()
    for i in range(0, len(scored), 100):
        ids = ",".join(str(x) for x in scored[i:i + 100])
        requests.patch(f"{SUPA_URL}/rest/v1/reports", headers=_headers(), params={"id": f"in.({ids})"},
                       data=json.dumps({"status": "scored"}), timeout=30).raise_for_status()
    print(f"[réputation] {len(scored)} signalements évalués, {len(delta)} appareils mis à jour, {n_ban} bannis")
    return dict(evalues=len(scored), appareils=len(delta), bannis=n_ban)


def as_seed_obs(reports, beaches):
    """Signalements utilisés comme points de départ de la dérive (poids = fiabilité)."""
    out = []
    for r in reports:
        if not r["seen"]:
            continue
        b = match_beach(r, beaches)
        t = trust(r["dev"])
        on, _ = onsite_factor(r, b) if b else (0.6, None)
        w = t * on
        if w < 0.3:
            continue
        out.append(dict(lat=r["lat"], lon=r["lon"], time=r["time"], taxon=r.get("species") or "?", source="signalement",
                        weight=round(min(1.5, w * (1 + 0.2 * (r.get("amount", 1) - 1))), 2), spread_km=1.5,
                        ref=f"rep:{r['id']}"))
    return out
