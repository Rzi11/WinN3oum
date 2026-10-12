"""
Calibrage automatique : le modèle apprend lui-même ses poids à partir de data/training.csv.

  python drift/calibrate.py

Méthode :
  - régression logistique régularisée (simple, robuste, lisible : un poids par facteur)
  - validation croisée PAR GROUPES (une simulation = un groupe) : on teste toujours sur des jours et des
    régions que le modèle n'a pas vus pendant l'apprentissage → pas de « par cœur »
  - comparaison avec les poids manuels actuels sur exactement les mêmes cas (AUC : 0,5 = hasard, 1 = parfait)
  - les poids appris ne sont ADOPTÉS que s'il y a assez d'observations et qu'ils battent nettement les poids manuels
  - l'« effort d'observation » est appris puis neutralisé à la prévision (on veut la présence des méduses,
    pas la présence des observateurs)

Résultat : data/calibration.json, lu automatiquement par la prévision de la nuit.
"""
import os, sys, csv, json, math, datetime as dt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P_TRAIN = os.path.join(ROOT, "data", "training.csv")
P_CALIB = os.path.join(ROOT, "data", "calibration.json")
FEATS = ["drift", "obs", "calm", "habitat", "exposure", "log_arrivals", "from_obs", "season", "effort"]


def load():
    rows = []
    with open(P_TRAIN, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                x = []
                for k in FEATS:
                    v = r.get(k, "")
                    x.append(float(v) if v not in ("", None) else (0.5 if k == "habitat" else 0.0))
                rows.append((x, int(r["label"]), float(r["weight"] or 1), f"{r['region']}|{r['run']}", r["region"]))
            except Exception:
                continue
    X = np.array([r[0] for r in rows], float)
    y = np.array([r[1] for r in rows], int)
    w = np.array([r[2] for r in rows], float)
    g = np.array([r[3] for r in rows])
    reg = np.array([r[4] for r in rows])
    return X, y, w, g, reg


def manual_score(X):
    """Score des poids manuels actuels (config.W), recalculé à partir des mêmes facteurs."""
    i = {k: FEATS.index(k) for k in FEATS}
    W = C.W
    P = (W["drift"] * X[:, i["drift"]] + W["habitat"] * X[:, i["habitat"]] + W["obs"] * X[:, i["obs"]] +
         W["calm"] * X[:, i["calm"]]) / (W["drift"] + W["habitat"] + W["obs"] + W["calm"] + W["news"])
    season = np.where(X[:, i["season"]] > 0, X[:, i["season"]], 0.5)
    return season * (C.BASE_PRESENCE + (1 - C.BASE_PRESENCE) * P) * X[:, i["exposure"]]


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    if not os.path.exists(P_TRAIN):
        print("[calibrage] pas encore de training.csv"); return
    X, y, w, g, reg = load()
    n_pos, n = int(y.sum()), len(y)
    print(f"[calibrage] {n} exemples, {n_pos} avec méduses, {len(set(g))} simulations, régions : {sorted(set(reg))}")
    out = dict(generated=dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z", version=C.VERSION,
               n_rows=n, n_pos=n_pos, n_runs=len(set(g)), regions=sorted(set(reg)), adopted=False)
    if n_pos < 10 or len(set(g)) < 4 or n_pos == n:
        out["reason"] = "pas assez de données"
        json.dump(out, open(P_CALIB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("[calibrage] pas assez de données pour apprendre"); return

    folds = min(5, len(set(g)))
    gkf = GroupKFold(n_splits=folds)
    best = None
    for Cr in (0.03, 0.1, 0.3, 1.0):
        pred = np.zeros(n); man = np.zeros(n)
        for tr, te in gkf.split(X, y, g):
            if len(set(y[tr])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(C=Cr, class_weight="balanced", max_iter=2000)
            m.fit(sc.transform(X[tr]), y[tr], sample_weight=w[tr])
            Xte = X[te].copy()
            Xte[:, FEATS.index("effort")] = X[tr][:, FEATS.index("effort")].mean()   # effort neutralisé
            pred[te] = m.predict_proba(sc.transform(Xte))[:, 1]
            man[te] = manual_score(X[te])
        if len(set(y)) < 2:
            continue
        auc = roc_auc_score(y, pred, sample_weight=w)
        auc_man = roc_auc_score(y, man, sample_weight=w)
        print(f"[calibrage] régularisation C={Cr} : AUC appris {auc:.3f} vs manuel {auc_man:.3f}")
        if best is None or auc > best[1]:
            best = (Cr, auc, auc_man)
    Cr, auc, auc_man = best
    # AUC par région (sur la meilleure régularisation, recalculée)
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=Cr, class_weight="balanced", max_iter=2000).fit(sc.transform(X), y, sample_weight=w)
    coef_std = m.coef_[0]
    coef = coef_std / sc.scale_
    intercept = float(m.intercept_[0] - np.sum(coef_std * sc.mean_ / sc.scale_))
    effort_mean = float(X[:, FEATS.index("effort")].mean())
    Xn = X.copy(); Xn[:, FEATS.index("effort")] = effort_mean
    p_all = 1 / (1 + np.exp(-(Xn @ coef + intercept)))
    q = np.quantile(p_all, [0.8, 0.95])
    qs = np.quantile(p_all, np.linspace(0, 1, 101))           # pour convertir une probabilité en rang (percentile)
    lo = X.min(axis=0); hi = X.max(axis=0)
    adopt = n_pos >= C.CALIB_MIN_POS and auc >= auc_man + C.CALIB_MIN_GAIN
    out.update(adopted=bool(adopt), auc_cv=round(float(auc), 3), auc_manual=round(float(auc_man), 3), C=Cr,
               features=FEATS, coef=[round(float(c), 5) for c in coef], intercept=round(intercept, 5),
               effort_mean=round(effort_mean, 4),
               importance={k: round(float(v), 3) for k, v in zip(FEATS, coef_std)},
               levels=[round(float(q[0]), 4), round(float(q[1]), 4)],
               quantiles=[round(float(v), 6) for v in qs],
               feat_min=[round(float(v), 4) for v in lo], feat_max=[round(float(v), 4) for v in hi],
               reason=("adopté" if adopt else
                       f"non adopté : {'pas assez d’observations' if n_pos < C.CALIB_MIN_POS else 'ne bat pas assez les poids manuels'}"))
    os.makedirs(os.path.dirname(P_CALIB), exist_ok=True)
    with open(P_CALIB, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[calibrage] AUC appris {auc:.3f} / manuel {auc_man:.3f} → {out['reason']}")
    print(f"[calibrage] importance (poids standardisés) : {out['importance']}")


if __name__ == "__main__":
    main()
