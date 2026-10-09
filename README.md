# Win N3oum ?

La bonne plage, au bon moment. Application web de baignade pour Djerba (bientôt toute la Tunisie).

- `index.html` : l'application (hébergée par GitHub Pages).
- `drift/run_drift.py` : modèle de dérive des méduses (courants Copernicus Marine + vent Open-Meteo).
- `.github/workflows/meduses.yml` : lance le modèle chaque nuit et met à jour `data/meduses.json`.
- `data/meduses.json` : le risque méduses par plage pour aujourd'hui, demain et après-demain.

Secrets nécessaires (Settings → Secrets and variables → Actions) : `COPERNICUS_USER`, `COPERNICUS_PASSWORD`.

Indicatif seulement : le modèle méduses est expérimental et sera réglé avec les signalements de l'été 2027.
