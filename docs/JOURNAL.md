# Journal

## 4 octobre 2026

### Installation
- Pas de GPU, Mac M1 8 Go. LeRobot dit que l'extra `libero` est Linux only (marqueur `sys_platform == "linux"` sur `hf-libero`). En fait c'est seulement `egl-probe` (dépendance de robomimic) qui ne compile pas sur macOS. LIBERO n'en a pas besoin : `hf-libero`, `robosuite==1.4.0`, `bddl` installés en `--no-deps`, puis les vraies dépendances à la main. MuJoCo rend en offscreen avec `MUJOCO_GL=cgl`. Tout tourne en local, Colab ne sert que pour passer à l'échelle.
- Versions : lerobot 0.6.1, torch 2.11.0, transformers 5.5.4, mujoco 3.8.1, robosuite 1.4.0, hf-libero 0.1.4. Checkpoint `lerobot/smolvla_libero` révision `31d453f7edd7`.
- Le téléchargement du checkpoint bloquait à 0 octet via xet. `HF_HUB_DISABLE_XET=1` a réglé le problème.

### Pièges rencontrés
- **Noms de caméras.** Le checkpoint attend `observation.images.camera1/2/3`, l'env fournit `image/image2`. `make_policy` refuse de charger sans `rename_map`. Le preprocessor sauvegardé contient déjà le renommage, mais `lerobot-eval` l'écrase avec `--rename_map` (vide par défaut) : il faut le passer explicitement en CLI. `camera3` reste absente, SmolVLA la traite comme caméra vide.
- **Batch.** `LiberoProcessorStep` attend une dimension batch sur l'état ; avec un env non vectorisé il faut l'ajouter soi-même.
- **Perturbations invisibles (bug silencieux).** Première grille d'images : caméra et lumière ne changeaient rien. Deux causes : la pose monde des caméras n'est recalculée que par `mj_forward`, et robosuite garde les observations en cache entre deux pas de simulation (`_get_observations(force_update=True)` nécessaire). Le décalage articulaire marchait parce qu'il fait des pas de simulation. Ajout de tests qui vérifient que l'image change avec l'intensité.
- **Mémoire.** Avec les poids en fp32 et un env gardé en mémoire par tâche, le process montait à 6,5 Go, 9 Go de swap, épisodes 2-3 fois plus lents. Passage à un seul env vivant (ordre des jobs par tâche) et poids gardés en bf16 (format du checkpoint, comme sur CUDA). fp32 est ~20 % plus rapide par appel en isolé (0,93 s contre 1,15 s), mais pas quand la machine swappe. Les 4 épisodes déjà faits en fp32 ont été écartés pour ne pas mélanger les précisions.
- Un seul process d'évaluation à la fois sur 8 Go ; les configs passent l'une après l'autre (`scripts/run_queue.sh`).

### Choix
- Chunk de 50 actions exécuté en entier (`n_action_steps=50`, valeur du checkpoint). ~3 appels au modèle par épisode réussi, 6 par échec.
- Résolution 360x360 (défaut LeRobot pour LIBERO et valeur du train config du checkpoint).
- Épisode *e* = état initial *e* et seed *e* (y compris le bruit du flow matching) quelle que soit la condition : comparaisons appariées possibles (McNemar).
- Perturbations retenues après lecture de LIBERO-plus (qui trouve caméra et position initiale très nocives, et un langage ignoré) :
  - orbite de la caméra principale autour du point visé sur la table, 0 à 30°, signe alterné selon la parité de l'épisode ; la caméra poignet ne bouge pas ;
  - décalage de ±δ sur chacune des 7 articulations, δ de 0 à 0,2 rad, signes tirés par épisode ; les objets ne bougent pas ;
  - baisse de toutes les lumières (2 directionnelles + headlight) de 0 à 90 % ;
  - bruit gaussien sur les pixels (écart-type 0 à 60) ;
  - masquage d'une caméra (ablation).
  La grille d'images (`scripts/perturbation_grid.py`) montre que la scène reste lisible à l'intensité max, sauf la lumière à 90 % qui est très sombre.
- Test d'écoute : dans LIBERO-Goal les 10 tâches ont exactement les mêmes objets, donc je peux évaluer les 10 prédicats de but dans n'importe quel env. On sait ainsi ce que le robot a vraiment fait quand on lui dit autre chose, pas seulement s'il a raté le but de l'env.

### Premiers résultats (provisoires)
- Baseline LIBERO-Spatial : 59/100. Bien en dessous des ~90 % publiés pour SmolVLA. Les tâches 5 et 8 sont presque toujours ratées. À vérifier avec `lerobot-eval` sur les mêmes tâches pour séparer « mon pipeline » de « le checkpoint sur MPS ». Toutes les conclusions sont relatives à cette baseline.
- Durée moyenne : ~25 s par épisode réussi, ~35-50 s par échec (280 pas).
- Lumière : plateau jusqu'à 67,5 % de lumière retirée, puis effondrement à 90 % (4 %). Rupture nette, pas une dégradation lente.
- Décalage articulaire : chute dès 0,05 rad par articulation (~3°), de 59 % à 30 %. Le modèle est très sensible à la pose de départ du bras.
- Caméra : déclin progressif, fortement dépendant de la tâche.
- Matrice d'écoute (en cours) : quand on donne la consigne d'une autre tâche, le robot atteint souvent le but de la consigne donnée, pas celui de l'env. Il écoute.

### Suite
- Cible RL : décalage articulaire, parce que la perturbation est visible dans la proprioception (le correcteur n'a pas d'images). Tâche 2 de LIBERO-Spatial (100 % sans perturbation), δ = 0,05 rad. États initiaux 0-39 pour l'entraînement, 40-49 gardés pour le test.
- Avant d'entraîner : vérifier qu'un correcteur nul redonne exactement les résultats du VLA seul, et mesurer les pas/s (go/no-go).
