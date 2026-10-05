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

## 5 octobre 2026 (nuit)

### Matrice d'écoute terminée (300 épisodes)
- Hors diagonale, le but de l'env n'est **jamais** atteint (0/270), mais le but de la consigne donnée l'est dans 197/270 épisodes (73 %), contre 25/30 (83 %) sur la diagonale. Seulement 2/270 épisodes atteignent un troisième but sans rapport.
- Conclusion : sur LIBERO-Goal, SmolVLA suit la consigne. Ça va à l'encontre de la lecture de LIBERO-plus (« les VLA ignorent le langage »). Les deux tests ne mesurent pas la même chose : une paraphrase ne change pas le but, un échange de consigne si.
- Consignes les plus mal suivies : tâche 3 (bol dans le tiroir du haut) et tâche 6 (fromage dans le bol). Bizarrement la tâche 6 échoue 3/3 dans son propre env mais réussit 2/3 dans plusieurs autres.

### Vérifications
- `lerobot-eval` officiel sur les tâches Spatial 1 et 3 : 14/20, contre 12/20 avec ma boucle (Fisher p = 0,74). Ma boucle n'explique pas la baseline basse. Le papier SmolVLA annonce 90 % sur Spatial mais le checkpoint du Hub a été entraîné séparément (pepijn223, 25k pas) ; d'autres réévaluations publiques trouvent aussi des chiffres bien plus bas.
- Correcteur résiduel nul (dernière couche à zéro) : résultats identiques au VLA seul épisode par épisode, y compris le nombre de pas. Le pipeline est donc déterministe même sur MPS, et la plomberie du résiduel est correcte.
- Débit RL : 7,5 / 8,4 / 9,5 pas/s avec 1 / 2 / 4 envs (100k pas en ~2,9 h avec 4 envs). Go, de justesse. Désactiver le rendu des caméras entre deux appels au VLA ne change rien (34 ms/pas dans les deux cas) : le coût est la physique, pas le rendu.

### Courbes dose-réponse
- Premier ajustement logistique avec s0 libre : il partait vers s0 = 1 avec une pente énorme et un x50 négatif. s0 est maintenant fixé au taux sans perturbation (mêmes épisodes), donc x50 = intensité qui divise le succès par deux. IC par bootstrap sur les tâches.
- x50 : caméra ≈ 30° [22, 40], bras ≈ 0,063 rad [0,044 ; 0,095], lumière ≈ 0,88 [0,69 ; 0,89].
- Ajout de niveaux : bras entre 0 et 0,05 rad (là où tout se joue), lumière entre 67,5 et 90 % (où est la falaise), caméra jusqu'à 60° (la courbe n'était pas encore à la moitié à 30°). Vérifié sur la grille que les objets restent visibles à 60°.

### Erreurs de ma part
- Les vidéos ont planté à l'écriture (imageio a choisi le plugin PyAV qui n'accepte pas `quality`) après 10 min de simulation. Maintenant : MP4 via `write_video` de LeRobot, et les frames sont mises en cache avant l'écriture.
- Un `git rebase` pendant qu'une évaluation tournait a recréé son CSV : le process écrivait dans un fichier supprimé. Éval relancée. Règle : pas de rebase/checkout/pull pendant un run.

### RL résiduel (lancé)
- Tâche 2 de LIBERO-Spatial, décalage articulaire 0,05 rad (intensité 0,25), α = 0,2, 4 envs, rollout 512 pas/env (~8 épisodes par mise à jour, ~50 mises à jour en 100k pas), γ = 0,995. États initiaux 0-39 pour l'entraînement, éval sur 40-49 (jamais vus) en hard reset.

### RL résiduel, seed 0 (5 octobre, 3h30)
- 100k pas en 3 h (49 mises à jour). Succès en entraînement (avec bruit d'exploration, états 0-39) : 0,60 sur les 10 premières mises à jour, 0,70 sur les 10 dernières. |Δ| moyen de 0,035 à 0,16.
- Éval déterministe en hard reset sur les états 40-49 jamais vus : 8/10 contre 8/10 sans perturbation, **4/10 contre 5/10** au niveau d'entraînement (0,05 rad), 1/10 contre 1/10 à 0,1 rad. Pas de gain, le seul épisode discordant est en défaveur du résiduel.
- Ce que le correcteur a appris : les épisodes déjà réussis finissent plus vite (121 → 92 pas, 103 → 90). Avec γ = 0,995 une réussite plus rapide rapporte plus, donc c'est exactement ce que PPO optimise ; rattraper un échec est beaucoup plus rare à observer avec ~400 épisodes.
- Diagnostics ajoutés avant de lancer d'autres seeds : (1) même éval sur les états d'entraînement 0-19 pour distinguer « n'a rien appris » de « ne généralise pas » ; (2) alternative sans apprentissage : exécuter 10 actions par chunk au lieu de 50 (le VLA replanifie 5 fois plus souvent).
- Seed 1 gardé avec la même config : changer les hyperparamètres maintenant ferait du réglage sur l'ensemble de test.

### Variantes de consigne (5 octobre, 6h)
- Consigne vide 0/50, « sing a song » 0/50 : aucun des 10 buts n'est atteint, le robot ne fait rien d'utile.
- Paraphrases écrites à la main : 35/150 (23 %). Proche 20/50, reformulée 11/50, éloignée 4/50. « turn the stove on » marche (4/5), « switch on the stove » jamais.
- Dans aucun des 135 échecs le robot n'atteint un autre but : pas de comportement par défaut.
- Lecture : la consigne sélectionne le comportement, mais elle fonctionne comme une clé (formulations vues à l'entraînement) plutôt que comme du langage compris. Ça nuance la matrice d'écoute : le modèle écoute, mais les mots exacts.

### Diagnostics du résiduel
- Sur les états d'entraînement 0-19 à 0,05 rad : VLA 10/20, VLA + résiduel 14/20 ; les 4 paires discordantes sont toutes en faveur du résiduel (McNemar p = 0,125). Épisodes réussis plus courts : 110 → 91 pas.
- Sur les états 40-49 jamais vus : pas de gain (4/10 contre 5/10).
- Donc le correcteur a appris des corrections spécifiques aux 40 états initiaux d'entraînement et ne généralise pas. Normal avec une entrée uniquement proprioceptive et 40 états : rien ne lui permet de déduire la correction pour une configuration nouvelle autrement que par interpolation.
- Replanifier toutes les 10 actions au lieu de 50 (sans apprentissage) : 10/10, 4/10, 3/10 contre 8/10, 5/10, 1/10. Rien de significatif avec 10 épisodes, et 2,4 fois plus lent.
