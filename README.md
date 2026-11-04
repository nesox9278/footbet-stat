# FOOTBET Stats — agent statistique gratuit

L'agent tourne gratuitement sur **GitHub Actions** environ toutes les heures (GitHub retarde les tâches planifiées) et publie une page web sur **GitHub Pages**. Aucun serveur à payer, aucun ordinateur à laisser allumé.

## Ce que fait l'agent (`agent/agent.py`)
1. Récupère résultats et calendrier de 8 compétitions (Ligue 1, Premier League, LaLiga, Serie A, Bundesliga, Liga Portugal, Eredivisie, Ligue des champions) sur **football-data.org** (offre gratuite), plus la Ligue Europa (C3) et la Ligue Conférence (C4) via l'API publique d'**ESPN** (sans clé), qui fournit aussi les compositions officielles environ 1 h avant les matchs.
2. Récupère les probabilités **ClubElo** (gratuit, sans clé) quand le service répond.
3. Calcule 3 modèles : **Poisson** pondéré dans le temps (force d'attaque/défense, avantage du terrain), **Elo FOOTBET** (toutes compétitions), **ClubElo**.
4. **Apprend de ses erreurs** : il garde ses pronostics d'avant-match, les compare aux résultats et donne plus de poids aux modèles qui se trompent le moins (log-loss), à partir de 30 matchs réglés.
5. Écrit `docs/data/stats.json` : probabilités 1/N/2, buts attendus, score le plus probable, +2,5 buts, les deux marquent, pronostic conseillé et niveau de confiance, fiabilité du modèle.

## Installation (10 minutes, gratuit)
1. Crée un compte sur https://github.com puis un **nouveau dépôt public** (ex. `footbet-stats`). Public est nécessaire pour que GitHub Pages et Actions restent gratuits.
2. Dans le dépôt : **Add file → Upload files**, glisse tout le contenu de ce dossier (y compris le dossier `.github`), puis **Commit changes**.
   - Si le dossier `.github` ne passe pas par glisser-déposer : **Add file → Create new file**, nomme-le `.github/workflows/agent.yml` et colle le contenu du fichier.
3. Crée ta clé gratuite sur https://www.football-data.org/client/register (elle arrive par e-mail).
4. Dans le dépôt : **Settings → Secrets and variables → Actions → New repository secret**. Nom : `FOOTBALL_DATA_TOKEN`, valeur : ta clé.
5. **Settings → Actions → General → Workflow permissions** : coche **Read and write permissions**, enregistre.
6. **Settings → Pages** : Source **Deploy from a branch**, branche **main**, dossier **/docs**, enregistre.
7. Onglet **Actions → Agent FOOTBET Stats → Run workflow** pour le premier passage (2 à 3 minutes la première fois).
8. Ta page : `https://<ton-pseudo>.github.io/footbet-stats/` — elle se recharge toute seule toutes les 10 minutes.

## Bon à savoir
- GitHub peut retarder les passages programmés de quelques minutes aux heures chargées.
- Offre gratuite football-data.org : 10 requêtes par minute, l'agent en fait 8 par passage (16 au tout premier pour la saison précédente).
- Les statistiques de tirs, compositions et cotes ne sont pas dans les offres gratuites : l'agent travaille sur les résultats.
- Tester en local : `python3 agent/agent.py` avec la variable `FOOTBALL_DATA_TOKEN`.

Aucun pari n'est sûr : ce sont des probabilités. Paris interdits aux mineurs. Joueurs Info Service : 09 74 75 13 13.

