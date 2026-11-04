#!/usr/bin/env python3
"""Agent statistique FOOTBET.

Environ toutes les heures (GitHub Actions) :
1. récupère les résultats et le calendrier (football-data.org, gratuit ; ESPN pour la Ligue Europa et la Ligue Conférence),
   les probabilités ClubElo, les cotes Betclic et, via ESPN, les compositions et absences ;
2. calcule trois modèles indépendants : Poisson pondéré dans le temps, Elo maison, ClubElo ;
3. les mélange avec des poids appris sur ses propres erreurs passées (log-loss des matchs déjà joués) ;
4. écrit docs/data/stats.json, lu par la page docs/index.html.

Python 3.10+, bibliothèque standard uniquement.
"""
import csv, difflib, io, json, math, os, re, sys, time, unicodedata, urllib.request
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "docs", "data")
CACHE = os.path.join(ROOT, "agent", "cache")
os.makedirs(DATA, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

TOKEN = os.environ.get("FOOTBALL_DATA_TOKEN", "").strip()
ODDS_KEY = os.environ.get("ODDS_API_KEY", "").strip()
try:
    ODDS_EVERY_H = float(os.environ.get("ODDS_EVERY_HOURS", "") or 24)
except ValueError:
    ODDS_EVERY_H = 24.0
OFFLINE = os.environ.get("AGENT_OFFLINE", "").strip()  # dossier de données de test
API = "https://api.football-data.org/v4"
COMPS = {
    "FL1": ("Ligue 1", 1570), "PL": ("Premier League", 1650), "PD": ("LaLiga", 1630),
    "SA": ("Serie A", 1615), "BL1": ("Bundesliga", 1615), "PPL": ("Liga Portugal", 1545),
    "DED": ("Eredivisie", 1535), "CL": ("Ligue des champions", None),
    "EL": ("Ligue Europa", None), "ECL": ("Ligue Conférence", None),
}
CUPS = ("CL", "EL", "ECL")
# ESPN (API publique sans clé) : C3 et C4 (absentes de l'offre gratuite football-data.org), compos et absences
ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer"
ESPN_LG = {"FL1": "fra.1", "PL": "eng.1", "PD": "esp.1", "SA": "ita.1", "BL1": "ger.1", "PPL": "por.1",
           "DED": "ned.1", "CL": "uefa.champions", "EL": "uefa.europa", "ECL": "uefa.europa.conf"}
ESPN_COMPS = ("EL", "ECL")
T0 = time.time()
DAYS_AHEAD = 8
NOW = datetime.now(timezone.utc)

# ---------------------------------------------------------------- utilitaires
def log(*a):
    print(datetime.now(timezone.utc).strftime("%H:%M:%S"), *a, flush=True)

def norm(s):
    s = unicodedata.normalize("NFD", str(s or "").lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"\b(fc|cf|afc|ac|sc|ssc|as|ss|us|rc|rcd|cd|ud|sd|sv|vfb|vfl|tsg|bv|fk|sk|club|de|calcio|1\.|1)\b", " ", s)
    return re.sub(r"[^a-z0-9]", "", s)

def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, obj, indent=None):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, separators=None if indent else (",", ":"))
    os.replace(tmp, path)

_last_call = [0.0]
def http(url, headers=None, timeout=30):
    wait = 6.3 - (time.time() - _last_call[0])  # 10 appels/minute maximum (offre gratuite)
    if "football-data" in url and wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": "footbet-agent/1.0", **(headers or {})})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                _last_call[0] = time.time()
                return r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            _last_call[0] = time.time()
            if e.code == 429 and attempt < 2:
                time.sleep(65); continue
            raise
        except Exception:
            if attempt < 2:
                time.sleep(5); continue
            raise

# ---------------------------------------------------------------- collecte
def season_of(d):
    return d.year if d.month >= 7 else d.year - 1

def fetch_matches(code, season=None):
    if OFFLINE:
        p = os.path.join(OFFLINE, f"{code}{'_' + str(season) if season else ''}.json")
        return load_json(p, {"matches": []}).get("matches", [])
    url = f"{API}/competitions/{code}/matches" + (f"?season={season}" if season else "")
    return json.loads(http(url, {"X-Auth-Token": TOKEN})).get("matches", [])

def side(score, k):
    ft = (score or {}).get("fullTime") or {}
    v = ft.get(k)
    if v is None:
        v = ft.get(k + "Team")
    return v

def simplify(m, code):
    h, a = m.get("homeTeam") or {}, m.get("awayTeam") or {}
    return {
        "id": m.get("id"), "c": code, "utc": m.get("utcDate"), "status": m.get("status"),
        "h": h.get("shortName") or h.get("name"), "a": a.get("shortName") or a.get("name"),
        "hid": h.get("id"), "aid": a.get("id"),
        "hg": side(m.get("score"), "home"), "ag": side(m.get("score"), "away"),
        "stage": m.get("stage"), "md": m.get("matchday"),
    }

def collect():
    cur = season_of(NOW)
    prev_cache = load_json(os.path.join(CACHE, "previous_season.json"), {})
    out, errors = {}, []
    for code in COMPS:
        if code in ESPN_COMPS:
            try:
                ms = espn_matches(code)
                out[code] = {"cur": ms, "prev": []}
                log(code, len(ms), "matchs (ESPN)")
            except Exception as e:
                errors.append(f"{code} (ESPN): {e}")
                log("ERREUR", code, e)
            continue
        try:
            ms = [simplify(m, code) for m in fetch_matches(code)]
            key = f"{code}_{cur - 1}"
            if key not in prev_cache:
                try:
                    prev_cache[key] = [x for x in (simplify(m, code) for m in fetch_matches(code, cur - 1)) if x["status"] == "FINISHED"]
                except Exception as e:  # saison précédente hors offre gratuite : on s'en passe
                    log("saison précédente indisponible", code, e)
                    prev_cache[key] = []
            out[code] = {"cur": ms, "prev": prev_cache.get(key, [])}
            log(code, len(ms), "matchs", len(prev_cache.get(key, [])), "matchs saison précédente")
        except Exception as e:
            errors.append(f"{code}: {e}")
            log("ERREUR", code, e)
    for k in [k for k in prev_cache if not k.endswith(str(cur - 1))]:
        del prev_cache[k]
    save_json(os.path.join(CACHE, "previous_season.json"), prev_cache)
    return out, errors

def espn_get(path, timeout=15):
    if OFFLINE:
        return load_json(os.path.join(OFFLINE, "espn_" + re.sub(r"[^A-Za-z0-9]+", "_", path).strip("_") + ".json"), {})
    req = urllib.request.Request(f"{ESPN}/{path}", headers={"User-Agent": "footbet-agent/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

def espn_event(e, code):
    c = (e.get("competitions") or [{}])[0]
    t = {x.get("homeAway"): x for x in c.get("competitors", [])}
    h, a = t.get("home") or {}, t.get("away") or {}
    st = (e.get("status") or {}).get("type") or {}
    done = bool(st.get("completed"))
    sc = lambda x: int(x["score"]) if done and str(x.get("score", "")).isdigit() else None
    name = lambda x: (x.get("team") or {}).get("displayName")
    return {"id": "E" + str(e.get("id")), "eid": str(e.get("id")), "c": code, "utc": (e.get("date") or "").replace("Z", ":00Z") if len(e.get("date") or "") == 17 else e.get("date"),
            "status": "FINISHED" if done else "SCHEDULED" if st.get("state") == "pre" else "IN_PLAY",
            "h": name(h), "a": name(a), "hid": (h.get("team") or {}).get("id"), "aid": (a.get("team") or {}).get("id"),
            "alias": [[(h.get("team") or {}).get(k) for k in ("shortDisplayName", "name")], [(a.get("team") or {}).get(k) for k in ("shortDisplayName", "name")]],
            "hg": sc(h), "ag": sc(a), "stage": None, "md": None}

def espn_matches(code):
    """Saison en cours mois par mois (ESPN refuse les plages de jours mais accepte dates=AAAAMM).
    Les mois terminés sont gardés en cache : en régime normal, 1 à 2 appels par passage."""
    path = os.path.join(CACHE, "espn_cups.json")
    cache = load_json(path, {})
    y, mo = season_of(NOW), 7
    end = NOW + timedelta(days=DAYS_AHEAD + 1)
    out = []
    while (y, mo) <= (end.year, end.month):
        key = f"{code}_{y}{mo:02d}"
        done = (y, mo) < (NOW.year, NOW.month) and key in cache and ago_h(cache[key]["at"]) < 24 * 40 and (y, mo) != ((NOW - timedelta(days=3)).year, (NOW - timedelta(days=3)).month)
        if not done:
            d = espn_get(f"{ESPN_LG[code]}/scoreboard?dates={y}{mo:02d}&limit=500")
            cache[key] = {"at": NOW.strftime("%Y-%m-%dT%H:%M:%SZ"), "ev": [m for m in (espn_event(e, code) for e in d.get("events", [])) if m["h"] and m["a"] and m["utc"]]}
        out += cache[key]["ev"]
        y, mo = (y + 1, 1) if mo == 12 else (y, mo + 1)
    for k in [k for k in cache if int(k.split("_")[1][:4]) < season_of(NOW)]:
        del cache[k]
    save_json(path, cache)
    seen = set()
    return [m for m in out if not (m["id"] in seen or seen.add(m["id"]))]

def unify_names(data):
    """Donne aux équipes ESPN (C3/C4) le nom football-data quand le club est connu, pour partager Elo et forme."""
    known = {}
    for code, d in data.items():
        if code not in ESPN_COMPS:
            for m in d["cur"] + d["prev"]:
                for t in (m["h"], m["a"]):
                    if t:
                        known[norm(t)] = t
    keys = list(known)
    cache = {}
    def find(names):
        for n in names:
            k = norm(n)
            if not k:
                continue
            if k in known:
                return known[k]
            c = difflib.get_close_matches(k, keys, n=1, cutoff=0.86)
            if c:
                return known[c[0]]
        return None
    for code in ESPN_COMPS:
        for m in (data.get(code) or {}).get("cur", []):
            al = m.pop("alias", [[], []])
            for side_, i in (("h", 0), ("a", 1)):
                key = m[side_]
                if key not in cache:
                    cache[key] = find([key] + [x for x in al[i] if x])
                if cache[key]:
                    m[side_] = cache[key]

def fetch_clubelo():
    """Probabilités ClubElo des prochains matchs : {(nom_normalisé_dom, nom_normalisé_ext): [p1, pN, p2]}."""
    try:
        txt = open(os.path.join(OFFLINE, "clubelo.csv"), encoding="utf-8").read() if OFFLINE else http("http://api.clubelo.com/Fixtures")
    except Exception as e:
        log("ClubElo indisponible", e)
        return {}
    res = {}
    for row in csv.DictReader(io.StringIO(txt)):
        p1 = pN = p2 = 0.0
        for k, v in row.items():
            if not k or not k.startswith("GD"):
                continue
            try:
                x = float(v)
            except Exception:
                continue
            m = re.match(r"GD\s*([<>=])\s*(-?\d+)", k)
            if not m:
                continue
            op, n = m.group(1), int(m.group(2))
            gd = n - 1 if op == "<" else n + 1 if op == ">" else n
            if gd > 0: p1 += x
            elif gd < 0: p2 += x
            else: pN += x
        s = p1 + pN + p2
        if s > 0.5 and row.get("Home") and row.get("Away"):
            res[(norm(row["Home"]), norm(row["Away"]))] = [p1 / s, pN / s, p2 / s]
    log("ClubElo", len(res), "matchs")
    return res

# ---------------------------------------------------------------- modèles
def pois(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)

def score_matrix(lh, la, n=9):
    return [[pois(i, lh) * pois(j, la) for j in range(n)] for i in range(n)]

def from_matrix(M):
    p1 = sum(M[i][j] for i in range(len(M)) for j in range(len(M)) if i > j)
    pN = sum(M[i][i] for i in range(len(M)))
    p2 = sum(M[i][j] for i in range(len(M)) for j in range(len(M)) if i < j)
    s = p1 + pN + p2
    o25 = sum(M[i][j] for i in range(len(M)) for j in range(len(M)) if i + j >= 3) / s
    btts = sum(M[i][j] for i in range(1, len(M)) for j in range(1, len(M))) / s
    best = max(((i, j) for i in range(len(M)) for j in range(len(M))), key=lambda x: M[x[0]][x[1]])
    return [p1 / s, pN / s, p2 / s], o25, btts, f"{best[0]}-{best[1]}"

def fit_poisson(matches, xi, k_prior=4.0):
    """Modèle de Maher pondéré dans le temps : lambda_dom = H * att_dom * def_ext."""
    teams = sorted({m["h"] for m in matches} | {m["a"] for m in matches})
    if len(matches) < 10:
        return None
    w = {}
    for m in matches:
        days = (NOW - datetime.fromisoformat(m["utc"].replace("Z", "+00:00"))).days
        w[m["id"]] = math.exp(-xi * max(0, days))
    sw = sum(w.values())
    gavg = sum(w[m["id"]] * (m["hg"] + m["ag"]) for m in matches) / (2 * sw)
    att = {t: 1.0 for t in teams}; dfn = {t: 1.0 for t in teams}; H = 1.2
    for _ in range(40):
        for t in teams:
            num = k_prior * gavg; den = k_prior * gavg
            for m in matches:
                ww = w[m["id"]]
                if m["h"] == t: num += ww * m["hg"]; den += ww * H * dfn[m["a"]] * gavg
                elif m["a"] == t: num += ww * m["ag"]; den += ww * dfn[m["h"]] * gavg
            att[t] = num / den
        for t in teams:
            num = k_prior * gavg; den = k_prior * gavg
            for m in matches:
                ww = w[m["id"]]
                if m["h"] == t: num += ww * m["ag"]; den += ww * att[m["a"]] * gavg
                elif m["a"] == t: num += ww * m["hg"]; den += ww * H * att[m["h"]] * gavg
            dfn[t] = num / den
        hn = sum(w[m["id"]] * m["hg"] for m in matches)
        hd = sum(w[m["id"]] * att[m["h"]] * dfn[m["a"]] * gavg for m in matches)
        H = max(0.9, min(1.6, hn / hd))
        ma = sum(att.values()) / len(att); md = sum(dfn.values()) / len(dfn)
        att = {t: v / ma for t, v in att.items()}; dfn = {t: v / md for t, v in dfn.items()}
    return {"att": att, "def": dfn, "H": H, "g": gavg}

def poisson_lambdas(model, h, a):
    if not model or h not in model["att"] or a not in model["att"]:
        return None
    g = model["g"]
    return model["H"] * model["att"][h] * model["def"][a] * g, model["att"][a] * model["def"][h] * g

def elo_ratings(all_matches, start, hfa=65, k=22):
    R = {}
    for m in sorted(all_matches, key=lambda x: x["utc"]):
        for t, c in ((m["h"], m["c"]), (m["a"], m["c"])):
            if t not in R:
                R[t] = start.get(c) or 1580
        e = 1 / (1 + 10 ** (-(R[m["h"]] + hfa - R[m["a"]]) / 400))
        s = 1.0 if m["hg"] > m["ag"] else 0.5 if m["hg"] == m["ag"] else 0.0
        gd = abs(m["hg"] - m["ag"])
        mult = 1 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
        d = k * mult * (s - e)
        R[m["h"]] += d; R[m["a"]] -= d
    return R

def elo_probs(rh, ra, hfa=65):
    e = 1 / (1 + 10 ** (-(rh + hfa - ra) / 400))
    pN = 0.29 * math.exp(-((e - 0.5) / 0.36) ** 2)
    p1 = max(0.02, e - pN / 2); p2 = max(0.02, 1 - e - pN / 2)
    s = p1 + pN + p2
    return [p1 / s, pN / s, p2 / s]

def elo_lambdas(rh, ra, gavg=1.35, hfa=65):
    d = (rh + hfa - ra) / 400
    return gavg * math.exp(0.55 * d), gavg * math.exp(-0.55 * d)

# ---------------------------------------------------------------- apprentissage
def logloss(p, r):
    return -math.log(max(1e-4, p[r]))

def learn_weights(history):
    settled = [h for h in history.values() if h.get("res") is not None]
    settled.sort(key=lambda h: h["utc"], reverse=True)
    settled = settled[:400]
    stats = {}
    for c in ("poisson", "elo", "clubelo", "final"):
        xs = [logloss(h["comp"][c], h["res"]) for h in settled if h.get("comp", {}).get(c)]
        stats[c] = {"n": len(xs), "logloss": round(sum(xs) / len(xs), 4) if xs else None}
    W = {"poisson": 0.4, "elo": 0.3, "clubelo": 0.3}
    usable = {c: stats[c]["logloss"] for c in W if stats[c]["n"] >= 30}
    if len(usable) >= 2:
        best = min(usable.values())
        W = {c: (math.exp(-(usable[c] - best) * 25) if c in usable else 0.25) for c in W}
        W = {c: max(0.1, v) for c, v in W.items()}
    s = sum(W.values())
    W = {c: round(v / s, 3) for c, v in W.items()}
    fin = [h for h in settled if h.get("final")]
    hit = sum(1 for h in fin if max(range(3), key=lambda i: h["final"][i]) == h["res"])
    brier = sum(sum((h["final"][i] - (1 if i == h["res"] else 0)) ** 2 for i in range(3)) for h in fin)
    picks = [h for h in fin if h.get("pick")]
    phit = sum(1 for h in picks if pick_ok(h["pick"]["sel"], h))
    bins = []
    for lo in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        sel = [h for h in fin if lo <= max(h["final"]) < lo + 0.1]
        if sel:
            ok = sum(1 for h in sel if max(range(3), key=lambda i: h["final"][i]) == h["res"])
            bins.append({"bin": f"{int(lo*100)}-{int(lo*100)+10} %", "n": len(sel), "pred": round(sum(max(h["final"]) for h in sel) / len(sel), 3), "real": round(ok / len(sel), 3)})
    recent = [{"utc": h["utc"], "c": h["c"], "match": h["match"], "pick": (h.get("pick") or {}).get("label"), "ok": pick_ok(h["pick"]["sel"], h) if h.get("pick") else None, "score": h.get("score")} for h in settled[:15]]
    return W, {
        "settled": len(fin), "hitRate": round(hit / len(fin), 3) if fin else None,
        "pickHitRate": round(phit / len(picks), 3) if picks else None, "picks": len(picks),
        "brier": round(brier / len(fin), 4) if fin else None, "components": stats, "calibration": bins, "recent": recent,
    }

WIN_SETS = {0: {"1", "1N", "12"}, 1: {"N", "1N", "N2"}, 2: {"2", "N2", "12"}}
GOAL_LINES = (0.5, 1.5, 2.5, 3.5, 4.5)
PICK_MIN = 0.70  # seuil de sûreté d'un pronostic

def pick_ok(sel, h):
    """Le pronostic sel est-il gagnant ? Jugé sur le score final ("2-1") ; à défaut, sur le résultat 1/N/2."""
    try:
        hg, ag = (int(x) for x in str(h.get("score") or "").split("-"))
    except ValueError:
        return sel in WIN_SETS[h["res"]] if h.get("res") is not None and sel in WIN_SETS[0] | WIN_SETS[1] | WIN_SETS[2] else None
    m = re.fullmatch(r"ou(\d+(?:\.\d+)?)-([01])", sel)
    if m:
        over = hg + ag > float(m.group(1))
        return over if m.group(2) == "0" else not over
    return {"1": hg > ag, "N": hg == ag, "2": hg < ag, "1N": hg >= ag, "12": hg != ag, "N2": hg <= ag,
            "bt0": hg > 0 and ag > 0, "bt1": not (hg > 0 and ag > 0)}.get(sel)

def goals_label(line, over):
    L = f"{line:g}".replace(".", ",")
    return f"{'Plus' if over else 'Moins'} de {L} but{'s' if line >= 2 else ''}"

def pick_options(p, h, a, overs, btts, odds):
    """Toutes les options de pari du match : (sel, libellé, probabilité, cote Betclic ou None)."""
    r = (odds or {}).get("r"); ou = (odds or {}).get("ou") or {}
    opts = [("1", f"Victoire de {h}", p[0], r[0] if r else None), ("N", "Match nul", p[1], r[1] if r else None),
            ("2", f"Victoire de {a}", p[2], r[2] if r else None)]
    for sel, lab, i, j in (("1N", f"{h} ou match nul", 0, 1), ("12", f"{h} ou {a}", 0, 2), ("N2", f"Match nul ou {a}", 1, 2)):
        opts.append((sel, lab, p[i] + p[j], 1 / (1 / r[i] + 1 / r[j]) if r else None))  # double chance estimée sur les cotes 1N2 Betclic
    for line, po in sorted(overs.items(), key=lambda x: float(x[0])):
        o = ou.get(line) or [None, None]
        opts.append((f"ou{line}-0", goals_label(float(line), True), po, o[0]))
        opts.append((f"ou{line}-1", goals_label(float(line), False), 1 - po, o[1]))
    opts.append(("bt0", "Les deux équipes marquent : oui", btts, None))
    opts.append(("bt1", "Les deux équipes marquent : non", 1 - btts, None))
    return opts

def pick_of(p, h, a, overs=None, btts=None, odds=None):
    """1) victoire si une équipe a 70 % ou plus ; 2) sinon l'option à 70 % ou plus qui a la cote la plus haute
    (cote Betclic si elle existe, sinon cote juste 1 / probabilité) ; 3) sinon double chance du favori."""
    for i in (0, 2):
        if p[i] >= PICK_MIN:
            return {"sel": "1" if i == 0 else "2", "label": f"Victoire de {h if i == 0 else a}", "prob": round(p[i], 3)}
    if overs is not None and btts is not None:
        best = None
        for sel, lab, prob, odd in pick_options(p, h, a, overs, btts, odds):
            if prob < PICK_MIN or prob >= 1:
                continue
            o = odd if isinstance(odd, (int, float)) and odd > 1 else 1 / prob
            if best is None or o > best[0]:
                best = (o, sel, lab, prob)
        if best:
            return {"sel": best[1], "label": best[2], "prob": round(best[3], 3)}
    if p[0] >= p[2]:
        return {"sel": "1N", "label": f"{h} ou match nul", "prob": round(p[0] + p[1], 3)}
    return {"sel": "N2", "label": f"Match nul ou {a}", "prob": round(p[1] + p[2], 3)}

# ---------------------------------------------------------------- cotes Betclic (The Odds API, offre gratuite)
SPORTS = {"FL1": "soccer_france_ligue_one", "PL": "soccer_epl", "PD": "soccer_spain_la_liga", "SA": "soccer_italy_serie_a",
          "BL1": "soccer_germany_bundesliga", "PPL": "soccer_portugal_primeira_liga", "DED": "soccer_netherlands_eredivisie",
          "CL": "soccer_uefa_champs_league", "EL": "soccer_uefa_europa_league", "ECL": "soccer_uefa_europa_conference_league"}
# Budget The Odds API (offre gratuite, 500 crédits/mois) : un appel /odds coûte marchés × régions = 2 (h2h + totals) × 1
# (betclic_fr seul = 1 région) = 2 crédits. Un relevé par jour (vers 9 h, Paris) des 10 compétitions coûterait
# 10 × 2 × 31 = 620 crédits : trop. On interroge donc d'abord /events (gratuit, 0 crédit) et on ne paie que pour les
# compétitions qui ont un match dans les 36 h : en pratique 4 à 6 par jour, soit environ 250 à 370 crédits par mois.
ODDS_COST = 2

def parse_event(e):
    bk = next((b for b in e.get("bookmakers", []) if b.get("key") == "betclic_fr"), None)
    if not bk:
        return None
    r, ou = None, {}
    for mk in bk.get("markets", []):
        oc = mk.get("outcomes", [])
        if mk.get("key") == "h2h":
            d = {o.get("name"): o.get("price") for o in oc}
            if all(isinstance(d.get(k), (int, float)) for k in (e["home_team"], "Draw", e["away_team"])):
                r = [d[e["home_team"]], d["Draw"], d[e["away_team"]]]
        elif mk.get("key") == "totals":
            for o in oc:
                pt = o.get("point")
                if isinstance(pt, (int, float)) and isinstance(o.get("price"), (int, float)):
                    ou.setdefault(f"{float(pt):g}", [None, None])[0 if o.get("name") == "Over" else 1] = o["price"]
    ou = {k: v for k, v in ou.items() if all(v)}
    if not r and not ou:
        return None
    return {"h": e["home_team"], "a": e["away_team"], "utc": e["commence_time"], "r": r, "ou": ou}

def fetch_odds():
    path = os.path.join(CACHE, "odds.json")
    cache = load_json(path, {"at": None, "remaining": None, "events": {}})
    now_iso = NOW.strftime("%Y-%m-%dT%H:%M:%SZ")
    for code in list(cache["events"]):  # on garde seulement les matchs à venir
        cache["events"][code] = [e for e in cache["events"][code] if e["utc"] > now_iso]
    if OFFLINE:
        for code in SPORTS:
            evs = load_json(os.path.join(OFFLINE, f"odds_{code}.json"), [])
            cache["events"][code] = [x for x in (parse_event(e) for e in evs) if x]
        cache["at"] = now_iso
        return cache
    age = (NOW - datetime.fromisoformat(cache["at"].replace("Z", "+00:00"))).total_seconds() / 3600 if cache.get("at") else 1e9
    from zoneinfo import ZoneInfo
    paris = NOW.astimezone(ZoneInfo("Europe/Paris"))
    last_day = datetime.fromisoformat(cache["at"].replace("Z", "+00:00")).astimezone(ZoneInfo("Europe/Paris")).date() if cache.get("at") else None
    morning = paris.hour >= 9 and last_day != paris.date() and age >= 6  # un relevé chaque matin vers 9 h
    if not ODDS_KEY or (age < ODDS_EVERY_H and not morning):
        return cache
    to = (NOW + timedelta(hours=36)).strftime("%Y-%m-%dT%H:%M:%SZ")
    got = 0
    frm = NOW.strftime("%Y-%m-%dT%H:%M:%SZ")
    for code, sport in SPORTS.items():
        if cache.get("remaining") is not None and cache["remaining"] < 4:
            log("crédits The Odds API presque épuisés : arrêt jusqu'au mois prochain"); break
        try:  # /events ne consomme aucun crédit : on vérifie qu'il y a des matchs avant de payer le relevé
            req = urllib.request.Request(f"https://api.the-odds-api.com/v4/sports/{sport}/events?apiKey={ODDS_KEY}"
                                         f"&commenceTimeFrom={frm}&commenceTimeTo={to}", headers={"User-Agent": "footbet-agent/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                if not json.loads(resp.read().decode("utf-8")):
                    cache["events"][code] = []
                    log("cotes", code, ": aucun match dans les 36 h, relevé sauté (0 crédit)")
                    continue
        except Exception as e:
            log("liste des matchs The Odds API indisponible", code, e)
        url = (f"https://api.the-odds-api.com/v4/sports/{sport}/odds?apiKey={ODDS_KEY}&bookmakers=betclic_fr"
               f"&markets=h2h,totals&oddsFormat=decimal&dateFormat=iso&commenceTimeTo={to}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "footbet-agent/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                evs = json.loads(resp.read().decode("utf-8"))
                rem = resp.headers.get("x-requests-remaining")
                if rem is not None:
                    cache["remaining"] = float(rem)
            cache["events"][code] = [x for x in (parse_event(e) for e in evs) if x]
            got += 1
        except Exception as e:
            log("cotes indisponibles", code, e)
    if got:
        cache["at"] = now_iso
    save_json(path, cache)
    log("cotes Betclic :", sum(len(v) for v in cache["events"].values()), "matchs ; crédits restants", cache.get("remaining"))
    return cache

def match_odds(cache, code, m):
    ko = datetime.fromisoformat(m["utc"].replace("Z", "+00:00"))
    best, bs = None, 0.0
    for e in cache["events"].get(code, []):
        et = datetime.fromisoformat(e["utc"].replace("Z", "+00:00"))
        if abs((et - ko).total_seconds()) > 3 * 3600:
            continue
        sc = sum(difflib.SequenceMatcher(None, norm(x), norm(y)).ratio() + (0.35 if norm(x) and (norm(x) in norm(y) or norm(y) in norm(x)) else 0)
                 for x, y in ((m["h"], e["h"]), (m["a"], e["a"])))
        if sc > bs:
            best, bs = e, sc
    return best if bs >= 1.1 else None

def over_probs(lh, la, extra=()):
    """Probabilité de « plus de L buts » pour chaque ligne (0,5 à 4,5 + lignes Betclic), par la matrice de Poisson."""
    M = score_matrix(lh, la, 11)
    tot = sum(map(sum, M))
    lines = sorted(set(GOAL_LINES) | {float(x) for x in extra})
    return {f"{l:g}": sum(M[i][j] for i in range(11) for j in range(11) if i + j > l) / tot for l in lines}

# ---------------------------------------------------------------- combinés (cotes Betclic réelles)
C_MIN, C_MAX, C_LEGS = 10.0, 16.0, 7
C_PLAN = [("Victoires", 2, "win"), ("Victoire/nul", 1, "res"), ("Buts", 1, "goal"), ("Mixte", 1, "mix")]

def combo_legs(rows):
    allm = []
    for r in rows:
        o = r.get("odds")
        if not o:
            continue
        p, pk = r["p"], r["pick"]["sel"]
        bad = {"r1", "r2", "dc2"} if pk in ("1", "1N") else {"r0", "r1", "dc0"} if pk in ("2", "N2") else set()
        base = {"id": r["id"], "c": r["c"], "match": f"{r['h']} – {r['a']}", "utc": r["utc"]}
        legs = []
        def add(kind, sel, label, odd, prob, est=False):
            if sel not in bad and isinstance(odd, (int, float)) and odd > 1.01 and 0 < prob < 1:
                legs.append(dict(base, kind=kind, sel=sel, label=label, odd=round(odd, 2), prob=round(prob, 3), est=est))
        if o.get("r"):
            for i, lab in enumerate([f"Victoire de {r['h']}", "Match nul", f"Victoire de {r['a']}"]):
                add("res", f"r{i}", lab, o["r"][i], p[i])
            for i, (lab, a, b) in enumerate([(f"{r['h']} ou match nul", 0, 1), (f"{r['h']} ou {r['a']}", 0, 2), (f"Match nul ou {r['a']}", 1, 2)]):
                add("res", f"dc{i}", lab, 1 / (1 / o["r"][a] + 1 / o["r"][b]), p[a] + p[b], True)
        for line, (ov, un) in (o.get("ou") or {}).items():
            po = r["overs"].get(line)
            if po is None:
                continue
            add("goal", f"ou{line}-0", goals_label(float(line), True), ov, po)
            add("goal", f"ou{line}-1", goals_label(float(line), False), un, 1 - po)
        if legs:
            allm.append(legs)
    return allm

def build_combos(rows, rng):
    import random as _r
    matches = combo_legs(rows)
    prod = lambda t, k: math.prod(l[k] for l in t)
    def allowed(l, mode):
        return (mode == "win" and l["sel"] in ("r0", "r2")) or (mode == "res" and l["kind"] == "res") or (mode == "goal" and l["kind"] == "goal") or mode == "mix"
    def valid(t, mode):
        return mode != "mix" or len({l["kind"] for l in t}) >= 2
    def build(mode, pmin):
        opts = [[l for l in ms if allowed(l, mode) and l["prob"] >= pmin] for ms in matches]
        opts = [o for o in opts if o]; rng.shuffle(opts); t = []
        for o in opts:
            o2 = sorted(o, key=lambda l: -l["prob"])
            t.append(o2[0] if rng.random() < 0.5 else rng.choice(o2[:3]))
            if prod(t, "odd") >= C_MIN: break
        if prod(t, "odd") < C_MIN: return None
        for i in range(len(t)):
            t2 = t[:i] + t[i + 1:]
            if t2 and prod(t2, "odd") >= C_MIN and valid(t2, mode):
                t = t2; break
        return t if prod(t, "odd") <= C_MAX and len(t) <= C_LEGS and valid(t, mode) else None
    sig = lambda t: frozenset((l["id"], l["sel"]) for l in t)
    chosen, alts = [], {}
    def ok(t, share, cap):
        s, use = sig(t), {}
        for _, g in chosen:
            if s == sig(g) or len(s & sig(g)) > share: return False
            for x in sig(g): use[x] = use.get(x, 0) + 1
        return all(use.get(x, 0) < cap for x in s)
    for name, n, mode in C_PLAN:
        pool = {}
        for lv, pmin in enumerate((0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3)):
            for _ in range(3000):
                t = build(mode, pmin)
                if t and sig(t) not in pool: pool[sig(t)] = (lv, t)
            if len(pool) >= 10 * n: break
        ranked = [t for lv, t in sorted(pool.values(), key=lambda x: (x[0], -prod(x[1], "prob")))]
        got = 0
        for share, cap in ((0, 1), (1, 2), (1, 3), (2, 3), (3, 4), (99, 99)):
            for t in ranked:
                if got >= n: break
                if ok(t, share, cap): chosen.append((name, t)); got += 1
            if got >= n: break
        used = {sig(t) for nm, t in chosen if nm == name}
        alts[name] = [t for t in ranked if sig(t) not in used][:8]
    def doc(name, t, k):
        t = sorted(t, key=lambda l: (l["utc"], l["match"]))
        return {"id": f"{name}-{k}-" + "-".join(sorted(f"{l['id']}{l['sel']}" for l in t))[:120], "type": name, "odd": round(prod(t, "odd"), 2), "prob": round(prod(t, "prob"), 4),
                "legs": [{x: l[x] for x in ("id", "c", "match", "utc", "sel", "label", "odd", "prob", "est")} for l in t]}
    return {"tickets": [doc(n, t, i) for i, (n, t) in enumerate(chosen)],
            "alts": {n: [doc(n, t, 100 + i) for i, t in enumerate(v)] for n, v in alts.items()},
            "matches": len(matches)}

# ---------------------------------------------------------------- table et forme
def table_and_form(matches):
    T, F = {}, {}
    for m in sorted(matches, key=lambda x: x["utc"]):
        for t in (m["h"], m["a"]):
            T.setdefault(t, {"pts": 0, "gd": 0, "gf": 0, "p": 0}); F.setdefault(t, [])
        hg, ag = m["hg"], m["ag"]
        for t, gf, ga in ((m["h"], hg, ag), (m["a"], ag, hg)):
            T[t]["p"] += 1; T[t]["gf"] += gf; T[t]["gd"] += gf - ga
            T[t]["pts"] += 3 if gf > ga else 1 if gf == ga else 0
            F[t].append("V" if gf > ga else "N" if gf == ga else "D")
    order = sorted(T, key=lambda t: (-T[t]["pts"], -T[t]["gd"], -T[t]["gf"]))
    pos = {t: i + 1 for i, t in enumerate(order)}
    return pos, {t: " ".join(reversed(v[-5:])) for t, v in F.items()}, T

# ---------------------------------------------------------------- compositions et absences (ESPN, sans clé)
# Absences : ESPN ne publie pas les blessés pour le football (réponses vides, vérifié en novembre 2026) ; si le résumé
# de match en contient un jour, ils sont repris. Aucun appel dédié n'est donc fait pour les absences.
NEWS_LINEUP_WINDOW_H = 2   # compos : seulement dans les 2 h avant le coup d'envoi
NEWS_MAX_CALLS = 60        # plafond d'appels ESPN par passage
NEWS_MAX_SECONDS = 420     # au-delà de 7 min de passage, on n'interroge plus ESPN

def ago_h(iso):
    return (NOW - datetime.fromisoformat(iso.replace("Z", "+00:00"))).total_seconds() / 3600 if iso else 1e9

def espn_day(code, day, cache):
    if (code, day) not in cache:
        cache[(code, day)] = [espn_event(e, code) for e in espn_get(f"{ESPN_LG[code]}/scoreboard?dates={day}").get("events", [])]
    return cache[(code, day)]

def same_team(x, names):
    k = norm(x)
    return any(n and (norm(n) == k or difflib.SequenceMatcher(None, norm(n), k).ratio() >= 0.8 or (len(k) >= 5 and (k in norm(n) or norm(n) in k))) for n in names)

def find_eid(r, cache):
    if r.get("eid"):
        return r["eid"]
    ko = datetime.fromisoformat(r["utc"].replace("Z", "+00:00"))
    for e in espn_day(r["c"], ko.strftime("%Y%m%d"), cache):
        et = datetime.fromisoformat(e["utc"].replace("Z", "+00:00"))
        al = e.get("alias") or [[], []]
        if abs((et - ko).total_seconds()) <= 3 * 3600 and same_team(r["h"], [e["h"]] + al[0]) and same_team(r["a"], [e["a"]] + al[1]):
            return e["eid"]
    return None

INJ_SURE = {"out", "suspension", "suspended", "injured reserve", "ir"}
def parse_summary(s, r):
    """Compos (rosters) et absences (injuries) d'un résumé de match ESPN, rangées [domicile, extérieur]."""
    comps = ((s.get("header") or {}).get("competitions") or [{}])[0].get("competitors") or []
    side_of = {str((c.get("team") or {}).get("id")): (0 if c.get("homeAway") == "home" else 1) for c in comps}
    lu = [None, None]
    for ro in s.get("rosters") or []:
        i = 0 if ro.get("homeAway") == "home" else 1
        xi = [x for x in ro.get("roster") or [] if x.get("starter")]
        if len(xi) >= 11:
            lu[i] = {"st": "officielle", "f": ro.get("formation"),
                     "xi": [{"n": (x.get("athlete") or {}).get("displayName"), "p": (x.get("position") or {}).get("abbreviation"), "j": x.get("jersey")} for x in xi]}
    ab, has_inj = [[], []], "injuries" in s
    for blk in s.get("injuries") or []:
        tm = blk.get("team") or {}
        i = side_of.get(str(tm.get("id")))
        if i is None:
            i = 0 if same_team(tm.get("displayName") or "", [r["h"]]) else 1
        for inj in blk.get("injuries") or []:
            st = str(inj.get("status") or (inj.get("type") or {}).get("description") or "").strip()
            det = inj.get("details") or {}
            why = det.get("type") or det.get("detail") or (inj.get("type") or {}).get("description") or st or "non précisée"
            ab[i].append({"n": (inj.get("athlete") or {}).get("displayName"), "r": str(why),
                          "s": "confirmé" if st.lower() in INJ_SURE or "susp" in str(why).lower() else "incertain"})
    return lu, ab, has_inj

def team_news(rows, errors):
    path = os.path.join(CACHE, "lineups.json")
    cache = load_json(path, {})
    live = {str(r["id"]) for r in rows}
    cache = {k: v for k, v in cache.items() if k in live}
    now_iso = NOW.strftime("%Y-%m-%dT%H:%M:%SZ")
    sb, calls, fails = {}, 0, 0
    for r in sorted(rows, key=lambda x: x["utc"]):
        k = str(r["id"])
        c = cache.setdefault(k, {})
        hours = (datetime.fromisoformat(r["utc"].replace("Z", "+00:00")) - NOW).total_seconds() / 3600
        need_lu = hours <= NEWS_LINEUP_WINDOW_H and not (c.get("lu") and all(c["lu"]))
        if need_lu and calls < NEWS_MAX_CALLS and time.time() - T0 < NEWS_MAX_SECONDS and fails < 4:
            try:
                eid = c.get("eid") or find_eid(r, sb)
                calls += 1
                if eid:
                    c["eid"] = eid
                    lu, ab, has_inj = parse_summary(espn_get(f"{ESPN_LG[r['c']]}/summary?event={eid}"), r)
                    calls += 1
                    if any(lu):
                        c["lu"], c["luAt"] = lu, now_iso
                    if has_inj:
                        c["abs"], c["absAt"], c["absOk"] = ab, now_iso, True
            except Exception as e:
                fails += 1
                log("ESPN indisponible pour", r["h"], "-", r["a"], e)
        lu = c.get("lu") or [None, None]
        r["news"] = {"src": "ESPN", "abs": c.get("abs"), "absAt": c.get("absAt"), "absOk": c.get("absOk"),
                     "lu": [x or {"st": "inconnue"} for x in lu], "luAt": c.get("luAt")}
    if fails >= 4:
        errors.append("ESPN (compos/absences) ne répond pas à ce passage")
    save_json(path, cache)
    log("compos/absences ESPN :", calls, "appels ;", sum(1 for r in rows if any(x.get("st") == "officielle" for x in r["news"]["lu"])), "compos officielles")

# ---------------------------------------------------------------- principal
def main():
    if not TOKEN and not OFFLINE:
        log("FOOTBALL_DATA_TOKEN manquant : ajoute-le dans Settings > Secrets and variables > Actions.")
    data, errors = collect()
    unify_names(data)
    clubelo = fetch_clubelo()
    odds = fetch_odds()
    ce_keys = list(clubelo.keys())
    hist_path = os.path.join(DATA, "history.json")
    history = load_json(hist_path, {})
    W, quality = learn_weights(history)
    log("poids appris", W)

    finished_all = []
    for code, d in data.items():
        for m in d["prev"] + d["cur"]:
            if m["status"] == "FINISHED" and m["hg"] is not None:
                finished_all.append(m)
    start = {c: v[1] for c, v in COMPS.items()}
    R = elo_ratings(finished_all, start)
    nplayed = {}
    for m in finished_all:
        for t in (m["h"], m["a"]):
            nplayed[t] = nplayed.get(t, 0) + 1

    out = []
    for code, d in data.items():
        fin_cur = [m for m in d["cur"] if m["status"] == "FINISHED" and m["hg"] is not None]
        fin_prev = [m for m in d["prev"] if m["hg"] is not None]
        model = fit_poisson(fin_cur + fin_prev, xi=0.006) if code not in CUPS else None
        pos, form, table = table_and_form(fin_cur) if code not in CUPS else ({}, {}, {})
        if code in CUPS:  # forme toutes compétitions pour les coupes d'Europe
            _, form, _ = table_and_form([m for m in finished_all])
        for m in d["cur"]:
            if m["status"] == "FINISHED" and m["hg"] is not None:
                h = history.get(str(m["id"]))
                if h and h.get("res") is None:
                    h["res"] = 0 if m["hg"] > m["ag"] else 1 if m["hg"] == m["ag"] else 2
                    h["score"] = f"{m['hg']}-{m['ag']}"
                continue
            if m["status"] not in ("SCHEDULED", "TIMED"):
                continue
            ko = datetime.fromisoformat(m["utc"].replace("Z", "+00:00"))
            if not (NOW < ko < NOW + timedelta(days=DAYS_AHEAD)) or not m["h"] or not m["a"]:
                continue
            comp = {}
            lp = poisson_lambdas(model, m["h"], m["a"])
            rh, ra = R.get(m["h"], start.get(code) or 1580), R.get(m["a"], start.get(code) or 1580)
            le = elo_lambdas(rh, ra, model["g"] if model else 1.4)
            if lp:
                comp["poisson"] = from_matrix(score_matrix(*lp))[0]
            comp["elo"] = elo_probs(rh, ra)
            kh, ka = norm(m["h"]), norm(m["a"])
            ce = clubelo.get((kh, ka))
            if not ce and ce_keys:
                cand = difflib.get_close_matches(kh + "|" + ka, [x + "|" + y for x, y in ce_keys], n=1, cutoff=0.82)
                if cand:
                    ce = clubelo.get(tuple(cand[0].split("|")))
            if ce:
                comp["clubelo"] = ce
            ev = match_odds(odds, code, m)
            evo = {"r": ev["r"], "ou": ev["ou"]} if ev else None
            src = None
            if code in ESPN_COMPS and min(nplayed.get(m["h"], 0), nplayed.get(m["a"], 0)) < 8:
                # équipe sans historique dans le modèle (petit championnat) : l'Elo maison ne vaut rien ici
                del comp["elo"]
                if ce:
                    src = "ClubElo (équipe sans historique dans le modèle)"
                elif evo and evo.get("r"):
                    inv = [1 / x for x in evo["r"]]
                    comp["betclic"] = [x / sum(inv) for x in inv]
                    src = "Cotes Betclic sans marge (équipe sans historique dans le modèle)"
                else:
                    comp["elo"] = elo_probs(rh, ra)
                    src = "Elo par défaut : équipe sans historique, ni ClubElo ni cote Betclic (fiabilité faible)"
                if "elo" not in comp:  # buts attendus tirés des probabilités retenues
                    q = comp.get("clubelo") or comp["betclic"]
                    e = min(0.97, max(0.03, q[0] + q[1] / 2))
                    dd = math.log10(e / (1 - e))
                    le = (1.35 * math.exp(0.55 * dd), 1.35 * math.exp(-0.55 * dd))
            ws = {c: W.get(c, 1.0) for c in comp}
            s = sum(ws.values())
            p = [sum(ws[c] * comp[c][i] for c in comp) / s for i in range(3)]
            lam = (0.65 * lp[0] + 0.35 * le[0], 0.65 * lp[1] + 0.35 * le[1]) if lp else le
            _, o25, btts, sc = from_matrix(score_matrix(*lam))
            overs = over_probs(*lam, extra=(evo or {}).get("ou") or {})
            pk = pick_of(p, m["h"], m["a"], overs, btts, evo)
            agree = len({max(range(3), key=lambda i: v[i]) for v in comp.values()}) == 1
            conf = "forte" if pk["prob"] >= 0.68 and agree else "moyenne" if pk["prob"] >= 0.58 else "faible"
            if src and src.startswith("Elo par défaut"):
                conf = "faible"  # aucune donnée fiable sur au moins une des équipes
            row = {
                "id": m["id"], "c": code, "utc": m["utc"], "h": m["h"], "a": m["a"], "md": m["md"],
                "pos": [pos.get(m["h"]), pos.get(m["a"])], "form": [form.get(m["h"], ""), form.get(m["a"], "")],
                "p": [round(x, 3) for x in p], "comp": {c: [round(x, 3) for x in v] for c, v in comp.items()},
                "xg": [round(lam[0], 2), round(lam[1], 2)], "score": sc, "over25": round(o25, 3), "btts": round(btts, 3),
                "elo": [round(rh), round(ra)], "pick": pk, "conf": conf, "agree": agree,
                "overs": {k: round(v, 3) for k, v in overs.items()},
            }
            if evo:
                row["odds"] = evo
            if src:
                row["src"] = src
            if m.get("eid"):
                row["eid"] = m["eid"]
            out.append(row)
            if ko - NOW < timedelta(days=3):
                history[str(m["id"])] = {"utc": m["utc"], "c": code, "match": f"{m['h']} – {m['a']}", "comp": row["comp"], "final": row["p"], "pick": pk, "res": None}

    cutoff = (NOW - timedelta(days=120)).isoformat()
    history = {k: v for k, v in history.items() if v["utc"] >= cutoff}
    save_json(hist_path, history)
    out.sort(key=lambda r: r["utc"])
    try:
        team_news(out, errors)
    except Exception as e:  # jamais bloquant
        errors.append(f"compos/absences ESPN: {e}")
        log("ERREUR compos/absences", e)
    top = sorted([r for r in out if r["utc"] < (NOW + timedelta(hours=36)).isoformat()], key=lambda r: -r["pick"]["prob"])[:8]
    import random
    soon = [r for r in out if r.get("odds") and r["utc"] < (NOW + timedelta(hours=36)).isoformat()]
    combos = build_combos(soon, random.Random(int(NOW.strftime("%Y%m%d"))))
    stats = {
        "updated": NOW.strftime("%Y-%m-%dT%H:%M:%SZ"), "everyMin": 60, "errors": errors,
        "competitions": {c: v[0] for c, v in COMPS.items()}, "weights": W, "quality": quality,
        "sources": ["football-data.org (résultats, calendrier)", "ESPN (Ligue Europa, Ligue Conférence, compositions, absences)", "ClubElo (probabilités Elo)" if clubelo else "ClubElo indisponible à ce passage", "Cotes Betclic via The Odds API" if odds.get("at") else "Cotes Betclic pas encore relevées"],
        "matches": out, "top": [r["id"] for r in top], "combos": combos,
        "odds": {"at": odds.get("at"), "remaining": odds.get("remaining"), "matched": sum(1 for r in out if r.get("odds")), "source": "Betclic (via The Odds API)"},
    }
    save_json(os.path.join(DATA, "stats.json"), stats)
    log("écrit", len(out), "matchs ;", len(history), "dans l'historique")
    if errors and not out:
        sys.exit(1)

if __name__ == "__main__":
    main()
