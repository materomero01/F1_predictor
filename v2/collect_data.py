"""Descarga resultados de F1 (carrera, clasificación y sprint) desde Jolpica-F1
(el reemplazo de Ergast) y arma data/results.csv + data/schedule.csv.

- Mucho más liviano que FastF1 `session.load()`: solo resultados, ~70 requests.
- Incluye fines de semana sprint (la v1 los salteaba).
- Los años cerrados se cachean en data/raw; el año en curso se baja siempre de nuevo.

Uso:
    python collect_data.py                # START_SEASON..año actual
    python collect_data.py --refresh      # ignora la caché
    python collect_data.py --from 2021
"""
import argparse
import json
import time

import pandas as pd
import requests

import config as C

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "f1-race-predictor-v2 (personal project)"


def _get(url: str, params: dict) -> dict:
    for attempt in range(8):
        r = SESSION.get(url, params=params, timeout=30)
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", 2 * (attempt + 1)))
            print(f"    rate limit, espero {wait:.0f}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        time.sleep(0.35)  # ~3 req/s, bajo el límite de Jolpica
        return r.json()["MRData"]
    raise RuntimeError(f"Demasiados reintentos: {url}")


def _paged(season: int, kind: str, refresh: bool) -> list[dict]:
    """Devuelve la lista de carreras (con sus resultados) de un endpoint paginado.
    Una carrera puede quedar partida entre páginas: se junta por ronda."""
    cache = C.RAW_DIR / f"{season}_{kind}.json"
    if cache.exists() and not refresh and season < C.END_SEASON:
        return json.loads(cache.read_text(encoding="utf-8"))

    races: dict[int, dict] = {}
    offset, limit = 0, 100
    key = {"results": "Results", "qualifying": "QualifyingResults", "sprint": "SprintResults"}[kind]
    while True:
        md = _get(f"{C.JOLPICA_URL}/{season}/{kind}.json", {"limit": limit, "offset": offset})
        total = int(md["total"])
        for race in md["RaceTable"]["Races"]:
            rnd = int(race["round"])
            if rnd not in races:
                races[rnd] = {k: v for k, v in race.items() if k != key}
                races[rnd][key] = []
            seen = {x["Driver"]["driverId"] for x in races[rnd][key]}
            races[rnd][key] += [x for x in race.get(key, []) if x["Driver"]["driverId"] not in seen]
        offset += limit
        if offset >= total:
            break
    out = [races[k] for k in sorted(races)]
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def _num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def collect(start: int, end: int, refresh: bool) -> pd.DataFrame:
    rows = []
    sched_rows = []
    for season in range(start, end + 1):
        print(f"=== {season} ===")
        sched = _get(f"{C.JOLPICA_URL}/{season}.json", {"limit": 100})["RaceTable"]["Races"]
        for s in sched:
            sched_rows.append({
                "season": season, "round": int(s["round"]), "race_name": s["raceName"],
                "circuit_id": s["Circuit"]["circuitId"], "circuit_name": s["Circuit"]["circuitName"],
                "locality": s["Circuit"]["Location"]["locality"], "country": s["Circuit"]["Location"]["country"],
                "date": s["date"], "is_sprint_weekend": int("Sprint" in s),
            })
        race = _paged(season, "results", refresh)
        quali = _paged(season, "qualifying", refresh)
        sprint = _paged(season, "sprint", refresh)
        qmap = {(int(r["round"]), x["Driver"]["driverId"]): _num(x["position"])
                for r in quali for x in r["QualifyingResults"]}
        smap = {(int(r["round"]), x["Driver"]["driverId"]): _num(x["position"])
                for r in sprint for x in r["SprintResults"]}
        sprint_rounds = {int(r["round"]) for r in sprint}
        for r in race:
            rnd = int(r["round"])
            for x in r["Results"]:
                d, c = x["Driver"], x["Constructor"]
                rows.append({
                    "season": season, "round": rnd, "race_name": r["raceName"],
                    "circuit_id": r["Circuit"]["circuitId"], "date": r["date"],
                    "driver_id": d["driverId"], "code": d.get("code", d["familyName"][:3].upper()),
                    "driver_name": f'{d["givenName"]} {d["familyName"]}',
                    "constructor_id": c["constructorId"], "constructor_name": c["name"],
                    "grid": _num(x.get("grid"), 0.0),
                    "quali_pos": qmap.get((rnd, d["driverId"])),
                    "sprint_pos": smap.get((rnd, d["driverId"])),
                    "is_sprint_weekend": int(rnd in sprint_rounds),
                    "position": _num(x.get("position")),
                    "position_text": x.get("positionText"),
                    "points": _num(x.get("points"), 0.0),
                    "laps": _num(x.get("laps"), 0.0),
                    "status": x.get("status", ""),
                })
        print(f"   carreras con resultado: {len(race)} | sprints: {len(sprint)}")

    df = pd.DataFrame(rows).sort_values(["season", "round", "position"]).reset_index(drop=True)
    df.to_csv(C.RESULTS_CSV, index=False)
    pd.DataFrame(sched_rows).to_csv(C.SCHEDULE_CSV, index=False)
    last = df.iloc[-1]
    print(f"\n✓ {len(df)} filas -> {C.RESULTS_CSV}")
    print(f"✓ Última carrera cargada: {last['season']} R{int(last['round'])} {last['race_name']} ({last['date']})")

    unknown = sorted(set(df["constructor_id"]) - set(C.TEAM_LINEAGE))
    if unknown:
        print(f"⚠ constructorId sin linaje en config.TEAM_LINEAGE (se tratan como equipos nuevos): {unknown}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=C.START_SEASON)
    ap.add_argument("--to", dest="end", type=int, default=C.END_SEASON)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    collect(a.start, a.end, a.refresh)
