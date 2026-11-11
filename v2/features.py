"""Feature engineering v2.

Diferencias clave respecto a la v1:
  * UN solo camino de código para entrenar y predecir (`FeatureBuilder.race_features`),
    así no hay desfase train/serve.
  * Estrictamente causal: para la carrera t solo se usan carreras con t' < t
    (incluye la temporada en curso).
  * Piloto y auto se separan: el piloto se mide contra su compañero de equipo
    (portable entre equipos) y el auto se mide por equipo dentro de la temporada.
  * Cambios de reglamento (config.REGIME_CHANGE_SEASONS): el prior del auto se
    encoge fuerte hacia la media del campo y pierde peso rápido con cada carrera.
  * Sin hand-tuning de rookies: la contracción hacia 0 (shrinkage) lo resuelve.
  * Features de circuito y de sprint/clasificación (en la v1 se calculaban pero
    no entraban al modelo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------

def _ew(values, half_life: float):
    """Media exponencial (el último valor es el más reciente). Devuelve (media, n_válidos)."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return np.nan, 0
    m = ~np.isnan(v)
    if not m.any():
        return np.nan, 0
    w = 0.5 ** (np.arange(v.size)[::-1] / half_life)
    return float((v[m] * w[m]).sum() / w[m].sum()), int(m.sum())


def _shrunk(values, half_life: float, k: float) -> float:
    """Media exponencial contraída hacia 0 con n/(n+k)."""
    mean, n = _ew(values, half_life)
    return 0.0 if n == 0 else mean * n / (n + k)


def canonical_team(constructor_id: str) -> str:
    return C.TEAM_LINEAGE.get(constructor_id, constructor_id)


# --------------------------------------------------------------------------
# Preparación de resultados históricos
# --------------------------------------------------------------------------

def prepare_results(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy().sort_values(["season", "round", "position"]).reset_index(drop=True)
    df["t"] = df.groupby(["season", "round"], sort=True).ngroup()
    df["team"] = df["constructor_id"].map(canonical_team)
    df["n_entries"] = df.groupby("t")["driver_id"].transform("count")

    n = df["n_entries"]
    df["pos"] = df["position"].fillna(n)
    df["grid_adj"] = df["grid"].where(df["grid"] > 0, n + 1)      # pit lane -> último
    df["quali_adj"] = df["quali_pos"].fillna(df["grid_adj"])
    st = df["status"].fillna("")
    df["finished"] = (st == "Finished") | st.str.startswith("+")
    df["dnf"] = (~df["finished"]).astype(float)

    key = [df["t"], df["team"]]
    # delta vs compañero (negativo = mejor que el compañero)
    fin_pos = df["pos"].where(df["finished"])
    s, c = fin_pos.groupby(key).transform("sum"), fin_pos.groupby(key).transform("count")
    mate = (s - fin_pos) / (c - 1)
    df["race_delta"] = (df["pos"] - mate).where(df["finished"] & (c >= 2))
    sq, cq = df["quali_adj"].groupby(key).transform("sum"), df["quali_adj"].groupby(key).transform("count")
    df["quali_delta"] = (df["quali_adj"] - (sq - df["quali_adj"]) / (cq - 1)).where(cq >= 2)
    df["gain"] = (df["grid_adj"] - df["pos"]).where(df["finished"])
    return df


# --------------------------------------------------------------------------
# Constructor de features
# --------------------------------------------------------------------------

FEATURE_COLS = [
    # grilla / pre-carrera
    "grid_adj", "grid_rel", "quali_adj", "grid_pen", "sprint_pos", "is_sprint",
    # auto (por equipo, con prior consciente del reglamento)
    "car_q", "car_f", "car_q_rank", "car_q_gap", "car_f_gap",
    "pu_car_q", "pu_car_f", "team_races_season", "has_prior", "regime_new",
    "grid_minus_car", "quali_minus_car",
    # piloto (relativo al compañero, portable entre equipos)
    "drv_race_delta", "drv_quali_delta", "drv_gain", "drv_dnf_rate", "drv_exp",
    "drv_races_with_team", "drv_circuit_delta", "drv_circuit_n",
    # circuito
    "circ_gain_abs", "circ_dnf_rate", "circ_n",
    # contexto
    "n_entries", "round_no",
]


class FeatureBuilder:
    def __init__(self, prepared: pd.DataFrame):
        self.df = prepared
        self.max_t = int(prepared["t"].max())

        self.drv = {}
        for d, g in prepared.groupby("driver_id"):
            g = g.sort_values("t")
            self.drv[d] = {
                "t": g["t"].to_numpy(), "race_delta": g["race_delta"].to_numpy(float),
                "quali_delta": g["quali_delta"].to_numpy(float), "gain": g["gain"].to_numpy(float),
                "dnf": g["dnf"].to_numpy(float), "team": g["team"].to_numpy(object),
                "circuit": g["circuit_id"].to_numpy(object),
            }

        # rendimiento del auto por (equipo, carrera), normalizado a [-1, 1]
        tr = prepared.groupby(["team", "t"]).agg(
            season=("season", "first"), n=("n_entries", "first"),
            q=("quali_adj", "mean"), f=("pos", "mean")).reset_index()
        tr["q_rel"] = (tr["q"] - (tr["n"] + 1) / 2) / (tr["n"] / 2)
        tr["f_rel"] = (tr["f"] - (tr["n"] + 1) / 2) / (tr["n"] / 2)
        self.tr = {team: g.sort_values("t") for team, g in tr.groupby("team")}

        # estadísticas por carrera para el circuito
        rt = prepared.groupby("t").agg(
            circuit=("circuit_id", "first"), gain_abs=("gain", lambda x: x.abs().mean()),
            dnf_rate=("dnf", "mean")).reset_index()
        self.race_tbl = rt

    # ---- auto -----------------------------------------------------------
    def _car(self, team: str, season: int, t0: int):
        """(q_blend, f_blend, n_cur, has_prior, q_cur_raw...) de un equipo para la carrera t0."""
        g = self.tr.get(team)
        regime = season in C.REGIME_CHANGE_SEASONS
        P = C.PRIOR_NEW_REGIME if regime else C.PRIOR_STABLE
        q_cur = f_cur = np.nan
        n_cur = 0
        q_pri = f_pri = np.nan
        if g is not None:
            before = g[g["t"] < t0]
            cur = before[before["season"] == season]
            n_cur = len(cur)
            if n_cur:
                w = cur.tail(C.CAR_WINDOW)
                q_cur, _ = _ew(w["q_rel"], C.CAR_HALF_LIFE)
                f_cur, _ = _ew(w["f_rel"], C.CAR_HALF_LIFE)
            prev = before[before["season"] < season]
            if len(prev):
                last_season = prev["season"].max()
                p = prev[prev["season"] == last_season].tail(8)
                q_pri, _ = _ew(p["q_rel"], C.CAR_HALF_LIFE)
                f_pri, _ = _ew(p["f_rel"], C.CAR_HALF_LIFE)
        has_prior = not np.isnan(q_pri)
        shrink = P["shrink"] if has_prior else 1.0
        q_p = 0.0 if not has_prior else (1 - shrink) * q_pri
        f_p = 0.0 if not has_prior else (1 - shrink) * f_pri
        w_p = P["prior_races"] * (0.85 ** n_cur) if has_prior else 1.0
        n_eff = min(n_cur, C.CAR_WINDOW)
        q = (n_eff * (q_cur if n_cur else 0.0) + w_p * q_p) / (n_eff + w_p)
        f = (n_eff * (f_cur if n_cur else 0.0) + w_p * f_p) / (n_eff + w_p)
        return q, f, n_cur, int(has_prior)

    # ---- carrera ----------------------------------------------------------
    def race_features(self, t0: int, season: int, rnd: int, circuit_id: str,
                      entries: pd.DataFrame) -> pd.DataFrame:
        """entries: driver_id, team (canónico), grid_adj, quali_adj, sprint_pos, is_sprint."""
        n = len(entries)
        regime_new = int(season in C.REGIME_CHANGE_SEASONS)

        # circuito
        rt = self.race_tbl[self.race_tbl["t"] < t0]
        glob_gain = rt["gain_abs"].mean() if len(rt) else 3.5
        glob_dnf = rt["dnf_rate"].mean() if len(rt) else C.DNF_PRIOR_RATE
        rc = rt[rt["circuit"] == circuit_id]
        k = 2.0
        circ_gain = (rc["gain_abs"].sum() + k * glob_gain) / (len(rc) + k)
        circ_dnf = (rc["dnf_rate"].sum() + k * glob_dnf) / (len(rc) + k)

        # autos por equipo
        teams = entries["team"].unique()
        car = {tm: self._car(tm, season, t0) for tm in teams}
        qs = pd.Series({tm: v[0] for tm, v in car.items()})
        fs = pd.Series({tm: v[1] for tm, v in car.items()})
        q_rank = qs.rank(method="min")
        q_gap, f_gap = qs - qs.min(), fs - fs.min()
        pu = {tm: C.power_unit(tm, season) for tm in teams}
        pu_q, pu_f = {}, {}
        for tm in teams:
            mates = [x for x in teams if pu[x] is not None and pu[x] == pu[tm]] or [tm]
            pu_q[tm], pu_f[tm] = qs[mates].mean(), fs[mates].mean()

        rows = []
        for e in entries.itertuples(index=False):
            h = self.drv.get(e.driver_id)
            if h is None:
                kk, rd, qd, gn, dn, tm_hist, ci_hist = 0, [], [], [], [], [], []
                hist_circuit_delta = []
            else:
                kk = int(np.searchsorted(h["t"], t0, side="left"))
                lo = max(0, kk - C.DRIVER_WINDOW)
                rd, qd, gn, dn = (h["race_delta"][lo:kk], h["quali_delta"][lo:kk],
                                  h["gain"][lo:kk], h["dnf"][lo:kk])
                tm_hist = h["team"][:kk]
                m = h["circuit"][:kk] == circuit_id
                hist_circuit_delta = h["race_delta"][:kk][m]
            dnf_ew, dn_n = _ew(dn, C.DRIVER_HALF_LIFE)
            dnf_rate = C.DNF_PRIOR_RATE if dn_n == 0 else (
                dnf_ew * dn_n + C.DNF_PRIOR_RATE * C.DNF_PRIOR_N) / (dn_n + C.DNF_PRIOR_N)
            circ_n_d = int(np.sum(~np.isnan(hist_circuit_delta))) if len(hist_circuit_delta) else 0
            circ_delta = 0.0 if circ_n_d == 0 else float(np.nanmean(hist_circuit_delta)) * circ_n_d / (circ_n_d + 3.0)
            tm = e.team
            car_q, car_f = qs[tm], fs[tm]
            grid_rel = (e.grid_adj - (n + 1) / 2) / (n / 2)
            quali_rel = (e.quali_adj - (n + 1) / 2) / (n / 2)
            rows.append({
                "driver_id": e.driver_id,
                "grid_adj": e.grid_adj, "grid_rel": grid_rel, "quali_adj": e.quali_adj,
                "grid_pen": max(0.0, e.grid_adj - e.quali_adj),
                "sprint_pos": e.sprint_pos, "is_sprint": int(e.is_sprint),
                "car_q": car_q, "car_f": car_f, "car_q_rank": q_rank[tm],
                "car_q_gap": q_gap[tm], "car_f_gap": f_gap[tm],
                "pu_car_q": pu_q[tm], "pu_car_f": pu_f[tm],
                "team_races_season": car[tm][2], "has_prior": car[tm][3], "regime_new": regime_new,
                "grid_minus_car": grid_rel - car_q, "quali_minus_car": quali_rel - car_q,
                "drv_race_delta": _shrunk(rd, C.DRIVER_HALF_LIFE, 3.0),
                "drv_quali_delta": _shrunk(qd, C.DRIVER_HALF_LIFE, 3.0),
                "drv_gain": _shrunk(gn, C.DRIVER_HALF_LIFE, 5.0),
                "drv_dnf_rate": dnf_rate, "drv_exp": min(kk, 100),
                "drv_races_with_team": int(np.sum(np.asarray(tm_hist, dtype=object) == tm)) if kk else 0,
                "drv_circuit_delta": circ_delta, "drv_circuit_n": circ_n_d,
                "circ_gain_abs": circ_gain, "circ_dnf_rate": circ_dnf, "circ_n": len(rc),
                "n_entries": n, "round_no": rnd,
            })
        return pd.DataFrame(rows)[["driver_id"] + FEATURE_COLS]


# --------------------------------------------------------------------------
# Tabla de entrenamiento
# --------------------------------------------------------------------------

def build_training_table(prepared: pd.DataFrame, warmup_races: int = 8) -> pd.DataFrame:
    fb = FeatureBuilder(prepared)
    out = []
    for t0, g in prepared.groupby("t", sort=True):
        if t0 < warmup_races:
            continue
        first = g.iloc[0]
        feats = fb.race_features(int(t0), int(first["season"]), int(first["round"]),
                                 first["circuit_id"], g.assign(is_sprint=g["is_sprint_weekend"]))
        meta = g[["driver_id", "code", "constructor_id", "race_name", "season", "round", "t",
                  "pos", "dnf", "points"]].reset_index(drop=True)
        out.append(meta.merge(feats, on="driver_id", how="left"))
    return pd.concat(out, ignore_index=True)


def build_inference_features(prepared_hist: pd.DataFrame, season: int, rnd: int, circuit_id: str,
                             entries_raw: pd.DataFrame) -> pd.DataFrame:
    """entries_raw: driver_id, constructor_id y (opcional) grid, quali_pos, sprint_pos, is_sprint.
    Usa el MISMO FeatureBuilder que el entrenamiento, con todo el historial disponible."""
    fb = FeatureBuilder(prepared_hist)
    e = entries_raw.copy()
    n = len(e)
    e["team"] = e["constructor_id"].map(canonical_team)
    if "grid" not in e or e["grid"].isna().any():
        raise ValueError("Falta la grilla (usá predict.estimate_grid para un pronóstico pre-clasificación).")
    e["grid_adj"] = e["grid"].where(e["grid"] > 0, n + 1)
    e["quali_adj"] = e["quali_pos"].fillna(e["grid_adj"]) if "quali_pos" in e else e["grid_adj"]
    if "sprint_pos" not in e:
        e["sprint_pos"] = np.nan
    if "is_sprint" not in e:
        e["is_sprint"] = int(e["sprint_pos"].notna().any())
    t0 = fb.max_t + 1
    feats = fb.race_features(t0, season, rnd, circuit_id, e)
    return feats
