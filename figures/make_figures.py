#!/usr/bin/env python3
"""
figures/make_figures.py — figures and tables for the professor update.

Part 1 of N: input loading and validation. No plotting yet.
Run from anywhere:  python figures/make_figures.py
"""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef)

ROOT    = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA    = ROOT / "data"
OUT     = ROOT / "figures"
RUNS    = RESULTS / "runs"

GRID_N    = (256, 512, 1024, 2048, 4096)
GRID_SEED = (0, 1, 2)
SYSTEMS   = ["S1 consequence rule", "S2 frozen probe", "S3 full FT",
             "S4 LoRA r=16", "S5 composite"]
STRATA    = ["overall", "LoF", "missense", "silent/non-coding", "hard-eval"]

FAIL = []


def check(cond, msg):
    """Record a failure and report it immediately, so nothing crashes silently."""
    if not cond:
        FAIL.append(msg)
        print(f"  FAIL: {msg}")
    return cond


def banner(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


# --------------------------------------------------------------------------
# 1. presence
# --------------------------------------------------------------------------
banner("1. INPUT FILES")

NEEDED = {
    "matrix":     RESULTS / "test_matrix.csv",
    "robustness": RESULTS / "test_robustness.json",
    "meta":       RESULTS / "test_meta.json",
    "splits":     DATA / "gene_splits_v2.csv",
    "card":       DATA / "dataset_card_v2.json",
    "val_meta":   RESULTS / "emb_val_1024bp.npz",
    "thresholds": RESULTS / "test_thresholds.json",
}
for k, p in NEEDED.items():
    ok = p.exists()
    print(f"  {'ok ' if ok else 'MISSING':<8} {k:<11} {p.relative_to(ROOT)}")
    check(ok, f"missing input: {p}")

check(RUNS.is_dir(), f"missing run directory: {RUNS}")
if FAIL:
    print("\n".join(FAIL))
    sys.exit(1)

OUT.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# 2. test matrix
# --------------------------------------------------------------------------
banner("2. test_matrix.csv")

MX = pd.read_csv(NEEDED["matrix"])
print(f"  {len(MX)} rows | {MX.system.nunique()} systems x {MX.stratum.nunique()} strata")

check(len(MX) == 25, f"expected 25 rows, got {len(MX)}")
check(set(MX.system) == set(SYSTEMS), f"unexpected systems: {sorted(set(MX.system))}")
check(set(MX.stratum) == set(STRATA), f"unexpected strata: {sorted(set(MX.stratum))}")

MX["system"] = pd.Categorical(MX.system, SYSTEMS, ordered=True)
MX["stratum"] = pd.Categorical(MX.stratum, STRATA, ordered=True)
MX = MX.sort_values(["stratum", "system"]).reset_index(drop=True)

OV = MX[MX.stratum == "overall"].set_index("system")
check(int(OV.n.iloc[0]) == 48328, f"overall n = {int(OV.n.iloc[0])}, expected 48,328")
check(int(OV.pos.iloc[0]) == 9284, f"overall pos = {int(OV.pos.iloc[0])}, expected 9,284")
check(OV.n.nunique() == 1, "systems disagree on overall row count")

s5 = MX[MX.system == "S5 composite"]
check(s5.auc.notna().all(), "S5 missing AUC in some stratum")
check(s5[["mcc", "f1"]].isna().all().all(), "S5 has threshold metrics — expected NaN")

print("\n  overall AUC / MCC / AUPRC")
for s in SYSTEMS:
    r = OV.loc[s]
    mcc = "—" if pd.isna(r.mcc) else f"{r.mcc:.4f}"
    print(f"    {s:<22} {r.auc:.4f}  {mcc:>6}  {r.auprc:.4f}")

print("\n  AUC by stratum")
piv = MX.pivot(index="system", columns="stratum", values="auc")
print(piv.to_string(float_format=lambda v: f"{v:.4f}", na_rep="—"))


# --------------------------------------------------------------------------
# 3. robustness
# --------------------------------------------------------------------------
banner("3. test_robustness.json")

ROB = json.loads(NEEDED["robustness"].read_text())
CI = ROB["clustered_ci"]
CI_KEYS = ["comp_gain_auc", "comp_gain_auprc", "mis_full_probe", "mis_lora_full",
           "lof_lora_full", "dod", "overall_lora_full"]

check(set(CI) == set(CI_KEYS), f"unexpected CI keys: {sorted(set(CI))}")
for k, v in CI.items():
    check(v["lo"] <= v["point"] <= v["hi"], f"{k}: point outside its interval")
    check(v["excludes_zero"] == (v["lo"] > 0 or v["hi"] < 0),
          f"{k}: excludes_zero disagrees with the interval")

excl = [k for k, v in CI.items() if v["excludes_zero"]]
check(set(excl) == {"comp_gain_auc", "comp_gain_auprc"},
      f"intervals excluding zero: {excl} — expected the two composite-gain ones")

# the composite gain should equal composite AUC minus rule AUC in the matrix
gain = OV.loc["S5 composite"].auc - OV.loc["S1 consequence rule"].auc
check(abs(gain - CI["comp_gain_auc"]["point"]) < 1e-6,
      f"comp_gain_auc {CI['comp_gain_auc']['point']:.6f} != matrix difference {gain:.6f}")

print(f"  {'quantity':<20}{'point':>9}{'95% CI':>22}{'excl 0':>8}")
for k in CI_KEYS:
    v = CI[k]
    print(f"  {k:<20}{v['point']:>+9.4f}   {v['lo']:>+8.4f} .. {v['hi']:>+8.4f}"
          f"{'yes' if v['excludes_zero'] else 'no':>8}")

DEC = ROB["decomposition"]
for k, v in DEC.items():
    check(v["within"] >= v["pooled"] - 1e-9,
          f"decomposition {k}: within {v['within']:.4f} below pooled {v['pooled']:.4f}")
print(f"\n  within-gene AUC exceeds pooled for all {len(DEC)} systems: "
      f"{all(v['within'] >= v['pooled'] for v in DEC.values())}")

PG = ROB["per_gene_missense"]
print(f"  per-gene missense: full FT > probe in {PG['full_gt_probe']}/{PG['n_genes']} "
      f"genes, sign test p={PG['sign_test_p']:.4f}, median lift {PG['median_lift']:+.4f}")


# --------------------------------------------------------------------------
# 4. meta
# --------------------------------------------------------------------------
banner("4. test_meta.json")

META = json.loads(NEEDED["meta"].read_text())
RATE = META["seq_per_s"]
check(set(RATE) == {"probe", "full", "lora"}, f"seq_per_s keys: {sorted(RATE)}")
check(META["n"] == int(OV.n.iloc[0]), "meta n disagrees with test_matrix")
check(META["pos"] == int(OV.pos.iloc[0]), "meta pos disagrees with test_matrix")

slow = 100 * (RATE["full"] - RATE["lora"]) / RATE["full"]
print(f"  inference: " + " | ".join(f"{k} {v:.1f}" for k, v in RATE.items()) + " seq/s")
print(f"  LoRA is {slow:.1f}% slower than full FT at inference")
MERGE = json.loads((NEEDED["meta"].parent / "inference_merge.json").read_text())
check(abs(MERGE["full_drift_pct"]) < 5,
      f"full FT drifted {MERGE['full_drift_pct']:+.1f}% across the T3C session")
check(abs(MERGE["seq_lora_unmerged"] / MERGE["seq_full"]
          - RATE["lora"] / RATE["full"]) < 0.03,
      "T3C live-LoRA ratio disagrees with T4")
print(f"  T3C: merged {MERGE['seq_lora_merged']:.0f} seq/s "
      f"({100*(MERGE['seq_lora_merged']/MERGE['seq_full']-1):+.1f}% vs full FT)"
      f" | merging recovers "
      f"{100*(MERGE['seq_lora_merged']/MERGE['seq_lora_unmerged']-1):+.1f}%")
print(f"  frozen thresholds: full {META['thr_full']:.4f} @ step {META['step_full']}"
      f" | LoRA {META['thr_lora']:.4f} @ step {META['step_lora']}")


# --------------------------------------------------------------------------
# 5. splits
# --------------------------------------------------------------------------
banner("5. gene_splits_v2.csv + dataset_card_v2.json")

GS = pd.read_csv(NEEDED["splits"], dtype={"gene": str})
CARD = json.loads(NEEDED["card"].read_text())

SPL = (GS.groupby("split")
         .agg(genes=("gene", "nunique"),
              rows=("n_variants", "sum"),
              positives=("n_pos", "sum"))
         .reindex(["train", "val", "test"]))
SPL["prevalence"] = SPL.positives / SPL.rows

check(len(GS) == 629, f"{len(GS)} genes, expected 629")
check(GS.gene.is_unique, "gene column is not unique")
check(int(SPL.rows.sum()) == 92019, f"{int(SPL.rows.sum())} rows, expected 92,019")
check(list(SPL.genes) == [434, 155, 40], f"genes per split: {list(SPL.genes)}")

for split, c in CARD["split_summary"].items():
    r = SPL.loc[split]
    check(c["genes"] == r.genes and c["variants"] == r.rows
          and c["positives"] == r.positives,
          f"card disagrees with gene_splits for {split}")

HARD_EVAL = int(GS.loc[GS.split == "test", "movers"].sum())
check(HARD_EVAL == 1216, f"hard-eval from movers = {HARD_EVAL}, expected 1,216")
check(HARD_EVAL == int(MX[(MX.stratum == "hard-eval")].n.iloc[0]),
      "hard-eval count disagrees with test_matrix")

print(SPL.assign(prevalence=lambda d: (100 * d.prevalence).round(1).astype(str) + "%")
         .to_string())
print(f"\n  hard-eval subset: {HARD_EVAL:,} (test-side reclassified VUS)")
print(f"  card and gene_splits agree on all three splits")


# --------------------------------------------------------------------------
# 6. grid runs
# --------------------------------------------------------------------------
banner("6. results/runs/*/metrics.json")

# The grid is one learning rate per arm, untagged. Everything else in runs/ is
# an lr sweep (lr_bb differs), a patience probe (tag "_nopatience", a duplicate
# of its untagged twin), a full-data run (N=37,241), or a non-grid artifact.
GRID_LR = {"full": 2e-5, "lora": 1e-4}

rows = []
for p in sorted(RUNS.glob("*/metrics.json")):
    m = json.loads(p.read_text())
    rows.append({"run_id": p.parent.name,
                 "arm": m.get("arm"), "N": m.get("N", m.get("n_train")),
                 "seed": m.get("seed"), "lr_bb": m.get("lr_bb"),
                 "tag": m.get("tag"), "auc": m.get("auc"), "mcc": m.get("mcc"),
                 "trainable": m.get("trainable"), "ms": m.get("ms_per_step"),
                 "gb": m.get("peak_gb"), "steps_run": m.get("steps_run"),
                 "session": m.get("session_id"),
                 "comparable": m.get("cost_comparable"),
                 "has_verdict": "verdict" in m,
                 "verdict": m.get("verdict", "—")})
R = pd.DataFrame(rows)
print(f"  {len(R)} run directories")

is_grid = (R.arm.isin(GRID_LR)
           & R.N.isin(GRID_N)
           & R.seed.isin(GRID_SEED)
           & (R.tag.fillna("non-grid") == "")
           & R.apply(lambda r: r.arm in GRID_LR
                     and r.lr_bb is not None
                     and np.isclose(r.lr_bb, GRID_LR.get(r.arm, np.nan)), axis=1))
G = R[is_grid].copy()
G["N"] = G.N.astype(int)
G["seed"] = G.seed.astype(int)

print(f"  excluded: {(~is_grid).sum()} runs "
      f"({(R.tag.fillna('') != '').sum()} tagged, "
      f"{(R.arm.isin(GRID_LR) & R.N.isin(GRID_N) & ~is_grid & (R.tag.fillna('') == '')).sum()} off-lr, "
      f"rest out of grid)")
check(len(G) == 30, f"{len(G)} grid cells, expected 30")

dupes = G[G.duplicated(["arm", "N", "seed"], keep=False)]
check(dupes.empty, f"duplicate (arm,N,seed): {list(dupes.run_id)}")
for col in ["auc", "trainable", "ms", "gb", "steps_run", "session"]:
    check(G[col].notna().all(),
          f"{col} missing in: {list(G.loc[G[col].isna(), 'run_id'])}")

TRAIN = G.groupby("arm").trainable.unique()
check(all(len(v) == 1 for v in TRAIN), f"trainable varies within an arm: {dict(TRAIN)}")
n_full, n_lora = int(TRAIN["full"][0]), int(TRAIN["lora"][0])
print(f"  trainable: full {n_full:,} | lora {n_lora:,} ({n_full / n_lora:.0f}x fewer)")

nb08 = G[~G.has_verdict]
if len(nb08):
    print(f"  nb08-era cells (no `verdict` field): {list(nb08.run_id)}")

P = G.pivot(index=["N", "seed"], columns="arm",
            values=["auc", "ms", "gb", "steps_run", "session", "comparable"])
P.columns = [f"{a}_{b}" for a, b in P.columns]
check(len(P) == 15, f"{len(P)} (N,seed) pairs, expected 15")
check(P[[c for c in P.columns if not c.startswith("comparable")]].notna().all().all(),
      "a (N,seed) pair is missing an arm")

# --- pre-declared pooled paired AUC difference (all 15 pairs) --------------
from scipy import stats

# --- pre-declared pooled paired AUC difference (all 15 pairs) --------------
# nb10 cell D2 uses the t interval on 14 df, not a normal approximation.
d = (P.auc_lora - P.auc_full).values
m, sd = d.mean(), d.std(ddof=1)
se = sd / np.sqrt(len(d))
tcrit = stats.t.ppf(0.975, len(d) - 1)
CI_LO, CI_HI = m - tcrit * se, m + tcrit * se
print(f"\n  pooled paired LoRA - full FT AUC, {len(d)} pairs:")
print(f"    {m:+.4f} +/- {sd:.4f} | SE {se:.4f} | "
      f"95% t CI {CI_LO:+.4f} .. {CI_HI:+.4f}")
check(abs(m - (-0.0045)) < 5e-4, f"pooled difference {m:+.4f}, expected -0.0045")
check(CI_LO < 0 < CI_HI, "pooled interval no longer contains zero")

# --- cost: two different pair sets, per nb09 -------------------------------
# ms/step  -> cost_comparable only (a per-step average, run length irrelevant)
# peak VRAM -> cost_comparable AND matched steps_run, because peak memory
#              depends on which batches were seen (nb09's stated reason)
P["flag_ok"] = P.comparable_full.eq(True) & P.comparable_lora.eq(True)
P["steps_ok"] = P.steps_run_full == P.steps_run_lora

MS = P[P.flag_ok]
GB = P[P.flag_ok & P.steps_ok]
check(len(MS) == 13, f"{len(MS)} pairs flagged comparable, expected 13")
check(len(GB) >= 4, f"only {len(GB)} matched-step comparable pairs")

dms = 100 * (MS.ms_lora - MS.ms_full) / MS.ms_full
dgb = GB.gb_lora - GB.gb_full
print(f"\n  cost")
print(f"    ms/step   {dms.mean():+.2f}% +/- {dms.std(ddof=1):.2f}   "
      f"({len(MS)} of {len(P)} pairs, cost_comparable)")
print(f"    peak VRAM {dgb.mean():+.4f} +/- {dgb.std(ddof=1):.4f} GB   "
      f"({len(GB)} of {len(P)} pairs, also matched steps_run)")
check(abs(dms.mean() - 9.01) < 0.2, f"ms/step {dms.mean():+.2f}%, expected +9.01%")
check(abs(dgb.mean() - (-0.0404)) < 5e-3,
      f"peak VRAM {dgb.mean():+.4f} GB, expected -0.0404")

print(f"    ms/step excludes:  {[tuple(i) for i in P[~P.flag_ok].index]}")
print(f"    VRAM also excludes: "
      f"{[tuple(i) for i in P[P.flag_ok & ~P.steps_ok].index]}")

# --- inference side, for the same figure -----------------------------------
print(f"\n  inference (test set, from test_meta.json)")
print(f"    {RATE['full']:.0f} seq/s full FT vs {RATE['lora']:.0f} seq/s LoRA "
      f"-> {100 * (RATE['full'] / RATE['lora'] - 1):+.0f}% slower")

COST = {"trainable_full": n_full, "trainable_lora": n_lora,
        "ms_pct": float(dms.mean()), "ms_sd": float(dms.std(ddof=1)),
        "ms_n": int(len(MS)),
        "gb_delta": float(dgb.mean()), "gb_sd": float(dgb.std(ddof=1)),
        "gb_n": int(len(GB)),
        "seq_full": RATE["full"], "seq_lora": RATE["lora"],
        "seq_full_t3c": MERGE["seq_full"],
        "seq_lora_live": MERGE["seq_lora_unmerged"],
        "seq_lora_merged": MERGE["seq_lora_merged"],
        "gb_inf_full": MERGE["gb_full"],
        "gb_inf_lora_merged": MERGE["gb_lora_merged"]}
GRID_STAT = {"mean": float(m), "sd": float(sd), "se": float(se),
             "lo": float(CI_LO), "hi": float(CI_HI), "n": int(len(d))}
# --------------------------------------------------------------------------
banner("VALIDATION")
if FAIL:
    print(f"{len(FAIL)} problem(s):")
    for f in FAIL:
        print(f"  - {f}")
    sys.exit(1)
print("all checks passed — inputs are consistent, ready to plot")

# ==========================================================================
# 7. STYLE
# ==========================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito, colourblind-safe. Fixed system -> colour mapping used everywhere.
CB = {"orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
      "purple": "#CC79A7", "black": "#000000", "grey": "#999999"}

SYS_COLOR = {"S1 consequence rule": CB["black"],
             "S2 frozen probe":     CB["grey"],
             "S3 full FT":          CB["blue"],
             "S4 LoRA r=16":        CB["orange"],
             "S5 composite":        CB["green"]}

SYS_SHORT = {"S1 consequence rule": "consequence\nrule",
             "S2 frozen probe":     "frozen\nprobe",
             "S3 full FT":          "full FT",
             "S4 LoRA r=16":        "LoRA r=16",
             "S5 composite":        "composite"}

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False,          # set per-axes via grid(), so the axis can vary
    "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "legend.frameon": False, "figure.autolayout": False,
})

WRITTEN = []


def save(fig, name):
    p = OUT / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    WRITTEN.append(p)
    print(f"    {p.relative_to(ROOT)}")


def write_table(df, name, floatfmt=None):
    """CSV for the record, markdown for pasting."""
    csv_p = OUT / f"{name}.csv"
    df.to_csv(csv_p, index=False)
    WRITTEN.append(csv_p)

    d = df.copy()
    if floatfmt:
        for c, f in floatfmt.items():
            if c in d:
                d[c] = d[c].map(lambda v: "—" if pd.isna(v) else f.format(v))
    d = d.astype(str)
    widths = [max(len(c), *(len(v) for v in d[c])) for c in d.columns]
    lines = ["| " + " | ".join(c.ljust(w) for c, w in zip(d.columns, widths)) + " |",
             "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + " |")
    md_p = OUT / f"{name}.md"
    md_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    WRITTEN.append(md_p)
    print(f"    {csv_p.relative_to(ROOT)} + .md")
    return "\n".join(lines)


# ==========================================================================
# 8. TABLES
# ==========================================================================
banner("8. TABLES")

# --- Table 1: dataset splits ----------------------------------------------
T1 = (SPL.reset_index()
         .rename(columns={"split": "Split", "genes": "Genes",
                          "rows": "Variants", "positives": "Pathogenic"}))
T1["Split"] = T1.Split.map({"train": "Train", "val": "Validation",
                            "test": "Test (sealed)"})
T1["Prevalence"] = (100 * T1.Pathogenic / T1.Variants).round(1).astype(str) + "%"
T1 = T1[["Split", "Genes", "Variants", "Pathogenic", "Prevalence"]]
T1 = pd.concat([T1, pd.DataFrame([{
    "Split": "  of which hard-eval", "Genes": "—", "Variants": HARD_EVAL,
    "Pathogenic": int(MX[MX.stratum == "hard-eval"].pos.iloc[0]),
    "Prevalence": f"{100 * MX[MX.stratum == 'hard-eval'].pos.iloc[0] / HARD_EVAL:.1f}%",
}])], ignore_index=True)
for c in ["Variants", "Pathogenic"]:
    T1[c] = T1[c].map(lambda v: f"{int(v):,}")

print("\n  Table 1 — dataset splits (gene-disjoint)")
print(write_table(T1, "table1_splits"))

# --- Table 2: overall test results ----------------------------------------
T2 = (OV.reset_index()[["system", "auc", "mcc", "auprc", "f1",
                        "precision", "recall"]]
        .rename(columns={"system": "System", "auc": "AUC", "mcc": "MCC",
                         "auprc": "AUPRC", "f1": "F1",
                         "precision": "Precision", "recall": "Recall"}))
T2["System"] = T2.System.astype(str)
print("\n  Table 2 — test set, all systems, overall stratum "
      f"({int(OV.n.iloc[0]):,} variants, {int(OV.pos.iloc[0]):,} pathogenic)")
print(write_table(T2, "table2_test_overall",
                  {"AUC": "{:.4f}", "MCC": "{:.4f}", "AUPRC": "{:.4f}",
                   "F1": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}"}))

# --- Table 4: test AUC by stratum ------------------------------------------
STRATA_T4 = ["LoF", "missense", "silent/non-coding", "hard-eval"]
COL_T4 = {"LoF": "LoF", "missense": "Missense",
          "silent/non-coding": "Silent / non-coding",
          "hard-eval": "Hard-eval"}

S4 = MX[MX.stratum.isin(STRATA_T4)]
check(len(S4) == len(SYSTEMS) * len(STRATA_T4),
      f"stratum matrix has {len(S4)} rows, expected {len(SYSTEMS) * len(STRATA_T4)}")

T4 = S4.pivot(index="system", columns="stratum", values="auc")
T4 = T4.reindex(index=SYSTEMS, columns=STRATA_T4)
check(not T4.isna().any().any(), "missing system x stratum cells in test_matrix")

# the rule is constant within a stratum, so its AUC is 0.5 by construction
check((T4.loc["S1 consequence rule"] == 0.5).all(),
      f"rule is not 0.5 in every stratum: {T4.loc['S1 consequence rule'].to_dict()}")
# the composite is a monotone transform of full FT within a stratum
check(np.allclose(T4.loc["S5 composite"], T4.loc["S3 full FT"], atol=1e-9),
      "composite differs from full FT within a stratum — it should not")

T4 = T4.drop(index=["S1 consequence rule", "S5 composite"])

N4 = S4.drop_duplicates("stratum").set_index("stratum").n
T4.columns = [f"{COL_T4[s]}\n(n = {int(N4[s]):,})" for s in T4.columns]
T4 = T4.reset_index().rename(columns={"system": "System"})
T4["System"] = T4.System.astype(str)

print("\n  Table 4 — test AUC by stratum")
print(f"    rule = 0.5000 in every stratum (constant score within a stratum); "
      f"composite identical to full FT (monotone transform) — both omitted")
print(write_table(T4, "table4_test_by_stratum",
                  {c: "{:.4f}" for c in T4.columns if c != "System"}))

# --- Table 3: validation, five systems ------------------------------------
VAL_RUNS = {"probe": ("step0_fulldata_n37241_s0", "val_logits_step0.npz", "p_s0"),
            "full":  ("full_lr2e-05_n37241_s0_fd_nopatience",
                      "val_logits_top3.npz", "p_s3200"),
            "lora":  ("lora_r16_lr0.0001_n37241_s0_fulldata",
                      "val_logits_top3.npz", "p_s3000")}

LOF_SO = {"SO:0001587", "SO:0001575", "SO:0001574"}   # nonsense, splice donor/acceptor
MIS_SO = {"SO:0001583", "SO:0001582"}                 # missense, initiator codon

EXP_STRATA = {"LoF": (582, 575), "missense": (1514, 535),
              "silent/non-coding": (4354, 15)}        # nb10 D4


def _stratum(s):
    """nb10 D4 / nb11 T5, verbatim."""
    ts = {t.split("|")[0] for t in str(s).split(",") if t}
    if ts & LOF_SO:
        return "LoF"
    if ts & MIS_SO:
        return "missense"
    return "silent/non-coding"


def table3():
    n0 = len(FAIL)

    # only `label` and `mc` are decompressed; the two 768-d arrays are untouched
    _z = np.load(NEEDED["val_meta"], allow_pickle=True)
    VF = pd.DataFrame({"label": _z["label"].astype(int), "mc": _z["mc"]})
    del _z
    check(len(VF) == 6450, f"emb_val has {len(VF)} rows, expected 6,450")
    y = VF.label.values
    check(int(y.sum()) == 1125, f"val positives {int(y.sum())}, expected 1,125")

    strat = np.array([_stratum(s) for s in VF.mc])
    counts = {k: (int((strat == k).sum()), int(y[strat == k].sum()))
              for k in EXP_STRATA}
    check(counts == EXP_STRATA,
          f"tier mapping gives {counts}, does not reproduce nb10 D4 {EXP_STRATA}")
    tier = np.select([strat == "LoF", strat == "missense"], [2.0, 1.0], 0.0)

    P, REF = {}, {}
    for arm, (rid, fn, key) in VAL_RUNS.items():
        z = np.load(RUNS / rid / fn)
        check(key in z.files, f"{rid}/{fn}: {key} absent, has {list(z.files)}")
        check(np.array_equal(z["label"].astype(int), y),
              f"{rid}: label order differs from emb_val — rows are misaligned")
        P[arm] = z[key]
        REF[arm] = json.loads((RUNS / rid / "metrics.json").read_text())

    thr_rule = json.loads(NEEDED["thresholds"].read_text())["thr"]["rule"]
    SPEC = [("S1 consequence rule", tier,       thr_rule,          None),
            ("S2 frozen probe",     P["probe"], REF["probe"]["thr"], "probe"),
            ("S3 full FT",          P["full"],  REF["full"]["thr"],  "full"),
            ("S4 LoRA r=16",        P["lora"],  REF["lora"]["thr"],  "lora"),
            ("S5 composite",        2.0 * tier + P["full"], None,    None)]

    rows = []
    for name, p, thr, ref in SPEC:
        r = {"system": name, "n": len(y), "pos": int(y.sum()),
             "auc": float(roc_auc_score(y, p)),
             "auprc": float(average_precision_score(y, p)),
             "thr": None if thr is None else float(thr)}
        if thr is None:                                   # S5: ranking only
            r.update(mcc=np.nan, f1=np.nan, precision=np.nan, recall=np.nan)
        else:
            yh = (p >= thr).astype(int)
            tp = int(((yh == 1) & (y == 1)).sum())
            fp = int(((yh == 1) & (y == 0)).sum())
            fn = int(((yh == 0) & (y == 1)).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            r.update(mcc=float(matthews_corrcoef(y, yh)),
                     f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                     precision=prec, recall=rec)
        if ref:                    # recomputed must equal what the run stored
            m = REF[ref]
            for k in ("auc", "mcc", "auprc", "f1", "precision", "recall"):
                check(abs(r[k] - m[k]) < 1e-6,
                      f"{name}: recomputed {k} {r[k]:.6f} "
                      f"!= metrics.json {m[k]:.6f}")
        rows.append(r)

    V = pd.DataFrame(rows)
    V["system"] = pd.Categorical(V.system, SYSTEMS, ordered=True)
    V = V.sort_values("system")

    ix = V.set_index("system")
    check(abs(ix.loc["S3 full FT", "thr"] - META["thr_full"]) < 1e-9,
          "full-FT validation threshold differs from the one frozen for test")
    check(abs(ix.loc["S4 LoRA r=16", "thr"] - META["thr_lora"]) < 1e-9,
          "LoRA validation threshold differs from the one frozen for test")

    if len(FAIL) > n0:
        print("\n".join(FAIL[n0:]))
        sys.exit(1)

    print(f"  strata: " + " | ".join(f"{k} {v[0]:,}/{v[1]}"
                                     for k, v in counts.items()))
    print(f"  checkpoints: probe step 0 | full step {REF['full']['step']:,} "
          f"| LoRA step {REF['lora']['step']:,}")

    T3 = V[["system", "auc", "mcc", "auprc", "f1", "thr"]].copy()
    T3.columns = ["System", "AUC", "MCC", "AUPRC", "F1", "Threshold"]
    return write_table(T3, "table3_validation",
                       floatfmt={"AUC": "{:.4f}", "MCC": "{:.4f}",
                                 "AUPRC": "{:.4f}", "F1": "{:.4f}",
                                 "Threshold": "{:.6f}"})


print(table3())



# ==========================================================================
# 9. FIGURES
# ==========================================================================
banner("9. FIGURES")

STRAT_LABEL = {"overall": "Overall\n48,328", "LoF": "Loss of\nfunction",
               "missense": "Missense", "silent/non-coding": "Silent /\nnon-coding",
               "hard-eval": "Hard-eval\n(reclassified)"}

# --- F1: AUC by system and stratum ----------------------------------------
def fig1():
    piv = MX.pivot(index="stratum", columns="system", values="auc").loc[STRATA]
    base = MX.groupby("stratum", observed=True)[["n", "pos"]].first().loc[STRATA]
    prev = 100 * base.pos / base.n

    fig, (ax, axp) = plt.subplots(
        2, 1, sharex=True, figsize=(7.4, 5.0),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.20})

    x = np.arange(len(STRATA))
    w = 0.15
    for i, s in enumerate(SYSTEMS):
        ax.bar(x + (i - 2) * w, piv[s].values, w,
               label=SYS_SHORT[s].replace("\n", " "),
               color=SYS_COLOR[s], edgecolor="white", linewidth=0.4)

    ax.axhline(0.5, color=CB["vermillion"], lw=0.9, ls="--", zorder=1)
    ax.annotate("chance", xy=(len(STRATA) - 0.45, 0.5), xytext=(0, 3),
                textcoords="offset points", fontsize=7,
                color=CB["vermillion"], ha="right")
    for xi, s in zip(x, STRATA):
        if s in ("LoF", "missense", "silent/non-coding"):
            ax.annotate("0.500\nby construction", xy=(xi - 2 * w, 0.5),
                        xytext=(0, -14), textcoords="offset points",
                        fontsize=6, ha="center", va="top", color=CB["grey"])

    ax.set_ylim(0.42, 1.02)
    ax.set_ylabel("ROC AUC")
    ax.set_title("The rule discriminates between strata, not within them")
    ax.grid(axis="y")

    axp.bar(x, prev.values, 0.55, color=CB["purple"],
            edgecolor="white", linewidth=0.4)
    for xi, v in zip(x, prev.values):
        axp.annotate(f"{v:.1f}%", xy=(xi, v), xytext=(0, 2),
                     textcoords="offset points", fontsize=7, ha="center")
    axp.set_ylim(0, 118)
    axp.set_yticks([0, 50, 100])
    axp.set_ylabel("% pathogenic")
    axp.grid(axis="y")
    axp.set_xticks(x)
    axp.set_xticklabels([f"{STRAT_LABEL[s]}\nn={base.n[s]:,}" for s in STRATA])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), columnspacing=1.2, handlelength=1.2)
    fig.subplots_adjust(bottom=0.22)
    save(fig, "fig1_auc_by_stratum")

def fig1a():
    """Prevalence by stratum — a property of the data (Chapter 3)."""
    base = MX.groupby("stratum", observed=True)[["n", "pos"]].first().loc[STRATA]
    prev = 100 * base.pos / base.n

    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    x = np.arange(len(STRATA))
    ax.bar(x, prev.values, 0.55, color=CB["purple"],
           edgecolor="white", linewidth=0.4)
    for xi, v in zip(x, prev.values):
        ax.annotate(f"{v:.1f}%", xy=(xi, v), xytext=(0, 2),
                    textcoords="offset points", fontsize=7, ha="center")
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("% pathogenic")
    ax.grid(axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{STRAT_LABEL[s]}\nn={base.n[s]:,}" for s in STRATA])
    fig.subplots_adjust(bottom=0.26)
    save(fig, "fig1a_prevalence_by_stratum")


def fig1b():
    """AUC by system and stratum — a result (Chapter 5)."""
    piv = MX.pivot(index="stratum", columns="system", values="auc").loc[STRATA]
    base = MX.groupby("stratum", observed=True)[["n", "pos"]].first().loc[STRATA]

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    x = np.arange(len(STRATA))
    w = 0.15
    for i, s in enumerate(SYSTEMS):
        ax.bar(x + (i - 2) * w, piv[s].values, w,
               label=SYS_SHORT[s].replace("\n", " "),
               color=SYS_COLOR[s], edgecolor="white", linewidth=0.4)

    ax.axhline(0.5, color=CB["vermillion"], lw=0.9, ls="--", zorder=1)
    ax.annotate("chance", xy=(len(STRATA) - 0.45, 0.5), xytext=(0, 3),
                textcoords="offset points", fontsize=7,
                color=CB["vermillion"], ha="right")
    for xi, s in zip(x, STRATA):
        if s in ("LoF", "missense", "silent/non-coding"):
            ax.annotate("0.500\nby construction", xy=(xi - 2 * w, 0.5),
                        xytext=(0, -14), textcoords="offset points",
                        fontsize=6, ha="center", va="top", color=CB["grey"])

    ax.set_ylim(0.42, 1.02)
    ax.set_ylabel("ROC AUC")
    ax.grid(axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{STRAT_LABEL[s]}\nn={base.n[s]:,}" for s in STRATA])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), columnspacing=1.2, handlelength=1.2)
    fig.subplots_adjust(bottom=0.26)
    save(fig, "fig1b_auc_by_stratum")

# --- F2: gene-clustered confidence intervals ------------------------------
def fig2():
    order = ["comp_gain_auc", "comp_gain_auprc", "overall_lora_full",
             "lof_lora_full", "mis_lora_full", "mis_full_probe", "dod"]
    label = {
        "comp_gain_auc":     "composite − rule, AUC",
        "comp_gain_auprc":   "composite − rule, AUPRC",
        "overall_lora_full": "LoRA − full FT, overall AUC",
        "lof_lora_full":     "LoRA − full FT, LoF AUC",
        "mis_lora_full":     "LoRA − full FT, missense AUC",
        "mis_full_probe":    "full FT − probe, missense AUC",
        "dod":               "difference of differences",
    }
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    y = np.arange(len(order))[::-1]
    for yi, k in zip(y, order):
        v = CI[k]
        col = CB["green"] if v["excludes_zero"] else CB["grey"]
        ax.plot([v["lo"], v["hi"]], [yi, yi], color=col, lw=2, solid_capstyle="butt")
        ax.plot([v["point"]], [yi], "o", color=col, ms=5,
                markeredgecolor="white", markeredgewidth=0.6)

    ax.axvline(0, color=CB["black"], lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([label[k] for k in order])
    ax.set_xlabel("difference (95% gene-clustered bootstrap CI, 1,000 resamples)")
    ax.set_title("Only the composite gain excludes zero")
    ax.grid(axis="x")
    ax.set_ylim(-0.8, len(order) - 0.2)

    ax.plot([], [], color=CB["green"], lw=2, label="excludes zero")
    ax.plot([], [], color=CB["grey"], lw=2, label="crosses zero")
    ax.legend(loc="lower right", fontsize=7)
    save(fig, "fig2_clustered_ci")


# --- F3: cost inversion ----------------------------------------------------
def fig3():
    panels = [
        ("Trainable\nparameters", "parameters",
         [("full FT", COST["trainable_full"]), ("LoRA", COST["trainable_lora"])],
         True, "{:.0f}x fewer"),
        ("Training\ntime per step", "ms / step",
         [("full FT", MS.ms_full.mean()), ("LoRA", MS.ms_lora.mean())],
         False, "{:+.1f}%"),
        ("Peak GPU memory\n(training)", "GB",
         [("full FT", GB.gb_full.mean()), ("LoRA", GB.gb_lora.mean())],
         False, "{:+.1f}%"),
        ("Inference\nthroughput", "sequences / s",
         [("full FT", COST["seq_full_t3c"]),
          ("LoRA\nlive", COST["seq_lora_live"]),
          ("LoRA\nmerged", COST["seq_lora_merged"])],
         False, None),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(7.4, 3.4))
    for axi, (title, unit, bars, logy, fmt) in zip(axes, panels):
        labels = [b[0] for b in bars]
        vals = [b[1] for b in bars]
        n = len(bars)
        cols = [SYS_COLOR["S3 full FT"]] + [SYS_COLOR["S4 LoRA r=16"]] * (n - 1)
        x = np.arange(n)
        rects = axi.bar(x, vals, 0.6, color=cols, edgecolor="white", linewidth=0.4)
        if n == 3:
            rects[2].set_hatch("//")
            rects[2].set_edgecolor("white")

        axi.set_xticks(x)
        axi.set_xticklabels(labels, fontsize=7 if n == 3 else 8)
        axi.set_title(title, fontsize=9, pad=10)
        axi.set_ylabel(unit, fontsize=8)
        axi.grid(axis="y")

        if logy:
            axi.set_yscale("log")
            axi.set_ylim(1e6, 5e8)
            note = fmt.format(vals[0] / vals[1])
            for xi, v in zip(x, vals):
                axi.annotate(f"{v/1e6:.1f}M", xy=(xi, v), xytext=(0, 3),
                             textcoords="offset points", fontsize=7, ha="center")
        else:
            axi.set_ylim(0, max(vals) * (1.42 if n == 3 else 1.30))
            for xi, v in zip(x, vals):
                axi.annotate(f"{v:,.0f}" if v > 10 else f"{v:.2f}",
                             xy=(xi, v), xytext=(0, 3),
                             textcoords="offset points", fontsize=7, ha="center")
            if n == 3:
                note = (f"live {100*(vals[1]/vals[0]-1):+.1f}%\n"
                        f"merged {100*(vals[2]/vals[0]-1):+.1f}%")
            else:
                note = fmt.format(100 * (vals[1] / vals[0] - 1))

        axi.annotate(note, xy=(0.5, 0.95), xycoords="axes fraction",
                     ha="center", va="top", fontsize=8, fontweight="bold",
                     color=CB["vermillion"])

    fig.suptitle("LoRA trains 37x fewer parameters and trains no faster", y=0.99)
    fig.text(0.5, 0.005,
             f"training panels: paired grid runs (time n={COST['ms_n']}, "
             f"memory n={COST['gb_n']} matched-step pairs); inference: full test "
             f"set, {META['n']:,} variants, one session, warm-up discarded, "
             f"full FT drift {MERGE['full_drift_pct']:+.1f}%",
             ha="center", fontsize=7, color=CB["grey"])
    fig.subplots_adjust(top=0.78, bottom=0.16, wspace=0.55)
    save(fig, "fig3_cost_inversion")


# --- F4: paired grid outcomes ---------------------------------------------
def fig4():
    NC = dict(zip(GRID_N, [CB["blue"], CB["skyblue"], CB["green"],
                           CB["orange"], CB["vermillion"]]))
    fig, axi = plt.subplots(figsize=(4.6, 4.4))

    lim = (0.60, 0.78)
    axi.plot(lim, lim, color=CB["black"], lw=0.9, zorder=1)
    axi.annotate("LoRA better", xy=(0.615, 0.725), fontsize=7, color=CB["grey"])
    axi.annotate("full FT better", xy=(0.700, 0.615), fontsize=7, color=CB["grey"])

    for N in GRID_N:
        sub = P.xs(N, level="N")
        axi.plot(sub.auc_full, sub.auc_lora, "o", ms=6, color=NC[N],
                 markeredgecolor="white", markeredgewidth=0.6,
                 label=f"N={N:,}", zorder=3)

    axi.set_xlim(*lim)
    axi.set_ylim(*lim)
    axi.set_aspect("equal")
    axi.set_xlabel("full fine-tuning, validation AUC")
    axi.set_ylabel("LoRA r=16, validation AUC")
    axi.set_title("15 paired runs, 5 sizes x 3 seeds")
    axi.grid(axis="both")
    axi.legend(fontsize=7, loc="lower right")

    axi.annotate(
        f"pooled paired difference\n{GRID_STAT['mean']:+.4f} "
        f"(95% CI {GRID_STAT['lo']:+.4f} to {GRID_STAT['hi']:+.4f})",
        xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=7.5)
    save(fig, "fig4_grid_paired")


fig1()
fig1a()
fig1b()
fig2()
fig3()
fig4()

banner("WRITTEN")
for p in WRITTEN:
    print(f"  {p.relative_to(ROOT)}")