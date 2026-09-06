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
        "seq_full": RATE["full"], "seq_lora": RATE["lora"]}
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
T2 = (OV.reset_index()[["system", "auc", "mcc", "auprc", "f1"]]
        .rename(columns={"system": "System", "auc": "AUC", "mcc": "MCC",
                         "auprc": "AUPRC", "f1": "F1"}))
T2["System"] = T2.System.astype(str)
print("\n  Table 2 — test set, all systems, overall stratum "
      f"({int(OV.n.iloc[0]):,} variants, {int(OV.pos.iloc[0]):,} pathogenic)")
print(write_table(T2, "table2_test_overall",
                  {"AUC": "{:.4f}", "MCC": "{:.4f}",
                   "AUPRC": "{:.4f}", "F1": "{:.4f}"}))

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
    n_by = MX.groupby("stratum", observed=True).n.first().loc[STRATA]

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    x = np.arange(len(STRATA))
    w = 0.15
    for i, s in enumerate(SYSTEMS):
        ax.bar(x + (i - 2) * w, piv[s].values, w, label=SYS_SHORT[s].replace("\n", " "),
               color=SYS_COLOR[s], edgecolor="white", linewidth=0.4)

    ax.axhline(0.5, color=CB["vermillion"], lw=0.9, ls="--", zorder=1)
    ax.annotate("chance", xy=(len(STRATA) - 0.45, 0.5), xytext=(0, 3),
                textcoords="offset points", fontsize=7, color=CB["vermillion"],
                ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{STRAT_LABEL[s]}\nn={n_by[s]:,}"
                        if s != "overall" else STRAT_LABEL[s] for s in STRATA])
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("ROC AUC")
    ax.set_title("Test-set discrimination by consequence stratum")
    ax.grid(axis="y")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              columnspacing=1.2, handlelength=1.2)
    save(fig, "fig1_auc_by_stratum")


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


fig1()
fig2()

banner("WRITTEN")
for p in WRITTEN:
    print(f"  {p.relative_to(ROOT)}")