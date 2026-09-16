# Fine-Tuning a Genomic Language Model for Cancer Variant Classification

Binary pathogenic/benign classification of single-nucleotide variants in cancer-associated
genes with [DNABERT-2-117M](https://huggingface.co/zhihan1996/DNABERT-2-117M), and a
controlled comparison of **full fine-tuning against LoRA** under a fixed low-compute budget.

Course project — Medical Informatics, PhD programme, University of Belgrade.

> **Primary question.** Does parameter-efficient fine-tuning match full fine-tuning of a
> genomic language model for cancer-variant classification in the low-data regime?
>
> **Answer.** Yes, within ±0.02 AUC — but the comparison turned out to be the least
> interesting result in the project.

---

## Headline results

All figures below are on a **sealed, gene-disjoint test set** of 48,328 variants across 40
genes, scored once with thresholds frozen on validation and never re-tuned.

| System | AUC | MCC | AUPRC |
|---|---|---|---|
| Consequence rule (no model, no sequence) | **0.9700** | 0.8372 | 0.9223 |
| Frozen probe (no fine-tuning) | 0.8090 | 0.5183 | 0.6431 |
| Full fine-tuning | 0.8359 | 0.6713 | 0.7649 |
| LoRA r=16 | 0.8594 | 0.6678 | 0.7781 |
| Composite (rule + model) | **0.9859** | — | 0.9651 |

**A three-tier rule reading only the predicted consequence class beats every fine-tuned
model.** This is expected rather than embarrassing: under the ACMG/AMP guideline, consequence
class is formal evidence curators use when assigning the ClinVar labels the models train on,
so the label is not independent of the annotation. Any benchmark reporting pooled AUC on
ClinVar measures that dependence as much as it measures the model.

**The model contributes where annotation cannot.** Composing the two, so the model only
reorders variants *within* a consequence tier, gains **+0.0159 AUC (95% CI +0.0110..+0.0202,
gene-clustered)** — the only quantity in the study that survives clustered inference.

**PEFT matches full fine-tuning and saves nothing at this scale.** Pooled paired difference
across 15 cells: −0.0045 ± 0.0301 (95% CI −0.0211..+0.0122). LoRA trains 37× fewer parameters
and is **9.01% slower per step** with **0.04 GB** less peak memory out of ~9 GB. Its 24%
inference penalty disappears entirely once adapters are merged (`merge_and_unload()`).

**Fine-tuning fails silently about half the time.** Full FT finished at or below its own
frozen probe in 6 of 15 grid cells, LoRA in 8 of 15 — at every training-set size including the
largest. A broken run is not identifiable without a frozen-probe reference to compare against.

---

## Method

### Siamese difference architecture

The obvious design — concatenate reference and alternate as `[CLS] ref [SEP] alt [SEP]` —
**fails completely**, reaching 0.5552 validation AUC. An ablation replacing the alternate
allele with a second copy of the reference changed predictions by 0.016 on average
(correlation 0.995): the model was classifying the locus and ignoring the substitution.

Two properties of DNABERT-2 explain it. Its ALiBi positional scheme penalises attention in
proportion to distance, and the comparison required spans ~100 tokens. And its segment
embedding table has an effectively untrained second row (L2 0.043 against 1.60), because it
was pretrained on single sequences — so segment identifiers had to be zeroed, leaving `[SEP]`
as the only boundary marker.

The adopted design encodes each allele separately through a shared backbone, mask-mean-pools
each to 768 dimensions, and classifies `h_alt − h_ref`. Locus content is common to both
vectors and cancels; a model ignoring the substitution would receive a zero vector. Backbone
output is cast to fp32 **before** pooling, since subtracting two nearly identical fp16 vectors
leaves mostly rounding error.

Head: `LayerNorm(768) → Linear(768,256) → GELU → Dropout(0.2) → Linear(256,2)`, 198,914
parameters. Every run uses **LP-FT** — the head is converged on cached frozen embeddings
before the backbone is unfrozen — which resolves the feature-distortion failure mode.

### Systems evaluated

| | Description |
|---|---|
| **S1 consequence rule** | tier 2 = LoF, 1 = missense, 0 = silent/non-coding. No model |
| **S2 frozen probe** | head at step 0 over a frozen backbone. *Not* zero-shot |
| **S3 full fine-tuning** | 116,676,866 trainable parameters |
| **S4 LoRA r=16** | 3,148,034 trainable (37× fewer); `Wqkv, dense, gated_layers, wo`; α=32 |
| **S5 composite** | `2·tier + p_full` — a lexicographic tie-break, unfitted |

DNABERT-2 has no classification head, so **no zero-shot baseline exists** for this task; the
frozen probe substitutes but is trained.

### Evaluation

Primary metric **MCC**, with ROC-AUC, AUPRC, F1, precision and recall. Thresholds are selected
on validation by maximising MCC and **frozen** for the test set. Results are reported overall,
within each consequence stratum, and on a held-out subset of 1,216 variants ClinVar later
reclassified from VUS.

**Splits are gene-disjoint** — no gene contributes variants to more than one split — which
removes both forms of circularity described by Grimm et al. (2015). It does not control for
homology between genes in different splits.

**Test intervals come from a gene-clustered bootstrap** over the 40 test genes, 1,000
resamples. A naive variant-level bootstrap gives intervals **2.6× too narrow**.

---

## Data

| Source | Role | Version |
|---|---|---|
| [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) (GRCh38 VCF) | Pathogenicity labels | 2026-07-06 |
| [Ensembl GRCh38](https://www.ensembl.org/) primary assembly | Reference windows | Release 116 |
| [COSMIC Cancer Gene Census](https://cancer.sanger.ac.uk/census) | Cancer-gene filter | v104, Tier 1 + 2 |

**Final dataset:** 92,019 variants across 629 genes, 1,024 bp windows centred on the variant.

| Split | Genes | Variants | Pathogenic | Prevalence |
|---|---|---|---|---|
| train | 434 | 37,241 | 5,309 | 14.3% |
| validation | 155 | 6,450 | 1,125 | 17.4% |
| test (sealed) | 40 | 48,328 | 9,284 | 19.2% |

Test genes were selected as those with the most ClinVar reclassifications, so the test split is
**not a random sample of cancer genes** — it is enriched for the most heavily curated ones.
Validation was rebalanced at whole-gene granularity after the initial split (7.3% → 17.4%
prevalence) so that MCC-maximising thresholds would transfer; the test split was left
byte-identical.

### Filtering recipe

- **Review status** ≥ 2 stars via `CLNREVSTAT` ∈ {`criteria_provided,_multiple_submitters,_no_conflicts`, `reviewed_by_expert_panel`, `practice_guideline`}
- **Label** from `CLNSIG`; VUS and conflicting classifications dropped
- **SNVs only**, enforcing `len(REF) == len(ALT) == 1` over `{A,C,G,T}`
- **Cancer-gene filter** joined on **Entrez GeneId** from `GENEINFO`, not gene symbol
- **Contigs** restricted to `1–22, X, Y, MT`
- `MC` retained; strata assigned by Sequence Ontology term — LoF = SO:0001587/0001575/0001574, missense = SO:0001583/0001582, else silent/non-coding

### Data licensing — read before cloning

**No source data is committed to this repository.**

- **COSMIC CGC** is free for academic use with registration, but its licence **prohibits
  redistribution**. Register at [cancer.sanger.ac.uk](https://cancer.sanger.ac.uk) and download
  the census yourself.
- **OncoKB is deliberately excluded**: its licence forbids training ML models. The cancer-gene
  list only *filters* public ClinVar variants.
- ClinVar and the Ensembl reference are freely redistributable but multi-GB, so they are
  fetched rather than vendored.

---

## Environment

Runs entirely on **free-tier Kaggle notebooks** (T4, ~16 GB VRAM). No local GPU. CPU sessions
handle all data preparation to conserve GPU quota.

```bash
pip install "transformers==4.43.4" einops pysam biopython scikit-learn
# peft 0.19.1 · torch 2.10.0+cu128 · Python 3.12.13
```

### Non-obvious constraints

These are load-bearing. Ignoring them produces silent failures rather than clean errors.

- **`transformers` pinned to 4.43.4.** DNABERT-2's vendored modelling code targets the
  4.28–4.29 era; later releases break it. 4.43.4 is the validated window.
- **Triton must be absent.** Its presence selects a FlashAttention path that fails to compile
  on non-A100 GPUs. Uninstalled, the model falls back to eager attention — correct maths,
  lower throughput. **All timing figures in this repository are eager-attention figures.**
  The warning `Unable to import Triton; defaulting ... to pytorch` is the *intended* state.
- **`bert.pooler.dense` newly initialised** is expected — classification uses a custom head and
  never touches `[CLS]`.
- **Model revision pinned** to `7bce263b15377fc15361f52cfab88f8b586abda0`. DNABERT-2 loads
  custom code from the Hub at runtime; without a pin an upstream change would silently alter
  the model mid-project.
- **Coordinate conversion:** VCF `POS` is 1-based, `pysam.fetch()` is 0-based half-open. An
  off-by-one corrupts every window silently.
- **Kaggle auto-decompresses top-level `.gz`** and mangles bgzip archives. Upload ClinVar
  decompressed or zipped.

---

## Repository structure

```
.
├── notebooks/
│   ├── 01_environment_setup      # pinned stack, DNABERT-2 load + forward-pass check
│   ├── 02_data_acquisition       # fetch and verify ClinVar, GRCh38, CGC
│   ├── 03_preprocessing          # filter, windows, gene-disjoint splits
│   ├── 03b_val_rebalance         # whole-gene validation rebalance; test left untouched
│   ├── 04_representation_check   # BPE behaviour, segment embeddings, token lengths
│   ├── 05_frozen_probe           # cached embeddings, pooling sweep, concat failure + ablation
│   ├── 06_siamese_finetune       # Siamese difference architecture
│   ├── 07_peft_probe             # LoRA configuration checks
│   ├── 08_lora_grid              # early grid
│   ├── 09_lpft_grid_runner       # 5 sizes x 3 seeds x 2 arms, plus full-data runs
│   ├── 10_section_d_analysis     # validation analysis, strata, composite
│   └── 11-test-scoring           # sealed test scoring, robustness, inference cost
├── figures/
│   ├── make_figures.py           # regenerates every figure and table, with assertions
│   ├── fig1a_prevalence_by_stratum.png
│   ├── fig1b_auc_by_stratum.png
│   ├── fig2_clustered_ci.png
│   ├── fig3_cost_inversion.png
│   ├── fig4_grid_paired.png
│   └── table1..table4 (.csv / .md)
├── results/
│   ├── runs/<run_id>/            # metrics.json + validation logits per run
│   ├── test_matrix.csv           # 25 rows: 5 systems x 5 strata
│   ├── test_robustness.json      # clustered CIs, per-gene, decomposition
│   ├── test_logits.npz, test_thresholds.json, test_meta.json, test_oracle.json
│   ├── inference_merge.json      # merged vs unmerged LoRA throughput
│   └── d4_strata.csv, d5_verified.csv
└── data/
    ├── dataset_card_v2.json
    └── gene_splits_v2.csv
```

`figures/make_figures.py` rebuilds every figure and table from `results/` and exits non-zero if
any cross-check between files fails.

> **Note:** the script also requires `results/emb_val_1024bp.npz` for the validation table,
> which is not committed here because of its size.

---

## Status

Complete. All experiments, analysis and test scoring are finished; the report is written.

**Out of scope, deliberately:** QLoRA and Nucleotide Transformer v2 (500M) were scoped for
timeline reasons and not attempted. No published variant-effect predictor (CADD, REVEL,
AlphaMissense) was evaluated as a comparator, which is the most useful single addition anyone
extending this should make.

## Known limitations

- The label's dependence on consequence annotation cannot be removed by any splitting protocol
- Full-data results rest on a **single seed per arm**; grid seed variation exceeds the
  between-arm difference
- Gene-disjoint splitting does not control for **homology** between genes in different splits
- The LoF test stratum has only 24 negatives; numbers reported, nothing inferred
- Timing is comparable **within a session only**; absolute rates vary across Kaggle allocations
- No system is usable as a clinical screen at its MCC-tuned operating point — recall is poor by
  construction
- The secondary question (regulatory variants) is **unanswered**; the estimable non-coding
  stratum is predominantly intronic and synonymous, most plausibly acting through splicing

## Acknowledgements

DNABERT-2 — Zhou et al. · ClinVar — NCBI · Cancer Gene Census — Wellcome Sanger Institute ·
Reference genome — Ensembl / EBI

## License

Code is released under the MIT License. **This covers the code only** — see *Data licensing*
above; COSMIC data may not be redistributed.
