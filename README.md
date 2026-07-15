# cancer-variant-peft
Does parameter-efficient fine-tuning match full fine-tuning of a genomic language model for cancer variant pathogenicity classification in the low-data regime? A controlled comparison of full FT vs. LoRA vs. QLoRA on DNABERT-2.
# Parameter-Efficient Fine-Tuning of Genomic Language Models for Cancer Variant Classification

A controlled comparison of **full fine-tuning vs. LoRA vs. QLoRA** on [DNABERT-2](https://huggingface.co/zhihan1996/DNABERT-2-117M) for binary pathogenic/benign classification of single-nucleotide variants in cancer-associated genes.

> **Research question.** Does parameter-efficient fine-tuning (PEFT) match full fine-tuning of a genomic language model for cancer-variant classification in the low-data regime?
>
> **Secondary (optional).** Do genomic LMs gain their advantage specifically on non-coding / regulatory cancer variants?

Course project — Medical Informatics, PhD programme, University of Belgrade.

---

## Motivation

Genomic language models are increasingly applied to variant effect prediction, but fine-tuning them is compute-hungry. PEFT methods (LoRA, QLoRA) promise near-equivalent performance at a fraction of the trainable parameters — a claim that is well studied for natural-language models but less so for genomic LMs, and especially not in the **low-data regime** that characterises curated cancer-variant sets, where the number of high-confidence labelled examples is small.

This project tests that claim under controlled conditions: same model, same data, same splits, same evaluation — varying only the fine-tuning strategy and the training-set size.

## Method

Four training regimes, compared across shrinking training-set sizes:

| Regime | Description |
|---|---|
| **Zero-shot baseline** | Frozen backbone, no fine-tuning |
| **Full fine-tuning** | All ~117M parameters updated |
| **LoRA** | Low-rank adapters, backbone frozen |
| **QLoRA** | LoRA on a 4-bit quantised backbone |

**Metrics.** Primary: **Matthews Correlation Coefficient (MCC)** — chosen because it is robust to the class imbalance inherent in ClinVar. Also reported: F1, ROC-AUC, AUPRC. Each run additionally logs **trainable parameters**, **peak VRAM**, and **wall-clock time**, so accuracy is read against its true cost.

**Leakage control.** Train/validation/test splits are **gene/locus-disjoint** — no gene contributes variants to more than one split. Random variant-level splitting would leak, because variants in the same gene share sequence context and the model could memorise the locus rather than learn variant effect.

## Data

| Source | Role | Version used |
|---|---|---|
| [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) (GRCh38 VCF) | Pathogenicity labels | 2026-07-06 |
| [Ensembl GRCh38](https://www.ensembl.org/) primary assembly | Reference sequence for variant windows | Release 116 |
| [COSMIC Cancer Gene Census](https://cancer.sanger.ac.uk/census) | Cancer-gene filter | v104 (Tier 1 + Tier 2, 768 genes) |

**Filtering recipe**

- **Review status** ≥ 2 stars via `CLNREVSTAT` ∈ {`criteria_provided,_multiple_submitters,_no_conflicts`, `reviewed_by_expert_panel`, `practice_guideline`}
- **Label** from `CLNSIG` — positive: `Pathogenic`, `Likely_pathogenic`, `Pathogenic/Likely_pathogenic`; negative: `Benign`, `Likely_benign`, `Benign/Likely_benign`. Variants of uncertain significance (VUS) and conflicting classifications are dropped.
- **SNVs only** via `CLNVC=single_nucleotide_variant`, enforcing `len(REF) == len(ALT) == 1` over `{A,C,G,T}`
- **Cancer-gene filter** joined on **Entrez GeneId** from `GENEINFO` — *not* gene symbol, which is lossy across synonyms. Validated join overlap: **762/765 (99.6%)** of CGC genes have ClinVar entries.
- **Contigs** restricted to `1–22, X, Y, MT` (ClinVar contains NCBI-style scaffolds absent from the Ensembl primary assembly)
- `MC` (molecular consequence) retained to support the coding vs. non-coding secondary analysis

### Data licensing — read before cloning

**No data is committed to this repository.**

- **COSMIC CGC** is free for academic/non-commercial use *with registration*, but its licence **prohibits redistribution**. You must register at [cancer.sanger.ac.uk](https://cancer.sanger.ac.uk) and download the census yourself.
- **OncoKB is deliberately excluded** from this project: its licence forbids training ML models. The cancer-gene list is used only to *filter* public ClinVar variants.
- ClinVar and the Ensembl reference are freely redistributable but are multi-GB and therefore fetched, not vendored.

Setup scripts fetch ClinVar and the reference genome automatically; the CGC requires a manual, registered download.

## Models

- **[DNABERT-2-117M](https://huggingface.co/zhihan1996/DNABERT-2-117M)** — primary. Pinned to revision `7bce263b15377fc15361f52cfab88f8b586abda0`.
- **Nucleotide Transformer v2 (500M)** — optional stretch goal, not a committed deliverable.

> **Why pin the revision?** DNABERT-2 ships custom modelling code loaded via `trust_remote_code`, fetched live from the Hub at load time. Without a pin, an upstream change would silently alter the model mid-project and break the controlled comparison.

## Environment

Runs entirely on **free-tier cloud notebooks** (Kaggle P100/T4, ~16 GB VRAM) — no local GPU required. CPU sessions handle all data preparation to conserve GPU quota.

```bash
pip install "transformers==4.43.4" "bitsandbytes==0.43.1" einops pysam biopython scikit-learn
# peft 0.19.1
```

### Non-obvious environment constraints

These are load-bearing. Ignoring them produces silent failures rather than clean errors.

- **`transformers` must be pinned to 4.43.4.** DNABERT-2's vendored modelling code targets the 4.28–4.29 era. Modern releases (5.x) break it outright; the ~4.45 `GenerationMixin`/`PreTrainedModel` refactor also breaks it. 4.43.4 is the validated window — recent enough for Python 3.12 wheels and 4-bit support, old enough to load the remote code.
- **Triton must be absent.** DNABERT-2 imports a Triton FlashAttention kernel that fails to compile on non-A100 GPUs. With Triton uninstalled the model falls back to eager PyTorch attention — correct maths, slightly lower throughput, irrelevant at 117M parameters. The warning `Unable to import Triton; defaulting ... to pytorch` is the **intended** state.
- **`Some weights ... newly initialized: bert.pooler.dense`** is expected and harmless — classification uses a custom head, not the checkpoint's pooler.
- **Coordinate conversion:** VCF `POS` is 1-based; `pysam.fetch()` is 0-based half-open. Off-by-one here corrupts every extracted window silently. Validated against ClinVar `1:66926 REF=AG` → FASTA returns `AG`.
- **Kaggle auto-decompresses top-level `.gz` files** and mangles **bgzip** archives, yielding a header-only stub. Upload ClinVar decompressed, or wrapped in a `.zip`.
- Kaggle pip installs do **not** survive session restarts; files in `/kaggle/working/` do.

## Project structure

```
.
├── notebooks/
│   ├── 01_environment_setup.ipynb     # pinned deps, DNABERT-2 load + forward-pass check
│   ├── 02_data_acquisition.ipynb      # fetch + verify ClinVar, GRCh38, CGC
│   ├── 03_preprocessing.ipynb         # filter, window extraction, gene-disjoint splits
│   ├── 04_finetune_full.ipynb
│   ├── 05_finetune_lora.ipynb
│   ├── 06_finetune_qlora.ipynb
│   └── 07_evaluation.ipynb            # MCC / F1 / ROC-AUC / AUPRC, cost curves
├── src/
│   ├── data/                          # filtering + window extraction
│   ├── models/                        # backbone + classification head
│   ├── training/                      # train loops, PEFT configs, checkpointing
│   └── eval/                          # metrics, plots
├── configs/
├── results/
├── requirements.txt
└── README.md
```

## Roadmap

- [x] **Phase 1** — Environment & data acquisition (pinned stack validated; DNABERT-2 loads; raw sources verified and persisted)
- [ ] **Phase 2** — Preprocessing: filter to cancer-gene SNVs, extract ref/alt windows, build gene-disjoint splits
- [ ] **Phase 3** — Full fine-tuning baseline
- [ ] **Phase 4** — LoRA and QLoRA
- [ ] **Phase 5** — Low-data sweep across shrinking training-set sizes
- [ ] **Phase 6** — Evaluation, analysis, writeup
- [ ] *Stretch* — Nucleotide Transformer v2 (500M); non-coding/regulatory variant analysis

## Expected findings & honest caveats

Stated up front, because they shape how the results should be read:

- **QLoRA's memory advantage is close to meaningless at 117M parameters.** Full fine-tuning of a 117M encoder fits comfortably in 16 GB, so 4-bit quantisation saves little and peak-VRAM will look nearly flat across regimes. The comparison remains valid as an **accuracy-parity** study framed around MCC and trainable-parameter count. VRAM only becomes an interesting axis at the 500M stretch goal — which is precisely what motivates that extension.
- **PEFT may well *not* match full fine-tuning here**, and that is a legitimate result rather than a failure. Published benchmarks report DNABERT-2 dropping meaningfully in MCC under LoRA relative to full fine-tuning — a larger gap than for bigger genomic LMs. A clean negative answer to the primary research question is a finding, not a null result.
- **ClinVar is imbalanced and ascertainment-biased.** Labels reflect what has been submitted and curated, not a random sample of variant space. MCC is used precisely because it does not flatter a classifier that exploits imbalance.

## Acknowledgements

- DNABERT-2 — Zhou et al.
- ClinVar — NCBI
- Cancer Gene Census — Wellcome Sanger Institute
- Reference genome — Ensembl / EBI

## License

Code is released under the MIT License. **This licence covers the code only** — see *Data licensing* above; the datasets carry their own terms, and COSMIC data may not be redistributed.
