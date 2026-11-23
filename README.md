# Query-Efficient Black-Box Knowledge Distillation from Pure Noise

**Student-side PGA with softmax-space diversity and student-guided query selection, on CIFAR-100**

ATDL course project, Vidyashilp University.

| Name | Roll number | Email |
|:---|:---|:---|
| Suneel | 2023UG000150 | 2023suneel@vidyashilp.edu.in |
| Shaik Sameer Roshan | 2024UG100001 | 2023shaiksameer.roshan@vidyashilp.edu.in |
| S Venkata Prasanna Bhargava | 2023UG000129 | 2023venkataprasanna.bhargava@vidyashilp.edu.in |
| Rajesh | 2023UG000184 | 2023rajesh.s@vidyashilp.edu.in |

📄 **Final report:** [report/report.pdf](report/report.pdf). The LaTeX source is in [report/latex/](report/latex/).

## Problem

A ResNet-34 **teacher** (82.18% on CIFAR-100) is a black box: we can only send it images and read its answer. The answer is either its full probability vector (*soft labels*) or only its class (*hard labels*). A ResNet-18 **student** must learn from it under these rules:

- **No real images.** The teacher is shown **only random noise**.
- **At most 250,000 teacher queries.** Queries to our own students are free.
- **PGA on the student.** Synthetic images are made by Projected Gradient Ascent on the student, with a KL-diversity term that keeps images of the same class far apart in the student's softmax space. **These images are never sent to the teacher.**

## Method (v2)

| Stage | What happens | Teacher queries |
|:---|:---|:---:|
| A | 50K noise images (DIP-style priors + RandAugment) are labelled by the teacher; 3 students are trained | 50K |
| B | In 4 rounds, the students score 400K free noise images each. The 50K most useful per round go to the teacher, chosen by class quota (label deficit) and score = z(log p̄_c) + 0.5·z(JS disagreement) | 200K |
| C | 3 fresh students are trained on all 250K, with class-balanced weights | 0 |
| D | PGA on the student ensemble: J = ln s_c + λ·SKL(r_i, r_j) + λ·min KL(s_new ‖ s_old). Images are kept only if all students agree with confidence ≥ 0.8 | 0 |
| E | The best student is fine-tuned on teacher labels plus ensemble labels (PGA images and 250K extra noise) | 0 |

## Main results (CIFAR-100 test accuracy, 250K teacher queries, 3 seeds)

| Method | Soft labels | Hard labels |
|:---|:---:|:---:|
| Baseline (1 student, random noise) | 35.0 ± 0.6 | 24.5 ± 0.2 |
| Professor's method (+ student-side PGA with diversity) | 36.1 | 25.7 |
| **v2: final single student** | **41.4 ± 0.6** | **30.5 ± 1.0** |
| **v2: ensemble of 3 students** | **43.0 ± 0.2** | **31.3 ± 0.6** |
| v2 with 500K queries | 44.5 | 34.0 |

- **Budget:** v2 with 250K queries beats the baseline with 500K queries (soft 41.4 vs. 37.9, hard 30.5 vs. 27.3).
- **Ablations:**
  - class-balanced, disagreement-based selection is the key component;
  - PGA and its diversity term change accuracy by less than one standard deviation once selection is used;
  - CutMix hurts by about 5 points.
- **Diversity:** the diversity term spreads same-class PGA images apart (mean pairwise symmetric KL 0.75 → 3.47).

See the report for all analyses.

## Repository structure

```
query_kd/                 code
  data.py                 CIFAR-100 loading, GPU RandAugment, crop/flip
  noise.py                noise priors (Gaussian, uniform, pink, Perlin, DIP-style)
  models.py               CIFAR-style ResNet-18/34
  oracle.py               black-box teacher oracle with a hard query budget
  trainer.py              KD training (KL / CE, class weights, mixup/CutMix option), evaluation
  pga_ascent.py           student-side PGA with softmax-space KL diversity
  run_multi_v2.py         full pipeline (stages A-E), crash-safe and resumable
  train_teacher.py        teacher training with a validation split
  pretrain_simclr.py      contrastive pre-training on noise (ablation)
  summarize_v2.py         results table over runs
experiments/              args.json, results.json and log.txt of every run (40 runs)
  teacher_val/            teacher training log (checkpoint chosen on validation)
  v2_kd/, v3_kd/          one folder per run, plus summary.md
report/                   report (PDF, Markdown, LaTeX), figures, scripts that rebuild them
```

Model checkpoints (`.pth`) and saved image sets (`.pt`) are not included because of their size (~36 GB). Every run can be reproduced from the commands below.

## Setup

```bash
pip install -r requirements.txt
# CIFAR-100 (python version), unpacked into ./cifar-100-python
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz && tar xzf cifar-100-python.tar.gz
```

A CUDA GPU is strongly recommended (we used one RTX 4000 Ada, 20 GB).

## Reproducing the experiments

**1. Teacher** (ResNet-34, 45K train / 5K validation, about 1.5 h):

```bash
python query_kd/train_teacher.py --data_dir cifar-100-python --arch resnet34 --val_size 5000 \
       --out experiments/teacher_val/teacher_resnet34_val.pth
```

**2. Any run of the study.** Every run uses these common arguments:

```bash
python query_kd/run_multi_v2.py --data_dir cifar-100-python \
       --teacher_ckpt experiments/teacher_val/teacher_resnet34_val.pth --teacher_arch resnet34 \
       --out_dir experiments/<v2_kd or v3_kd> --run_name <run> <arguments from the table below>
```

To reproduce run `sim_*`, first run `python query_kd/pretrain_simclr.py --out experiments/v3_kd/simclr_pre/simclr_noise.pth --n_images 300000 --epochs 25 --refresh 5`.

Runs that use `--reuse_from` need the referenced run to have finished first. They reuse its teacher-labelled data, so they make no new teacher queries.

Each stage is saved to disk, so an interrupted run resumes when the same command is started again.

| Run | Purpose | Final acc. (%) | Arguments (besides the common ones) |
|:---|:---|:---:|:---|
| `b100k_hard` | budget 100K | 21.40 | `--label_mode hard --seed_size 20000 --total_budget 100000 --pool_size 640000 --seed_epochs 100 --student_epochs 80` |
| `b500k_hard` | budget 500K | 33.95 | `--label_mode hard --seed_size 100000 --total_budget 500000 --pool_size 3200000` |
| `b50k_hard` | budget 50K | 14.99 | `--label_mode hard --seed_size 10000 --total_budget 50000 --pool_size 320000 --seed_epochs 100 --student_epochs 100` |
| `base500k_hard` | baseline, budget 500K | 27.34 | `--label_mode hard --n_students 1 --seed_size 500000 --total_budget 500000 --pool_size 1600000 --balance_power 0.0 --pga_images 0 --extra_noise 0 --seed_epochs 1 --final_epochs 0` |
| `base_hard` | baseline: 1 student, random noise | 24.35 | `--label_mode hard --n_students 1 --seed_size 250000 --pool_size 1600000 --balance_power 0.0 --pga_images 0 --extra_noise 0 --seed_epochs 1 --final_epochs 0` |
| `base_seed3_hard` | baseline, seed 3 | 24.61 | `--label_mode hard --n_students 1 --seed_size 250000 --pool_size 1600000 --balance_power 0.0 --pga_images 0 --extra_noise 0 --seed_epochs 1 --final_epochs 0 --seed 3` |
| `lam1_hard` | ablation: λ = 1.0 | 31.23 | `--label_mode hard --pool_size 1600000 --lam_intra 1.0 --lam_old 1.0 --reuse_from v2_hard` |
| `mix_hard` | ablation: + CutMix | 25.82 | `--label_mode hard --pool_size 1600000 --reuse_from v2_hard --reuse_stages AB --mix cutmix` |
| `nobal_hard` | ablation: no class balancing | 25.55 | `--label_mode hard --seed_size 250000 --pool_size 1600000 --balance_power 0.0 --seed_epochs 1` |
| `nodiv_hard` | ablation: PGA without diversity (λ=0) | 30.65 | `--label_mode hard --pool_size 1600000 --lam_intra 0.0 --lam_old 0.0 --reuse_from v2_hard` |
| `nopga_hard` | ablation: no PGA | 31.50 | `--label_mode hard --pool_size 1600000 --reuse_from v2_hard --pga_images 0` |
| `one_hard` | ablation: 1 student | 29.58 | `--label_mode hard --n_students 1 --pool_size 1600000` |
| `primer_hard` | ablation: softened hard labels | 30.29 | `--label_mode hard --pool_size 1600000 --reuse_from v2_hard --hard_soften 0.5` |
| `prof_hard` | professor's method (1 student + PGA) | 25.71 | `--label_mode hard --n_students 1 --seed_size 250000 --pool_size 1600000 --reuse_from base_hard --balance_power 0.0 --extra_noise 0 --seed_epochs 1` |
| `seed2_hard` | main method, seed 2 | 29.37 | `--label_mode hard --pool_size 1600000 --seed 2` |
| `seed3_hard` | main method, seed 3 | 31.07 | `--label_mode hard --pool_size 1600000 --seed 3` |
| `sim_hard` | ablation: + SimCLR init | 29.95 | `--label_mode hard --pool_size 1600000 --reuse_from v2_hard --reuse_stages AB --init_ckpt experiments/v3_kd/simclr_pre/simclr_noise.pth` |
| `v2_hard` | main method (seed 1) | 31.04 | `--label_mode hard --pool_size 1600000` |
| `v3_hard` | v3 (= v2, no add-on qualified) | 30.41 | `--label_mode hard --pool_size 1600000` |
| `v3b100k_hard` | v3 at 100K | 22.35 | `--label_mode hard --seed_size 20000 --total_budget 100000 --pool_size 640000 --seed_epochs 100 --student_epochs 80` |
| `b100k_soft` | budget 100K | 37.02 | `--seed_size 20000 --total_budget 100000 --pool_size 640000 --seed_epochs 100 --student_epochs 80` |
| `b500k_soft` | budget 500K | 44.53 | `--seed_size 100000 --total_budget 500000 --pool_size 3200000` |
| `b50k_soft` | budget 50K | 31.14 | `--seed_size 10000 --total_budget 50000 --pool_size 320000 --seed_epochs 100 --student_epochs 100` |
| `base500k_soft` | baseline, budget 500K | 37.90 | `--n_students 1 --seed_size 500000 --total_budget 500000 --pool_size 1600000 --balance_power 0.0 --pga_images 0 --extra_noise 0 --seed_epochs 1 --final_epochs 0` |
| `base_seed3_soft` | baseline, seed 3 | 34.60 | `--n_students 1 --seed_size 250000 --pool_size 1600000 --balance_power 0.0 --pga_images 0 --extra_noise 0 --seed_epochs 1 --final_epochs 0 --seed 3` |
| `base_soft` | baseline: 1 student, random noise | 35.39 | `--n_students 1 --seed_size 250000 --pool_size 1600000 --balance_power 0.0 --pga_images 0 --extra_noise 0 --seed_epochs 1 --final_epochs 0` |
| `lam1_soft` | ablation: λ = 1.0 | 41.85 | `--pool_size 1600000 --lam_intra 1.0 --lam_old 1.0 --reuse_from v2_soft` |
| `mix_soft` | ablation: + CutMix | 36.56 | `--pool_size 1600000 --reuse_from v2_soft --reuse_stages AB --mix cutmix` |
| `nobal_soft` | ablation: no class balancing | 34.96 | `--seed_size 250000 --pool_size 1600000 --balance_power 0.0 --seed_epochs 1` |
| `nodiv_soft` | ablation: PGA without diversity (λ=0) | 42.69 | `--pool_size 1600000 --lam_intra 0.0 --lam_old 0.0 --reuse_from v2_soft` |
| `nopga_soft` | ablation: no PGA | 41.66 | `--pool_size 1600000 --reuse_from v2_soft --pga_images 0` |
| `one_soft` | ablation: 1 student | 38.93 | `--n_students 1 --pool_size 1600000` |
| `pga100k_soft` | ablation: 100K PGA images | 42.63 | `--pool_size 1600000 --reuse_from v2_soft --pga_images 100000` |
| `prof_soft` | professor's method (1 student + PGA) | 36.10 | `--n_students 1 --seed_size 250000 --pool_size 1600000 --reuse_from base_soft --balance_power 0.0 --extra_noise 0 --seed_epochs 1` |
| `seed2_soft` | main method, seed 2 | 41.41 | `--pool_size 1600000 --seed 2` |
| `seed3_soft` | main method, seed 3 | 40.89 | `--pool_size 1600000 --seed 3` |
| `sim_soft` | ablation: + SimCLR init | 42.01 | `--pool_size 1600000 --reuse_from v2_soft --reuse_stages AB --init_ckpt experiments/v3_kd/simclr_pre/simclr_noise.pth` |
| `v2_soft` | main method (seed 1) | 42.03 | `--pool_size 1600000` |
| `v3_soft` | v3 (= v2, no add-on qualified) | 40.79 | `--pool_size 1600000` |
| `v3b100k_soft` | v3 at 100K | 36.62 | `--seed_size 20000 --total_budget 100000 --pool_size 640000 --seed_epochs 100 --student_epochs 80` |

**3. Report figures and PDF** (need the checkpoints produced above):

```bash
python report/make_figures.py      # figures + numbers from the saved runs
python report/build.py             # report.pdf (pandoc + Chrome) and report/latex/report.tex
```
