# Computer Vision A

**MLC Africa × Harvard CS197**: group repository for the Computer Vision A track of the Machine Learning Collective Africa programme.

This repo holds all of our work for the course. Each piece of work gets its own date-prefixed folder, so the history reads in order.

## Projects

| Date | Project | Summary | Headline result |
|---|---|---|---|
| 2026-11-13 | [SigLIP zero-shot vs frozen-backbone linear probe](2026-11-13_siglip-tomato-linear-probe/) | Tomato maturity (Immature / Mature) with `google/siglip-base-patch16-224`: zero-shot text-prompt baseline vs a 1,538-parameter linear head on the frozen vision encoder | **78.6% → 99.3%** test accuracy (+20.7 pts) |

## Repository layout

```
COMPUTER_VISION_A/
├── README.md                          ← you are here
├── .gitignore
└── YYYY-MM-DD_<short-project-name>/   ← one folder per project
    ├── README.md                      ← task, data, method, results
    ├── requirements.txt
    ├── <notebook>.ipynb               ← runnable top to bottom on Colab
    ├── slides.html                    ← presentation (open in a browser)
    └── outputs/                       ← figures, metrics (results.json), small checkpoints
```

## Conventions

- **Folder names:** `YYYY-MM-DD_kebab-case-name`, dated by the session the work is presented at.
- **Every project is self-contained.** It has its own README, `requirements.txt` and outputs, and it runs on Google Colab (T4 GPU) with no local setup.
- **Notebooks are committed with their outputs**, so results are readable on GitHub without re-running.
- **Data is never committed.** Notebooks download datasets at runtime (e.g. via `kagglehub`).
- **Results are reproducible:** fixed seeds, held-out test sets touched only once, and metrics saved to `outputs/results.json`.

## Running a project

1. Open the project's notebook in Colab (each project README has an "Open in Colab" badge).
2. **Runtime → Change runtime type → T4 GPU**, then **Run all**.
3. If a dataset needs Kaggle credentials, add `KAGGLE_USERNAME` and `KAGGLE_KEY` in Colab's 🔑 Secrets panel.
