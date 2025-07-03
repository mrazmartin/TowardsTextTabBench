## 🚀 Running the Code

This project provides a modular pipeline for:

- Downloading datasets
- Generating text embeddings
- Performing feature and row downsampling
- Training and evaluating machine learning models

---

### 📦 Environment Setup

We recommend using **Conda** to manage dependencies.

```bash
# 1. Create and activate a new conda environment
conda create -n t4t python=3.10 -y
conda activate t4t

# 2. Install dependencies
pip install -r requirements.txt
```

---

### 🏃 Running the Pipeline

Run the full pipeline using the module syntax:

```bash
python -u -m pipelines.main --dataset it_salary --download_datasets --generate_embeddings --run_pipe
```

#### ✅ Example: TabPFN Evaluation with Specific Downsampling

```bash
python -u -m pipelines.main \
    --dataset it_salary \
    --download_datasets \
    --generate_embeddings \
    --run_pipe \
    --eval_method tabpfn \
    --downsample_methods pca shap
```

---

### ⚙️ Command-Line Arguments

| Argument               | Description                                                                 | Default                |
|------------------------|-----------------------------------------------------------------------------|------------------------|
| `--dataset`            | Dataset name or `"all"`                                                     | `it_salary`            |
| `--embed_methods`      | Embedding methods to use (`fasttext`, `skrub`, `ag`, etc.)                  | `fasttext skrub ag`    |
| `--save_format`        | Format for saving embeddings (`npy` or `pkl`)                               | `npy`                  |
| `--project_root`       | Optional root directory for the project                                     | `None`                 |
| `--download_datasets`  | Run dataset preprocessing notebooks (downloads data)                        | `False`                |
| `--generate_embeddings`| Generate embeddings for textual columns                                     | `False`                |
| `--run_pipe`           | Run the full pipeline (load, embed, downsample, evaluate)                   | `False`                |
| `--eval_method`        | Model to use for evaluation: `xgb` or `tabpfn`                              | `tabpfn`               |
| `--downsample_methods` | Feature selection strategies (`pca`, `shap`, `anova`, etc.)                 | All listed in script   |
| `--no_text`            | Drop textual columns after embedding                                        | `False`                |

---

Let me know if you need help with datasets, results interpretation, or extending the model list!
