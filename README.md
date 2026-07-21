# 🎬 IMDb Transformer Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-DistilBERT-FFD21E?style=flat-square&logo=huggingface)
![Colab](https://img.shields.io/badge/Google%20Colab-GPU-F9AB00?style=flat-square&logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> Fine-tuning a pretrained **DistilBERT** Transformer model on the IMDb dataset for binary sentiment classification — and comparing it against a classical ML baseline.

---

## 📌 Project Overview

This project demonstrates a complete **modern NLP pipeline** using Transformer-based deep learning:

- 🔤 Text tokenization using HuggingFace `DistilBertTokenizer`
- 🧠 Fine-tuning `DistilBertForSequenceClassification` on 10,000 IMDb reviews
- 📊 Full evaluation: Accuracy, Precision, Recall, F1
- ⚖️ Performance comparison with Classical ML (TF-IDF + Logistic Regression)
- 🎯 Real-time inference demo on custom inputs

---

## 📁 Repository Structure

```
imdb-transformer-sentiment/
│
├── Images/
│   ├── 01_class_distribution.png
│   ├── 02_distilbert_architecture.png
│   ├── 03_training_metrics.png
│   ├── 04_confusion_matrix.png
│   ├── 05_model_comparison.png
│   └── 06_sample_predictions.png
│
├── imdb_transformer_sentiment.ipynb
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

| Property | Detail |
|----------|--------|
| Source | IMDb Movie Reviews |
| Total Reviews | 50,000 |
| Sampled | 10,000 (stratified) |
| Classes | Positive (1) / Negative (0) |
| Split | 80% Train / 20% Test |

---

## 🏗️ Model Architecture

**Text → Tokenizer → DistilBERT (6 Transformer Layers) → Classifier Head → Positive / Negative**

| Property | Detail |
|----------|--------|
| Model | `distilbert-base-uncased` |
| Parameters | 66,955,010 |
| Max Token Length | 128 |
| Batch Size | 16 |
| Epochs | 3 |
| Learning Rate | 2e-5 |
| Optimizer | AdamW + Linear Warmup |

---

## 📈 Training Results

![Training Metrics](Images/03_training_metrics.png)

| Epoch | Loss | Accuracy |
|-------|------|----------|
| 1 | 0.4449 | 77.61% |
| 2 | 0.2555 | 89.86% |
| 3 | 0.1615 | 94.44% |

---

## 🎯 Test Set Performance

![Confusion Matrix](Images/04_confusion_matrix.png)

| Metric | Score |
|--------|-------|
| Accuracy | 87.05% |
| Precision | 87.07% |
| Recall | 87.05% |
| F1 Score | 87.05% |

---

## ⚖️ DistilBERT vs Classical ML

![Model Comparison](Images/05_model_comparison.png)

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| TF-IDF + Logistic Regression | 89.00% | 89.00% |
| DistilBERT (Fine-tuned) | 87.05% | 87.05% |

> 💡 Classical ML slightly edges out DistilBERT here due to limited fine-tuning data (10k samples). With the full 50k dataset and more epochs, Transformer models consistently outperform classical approaches.

---

## 🔍 Inference Demo

| Review | Prediction | Confidence |
|--------|-----------|------------|
| "This movie was absolutely amazing..." | Positive 😊 | 99.50% |
| "Worst movie I have ever seen..." | Negative 😞 | 99.69% |
| "It was okay, nothing special..." | Negative 😞 | 93.03% |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| PyTorch 2.x | Deep learning framework |
| HuggingFace Transformers | DistilBERT model & tokenizer |
| scikit-learn | Metrics & train/test split |
| Matplotlib / Seaborn | Visualizations |
| Google Colab (T4 GPU) | Training environment |

---

## 📦 Requirements

```bash
pip install -r requirements.txt
```

| Package | Version |
|---------|---------|
| torch | >=2.0.0 |
| transformers | >=4.30.0 |
| datasets | >=2.12.0 |
| pandas | >=1.5.0 |
| numpy | >=1.23.0 |
| scikit-learn | >=1.2.0 |
| matplotlib | >=3.6.0 |
| seaborn | >=0.12.0 |

---

## 🚀 How to Run

1. Open `imdb_transformer_sentiment.ipynb` in Google Colab
2. Set runtime to **T4 GPU** (Runtime → Change runtime type)
3. Mount your Google Drive and place `IMDB_Dataset.csv` in root
4. Run all cells in order

---

## 👤 Author

<div align="center">
<img src="https://avatars.githubusercontent.com/Junaid-Ahmed-Rupok" width="100" style="border-radius:50%"/>

### Sarder Junaid Ahmed
**Data Scientist & Machine Learning Engineer**

*Transforming complex data into strategic decisions through rigorous statistical modeling and production-ready machine learning systems.*

[![GitHub](https://img.shields.io/badge/GitHub-Junaid--Ahmed--Rupok-181717?logo=github)](https://github.com/Junaid-Ahmed-Rupok)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sarder%20Junaid%20Ahmed-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sarder-junaid-ahmed-059b68240/)
[![Portfolio](https://img.shields.io/badge/Portfolio-junaid--ahmed--rupok.github.io-1E88E5?logo=githubpages&logoColor=white)](https://junaid-ahmed-rupok.github.io/__portfolio__Yes/)
[![Email](https://img.shields.io/badge/Email-junaidahmedrupok%40gmail.com-EA4335?logo=gmail&logoColor=white)](mailto:junaidahmedrupok@gmail.com)
</div>

**Specializations:** Statistical ML · Causal Inference · Trustworthy AI · Fairness-Aware ML · RAG Systems

**Selected Research:**
- 📄 **Ahmed, S.J.** et al. (2026). *Machine Learning for Crime Classification: A Fairness-Aware Approach to Class Imbalance.* Journal of Machine Learning and Applications, 2(1), 9–17. [DOI: 10.61577/jmla.2026.100002](https://doi.org/10.61577/jmla.2026.100002)
- 📄 **Ahmed, S.J.** et al. (2026). *Machine Learning for Crime Classification: A Fairness-Aware Approach to Class Imbalance.* IEEE SPICSCON 2026, BAUET, Bangladesh (Aug 13–14, 2026). **Accepted for Presentation** — IEEE Xplore.
- 📄 **Ahmed, S.J.** et al. (2026). *CF-EGAT: A Causal Fairness-Aware Equity Graph Attention Network for Country-Level Environmental Livability Classification.* SPECTRA 2026. 🏆 **1st Best Paper Award**
- 📄 **Ahmed, S.J.** (2025). *Multi-Dimensional Statistical Similarity for Governance Classification: Beyond Arbitrary Thresholds.* APMEE 2025. 🏆 **Best Research Paper Award**
- 📄 **Ahmed, S.J.** (2026). *DeepEnMap: Ordinal-Aware Multi-Modal Deep Learning for Energy Poverty Risk Mapping.* IEMIS 2026, University of British Columbia, Vancouver, Canada (Aug 10–12, 2026). **Accepted for Presentation** — Springer LNNS Series (Scopus, EI-Compendex, DBLP, ISI Proceedings).
- 📄 **Ahmed, S.J.** (2026). *Density-Decoupled, Mask-Ablated Segmentation-Guided Diffusion for Controllable Mammography Synthesis: A Preliminary Study.* IEMIS 2026, University of British Columbia, Vancouver, Canada (Aug 10–12, 2026). **Accepted for Presentation** — Springer LNNS Series (Scopus, EI-Compendex, DBLP, ISI Proceedings).
- 📄 **Ahmed, S.J.**, Islam Nahian, M.T., & Kwoshik, M.H.R. (2026). *Environmental Livability Assessment via Adaptive Bootstrap-Retrained SHAP and Statistically-Constrained Pareto Counterfactuals: A Cross-National Analysis.* IEEE SPICSCON 2026, BAUET, Bangladesh (Aug 13–14, 2026). **Accepted for Presentation** — IEEE Xplore.
- 📄 **Ahmed, S.J.** (2026). *DemocracyGuard: Testing a Divergence-Index Reconciliation of Subjective and Objective Democracy Indicators for Forecasting Adverse Regime Transitions.* **Under Review**, Transactions on Machine Learning Research (TMLR) — Q1, Top-Tier Journal.
- 📄 **Ahmed, S.J.** (2026). *FAI: Feature-Wise Adaptive Imputation via Downstream-Aware Method Selection.* **Under Review**, ICISET 2026 (IEEE Xplore).

**Other Deployed Projects:**
- 🔬 [ReproHub](https://reproapp-8jb7vbhnqyltxq23bsr8xn.streamlit.app/) — Automated research reproducibility platform with composite scoring across 11 statistical tests
- 📊 [StatsPro](https://statistical-analysis-app-7axetqtx75ncuu7fr8irxj.streamlit.app/) — AI-powered statistical analysis platform with automated CSV-to-report workflows

**Honors:**
🏆 1st Best Paper — SPECTRA 2026 &nbsp;·&nbsp;
🏆 Best Research Paper — APMEE 2025 &nbsp;·&nbsp;
🎖️ Esteemed Alumni Award — YLRL RUET 2024 &nbsp;·&nbsp;
⭐ Perfect GPA 5.00/5.00 — SSC & HSC &nbsp;·&nbsp;
🎓 National Merit Scholarship — 2009 & 2013

---

## 📄 License

This project is licensed under the MIT License.
