\# Financial News Sentiment Classifier



\## Problem Statement



Classify financial news headlines and sentences as \*\*positive\*\*, \*\*negative\*\*, or \*\*neutral\*\* to assess market sentiment. This project compares a classical baseline approach (TF-IDF + Logistic Regression) with a modern zero-shot transformer-based approach (HuggingFace BART), providing confusion matrices, metrics, and detailed error analysis.



\## Dataset



\*\*Source:\*\* \[Sentiment Analysis for Financial News (Kaggle)](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)



\- \*\*Format:\*\* CSV with columns `Sentence` (financial news headline/article text) and `Sentiment` (positive/negative/neutral)

\- \*\*Size:\*\* ~4,800+ samples (varies with download)

\- \*\*Classes:\*\* 3 (positive, negative, neutral)



The model automatically downloads the data from an online mirror during training. If the download fails, sample data is used for demonstration.



\## Design \& Architecture



\### Models



1\. \*\*Baseline Model: TF-IDF + Logistic Regression\*\*

&nbsp;  - Vectorizer: TF-IDF with max 500 features and bigrams

&nbsp;  - Classifier: L2-regularized Logistic Regression

&nbsp;  - Training time: ~1-2 seconds

&nbsp;  - Pros: Fast, interpretable, low computational cost

&nbsp;  - Cons: Limited contextual understanding



2\. \*\*Zero-Shot Model: HuggingFace BART (facebook/bart-large-mnli)\*\*

&nbsp;  - Architecture: Transformer-based zero-shot classifier

&nbsp;  - Labels: \["positive", "negative", "neutral"]

&nbsp;  - Training time: None (pre-trained)

&nbsp;  - Pros: Strong contextual understanding, no fine-tuning needed

&nbsp;  - Cons: Slower inference, requires GPU for practical speed



\### Evaluation Metrics



\- \*\*Accuracy:\*\* Overall correctness

\- \*\*Precision:\*\* True positives / (true positives + false positives) per class

\- \*\*Recall:\*\* True positives / (true positives + false negatives) per class

\- \*\*F1-Score:\*\* Harmonic mean of precision and recall

\- \*\*Confusion Matrix:\*\* Detailed classification breakdown



\### Error Analysis



For each model, we analyze:

\- Total misclassifications and overall error rate

\- Error rate per class (which classes are harder to predict?)

\- Sample misclassifications (examples of wrong predictions with true/predicted labels)



\## Assumptions \& Limitations



\### Assumptions

\- Financial news can be accurately classified into three sentiment buckets

\- Single-sentence or headline-level input is sufficient for sentiment classification

\- Pre-trained models (BART) generalize well to financial domain without fine-tuning



\### Limitations

\- \*\*Class imbalance:\*\* Financial news datasets often have imbalanced sentiment distributions

\- \*\*Domain mismatch:\*\* General-purpose transformers may not capture domain-specific financial terminology

\- \*\*Short context:\*\* Single-sentence inputs lack paragraph-level context sometimes needed for nuanced sentiment

\- \*\*Label noise:\*\* Manual annotations may have subjective disagreements

\- \*\*Cold-start:\*\* Baseline model requires training data; zero-shot model has no domain adaptation





\## Setup \& Installation



\### Prerequisites

\- Python 3.8+

\- pip or conda

\- ~2 GB disk space (for transformer models)

\- GPU (optional, but recommended for zero-shot inference)



\### Install Dependencies



```bash

pip install -r requirements.txt

```



\### (Optional) Using Conda



```bash

conda create -n sentiment-classifier python=3.10

conda activate sentiment-classifier

pip install -r requirements.txt

```



\## Usage



\### 1. Training \& Evaluation (Recommended First Run)



Train the baseline model and evaluate both models on test data:



```bash

python app.py --mode train --train-split 0.8

```



\*\*Options:\*\*

\- `--train-split 0.8` (default): 80% train, 20% test

\- `--sample-data`: Use built-in sample data instead of downloading from Kaggle

\- `--mode train`: (default) Train and evaluate



\*\*Output:\*\*

\- Prints accuracy, precision, recall, F1 for both models

\- Confusion matrices for each model

\- Error analysis (misclassified samples)

\- Saves baseline model to `models/baseline.pkl`

\- Saves results summary to `results/evaluation\_results.json`



\### 2. Running FastAPI Service



After training, start a FastAPI server for real-time predictions:



```bash

python app.py --mode serve --port 8000

```



\*\*API Endpoints:\*\*



\#### Health Check

```bash

curl http://localhost:8000/health

```



\#### Predict Sentiment

```bash

\# Using baseline model

curl -X POST http://localhost:8000/predict \\

&nbsp; -H "Content-Type: application/json" \\

&nbsp; -d '{"text": "Strong earnings beat expectations", "model": "baseline"}'



\# Using zero-shot model

curl -X POST http://localhost:8000/predict \\

&nbsp; -H "Content-Type: application/json" \\

&nbsp; -d '{"text": "Strong earnings beat expectations", "model": "zero-shot"}'

```



\*\*Response:\*\*

```json

{

&nbsp; "text": "Strong earnings beat expectations",

&nbsp; "model": "baseline",

&nbsp; "prediction": "positive",

&nbsp; "probabilities": {

&nbsp;   "positive": 0.92,

&nbsp;   "negative": 0.05,

&nbsp;   "neutral": 0.03

&nbsp; }

}

```



\### 3. Using Sample Data (No Download Required)



For quick testing without downloading the full dataset:



```bash

python app.py --mode train --sample-data

```



\## Evaluation Notes



\### Metrics



After training on the full dataset, expect:



| Model | Accuracy | Precision | Recall | F1-Score |

|-------|----------|-----------|--------|----------|

| Baseline (TF-IDF + LR) | ~0.72 | ~0.72 | ~0.72 | ~0.71 |

| Zero-Shot (BART) | ~0.75 | ~0.75 | ~0.74 | ~0.74 |



\*(Actual results depend on data split and class distribution)\*



\### Key Findings



1\. \*\*Baseline Model\*\* performs decently for a classical approach; benefits from simple interpretable features

2\. \*\*Zero-Shot Model\*\* typically outperforms baseline by ~3-5% due to contextual understanding

3\. \*\*Error Analysis\*\* often reveals:

&nbsp;  - Neutral class is hardest to distinguish (overlaps with weak positive/negative signals)

&nbsp;  - Baseline struggles with domain-specific vocabulary

&nbsp;  - Zero-shot sometimes misses sarcasm or implicit sentiment



\### Testing Guardrails



\- Models validate input text is non-empty

\- API returns 400 error if model not found

\- Confusion matrices align predictions and labels automatically

\- Error analysis handles all class combinations



\## Reproducibility



\### Random Seeds

Set in `BaselineModel` and across sklearn operations for deterministic results.



\### Data Reproducibility

\- Script auto-downloads from stable Kaggle mirror

\- Alternatively, download manually and modify `download\_financial\_sentiment\_data()`

\- Sample data is hardcoded for full reproducibility without network



\### Git Tracking

To initialize version control:



```bash

git init

git add .

git commit -m "Initial commit: baseline + zero-shot sentiment classifier"

```



\## Dependencies



See \[requirements.txt](requirements.txt) for exact versions. Key libraries:



\- \*\*pandas\*\*: Data handling

\- \*\*scikit-learn\*\*: Baseline model (TF-IDF, LogisticRegression, metrics)

\- \*\*transformers\*\*: HuggingFace zero-shot pipeline

\- \*\*torch\*\*: Deep learning backend

\- \*\*fastapi\*\*: REST API framework

\- \*\*uvicorn\*\*: ASGI server



\## Future Work \& Extensions



1\. \*\*Fine-tuning:\*\* Train transformers on financial data for domain adaptation

2\. \*\*Ensemble:\*\* Combine baseline + zero-shot via voting or stacking

3\. \*\*Multi-label:\*\* Support mixed sentiments (e.g., "good news but high cost")

4\. \*\*Explainability:\*\* Add SHAP/LIME for feature importance

5\. \*\*Web UI:\*\* Build frontend dashboard for interactive predictions

6\. \*\*Real-time monitoring:\*\* Track model drift over time



\## Author \& Attribution



\*\*Project:\*\* Financial News Sentiment Classifier  

\*\*Data Source:\*\* \[Sentiment Analysis for Financial News – Kaggle](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)  

\*\*Stack:\*\* Python, scikit-learn, HuggingFace Transformers, FastAPI



\## License



This project is provided as-is for educational purposes. See Kaggle dataset license for data usage terms.



---



\*\*Questions or Issues?\*\* Refer to the inline code comments in `app.py` or adjust hyperparameters and re-train.

