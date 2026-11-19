# Sentiment Analysis on Amazon Reviews

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Natural Language Processing (NLP) project that performs **sentiment analysis and star rating prediction** on customer reviews from the **Amazon Fine Food Reviews** dataset (1999–2012). This project follows the structured 4-sprint roadmap, encompassing exploratory data analysis, text preprocessing, multi-model benchmarking, hyperparameter tuning, model explainability, and deployment via an interactive Streamlit web application.

---

## 📁 Project Directory Structure

```text
NLP_PROJECT/
│
├── Sentiment Analysis on Amazon Reviews.pdf   # Project requirements & roadmap
├── Reviews.csv                                # Raw Amazon reviews dataset (~568k reviews)
├── cleaned_reviews.csv                        # Cleaned & preprocessed dataset (~390k reviews)
├── best_model.joblib                          # Serialized champion model (Linear SVM + TF-IDF)
│
├── Text Data Modelling/                       # Required project folder structure
│   ├── EDA and Feature Engineering.ipynb      # Sprint 1 notebook: Cleaning, Preprocessing & EDA
│   └── Model_Building and Evaluation.ipynb    # Sprint 2 & 3 notebook: Model Training & Tuning
│
├── app.py                                     # Sprint 4: Interactive Streamlit web application
├── requirements.txt                           # Project dependencies
└── README.md                                  # Complete project documentation
```

---

## 📊 Dataset Overview

- **Source**: Amazon Fine Food Reviews (Kaggle / Stanford SNAP)
- **Timeframe**: October 1999 – October 2012 (>10 years)
- **Total Raw Reviews**: 568,454
- **Total Unique Users**: 256,059
- **Total Unique Products**: 74,258
- **Target Variable**:
  - `Sentiment`: Multi-class sentiment category:
    - **Positive**: Ratings 4 and 5 (78.0% of data)
    - **Neutral**: Rating 3 (7.8% of data)
    - **Negative**: Ratings 1 and 2 (14.2% of data)
  - `Score`: Exact customer star rating (1–5)

---

## 🚀 Sprint Roadmap & Deliverables

### Sprint 1 – Exploratory Data Analysis & Text Preprocessing
*Delivered in `Text Data Modelling/EDA and Feature Engineering.ipynb` & `cleaned_reviews.csv`*

1. **Data Cleaning**:
   - **Missing Values**: Handled null values in reviewer profile names and summaries.
   - **Duplicate Removal**: Identified and pruned duplicate reviews sharing identical `(UserId, ProfileName, Time, ReviewText)` combinations (reduced from ~568k to ~393k).
   - **Helpfulness Validation**: Filtered out physically impossible records where `HelpfulnessNumerator > HelpfulnessDenominator`.
   - **Timestamp Conversion**: Parsed Unix epoch timestamps into standard datetime formats, extracting `Year`, `Month`, and `DayOfWeek`.

2. **Text Preprocessing Pipeline**:
   - HTML entity decoding and tag stripping (`<br />`, etc.).
   - URL removal (`http://...`, `www...`).
   - Contraction expansion (`won't` $\rightarrow$ `will not`, `can't` $\rightarrow$ `can not`, etc.).
   - Negation preservation: Carefully retained `not`, `no`, and `nor` in stopwords to preserve sentiment-flipping phrases.
   - Two-pass lemmatization via NLTK `WordNetLemmatizer` (verb and noun reduction).
   - Case normalization and tokenization.

3. **Exploratory Data Analysis (EDA)**:
   - Rating distributions and rating percentages (highlighting severe 78% positive skew).
   - Top frequent unigrams and bigrams for positive (4–5★) vs negative (1–2★) reviews.
   - Word clouds for positive vs negative reviews.
   - Helpfulness ratio distribution and rating-vs-helpfulness relationship.
   - Review length distributions by rating (negative reviews tend to be longer and more descriptive).
   - Top reviewed products and most active reviewers.
   - Temporal trends: Review volume expansion and sentiment stability over a decade.
   - **Observations documented for every single visualization.**

---

### Sprint 2 – Machine Learning Model Development
*Delivered in `Text Data Modelling/Model_Building and Evaluation.ipynb`*

1. **Feature Engineering**:
   - **Bag of Words (CountVectorizer)**: Unigrams + Bigrams (`ngram_range=(1, 2)`), 5,000 max features.
   - **TF-IDF Vectorizer**: Unigrams + Bigrams (`ngram_range=(1, 2)`), sublinear TF scaling, 5,000 max features.

2. **Algorithms Trained & Evaluated (12 Configurations)**:
   - **Naive Bayes** (`MultinomialNB`)
   - **Logistic Regression** (`LogisticRegression(class_weight="balanced")`)
   - **Support Vector Machine** (`LinearSVC(class_weight="balanced")`)
   - **Decision Tree** (`DecisionTreeClassifier`)
   - **Random Forest** (`RandomForestClassifier`)
   - **K-Nearest Neighbors** (`KNeighborsClassifier`)

3. **Evaluation Metrics**:
   - **Accuracy**, **Precision (macro)**, **Recall (macro)**, **F1-Score (macro)**.
   - Full confusion matrices and per-class classification reports.
   - **Why Macro-F1?** Due to the 78% positive class imbalance, raw Accuracy is misleading (a naive dummy classifier achieves 78% accuracy by predicting only Positive). Macro-F1 fairly penalizes models that fail on the minority Neutral (8%) and Negative (14%) classes.

---

### Sprint 3 – Hyperparameter Tuning & Model Selection
*Delivered in `Text Data Modelling/Model_Building and Evaluation.ipynb` & `best_model.joblib`*

1. **Hyperparameter Optimization**:
   - Conducted 3-fold cross-validation `GridSearchCV` on Logistic Regression regularization parameter $C \in [0.1, 0.5, 1.0, 5.0, 10.0]$ optimizing for `f1_macro`.
   - Compared baseline vs tuned configurations.

2. **Benchmark Summary**:

| Rank | Model | Vectorizer | Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **Linear SVM** | **TF-IDF** | **82.66%** | **0.6176** | **0.6434** | **0.6288** |
| 🥈 2 | Logistic Regression (tuned) | TF-IDF | 77.63% | 0.5981 | 0.6548 | 0.6190 |
| 🥉 3 | Logistic Regression | TF-IDF | 77.88% | 0.5962 | 0.6517 | 0.6171 |
| 4 | Naive Bayes | BoW | 80.40% | 0.5980 | 0.6111 | 0.6037 |
| 5 | Linear SVM | BoW | 81.52% | 0.5990 | 0.6166 | 0.6071 |
| 6 | Random Forest | BoW | 83.52% | 0.6385 | 0.5478 | 0.5739 |
| 7 | Random Forest | TF-IDF | 83.35% | 0.6322 | 0.5284 | 0.5443 |
| 8 | KNN | BoW | 72.84% | 0.5376 | 0.5057 | 0.5186 |
| 9 | KNN | TF-IDF | 78.43% | 0.5843 | 0.4851 | 0.5097 |
| 10 | Decision Tree | TF-IDF | 71.98% | 0.4687 | 0.4578 | 0.4629 |
| 11 | Naive Bayes | TF-IDF | 82.52% | 0.6974 | 0.4411 | 0.4655 |
| 12 | Decision Tree | BoW | 71.53% | 0.4554 | 0.4447 | 0.4497 |

3. **Champion Model Selection**:
   - **Linear SVM + TF-IDF** is the definitive champion. It achieves the best balance between high accuracy (82.7%) and the highest macro-F1 (0.629).
   - Linear models efficiently exploit the sparse 5,000-dimensional TF-IDF feature space in seconds, whereas KNN and Tree ensembles struggle with dimensionality and minority class recall.

4. **Model Interpretability (Top Predictive N-Grams)**:
   - **Strongest Negative Features**: `disappoint`, `not recommend`, `terrible`, `not`, `awful`, `worst`, `nasty`, `not worth`.
   - **Strongest Neutral Features**: `ok`, `not bad`, `not favorite`, `however`, `average`, `okay`, `unfortunately`.
   - **Strongest Positive Features**: `best`, `great`, `love`, `delicious`, `not disappoint`, `perfect`, `excellent`, `wonderful`.

5. **Model Serialization**:
   - The champion pipeline was serialized and exported to `best_model.joblib`.

---

### Sprint 4 – Interactive Streamlit Web Application
*Delivered in `app.py`*

The Streamlit web application offers a clean interface for live sentiment analysis:
- **Customer Review Text Area**: Multi-line review input with live character and word counters.
- **Predict Button**: Real-time inference triggering text normalization and model scoring.
- **Predicted Rating Display**:
  - Star visualization: ⭐⭐⭐⭐⭐ (1 to 5 stars).
  - Continuous calibrated score estimation (e.g., `4.82 / 5.00`) with visual progress gauge.
- **Predicted Sentiment Category**:
  - Color-coded badges: 🟢 Positive, 🟡 Neutral, 🔴 Negative.
- **Class Confidence Distribution**:
  - Multi-class breakdown displaying percentage confidence for Negative, Neutral, and Positive classes using softmax calibrated decision margins.
  - Interactive distribution bar chart.
- **Explainability & Pipeline Breakdown**:
  - Live inspection drawer showing raw text vs. cleaned/lemmatized tokens.
  - Detected keyword features influencing prediction.
- **Preset Quick-Test Buttons**:
  - One-click testing for 5★ Ecstatic, 3★ Mixed, and 1★ Terrible reviews.

---

## 💻 Installation & Usage

### 1. Environment Setup
Clone or navigate to the repository directory:
```bash
cd /Users/charanteja/Desktop/NLP_PROJECT
```

Install required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Exploring the Notebooks
Launch Jupyter Notebook or JupyterLab:
```bash
jupyter lab
```
Open and execute:
- `Text Data Modelling/EDA and Feature Engineering.ipynb`
- `Text Data Modelling/Model_Building and Evaluation.ipynb`

### 3. Launching the Streamlit Web Application
Start the interactive application locally:
```bash
streamlit run app.py
```
Open your browser at the displayed local URL (default: `http://localhost:8501`).

---

## 📈 Key Insights & Conclusions

1. **Linear Models Excel on Sparse Text**: Linear SVM and Logistic Regression with TF-IDF significantly outperformed tree ensembles and distance-based classifiers (KNN) both in predictive accuracy and computational speed.
2. **Class Imbalance Matters**: Positive reviews represent 78% of all feedback. Balancing class weights was crucial to prevent models from ignoring negative and neutral sentiments.
3. **Bigrams and Negations are Critical**: Keeping negative stopword tokens (`not`, `no`) and extracting bigrams allowed the model to accurately differentiate between *"recommend"* and *"not recommend"*, or *"disappoint"* and *"not disappoint"*.
4. **The Neutral Class Challenge**: 3-star reviews represent mixed feedback (praise combined with reservations), making them the hardest to classify cleanly. Binary sentiment modeling (Positive vs. Negative) yields even higher classification metrics (>92% F1).

---
*Created as part of the Innomatics Research Labs Data Science & NLP Internship Assessment.*
