# PRD — LLM-Assisted Mood Detection from Keystroke Dynamics

## 1. Project Overview

**Project:** LLM-Assisted Mood Detection from Keystroke Dynamics  
**Type:** College-level Machine Learning project  
**Primary focus:** R02 + R06  
**Dataset:** 100,000-event synthetic keystroke dataset  
**Core ML:** Random Forest / XGBoost  
**LLM role:** Feature-grounded explanation and contextual reasoning

The project aims to investigate whether privacy-preserving keystroke behavior can be used to estimate affective state while avoiding collection of the user's actual typed content.

The project will use synthetic data because participant data is not currently available. The synthetic dataset is intended to validate the complete ML pipeline—data cleaning, EDA, feature extraction, personalization, prediction, and explanation—not to claim real-world emotion-recognition accuracy.

---

## 2. Problem Statement

Typing behavior contains measurable temporal patterns such as typing speed, dwell time, flight time, pauses, and correction behavior. These patterns may change with a person's affective state.

Existing work demonstrates that keyboard/touch interaction features can be associated with stress, valence, arousal, and emotion, but collecting human-labeled affective data requires participants and repeated self-reports.

This project therefore develops a privacy-preserving, ML-based pipeline that:

1. Works only with keystroke timing/behavioral metadata.
2. Does not store actual typed characters.
3. Extracts meaningful behavioral features.
4. Builds a personal typing baseline.
5. Measures deviations from that baseline.
6. Predicts an affective state.
7. Uses an LLM to produce a human-readable explanation based only on structured features.

---

## 3. Research Objectives

### R02 — Privacy-Preserving Keystroke Representation

Represent keyboard behavior using timing and behavioral characteristics without storing actual characters.

Core features include:

- Dwell time
- Flight time
- Typing speed
- Pause duration/frequency
- Correction rate
- Variability of timing

The research mapping also proposes digraph latency, burst structure and entropy as possible extensions.

### R06 — LLM-Assisted Personalized Mood Inference

Combine:

- Current keystroke features
- Personal typing baseline
- Deviation from baseline
- Recent temporal context

and provide the structured information to an LLM for affective-state reasoning and explanation.

---

## 4. Scope

### Included

- Synthetic raw keystroke event generation
- Data cleaning
- Exploratory Data Analysis
- Feature extraction
- 30-second window aggregation
- Personal baseline calculation
- Baseline deviation features
- Simple temporal trend features
- Classical ML classification/regression
- User-independent evaluation
- LLM-assisted explanation
- Simple dashboard/reporting

### Not Included

To keep the project suitable for a college ML course:

- No Transformer trained from scratch
- No custom LSTM/1D-CNN required
- No real-time production keyboard logger
- No medical/clinical diagnosis
- No claim that synthetic accuracy represents real human emotion accuracy
- No storage of typed text
- No unnecessarily complex cloud architecture

---

# 5. Dataset Requirements

## 5.1 Dataset Size

Target:

**100,000 raw keystroke events**

Structure:

- 50 synthetic users
- 20 sessions per user
- 100 events per session
- Total = 100,000 events

The dataset should contain user-level variability so that every synthetic user has a different typing baseline.

## 5.2 Raw Dataset Schema

| Column | Description |
|---|---|
| user_id | Synthetic user identifier |
| session_id | Session identifier |
| event_index | Event order |
| timestamp | Event timestamp |
| key_event | Abstract key category/event |
| dwell_time_ms | Key hold duration |
| flight_time_ms | Interval between successive key events |
| is_correction | Whether the event represents a correction |
| valence | Synthetic continuous valence label |
| arousal | Synthetic continuous arousal label |
| mood_state | Synthetic affective-state category |

Actual typed sentences/characters must not be stored.

## 5.3 Synthetic Data Generation

The generator should model realistic variation rather than producing identical distributions for every user.

Each synthetic user receives individual baseline parameters such as:

- Average typing speed
- Average dwell time
- Average flight time
- Timing variability
- Correction tendency

Affective states modify these behavioral distributions.

Initial states:

- Calm
- Positive
- Stressed
- Low

The generator also introduces:

- Random noise
- Occasional long pauses
- Missing timing values
- Invalid timing values
- Outliers

This makes the dataset suitable for demonstrating the complete data-science workflow.

### Important limitation

The synthetic mood labels are generated assumptions. They are **not human ground truth**.

Therefore, model performance on this dataset must be described as:

> "Performance on the synthetic benchmark"

and not as:

> "Real-world emotion detection accuracy."

Synthetic keystroke generation itself is supported by prior research using statistical models to generate realistic keystroke timing data. citeturn0search2

---

# 6. Data Processing Pipeline

```text
Synthetic Raw Keystrokes
          ↓
Data Inspection
          ↓
Cleaning
          ↓
EDA
          ↓
30-Second Windowing
          ↓
Feature Extraction
          ↓
Personal Baseline
          ↓
Baseline Deviation
          ↓
Temporal Features
          ↓
ML Model
          ↓
Prediction
          ↓
LLM Explanation
```

---

# 7. Data Cleaning

The cleaning stage should demonstrate standard ML preprocessing.

### Tasks

1. Check duplicate rows.
2. Check missing values.
3. Detect negative timing values.
4. Detect unrealistic timing values.
5. Handle missing dwell/flight values.
6. Check timestamp ordering.
7. Verify session boundaries.
8. Check class distribution.
9. Verify user-level sample counts.

### Example rules

- Negative dwell time → invalid → replace/remove.
- Missing flight time → impute using appropriate session/user statistics.
- Extreme pauses → either retain as meaningful pauses or cap only after EDA.
- Duplicate events → remove where appropriate.

The cleaning decisions should be justified using EDA rather than arbitrary deletion.

---

# 8. Exploratory Data Analysis

EDA should be a visible part of the project.

### Required plots

1. Dwell-time distribution
2. Flight-time distribution
3. Typing-speed distribution
4. Pause-duration distribution
5. Correction-rate distribution
6. Mood-state distribution
7. User-level typing-speed comparison
8. Correlation heatmap
9. Feature distributions by mood
10. Valence vs arousal scatter plot

### Questions to investigate

- Are timing distributions skewed?
- Do users have noticeably different typing baselines?
- Which features vary most between synthetic affective states?
- Are valence and arousal correlated?
- Are there outliers?
- Does correction behavior differ between states?

---

# 9. Feature Engineering

Raw events will be converted into fixed time windows.

### Window

Initial choice:

**30 seconds per window**

Each window becomes one ML sample.

### Core features

#### Typing behavior

- `typing_speed`
- `mean_dwell`
- `std_dwell`
- `mean_flight`
- `std_flight`

#### Pauses

- `mean_pause`
- `pause_count`
- `long_pause_count`

#### Corrections

- `correction_rate`

### Optional extensions

If time permits:

- Median dwell
- Dwell coefficient of variation
- Flight coefficient of variation
- Digraph latency
- Burst length
- Keystroke entropy

The first implementation should remain limited to the core features.

---

# 10. Personal Baseline

A major component of R06 is personalization.

Instead of assuming that the same typing speed means the same emotional state for everyone, calculate a baseline for each user.

For each user:

```text
baseline_mean(feature)
baseline_std(feature)
```

For a new window:

```text
z = (current_value - baseline_mean) / baseline_std
```

Example:

```text
User baseline typing speed = 4.2 keys/sec
Current typing speed       = 3.2 keys/sec

Deviation = -1.0 keys/sec
```

The model can therefore learn whether the user is behaving differently from their normal pattern.

---

# 11. Temporal Context

Keep temporal modelling simple.

For each current window, calculate recent trends using previous windows.

Examples:

- Change in typing speed
- Change in dwell time
- Change in flight time
- Change in pause frequency
- Change in correction rate

Possible features:

```text
typing_speed_change
dwell_change
flight_change
pause_change
correction_change
```

A simple rolling mean or slope over the previous 3–5 windows is sufficient.

No LSTM/Transformer is required.

---

# 12. Machine Learning

## Primary models

### Model 1 — Random Forest

Used as a simple interpretable baseline.

### Model 2 — XGBoost

Primary ML model because it handles nonlinear feature relationships well and works effectively with tabular data.

### Optional Model 3 — SVM

Can be used as an additional comparison if time permits.

---

# 13. Prediction Target

Two approaches can be explored.

## Option A — Mood Classification

Classes:

```text
Calm
Positive
Stressed
Low
```

Metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Option B — Valence/Arousal Regression

Predict:

```text
Valence: -1 to +1
Arousal: -1 to +1
```

Metrics:

- MAE
- RMSE
- R²

### Recommended

Use **mood classification as the main ML demonstration**, while retaining valence/arousal as the underlying affective representation.

---

# 14. Train/Test Strategy

A normal random row split can leak information from the same user into both training and testing.

Therefore, the primary evaluation should be **user-independent**.

Recommended:

**Group-based train/test split by `user_id`.**

Example:

```text
40 users → training
10 users → testing
```

Optional advanced evaluation:

**Leave-One-Subject/User-Out validation**

This tests whether the model can generalize to unseen users.

---

# 15. LLM Component

The LLM should not simply receive raw keystrokes.

It receives a structured representation such as:

```json
{
  "current": {
    "typing_speed": 3.1,
    "mean_dwell": 124,
    "mean_flight": 168,
    "correction_rate": 0.08
  },
  "baseline_deviation": {
    "typing_speed_z": -1.4,
    "dwell_z": 0.9,
    "flight_z": 1.2,
    "correction_rate_z": 1.5
  },
  "recent_trend": {
    "typing_speed": "decreasing",
    "pause_frequency": "increasing",
    "correction_rate": "increasing"
  },
  "ml_prediction": {
    "state": "stressed",
    "confidence": 0.81
  }
}
```

The LLM returns:

```json
{
  "predicted_state": "stressed",
  "confidence": "moderate",
  "explanation": "...",
  "important_features": [
    "increased pauses",
    "higher correction rate",
    "slower typing than baseline"
  ]
}
```

The LLM must not receive typed text.

---

# 16. LLM Prompt Design

The LLM should be instructed:

1. Use only the supplied numerical/structured features.
2. Do not invent user information.
3. Do not claim certainty.
4. Treat the ML prediction as an input rather than unquestionable truth.
5. Explain which features support the prediction.
6. Clearly state uncertainty.
7. Avoid medical or psychological diagnosis.

Example:

> "Given the user's current keystroke statistics, personal baseline deviations, recent trends, and ML prediction, provide a concise feature-grounded interpretation. Do not infer or mention the actual content typed by the user."

---

# 17. Explainability

The system should provide two explanation layers.

### ML explanation

Use:

- Feature importance
- Optional SHAP values

Example:

```text
Most influential features:
1. Correction rate
2. Flight-time deviation
3. Pause frequency
4. Typing-speed deviation
```

### LLM explanation

Convert these numerical findings into understandable language.

Example:

> "The model predicted a stressed state mainly because typing speed decreased relative to the user's baseline, while pauses and correction frequency increased."

This keeps the LLM explanation grounded in measurable features.

---

# 18. Dashboard

A simple Streamlit dashboard is sufficient.

### Dashboard sections

#### Current State

```text
Predicted State: STRESSED
Confidence: 81%
```

#### Current Features

```text
Typing Speed       3.1 keys/sec
Mean Dwell        124 ms
Mean Flight       168 ms
Correction Rate   8%
```

#### Baseline Comparison

```text
Typing Speed      ↓ 1.4 SD
Dwell Time        ↑ 0.9 SD
Flight Time       ↑ 1.2 SD
Corrections       ↑ 1.5 SD
```

#### Temporal Trend

Show the last several windows.

#### Explanation

Display the LLM-generated feature-grounded explanation.

#### Privacy

```text
Typed content: NOT COLLECTED
```

---

# 19. Evaluation

The project should evaluate four things.

## 19.1 ML Performance

- Accuracy
- Precision
- Recall
- F1
- Confusion matrix

## 19.2 Personalization

Compare:

```text
Model without baseline
vs
Model with baseline deviation
```

Question:

> Does personalization improve prediction?

## 19.3 Temporal Context

Compare:

```text
Current features only
vs
Current features + temporal trends
```

Question:

> Does recent behavior improve prediction?

## 19.4 LLM Explanation

Evaluate qualitatively:

- Does explanation reference actual features?
- Does it agree with model output?
- Does it avoid unsupported claims?
- Does it communicate uncertainty?

---

# 20. Ablation Study

A simple ablation study will strengthen the project.

| Experiment | Features |
|---|---|
| A | Raw aggregated keystroke features |
| B | + Personal baseline deviations |
| C | + Temporal trend features |
| D | + LLM explanation |

The key quantitative comparison should be A vs B vs C.

D is primarily an explanation layer, not necessarily an accuracy improvement.

---

# 21. Project Folder Structure

```text
llm-keystroke-mood/
│
├── data/
│   ├── raw/
│   │   └── synthetic_keystroke_raw_100k.csv
│   └── processed/
│       └── window_features.csv
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
│
├── src/
│   ├── data_generator.py
│   ├── cleaning.py
│   ├── features.py
│   ├── baseline.py
│   ├── model.py
│   └── llm.py
│
├── app/
│   └── dashboard.py
│
├── models/
│   └── xgboost_model.pkl
│
├── requirements.txt
├── README.md
└── PRD.md
```

---

# 22. Technology Stack

### Programming

- Python

### Data

- pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- scikit-learn
- XGBoost

### Explainability

- SHAP (optional)

### LLM

- OpenAI API or another available LLM API

### Dashboard

- Streamlit

---

# 23. Expected Deliverables

### Dataset

- 100,000 raw synthetic keystroke events

### Data Science

- Cleaned dataset
- EDA notebook
- Feature-engineered dataset
- Baseline/deviation dataset

### ML

- Trained Random Forest
- Trained XGBoost
- Evaluation metrics
- Confusion matrix
- Ablation study

### LLM

- Structured feature serializer
- Prompt
- LLM response
- Feature-grounded explanation

### Application

- Simple Streamlit dashboard

### Documentation

- README
- Methodology
- Limitations
- Results
- Final report
- Presentation

---

# 24. Key Research Question

> **Can privacy-preserving keystroke behavioral features, personalized against a user's normal typing baseline and combined with simple temporal context, support affective-state prediction, and can an LLM provide feature-grounded explanations without accessing typed content?**

---

# 25. Hypotheses

### H1

Keystroke behavioral features contain information useful for distinguishing the synthetic affective states.

### H2

Personal baseline deviations improve prediction compared with population-level features alone.

### H3

Simple temporal trend features improve prediction compared with current-window features alone.

### H4

An LLM can generate understandable explanations from structured keystroke features without requiring typed content.

These hypotheses are evaluated on the synthetic benchmark and should not be interpreted as proof of real-world human emotion recognition.

---

# 26. Limitations

The most important limitation is the synthetic nature of the dataset.

The project does **not** establish that:

- Real users type exactly like the synthetic users.
- The generated timing distributions represent true emotional responses.
- The reported ML accuracy will transfer to real people.
- The system can diagnose stress, depression, anxiety, or other medical conditions.

The appropriate future validation is a human-participant study where keystroke timing is collected without text content and affective labels are obtained through periodic self-report.

Prior research has demonstrated personalized emotion inference from interaction features and user-provided labels, including privacy-preserving approaches that avoid storing text content. citeturn0search9

---

# 27. Future Work

If additional time/data becomes available:

1. Collect real participant data.
2. Replace synthetic labels with self-reported valence/arousal.
3. Compare synthetic and real timing distributions.
4. Add digraph/trigraph features.
5. Test Leave-One-User-Out validation.
6. Investigate LSTM/Transformer temporal models.
7. Test LLM reasoning against classical ML explanations.
8. Conduct privacy and fairness analysis.

---

# 28. Success Criteria

The project is considered successful if it can demonstrate:

- A reproducible 100K-event synthetic dataset.
- Proper data cleaning.
- Meaningful EDA.
- Extraction of privacy-preserving keystroke features.
- User-specific baseline calculation.
- Baseline deviation features.
- A working ML classifier.
- User-independent evaluation.
- An ablation showing the effect of personalization/temporal features.
- An LLM that produces feature-grounded explanations.
- No typed content being required by the pipeline.

---

# 29. Viva Explanation

If asked why synthetic data was used:

> "We did not have access to enough human participants, so we created a controlled synthetic benchmark based on established keystroke-dynamics timing concepts. The purpose is to validate our complete data-science and ML pipeline. We explicitly do not claim that the resulting accuracy represents real-world emotion-recognition accuracy."

If asked how the emotions were generated:

> "The affective labels are synthetic assumptions, and the generator modifies timing and behavioral distributions to represent different simulated affective conditions. These assumptions are used to create a controlled benchmark, not presented as human ground truth."

If asked why this is useful:

> "It allows us to test cleaning, EDA, feature extraction, personalization, temporal context, ML evaluation and LLM explanation end-to-end before human-data validation."

---

# 30. Final Project Architecture

```text
                    ┌───────────────────────┐
                    │ Synthetic Raw Dataset │
                    │     100,000 Events    │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Data Cleaning + EDA │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   30-sec Windowing    │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Feature Extraction  │
                    │ Speed / Dwell / Flight│
                    │ Pauses / Corrections  │
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
          ┌─────────────────┐     ┌──────────────────┐
          │ Personal        │     │ Temporal Context │
          │ Baseline        │     │ Recent Trends    │
          └────────┬────────┘     └─────────┬────────┘
                   └──────────────┬─────────┘
                                  ▼
                       ┌────────────────────┐
                       │   XGBoost / RF     │
                       │ Mood Prediction    │
                       └──────────┬─────────┘
                                  │
                                  ▼
                       ┌────────────────────┐
                       │ Structured Feature │
                       │ Representation     │
                       └──────────┬─────────┘
                                  │
                                  ▼
                       ┌────────────────────┐
                       │       LLM          │
                       │ Explanation/Reason │
                       └──────────┬─────────┘
                                  │
                                  ▼
                       ┌────────────────────┐
                       │ Streamlit Dashboard│
                       └────────────────────┘
```

## Final Principle

**Keep the project simple enough to implement and defend, but complete enough to demonstrate the full ML lifecycle.**

The central contribution is not "we invented a perfect emotion detector." It is:

> **A privacy-preserving, personalized keystroke-to-affective-state pipeline with feature-grounded LLM explanations, demonstrated on a controlled synthetic benchmark.**
