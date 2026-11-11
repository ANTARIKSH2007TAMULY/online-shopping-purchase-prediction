# Will This Session Convert? — Online Shopping Purchase Prediction

A machine learning project for predicting whether an online shopping browsing session will end in a purchase. The project was completed as part of the **Scaler CML (Applied ML, Intro) End-Term Competition 2026**.

The task is a **binary classification problem**, but the model must output a probability for each session rather than a hard `0` or `1` prediction. Model performance is evaluated using **ROC AUC**.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Competition Details](#competition-details)
3. [Problem Statement](#problem-statement)
4. [Dataset Description](#dataset-description)
5. [Files](#files)
6. [Feature Description](#feature-description)
7. [Important Dataset Characteristics](#important-dataset-characteristics)
8. [Evaluation Metric](#evaluation-metric)
9. [Project Workflow](#project-workflow)
10. [Exploratory Data Analysis](#exploratory-data-analysis)
11. [Data Cleaning and Feature Engineering](#data-cleaning-and-feature-engineering)
12. [Categorical Feature Handling](#categorical-feature-handling)
13. [Models and Training Strategy](#models-and-training-strategy)
14. [Validation and Blending](#validation-and-blending)
15. [Results](#results)
16. [Generating the Submission](#generating-the-submission)
17. [How to Run the Notebook](#how-to-run-the-notebook)
18. [Libraries and Tools](#libraries-and-tools)
19. [Limitations and Considerations](#limitations-and-considerations)
20. [AI Assistance and Academic Transparency](#ai-assistance-and-academic-transparency)
21. [Acknowledgements and Dataset License](#acknowledgements-and-dataset-license)

---

## Project Overview

Online stores receive many browsing sessions, but only a fraction of visitors complete a purchase. Understanding which sessions are more likely to convert can help an online business make better decisions about discounts, recommendations, customer support, and marketing follow-ups.

This project uses session-level browsing information to estimate the probability of a purchase. The input data includes:

- The number of administrative, informational, and product-related pages visited.
- The time spent on different types of pages.
- Bounce rates, exit rates, and page values.
- The month and whether the session occurred on a weekend.
- Visitor type, operating system, browser, region, and traffic source.

The notebook follows a complete applied machine learning workflow: loading the data, exploratory data analysis, feature engineering, preprocessing, model training, hyperparameter tuning, cross-validation, ensemble prediction, and submission generation.

## Competition Details

| Item | Details |
|---|---|
| Competition | **Will This Session Convert?** |
| Course | Scaler CML — Applied ML, Intro |
| Competition type | Private, individual, in-class end-term competition |
| Task | Binary classification |
| Target | `Revenue` |
| Positive class | `1` — session ended in a purchase |
| Negative class | `0` — session did not end in a purchase |
| Evaluation metric | ROC AUC |
| Team size | 1 (individual) |
| Competition start | 10 October 2026, 2:00 PM IST |
| Final submission deadline | 18 October 2026, 11:59 PM IST |
| Notebook and write-up deadline | 21 October 2026, 11:59 PM IST |

The competition is private and intended for the Scaler CML batch. Participation requires the invitation link shared by the instructor.

## Problem Statement

Given the browsing information for an online shopping session, predict:

\[
P(\text{Revenue}=1 \mid \text{session features})
\]

In other words, the model should assign a higher score to sessions that are more likely to result in a purchase.

This is a probability-ranking task. The objective is not simply to classify each session as a buyer or non-buyer using a fixed threshold. The model should distinguish likely purchasers from non-purchasers as accurately as possible.

## Dataset Description

The competition uses a **synthetic dataset** generated to preserve statistical distributions and relationships found in the UCI *Online Shoppers Purchasing Intention* dataset (UCI dataset #468).

Although the synthetic data is based on the statistical characteristics of that source dataset, **no row in the competition dataset is a copy of an original real-world session**.

The original UCI dataset must not be downloaded or used for this competition. Only the competition-provided files should be used.

### Dataset size

| Dataset | Number of rows | Target availability |
|---|---:|---|
| Training set | 19,568 | `Revenue` is available |
| Test set | 10,431 | `Revenue` is not provided |
| Sample submission | 10,431 | Contains example predictions |
| Total | 30,430 | — |

The training target is imbalanced: approximately **15% of sessions result in a purchase**, while approximately **85% do not**.

The training and test sets contain the same input columns, except that the test set does not contain the target column `Revenue`.

The competition description states that there are no missing values in the original provided data. However, the notebook also treats negative duration values as invalid measurements and converts them to missing values during feature engineering.

## Files

The competition provides three CSV files:

| File | Purpose |
|---|---|
| `train.csv` | Training data containing session features and the `Revenue` target |
| `test.csv` | Unlabelled session data for which purchase probabilities must be predicted |
| `sample_submission.csv` | Submission template containing the required `id` and `Revenue` columns |

### Expected submission format

The final submission must contain exactly two columns, in this order:

```csv
id,Revenue
1,0.1547
3,0.0213
6,0.8731
```

Requirements:

- Include every `id` from `test.csv` exactly once.
- Keep the column names and their order unchanged.
- Predictions must be numeric probabilities between `0` and `1`.
- Do not submit hard class labels when probability predictions are required.
- The submission must contain 10,431 data rows, plus the header.

## Feature Description

The dataset contains session identifiers, browsing behaviour, engagement measurements, temporal information, visitor information, and the target.

| Column | Type | Description |
|---|---|---|
| `id` | Integer identifier | Unique session identifier. It is not treated as a predictive feature. |
| `Administrative` | Count | Number of administrative pages visited. |
| `Administrative_Duration` | Seconds | Total time spent on administrative pages. |
| `Informational` | Count | Number of informational pages visited. |
| `Informational_Duration` | Seconds | Total time spent on informational pages. |
| `ProductRelated` | Count | Number of product-related pages visited. |
| `ProductRelated_Duration` | Seconds | Total time spent on product-related pages. |
| `BounceRates` | Fraction | Average bounce rate of the pages visited. |
| `ExitRates` | Fraction | Average exit rate of the pages visited. |
| `PageValues` | Numeric | Average page value associated with the pages visited. It is zero for most sessions. |
| `SpecialDay` | Numeric, 0–1 | Proximity of the session date to a special shopping day. |
| `Month` | Category | Month of the session. |
| `OperatingSystems` | Category code | Visitor's operating system, coded from 1 to 8. |
| `Browser` | Category code | Visitor's browser, coded from 1 to 13. |
| `Region` | Category code | Visitor's region, coded from 1 to 9. |
| `TrafficType` | Category code | Traffic source category, coded from 1 to 20. |
| `VisitorType` | Category | `Returning_Visitor`, `New_Visitor`, or `Other`. |
| `Weekend` | Boolean | Whether the session occurred on a weekend. |
| `Revenue` | Binary target | `1` if the session resulted in a purchase, otherwise `0`. Present only in `train.csv`. |

### Important interpretation of categorical codes

`OperatingSystems`, `Browser`, `Region`, and `TrafficType` contain integer values, but these values are **category labels**, not measurements.

For example, browser code `8` does not mean that the browser is four times greater than browser code `2`. These columns should therefore be handled as categorical variables rather than ordinary continuous numeric features.

## Important Dataset Characteristics

A few properties of the dataset influence the modelling approach:

1. **Class imbalance:** Around 85% of sessions do not result in a purchase. Accuracy alone can therefore be misleading.
2. **Right-skewed measurements:** Page counts and duration columns have many small values and zeros, with a smaller number of large values.
3. **Many zero values:** A session may contain no visits to a particular page category, resulting in a count and duration of zero.
4. **Duration interpretation:** If a page count is zero, its duration is zero. A positive page count with a duration of zero is also possible.
5. **Page values:** `PageValues` is zero for most sessions and can be highly informative for purchase prediction.
6. **Categorical integer codes:** Operating system, browser, region, and traffic type codes represent categories.
7. **Month labels:** The dataset uses month names as categories. `June` is written in full, while other month labels may use shorter forms. January and April do not appear in the provided data.

These characteristics make careful EDA, feature engineering, categorical preprocessing, and ROC AUC-based validation important.

## Evaluation Metric

The competition uses **Area Under the Receiver Operating Characteristic Curve (ROC AUC)**.

ROC AUC measures how well a model ranks positive examples above negative examples.

A useful interpretation is:

- `0.50`: Random ranking; no discrimination beyond chance.
- `1.00`: Perfect ranking.
- A higher score indicates better separation between purchasing and non-purchasing sessions.

### Why accuracy is not the main metric

Because approximately 85% of sessions do not result in a purchase, a model that predicts `0` for every session could achieve approximately 85% accuracy while failing to identify any potential purchasers.

ROC AUC is more suitable because it evaluates the ranking of prediction scores across thresholds.

### Why probabilities are submitted

The notebook uses:

```python
model.predict_proba(X_test)[:, 1]
```

This extracts the estimated probability of the positive class (`Revenue = 1`).

Submitting probabilities preserves the model's ranking information. Converting them into hard `0` and `1` labels would discard useful information and can reduce ROC AUC.

### Public and private leaderboard

The test data is divided into two parts:

- **Public leaderboard:** Approximately 30% of the test rows.
- **Private leaderboard:** Approximately 70% of the test rows.

The private leaderboard determines the final competition ranking and grades. For this reason, cross-validation performance should be considered alongside public leaderboard performance. Repeatedly tuning against a small public leaderboard can lead to overfitting.

## Project Workflow

The notebook is organised into the following stages:

```text
Load CSV files
      |
      v
Inspect dataset shape, columns and statistics
      |
      v
Exploratory Data Analysis
      |
      v
Clean invalid duration values
      |
      v
Create behavioural and interaction features
      |
      v
Separate target, identifiers and input features
      |
      v
Identify numerical, binary and categorical columns
      |
      v
Build preprocessing pipeline
      |
      v
Train and tune XGBoost
      |
      v
Train and tune CatBoost
      |
      v
Train XGBoost, CatBoost and LightGBM with
repeated stratified cross-validation
      |
      v
Generate out-of-fold and test predictions
      |
      v
Rank-average model predictions
      |
      v
Create final CSV submission
```

## Exploratory Data Analysis

The notebook begins by loading the training, test, and sample submission files using pandas.

The following checks and visualisations are performed.

### 1. Dataset structure and descriptive statistics

- Display the first few rows of the datasets.
- Check the shape of the training and test sets.
- Inspect data types and non-null counts using `info()`.
- Review descriptive statistics using `describe()`.

This provides an initial understanding of feature ranges, distributions, and data types.

### 2. Distribution analysis

Histograms are plotted for selected features, including:

- `Administrative_Duration`
- `ProductRelated`
- `ProductRelated_Duration`
- `BounceRates`
- `ExitRates`
- `PageValues`

The purpose is to understand skewness, zero-heavy distributions, and the presence of large values.

### 3. Correlation heatmap

A correlation heatmap is created for numeric columns, excluding the session identifier.

It helps inspect linear relationships between numerical variables. Correlation is treated as an exploratory measure, not proof of causation or a complete measure of feature importance.

### 4. Target distribution

A count plot displays the number and percentage of sessions in each `Revenue` class.

This makes the imbalance between purchase and non-purchase sessions visible.

### 5. Weekday versus weekend purchase rate

The average `Revenue` value is calculated separately for weekend and weekday sessions.

Since the target is binary, its mean represents the purchase rate for each group.

### 6. Page value by purchase outcome

A box plot compares `PageValues` for sessions with `Revenue = 0` and `Revenue = 1`.

This helps examine whether sessions associated with higher page values show different purchasing behaviour.

### 7. Unique-value inspection

The number of unique values in each column is checked. This supports the decision about which features should be treated as categorical, binary, or numeric.

## Data Cleaning and Feature Engineering

Feature engineering is an important part of this project. The aim is to represent browsing behaviour in a way that helps the models identify patterns associated with purchases.

The same `add_features()` function is applied separately to both training and test data to keep the feature-generation logic consistent.

### 1. Handling negative duration values

The duration columns are:

- `Administrative_Duration`
- `Informational_Duration`
- `ProductRelated_Duration`

Negative duration values are not meaningful as actual time measurements. They are replaced with missing values:

```python
d[c] = d[c].where(d[c] >= 0)
```

This avoids treating negative values as valid durations. The tree-based models used in the notebook can work with missing numeric values.

### 2. Total pages

The total number of pages visited is calculated as:

\[
\text{total\_pages} =
\text{Administrative} +
\text{Informational} +
\text{ProductRelated}
\]

This provides a single measure of the visitor's overall browsing activity.

### 3. Total duration

The total time spent across the three page categories is calculated as:

\[
\text{total\_duration} =
\text{Administrative\_Duration} +
\text{Informational\_Duration} +
\text{ProductRelated\_Duration}
\]

The implementation uses `min_count=1` so that the sum remains missing if all component durations are missing.

### 4. Average duration per page

The feature `dur_per_page` is calculated as:

\[
\text{dur\_per\_page} =
\frac{\text{total\_duration}}{\text{total\_pages}+1}
\]

The added `1` prevents division by zero when no pages have been visited.

### 5. Product-related duration per page

\[
\text{prod\_dur\_per\_page} =
\frac{\text{ProductRelated\_Duration}}{\text{ProductRelated}+1}
\]

This represents the approximate time spent per product-related page.

### 6. Product page share

\[
\text{prod\_share} =
\frac{\text{ProductRelated}}{\text{total\_pages}+1}
\]

This captures the proportion of browsing activity directed towards product-related pages.

### 7. Exit rate minus bounce rate

\[
\text{exit\_minus\_bounce} =
\text{ExitRates} - \text{BounceRates}
\]

This feature provides another representation of the relationship between exit and bounce behaviour.

### 8. Page value indicator

The binary feature `pv_gt0` indicates whether a session has a positive page value:

\[
\text{pv\_gt0} =
\begin{cases}
1 & \text{if PageValues} > 0 \\
0 & \text{otherwise}
\end{cases}
\]

It separates sessions with a non-zero page value from those with a zero page value.

### 9. Log-transformed page value

\[
\text{log\_pv} = \log(1+\max(\text{PageValues},0))
\]

The transformation reduces the influence of very large page values and gives the model another representation of this highly skewed feature.

### 10. Page value and browsing interactions

The notebook creates three additional features:

\[
\text{pv\_x\_prod} =
\text{PageValues} \times \text{ProductRelated}
\]

\[
\text{pv\_x\_dur} =
\text{PageValues} \times \text{total\_duration}
\]

\[
\text{pv\_per\_page} =
\frac{\text{PageValues}}{\text{total\_pages}+1}
\]

These features combine page value with browsing volume and time spent. They allow the models to learn patterns involving more than one original variable.

### 11. Month conversion

The month names are converted into numeric month values using a mapping from January to December.

For example:

| Month | Numeric value |
|---|---:|
| Feb | 2 |
| Mar | 3 |
| May | 5 |
| June | 6 |
| Jul | 7 |
| Oct | 10 |
| Dec | 12 |

The notebook takes the first three characters of the month string, converts them to lowercase, and maps them to the corresponding month number. The original `Month` column is retained for categorical preprocessing, while `Month_num` provides an ordered numeric representation.

### Feature engineering summary

The notebook creates the following additional features:

| Feature | Purpose |
|---|---|
| `total_pages` | Total browsing activity |
| `total_duration` | Total time spent across page categories |
| `dur_per_page` | Approximate time spent per page |
| `prod_dur_per_page` | Approximate time spent per product page |
| `prod_share` | Share of total pages that are product-related |
| `exit_minus_bounce` | Difference between exit and bounce rates |
| `pv_gt0` | Whether page value is positive |
| `log_pv` | Log-transformed page value |
| `pv_x_prod` | Interaction between page value and product page count |
| `pv_x_dur` | Interaction between page value and total duration |
| `pv_per_page` | Page value relative to total page count |
| `Month_num` | Numeric representation of the month |

After feature engineering, the notebook reports:

| Dataset | Rows | Features used by the model |
|---|---:|---:|
| Training features (`X`) | 19,568 | 29 |
| Test features (`X_test`) | 10,431 | 29 |

The `id` column is retained separately for submission generation and excluded from model training. The `Revenue` column is separated as the target.

## Categorical Feature Handling

The notebook uses different preprocessing strategies depending on the type and number of unique values in a feature.

### Feature groups

The columns are divided into four groups:

1. **Binary columns:** Numeric features with exactly two unique values.
2. **Low-cardinality categorical columns:** Text/category columns with at most 10 unique values, along with the integer-coded category columns.
3. **Passthrough columns:** Selected numeric features with at most 20 unique values, excluding binary and identifier-like categorical columns.
4. **Remaining numeric columns:** Other numeric features, passed through without scaling.

The notebook identifies the following binary columns:

- `Weekend`
- `pv_gt0`

The low-cardinality categorical group includes:

- `Month`
- `VisitorType`
- `OperatingSystems`
- `Browser`
- `Region`
- `TrafficType`

The selected passthrough columns are:

- `Informational`
- `SpecialDay`
- `Month_num`

### One-hot encoding

`OneHotEncoder` is used for categorical columns.

- Low-cardinality categorical columns use `drop="first"`.
- Binary columns use `drop="if_binary"`.
- `handle_unknown="ignore"` allows the transformer to handle categories that were not seen during fitting.
- The remaining selected numeric features are passed through.

The preprocessing is organised using a scikit-learn `ColumnTransformer` and `Pipeline`.

### CatBoost categorical handling

For CatBoost, categorical columns are passed as strings and supplied through the `cat_features` parameter. This allows CatBoost to process categorical variables natively rather than requiring the same one-hot encoding used by the XGBoost and LightGBM pipeline.

## Models and Training Strategy

Three gradient-boosting models are used in the final ensemble:

- XGBoost
- CatBoost
- LightGBM

The notebook first experiments with individual models and then combines their predictions.

### 1. XGBoost

XGBoost is trained using an `XGBClassifier` inside a scikit-learn pipeline.

The pipeline combines:

1. The `ColumnTransformer` preprocessing step.
2. The XGBoost binary classification model.

Important configuration choices include:

| Parameter | Setting |
|---|---|
| Objective | `binary:logistic` |
| Evaluation metric | `auc` |
| Tree method | `hist` |
| Random state | `42` |
| Parallel jobs | `-1` |

#### Hyperparameter tuning

`RandomizedSearchCV` is used to search across 40 parameter combinations with 5-fold stratified cross-validation.

The search explores:

- `n_estimators`
- `learning_rate`
- `max_depth`
- `min_child_weight`
- `gamma`
- `subsample`
- `colsample_bytree`
- `reg_lambda`
- `reg_alpha`

The search uses ROC AUC as its scoring metric and refits the best estimator on the complete training data.

The best recorded 5-fold cross-validation ROC AUC for this XGBoost search is **0.94313**.

### 2. CatBoost

CatBoost is trained using its native categorical feature support.

Before training:

- The categorical columns are identified.
- Missing categorical values are filled with a placeholder.
- Categorical values are converted to strings.
- The test columns are aligned with the training feature columns.

`RandomizedSearchCV` evaluates 25 parameter combinations using 3-fold stratified cross-validation.

The search explores:

- `iterations`
- `depth`
- `learning_rate`
- `l2_leaf_reg`
- `random_strength`
- `border_count`

The evaluation metric is ROC AUC, and the best estimator is refitted using the full training data.

### 3. LightGBM

LightGBM is included in the final ensemble as another gradient-boosting model.

It uses the same general preprocessing strategy as the XGBoost pipeline. The notebook trains LightGBM with a relatively shallow tree configuration and early stopping.

Important settings include:

| Parameter | Setting |
|---|---|
| Estimators | Up to 2,000 |
| Learning rate | 0.02 |
| Number of leaves | 15 |
| Maximum depth | 5 |
| Minimum child samples | 40 |
| Subsample | 0.8 |
| Column sample | 0.7 |
| L2 regularisation | 5.0 |
| Early stopping | 100 rounds |
| Evaluation metric | AUC |

Early stopping monitors validation performance and stops training when the validation score does not improve for the specified number of rounds.

## Validation and Blending

The final ensemble uses **repeated stratified cross-validation** to generate out-of-fold predictions.

### Repeated stratified folds

The notebook uses:

- 5 folds per run.
- 3 random seeds: `42`, `7`, and `2024`.
- Stratified splitting to preserve the purchase/non-purchase ratio in each fold.

For each seed and fold:

1. A model is trained on the fold's training portion.
2. Predictions are generated for the held-out validation portion.
3. Predictions are generated for the competition test set.
4. The predictions are accumulated and averaged across folds and seeds.

The models are trained independently:

- XGBoost uses the tuned pipeline.
- CatBoost uses the tuned CatBoost parameters and native categorical columns.
- LightGBM uses the configured shallow model with early stopping.

### Out-of-fold predictions

Out-of-fold (OOF) predictions are generated for the training rows. Each OOF prediction comes from a model that did not train on that row.

OOF predictions are useful for comparing models and evaluating blending strategies without directly evaluating a model on the same rows it was trained on.

### Rank averaging

The final ensemble uses rank averaging instead of directly averaging raw predicted probabilities.

For each model, predictions are converted into normalised ranks:

\[
R(x) = \frac{\operatorname{rank}(x)-1}{n-1}
\]

The ranks from XGBoost, CatBoost, and LightGBM are then averaged:

\[
\text{Blended Rank} =
\frac{R_{\text{XGB}}+R_{\text{CatBoost}}+R_{\text{LightGBM}}}{3}
\]

This approach combines the relative ordering learned by each model. Since ROC AUC depends on ranking, rank averaging is a reasonable ensemble strategy to test.

The same rank transformation and averaging process is applied to the test predictions to create the final submission.

## Results

The recorded validation results from the final repeated cross-validation workflow are:

| Model | OOF ROC AUC |
|---|---:|
| XGBoost | 0.943266 |
| CatBoost | 0.945420 |
| LightGBM | 0.944073 |
| **Rank-averaged ensemble** | **0.945101** |

These values are **local out-of-fold validation results**, not public or private Kaggle leaderboard scores.

The tuned XGBoost search separately recorded a best 5-fold CV ROC AUC of **0.94313**. This value comes from its hyperparameter-search stage and should not be confused with the repeated OOF result in the ensemble stage.

The ensemble submission generated by the notebook is:

```text
submission_xgb_catboost_lightgbm.csv
```

It contains 10,431 rows and the two required columns: `id` and `Revenue`.

## Generating the Submission

The notebook creates the final submission using the competition's sample submission as a template:

```python
submission_3model = sample_submission.copy()
submission_3model[target_col] = final_predictions

submission_3model.to_csv(
    "submission_xgb_catboost_lightgbm.csv",
    index=False
)
```

Using the sample submission as a template helps preserve the required column names and row order.

Before submitting, verify:

```python
submission_3model.shape
submission_3model.head()
submission_3model.isna().sum()
submission_3model["Revenue"].between(0, 1).all()
```

Also confirm that the submission IDs match the IDs in `test.csv`:

```python
submission_3model["id"].equals(test["id"])
```

The final CSV should have:

- 10,431 data rows.
- Exactly two columns: `id`, `Revenue`.
- No missing predictions.
- Probabilities between 0 and 1.
- The same test ID order as the sample submission.

## How to Run the Notebook

The main notebook is:

```text
25bcs10450-cml-kaggle (13).ipynb
```

It is designed for the Kaggle Notebook environment and reads the competition files from the Kaggle input directory.

### Steps

1. Join the private competition using the invitation link provided by the instructor.
2. Open the competition's Kaggle notebook environment.
3. Add the competition dataset to the notebook.
4. Open or upload the project notebook.
5. Confirm that the CSV paths in the data-loading cell match the dataset directory available in your Kaggle environment.
6. Run the notebook cells in order, from data loading through final submission generation.
7. Check the printed cross-validation scores and the generated submission shape.
8. Download or submit `submission_xgb_catboost_lightgbm.csv` through the competition's Submission page.

### Important note about paths

The notebook uses Kaggle input paths. If the dataset directory differs in another Kaggle environment, update only the `read_csv()` paths in the data-loading cell. The expected filenames remain:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

## Libraries and Tools

The notebook uses the following Python libraries:

| Library | Purpose |
|---|---|
| `pandas` | Loading and manipulating tabular data |
| `numpy` | Numerical operations and feature engineering |
| `matplotlib` | Plotting distributions and charts |
| `seaborn` | Statistical visualisations and heatmaps |
| `scikit-learn` | Preprocessing, pipelines, cross-validation, hyperparameter search, and ROC AUC |
| `XGBoost` | Gradient-boosting classification |
| `CatBoost` | Gradient boosting with native categorical feature support |
| `LightGBM` | Gradient-boosting classification |
| `SciPy` | Parameter distributions and rank calculations |

The notebook also uses Kaggle's hosted notebook environment for data access and execution.

## Limitations and Considerations

- The dataset is synthetic. Model performance on this competition should not automatically be interpreted as real-world business performance.
- The original UCI dataset is not permitted as an additional training source for this competition.
- The target is imbalanced, so accuracy is not an appropriate primary evaluation metric.
- The public leaderboard uses only part of the test data. A strong public score does not guarantee a similarly strong private score.
- The OOF scores are estimates based on the chosen folds and random seeds. Different splits or random seeds may produce different results.
- Rank averaging combines model orderings, but it does not guarantee an improvement over every individual model. In the recorded OOF results, CatBoost has a higher score than the rank-averaged ensemble.
- The notebook's preprocessing and feature engineering are designed for the provided competition data. Any changes to the data or feature definitions should be validated again.
- The notebook may print warnings when a validation or test set contains an unseen category. The one-hot encoder is configured with `handle_unknown="ignore"` so those categories do not stop prediction.

## AI Assistance and Academic Transparency

AI tools were used as assistance during parts of the project, particularly for generating or refining code for selected machine learning algorithms, experimenting with model configurations, and improving the written explanation.

The workflow was not based solely on accepting generated code. The dataset was explored, features were examined, models were trained, validation scores were compared, and different modelling approaches were tested.

Some code blocks were also reused from a related machine learning project that had already been implemented and tested as part of coursework. The three-model blending workflow received the most AI assistance.

The final notebook should be reviewed and understood by the author, including the feature engineering, model validation, and ensemble-generation steps.

## Acknowledgements and Dataset License

The competition dataset is synthetic and was generated using the statistical properties of:

**Online Shoppers Purchasing Intention Dataset — UCI Machine Learning Repository, Dataset #468**

- Dataset: https://archive.ics.uci.edu/dataset/468
- License: **Creative Commons Attribution 4.0 International (CC BY 4.0)**
- Reference: Sakar, C. O., Polat, S. O., Katircioglu, M., & Kastro, Y. (2019), *Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks*, Neural Computing and Applications.

The competition data itself is synthetic. No original UCI row is copied into the competition dataset. The original UCI file must not be used for this competition.

---

**Project:** Will This Session Convert?  
**Course:** Scaler CML — Applied ML, Intro  
**Task:** Predict online shopping purchase probability  
**Primary metric:** ROC AUC  
