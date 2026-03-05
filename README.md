# 🎮 Video Game Sales Analysis & Prediction

> A comprehensive data science project applying **Classification**, **Clustering**, and **Regression** techniques on real-world video game sales data using R.

---

## 📌 Project Overview

This project analyzes **16,000+ video game records** from VGChartz combined with Metacritic ratings to uncover patterns in the gaming industry. Three core machine learning tasks were performed end-to-end — from data collection and exploratory analysis through preprocessing, modeling, and evaluation.

| Aspect | Details |
|---|---|
| **Language** | R |
| **Dataset** | [Video Game Sales with Ratings](https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings) (Kaggle) |
| **Records** | ~16,700 games (complete cases ~6,900) |
| **Domain** | Entertainment / Gaming Industry |

---

## 📂 Repository Structure

```
├── C-G10-final-classification.R      # Decision Tree — ESRB Rating Prediction
├── C-G10-final-clustering.R          # K-Means — Game Market Segmentation
├── C-G10-final-regression.R          # Linear Regression — Global Sales Prediction
├── Video_Games_Sales_as_at_22_Dec_2016.csv   # Source dataset
├── previous code/                    # Earlier lab exercises & practice scripts
└── README.md
```

---

## 📊 Dataset Description

The dataset merges web-scraped sales data from **VGChartz** with review scores from **Metacritic**. Some platforms lack Metacritic coverage, resulting in missing observations.

| Feature | Type | Description |
|---|---|---|
| `Name` | Categorical | Title of the game |
| `Platform` | Categorical | Console/platform (PS4, X360, Wii, PC, etc.) |
| `Year_of_Release` | Numeric | Release year |
| `Genre` | Categorical | Game genre (Action, Sports, Shooter, etc.) |
| `Publisher` | Categorical | Publishing company |
| `NA_Sales` | Numeric | North America sales (millions) |
| `EU_Sales` | Numeric | Europe sales (millions) |
| `JP_Sales` | Numeric | Japan sales (millions) |
| `Other_Sales` | Numeric | Rest of world sales (millions) |
| `Global_Sales` | Numeric | Total worldwide sales (millions) |
| `Critic_Score` | Numeric | Aggregate critic score by Metacritic staff (0–100) |
| `Critic_Count` | Numeric | Number of critics contributing to the score |
| `User_Score` | Numeric | Metacritic subscriber score (0–10) |
| `User_Count` | Numeric | Number of users who rated |
| `Developer` | Categorical | Studio that developed the game |
| `Rating` | Categorical | ESRB content rating (E, E10+, T, M, etc.) |

---

## 🔬 Project Pipeline

Each analysis follows a structured **5-stage pipeline**:

```
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌────────────┐    ┌──────────────┐
│ A. Data      │ -> │ B. EDA &         │ -> │ C. Data          │ -> │ D. Model   │ -> │ E. Evaluation│
│ Collection   │    │ Exploration      │    │ Preprocessing    │    │ Building   │    │ & Insights   │
└──────────────┘    └──────────────────┘    └──────────────────┘    └────────────┘    └──────────────┘
```

---

## 1️⃣ Classification — ESRB Rating Prediction

> **File:** `C-G10-final-classification.R`
>
> **Goal:** Predict the ESRB content rating (E, E10+, T, M) of a video game using a Decision Tree classifier.

### Approach

| Step | What Was Done |
|---|---|
| **Data Collection** | Loaded 16,700+ records from Kaggle via Google Drive |
| **EDA** | 8 visualizations — rating distribution bar chart, genre distribution, boxplots of sales by rating, critic score histogram, outlier detection, critic vs. user score scatter, correlation heatmap, rating proportions across genres |
| **Preprocessing** | Removed missing/rare ratings → kept only E, E10+, T, M; median imputation for missing numerics; IQR-based outlier capping on sales columns; platform grouping (Nintendo, PlayStation, Xbox, PC, Other); z-score normalization; log transformation on sales; engineered features — regional sales ratios, score difference, game age |
| **Modeling** | Decision Tree (`rpart`) with `cp = 0.01`, `maxdepth = 10`; 70/30 stratified train-test split |
| **Evaluation** | Confusion matrix, overall accuracy, per-class precision/recall/F1, macro-averaged metrics, variable importance ranking |

### Key Visualizations Produced

- Distribution of ESRB Ratings (bar chart)
- Global Sales Distribution by Rating (boxplot)
- Rating Distribution Across Genres (stacked bar)
- Correlation Heatmap of numerical features
- Decision Tree structure visualization
- Confusion Matrix heatmap
- Variable Importance bar chart

### Key Results

| Metric | Description |
|---|---|
| **Algorithm** | Decision Tree (CART) |
| **Target** | ESRB Rating — 4 classes (E, E10+, T, M) |
| **Split** | 70% train / 30% test |
| **Top Predictors** | Genre, regional sales patterns, Critic Score, Japanese sales ratio |

### Insights Discovered

- **Action/Shooter** games strongly associate with **M (Mature)** ratings
- **Sports/Racing** games predominantly receive **E (Everyone)** ratings
- Higher **Critic Scores** tend to correlate with more mature ratings (T/M)
- **Japanese sales ratio** is a meaningful predictor of game content ratings
- Genre alone is one of the strongest single predictors of ESRB rating

---

## 2️⃣ Clustering — Game Market Segmentation

> **File:** `C-G10-final-clustering.R`
>
> **Goal:** Discover natural groupings of video games based on sales performance and review metrics using K-Means clustering.

### Approach

| Step | What Was Done |
|---|---|
| **Data Collection** | Same Kaggle dataset loaded via Google Drive |
| **EDA** | 8 visualizations — global sales histogram, critic score distribution, sales by genre boxplot, outlier detection, NA vs. EU sales scatter, critic score vs. global sales, correlation heatmap, regional sales bar comparison |
| **Preprocessing** | Filtered out zero/missing sales; median imputation; IQR outlier capping; log transformation on sales; feature engineering (sales ratios, avg score, score difference, game age); z-score standardization (critical for K-Means distance calculations) |
| **Modeling** | K-Means with **Elbow Method** (WSS for K=1–10) and **Silhouette Analysis** (K=2–10) to find optimal K; final model with **K=4** clusters, `nstart=25`, `iter.max=100` |
| **Evaluation** | Silhouette score & plot, PCA-based 2D cluster visualization, cluster profiling with mean feature values |

### Clustering Features Used

```
NA_Sales  ·  EU_Sales  ·  JP_Sales  ·  Other_Sales
Critic_Score  ·  User_Score  ·  Critic_Count  ·  User_Count
```

### Key Visualizations Produced

- Elbow Curve (WSS vs. K)
- Silhouette Score across K values
- PCA 2D Cluster Visualization with convex hulls
- Cluster Size bar chart
- Cluster Centers by Feature (grouped bar chart)
- Feature Distributions by Cluster (faceted boxplots)
- NA Sales vs. EU Sales colored by cluster (scatter)
- Critic Score vs. Global Sales colored by cluster (scatter)
- Silhouette plot per observation

### Cluster Profiles (K=4)

| Cluster | Profile | Characteristics |
|---|---|---|
| **1** | Blockbuster / AAA Games | High sales across NA & EU, high critic/user scores, many reviews |
| **2** | Japan-Focused Titles | Strong JP market sales, moderate critic scores |
| **3** | Niche / Low-Performing Games | Below-average sales across all regions |
| **4** | Average / Mid-Tier Games | Moderate sales and review metrics |

### Key Results

| Metric | Value |
|---|---|
| **Algorithm** | K-Means Clustering |
| **Optimal K** | 4 (validated via Elbow + Silhouette) |
| **Between SS / Total SS** | Measures compactness of clusters |
| **Silhouette Score** | Evaluated for cluster quality |

### Insights Discovered

- The gaming market naturally segments into **distinct tiers** — from blockbusters to niche titles
- **Regional sales patterns** differ significantly: some games dominate in Japan but underperform in the West, and vice versa
- **Critic scores and review counts** help differentiate quality tiers among games
- Cluster analysis can power **recommendation systems** (similar game profiles) and **targeted marketing strategies**

---

## 3️⃣ Regression — Global Sales Prediction

> **File:** `C-G10-final-regression.R`
>
> **Goal:** Predict a video game's worldwide sales (in millions) using Linear Regression.

### Approach

| Step | What Was Done |
|---|---|
| **Data Collection** | Same Kaggle dataset loaded via Google Drive |
| **EDA** | 9 visualizations — global sales histogram, log-transformed sales distribution, genre boxplot, outlier detection boxplot, critic score vs. sales scatter, user score vs. sales scatter, correlation heatmap, yearly average sales trend line, regional sales comparison bar chart |
| **Preprocessing** | Removed zero/missing Global Sales; median imputation; IQR outlier capping; categorical encoding (Genre, Platform grouped, Rating); log transformation for skewness reduction; z-score normalization; engineered features — sales ratios, score difference, game age, total reviews, average score |
| **Modeling** | Multiple Linear Regression via `lm()`; 70/30 train-test split; negative predictions clipped to 0 |
| **Evaluation** | RMSE, MAE, R², Adjusted R², MAPE; 5 diagnostic plots — actual vs. predicted scatter, residual plot, residual distribution histogram, Q-Q plot for normality, absolute error distribution |

### Key Visualizations Produced

- Global Sales distribution (raw + log-transformed)
- Critic Score vs. Global Sales (scatter with regression line)
- Average Global Sales trend over years (line chart)
- Top 15 Regression Coefficients (bar chart)
- Actual vs. Predicted scatter plot (with R² annotation)
- Residual Plot (predicted vs. residuals)
- Q-Q Plot of residuals (normality check)
- Residual Distribution histogram

### Key Results

| Metric | Description |
|---|---|
| **Algorithm** | Multiple Linear Regression |
| **Target** | `Global_Sales` (continuous, in millions) |
| **Split** | 70% train / 30% test |
| **Metrics** | RMSE, MAE, R², Adjusted R², MAPE |

### Significant Predictors

| Predictor | Effect on Sales |
|---|---|
| `Critic_Score` | Higher scores → higher sales |
| `Platform_Category` | Different platforms have different sales potential |
| `Genre` | Action & Sports genres tend to sell more |
| `Total_Reviews` | More reviews correlate with higher sales |
| `Game_Age` | Older games may have accumulated more lifetime sales |

### Insights Discovered

- **Critic Score** is one of the strongest predictors of commercial success
- Sales data is **highly right-skewed** — a small number of blockbusters dominate, while most games sell modestly
- **Linear models have limitations** here: non-linear relationships and external factors (marketing budget, franchise power, release timing) are not captured
- The model reveals which measurable game attributes most influence sales potential

---

## 🛠️ Technologies & Libraries

| Library | Purpose |
|---|---|
| `caret` | Data partitioning, model training & confusion matrix |
| `rpart` + `rpart.plot` | Decision Tree construction & visualization |
| `ggplot2` | All visualizations (bar, scatter, box, histogram, heatmap, etc.) |
| `dplyr` | Data manipulation & transformation |
| `corrplot` | Correlation heatmap matrices |
| `cluster` | Silhouette analysis for clustering |
| `factoextra` | Elbow method, silhouette plots, PCA cluster visualization |
| `tidyr` | Data reshaping (pivot_longer) |
| `e1071` | Skewness calculation |
| `Metrics` | Regression evaluation metrics |
| `car` | Regression diagnostics |
| `GGally` | Pair plots and extended ggplot2 functionality |

---

## 🧠 What I Learned

### Data Science Skills

- **End-to-end ML workflow**: From raw data → EDA → preprocessing → modeling → evaluation → interpretation
- **Exploratory Data Analysis**: Using multiple visualization types to understand distributions, relationships, and outliers before modeling
- **Data Preprocessing**: Handling missing values (median imputation), outlier treatment (IQR capping), encoding categorical variables, feature scaling (z-score), and log transformations for skewness reduction
- **Feature Engineering**: Creating meaningful derived features (sales ratios, score differences, game age) that improve model performance

### Machine Learning Techniques

- **Supervised Learning — Classification**: Building Decision Trees for multi-class prediction; understanding how tree splits are made based on information gain; interpreting variable importance
- **Unsupervised Learning — Clustering**: Using K-Means to discover hidden patterns; selecting optimal K via Elbow and Silhouette methods; profiling and interpreting cluster characteristics
- **Supervised Learning — Regression**: Fitting Linear Regression models; understanding coefficients, R², and residual diagnostics; evaluating with RMSE, MAE, and MAPE

### Practical Takeaways

- Real-world datasets are **messy** — missing values, mixed types (`User_Score` contains `"tbd"`), inconsistent categories, and heavy outliers are the norm, not the exception
- **Feature scaling matters** — K-Means is distance-based and produces meaningless results without normalization
- **Domain knowledge helps** — grouping 30+ gaming platforms into manufacturer categories (Nintendo, PlayStation, Xbox, PC, Other) was essential for usable categorical encoding
- **No model is perfect** — understanding limitations (linear assumptions, unexplained variance, skewed targets) is as important as building the model

---

## ▶️ How to Run

1. **Install R** (version 4.0+) and optionally RStudio
2. **Clone** this repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   ```
3. **Open** any of the three `.R` files in RStudio
4. **Run the script** — all required packages are auto-installed if missing, and the dataset is loaded directly from Google Drive (no manual download needed)

> **Note:** The scripts are self-contained. Each one loads the dataset, performs full EDA, preprocessing, modeling, and evaluation independently.

---

## 📎 Acknowledgements

- **Dataset**: Originally scraped from [VGChartz](https://www.vgchartz.com/) and extended with [Metacritic](https://www.metacritic.com/) data. Published on Kaggle by [Rush Kirubi](https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings).
- **Scraper**: Adapted from [wtamu-cisresearch/scraper](https://github.com/wtamu-cisresearch/scraper).

---

<p align="center"><i>Built as a university Data Science course final project</i></p>
