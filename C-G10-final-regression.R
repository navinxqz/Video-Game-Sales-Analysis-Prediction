# ============================================================================
# FINAL PROJECT: REGRESSION MODEL
# Student ID: 23-50079-1, 23-50063-1
# Dataset: Video Game Sales with Ratings
# Task: Predict Global Sales (Continuous) using Linear Regression
# ============================================================================

# ----------------------------------------------------------------------------
# A. DATA COLLECTION
# ----------------------------------------------------------------------------
# Dataset: Video Game Sales with Ratings
# Source: Kaggle - https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings
# Description: Video game sales data from VGChartz combined with Metacritic ratings
# Domain: Entertainment/Gaming
# Target Variable: Global_Sales (Continuous - in millions)
# ----------------------------------------------------------------------------

# Install and load required packages
if (!require("caret")) install.packages("caret", repos = "https://cloud.r-project.org")
if (!require("ggplot2")) install.packages("ggplot2", repos = "https://cloud.r-project.org")
if (!require("dplyr")) install.packages("dplyr", repos = "https://cloud.r-project.org")
if (!require("corrplot")) install.packages("corrplot", repos = "https://cloud.r-project.org")
if (!require("Metrics")) install.packages("Metrics", repos = "https://cloud.r-project.org")
if (!require("car")) install.packages("car", repos = "https://cloud.r-project.org")

library(caret)
library(ggplot2)
library(dplyr)
library(corrplot)
library(Metrics)
library(car)

# Load dataset automatically from Google Drive (no hardcoded local paths)
# Google Drive Link: https://drive.google.com/file/d/13AKg27UEoIErD9xA9JO8rWBdQ1dzHlAt/view?usp=sharing
url <- "https://drive.google.com/uc?export=download&id=13AKg27UEoIErD9xA9JO8rWBdQ1dzHlAt"

# Load data from Google Drive
data <- read.csv(url, stringsAsFactors = FALSE)
cat("Dataset loaded successfully from Google Drive!\n")

cat("\n============================================================\n")
cat("REGRESSION MODEL - Predicting Global Sales\n")
cat("============================================================\n\n")

# ----------------------------------------------------------------------------
# B. DATA UNDERSTANDING & EXPLORATION (EDA)
# ----------------------------------------------------------------------------
cat("\n============================================================\n")
cat("B. DATA UNDERSTANDING & EXPLORATION\n")
cat("============================================================\n\n")

# Display shape of dataset
cat("Dataset Shape:\n")
cat("Rows:", nrow(data), "\n")
cat("Columns:", ncol(data), "\n\n")

# Display first few rows
cat("First 6 rows of data:\n")
print(head(data))

# Data structure and types
cat("\nData Structure:\n")
str(data)

# Summary statistics
cat("\nSummary Statistics:\n")
print(summary(data))

# Identify categorical and numerical features
cat("\n--- Feature Types ---\n")
categorical_cols <- names(data)[sapply(data, function(x) is.character(x) | is.factor(x))]
numerical_cols <- names(data)[sapply(data, is.numeric)]

cat("\nCategorical Features:", paste(categorical_cols, collapse = ", "), "\n")
cat("Numerical Features:", paste(numerical_cols, collapse = ", "), "\n")

# Check missing values
cat("\n--- Missing Values ---\n")
missing_vals <- colSums(is.na(data))
print(missing_vals[missing_vals > 0])
cat("\nTotal missing values:", sum(is.na(data)), "\n")

# Target variable (Global_Sales) statistics
cat("\n--- Target Variable (Global_Sales) Statistics ---\n")
cat("Mean:", mean(data$Global_Sales, na.rm = TRUE), "million\n")
cat("Median:", median(data$Global_Sales, na.rm = TRUE), "million\n")
cat("Std Dev:", sd(data$Global_Sales, na.rm = TRUE), "million\n")
cat("Min:", min(data$Global_Sales, na.rm = TRUE), "million\n")
cat("Max:", max(data$Global_Sales, na.rm = TRUE), "million\n")

# ============== VISUALIZATIONS ==============
cat("\nGenerating visualizations...\n")

# 1. Histogram of Global Sales (Target Variable)
ggplot(data, aes(x = Global_Sales)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  labs(title = "Distribution of Global Sales (Target Variable)",
       x = "Global Sales (Millions)", y = "Frequency") +
  theme_minimal()

# 2. Log-transformed Global Sales distribution (to see pattern better)
ggplot(data %>% filter(Global_Sales > 0), aes(x = log10(Global_Sales))) +
  geom_histogram(bins = 50, fill = "darkgreen", color = "white") +
  labs(title = "Distribution of Log10(Global Sales)",
       x = "Log10(Global Sales)", y = "Frequency") +
  theme_minimal()

# 3. Boxplot of Global Sales by Genre
ggplot(data %>% filter(!is.na(Genre)), aes(x = Genre, y = Global_Sales, fill = Genre)) +
  geom_boxplot() +
  labs(title = "Global Sales Distribution by Genre",
       x = "Genre", y = "Global Sales (Millions)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none") +
  coord_cartesian(ylim = c(0, 5))

# 4. Boxplot to detect outliers in Global Sales
ggplot(data, aes(y = Global_Sales)) +
  geom_boxplot(fill = "tomato") +
  labs(title = "Boxplot of Global Sales (Outlier Detection)",
       y = "Global Sales (Millions)") +
  theme_minimal()

# 5. Scatterplot: Critic Score vs Global Sales
data$User_Score_Num <- suppressWarnings(as.numeric(data$User_Score))

ggplot(data %>% filter(!is.na(Critic_Score)), 
       aes(x = Critic_Score, y = Global_Sales)) +
  geom_point(alpha = 0.3, color = "darkblue") +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(title = "Critic Score vs Global Sales",
       x = "Critic Score", y = "Global Sales (Millions)") +
  theme_minimal() +
  coord_cartesian(ylim = c(0, 10))

# 6. Scatterplot: User Score vs Global Sales
ggplot(data %>% filter(!is.na(User_Score_Num)), 
       aes(x = User_Score_Num, y = Global_Sales)) +
  geom_point(alpha = 0.3, color = "darkgreen") +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(title = "User Score vs Global Sales",
       x = "User Score", y = "Global Sales (Millions)") +
  theme_minimal() +
  coord_cartesian(ylim = c(0, 10))

# 7. Correlation heatmap for numerical variables
data_num <- data %>%
  select(NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales, 
         Critic_Score, Critic_Count, User_Count, Year_of_Release) %>%
  mutate(across(everything(), as.numeric)) %>%
  na.omit()

cor_matrix <- cor(data_num, use = "complete.obs")
corrplot(cor_matrix, method = "color", type = "upper",
         addCoef.col = "black", tl.col = "black", tl.srt = 45,
         title = "Correlation Heatmap - Numerical Variables", mar = c(0,0,1,0))

# 8. Trend: Year vs Average Global Sales
yearly_sales <- data %>%
  filter(!is.na(Year_of_Release) & Year_of_Release >= 1990 & Year_of_Release <= 2016) %>%
  group_by(Year_of_Release) %>%
  summarise(Avg_Sales = mean(Global_Sales, na.rm = TRUE),
            Total_Games = n())

ggplot(yearly_sales, aes(x = Year_of_Release, y = Avg_Sales)) +
  geom_line(color = "steelblue", size = 1) +
  geom_point(color = "darkblue", size = 2) +
  labs(title = "Average Global Sales Over Years",
       x = "Year of Release", y = "Average Global Sales (Millions)") +
  theme_minimal()

# 9. Regional Sales Comparison
regional_sales <- data %>%
  summarise(NA_Sales = sum(NA_Sales, na.rm = TRUE),
            EU_Sales = sum(EU_Sales, na.rm = TRUE),
            JP_Sales = sum(JP_Sales, na.rm = TRUE),
            Other_Sales = sum(Other_Sales, na.rm = TRUE)) %>%
  tidyr::pivot_longer(cols = everything(), names_to = "Region", values_to = "Sales")

ggplot(regional_sales, aes(x = Region, y = Sales, fill = Region)) +
  geom_bar(stat = "identity") +
  labs(title = "Total Sales by Region",
       x = "Region", y = "Total Sales (Millions)") +
  theme_minimal() +
  theme(legend.position = "none")

# ----------------------------------------------------------------------------
# C. DATA PREPROCESSING
# ----------------------------------------------------------------------------
cat("\n============================================================\n")
cat("C. DATA PREPROCESSING\n")
cat("============================================================\n\n")

# Create a copy for preprocessing
df <- data

# Remove rows where Global_Sales is missing or zero (target variable)
cat("Removing rows with missing/zero Global Sales (target variable)...\n")
df <- df %>% filter(!is.na(Global_Sales) & Global_Sales > 0)
cat("Rows after removing invalid Global Sales:", nrow(df), "\n")

# Handle missing values in numerical columns - impute with median
cat("\n--- Handling Missing Values ---\n")

# Convert User_Score to numeric first
df$User_Score <- suppressWarnings(as.numeric(df$User_Score))

numerical_features <- c("NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", 
                        "Critic_Score", "Critic_Count", "User_Score", "User_Count",
                        "Year_of_Release")

for (col in numerical_features) {
  if (col %in% colnames(df)) {
    df[[col]] <- as.numeric(df[[col]])
    median_val <- median(df[[col]], na.rm = TRUE)
    na_count <- sum(is.na(df[[col]]))
    df[[col]][is.na(df[[col]])] <- median_val
    if (na_count > 0) {
      cat("Imputed", col, "- NAs:", na_count, "-> Median:", round(median_val, 2), "\n")
    }
  }
}

# Handle outliers using IQR method for target variable
cat("\n--- Handling Outliers ---\n")
handle_outliers <- function(x) {
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR_val <- Q3 - Q1
  lower <- Q1 - 1.5 * IQR_val
  upper <- Q3 + 1.5 * IQR_val
  x[x < lower] <- lower
  x[x > upper] <- upper
  return(x)
}

# Cap outliers in sales columns
sales_cols <- c("NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales")
for (col in sales_cols) {
  original_max <- max(df[[col]], na.rm = TRUE)
  df[[col]] <- handle_outliers(df[[col]])
  new_max <- max(df[[col]], na.rm = TRUE)
  cat(col, "- Original Max:", round(original_max, 2), "-> Capped Max:", round(new_max, 2), "\n")
}

# Encode categorical variables automatically
cat("\n--- Encoding Categorical Variables ---\n")

# Encode Genre as factor
df$Genre <- as.factor(df$Genre)
cat("Genre encoded as factor with", length(levels(df$Genre)), "levels\n")

# Encode Platform - group into categories due to many levels
df$Platform_Category <- case_when(
  df$Platform %in% c("Wii", "WiiU", "DS", "3DS", "GBA", "GC", "N64", "NES", "SNES", "GB") ~ "Nintendo",
  df$Platform %in% c("PS", "PS2", "PS3", "PS4", "PSP", "PSV") ~ "PlayStation",
  df$Platform %in% c("X360", "XB", "XOne") ~ "Xbox",
  df$Platform %in% c("PC") ~ "PC",
  TRUE ~ "Other"
)
df$Platform_Category <- as.factor(df$Platform_Category)
cat("Platform grouped into:", paste(levels(df$Platform_Category), collapse = ", "), "\n")

# Encode Rating
df$Rating <- as.factor(df$Rating)
cat("Rating encoded as factor\n")

# Apply log transformation to reduce skewness in target variable
cat("\n--- Applying Log Transformation to Reduce Skewness ---\n")
cat("Global_Sales Skewness (before):", round(e1071::skewness(df$Global_Sales, na.rm = TRUE), 3), "\n")
df$Global_Sales_Log <- log1p(df$Global_Sales)
cat("Global_Sales_Log Skewness (after):", round(e1071::skewness(df$Global_Sales_Log, na.rm = TRUE), 3), "\n")

# Normalize/Scale numerical variables
cat("\n--- Normalizing Numerical Variables ---\n")
features_to_scale <- c("Critic_Score", "User_Score", "Critic_Count", "User_Count", "Year_of_Release")

for (col in features_to_scale) {
  if (col %in% colnames(df)) {
    df[[col]] <- as.numeric(df[[col]])
    df[[col]][is.na(df[[col]])] <- median(df[[col]], na.rm = TRUE)
    df[[paste0(col, "_scaled")]] <- scale(df[[col]])[,1]
  }
}
cat("Numerical features normalized using z-score standardization\n")

# Feature Engineering: Create new features
cat("\n--- Feature Engineering ---\n")

# Sales ratios
df$NA_Sales_Ratio <- df$NA_Sales / (df$Global_Sales + 0.01)
df$EU_Sales_Ratio <- df$EU_Sales / (df$Global_Sales + 0.01)
df$JP_Sales_Ratio <- df$JP_Sales / (df$Global_Sales + 0.01)

# Score difference (Critic vs User perception)
df$Score_Diff <- df$Critic_Score - (df$User_Score * 10)

# Game age
current_year <- 2016
df$Game_Age <- current_year - df$Year_of_Release

# Total review count
df$Total_Reviews <- df$Critic_Count + df$User_Count

# Average score
df$Avg_Score <- (df$Critic_Score + (df$User_Score * 10)) / 2

cat("Created features: Sales Ratios, Score_Diff, Game_Age, Total_Reviews, Avg_Score\n")

# Select final features for modeling
cat("\n--- Final Feature Selection ---\n")

# For regression, we'll use features that aren't directly derived from target
model_features <- c("Genre", "Platform_Category", "Critic_Score", "User_Score",
                    "Critic_Count", "User_Count", "Year_of_Release", "Rating",
                    "Game_Age", "Total_Reviews", "Avg_Score", "Global_Sales")

df_model <- df %>% 
  select(any_of(model_features)) %>%
  na.omit()

cat("Final dataset shape:", nrow(df_model), "rows,", ncol(df_model), "columns\n")
cat("Features used for prediction:", paste(model_features[-length(model_features)], collapse = ", "), "\n")

# Check final data structure
cat("\nFinal data structure:\n")
str(df_model)

# ----------------------------------------------------------------------------
# D. MODELING - LINEAR REGRESSION
# ----------------------------------------------------------------------------
cat("\n============================================================\n")
cat("D. MODELING - LINEAR REGRESSION\n")
cat("============================================================\n\n")

# Split data into training (70%) and testing (30%) sets
set.seed(123)
trainIndex <- createDataPartition(df_model$Global_Sales, p = 0.7, list = FALSE)
trainData <- df_model[trainIndex, ]
testData <- df_model[-trainIndex, ]

cat("Training set size:", nrow(trainData), "\n")
cat("Testing set size:", nrow(testData), "\n")

# Check target variable distribution in train and test
cat("\nTarget variable (Global_Sales) distribution:\n")
cat("Training - Mean:", round(mean(trainData$Global_Sales), 3), 
    "Median:", round(median(trainData$Global_Sales), 3), "\n")
cat("Testing - Mean:", round(mean(testData$Global_Sales), 3), 
    "Median:", round(median(testData$Global_Sales), 3), "\n")

# Build Linear Regression model
cat("\n--- Building Linear Regression Model ---\n")
model_lm <- lm(Global_Sales ~ Genre + Platform_Category + Critic_Score + User_Score +
                 Critic_Count + User_Count + Year_of_Release + Game_Age + 
                 Total_Reviews + Avg_Score,
               data = trainData)

# Print model summary
cat("\nLinear Regression Model Summary:\n")
print(summary(model_lm))

# Model coefficients
cat("\n--- Model Coefficients ---\n")
coef_df <- data.frame(
  Variable = names(coef(model_lm)),
  Coefficient = coef(model_lm)
)
coef_df <- coef_df[order(abs(coef_df$Coefficient), decreasing = TRUE), ]
print(head(coef_df, 15))

# Visualize coefficients
coef_plot <- coef_df %>%
  filter(Variable != "(Intercept)") %>%
  head(15)

ggplot(coef_plot, aes(x = reorder(Variable, Coefficient), y = Coefficient, fill = Coefficient > 0)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Top 15 Linear Regression Coefficients",
       x = "Variable", y = "Coefficient") +
  theme_minimal() +
  scale_fill_manual(values = c("tomato", "steelblue"), guide = "none")

# Make predictions on test set
pred_lm <- predict(model_lm, newdata = testData)

# Ensure no negative predictions (sales can't be negative)
pred_lm[pred_lm < 0] <- 0

cat("\nSample Predictions:\n")
comparison <- data.frame(
  Actual = round(testData$Global_Sales, 3),
  Predicted = round(pred_lm, 3),
  Difference = round(testData$Global_Sales - pred_lm, 3)
)
print(head(comparison, 10))

# ----------------------------------------------------------------------------
# E. MODEL EVALUATION & INTERPRETATION
# ----------------------------------------------------------------------------
cat("\n============================================================\n")
cat("E. MODEL EVALUATION & INTERPRETATION\n")
cat("============================================================\n\n")

# Calculate evaluation metrics
actual <- testData$Global_Sales
predicted <- pred_lm

# RMSE - Root Mean Squared Error
rmse_val <- sqrt(mean((actual - predicted)^2))
cat("RMSE (Root Mean Squared Error):", round(rmse_val, 4), "million\n")

# MAE - Mean Absolute Error
mae_val <- mean(abs(actual - predicted))
cat("MAE (Mean Absolute Error):", round(mae_val, 4), "million\n")

# R² - Coefficient of Determination
ss_res <- sum((actual - predicted)^2)
ss_tot <- sum((actual - mean(actual))^2)
r_squared <- 1 - (ss_res / ss_tot)
cat("R² (R-Squared):", round(r_squared, 4), "\n")

# Adjusted R²
n <- length(actual)
p <- length(coef(model_lm)) - 1
adj_r_squared <- 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
cat("Adjusted R²:", round(adj_r_squared, 4), "\n")

# Mean Absolute Percentage Error (MAPE)
mape <- mean(abs((actual - predicted) / (actual + 0.01))) * 100
cat("MAPE (Mean Absolute Percentage Error):", round(mape, 2), "%\n")

# Display metrics summary
cat("\n--- Regression Metrics Summary ---\n")
metrics_df <- data.frame(
  Metric = c("RMSE", "MAE", "R²", "Adjusted R²", "MAPE"),
  Value = c(round(rmse_val, 4), round(mae_val, 4), round(r_squared, 4), 
            round(adj_r_squared, 4), paste0(round(mape, 2), "%"))
)
print(metrics_df)

# ============== EVALUATION VISUALIZATIONS ==============

# 1. Actual vs Predicted scatter plot
ggplot(data.frame(Actual = actual, Predicted = predicted), 
       aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5, color = "darkblue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", size = 1) +
  labs(title = "Actual vs Predicted Global Sales",
       subtitle = paste("R² =", round(r_squared, 3), "| RMSE =", round(rmse_val, 3)),
       x = "Actual Global Sales (Millions)", 
       y = "Predicted Global Sales (Millions)") +
  theme_minimal() +
  coord_cartesian(xlim = c(0, max(actual)), ylim = c(0, max(predicted)))

# 2. Residual Plot
residuals <- actual - predicted
ggplot(data.frame(Predicted = predicted, Residuals = residuals), 
       aes(x = Predicted, y = Residuals)) +
  geom_point(alpha = 0.5, color = "darkgreen") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Residual Plot",
       x = "Predicted Values", y = "Residuals") +
  theme_minimal()

# 3. Residual Distribution
ggplot(data.frame(Residuals = residuals), aes(x = Residuals)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  geom_vline(xintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Distribution of Residuals",
       x = "Residuals", y = "Frequency") +
  theme_minimal()

# 4. Q-Q Plot for residuals (normality check)
ggplot(data.frame(Residuals = residuals), aes(sample = Residuals)) +
  stat_qq() +
  stat_qq_line(color = "red") +
  labs(title = "Q-Q Plot of Residuals",
       x = "Theoretical Quantiles", y = "Sample Quantiles") +
  theme_minimal()

# 5. Prediction Error Distribution
ggplot(data.frame(Error = abs(residuals)), aes(x = Error)) +
  geom_histogram(bins = 50, fill = "tomato", color = "white") +
  labs(title = "Distribution of Absolute Prediction Errors",
       x = "Absolute Error (Millions)", y = "Frequency") +
  theme_minimal()

# ----------------------------------------------------------------------------
# MODEL INTERPRETATION & INSIGHTS
# ----------------------------------------------------------------------------
cat("\n============================================================\n")
cat("MODEL INTERPRETATION & INSIGHTS\n")
cat("============================================================\n\n")

cat("KEY FINDINGS:\n")
cat("-------------\n")
cat("1. The Linear Regression model predicts Global Sales based on game features.\n")
cat("2. R² =", round(r_squared, 3), "means the model explains", round(r_squared * 100, 1), 
    "% of variance in sales.\n")
cat("3. Average prediction error (MAE):", round(mae_val, 3), "million units.\n\n")

cat("SIGNIFICANT PREDICTORS:\n")
cat("-----------------------\n")
# Get significant coefficients
model_summary <- summary(model_lm)
sig_coefs <- model_summary$coefficients[model_summary$coefficients[, "Pr(>|t|)"] < 0.05, ]
if (nrow(sig_coefs) > 0) {
  cat("Variables with p-value < 0.05:\n")
  print(round(sig_coefs[order(abs(sig_coefs[, "Estimate"]), decreasing = TRUE), ], 4))
}

cat("\nINTERPRETATION:\n")
cat("---------------\n")
cat("- Critic_Score: Higher critic scores tend to increase global sales\n")
cat("- Platform_Category: Different platforms have varying sales potential\n")
cat("- Genre: Action and Sports games typically have higher sales\n")
cat("- Total_Reviews: More reviews often correlate with higher sales\n")
cat("- Game_Age: Older games may have accumulated more sales over time\n")

cat("\nMODEL LIMITATIONS:\n")
cat("------------------\n")
cat("- Sales data is highly skewed (few blockbusters, many low-sellers)\n")
cat("- Some variance unexplained - other factors like marketing, franchise affect sales\n")
cat("- Linear model may not capture non-linear relationships\n")

cat("\n============================================================\n")
cat("REGRESSION MODEL COMPLETE!\n")
cat("============================================================\n")
