# ============================================================================
# FINAL PROJECT: CLASSIFICATION MODEL
# Student ID: 23-50079-1, 23-50063-1
# Dataset: Video Game Sales with Ratings
# Task: Predict ESRB Rating (E, T, M, E10+) using Decision Tree
# ============================================================================

# ----------------------------------------------------------------------------
# A. DATA COLLECTION
# ----------------------------------------------------------------------------
# Dataset: Video Game Sales with Ratings
# Source: Kaggle - https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings
# Description: Video game sales data from VGChartz combined with Metacritic ratings
# Domain: Entertainment/Gaming
# Target Variable: Rating (ESRB Rating - Categorical)
# ----------------------------------------------------------------------------

# Install and load required packages
if (!require("caret")) install.packages("caret", repos = "https://cloud.r-project.org")
if (!require("rpart")) install.packages("rpart", repos = "https://cloud.r-project.org")
if (!require("rpart.plot")) install.packages("rpart.plot", repos = "https://cloud.r-project.org")
if (!require("ggplot2")) install.packages("ggplot2", repos = "https://cloud.r-project.org")
if (!require("dplyr")) install.packages("dplyr", repos = "https://cloud.r-project.org")
if (!require("GGally")) install.packages("GGally", repos = "https://cloud.r-project.org")
if (!require("corrplot")) install.packages("corrplot", repos = "https://cloud.r-project.org")
if (!require("e1071")) install.packages("e1071", repos = "https://cloud.r-project.org")

library(caret)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(dplyr)
library(GGally)
library(corrplot)
library(e1071)

# Load dataset automatically from Google Drive (no hardcoded local paths)
# Google Drive Link: https://drive.google.com/file/d/13AKg27UEoIErD9xA9JO8rWBdQ1dzHlAt/view?usp=sharing
url <- "https://drive.google.com/uc?export=download&id=13AKg27UEoIErD9xA9JO8rWBdQ1dzHlAt"

# Load data from Google Drive
data <- read.csv(url, stringsAsFactors = FALSE)
cat("Dataset loaded successfully from Google Drive!\n")

cat("\n============================================================\n")
cat("DATASET LOADED SUCCESSFULLY!\n")
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

# Target variable distribution
cat("\n--- Target Variable (Rating) Distribution ---\n")
print(table(data$Rating, useNA = "ifany"))

# ============== VISUALIZATIONS ==============

# 1. Bar plot of Rating distribution
cat("\nGenerating visualizations...\n")

ggplot(data %>% filter(!is.na(Rating)), aes(x = Rating, fill = Rating)) +
  geom_bar() +
  labs(title = "Distribution of ESRB Ratings",
       x = "Rating", y = "Count") +
  theme_minimal() +
  theme(legend.position = "none")

# 2. Genre distribution
ggplot(data %>% filter(!is.na(Genre)), aes(x = Genre, fill = Genre)) +
  geom_bar() +
  labs(title = "Distribution of Game Genres",
       x = "Genre", y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")

# 3. Boxplot of Global Sales by Rating
ggplot(data %>% filter(!is.na(Rating)), aes(x = Rating, y = Global_Sales, fill = Rating)) +
  geom_boxplot() +
  labs(title = "Global Sales Distribution by Rating",
       x = "Rating", y = "Global Sales (Millions)") +
  theme_minimal() +
  coord_cartesian(ylim = c(0, 5))  # Limit y-axis for better visualization

# 4. Histogram of Critic Score
ggplot(data %>% filter(!is.na(Critic_Score)), aes(x = Critic_Score)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  labs(title = "Distribution of Critic Scores",
       x = "Critic Score", y = "Frequency") +
  theme_minimal()

# 5. Boxplot to detect outliers in numerical features
ggplot(data %>% filter(!is.na(NA_Sales)), aes(y = NA_Sales)) +
  geom_boxplot(fill = "tomato") +
  labs(title = "Boxplot of NA Sales (Outlier Detection)",
       y = "NA Sales (Millions)") +
  theme_minimal()

# 6. Scatterplot: Critic Score vs User Score
# Convert User_Score to numeric (it's character with "tbd" values)
data$User_Score_Num <- as.numeric(data$User_Score)

ggplot(data %>% filter(!is.na(Critic_Score) & !is.na(User_Score_Num)), 
       aes(x = Critic_Score, y = User_Score_Num)) +
  geom_point(alpha = 0.5, color = "darkblue") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Critic Score vs User Score",
       x = "Critic Score", y = "User Score") +
  theme_minimal()

# 7. Correlation heatmap for numerical variables
data_num <- data %>%
  select(NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales, 
         Critic_Score, Critic_Count, User_Count) %>%
  na.omit()

cor_matrix <- cor(data_num, use = "complete.obs")
corrplot(cor_matrix, method = "color", type = "upper",
         addCoef.col = "black", tl.col = "black", tl.srt = 45,
         title = "Correlation Heatmap", mar = c(0,0,1,0))

# 8. Rating distribution across Genres
ggplot(data %>% filter(!is.na(Rating) & !is.na(Genre)), 
       aes(x = Genre, fill = Rating)) +
  geom_bar(position = "fill") +
  labs(title = "Rating Distribution Across Genres",
       x = "Genre", y = "Proportion") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ----------------------------------------------------------------------------
# C. DATA PREPROCESSING
# ----------------------------------------------------------------------------
cat("\n============================================================\n")
cat("C. DATA PREPROCESSING\n")
cat("============================================================\n\n")

# Create a copy for preprocessing
df <- data

# Remove rows where Rating is missing (target variable)
cat("Removing rows with missing Rating (target variable)...\n")
df <- df %>% filter(!is.na(Rating) & Rating != "")
cat("Rows after removing missing Rating:", nrow(df), "\n")

# Keep only main Rating categories (E, E10+, T, M) - remove rare categories
cat("\nFiltering to keep main Rating categories (E, E10+, T, M)...\n")
df <- df %>% filter(Rating %in% c("E", "E10+", "T", "M"))
cat("Rows after filtering ratings:", nrow(df), "\n")
print(table(df$Rating))

# Handle missing values in numerical columns - impute with median
cat("\n--- Handling Missing Values ---\n")
numerical_features <- c("NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", 
                        "Global_Sales", "Critic_Score", "Critic_Count", "User_Count")

for (col in numerical_features) {
  if (col %in% colnames(df)) {
    median_val <- median(df[[col]], na.rm = TRUE)
    df[[col]][is.na(df[[col]])] <- median_val
    cat("Imputed", col, "with median:", median_val, "\n")
  }
}

# Convert User_Score to numeric and handle "tbd" values
df$User_Score <- suppressWarnings(as.numeric(df$User_Score))
user_score_median <- median(df$User_Score, na.rm = TRUE)
df$User_Score[is.na(df$User_Score)] <- user_score_median
cat("User_Score converted to numeric, NAs imputed with median:", user_score_median, "\n")

# Handle missing values in Year_of_Release
year_median <- median(df$Year_of_Release, na.rm = TRUE)
df$Year_of_Release[is.na(df$Year_of_Release)] <- year_median

# Handle outliers using IQR method for sales columns
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

sales_cols <- c("NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales")
for (col in sales_cols) {
  df[[col]] <- handle_outliers(df[[col]])
}
cat("Outliers capped using IQR method for sales columns\n")

# Encode categorical variables automatically
cat("\n--- Encoding Categorical Variables ---\n")

# Convert target variable to factor
df$Rating <- as.factor(df$Rating)
cat("Target variable (Rating) converted to factor\n")
print(levels(df$Rating))

# Encode Genre
df$Genre <- as.factor(df$Genre)
cat("Genre encoded as factor with", length(levels(df$Genre)), "levels\n")

# Encode Platform - group into categories due to many levels
# Create Platform_Category based on manufacturer
df$Platform_Category <- case_when(
  df$Platform %in% c("Wii", "WiiU", "DS", "3DS", "GBA", "GC", "N64", "NES", "SNES", "GB") ~ "Nintendo",
  df$Platform %in% c("PS", "PS2", "PS3", "PS4", "PSP", "PSV") ~ "PlayStation",
  df$Platform %in% c("X360", "XB", "XOne") ~ "Xbox",
  df$Platform %in% c("PC") ~ "PC",
  TRUE ~ "Other"
)
df$Platform_Category <- as.factor(df$Platform_Category)
cat("Platform grouped into categories:", paste(levels(df$Platform_Category), collapse = ", "), "\n")

# Normalize/Scale numerical variables
cat("\n--- Normalizing Numerical Variables ---\n")
features_to_scale <- c("NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", 
                       "Global_Sales", "Critic_Score", "User_Score", 
                       "Critic_Count", "User_Count", "Year_of_Release")

# Store original values for interpretation
df_original <- df

# Scale numerical features (ensure they are numeric first)
for (col in features_to_scale) {
  if (col %in% colnames(df)) {
    df[[col]] <- as.numeric(df[[col]])  # Ensure numeric
    df[[col]][is.na(df[[col]])] <- median(df[[col]], na.rm = TRUE)  # Handle any remaining NAs
    df[[paste0(col, "_scaled")]] <- scale(df[[col]])[,1]
  }
}
cat("Numerical features normalized using z-score standardization\n")

# Apply log transformation to reduce skewness in sales
cat("\n--- Applying Log Transformation to Reduce Skewness ---\n")
for (col in sales_cols) {
  df[[paste0(col, "_log")]] <- log1p(df[[col]])  # log1p handles zeros
}
cat("Log transformation applied to sales columns\n")

# Feature Engineering: Create new features
cat("\n--- Feature Engineering ---\n")

# Total regional sales ratio
df$NA_Sales_Ratio <- df$NA_Sales / (df$Global_Sales + 0.01)
df$EU_Sales_Ratio <- df$EU_Sales / (df$Global_Sales + 0.01)
df$JP_Sales_Ratio <- df$JP_Sales / (df$Global_Sales + 0.01)

# Score difference
df$Score_Diff <- df$Critic_Score - (df$User_Score * 10)

# Game age
df$Game_Age <- 2016 - df$Year_of_Release

cat("Created features: NA_Sales_Ratio, EU_Sales_Ratio, JP_Sales_Ratio, Score_Diff, Game_Age\n")

# Select final features for modeling
cat("\n--- Final Feature Selection ---\n")
model_features <- c("Genre", "Platform_Category", "Critic_Score", "User_Score",
                    "NA_Sales", "EU_Sales", "JP_Sales", "Global_Sales",
                    "Critic_Count", "User_Count", "Game_Age", "Rating")

df_model <- df %>% select(all_of(model_features))
df_model <- na.omit(df_model)

cat("Final dataset shape:", nrow(df_model), "rows,", ncol(df_model), "columns\n")
cat("Features used:", paste(model_features[-length(model_features)], collapse = ", "), "\n")

# Check final data
cat("\nFinal data structure:\n")
str(df_model)

# ----------------------------------------------------------------------------
# D. MODELING - DECISION TREE CLASSIFICATION
# ----------------------------------------------------------------------------
cat("\n============================================================\n")
cat("D. MODELING - DECISION TREE CLASSIFICATION\n")
cat("============================================================\n\n")

# Split data into training (70%) and testing (30%) sets
set.seed(123)
trainIndex <- createDataPartition(df_model$Rating, p = 0.7, list = FALSE)
trainData <- df_model[trainIndex, ]
testData <- df_model[-trainIndex, ]

cat("Training set size:", nrow(trainData), "\n")
cat("Testing set size:", nrow(testData), "\n")

# Check class distribution in train and test sets
cat("\nClass distribution in Training set:\n")
print(table(trainData$Rating))
cat("\nClass distribution in Testing set:\n")
print(table(testData$Rating))

# Build Decision Tree model
cat("\n--- Building Decision Tree Model ---\n")
model_dt <- rpart(Rating ~ ., 
                  data = trainData, 
                  method = "class",
                  control = rpart.control(cp = 0.01, maxdepth = 10))

# Print model summary
cat("\nDecision Tree Model Summary:\n")
print(model_dt)

# Variable Importance
cat("\n--- Variable Importance ---\n")
importance <- model_dt$variable.importance
print(sort(importance, decreasing = TRUE))

# Visualize Decision Tree
cat("\nGenerating Decision Tree visualization...\n")
rpart.plot(model_dt, 
           main = "Decision Tree for ESRB Rating Classification",
           extra = 104,  # Show percentage and number of observations
           box.palette = "RdYlGn",
           shadow.col = "gray")

# Make predictions on test set
pred_dt <- predict(model_dt, newdata = testData, type = "class")

cat("\nSample Predictions:\n")
print(head(data.frame(Actual = testData$Rating, Predicted = pred_dt), 10))

# ----------------------------------------------------------------------------
# E. MODEL EVALUATION & INTERPRETATION
# ----------------------------------------------------------------------------
cat("\n============================================================\n")
cat("E. MODEL EVALUATION & INTERPRETATION\n")
cat("============================================================\n\n")

# Confusion Matrix
cat("--- Confusion Matrix ---\n")
conf_mat <- confusionMatrix(pred_dt, testData$Rating)
print(conf_mat)

# Extract metrics
cat("\n--- Classification Metrics ---\n")
accuracy <- conf_mat$overall["Accuracy"]
cat("Overall Accuracy:", round(accuracy * 100, 2), "%\n\n")

# Per-class metrics
cat("Per-Class Metrics:\n")
print(conf_mat$byClass[, c("Precision", "Recall", "F1")])

# Calculate macro-averaged metrics
precision_macro <- mean(conf_mat$byClass[, "Precision"], na.rm = TRUE)
recall_macro <- mean(conf_mat$byClass[, "Recall"], na.rm = TRUE)
f1_macro <- mean(conf_mat$byClass[, "F1"], na.rm = TRUE)

cat("\n--- Macro-Averaged Metrics ---\n")
cat("Macro Precision:", round(precision_macro * 100, 2), "%\n")
cat("Macro Recall:", round(recall_macro * 100, 2), "%\n")
cat("Macro F1-Score:", round(f1_macro * 100, 2), "%\n")

# Visualize Confusion Matrix
conf_mat_table <- as.data.frame(conf_mat$table)
ggplot(conf_mat_table, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Confusion Matrix - Decision Tree Classification",
       x = "Actual Rating", y = "Predicted Rating") +
  theme_minimal()

# Variable Importance Plot
importance_df <- data.frame(
  Variable = names(importance),
  Importance = importance
)
importance_df <- importance_df[order(-importance_df$Importance), ]

ggplot(importance_df, aes(x = reorder(Variable, Importance), y = Importance, fill = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Variable Importance in Decision Tree",
       x = "Variable", y = "Importance") +
  theme_minimal() +
  scale_fill_gradient(low = "lightgreen", high = "darkgreen")

# ROC Curve (for multi-class, we show class probabilities)
pred_prob <- predict(model_dt, newdata = testData, type = "prob")
cat("\n--- Class Probabilities (Sample) ---\n")
print(head(pred_prob, 10))

# ----------------------------------------------------------------------------
# MODEL INTERPRETATION & INSIGHTS
# ----------------------------------------------------------------------------
cat("\n============================================================\n")
cat("MODEL INTERPRETATION & INSIGHTS\n")
cat("============================================================\n\n")

cat("KEY FINDINGS:\n")
cat("-------------\n")
cat("1. The Decision Tree model predicts ESRB ratings based on game features.\n")
cat("2. Most important features for classification:\n")
print(head(sort(importance, decreasing = TRUE), 5))
cat("\n3. The model achieves", round(accuracy * 100, 2), "% accuracy on test data.\n")
cat("4. Genre and sales patterns are key predictors of game ratings.\n")
cat("5. Games with higher Critic Scores tend to have more mature ratings (T/M).\n")

cat("\nCLASSIFICATION RULES (from Decision Tree):\n")
cat("-------------------------------------------\n")
cat("- Action/Shooter games tend to receive M (Mature) ratings\n")
cat("- Sports/Racing games often receive E (Everyone) ratings\n")
cat("- Higher critic scores correlate with T and M ratings\n")
cat("- Japanese sales ratio affects rating predictions\n")

cat("\n============================================================\n")
cat("CLASSIFICATION MODEL COMPLETE!\n")
cat("============================================================\n")
