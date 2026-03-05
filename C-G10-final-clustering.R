# ============================================================================
# FINAL PROJECT: CLUSTERING MODEL
# Student ID: 23-50079-1, 23-50063-1
# Dataset: Video Game Sales with Ratings
# Task: Cluster Similar Video Games using K-Means Clustering
# ============================================================================

# ----------------------------------------------------------------------------
# A. DATA COLLECTION
# ----------------------------------------------------------------------------
# Dataset: Video Game Sales with Ratings
# Source: Kaggle - https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings
# Description: Video game sales data from VGChartz combined with Metacritic ratings
# Domain: Entertainment/Gaming
# Task: Unsupervised clustering to group similar games based on sales and scores

if (!require("ggplot2")) install.packages("ggplot2", repos = "https://cloud.r-project.org")
if (!require("dplyr")) install.packages("dplyr", repos = "https://cloud.r-project.org")
if (!require("cluster")) install.packages("cluster", repos = "https://cloud.r-project.org")
if (!require("factoextra")) install.packages("factoextra", repos = "https://cloud.r-project.org")
if (!require("corrplot")) install.packages("corrplot", repos = "https://cloud.r-project.org")
if (!require("tidyr")) install.packages("tidyr", repos = "https://cloud.r-project.org")

library(ggplot2)
library(dplyr)
library(cluster)
library(factoextra)
library(corrplot)
library(tidyr)

url <- "https://drive.google.com/uc?export=download&id=13AKg27UEoIErD9xA9JO8rWBdQ1dzHlAt"
data <- read.csv(url, stringsAsFactors = FALSE)
cat("Dataset loaded successfully from Google Drive!\n")

cat("CLUSTERING MODEL - Grouping Similar Video Games\n")

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
cat("\nGenerating EDA visualizations...\n")

# 1. Distribution of Global Sales
p1 <- ggplot(data, aes(x = Global_Sales)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  labs(title = "Distribution of Global Sales",
       x = "Global Sales (Millions)", y = "Frequency") +
  theme_minimal()
print(p1)

# 2. Distribution of Critic Scores
p2 <- ggplot(data %>% filter(!is.na(Critic_Score)), aes(x = Critic_Score)) +
  geom_histogram(bins = 30, fill = "darkgreen", color = "white") +
  labs(title = "Distribution of Critic Scores",
       x = "Critic Score", y = "Frequency") +
  theme_minimal()
print(p2)

# 3. Boxplot of Sales by Genre
p3 <- ggplot(data %>% filter(!is.na(Genre)), aes(x = Genre, y = Global_Sales, fill = Genre)) +
  geom_boxplot() +
  labs(title = "Global Sales Distribution by Genre",
       x = "Genre", y = "Global Sales (Millions)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none") +
  coord_cartesian(ylim = c(0, 3))
print(p3)

# 4. Boxplot to detect outliers
p4 <- ggplot(data, aes(y = Global_Sales)) +
  geom_boxplot(fill = "tomato") +
  labs(title = "Boxplot of Global Sales (Outlier Detection)",
       y = "Global Sales (Millions)") +
  theme_minimal()
print(p4)

# 5. Scatterplot: NA Sales vs EU Sales
p5 <- ggplot(data, aes(x = NA_Sales, y = EU_Sales)) +
  geom_point(alpha = 0.3, color = "darkblue") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "NA Sales vs EU Sales",
       x = "NA Sales (Millions)", y = "EU Sales (Millions)") +
  theme_minimal() +
  coord_cartesian(xlim = c(0, 5), ylim = c(0, 3))
print(p5)

# 6. Scatterplot: Critic Score vs Global Sales
data$User_Score_Num <- suppressWarnings(as.numeric(data$User_Score))

p6 <- ggplot(data %>% filter(!is.na(Critic_Score)), 
       aes(x = Critic_Score, y = Global_Sales)) +
  geom_point(alpha = 0.3, color = "purple") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Critic Score vs Global Sales",
       x = "Critic Score", y = "Global Sales (Millions)") +
  theme_minimal() +
  coord_cartesian(ylim = c(0, 5))
print(p6)

# 7. Correlation heatmap for numerical variables
data_num <- data %>%
  select(NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales, 
         Critic_Score, Critic_Count, User_Count) %>%
  mutate(across(everything(), as.numeric)) %>%
  na.omit()

cor_matrix <- cor(data_num, use = "complete.obs")
corrplot(cor_matrix, method = "color", type = "upper",
         addCoef.col = "black", tl.col = "black", tl.srt = 45,
         title = "Correlation Heatmap - Numerical Variables", mar = c(0,0,1,0))

# 8. Regional Sales Comparison
regional_sales <- data %>%
  summarise(NA_Sales = sum(NA_Sales, na.rm = TRUE),
            EU_Sales = sum(EU_Sales, na.rm = TRUE),
            JP_Sales = sum(JP_Sales, na.rm = TRUE),
            Other_Sales = sum(Other_Sales, na.rm = TRUE)) %>%
  pivot_longer(cols = everything(), names_to = "Region", values_to = "Sales")

p8 <- ggplot(regional_sales, aes(x = Region, y = Sales, fill = Region)) +
  geom_bar(stat = "identity") +
  labs(title = "Total Sales by Region",
       x = "Region", y = "Total Sales (Millions)") +
  theme_minimal() +
  theme(legend.position = "none")
print(p8)

# ----------------------------------------------------------------------------
# C. DATA PREPROCESSING
# ----------------------------------------------------------------------------
cat("\n============================================================\n")
cat("C. DATA PREPROCESSING\n")
cat("============================================================\n\n")

# Create a copy for preprocessing
df <- data

# Remove rows with too many missing values
cat("Removing rows with missing critical values...\n")
df <- df %>% filter(!is.na(Global_Sales) & Global_Sales > 0)
cat("Rows after filtering:", nrow(df), "\n")

# Handle missing values in numerical columns - impute with median
cat("\n--- Handling Missing Values ---\n")

# Convert User_Score to numeric
df$User_Score <- suppressWarnings(as.numeric(df$User_Score))

numerical_features <- c("NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", 
                        "Global_Sales", "Critic_Score", "Critic_Count", 
                        "User_Score", "User_Count", "Year_of_Release")

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

# Handle outliers using IQR method
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
  original_max <- max(df[[col]], na.rm = TRUE)
  df[[col]] <- handle_outliers(df[[col]])
  new_max <- max(df[[col]], na.rm = TRUE)
  cat(col, "- Original Max:", round(original_max, 2), "-> Capped Max:", round(new_max, 2), "\n")
}

# Encode categorical variables
cat("\n--- Encoding Categorical Variables ---\n")

df$Genre <- as.factor(df$Genre)
cat("Genre encoded as factor with", length(levels(df$Genre)), "levels\n")

df$Platform_Category <- case_when(
  df$Platform %in% c("Wii", "WiiU", "DS", "3DS", "GBA", "GC", "N64", "NES", "SNES", "GB") ~ "Nintendo",
  df$Platform %in% c("PS", "PS2", "PS3", "PS4", "PSP", "PSV") ~ "PlayStation",
  df$Platform %in% c("X360", "XB", "XOne") ~ "Xbox",
  df$Platform %in% c("PC") ~ "PC",
  TRUE ~ "Other"
)
df$Platform_Category <- as.factor(df$Platform_Category)
cat("Platform grouped into:", paste(levels(df$Platform_Category), collapse = ", "), "\n")

# Apply log transformation to reduce skewness
cat("\n--- Applying Log Transformation to Reduce Skewness ---\n")
for (col in sales_cols) {
  df[[paste0(col, "_log")]] <- log1p(df[[col]])
}
cat("Log transformation applied to sales columns\n")

# Feature Engineering
cat("\n--- Feature Engineering ---\n")

# Sales ratios (regional contribution)
df$NA_Sales_Ratio <- df$NA_Sales / (df$Global_Sales + 0.01)
df$EU_Sales_Ratio <- df$EU_Sales / (df$Global_Sales + 0.01)
df$JP_Sales_Ratio <- df$JP_Sales / (df$Global_Sales + 0.01)

# Score metrics
df$Avg_Score <- (df$Critic_Score + (df$User_Score * 10)) / 2
df$Score_Diff <- df$Critic_Score - (df$User_Score * 10)

# Game age
df$Game_Age <- 2016 - df$Year_of_Release

cat("Created features: Sales Ratios, Avg_Score, Score_Diff, Game_Age\n")

# Select features for clustering (only numerical features)
cat("\n--- Selecting Features for Clustering ---\n")

# For K-Means, we use numerical features that represent game characteristics
clustering_features <- c("NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales",
                         "Critic_Score", "User_Score", "Critic_Count", "User_Count")

df_cluster <- df %>%
  select(all_of(clustering_features)) %>%
  na.omit()

cat("Features for clustering:", paste(clustering_features, collapse = ", "), "\n")
cat("Rows for clustering:", nrow(df_cluster), "\n")

# Normalize/Scale numerical variables (CRITICAL for K-Means)
cat("\n--- Normalizing Features for K-Means ---\n")
df_scaled <- scale(df_cluster)
cat("All features normalized using z-score standardization\n")
cat("Scaled data dimensions:", nrow(df_scaled), "rows x", ncol(df_scaled), "columns\n")

# Check scaled data summary
cat("\nScaled data summary (should have mean~0, sd~1):\n")
print(round(colMeans(df_scaled), 4))

# ----------------------------------------------------------------------------
# D. MODELING - K-MEANS CLUSTERING
# ----------------------------------------------------------------------------
cat("\n============================================================\n")
cat("D. MODELING - K-MEANS CLUSTERING\n")
cat("============================================================\n\n")

# Determine optimal number of clusters using Elbow Method
cat("--- Finding Optimal Number of Clusters ---\n")

# Elbow Method
set.seed(123)
wss <- sapply(1:10, function(k) {
  kmeans(df_scaled, centers = k, nstart = 25, iter.max = 100)$tot.withinss
})

# Plot Elbow curve
elbow_df <- data.frame(k = 1:10, WSS = wss)
p_elbow <- ggplot(elbow_df, aes(x = k, y = WSS)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(color = "darkblue", size = 3) +
  labs(title = "Elbow Method for Optimal K",
       x = "Number of Clusters (K)", y = "Within-Cluster Sum of Squares (WSS)") +
  scale_x_continuous(breaks = 1:10) +
  theme_minimal()
print(p_elbow)

cat("Elbow plot generated - look for the 'elbow' point\n")

# Silhouette Method for optimal K
cat("\n--- Silhouette Analysis for Optimal K ---\n")

# Sample data for faster silhouette computation (if dataset is large)
if (nrow(df_scaled) > 5000) {
  set.seed(123)
  sample_idx <- sample(1:nrow(df_scaled), 5000)
  df_scaled_sample <- df_scaled[sample_idx, ]
} else {
  df_scaled_sample <- df_scaled
}

silhouette_scores <- sapply(2:10, function(k) {
  km <- kmeans(df_scaled_sample, centers = k, nstart = 25, iter.max = 100)
  ss <- silhouette(km$cluster, dist(df_scaled_sample))
  mean(ss[, 3])
})

sil_df <- data.frame(k = 2:10, Silhouette = silhouette_scores)
p_sil <- ggplot(sil_df, aes(x = k, y = Silhouette)) +
  geom_line(color = "darkgreen", linewidth = 1) +
  geom_point(color = "green", size = 3) +
  labs(title = "Silhouette Score for Different K Values",
       x = "Number of Clusters (K)", y = "Average Silhouette Score") +
  scale_x_continuous(breaks = 2:10) +
  theme_minimal()
print(p_sil)

optimal_k <- sil_df$k[which.max(sil_df$Silhouette)]
cat("Optimal K based on Silhouette Score:", optimal_k, "\n")
cat("Maximum Silhouette Score:", round(max(silhouette_scores), 4), "\n")

# Build K-Means model with optimal K (using K=4 as a good balance)
cat("\n--- Building K-Means Model ---\n")
K <- 4  # Can be adjusted based on elbow/silhouette analysis
cat("Using K =", K, "clusters\n")

set.seed(123)
km_model <- kmeans(df_scaled, centers = K, nstart = 25, iter.max = 100)

# Print model summary
cat("\nK-Means Model Summary:\n")
cat("Cluster sizes:", km_model$size, "\n")
cat("Total WSS:", round(km_model$tot.withinss, 2), "\n")
cat("Between SS / Total SS:", round(km_model$betweenss / km_model$totss * 100, 2), "%\n")

# Cluster centers (scaled values)
cat("\n--- Cluster Centers (Scaled) ---\n")
print(round(km_model$centers, 3))

# Add cluster assignments to data
df_cluster$Cluster <- as.factor(km_model$cluster)

# ----------------------------------------------------------------------------
# E. MODEL EVALUATION & INTERPRETATION
# ----------------------------------------------------------------------------
cat("\n============================================================\n")
cat("E. MODEL EVALUATION & INTERPRETATION\n")
cat("============================================================\n\n")

# Calculate Silhouette Score
cat("--- Silhouette Score ---\n")

# Use sampled data if too large for silhouette computation
if (nrow(df_scaled) > 5000) {
  set.seed(123)
  sample_idx <- sample(1:nrow(df_scaled), 5000)
  df_scaled_eval <- df_scaled[sample_idx, ]
  cluster_eval <- km_model$cluster[sample_idx]
} else {
  df_scaled_eval <- df_scaled
  cluster_eval <- km_model$cluster
}

sil <- silhouette(cluster_eval, dist(df_scaled_eval))
avg_sil <- mean(sil[, 3])
cat("Average Silhouette Score:", round(avg_sil, 4), "\n")

# Interpret silhouette score
if (avg_sil > 0.5) {
  cat("Interpretation: Strong cluster structure\n")
} else if (avg_sil > 0.25) {
  cat("Interpretation: Reasonable cluster structure\n")
} else {
  cat("Interpretation: Weak cluster structure, clusters may overlap\n")
}

# Silhouette plot
p_sil_plot <- fviz_silhouette(sil, palette = "jco") +
  labs(title = paste("Silhouette Plot (Avg Width =", round(avg_sil, 3), ")")) +
  theme_minimal()
print(p_sil_plot)

# ============== CLUSTER VISUALIZATIONS ==============
cat("\n--- Generating Cluster Visualizations ---\n")

# 1. Cluster Visualization using PCA (2D projection)
p_cluster <- fviz_cluster(km_model, data = df_scaled,
             palette = "jco",
             geom = "point",
             ellipse.type = "convex",
             ggtheme = theme_minimal(),
             main = "K-Means Clustering - PCA Visualization")
print(p_cluster)

# 2. Cluster sizes bar plot
cluster_sizes <- data.frame(
  Cluster = factor(1:K),
  Size = km_model$size
)

p_sizes <- ggplot(cluster_sizes, aes(x = Cluster, y = Size, fill = Cluster)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = Size), vjust = -0.5) +
  labs(title = "Number of Games in Each Cluster",
       x = "Cluster", y = "Number of Games") +
  theme_minimal() +
  theme(legend.position = "none")
print(p_sizes)

# 3. Cluster centers visualization (radar/parallel coordinates)
centers_df <- as.data.frame(km_model$centers)
centers_df$Cluster <- factor(1:K)
centers_long <- centers_df %>%
  pivot_longer(cols = -Cluster, names_to = "Feature", values_to = "Value")

p_centers <- ggplot(centers_long, aes(x = Feature, y = Value, fill = Cluster, group = Cluster)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Cluster Centers by Feature",
       x = "Feature", y = "Scaled Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_brewer(palette = "Set1")
print(p_centers)

# 4. Boxplots of features by cluster
df_cluster_long <- df_cluster %>%
  pivot_longer(cols = -Cluster, names_to = "Feature", values_to = "Value")

p_boxplots <- ggplot(df_cluster_long, aes(x = Cluster, y = Value, fill = Cluster)) +
  geom_boxplot() +
  facet_wrap(~Feature, scales = "free_y") +
  labs(title = "Feature Distributions by Cluster",
       x = "Cluster", y = "Value") +
  theme_minimal() +
  theme(legend.position = "none")
print(p_boxplots)

# 5. Scatter plot: NA_Sales vs EU_Sales colored by cluster
p_scatter1 <- ggplot(df_cluster, aes(x = NA_Sales, y = EU_Sales, color = Cluster)) +
  geom_point(alpha = 0.6) +
  labs(title = "NA Sales vs EU Sales by Cluster",
       x = "NA Sales (Millions)", y = "EU Sales (Millions)") +
  theme_minimal() +
  scale_color_brewer(palette = "Set1")
print(p_scatter1)

# 6. Scatter plot: Critic Score vs Global Sales colored by cluster
df_cluster$Global_Sales <- df_cluster$NA_Sales + df_cluster$EU_Sales + 
                            df_cluster$JP_Sales + df_cluster$Other_Sales

p_scatter2 <- ggplot(df_cluster, aes(x = Critic_Score, y = Global_Sales, color = Cluster)) +
  geom_point(alpha = 0.6) +
  labs(title = "Critic Score vs Global Sales by Cluster",
       x = "Critic Score", y = "Global Sales (Millions)") +
  theme_minimal() +
  scale_color_brewer(palette = "Set1")
print(p_scatter2)

# ----------------------------------------------------------------------------
# CLUSTER PROFILING & INSIGHTS
# ----------------------------------------------------------------------------
cat("\n============================================================\n")
cat("CLUSTER PROFILING & INSIGHTS\n")
cat("============================================================\n\n")

# Calculate cluster statistics (original scale)
cat("--- Cluster Profiles (Mean Values) ---\n")
cluster_profile <- df_cluster %>%
  group_by(Cluster) %>%
  summarise(
    Count = n(),
    Avg_NA_Sales = round(mean(NA_Sales), 3),
    Avg_EU_Sales = round(mean(EU_Sales), 3),
    Avg_JP_Sales = round(mean(JP_Sales), 3),
    Avg_Critic_Score = round(mean(Critic_Score), 1),
    Avg_User_Score = round(mean(User_Score), 2),
    Avg_Reviews = round(mean(Critic_Count + User_Count), 0)
  )

print(cluster_profile)

# Interpretation of clusters
cat("\n--- Cluster Interpretation ---\n")
cat("\nBased on cluster centers, the games are grouped as:\n\n")

for (i in 1:K) {
  center <- km_model$centers[i, ]
  cat("CLUSTER", i, "(", km_model$size[i], "games):\n")
  
  # Determine cluster characteristics
  if (center["NA_Sales"] > 0.5 && center["EU_Sales"] > 0.5) {
    cat("  -> High-performing games with strong Western sales\n")
  } else if (center["JP_Sales"] > 0.5) {
    cat("  -> Japan-focused games with strong JP market sales\n")
  } else if (center["Critic_Score"] > 0.5) {
    cat("  -> Well-reviewed games with high critic scores\n")
  } else if (all(center[c("NA_Sales", "EU_Sales", "JP_Sales")] < 0)) {
    cat("  -> Low-performing/Niche games with below-average sales\n")
  } else {
    cat("  -> Average-performing games with moderate sales\n")
  }
  cat("\n")
}

cat("\nKEY FINDINGS:\n")
cat("-------------\n")
cat("1. K-Means successfully grouped", nrow(df_cluster), "games into", K, "clusters\n")
cat("2. Silhouette Score:", round(avg_sil, 3), "- indicates", 
    ifelse(avg_sil > 0.25, "reasonable", "weak"), "cluster separation\n")
cat("3. Clusters reveal different game market segments:\n")
cat("   - Some clusters represent blockbuster/AAA games\n")
cat("   - Others represent niche/regional market games\n")
cat("   - Sales patterns differ significantly across regions\n")
cat("4. Critic scores and review counts help differentiate game quality tiers\n")

cat("\nBUSINESS INSIGHTS:\n")
cat("------------------\n")
cat("- Cluster analysis helps identify market segments for targeted marketing\n")
cat("- Games with similar profiles can be recommended together\n")
cat("- Regional sales patterns suggest localization strategies\n")
cat("- High-review clusters may indicate franchise/sequel opportunities\n")

cat("\n============================================================\n")
cat("CLUSTERING MODEL COMPLETE!\n")
cat("============================================================\n")
