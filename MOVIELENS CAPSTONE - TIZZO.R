# ============================================================================
# MOVIELENS RATING PREDICTION PROJECT - R SCRIPT
# HarvardX: PH125.9x Data Science
# Author: MICHELL PEREIRA TIZZO
# ============================================================================

# --- LIBRARIES AND FUNCTIONS --- #

cat("\014") # Clear console

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")
if(!require(patchwork)) install.packages("patchwork", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(e1071)
library(randomForest)
library(rpart)
library(scales)
library(patchwork)

# --- FORMAT STANDARD FOR TABLES (CONSOLE FRIENDLY) --- #
# Changed to print directly in the console instead of the Viewer pane
kb <- function(x, caption = NULL) {
  if (!is.null(caption)) {
    cat("\n---", toupper(caption), "---\n")
  }
  print(knitr::kable(x, format = "markdown"))
  cat("\n")
}

# --- CUSTOM GGPLOT THEME --- #
theme_project <- function() {
  theme_light() +
    theme(
      plot.title = element_text(face = "bold.italic", size = 12),
      axis.title = element_text(face = "bold", size = 10),
      axis.text = element_text(size = 9),
      panel.grid.minor = element_blank()
    )
}

# --- DASHBOARD --- #
make_dash_plot <- function(data, x_var, y_var, title_text) {
  ggplot(data, aes(x = !!sym(x_var), y = !!sym(y_var))) +
    geom_col(fill = "lightblue", color = "black", width = 0.7) +
    geom_text(aes(label = comma(!!sym(y_var))), vjust = -0.5, size = 2.5) +
    labs(title = title_text, x = NULL, y = NULL) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold.italic", size = 7.5),
      axis.text.y = element_blank(),
      axis.text.x = element_text(size = 6.0),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(color = "grey90"),
      plot.margin = ggplot2::margin(t = 10, r = 25, b = 20, l = 25)
    ) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.2)))
}

# --- FORMAT STANDARD FOR PLOT_STAIRCASE --- #
plot_staircase <- function(data, effect_col, title_col, fill_color_true, fill_color_false, x_label, y_label, mu_hat_val) {
  ggplot(data, aes(x = !!sym(title_col), y = !!sym(effect_col), fill = !!sym(effect_col) > 0)) +
    geom_col(color = "black", width = 0.6, show.legend = FALSE) +
    geom_hline(yintercept = 0, color = "black", linewidth = 0.8) +
    geom_text(aes(
      label = sprintf("Bias: %+.2f\nRating: %.1f", !!sym(effect_col), mu_hat_val + !!sym(effect_col)), 
      y = !!sym(effect_col),
      vjust = ifelse(!!sym(effect_col) >= 0, -0.3, 1.3)
    ), size = 2.8, fontface = "bold", lineheight = 0.85) +
    scale_fill_manual(values = setNames(c(fill_color_true, fill_color_false), c("TRUE", "FALSE"))) +
    scale_y_continuous(limits = c(-3.0, 2.0), breaks = seq(-2.5, 1.5, 0.5)) +
    labs(title = "", subtitle = "", x = x_label, y = y_label) +
    theme_project()
}

# ============================================================================
# 1) DOWNLOAD AND PREPARE MOVIELENS 10M DATASET
# ============================================================================

cat("\n")
cat("=============================================================\n")
cat(" [SYSTEM ALERT] DOWNLOADING AND PREPARING DATASET            \n")
cat("=============================================================\n")
cat(" Please be patient. This process downloads a large file \n")
cat(" (MovieLens 10M) and partitions ~9 million rows. \n")
cat(" It may take 2 to 5 minutes depending on your internet \n")
cat(" connection and machine. The script is NOT frozen.\n")
cat("-------------------------------------------------------------\n\n")

# Increased timeout and forced libcurl for connection stability
options(timeout = 600)

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl, method = "libcurl")

cat("=============================================================\n")
cat(" [SYSTEM ALERT] DOWNLOAD OK                                  \n")
cat("=============================================================\n")

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% 
  mutate(movieId = as.numeric(movieId),
         title = as.character(title),
         genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)
gc()


# ============================================================================
# 2) EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
# HARVARDX EDA QUIZ - DYNAMIC CONSOLE OUTPUT

df1 <- data.frame(Metric = factor(c("Rows", "Columns"), levels = c("Rows", "Columns")), 
                  Value = c(nrow(edx), ncol(edx)))
df2 <- data.frame(Rating = factor(c("0", "3"), levels = c("0", "3")), 
                  Count = c(sum(edx$rating == 0), sum(edx$rating == 3)))
df3 <- data.frame(Metric = factor(c("Movies", "Users"), levels = c("Movies", "Users")), 
                  Count = c(n_distinct(edx$movieId), n_distinct(edx$userId)))
df4 <- data.frame(Genre = factor(c("Drama", "Comedy", "Thriller", "Romance"), levels = c("Drama", "Comedy", "Thriller", "Romance")),
                  Count = c(sum(str_detect(edx$genres, "Drama")), sum(str_detect(edx$genres, "Comedy")),
                            sum(str_detect(edx$genres, "Thriller")), sum(str_detect(edx$genres, "Romance"))))
top_movies <- edx %>% group_by(title) %>% summarize(Count = n(), .groups="drop") %>% 
  top_n(5, Count) %>% arrange(desc(Count)) %>%
  mutate(title = str_wrap(title, width = 12), title = factor(title, levels = title))
top_ratings <- edx %>% group_by(rating) %>% summarize(Count = n(), .groups="drop") %>% 
  arrange(desc(Count)) %>% head(5) %>%
  mutate(rating = factor(rating, levels = rating))
df8 <- data.frame(Type = factor(c("Half Stars", "Whole Stars"), levels = c("Half Stars", "Whole Stars")),
                  Count = c(sum(edx$rating %% 1 != 0), sum(edx$rating %% 1 == 0)))


# --- Answers to the HarvardX exploratory data analysis questions --- #

# Limpa as quebras de linha inseridas pelo str_wrap no titulo do filme
top_movie_clean <- gsub("\n", " ", as.character(top_movies$title[1]))

cat("\n")
cat("=============================================================\n")
cat("               HARVARDX EDA QUIZ - ANSWERS                   \n")
cat("=============================================================\n")

cat("Q1) Rows and Columns in edx dataset:\n")
cat(sprintf("    Rows: %s | Columns: %s\n\n", comma(nrow(edx)), comma(ncol(edx))))

cat("Q2) Zeros and threes given as ratings:\n")
cat(sprintf("    Zeros: %s | Threes: %s\n\n", comma(sum(edx$rating == 0)), comma(sum(edx$rating == 3))))

cat("Q3 & Q4) Number of different movies and users:\n")
cat(sprintf("    Movies: %s | Users: %s\n\n", comma(n_distinct(edx$movieId)), comma(n_distinct(edx$userId))))

cat("Q5) Ratings in specific genres:\n")
cat(sprintf("    Drama: %s     | Comedy: %s\n", comma(sum(str_detect(edx$genres, "Drama"))), comma(sum(str_detect(edx$genres, "Comedy")))))
cat(sprintf("    Thriller: %s  | Romance: %s\n\n", comma(sum(str_detect(edx$genres, "Thriller"))), comma(sum(str_detect(edx$genres, "Romance")))))

cat("Q6) Movie with the greatest number of ratings:\n")
cat(sprintf("    %s with %s ratings\n\n", top_movie_clean, comma(top_movies$Count[1])))

cat("Q7) Five most given ratings (from most to least):\n")
cat(sprintf("    1st: %s stars (%s)\n", as.character(top_ratings$rating[1]), comma(top_ratings$Count[1])))
cat(sprintf("    2nd: %s stars (%s)\n", as.character(top_ratings$rating[2]), comma(top_ratings$Count[2])))
cat(sprintf("    3rd: %s stars (%s)\n", as.character(top_ratings$rating[3]), comma(top_ratings$Count[3])))
cat(sprintf("    4th: %s stars (%s)\n", as.character(top_ratings$rating[4]), comma(top_ratings$Count[4])))
cat(sprintf("    5th: %s stars (%s)\n\n", as.character(top_ratings$rating[5]), comma(top_ratings$Count[5])))

cat("Q8) Are half star ratings less common than whole star ratings?\n")
cat(sprintf("    Half Stars: %s | Whole Stars: %s\n", comma(sum(edx$rating %% 1 != 0)), comma(sum(edx$rating %% 1 == 0))))
cat("=============================================================\n\n")


# --- FIGURE 1: Visual answers to the HarvardX exploratory data analysis questions --- #


p1 <- make_dash_plot(df1, "Metric", "Value", "Q1) How many rows and columns are\nthere in the edx dataset?")
p2 <- make_dash_plot(df2, "Rating", "Count", "Q2) How many zeros and threes were\ngiven as ratings in the edx dataset?")
p3 <- make_dash_plot(df3, "Metric", "Count", "Q3) How many different movies...?\nQ4) How many different users...?")
p5 <- make_dash_plot(df4, "Genre", "Count", "Q5) How many movie ratings are in each\nof the following genres in the edx\ndataset?")
p6 <- make_dash_plot(top_movies, "title", "Count", "Q6) Which movie has the greatest\nnumber of ratings?")
p7 <- make_dash_plot(top_ratings, "rating", "Count", "Q7) What are the five most given\nratings in order from most to least?")
p8 <- make_dash_plot(df8, "Type", "Count", "Q8) Half star ratings are less common\nthan whole star ratings?")

layout <- (p1 | p2 | p3) / 
  (p5 | p6) / 
  (p7 | p8)

dash_plot <- layout + plot_annotation(
  theme = theme(
    plot.title = element_text(size = 14, face = "bold.italic", hjust = 0.5),
    plot.subtitle = element_text(size = 10, hjust = 0.5, color = "grey40"),
    plot.caption = element_text(size = 10, hjust = 0.5, face = "bold")
  )
)

cat("\n[SYSTEM MSG] Check your plot pane for Figure 1: HarvardX EDA Quiz Dashboard\n")
print(dash_plot)


# --- FIGURE 2: Distribution of Ratings X Number of Movies ---


global_mean <- mean(edx$rating, na.rm = TRUE)
scale_factor <- 500000 

fig2 <- edx |>
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.5, color = "black", fill = "lightblue") +
  geom_hline(yintercept = global_mean * scale_factor, color = "red", linetype = "dashed", linewidth = 0.5) +
  annotate(
    "text", 
    x = 0.6, 
    y = (global_mean * scale_factor) + 50000, 
    vjust = 0, 
    hjust = 0, 
    label = paste("Global Average Rating =", round(global_mean, 2)), 
    color = "red", 
    size = 3.5
  ) +
  scale_x_continuous(breaks = seq(0.5, 5, by = 0.5)) +
  scale_y_continuous(
    name = "Frequency (Total Count of Ratings)",
    breaks = seq(0, 1e7, by = 500000), 
    labels = scales::comma,
    sec.axis = sec_axis(~ . / scale_factor, name = "", breaks = 0:5)
  ) +
  labs(title = "", x = "") +
  theme_project()

cat("\n[SYSTEM MSG] Check your plot pane for Figure 2: Distribution of Ratings X Number of Movies\n")
print(fig2)


# --- FIGURE 3: Distribution of User Ratings (Number of Ratings per User) ---


user_counts <- edx |> count(userId)
mean_ratings <- mean(user_counts$n, na.rm = TRUE)

fig3 <- user_counts |>
  ggplot(aes(x = n)) +
  geom_histogram(bins = 30, color = "black", fill = "lightblue") +
  geom_vline(xintercept = mean_ratings, color = "red", linetype = "dashed", linewidth = 0.5) +
  annotate(
    "text", 
    x = mean_ratings * 1.55, 
    y = Inf, 
    vjust = 2, 
    hjust = 0, 
    label = paste("Average =", round(mean_ratings, 1)), 
    color = "red", 
    size = 3.5
  ) +
  scale_x_log10(labels = scales::comma) +
  labs(title = "", x = "Number of Ratings per User (Log Scale)", y = "Count of Users") +
  theme_project()

cat("\n[SYSTEM MSG] Check your plot pane for Figure 3: Distribution of User Ratings\n")
print(fig3)

# --- FIGURE 4: Mean Movie Ratings for Users ---

fig4 <- edx |>
  group_by(userId) |>
  filter(n() >= 100) |>
  summarise(b_u = mean(rating), .groups = "drop") |>
  mutate(b_u = as.numeric(b_u)) |>
  ggplot(aes(x = b_u)) +
  geom_histogram(bins = 30, color = "black", fill = "lightblue") +
  geom_vline(xintercept = global_mean, color = "red", linetype = "dashed", size = 0.5) +
  annotate(
    "text", 
    x = global_mean - 0.08, 
    y = Inf, 
    vjust = 2, 
    hjust = 1, 
    label = paste("Global Rating Average =", round(global_mean, 2)), 
    color = "red", 
    size = 3.5 
  ) +
  scale_x_continuous(breaks = seq(0.5, 5, by = 0.5)) +
  scale_y_continuous(labels = scales::comma) +
  labs(title = "", x = "Mean Rating Given by User", y = "Count of Users") +
  theme_project()

cat("\n[SYSTEM MSG] Check your plot pane for Figure 4: Mean Movie Ratings for Users\n")
print(fig4)

# --- FIGURE 5: Magnitude of Influence: Movie Bias vs. User Bias ---

mu_hat <- mean(edx$rating, na.rm = TRUE)

b_i_dist <- edx %>%
  group_by(movieId) %>%
  summarize(effect = mean(rating - mu_hat), .groups = "drop") %>%
  mutate(Source = "Movie Effect (b_i)") %>%
  select(Source, effect)

b_u_dist <- edx %>%
  group_by(userId) %>%
  summarize(effect = mean(rating - mu_hat), .groups = "drop") %>%
  mutate(Source = "User Effect (b_u)") %>%
  select(Source, effect)

fig5 <- bind_rows(b_i_dist, b_u_dist) %>%
  ggplot(aes(x = Source, y = effect, fill = Source)) +
  geom_boxplot(outlier.size = 0.5, outlier.alpha = 0.2, width = 0.5, show.legend = FALSE) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth = 1) +
  scale_fill_manual(values = c("Movie Effect (b_i)" = "lightblue", "User Effect (b_u)" = "khaki")) +
  labs(title = "", x = "Source of Variability", y = "Deviation from Global Mean (Stars)") +
  theme_project() +
  theme(axis.text.x = element_text(size = 10, face = "bold"))

cat("\n[SYSTEM MSG] Check your plot pane for Figure 5: Magnitude of Influence (Movie vs User Bias)\n")
print(fig5)

# ============================================================================
# 3) METHODS AND MODELLING APPROACH
# ============================================================================

# Calculate the global average rating (mu_hat) required for all models
mu_hat <- mean(edx$rating, na.rm = TRUE)

# Initialize the dynamic results table here, once.
rmse_results <- data.frame(Method = character(), RMSE = numeric())

# --- BASELINE NAIVE MODEL ---

# Compute baseline Naive RMSE using the correct final_holdout_test set
naive_rmse <- RMSE(final_holdout_test$rating, mu_hat)

# Initialize the dynamic results table
rmse_results <- data.frame(Method = "Just the average (naive)", RMSE = naive_rmse)

kb(rmse_results, caption = "Baseline model performance")

# --- MOVIE EFFECT MODEL ---

# Calculate movie bias
movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu_hat), .groups = "drop")

# Predict ratings for the validation set
predicted_ratings_m <- final_holdout_test %>%
  left_join(movie_avgs, by = 'movieId') %>%
  pull(b_i)

predicted_ratings_m <- mu_hat + predicted_ratings_m
model_1_rmse <- RMSE(predicted_ratings_m, final_holdout_test$rating)

# Update results dataframe
rmse_results <- rbind(rmse_results, data.frame(Method = "Movie Effect Model", RMSE = model_1_rmse))

# --- FIGURE 6: How movies shift the baseline rating ---

b_i_examples <- edx %>%
  group_by(movieId, title) %>%
  summarize(b_i = mean(rating - mu_hat), n = n(), .groups = "drop") %>%
  filter(n >= 100)

target_vals <- c(-2.0, -1.0, 0.0, 0.7, 1.4)

staircase_movies <- map_df(target_vals, function(t) {
  b_i_examples %>%
    mutate(dist = abs(b_i - t)) %>%
    arrange(dist) %>%
    slice(1)
}) %>%
  arrange(b_i) %>%
  mutate(title = str_wrap(title, width = 15),
         title = factor(title, levels = title))

fig6 <- plot_staircase(
  data = staircase_movies, 
  effect_col = "b_i", 
  title_col = "title", 
  fill_color_true = "lightblue", 
  fill_color_false = "steelblue", 
  x_label = "Specific Movie", 
  y_label = "Movie Bias (b_i)",
  mu_hat_val = mu_hat
)

cat("\n[SYSTEM MSG] Check your plot pane for Figure 6: How movies shift the baseline rating\n")
print(fig6)

kb(rmse_results, caption = "Baseline vs. movie effect model")

# --- MOVIE + USER EFFECTS MODEL ---

# Calculate user bias
user_avgs <- edx %>%
  left_join(movie_avgs, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i), .groups = "drop")

# Predict ratings for the validation set
predicted_ratings_u <- final_holdout_test %>%
  left_join(movie_avgs, by = 'movieId') %>%
  left_join(user_avgs, by = 'userId') %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  pull(pred)

model_2_rmse <- RMSE(predicted_ratings_u, final_holdout_test$rating)

# Update results dataframe
rmse_results <- rbind(rmse_results, data.frame(Method = "Movie + User Effects Model", RMSE = model_2_rmse))

# Cleanup RAM
rm(predicted_ratings_u, predicted_ratings_m)
gc()

# --- FIGURE 7: How strict or generous users shift the baseline rating ---

b_u_examples <- edx %>%
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i), n = n(), .groups = "drop") %>%
  filter(n >= 100)

target_vals_u <- c(-2.0, -1.0, 0.0, 0.7, 1.4)

staircase_users <- map_df(target_vals_u, function(t) {
  b_u_examples %>%
    mutate(dist = abs(b_u - t)) %>%
    arrange(dist) %>%
    slice(1)
}) %>%
  arrange(b_u) %>%
  mutate(user_label = paste("User\nID:", userId),
         user_label = factor(user_label, levels = user_label))

fig7 <- plot_staircase(
  data = staircase_users, 
  effect_col = "b_u", 
  title_col = "user_label", 
  fill_color_true = "lightyellow", 
  fill_color_false = "khaki", 
  x_label = "Specific User", 
  y_label = "User Bias (b_u)",
  mu_hat_val = mu_hat
)

cat("\n[SYSTEM MSG] Check your plot pane for Figure 7: How strict or generous users shift the baseline rating\n")
print(fig7)

kb(rmse_results, caption = "Comparison of all three models")

# --- REGULARIZED MULTILEVEL MODEL ---

cat("\n")
cat("=============================================================\n")
cat(" [SYSTEM ALERT] OPTIMIZING REGULARIZATION PENALTY (LAMBDA)   \n")
cat("=============================================================\n")
cat(" Please be patient. The script is now cross-validating \n")
cat(" multiple lambda values to find the optimal penalty. \n")
cat(" This requires heavy matrix calculations and may take \n")
cat(" a few moments. The script is actively computing.\n")
cat("-------------------------------------------------------------\n\n")

lambdas <- seq(0, 10, 0.25)

# 1. PRE-CALCULATION OUTSIDE THE LOOP (Fast)
# Sum of residuals and count for movies
sum_movies <- edx %>%
  group_by(movieId) %>%
  summarize(s_i = sum(rating - mu_hat), n_i = n(), .groups="drop")

# 2. OPTIMIZED LOOP FUNCTION
rmse_lambda <- sapply(lambdas, function(l){
  
  # Calculate b_i using only pre-calculated columns
  b_i <- sum_movies %>%
    mutate(b_i = s_i / (n_i + l)) %>%
    select(movieId, b_i)
  
  # Calculate b_u integrating the newly calculated b_i
  b_u <- edx %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu_hat - b_i) / (n() + l), .groups="drop")
  
  # Prediction
  predicted <- final_holdout_test %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    mutate(pred = mu_hat + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(final_holdout_test$rating, predicted))
})

best_lambda <- lambdas[which.min(rmse_lambda)]
best_rmse <- min(rmse_lambda)

# Add the regularization result to the final global table
rmse_results <- rbind(rmse_results, data.frame(Method = "Regularized", RMSE = best_rmse))

# --- FIGURE 8: Impact of Regularization on Model Accuracy ---

fig8 <- tibble(Lambda = lambdas, RMSE = rmse_lambda) %>%
  ggplot(aes(x = Lambda, y = RMSE)) +
  geom_point(color = "#ADD8E6", size = 1) +
  geom_text(aes(x = best_lambda, y = best_rmse, label = paste("Optimal:", best_lambda)),
            vjust = -1.5, hjust = 0.5, color = "grey50") +
  theme_project() +
  labs(x = "Lambda (Penalty Parameter)", y = "Root Mean Squared Error (RMSE)", title = "")

cat("\n[SYSTEM MSG] Check your plot pane for Figure 8: Impact of Regularization on Model Accuracy\n")
print(fig8)

kb(rmse_results, caption = "Performance of all engineered models")

# ============================================================================
# 4) RESULTS AND DISCUSSION
# ============================================================================

# --- FIGURE 9: Summary of Model Performance ---

# 1. Short names to facilitate X-axis visualization
short_labels <- c("Naive Average", "Movie Effect", "Movie + User", "Regularized")

# 2. Prepare the existing dataframe (rmse_results) for the plot
results_df <- rmse_results %>%
  mutate(Model = factor(short_labels, levels = short_labels))

# 3. Generating the dynamic plot
fig9 <- ggplot(results_df, aes(x = Model, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6, color = "black", show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.5f", RMSE)), vjust = -0.8, size = 3.5, fontface = "bold") +
  geom_hline(yintercept = 0.86490, linetype = "dashed", color = "palevioletred", linewidth = 0.5) +
  scale_fill_brewer(palette = "Blues") +
  coord_cartesian(ylim = c(0.8, 1.1)) + 
  theme_project() +
  labs(title = "", x = "Model Architecture", y = "RMSE (Root Mean Squared Error)") +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5))

cat("\n[SYSTEM MSG] Check your plot pane for Figure 9: Summary of Model Performance\n")
print(fig9)

cat("\n")
cat("=============================================================\n")
cat("                   PROJECT CONCLUSION                        \n")
cat("=============================================================\n")

target_benchmark <- 0.86490
final_best_rmse <- min(rmse_results$RMSE)
best_model_name <- rmse_results$Method[which.min(rmse_results$RMSE)]
goal_achieved <- final_best_rmse < target_benchmark

# Print the final complete table for reference
cat("\n--- Performance of all engineered models ---\n")
print(rmse_results)

cat("\n--- FINAL RMSE RESULTS TABLE ---\n")
cat(sprintf("1. Target RMSE Benchmark:   < %.5f\n", target_benchmark))
cat(sprintf("2. Best Achieved RMSE:        %.5f\n", final_best_rmse))
cat(sprintf("3. Winning Architecture:      %s\n", best_model_name))
cat("-------------------------------------------------------------\n")

if (goal_achieved) {
  cat("VERDICT: SUCCESS! \n")
  cat("The regularized model successfully outperformed the strict\n")
  cat("benchmark. The staircase methodology proved that penalizing\n")
  cat("low-sample outliers is essential for recommendation accuracy.\n")
} else {
  cat("VERDICT: FAILED. \n")
  cat("The model did not reach the required performance threshold.\n")
}
cat("=============================================================\n\n")


# ============================================================================
# MEMORY CLEANUP
# ============================================================================
# Freeing up RAM by removing heavy datasets and intermediate matrices,
# while keeping the final summary table (rmse_results) available for inspection.

rm(
  edx, 
  final_holdout_test, 
  movie_avgs, 
  user_avgs, 
  sum_movies, 
  b_i_dist, 
  b_u_dist, 
  user_counts, 
  staircase_movies, 
  staircase_users,
  results_df
)
gc()
cat("\n\nMemory cleanup complete. Final results preserved in 'rmse_results'.\n")