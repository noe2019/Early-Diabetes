
# Replicating Early-Onset Type 2 Diabetes Analysis from NHANES Data (2001-2018)
## Table of Contents

- [Overview](#overview)
- [Data Requirements](#data-requirements)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Logistic Regression Model](#logistic-regression-model)
- [Machine Learning Models](#machine-learning-models)
- [Visualization](#visualization)
  - [Distribution Plots](#distribution-plots)
  - [Odds Ratios](#odds-ratios)
  - [Random Forest Feature Importance](#random-forest-feature-importance)
  - [ROC Curve](#roc-curve)
  - [Confusion Matrix](#confusion-matrix)
  - [Other Plots](#other-plots)
- [License](#license)

This repository contains the R code for replicating the analysis from the paper: **Race/Ethnicity and Other Predictors of Early-Onset Type 2 Diabetes Mellitus in the US Population**. The analysis uses data from the National Health and Nutrition Examination Survey (NHANES) from 2001 to 2018 to estimate the prevalence of early-onset type 2 diabetes (T2DM) and identify associated predictors using logistic regression and machine learning techniques.

## Overview

This project aims to replicate the findings of a study that explores the relationship between race/ethnicity, socioeconomic factors, and early-onset type 2 diabetes. The study uses a combination of statistical (logistic regression) and machine learning models to predict early-onset T2DM.

## Data Requirements

The analysis uses the NHANES dataset from the years 2001-2018. You can download the data from [NHANES website](https://www.cdc.gov/nchs/nhanes/index.htm). The specific datasets required include demographic information (`DEMO`), diabetes diagnosis (`DIQ`), and variables such as BMI, acculturation, smoking status, and more.

Once downloaded, the datasets must be merged across all years to perform the analysis.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/early-onset-t2dm-analysis.git
   cd early-onset-t2dm-analysis
   ```

2. Install the required R packages:
   ```r
   install.packages(c("nhanesA", "dplyr", "ggplot2", "randomForest", "pROC", "caret", "reshape2"))
   ```

## Data Preprocessing

To merge and clean the NHANES data across the relevant years, use the following code:

```r
# Load necessary libraries
library(nhanesA)
library(dplyr)

# Example for downloading and merging data from NHANES (for 2018):
demo_2018 <- nhanes('DEMO_I')
diabetes_2018 <- nhanes('DIQ_I')

# Merge demographic and diabetes data
merged_2018 <- merge(demo_2018, diabetes_2018, by = "SEQN")

# Repeat for all NHANES cycles from 2001-2018 and bind them together
combined_data <- bind_rows(merged_data_list)
```

### Creating Variables

```r
# Create the early-onset diabetes variable, acculturation score, and other predictors
combined_data <- combined_data %>%
  mutate(early_onset_T2DM = ifelse(DIABAGE < 50.5, 1, 0),
         acculturation_score = case_when(
           CountryOfBirth == "USA" ~ 5,
           CountryOfBirth != "USA" & YearsInUSA > 19 ~ 4,
           CountryOfBirth != "USA" & YearsInUSA <= 19 ~ 3,
           Language == "English" ~ 2,
           TRUE ~ 1
         ),
         smoking_status = ifelse(Smoking == "Yes", 1, 0),
         obesity = ifelse(BMI >= 30, 1, 0),
         marital_status = ifelse(marital_status %in% c("Married", "Partnered"), "Married", "Single"))
```

## Logistic Regression Model

The logistic regression model is used to estimate the odds of early-onset T2DM across different predictors.

```r
# Run stepwise logistic regression
stepwise_model <- glm(early_onset_T2DM ~ race_ethnicity + smoking_status + obesity + acculturation_score +
                      education_level + marital_status + hypertension, 
                      data = combined_data, family = binomial)

summary(stepwise_model)
```

## Machine Learning Models

This analysis also uses Random Forest as a machine learning model to classify individuals with early-onset T2DM.

```r
# Load library and split the data
library(randomForest)

# Split data into training and testing sets
set.seed(123)
train_indices <- sample(1:nrow(combined_data), 0.7 * nrow(combined_data))
train_data <- combined_data[train_indices, ]
test_data <- combined_data[-train_indices, ]

# Train Random Forest model
rf_model <- randomForest(early_onset_T2DM ~ race_ethnicity + smoking_status + obesity + acculturation_score +
                         education_level + marital_status + hypertension, 
                         data = train_data)

# Predict and evaluate
predictions <- predict(rf_model, test_data)
confusionMatrix(predictions, test_data$early_onset_T2DM)
```

## Visualization

### Distribution Plots

To visualize the distribution of early-onset T2DM by race/ethnicity, use this code:

```r
library(ggplot2)

ggplot(combined_data, aes(x = race_ethnicity, fill = factor(early_onset_T2DM))) +
  geom_bar(position = "fill") +
  labs(title = "Proportion of Early-Onset T2DM by Race/Ethnicity",
       x = "Race/Ethnicity", y = "Proportion", fill = "Early-Onset T2DM") +
  theme_minimal()
```

### Odds Ratios

You can plot the odds ratios from the logistic regression model:

```r
# Extract coefficients and CIs
logistic_coeffs <- summary(stepwise_model)$coefficients
odds_ratios <- exp(logistic_coeffs[, 1])
conf_intervals <- exp(confint(stepwise_model))

# Create plot data
odds_data <- data.frame(
  Variable = rownames(logistic_coeffs),
  OddsRatio = odds_ratios,
  LowerCI = conf_intervals[, 1],
  UpperCI = conf_intervals[, 2]
)

# Plot odds ratios
ggplot(odds_data, aes(x = Variable, y = OddsRatio)) +
  geom_point() +
  geom_errorbar(aes(ymin = LowerCI, ymax = UpperCI), width = 0.2) +
  coord_flip() +
  scale_y_log10() +
  labs(title = "Odds Ratios and 95% CI from Logistic Regression",
       x = "Variables", y = "Odds Ratio (log scale)") +
  theme_minimal()
```

### Random Forest Feature Importance

To plot the feature importance from the Random Forest model:

```r
varImpPlot(rf_model, main = "Variable Importance from Random Forest")
```

### ROC Curve

Plot the ROC curve for the logistic regression or machine learning model:

```r
library(pROC)

# Logistic regression predictions
logit_pred <- predict(stepwise_model, newdata = test_data, type = "response")
roc_curve <- roc(test_data$early_onset_T2DM, logit_pred)

# Plot ROC
plot(roc_curve, main = "ROC Curve for Logistic Regression Model")
auc(roc_curve)
```

### Confusion Matrix

To visualize the confusion matrix for the Random Forest model as a heatmap:

```r
conf_df <- as.data.frame(confusionMatrix(predictions, test_data$early_onset_T2DM)$table)

ggplot(conf_df, aes(Prediction, Reference)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "lightblue", high = "red") +
  labs(title = "Confusion Matrix for Random Forest Model", x = "Predicted", y = "Actual") +
  theme_minimal()
```

### Other Plots

#### Acculturation Score by Race/Ethnicity

```r
ggplot(combined_data, aes(x = race_ethnicity, y = acculturation_score, fill = race_ethnicity)) +
  geom_boxplot() +
  labs(title = "Acculturation Score by Race/Ethnicity",
       x = "Race/Ethnicity", y = "Acculturation Score") +
  theme_minimal()
```

#### BMI Distribution by Early-Onset T2DM

```r
ggplot(combined_data, aes(x = factor(early_onset_T2DM), y = BMI, fill = factor(early_onset_T2DM))) +
  geom_boxplot() +
  labs(title = "BMI Distribution by Early-Onset T2DM Status",
       x = "Early-Onset T2DM", y = "BMI") +
  theme_minimal()
```

#### Correlation Heatmap

```r
library(reshape2)

# Calculate correlation matrix
numeric_vars <- combined_data %>% select(BMI, acculturation_score, hypertension, smoking_status)
corr_matrix <- cor(numeric_vars, use = "complete.obs")

# Plot heatmap
melted_corr_matrix <- melt(corr_matrix

)

ggplot(melted_corr_matrix, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  labs(title = "Correlation Matrix Heatmap") +
  theme_minimal()
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

This README file is structured to give clear instructions to users of the repository, including how to replicate the data analysis, run machine learning models, and create visualizations. You can customize it as needed based on your project requirements and any additional analysis you wish to perform.
