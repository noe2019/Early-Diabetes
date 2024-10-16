# Race/Ethnicity and Other Predictors of Early-Onset Type 2 Diabetes Mellitus in the US Population

This repository contains the R code used for the analysis described in the article *Race/Ethnicity and Other Predictors of Early-Onset Type 2 Diabetes Mellitus in the US Population*. The analysis is based on data from the National Health and Nutrition Examination Survey (NHANES) from 2001 to 2018.

## Table of Contents

- [Abstract](#abstract)
- [Objectives](#objectives)
- [Methods](#methods)
  - [Study Design and Participants](#study-design-and-participants)
  - [Definition of Diabetes](#definition-of-diabetes)
  - [Variables Under Study](#variables-under-study)
  - [Data Analysis](#data-analysis)
- [Results](#results)
  - [Statistical Analysis](#statistical-analysis)
  - [Machine Learning Models](#machine-learning-models)
- [Plots and Visualizations](#plots-and-visualizations)
- [Conclusions](#conclusions)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Abstract

The study aims to estimate the prevalence of early-onset type 2 diabetes mellitus (T2DM) in the U.S. and explore associations between early-onset T2DM and race/ethnicity, among other predictors. Using stepwise logistic regression and machine learning, the study identifies significant predictors for early-onset T2DM.

## Objectives

Among U.S. adults aged 20+ years with diagnosed T2DM, this study aims to:

1. Estimate the prevalence of early-onset T2DM (onset at age <50.5 years).
2. Test associations between early-onset T2DM and race/ethnicity, and other hypothesized predictors such as acculturation, smoking, obesity, and marital status.

## Methods

### Study Design and Participants

Data from the 2001-2018 NHANES were used for this study. NHANES is a cross-sectional survey conducted by the CDC that collects health-related data through interviews, physical exams, and laboratory tests.

### Definition of Diabetes

Early-onset T2DM is defined as T2DM diagnosed before the age of 50.5 years. The dataset includes individuals who were either diagnosed with diabetes by a healthcare professional or were taking antidiabetic drugs.

### Variables Under Study

- **Race/Ethnicity**: Categorized as Non-Hispanic White (NHW), Non-Hispanic Black (NHB), Hispanic, and Other.
- **Acculturation Score**: Constructed based on country of birth, length of time in the USA, and language spoken at home.
- **Other Predictors**: Smoking status, marital status, education level, BMI, hypertension, and poverty.

### Data Analysis

Data was analyzed using both traditional statistical methods (stepwise logistic regression) and machine learning approaches to predict early-onset T2DM and assess the role of race/ethnicity and other factors.

### Preprocessing the NHANES Data

Download the NHANES data using the `nhanesA` package, and merge multiple cycles from 2001 to 2018.

```r
# Install and load necessary packages
install.packages("nhanesA")
library(nhanesA)
library(dplyr)

# Download demographic and diabetes data for 2018 as an example
demo_2018 <- nhanes('DEMO_I')
diabetes_2018 <- nhanes('DIQ_I')

# Merge datasets by participant identifier
merged_2018 <- merge(demo_2018, diabetes_2018, by = "SEQN")

# Repeat for other years and combine into a single dataset
combined_data <- bind_rows(merged_data_list)
```

### Variable Creation

Create new variables, including early-onset T2DM, acculturation score, smoking status, and obesity.

```r
# Create early-onset diabetes variable
combined_data <- combined_data %>%
  mutate(early_onset_T2DM = ifelse(DIABAGE < 50.5, 1, 0))

# Create acculturation score
combined_data <- combined_data %>%
  mutate(acculturation_score = case_when(
    CountryOfBirth == "USA" ~ 5,
    CountryOfBirth != "USA" & YearsInUSA > 19 ~ 4,
    CountryOfBirth != "USA" & YearsInUSA <= 19 ~ 3,
    Language == "English" ~ 2,
    TRUE ~ 1
  ))

# Recode other variables
combined_data <- combined_data %>%
  mutate(
    smoking_status = ifelse(Smoking == "Yes", 1, 0),
    obesity = ifelse(BMI >= 30, 1, 0),
    marital_status = case_when(
      marital_status %in% c("Married", "Partnered") ~ "Married",
      marital_status %in% c("Single", "Divorced") ~ "Single"
    )
  )
```

## Results

### Statistical Analysis

A stepwise logistic regression was performed to assess the association between early-onset T2DM and various predictors.

```r
# Logistic regression model
stepwise_model <- glm(early_onset_T2DM ~ race_ethnicity + smoking_status + obesity + acculturation_score +
                      education_level + marital_status + hypertension,
                      data = combined_data, family = binomial)

# View the model summary
summary(stepwise_model)
```

### Machine Learning Models

Eleven supervised machine learning algorithms were trained, including Random Forest, Gradient Boosting, and others. Below is an example of a Random Forest implementation.

```r
# Load necessary libraries
library(randomForest)
library(caret)

# Split the data into training and testing sets
set.seed(123)
train_indices <- sample(1:nrow(combined_data), 0.7 * nrow(combined_data))
train_data <- combined_data[train_indices, ]
test_data <- combined_data[-train_indices, ]

# Train Random Forest model
rf_model <- randomForest(early_onset_T2DM ~ race_ethnicity + smoking_status + obesity + acculturation_score +
                         education_level + marital_status + hypertension, 
                         data = train_data)

# Evaluate the model
predictions <- predict(rf_model, test_data)
confusionMatrix(predictions, test_data$early_onset_T2DM)
```

## Plots and Visualizations

### Distribution of Early-Onset T2DM by Race/Ethnicity

```r
# Load ggplot2 for visualization
library(ggplot2)

# Plot the distribution of early-onset T2DM by race/ethnicity
ggplot(combined_data, aes(x = race_ethnicity, fill = factor(early_onset_T2DM))) +
  geom_bar(position = "fill") +
  labs(title = "Proportion of Early-Onset T2DM by Race/Ethnicity",
       x = "Race/Ethnicity",
       y = "Proportion",
       fill = "Early-Onset T2DM") +
  theme_minimal()
```

### ROC Curve for Logistic Regression

```r
# Load pROC for ROC analysis
library(pROC)

# Predict probabilities for ROC curve
logit_pred <- predict(stepwise_model, newdata = test_data, type = "response")
roc_curve <- roc(test_data$early_onset_T2DM, logit_pred)

# Plot ROC curve
plot(roc_curve, main = "ROC Curve for Logistic Regression")
auc(roc_curve)  # Print the AUC
```

### Feature Importance from Random Forest

```r
# Plot variable importance from Random Forest
varImpPlot(rf_model, main = "Variable Importance from Random Forest")
```

## Conclusions

The study found that early-onset T2DM is more prevalent among Non-Hispanic Black (NHB) and Hispanic populations compared to Non-Hispanic Whites (NHW). Acculturation score, tobacco smoking, and obesity emerged as important predictors of early-onset T2DM in both males and females.

## Acknowledgements

We thank the study participants and acknowledge the contributions of the collaborators and the CDC for providing the NHANES dataset.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
```
