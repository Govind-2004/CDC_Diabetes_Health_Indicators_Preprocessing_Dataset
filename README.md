# CDC Diabetes Health Indicators – Data Preprocessing & Exploratory Data Analysis

## Overview

This project demonstrates a comprehensive **Exploratory Data Analysis (EDA)** and **Data Preprocessing** workflow on the **CDC Diabetes Health Indicators** dataset from the **UCI Machine Learning Repository**. The notebook focuses on data exploration, cleaning, visualization, feature analysis, and preprocessing techniques to prepare the dataset for machine learning applications.

**Dataset Source:** https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

---

# Project Structure

## 1. Libraries
- Import required Python libraries
- Configure visualization settings

---

## 2. Reading Data
- Load the dataset using Pandas
- Inspect dataset shape
- Display feature names
- Examine data types
- Generate descriptive statistics

---

# Dataset Features

The dataset consists of **22 attributes**, including **21 predictor variables** and **1 target variable**, describing demographic, lifestyle, and health-related factors associated with diabetes.

| Feature | Description |
|---------|-------------|
| **Diabetes_012** | Target variable indicating diabetes status (0 = No Diabetes, 1 = Prediabetes, 2 = Diabetes). |
| **HighBP** | Indicates whether the individual has high blood pressure (0 = No, 1 = Yes). |
| **HighChol** | Indicates whether the individual has high cholesterol (0 = No, 1 = Yes). |
| **CholCheck** | Indicates whether cholesterol has been checked within the last five years (0 = No, 1 = Yes). |
| **BMI** | Body Mass Index of the individual. |
| **Smoker** | Indicates whether the individual has smoked at least 100 cigarettes in their lifetime (0 = No, 1 = Yes). |
| **Stroke** | Indicates whether the individual has ever had a stroke (0 = No, 1 = Yes). |
| **HeartDiseaseorAttack** | Indicates history of coronary heart disease or heart attack (0 = No, 1 = Yes). |
| **PhysActivity** | Indicates whether the individual participated in physical activity during the past 30 days (0 = No, 1 = Yes). |
| **Fruits** | Indicates regular fruit consumption (0 = No, 1 = Yes). |
| **Veggies** | Indicates regular vegetable consumption (0 = No, 1 = Yes). |
| **HvyAlcoholConsump** | Indicates heavy alcohol consumption (0 = No, 1 = Yes). |
| **AnyHealthcare** | Indicates whether the individual has any form of healthcare coverage (0 = No, 1 = Yes). |
| **NoDocbcCost** | Indicates whether medical care was not sought due to cost (0 = No, 1 = Yes). |
| **GenHlth** | Self-reported general health (1 = Excellent, 2 = Very Good, 3 = Good, 4 = Fair, 5 = Poor). |
| **MentHlth** | Number of mentally unhealthy days during the past 30 days (0–30). |
| **PhysHlth** | Number of physically unhealthy days during the past 30 days (0–30). |
| **DiffWalk** | Indicates serious difficulty walking or climbing stairs (0 = No, 1 = Yes). |
| **Sex** | Gender of the individual (0 = Female, 1 = Male). |
| **Age** | Age group represented as ordinal categories (1–13). |
| **Education** | Highest level of education completed (1–6). |
| **Income** | Annual household income represented as ordinal categories (1–8). |

---

# Exploratory Data Analysis (EDA)

### Histogram Analysis
- Visualize numerical feature distributions
- Analyze skewness of continuous and count variables
- Identify data distribution patterns

### Box Plot Analysis
- Detect potential outliers
- Analyze spread of numerical features

### Correlation Analysis
- Generate a correlation heatmap
- Examine relationships among predictor variables
- Identify potential multicollinearity

---

# Preprocessing

## Data Cleaning

### Missing Value Analysis
- Check for missing values
- Verify dataset completeness

### Duplicate Management
- Identify duplicate records
- Remove duplicate observations

### Outlier Analysis
- Detect outliers using box plots
- Retain outliers as they represent valid health observations

---

# Data Transformation

### Encoding
- Verify categorical feature encoding
- No additional encoding required since all categorical variables are already numerically encoded

### Scaling
- Apply Min-Max Scaling to the continuous numerical feature (BMI)

### Feature Selection
- Compute Mutual Information scores
- Analyze feature importance for diabetes prediction

---

# Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
  - MinMaxScaler
  - Mutual Information Classifier

---

# Contributors

| Team Member | Contribution |
|-------------|--------------|
| **Aathil** | Imported required libraries, loaded the dataset, performed initial data exploration, analyzed dataset shape, inspected unique values, and generated value counts. |
| **Namitha** | Performed duplicate detection and removal, missing value analysis, and outlier analysis. |
| **Abhishek** | Created data visualizations, conducted correlation analysis, and computed Mutual Information for feature importance analysis. |
| **Govind** | Performed data transformation, including verification of feature encoding and application of feature scaling. |

# Repository Structure

```text
CDC_Diabetes_Preprocessing_EDA/
│
├── CDC_Diabetes_Preprocessing_EDA.ipynb
├── diabetes_012_health_indicators_BRFSS2015.csv
├── LICENSE
└── README.md
```
