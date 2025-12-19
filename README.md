# Massachusetts Schools Early Warning System
This repository provides an **early-warning prediction model** for Massachusetts schools, identifying schools at risk of falling into the bottom quartile of math proficiency.

## Table of Contents
- [About The Project](#about-the-project)
- [Repository Structure](#repository-structure)
- [Model Details](#model-details)
- [Usage](#usage)
- [Results](#results)


## About The Project

This project serves as an early-warning system, identifying at-risk schools based on yearly tracked data. The data included as features are reapeated yearly, without limitations, making it applicable for potential policymakers to identify at-risk schools a full year in advance, before new tests are released. 

Identification of at-risk schools could help guide resource allocation and further investigation, while schools that are overperforming relative to predictions (e.g., predicted to have low proficiency but actually have high proficiency) can be examined more closely to understand what practices aid in their success. 


## Repository Structure
### Code (Final_project/code)
The main analysis and modeling pipeline consists of four scripts:

[`Exploratory.ipynb`](Code/Exploratory.ipynb): Exploratory analysis evaluating multiple modeling approaches:
- Regular logistic regression
- Ridge logistic regression
- K-Nearest Neighbors (KNN)
- Decision Tree
  
[`Clean_data.py`](Code/Clean_data.py): Data cleaning functions used in the pipeline

[`Train&save.py`](Code/Train&save.py): Trains the final model and saves the fitted model along with preprocessing objects (imputer, scaler, etc.)

[`Predict.py`](Code/Predict.py): Loads new data (e.g., single-row CSV) and generates predictions, including:
- Classification (low vs. high proficiency)
- Probability of low proficiency

### Raw_data
[EducationData.csv`](<Raw Data/EducationData.csv>): Complete dataset spanning 2016-2018

[Excel_template.csv`](<Raw Data/Excel_template.csv>): Template for single-row predictions

[Data_dictionary.csv`](<Raw Data/Data_dictionary.csv>): Variable descriptions and possible value ranges

### Output
**Final model ROC curve**

<p align="right">(<a href="#Table-of-Contents">back to top</a>)</p>

## Model Details
Temporal Validation Framework:
- Training: 2016-2017 data
- Testing: 2018 data

Final Model:
- Ridge logistic regression best performed on unseen data, most likely due to multicollinearity in some poverty predictors.
- Tested multiple interactions and polynomial terms, and they all resulted in worse fit, indicating a simple model is best.

Features include enrollment, charter status, school level (elementary, middle, high), school type (regular, specialized), Title I status, percent of poverty, and direct certification.  

Key features of the modeling approach:
- Cross-validation for hyperparameter selection
- Class imbalance handled via class weighting
- Standardization of continuous predictors
- Evaluation using accuracy, ROC-AUC, and F1 score

The goals were to optimize the F1 score and capture as many low-performing schools as possible.

<p align="right">(<a href="#Table-of-Contents">back to top</a>)</p>

## Usage

### 1. Download the repository
Clone or download this repository from GitHub and ensure the directory structure is preserved. Alternatively, download the ZIP file from GitHub and unzip it locally.

```bash
git clone https://github.com/KaylenaMann/Final_Project.git
```
### 2. Install dependencies

This project requires Python 3.9 or later.
Install all required packages using
```bash
pip install -r requirements.txt
```
### 3. Train the model
To train the model and save all fitted objects (model, scaler, feature list, and cutoff), run

```bash
python Code/TrainSave.py
```
This script:
- cleans and preprocesses the data
- splits data into training (2016–2017) and test (2018) sets
- fits a cross-validated logistic regression model
- saves trained objects to the model/ directory

### 4. Make a prediction using the Excel template
- Open Raw Data/Excel_template.csv
- Enter values for a single school using the same variable definitions and units as the training data
- Save the file
- Run the following:
```bash
python Code/Predict.py
```
This script will:
- load the trained model and preprocessing objects
- apply the same data cleaning and feature engineering steps used during training
- output a predicted classification (bottom quartile vs. not bottom quartile)
- display the predicted probability of bottom-quartile performance

<p align="right">(<a href="#Table-of-Contents">back to top</a>)</p>

## Results

In terms of prediction, our final model performed well with new data. Final Accuracy was 0.84, with an ROC-AUC of 0.89, and an F1 Score: 0.72. The goal here was to detect at-risk schools despite class imbalance, so the F1 score provides the most realistic measure. Below is the classificaiton report, with higher recall than prediction for the bottom quartile, reflecting this goal of identifying more at risk schools even if it means some false positives. 

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Not Bottom Quartile | 0.93 | 0.85 | 0.89 | 1,398 |
| Bottom Quartile | 0.65 | 0.80 | 0.72 | 474 |
| **Accuracy** | | | **0.84** | **1,872** |
| Macro Avg | 0.79 | 0.83 | 0.80 | 1,872 |
| Weighted Avg | 0.86 | 0.84 | 0.85 | 1,872 |

Additionally, there were 206 schools incorrectly flagged as low-performance, with an average predicted probability of 0.75. When evaluating the means, they had very similar results overall to the correctly flagged schools. These may serve as potential case studies to investigate protective factors.

The strongest predictors of academic performance were poverty status and school level. 

<p align="right">(<a href="#Table-of-Contents">back to top</a>)</p>
