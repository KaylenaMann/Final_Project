# Massachusetts Schools Early Warning System
This repository provides an **early-warning prediction model** for Massachusetts schools, identifying schools at risk of having 25% or fewer students achieving proficient math scores.

## Table of Contents
- [About The Project](#about-the-project)
- [Repository Structure](#repository-structure)
- [Model Details](#model-details)
- [Usage](#usage)
- [Results](#results)


## About The Project

This project serves as an early-warning system, identifying at-risk schools based on yearly tracked data. The data included as features are reapeated yearly, without limitations, making it applicable to potential policymakers to  identify at-risk schools a full year in advance, before new tests are released. 

Identification of at-risk schools could help guide resource allocation and further investigation, while schools that are overperforming relative to predictions (e.g., predicted to have low proficiency but actually have high proficiency) can be examined more closely to understand what practices aid in their success. 


## Repository Structure
### Code (Final_project/code)
The main analysis and modeling pipeline consists of four scripts:

[`Exploratory.ipynb`](Code/Exploratory.ipynb): Exploratory analysis evaluating multiple modeling approaches:
- Regular logistic regression
- Ridge logistic regression
- K-Nearest Neighbors (KNN)
- Decision Tree
  
[`Clean_data.py`](Code/Clean_data.py) Data cleaning functions used in the pipeline

[`Train&save.py`](Code/Train&save.py) Trains the final model and saves the fitted model along with preprocessing objects (imputer, scaler, etc.)

[`Predict.py`](Code/Predict.py)Loads new data (e.g., single-row CSV) and generates predictions including:
- Classification (low vs. high proficiency)
- Probability of low proficiency

### Raw_data
[EducationData.csv`](<Raw Data/EducationData.csv>): Complete dataset spanning 2016-2018

[Excel_template.csv`](<Raw Data/Excel_template.csv>):Template for single-row predictions

[Data_dictionary.csv`](<Raw Data/Data_dictionary.csv>): Variable descriptions and possible value ranges

### Output
**Final model ROC curve**

<p align="right">(<a href="#Table-of-Contents">back to top</a>)</p>

## Model Details
Temporal Validation Framework:
- Training: 2016-2017 data
- Testing: 2018 data

Final Model:
- Ridge logistic regression with augmented features performed best on unseen data:
- Non-linear poverty term (guided by Box-Tidwell analysis)
- School level × enrollment interaction (guided by significant predictors and prior research)

These enhancements improved the F1 score and better captured low-performing schools—the primary objective of this prediction pipeline.

<p align="right">(<a href="#Table-of-Contents">back to top</a>)</p>

## Usage

<p align="right">(<a href="#Table-of-Contents">back to top</a>)</p>

## Results

In terms of prediciton, our final model performed well with new data. Final Accuracy was 0.83, with an ROC-AUC of 0.89, and an F1 Score: 0.71. The goal here was to detect at-risk schools despite class imbalance, so the F1 score provides the most realistic measure. Below is the classificaiton report, with higher recall than prediction for the bottom quartile, reflecting this goal of identifying more at risk schools even if it means some false positives. 


Classification Report (2018):
                     precision    recall  f1-score   support

Not Bottom Quartile       0.93      0.84      0.88      1398
    Bottom Quartile       0.63      0.82      0.71       474

           accuracy                           0.83      1872
          macro avg       0.78      0.83      0.80      1872
       weighted avg       0.86      0.83      0.84      1872


Additionally, there were 193 schools incorrectly flagged as low-performance, with an average predicted probability of 0.74. When evaluating the means, they had very similar results overall to the correctly flagged schools. These may serve as potential case studies to investigate protective factors.

The strongest predictors of academic performance were poverty status, enrollment status, and charter status. With higher poverty schools predicting low proficiency, and lower enrollment numbers predicting low proficiency. Non-charter schools were more likely to not be proficient, compared to charter. Basically, the schools more at risk are those with a higher proportion of students facing economic issues, those with lower enrollment, and non-charter schools. 

<p align="right">(<a href="#Table-of-Contents">back to top</a>)</p>
