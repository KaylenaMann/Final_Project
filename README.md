# Massachusetts Schools Early Warning System
This repository provides an **early-warning prediction model** for Massachusetts schools, identifying schools at risk of having 25% or fewer students achieving proficient math scores.

## Table of Contents
- [About The Project](#about-the-project)
- [Repository Structure](#repository-structure)
- [Model Details](#model-details)
- [Usage](#usage)
- [Results](#results)
  
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

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Model Details
Temporal Validation Framework:
- Training: 2016-2017 data
- Testing: 2018 data

Final Model:
- Ridge logistic regression with augmented features performed best on unseen data:
- Non-linear poverty term (guided by Box-Tidwell analysis)
- School level × enrollment interaction (guided by significant predictors and prior research)

These enhancements improved the F1 score and better captured low-performing schools—the primary objective of this prediction pipeline.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Results

<p align="right">(<a href="#readme-top">back to top</a>)</p>
