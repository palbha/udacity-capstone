                                                                                                                                                                                                                                                                                                                                                                     # # Capstone Project - Azure Machine Learning Engineer


# Table of Contents

[](https://github.com/palbha/udacity-capstone#table-of-contents)

1.  [Overview](https://github.com/palbha/udacity-capstone/blob/master)
2.  [Project Setup and Installation](https://github.com/palbha/udacity-capstone/blob/master)
3.  [Dataset](https://github.com/palbha/udacity-capstone/blob/master)
4.  [Azure ML Pipeline](https://github.com/palbha/udacity-capstone/blob/master)
5.  [Hyperparameter Tuning](https://github.com/palbha/udacity-capstone/blob/master)
6.  [Model Deployment](https://github.com/palbha/udacity-capstone/blob/master)
7.  [Screen Recording](https://github.com/palbha/udacity-capstone/blob/master)
8.  [Standout Suggestions](https://github.com/palbha/udacity-capstone/blob/master)
9.  [Future Work](https://github.com/palbha/udacity-capstone/blob/master)

## Overview

[](https://github.com/palbha/udacity-capstone#overview)

This is the **Capstone project** for the **Udacity Azure ML Nanodegree** . This project is aimed at demonstrating the capabilities of the Azure ML studio in training & deploying a model.
There are two ways we can achieve this in Azure ML studio :
1. Through AUTOML, a codeless configuration that automates machine learning. 
2. Another, is the HYPERDRIVE, a custom hyperparameter tuning functionality for optimizing a ML model's performance. 
From these two functionalities of the Azure ML studio, a production model was identified &deployed for an  Azure ML End-to-End production pipeline .

High Level Project Overview
![image](https://github.com/palbha/udacity-capstone/assets/20269788/e5de4726-ce2e-48c4-a446-f7edec258f1c)


## Dataset Overview

![image](https://github.com/palbha/udacity-capstone/assets/20269788/bbf57b16-bff6-435d-b09b-aeb7baef0b38)
![image](https://github.com/palbha/udacity-capstone/assets/20269788/b206b5bf-a412-4a34-aed1-3e4d4d1d2823)

## Data Task
The goal is to create a regression model that predicts the total_udpr which is critical for while observing early stage parksinons disease patients.
We have telemetric recording & corrresponding informations observed which can help in building a prediction model to acheive the same.

### Access
The data is uploaded to github repo & hence able to access the data through Python sdk & create a dataset in Azure ML studio
We have created compute to be ablet to run Automl & Hyperdrive parametr tuning

![image](https://github.com/palbha/udacity-capstone/assets/20269788/2468a622-1e0a-40b7-b5f9-e1f6588611a0)


## Automated ML
Triggered an AUtomated ML from python sdk & analyse result to be able to identify a well performing Model

![image](https://github.com/palbha/udacity-capstone/assets/20269788/be3dd8e3-8158-464d-b85d-ca5e64f5e484)
![image](https://github.com/palbha/udacity-capstone/assets/20269788/68192533-cd9f-4a86-b801-c07cb7a48876)
![image](https://github.com/palbha/udacity-capstone/assets/20269788/24efc6b2-7593-4d19-8d8b-623d5a4711e0)


### Results
We could see the Stacking Ensemble Model is performing the best & we could see the influencing Parameters for the same
![image](https://github.com/palbha/udacity-capstone/assets/20269788/8db5d17b-9c00-4a4a-8453-f0627204be83)


## Register Best Model
![image](https://github.com/palbha/udacity-capstone/assets/20269788/73bfe018-d505-458e-af21-aec50f94e3d0)


## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search
I went ahead by Finetuning Randomforest regrssor for different parameters like     "--n_estimators"( Number of trees), "--max_depth" ( Maximum depth of trees) & "--min_samples_split"( Minimum samples to split)
There are primary three reasons why I chose this model 
1. In my expereince Randomforest regressor does perform well in regrssion data if it doesn't have a linear relationship
2.  I wanted to see if by finetuning hyperparameters the accuracy can be imporved & made comparable with the Automl Model
3.  Also analysing the feature importance & undertsanding different tree created within Random forest is something intresting as it helps come up with insights
   

### Results
Automl Model r2 score is way better then the random forest regressor fine tuned . hence will go ahead & deploy AML Model

## Model Deployment
Model is deployed from Python Code & SUccesfully tested using the python code

![image](https://github.com/palbha/udacity-capstone/assets/20269788/336cf4e1-a329-4acc-bb81-75983c491f69)

![image](https://github.com/palbha/udacity-capstone/assets/20269788/3f145f11-ba7e-4271-a1c2-7808fe11b9a5)


## Screen Recording
Screencast Link fro drive , since unable to load on github /Youtube
- (https://drive.google.com/file/d/1rvtBhtmIebGCXhdMhvM6tKfNc5WRzYYM/view?usp=sharing)

## Future Work
I think I can try changing the Model I did hyperparameter tuning for,
Also spending more time on exploration of the data & also while training the model I can do train test split as its quite possible that current AUTOML model is overfitting as I havent evlauted on a test data

