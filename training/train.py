# Import necessary libraries
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from azureml.core.run import Run

from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
  

def read_data():
    # fetch dataset 
    data_path = 'https://raw.githubusercontent.com/palbha/udacity-capstone/master/parkinson_data_v2.csv'


    df=pd.read_csv(data_path)
    return df
    


def main():
    '''Perform the model training'''
    
    
    df=read_data()

    # Assuming you have a DataFrame 'df' with features and target variable
    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        df.drop(columns='total_UPDRS'), df.total_UPDRS, test_size=0.30, random_state=42)

    # Get Azure Machine Learning context
    run = Run.get_context()

    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n_estimators',
        type=int,
        default=100,
        help="Number of trees in the forest")

    parser.add_argument(
        '--max_depth',
        type=int,
        default=None,
        help="Maximum depth of each tree")

    parser.add_argument(
        '--min_samples_split',
        type=float,
        default=2,
        help="Minimum number of samples required to split an internal node")

    # Add other relevant parameters as needed

    args = parser.parse_args()

    # Create and train the random forest regressor
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=42)  # You can adjust other hyperparameters here

    model.fit(x_train, y_train)

    # Evaluate the model on the test set (you can use R-squared or other metrics)
    r2_score = model.score(x_test, y_test)
    run.log("r2_score", np.float(r2_score))



if __name__ == '__main__':
    main()