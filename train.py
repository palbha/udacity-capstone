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
from ucimlrepo import fetch_ucirepo 
  

def read_data():
    # fetch dataset 
    parkinsons_telemonitoring = fetch_ucirepo(id=189) 
    
    # data (as pandas dataframes) 
    X = parkinsons_telemonitoring.data.features 
    y = parkinsons_telemonitoring.data.targets
    X.reset_index(inplace=True)
    y.reset_index(inplace=True) 
    df=pd.merge(X,y,on='index')

    df.drop(columns=['index','motor_UPDRS'],inplace=True)
    return df
    


def main():
    '''Perform the model training'''
    ws = Workspace(
    subscription_id="610d6e37-4747-4a20-80eb-3aad70a55f43",
    resource_group="aml-quickstarts-259725",
    workspace_name="quick-starts-ws-259725",
    )   
    
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
        data_folder = args.data_folder
        training_data_file = os.path.join(data_folder, args.training_data)

        # Load your training data (assuming it's a CSV file)
        df = pd.read_csv(training_data_file)
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
    run.log("R-squared (Accuracy)", np.float(r2_score))



if __name__ == '__main__':
    main()