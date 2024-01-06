import pandas as pd

# Load the Excel file
data = pd.read_excel('C:/Users/Archit Jain/Downloads/data.xlsx')

# Function to extract and preprocess data for each energy source
def extract_corrected_data(df, start_col):
    cols = [df.columns[start_col], df.columns[start_col + 1], df.columns[start_col + 2], df.columns[start_col + 3]]
    data_source = df[cols].dropna().reset_index(drop=True).iloc[1:]
    data_source.columns = ['O', 'S', 'D', 'RPN']
    return data_source.apply(pd.to_numeric)

# Extracting data for each energy source
hydropower_data = extract_corrected_data(data, 2)  # Adjust the index as per the Excel structure
# Repeat for other energy sources (wind, solar, etc.) with correct column indices


# DEA Calculation (install pyDEA package if not installed)
import pyDEA
from pyDEA.core.data_processing.solution_parser import parse_solution
from pyDEA.core.models.bootstrap_model import BootstrapInputOrientedModel

# Function to perform DEA analysis
def perform_dea(data, inputs, output):
    categories = data.index
    data = data[inputs + [output]]
    dea_model = BootstrapInputOrientedModel(data, categories, inputs, [output], returns_to_scale='CRS')
    dea_model.run()
    return parse_solution(dea_model)

# Example usage for Hydropower data
dea_results_hydropower = perform_dea(hydropower_data, ['O', 'S', 'D'], 'RPN')

# import numpy as np
# from DEApy import DEA

# # Function to perform DEA analysis using DEApy
# def perform_dea_deapy(data, inputs, output):
#     # Prepare data for DEApy
#     X = data[inputs].to_numpy()  # Inputs
#     y = data[[output]].to_numpy()  # Output
#     dmus = data.index.to_list()  # DMU names

#     # Create DEA model (CRS input-oriented by default)
#     dea_model = DEA(X, y, returns_to_scale='CRS', orient='in')

#     # Run DEA analysis
#     efficiency_scores = dea_model.fit()

#     # Create a dictionary to hold DMU names and their efficiency scores
#     dea_results = dict(zip(dmus, efficiency_scores))

#     return dea_results

# # Example usage for Hydropower data
# dea_results_hydropower = perform_dea_deapy(hydropower_data, ['O', 'S', 'D'], 'RPN')



from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Preparing data for ML model
X = hydropower_data[['O', 'S', 'D']]  # Input features
y = hydropower_data['RPN']            # Target

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the model
ml_model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500)
ml_model.fit(X_train, y_train)

# Predictions and evaluation
predictions = ml_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

import matplotlib.pyplot as plt

# Example visualization comparing DEA and ML results
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Comparison of DEA and ML Predictions')
plt.show()
