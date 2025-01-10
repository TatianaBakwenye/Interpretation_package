# Interpretation_package

## Overview
The interpretation_package is a Python package designed to generate plots based on SHAP and Shapley values for interpreting model results. It is divided into two main sections: interpretation and interpretation_time. Each section contains scripts, data, models, and visualizations.

## Folder Descriptions
- data: Synthetically created datasets used for testing purposes. 
- models: These folders contain synthetic models that are used alongside the data to test interpretation functions.
- visualization: This folder stores all generated visual outputs after running the execute_interpretation.py and execute_interpretation_time.py scripts.

## Directory Structure
### 1. interpretation
This section focuses on data interpretation of time-independent model results.
```bash
interpretation/
│
├── data/                        # Contains synthetically generated data for testing
│
├── models/                      # Contains synthetically generated models for testing
│
├── visualization/               # Contains plots generated after running execute scripts
│
├── interpretation.py            # Main script defining interpretation classes and methods
│
└── execute_interpretation.py    # Script to execute classes and methods from interpretation.py with syntheric data and models
```
#### Key Components
- PlotConfig: Configures plot settings (e.g., font and output path).
- DataLoader: Loads data from CSV files for features (X) and target labels (y). **Note that feature files should start with X_ and target files with y_. Data for regression should contain 'reg' in its name, and data for classification should contain 'clf'.**
- ModelLoader: Loads pre-trained models from .pkl files. **Note that regression models should contain 'reg' in their name, and classification models should contain 'clf'.**
- ShapVisualizer: Generates SHAP-based visualizations, including force plots, scatter plots for individual feature importance, and overall feature importance plots.

#### Arguments Explanation:
#### PlotConfig Class:
- output_path (str): The folder path where the generated plots will be saved. Default is ./visualization.
#### DataLoader Class:
- data_folder (str): The folder path where the CSV files for features (X_*) and target labels (y_*) are stored.
##### Methods:
###### load_data(): Loads the data from the CSV files in the specified folder. It reads feature data (X_*) and target data (y_*) and returns two dictionaries:
- X_data: A dictionary where the key is the dataset name, and the value is a pandas DataFrame containing the feature data.
- y_data: A dictionary where the key is the dataset name, and the value is a pandas Series containing the target labels.
#### ModelLoader Class:
- model_folder (str): The folder path where the pre-trained models are stored as .pkl files.
##### Methods:
###### load_models(): Loads all the .pkl files from the specified model folder and stores them in a dictionary:
- models: A dictionary where the key is the model name (file name without .pkl extension), and the value is the pre-trained model.
#### ShapVisualizer Class:
- output_path (str): The folder path where the generated SHAP plots will be saved.
##### Methods:
###### gen_shap_individual(): Generates SHAP force plots for individual observations.
- model: The trained model used for SHAP explanation (either classifier or regressor).
- observation: The observation (row of data) for which the SHAP values are computed.
- model_name: The name of the model being explained.
- output_path: The path where the generated force plot will be saved.
- idx: The identifier of the observation for saving the plot.
- ordinal_function: A function to convert numeric ranks to ordinal strings (e.g., '1st', '2nd').
- selected_classes (optional): The list of class indices to generate plots for in the case of classification models (default is class 1).
###### gen_shap_for_features(): Generates SHAP scatter plots for individual features.
- model: The trained model used for SHAP explanation.
- observation: The data (features) used for the SHAP explanation.
- feature_index: The index of the feature to be plotted.
- title: The title of the plot.
- output_path: The path where the scatter plot will be saved.
- selected_classes (optional): The list of class indices to generate plots for in the case of classification models.
###### generate_plots(): Generates all SHAP-based visualizations (force plots, scatter plots for feature importance, and overall feature importance plots) for multiple models and datasets.
- models: A dictionary of models (key: model name, value: model object).
- X_data: A dictionary of features data (key: dataset name, value: DataFrame).
- y_data: A dictionary of target labels data (key: dataset name, value: Series).
- selected_ids: A list of identifiers (IDs) of the observations for which to generate SHAP force plots.
- ordinal_function: A function to convert numeric ranks to ordinal strings.
- id_column: The name of the column that contains unique identifiers for observations.
- selected_classes (optional): The list of class indices to generate plots for in the case of classification models.
  
### 2. interpretation_time
This section focuses on interpreting the results of a model fitted to specific time points or timestamps, such as for a rolling forecast. It aims to explain how the feature contributions change over time.
```bash
interpretation_over_time/
│
├── data/                        # Contains synthetically generated time-series data for testing
│
├── visualization/               # Contains plots generated after running execute scripts
│
├── interpretation_time.py  # Main script defining classes and methods for interpretation over time
│
└── execute_interpretation_time.py  # Script to execute classes and methods from interpretation_over_time.py with synthetic data
```
#### Key Components:
- ShapleyFeaturePlot: Generates and saves Shapley feature importance plots, both for all data points and for specific individual observations (id values).
- 
#### Arguments Explanation:
#### ShapleyFeaturePlot Class:
- df_shap (DataFrame): A pandas DataFrame containing Shapley values for each feature. The rows represent observations, and the columns represent features. The Shapley values indicate the contribution of each feature to the model's predictions. **Note that the date column should be transformed into the index column.**
- id_column (str): The name of the column containing unique identifiers (e.g., 'id') for the observations in df_shap.
##### Methods:
###### _plot_shapley(): This method generates a Shapley feature importance plot based on the Shapley values for the features of the dataset over time. It visualizes the contribution of each feature to the model's prediction, breaking it down into positive and negative contributions.
- df_to_plot (DataFrame): The subset of df_shap containing the Shapley values to be plotted.
- filename (str): The name of the file where the plot will be saved.
###### plot_main(): Generates and saves a main Shapley feature importance plot for all observations in the provided dataset (df_shap).
###### plot_for_id(): This method filters df_shap to include only the specified observation and then generates and saves the corresponding Shapley feature importance plot for that particular id_value.
- id_value (str or int): The unique identifier (from the id_column) for a specific observation whose Shapley feature importance plot will be generated.

## License
This project is licensed under the MIT License. See the LICENSE file for more information.
