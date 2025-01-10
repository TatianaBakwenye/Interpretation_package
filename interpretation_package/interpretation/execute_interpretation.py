from interpretation_package import PlotConfig, DataLoader, ModelLoader, ShapVisualizer

# Initialize the plot configuration, data loader, model loader, and SHAP visualizer
plot_config = PlotConfig()
data_loader = DataLoader(data_folder='./data')
model_loader = ModelLoader(model_folder='./models')
shap_visualizer = ShapVisualizer(output_path=plot_config.output_path)

# Load the data and models
X_data, y_data = data_loader.load_data()
models = model_loader.load_models()

# Specify the selected observation IDs and classes
selected_ids = ['id_0', 'id_11', 22] # value 22 doesn't exist in the datasets i'm working with, so it will be skipped
selected_classes = None # This can be updated as needed
id_column = 'id' # This can be updated as needed

# Drop the 'date' column here (or any other model you didn't use as features)
for dataset_name in X_data:
    X_data[dataset_name] = X_data[dataset_name].drop(columns=['date'])

# Generate the SHAP plots
shap_visualizer.generate_plots(models, X_data, y_data, selected_ids, plot_config.ordinal, id_column, selected_classes=selected_classes)