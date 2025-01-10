from interpretation_package import ShapleyFeaturePlot
import random
import pandas as pd

# Load feature importance data from CSV
df_shap = pd.read_csv('./data/feature_importances.csv', index_col=0)

# Specify the column name for 'id'
id_column = 'id'

# Initialize the plotter
plotter = ShapleyFeaturePlot(df_shap, id_column)

# Generate and save the main Shapley feature importance plot
plotter.plot_main()

# Generate and save ID-specific plots for two random IDs
random_ids = random.sample(df_shap[id_column].unique().tolist(), 2)
for id_value in random_ids:
    plotter.plot_for_id(id_value)