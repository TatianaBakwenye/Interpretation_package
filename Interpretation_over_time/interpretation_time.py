import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random

class ShapleyFeaturePlot:
    def __init__(self, df_shap, id_column):
        """Initialize the ShapleyFeaturePlot class.
        
        Parameters:
            df_shap (DataFrame): DataFrame containing Shapley values.
            id_column (str): Name of the column containing 'id' values.
        """
        self.df_shap = df_shap
        self.id_column = id_column
        self.df_shap_without_id = self.df_shap.drop(columns=[self.id_column])

        # Define colors for plotting
        self.colors = dict(zip(
            self.df_shap_without_id.columns.tolist(), sns.color_palette('tab20', n_colors=len(self.df_shap_without_id.columns))
        ))

        # Define visualization directory
        self.visualization_dir = './visualization'
        os.makedirs(self.visualization_dir, exist_ok=True)

    def _plot_shapley(self, df_to_plot, filename):
        """Generate and save the Shapley feature importance plot."""
        fig, ax = plt.subplots(figsize=[10, 5])

        for dt, fi in df_to_plot.T.items():
            v_pos = 0
            v_neg = 0

            for i, (key, val) in enumerate(fi.items()):
                color = self.colors[key]
                if val > 0:
                    ax.plot([dt, dt], [v_pos, v_pos + val], zorder=-i, c=color, lw=4, solid_capstyle='butt')
                    v_pos += val
                elif val < 0:
                    ax.plot([dt, dt], [v_neg, v_neg + val], zorder=i, c=color, lw=4, solid_capstyle='butt')
                    v_neg += val

        for key in self.colors:
            ax.plot(np.nan, np.nan, color=self.colors[key], label=key, lw=4)

        ax.axhline(0, lw=0.6, c='k')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend()
        ax.set_title('Shapley feature importance (additive)', loc='left')

        # Save the plot to the specified path
        output_path = os.path.join(self.visualization_dir, filename)
        fig.savefig(output_path)
        print(f"Plot saved to {output_path}")

    def plot_main(self):
        """Generate and save the main Shapley feature importance plot."""
        self._plot_shapley(self.df_shap_without_id, 'main_plot.png')

    def plot_for_id(self, id_value):
        """Generate and save the Shapley feature importance plot for a specific 'id' value."""
        df_filtered = self.df_shap[self.df_shap[self.id_column] == id_value].drop(columns=[self.id_column])
        filename = f'plot_for_id_{id_value}.png'
        self._plot_shapley(df_filtered, filename)

# if __name__ == "__main__":
#     # Load feature importance data from CSV
#     df_shap = pd.read_csv('./data/feature_importances.csv', index_col=0)

#     # Specify the column name for 'id'
#     id_column = 'id'

#     # Initialize the plotter
#     plotter = ShapleyFeaturePlot(df_shap, id_column)

#     # Generate and save the main Shapley feature importance plot
#     plotter.plot_main()

#     # Generate and save ID-specific plots for two random IDs
#     random_ids = random.sample(df_shap[id_column].unique().tolist(), 2)
#     for id_value in random_ids:
#         plotter.plot_for_id(id_value)
