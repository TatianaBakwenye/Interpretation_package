import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt
import os
import shutil
from sklearn.base import ClassifierMixin, RegressorMixin

# Class for plot configuration
class PlotConfig:
    def __init__(self, output_path='./visualization'):
        self.output_path = output_path
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.sans-serif': ['Arial'], 'pdf.fonttype': 42})

    @staticmethod
    def ordinal(n):
        return "%d%s" % (n, "tsnrhtdd"[((n // 10 % 10 != 1) * (n % 10 < 4) * n % 10)::4])

# DataLoader class to load data
class DataLoader:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.X_data = {}
        self.y_data = {}

    def load_data(self):
        for file_name in os.listdir(self.data_folder):
            if file_name.startswith("X_") and file_name.endswith(".csv"):
                dataset_name = file_name[2:-4]
                self.X_data[dataset_name] = pd.read_csv(os.path.join(self.data_folder, file_name))
            elif file_name.startswith("y_") and file_name.endswith(".csv"):
                dataset_name = file_name[2:-4]
                self.y_data[dataset_name] = pd.read_csv(os.path.join(self.data_folder, file_name))["Target"]
        return self.X_data, self.y_data

# ModelLoader class to load models
class ModelLoader:
    def __init__(self, model_folder):
        self.model_folder = model_folder
        self.models = {}

    def load_models(self):
        for file_name in os.listdir(self.model_folder):
            if file_name.endswith(".pkl"):
                model_name = file_name[:-4]
                with open(os.path.join(self.model_folder, file_name), "rb") as f:
                    self.models[model_name] = pickle.load(f)
        return self.models

# Main class for shap visualization
class ShapVisualizer:
    def __init__(self, output_path):
        self.output_path = output_path

    def gen_shap_individual(self, model, observation, model_name, output_path, idx, ordinal_function, selected_classes=None):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(observation)
        observation2 = (observation.rank(pct=True) * 10).round(0)
        observation2 = observation2.applymap(lambda x: ordinal_function(int(x)))

        if isinstance(model, ClassifierMixin):
            if selected_classes is None:
                selected_classes = [1]
            for class_idx in selected_classes:
                shap.plots.force(base_value=explainer.expected_value[class_idx],
                                 shap_values=shap_values[class_idx],
                                 features=observation.round(0), # Sophie used observation2
                                 feature_names=observation.columns,
                                 matplotlib=True,
                                 show=False)
                plt.savefig(f"{output_path}/force_plot_{model_name}_class_{class_idx}_{idx}.pdf", bbox_inches='tight')
                plt.close()
        elif isinstance(model, RegressorMixin):
            shap.plots.force(base_value=explainer.expected_value,
                             shap_values=shap_values,
                             features=observation.round(0), # Sophie used observation2
                             feature_names=observation.columns,
                             matplotlib=True,
                             show=False)
            plt.savefig(f"{output_path}/force_plot_{model_name}_{idx}.pdf", bbox_inches='tight')
            plt.close()

    def gen_shap_for_features(self, model, observation, feature_index, title, output_path, selected_classes=None):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(observation)

        if isinstance(model, ClassifierMixin):
            if selected_classes is None:
                selected_classes = [1]
            for class_idx in selected_classes:
                shap_values_feature = shap_values[class_idx][:, feature_index]
                data_feature = observation.values[:, feature_index].reshape(-1, 1)
                shap_exp = shap.Explanation(values=shap_values_feature, data=data_feature,
                                            feature_names=observation.columns[feature_index])
                shap.plots.scatter(shap_values=shap_exp, title=f"{title}_class_{class_idx}",
                                   dot_size=80, alpha=0.3, show=False)
                plt.tight_layout()
                plt.savefig(f"{output_path}/{title}_class_{class_idx}.pdf")
                plt.close()
        elif isinstance(model, RegressorMixin):
            shap_values_feature = shap_values[:, feature_index]
            data_feature = observation.values[:, feature_index].reshape(-1, 1)
            shap_exp = shap.Explanation(values=shap_values_feature, data=data_feature,
                                        feature_names=observation.columns[feature_index])
            shap.plots.scatter(shap_values=shap_exp, title=title, dot_size=80, alpha=0.3, show=False)
            plt.tight_layout()
            plt.savefig(f"{output_path}/{title}_regression.pdf")
            plt.close()

    def generate_plots(self, models, X_data, y_data, selected_ids, ordinal_function, id_column, selected_classes=None):
        if selected_classes is None:
            selected_classes = [1]
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path)

        for model_name, model in models.items():
            print(f"Generating plots for {model_name}...")

            # Subfolder for each type of plot
            feature_importance_folder = f"{self.output_path}/feature_importance"
            force_plot_folder = f"{self.output_path}/force_plots"
            feature_folder = f"{self.output_path}/feature"
            os.makedirs(feature_importance_folder, exist_ok=True)
            os.makedirs(force_plot_folder, exist_ok=True)
            os.makedirs(feature_folder, exist_ok=True)

            for dataset_name, X in X_data.items():
                y = y_data[dataset_name]
                if 'reg' in dataset_name:
                    if isinstance(model, ClassifierMixin):
                        continue
                elif 'clf' in dataset_name:
                    if isinstance(model, RegressorMixin):
                        continue
                else:
                    raise ValueError(f"Task type could not be determined from the dataset name: {dataset_name}")

                # Overall Feature Importance
                features = X.drop(columns=[id_column]).columns
                if hasattr(model, "feature_importances_"):
                    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

                    # Optionally save feature importances to CSV
                    importance_df = importance.reset_index()
                    importance_df.columns = ['Feature', 'Importance']
                    #importance_df.to_csv(f"{feature_importance_folder}/{model_name}_overall_feature_importance_{dataset_name}.csv", index=False)

                    # Plot the feature importance
                    title = f"{model_name} Overall Feature Importance ({dataset_name})"
                    importance.plot(kind='bar', title=title, ylabel="Importance")
                    plt.tight_layout()
                    plt.savefig(f"{feature_importance_folder}/{model_name}_overall_feature_importance_{dataset_name}.pdf")
                    plt.close()

                # Generate SHAP scatter plots for individual features
                for feature_index, feature_name in enumerate(features):
                    title = f"{model_name}_{feature_name}_{dataset_name}"
                    self.gen_shap_for_features(model, X.drop(columns=[id_column]), feature_index, title, feature_folder, selected_classes)

                # Generate SHAP force plots for selected observations
                for idx in selected_ids:
                    if idx not in X[id_column].values:
                        continue
                    observation_row = X[X[id_column] == idx].drop(columns=[id_column])
                    self.gen_shap_individual(model, observation_row, model_name, force_plot_folder, idx, ordinal_function, selected_classes)

# def main():
#     plot_config = PlotConfig()
#     data_loader = DataLoader(data_folder='./data')
#     model_loader = ModelLoader(model_folder='./models')
#     shap_visualizer = ShapVisualizer(output_path=plot_config.output_path)

#     X_data, y_data = data_loader.load_data()
#     models = model_loader.load_models()
#     selected_ids = ['id_0', 'id_11', 22] # value 22 doesn't exist in the datasets i'm working with, so it will be skipped
#     id_column = 'id'  # This can be updated as needed
#     selected_classes = None # This can be updated as needed

#     # Drop the 'date' column here (or any other columns you didn't use as features)
#     for dataset_name in X_data:
#         X_data[dataset_name] = X_data[dataset_name].drop(columns=['date'])

#     shap_visualizer.generate_plots(models, X_data, y_data, selected_ids, plot_config.ordinal, id_column, selected_classes)

# if __name__ == "__main__":
#     main()