import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from latent_model.supervised import SupervisedRunner
from latent_model.loaders.point_loader import PointLoader
from latent_model.models.supervised_model import SupervisedClassifier


RESULTS_DIR = "results"
DATA_DIR = "data"
EXPERIMENT_NAME = "arch_simple_cancer"

def main() -> None:
    """Main function to run supervised emulation using global configuration variables."""
    conf_name = EXPERIMENT_NAME
    results_dir = RESULTS_DIR

    output_dir = os.path.join(results_dir, conf_name, 'emulation')
    os.makedirs(output_dir, exist_ok=True)
    
    runner = SupervisedRunner()
    
    try:
        results = runner.run_emulator(conf_name)
        
        if results:
            results_path = os.path.join(output_dir, 'emulation_results.json')
            save_results(results, results_path)

            vis_dir = os.path.join(output_dir, 'visualizations')
            visualize_results(results, vis_dir)
    except Exception as e:
        raise

def visualize_results(results, output_dir: str) -> None:
    """
    Visualize emulation results with plots.
    
    Parameters
    ----------
    results : SupervisedResults
        Results from running emulation
    output_dir : str
        Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract results into a DataFrame for easier plotting
    records = []
    
    for dataset_name, dataset_result in results.dataset_results.items():
        for encoder_name, encoder_result in dataset_result.encoder_model_results.items():
            for label, label_result in encoder_result.label_results.items():
                for model_type, result in label_result.model_results.items():
                    records.append({
                        'dataset': dataset_name,
                        'encoder': encoder_name,
                        'label': label,
                        'model': model_type,
                        'r2_score': result.score
                    })
    
    df = pd.DataFrame(records)
    
    # Set Seaborn style
    sns.set(style="whitegrid")
    
    # Plot 1: R² scores by encoder model for each target label
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='label', y='r2_score', hue='encoder', data=df)
    plt.title('R² Scores by Encoder Model and Target Label')
    plt.xlabel('Target Label')
    plt.ylabel('R² Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r2_by_encoder_label.png'))
    plt.close()
    
    # Plot 2: R² scores by regression model for each target label
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='label', y='r2_score', hue='model', data=df)
    plt.title('R² Scores by Regression Model and Target Label')
    plt.xlabel('Target Label')
    plt.ylabel('R² Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r2_by_regressor_label.png'))
    plt.close()
    
    # Plot 3: Heatmap of average R² scores for each encoder-regressor pair
    pivot_df = df.pivot_table(
        index='encoder', 
        columns='model', 
        values='r2_score', 
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, cmap='viridis', vmin=0, vmax=1)
    plt.title('Mean R² Scores by Encoder and Regression Model')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r2_heatmap.png'))
    plt.close()
    
    # Save results as CSV for further analysis
    df.to_csv(os.path.join(output_dir, 'emulation_results.csv'), index=False)
    
    print(f"Visualizations saved to {output_dir}")

def save_results(results, output_path: str) -> None:
    """
    Save emulation results to a JSON file.
    
    Parameters
    ----------
    results : SupervisedResults
        Results from running emulation
    output_path : str
        Path to save JSON results
    """
    results_dict = results.to_dict()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"Results saved to {output_path}")

def create_loaders(results_path: str, time_point_idx: int | None = None) -> dict[str, dict[str, PointLoader]]:
    """Creates data loaders for all of the best performing models in a given encoder result"""
    loaders = {}
    for model_name in os.listdir(results_path):
        loaders[model_name] = {}
        model_path = f"{results_path}/{model_name}/_best_model"
        for dataset_name in os.listdir(model_path):
            loaders[model_name][dataset_name] = PointLoader(
                f"{model_path}/{dataset_name}/encoded_data.csv", time_point_idx=time_point_idx
            )
    return loaders

def create_models_list() -> list[SupervisedClassifier]:
    model_classes = ["logistic_regression", "random_forest", "svm"]
    models = []
    for model_class in model_classes:
        model = SupervisedClassifier(model_type = model_class)
        models.append(model)

    return models

if __name__ == "__main__":
    main()