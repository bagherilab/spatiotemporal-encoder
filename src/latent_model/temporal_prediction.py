import os

import pandas as pd

from simulation_encoder.writer import Writer
from simulation_encoder.plotter import Plotter
from simulation_encoder.logger import Logger
from latent_model.rnn import RNN
from latent_model.sequence_loader import SequenceLoader

from conf.utils.generate_hyperparams import generate_hyperparameters

def predict_next_timepoint(self):
    pass



if __name__ == "__main__":
    results_dir = "results"
    results_path = f"{results_dir}/test"

    encoded_data = {}
    for model in os.listdir(results_path):
        encoded_data[model] = {}
        best_model_path = f"{results_path}/{model}/_best_model"

        for dataset in os.listdir(best_model_path):
            loader = SequenceLoader(f"{best_model_path}/{dataset}/encoded_data.csv")
            test_loader = loader.get_dataloader("test")

            for latent_embeds, sample_ids in test_loader:
                print(sample_ids)

        # encoded_data[model][dataset] = loader
    
    # temporal_model = TemporalModel()
    # temporal_model.predict_next_timepoint()