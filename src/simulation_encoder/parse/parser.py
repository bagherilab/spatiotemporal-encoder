import os
import json
import ntpath

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.s3_utils import download_file, list_s3_files, upload_file
from parse.graph_parser import GraphParser
from parse.cell_parser import CellParser




class Parser(ABC):
    @abstractmethod
    def parse_files_to_csv(self, key) -> None:
        pass


class ArcadeParser:
    def __init__(self):
        self.timepoints = [(x / 2.0) for x in range(0, 31)]
        self.data_dir = "../../data/ARCADE"
        self.bucket = "bagherilab-working"
        self.load_prefix = "jcain/emulation_sims/outputs"
        self.save_prefix = "jevarts/encoder/parsed_sims"

    def parse_files_to_csv(self, keys: list[str]) -> None:
        for key in keys:
            print(key)

            # Cell parsing
            cell_files = list_s3_files(
                self.bucket, self.load_prefix, include=[key], exclude=[".GRAPH"]
            )
            file_pd = pd.DataFrame()
            for file_name in cell_files:

                local_path = f"{self.data_dir}/{file_name}"
                s3_path = f"{self.load_prefix}/{file_name}"
                download_file(self.bucket, s3_path, local_path)
                parsed_df = self._parse_cells(local_path)     
                unique_seeds = parsed_df['seed'].unique()
                unique_times = parsed_df['time'].unique()
                for seed in unique_seeds:
                    for time in unique_times:
                        df = parsed_df[(parsed_df['seed'] == seed) & (parsed_df['time'] == time)]
                        df.drop(columns=['seed', 'time'], inplace=True)
                        df.rename(columns=lambda x: x.lower(), inplace=True)
                        time = str(int(10 * time)).zfill(3)
                        name = f"{key}{seed}_{time}"
                        df.to_csv(f"{self.data_dir}/{name}.csv", index=False)
                        upload_file(f"{self.data_dir}/{name}.csv", self.bucket, f"{self.save_prefix}/individual/{name}.csv")
                        os.remove(f"{self.data_dir}/{name}.csv")

                file_pd = pd.concat([file_pd, parsed_df], ignore_index=True)
                os.remove(local_path)

            file_pd.to_csv(f"{self.data_dir}/{key}cell.csv", index=False)
            upload_file(f"{self.data_dir}/{key}cell.csv", self.bucket, f"{self.save_prefix}/raw/{key}cell.csv")
            os.remove(f"{self.data_dir}/{key}cell.csv")

            # Vasculature parsing
            graph_files = list_s3_files(self.bucket, self.load_prefix, include=[key, ".GRAPH"])
            file_pd = pd.DataFrame()
            for file_name in graph_files:
                local_path = f"{self.data_dir}/{file_name}"
                s3_path = f"{self.load_prefix}/{file_name}"
                download_file(self.bucket, s3_path, local_path)
                parsed_df = self._parse_graph(local_path)
                unique_seeds = parsed_df['seed'].unique()
                unique_times = parsed_df['time'].unique()
                for seed in unique_seeds:
                    for time in unique_times:
                        df = parsed_df[(parsed_df['seed'] == seed) & (parsed_df['time'] == time)]
                        df.drop(columns=['seed', 'time'], inplace=True)
                        df.rename(columns=lambda x: x.lower(), inplace=True)
                        df.rename(columns={'code': 'type'}, inplace=True)
                        time = str(int(10 * time)).zfill(3)
                        name = f"{key}{seed}_{time}_graph"
                        df.to_csv(f"{self.data_dir}/{name}.csv", index=False)
                        upload_file(f"{self.data_dir}/{name}.csv", self.bucket, f"{self.save_prefix}/individual/{name}.csv")
                        os.remove(f"{self.data_dir}/{name}.csv")

                file_pd = pd.concat([file_pd, parsed_df], ignore_index=True)
                os.remove(local_path)

            # file_pd.to_csv(f"{self.data_dir}/{key}graph.csv", index=False)
            # upload_file(f"{self.data_dir}/{key}graph.csv", self.bucket, f"{self.save_prefix}/raw/{key}graph.csv")
            # os.remove(f"{self.data_dir}/{key}graph.csv")

    def parse_graph_metrics_to_csv(self, keys: list[str]) -> None:
        graph_parser = GraphParser()
        for key in keys:
            graph_parser.parse_graph_metrics(key)

    def parse_cell_metrics_to_csv(self, keys: list[str]) -> None:
        cell_parser = CellParser()
        for key in keys:
            cell_parser.parse_cell_metrics(key)

    def _parse_cells(self, sim_file) -> pd.DataFrame:
        parsed_data = []
        parsed_df = pd.DataFrame()

        with open(sim_file, "r") as f:
            sim_json = json.load(f)

        for time_index, timepoint in enumerate(self.timepoints):
            sim_timepoint = sim_json["timepoints"][time_index]["cells"]

            for location, cells in sim_timepoint:
                u, v, w, z = map(int, location[:4])

                population_counts = [0] * self.num_populations
                state_counts = [0] * self.num_states
                total_volume = 0
                total_cells = 0

                for cell in cells:
                    population, state, position, volume, cell_cycle = cell[1:6]
                    cycle = np.round(np.mean(cell_cycle)) if cell_cycle else 0

                    population_counts[int(population)] += 1
                    state_counts[int(state)] += 1
                    total_volume += volume
                    total_cells += 1

                data_list = [
                    timepoint,
                    u,
                    v,
                    w,
                    z,
                    position,
                    total_volume,
                    population_counts[0],
                    population_counts[1],
                    total_cells,
                    *state_counts,
                    cycle,
                    self._get_seed(sim_file),
                ]

                parsed_data.append(data_list)

        columns = [
            "timepoint",
            "u",
            "v",
            "w",
            "z",
            "position",
            "volume",
            "pop_healthy",
            "pop_cancer",
            "count",
            "state_neutral",
            "state_apoptotic",
            "state_quiescent",
            "state_migrating",
            "state_proliferating",
            "state_senescent",
            "state_necrotic",
            "cycle",
            "seed",
        ]

        if parsed_data:
            parsed_df = pd.DataFrame(parsed_data, columns=columns)

        return parsed_df

    def _parse_graph(self, graph_file) -> pd.DataFrame:
        parsed_data = []
        parsed_df = pd.DataFrame()

        with open(graph_file, "r") as f:
            graph_json = json.load(f)

        seed = graph_json["seed"]
        csv_data = [
            "seed",
            "time",
            "fromx",
            "fromy",
            "fromz",
            "frompressure",
            "fromoxygen",
            "tox",
            "toy",
            "toz",
            "topressure",
            "tooxygen",
            "CODE",
            "RADIUS",
            "LENGTH",
            "WALL",
            "SHEAR",
            "CIRCUM",
            "FLOW",
        ]

        for timepoint in graph_json["timepoints"]:
            time = timepoint["time"]

            for edge in timepoint["graph"]:
                fromx, fromy, fromz, frompressure, fromoxygen = edge[0][:5]
                tox, toy, toz, topressure, tooxygen = edge[1][:5]
                code, radius, length, wall, shear, circum, flow = edge[2][:7]

                data_list = [
                    seed,
                    time,
                    fromx,
                    fromy,
                    fromz,
                    frompressure,
                    fromoxygen,
                    tox,
                    toy,
                    toz,
                    topressure,
                    tooxygen,
                    code,
                    radius,
                    length,
                    wall,
                    shear,
                    circum,
                    flow,
                ]

                parsed_data.append(data_list)

        parsed_df = pd.DataFrame(parsed_data, columns=csv_data)

        return parsed_df

    def _get_seed(self, f) -> None:
        f_path = ntpath.dirname(f)
        base = ntpath.basename(f)
        remove_extension = os.path.splitext(base)[0]
        remove_suffix = os.path.splitext(remove_extension)[0]
        seed = int(remove_suffix[-2:])
        extension = os.path.splitext(base)[1]
        key = remove_suffix[:-3]

        return seed
    

if __name__ == "__main__":
    arcade_parser = ArcadeParser()
    arcade_parser.parse_files_to_csv(["C_Lav_", "C_Lava_", "C_Lvav_", "C_Sav_", "C_Savav_", "CH_Lav_", "CH_Lava_", "CH_Lvav_", "CH_Sav_", "CH_Savav_"])
    # arcade_parser.parse_cell_metrics_to_csv(["C_Lav_"])
    # arcade_parser.parse_graph_metrics_to_csv(["C_Lav_"])