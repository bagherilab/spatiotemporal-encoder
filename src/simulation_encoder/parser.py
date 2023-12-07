import os
import ntpath

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

from utils.s3_utils import download_file, list_s3_files


class Parser(ABC):
    @abstractmethod
    def parse_timepoint(self, sim_data, time_index, timepoint) -> None:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass


class Arcade_Parser(Parser):
    def __init__(self, key: str):
        self.timepoints = [((x / 2.0) - 1) for x in range(0, 31)]
        self.data = pd.DataFrame()
        self.key = key
        self.bucket = "bagherilab-working"
        self.object_prefix = "jcain/emulation_sims/outputs"
        self.parse_files()

    def parse_files(self) -> None:
        files = list_s3_files("arcade-simulations", self.key)
        for f in files:
            print(f)
            # self.parse_file(f)

    def parse_timepoint(self, sim_json, time_index, timepoint) -> None:
        parsed_data = []
        sim_timepoint = sim_json["timepoints"][time_index]["cells"]

        for location, cells in sim_timepoint:
            u = int(location[0])
            v = int(location[1])
            w = int(location[2])
            z = int(location[3])

            for cell in cells:
                population = cell[1]
                state = cell[2]
                position = cell[3]
                volume = np.round(cell[4])
                cycle = np.round(np.mean(cell[5]))

                data_list = [
                    timepoint,
                    u,
                    v,
                    w,
                    z,
                    position,
                    int(population),
                    int(state),
                    volume,
                    cycle,
                ]

                parsed_data.append(data_list)

        columns = [
            "timepoint",
            "u",
            "v",
            "w",
            "z",
            "position",
            "population",
            "state",
            "volume",
            "cycle",
        ]
        parsed_df = pd.DataFrame(parsed_data, columns=columns)
        return

    def save(self, path: str) -> None:
        self.data.to_csv(path, index=False)

    def _get_seed(self, f) -> None:
        f_path = ntpath.dirname(f)
        base = ntpath.basename(f)
        remove_extension = os.path.splitext(base)[0]
        remove_suffix = os.path.splitext(remove_extension)[0]
        seed = int(remove_suffix[-2:])
        extension = os.path.splitext(base)[1]
        key = remove_suffix[:-3]

        return seed
