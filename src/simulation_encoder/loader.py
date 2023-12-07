import json

import numpy as np
import pandas as pd

from parser import Arcade_Parser


class Loader:
    def __init__(self):
        pass

    def load(self, path) -> None:
        # with open(path, "r") as f:
        #     data = json.load(f)
        self.parse_file(path)
        # for time_index, timepoint in enumerate(self.timepoints):
        #     self.parse_timepoint(data, time_index, timepoint)

    def get(self) -> dict:
        return self.data


if __name__ == "__main__":
    loader = Loader()
    loader.load("data/ARCADE/VASCULAR_FUNCTION_C_Lav_00.json")
    print(loader.data)
    # data = loader.get()
    # print(data)
