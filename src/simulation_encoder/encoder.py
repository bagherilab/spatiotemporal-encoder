from loader import Loader
from parser import Arcade_Parser


def main():
    parser = Arcade_Parser()
    parser.parse_files_to_csv(key="C_Lav_")
    # parser.parse_graph_metrics_to_csv(key="C_Lav_")
    # parser.parse_cell_metrics_to_csv(key="C_Lav_")

if __name__ == "__main__":
    main()
