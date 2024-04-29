import os
import matplotlib.pyplot as plt


def line_plot(
    data: dict[str, list[float]],
    title: str,
    uuid: str,
    model_name: str,
    x_label: str = "",
    y_label: str = "",
) -> None:
    """Creates a line plot of the data"""
    print(title)
    print(data)
    plt.figure()
    for _, values in data.items():
        plt.plot(range(len(values)), values)
    plt.legend(data.keys())
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    _create_dir(f"results/{uuid}/{model_name}/figures/")
    plt.savefig(f"results/{uuid}/{model_name}/figures/{title}.png")


def _create_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
