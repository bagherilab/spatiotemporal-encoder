import matplotlib.pyplot as plt


def line_plot(
    data: dict[str, list[float]], title: str, uuid: str, x_label: str = "", y_label: str = ""
) -> None:
    """Creates a line plot of the data"""
    plt.figure()
    for _, values in data.items():
        plt.plot(range(len(values)), values)
    plt.legend(data.keys())
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f"results/{uuid}/figures/{title}.png")
