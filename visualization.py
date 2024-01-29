import plotnine as p
import pandas as pd
from mizani.formatters import date_format

colors = ["#00a082", "#FEE12B", "blue", "red", "orange"]


class Visualization:
    @staticmethod
    def lineplot(
        data: pd.DataFrame,
        x: str,
        y: str,
        color: str,
        title: str,
        xlabel: str,
        ylabel: str,
    ):

        plot = (
            p.ggplot(data)
            + p.aes(x=x, y=y, color=color)
            + p.geom_line(size=1)
            + p.scale_color_manual(values=colors[: data[color].nunique()])
            + p.labs(title=title, x=xlabel, y=ylabel)
            + p.scale_x_datetime(labels=date_format("%d-%m"))
            + p.theme(panel_grid_major=p.element_line(color="gray", size=0.2))
        )

        return plot
