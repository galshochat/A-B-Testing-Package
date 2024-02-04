import plotnine as p
import pandas as pd
from mizani.formatters import date_format
from rich.table import Table
from rich.console import Console

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
            + p.theme(
                axis_line_x=p.element_line(color="gray", size=0.7),
                axis_line_y=p.element_line(color="gray", size=0.7),
                panel_grid_minor=p.element_line(color="gray", size=0.3),
            )
        )

        return plot

    @staticmethod
    def scorecard(data: pd.DataFrame, title: str):
        table = Table(title=title, leading=1, expand=True)
        for column in data.keys():
            table.add_column(column)
        for _idx, row in data.iterrows():
            values = row.values
            for ind, value in enumerate(values):
                if isinstance(value, bool):
                    values[ind] = str(value)
                elif value is None:
                    values[ind] = str(value)
                elif isinstance(value, str):
                    values[ind] = value
                else:
                    values[ind] = str(round(value, 4))

            table.add_row(*values)
        console = Console()
        return console.print(table)
