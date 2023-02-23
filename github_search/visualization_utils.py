import pandas as pd
import plotly.express as px


def make_2d_data_plot(data, text_label, cls):
    cls_numbering = {c: i for (i, c) in enumerate(set(cls))}
    source_df = pd.DataFrame(
        {
            "x": data[:, 0],
            "y": data[:, 1],
            "task": text_label,
            "area": cls,
            "color": [cls_numbering[c] for c in cls],
        }
    )
    plot = px.scatter(
        data_frame=source_df,
        x="x",
        y="y",
        hover_data=["task", "area"],
        color="area",
        width=1200,
        height=800,
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )

    return plot
