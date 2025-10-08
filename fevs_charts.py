# fevs_charts.py
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def gauge(title: str, value: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"thickness": 0.35}}
    ))
    fig.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10), title=title)
    return fig

def stacked_bar(df: pd.DataFrame, label: str, pos: str, neu: str, neg: str, title: str):
    fig = go.Figure()
    fig.add_bar(name="Positive", x=df[label], y=df[pos])
    fig.add_bar(name="Neutral",  x=df[label], y=df[neu])
    fig.add_bar(name="Negative", x=df[label], y=df[neg])
    fig.update_layout(
        barmode="stack", title=title, yaxis_title="Percent", xaxis_title=None,
        height=420, margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

def response_rate_line(rr_df: pd.DataFrame):
    """Render a bar chart of response rate by fiscal year."""

    rr_df = rr_df.sort_values("FY").dropna(subset=["FY", "rate"])
    if rr_df.empty:
        return px.bar()

    tick_vals = rr_df["FY"].astype(int).tolist()
    fig = px.bar(rr_df, x="FY", y="rate")
    fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
    fig.update_layout(
        yaxis_title="Response Rate (%)",
        xaxis_title=None,
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig.update_yaxes(range=[0, 100])
    if tick_vals:
        fig.update_xaxes(
            tickmode="array",
            tickvals=tick_vals,
            ticktext=[str(val) for val in tick_vals],
        )
    return fig
