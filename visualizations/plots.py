import os
import numpy as np
import torch

import plotly.graph_objects as go
import plotly.subplots as subplots
import plotly.io as pio
import plotly.express as px



pio.renderers.default = "browser"
pio.kaleido.scope.mathjax = None  # plotly issue https://github.com/plotly/plotly.py/issues/3469


def plot_sample_forecast(sample, fcast, feature_names=None, title=None, display=True):
    """Plots input, target, and forecast for one sample with an arbitrary number of features.
    Args:
        sample: tuple of torch.Tensors, (x1, x2), where x1 is input and x2 is target
        fcast: torch.Tensor, shape (pred_len, nb_of_features), forecast of sample
        feature_names: list of str, names of the features (optional)
        title: str, title of plot (optional)
        display: bool, if True, plot is displayed, else returned
    """
    x1, x2 = sample
    pred_x2 = np.squeeze(fcast)
    num_features = x1.shape[1]

    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(num_features)]

    assert len(feature_names) == num_features, "Number of feature names must match the number of features"

    colors = px.colors.sample_colorscale("turbo", [n/(num_features - 1) for n in range(num_features)])

    fig = go.Figure()

    for i in range(num_features):
        color = colors[i]

        # Plot x1 (input)
        fig.add_trace(go.Scatter(
            x=np.arange(x1.shape[0]),
            y=x1[:, i],
            name=f"{feature_names[i]} (Input)",
            mode="lines",
            line=dict(color=color)
        ))

        # Plot x2 (target)
        fig.add_trace(go.Scatter(
            x=np.arange(x1.shape[0], x1.shape[0] + x2.shape[0]),
            y=x2[:, i],
            name=f"{feature_names[i]} (Target)",
            mode="lines",
            line=dict(color=color)
        ))

        # Plot fcast (forecast)
        fig.add_trace(go.Scatter(
            x=np.arange(x1.shape[0], x1.shape[0] + pred_x2.shape[0]),
            y=pred_x2[:, i],
            name=f"{feature_names[i]} (Forecast)",
            mode="lines",
            line=dict(color=color, dash="dash")
        ))

    fig.add_vline(x=x1.shape[0], line=dict(color="black", width=2, dash="dash"), 
                  annotation_text="Prediction Start", annotation_position="top right")  # TODO does not work for fcast overview?
    fig.update_xaxes(title_text='Time')
    fig.update_layout(width=800, height=500, title=title, 
                      font_family="Serif", font_size=14,
                      margin=dict(l=5, t=50, b=5, r=5))
    if display:
        fig.show()
    else:
        return fig


def add_forecast_to_subplot(fig, sample, fcast, feature_names, scenario, subplot_idx, n_rows, n_cols):
    fcast_plot = plot_sample_forecast(sample, fcast, feature_names=feature_names, display=False)
    row, col = (subplot_idx // n_cols) + 1, (subplot_idx % n_cols) + 1
    for trace in fcast_plot.data:
        fig.add_trace(trace, row=row, col=col)


def fcast_overview(datamodule, model, idx=0, title=None, save_path=None):
    model = model.to(model.visualization_device)
    model.eval()
    datasets = datamodule.ds_test_dict

    n_rows, n_cols = 3, 3
    fig = subplots.make_subplots(rows=n_rows, cols=n_cols, shared_xaxes=True, vertical_spacing=0.02)

    for i, (scenario, ds) in enumerate(datasets.items()):
        if i >= n_rows * n_cols:
            break
        sample = ds[idx]
        x = torch.tensor(sample[0]).unsqueeze(0)
        fcast = model(x).cpu().detach().numpy()
        feature_names = datamodule.ds_test_dict["normal"].df.columns
        add_forecast_to_subplot(fig, sample, fcast, feature_names, scenario, i, n_rows, n_cols)

        fig.update_yaxes(title_text=scenario, row=(i//n_cols)+1, col=(i%n_cols)+1, title_standoff=0.5)
    
    fig.update_layout(showlegend=False, title=f"Forecasts for each scenario by {title}")
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.write_image(os.path.join(save_path, f"fcast_{title}.png"), width=1200, height=800)
    else:
        fig.show()


# def plot_performance_scores(performance_scores, metric_name):
#     fig = go.Figure()
#     quantile_cols = ["min_perf_score", "q25_perf_score", "q50_perf_score", "q75_perf_score", "max_perf_score"]
#     colors = ["rgba(150,0,0,0.4)", "rgba(150,0,0,0.3)", "rgba(150,0,0,0.2)", "rgba(150,0,0,0.1)"]
#     # Plot the worst performance score
#     fig.add_trace(go.Scatter(
#         x=performance_scores.index,
#         y=performance_scores["min_perf_score"],
#         mode="lines",
#         line=dict(width=3.5, color="red"), 
#         name="min_perf_score"
#     ))
#     # Plot the other performance scores    
#     for i in range(1, len(performance_scores.columns)):
#         fig.add_trace(go.Scatter(
#             x=performance_scores.index,
#             y=performance_scores[quantile_cols[i]],
#             mode="lines",
#             fill="tonexty",
#             fillcolor=colors[i-1],
#             line=dict(width=0, color=colors[i-1]),
#             name=quantile_cols[i]
#         ))
#     fig.update_layout(
#         title=f"Performance Scores Across Severity Levels Using {metric_name}",
#         xaxis_title="Severity Level",
#         yaxis_title="Performance Score",
#         yaxis_range=[0, 1],
#         font_family="Serif", 
#         font_size=14,
#         margin=dict(l=5, t=50, b=5, r=5),
#         showlegend=True,
#         legend_traceorder="normal",
#     )
#     return fig


def plot_scenario_performance(result_df, metric_name, model_architecture):
    scenario_order = [
        "Drift", "DyingSignal", "Noise", "FlatSensor", "MissingData",
        "FasterSampling", "SlowerSampling", "Outlier", "WrongDiscreteValue", "OscillatingSensor"
    ]
    fig = go.Figure()
    df = result_df.copy(deep=True)
    df["scenario"] = df["scenario"].apply(
        lambda s: "".join(word.capitalize() for word in s.split("_"))
    )
    for scenario in scenario_order:
        df_ = df[df["scenario"] == scenario]
        if len(df_) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=df_["severity"],
            y=df_["rel_perf"],
            mode="lines",
            name=scenario
        ))
    del df 
    fig.update_layout(
        width=600, height=350,
        title={"text": f"Relative Performance of {model_architecture} Across Severity Levels", "y": 0.95, "x": 0.5, "xanchor": "center", "yanchor": "top"},
        # title={"text": f"Relative Performance Based on {metric_name} Across Severity Levels for {model_architecture}", "y": 0.95, "x": 0.5, "xanchor": "center", "yanchor": "top"},
        xaxis_title="Severity Level",
        yaxis_title="Relative Performance",
        yaxis_range=[0, 1.1],
        font=dict(size=16),
        font_family="Serif", 
        title_font=dict(size=20),
        margin=dict(l=5, t=50, b=5, r=5),
        showlegend=True,
        legend_traceorder="normal",
        template="plotly_white"
    )
    return fig


def plot_scenario_performance_hist(scenario_df, metric_name):
    scenario_order = [
        "Drift", "DyingSignal", "Noise", "FlatSensor", "MissingData",
        "FasterSampling", "SlowerSampling", "Outlier", "WrongDiscreteValue", "OscillatingSensor"
    ]
    scenario_columns = [col for col in scenario_df.columns if col not in ["dataset", "model"]]
    df_melted = scenario_df.melt(
        id_vars=["dataset", "model"],
        value_vars=scenario_columns,
        var_name="scenario",
        value_name="rel_performance"
    )
    df_melted["scenario"] = df_melted["scenario"].apply(
        lambda s: "".join(word.capitalize() for word in s.split("_"))
    )
    performance_summary = df_melted.groupby(["scenario", "model"]).agg(
        mean_performance=("rel_performance", "mean"),
        std_performance=("rel_performance", "std")
    ).reset_index()
    model_order = ["DLinear", "MLP", "TCN", "LSTM", "GRU", "RIMs", "Transformer", "Informer", "Mamba"]
    fig = px.bar(
        performance_summary,
        x="scenario",
        y="mean_performance",
        color="model",
        barmode="group",
        error_y="std_performance",
        category_orders={"model": model_order, "scenario": scenario_order},
        labels={"mean_performance": "Mean Disturbance Robustness Score", "scenario": "Scenario", "model": "Model"}
    )
    fig.update_layout(
        width=1200, height=500,
        title={"text": f"Disturbance Robustness Scores Across Scenarios", "y": 0.95, "x": 0.5, "xanchor": "center", "yanchor": "top"},
        # title={"text": f"Disturbance Robustness Scores Based on {metric_name} Across Scenarios", "y": 0.95, "x": 0.5, "xanchor": "center", "yanchor": "top"},
        font=dict(size=20),
        font_family="Serif", 
        title_font=dict(size=24),
        xaxis=dict(
            title_font=dict(size=20),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            title_font=dict(size=20),
            tickfont=dict(size=16)
        ),
        margin={"t": 50},
        template="plotly_white"
    )
    return fig


def plot_performance_scores(performance_scores, metric_name):
    fig = go.Figure()
    quantile_cols = ["min_perf_score", "q25_perf_score", "q50_perf_score", "q75_perf_score", "max_perf_score"]
    colors = ["rgba(150,0,0,0.1)", "rgba(150,0,0,0.3)", "rgba(150,0,0,0.3)", "rgba(150,0,0,0.1)"]
    # Plot the worst performance score
    fig.add_trace(go.Scatter(
        x=performance_scores.index,
        y=performance_scores["min_perf_score"],
        mode="lines",
        line=dict(width=3.5, color="red"), 
        name="Minimum"
    ))
    # Plot the median performance score
    fig.add_trace(go.Scatter(
        x=performance_scores.index,
        y=performance_scores["q50_perf_score"],
        mode="lines",
        line=dict(width=2, color="gray", dash="dash"),
        name="Median"
    ))
    # Now plot interquartile ranges
    fig.add_trace(go.Scatter(
        x=performance_scores.index,
        y=performance_scores["min_perf_score"],
        mode="lines",
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=performance_scores.index,
        y=performance_scores["q25_perf_score"],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(150,0,0,0.1)",
        line=dict(width=0),
        name="Lower Quartile"
    ))
    fig.add_trace(go.Scatter(
        x=performance_scores.index,
        y=performance_scores["q75_perf_score"],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(150,0,0,0.3)",
        line=dict(width=0),
        name="Interquartile Range"
    ))
    fig.add_trace(go.Scatter(
        x=performance_scores.index,
        y=performance_scores["max_perf_score"],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(150,0,0,0.1)",
        line=dict(width=0),
        name="Upper Quartile"
    ))
    fig.update_layout(
        title=f"Relative Performance Across Severity Levels",
        # title=f"Relative Performance Across Severity Levels Using {metric_name}",
        xaxis_title="Severity Level",
        yaxis_title="Relative Performance",
        yaxis_range=[0, 1],
        font_family="Serif", 
        font_size=14,
        margin=dict(l=5, t=50, b=5, r=5),
        showlegend=True,
        legend_traceorder="normal",
    )
    return fig