# =============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
import plotly.graph_objects as go
import plotly.express as px

matplotlib.use('agg')
import matplotlib.pyplot as plt


class Visualizers:

    @staticmethod
    def histogram_visualizer(golden_data, target_data, dest):

        # Calculate mean, median, and standard deviation
        mean_golden = np.mean(golden_data)
        mean_target = np.mean(target_data)
        median_golden = np.median(golden_data)
        median_target = np.median(target_data)
        std_golden = np.std(golden_data)
        std_target = np.std(target_data)

        # Create a histogram trace for golden data
        trace_golden = go.Histogram(x=golden_data, name='Golden Data', opacity=0.6)

        # Create a histogram trace for target data
        trace_target = go.Histogram(x=target_data, name='Target Data', opacity=0.6)

        # Create the layout
        layout = go.Layout(
            title='Comparison of Golden Data vs. Target Data',
            xaxis=dict(title='Tensor value Range'),
            yaxis=dict(title='Frequency'),
            barmode='overlay'  # Overlay histograms
        )

        # Create the figure
        fig = go.Figure(data=[trace_golden, trace_target], layout=layout)

        # Add annotations for mean, median and standard deviation

        info_text = f"Golden: Mean: {mean_golden:.2f}\nMedian: {median_golden:.2f}\nStd Dev: {std_golden:.2f}<br>" \
                    f"Target: Mean: {mean_target:.2f}\nMedian: {median_target:.2f}\nStd Dev: {std_target:.2f}"

        fig.add_annotation(xref='paper', yref='paper', x=1, y=1, text=info_text, showarrow=False,
                           align='left')

        # Save the plot to an HTML file
        fig.write_html(dest)

    @staticmethod
    def diff_visualizer(golden_data, target_data, dest):
        # Create a DataFrame with the difference between target and golden data
        df = pd.DataFrame({
            'Index': np.arange(len(golden_data)),
            'Difference': golden_data - target_data,
            'Golden Data': golden_data,
            'Target Data': target_data
        })

        # Create the scatter plots
        fig = px.scatter(title='Golden vs. Target Values')

        fig.add_scatter(x=df['Index'], y=df['Golden Data'], mode='markers', name='Golden Data',
                        marker=dict(color='green'),
                        hovertemplate="Index: %{x}<br>Golden Data: %{y}")

        fig.add_scatter(x=df['Index'], y=df['Target Data'], mode='markers', name='Target Data',
                        marker=dict(color='blue'), hovertemplate="Index: %{x}<br>Target Data: %{y}")

        # Create the line plot for the difference
        fig.add_scatter(
            x=df['Index'], y=df['Difference'], mode='lines', name='Difference',
            line=dict(color='red'), hovertemplate=
            "Index: %{x}<br>Golden Data: %{customdata[0]}<br>Target Data: %{customdata[1]}<br>Difference: %{y:.2f}",
            customdata=np.column_stack((df['Golden Data'], df['Target Data'])))

        fig.update_xaxes(title_text='Index')
        fig.update_yaxes(title_text='Value')
        fig.write_html(dest)

    @staticmethod
    def cdf_visualizer(golden_data, target_data, dest):

        golden_data_hist, golden_data_edges = np.histogram(golden_data, 256)
        golden_data_centers = (golden_data_edges[:-1] + golden_data_edges[1:]) / 2
        golden_data_cdf = np.cumsum(golden_data_hist / golden_data_hist.sum())

        target_data_hist, target_data_edges = np.histogram(target_data, 256)
        target_data_centers = (target_data_edges[1:] + target_data_edges[:-1]) / 2
        target_data_cdf = np.cumsum(target_data_hist / target_data_hist.sum())

        # Create the Plotly figure
        fig = go.Figure()

        # Add traces for CDFs
        fig.add_trace(
            go.Scatter(x=golden_data_centers, y=golden_data_cdf, mode='lines',
                       name='Golden data CDF', line=dict(color='green')))
        fig.add_trace(
            go.Scatter(x=target_data_centers, y=target_data_cdf, mode='lines',
                       name='Target data CDF', line=dict(color='blue')))

        # Customize the layout
        fig.update_layout(
            title='CDF Comparison: Golden Data vs. Target Data',
            xaxis_title='Data Values',
            yaxis_title='Cumulative Probability',
            legend=dict(x=0.85, y=0.8),
        )
        # Save the plot to an HTML file
        fig.write_html(dest)

    @staticmethod
    def distribution_visualizer(data, dest, target_min, target_max, calibrated_min, calibrated_max):

        plt.figure()
        pd.DataFrame(data, columns=['Target data distribution']).plot(kind='kde', color='blue')

        plt.axvline(x=target_min, color='red')
        plt.text(target_min, 0, f'{target_min:.2f}', rotation=90)

        plt.axvline(x=target_max, color='red')
        plt.text(target_max, 0, f'{target_max:.2f}', rotation=90)

        plt.axvline(x=calibrated_min, color='green')
        plt.text(calibrated_min, 0, f'{calibrated_min:.2f}', rotation=90)

        plt.axvline(x=calibrated_max, color='green')
        plt.text(calibrated_max, 0, f'{calibrated_max:.2f}', rotation=90)

        target_data_distribution = matplotlib.patches.Patch(color='blue',
                                                            label='Target data distribution')
        target_hightlight = matplotlib.patches.Patch(color='red', label='Target min/max')
        calibrated_hightlight = matplotlib.patches.Patch(color='green', label='Calibrated min/max')
        plt.legend(handles=[target_data_distribution, target_hightlight, calibrated_hightlight])

        plt.savefig(dest)
        plt.close()
