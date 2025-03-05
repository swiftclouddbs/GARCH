import gradio as gr
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from arch import arch_model

def fetch_and_plot_garch(ticker, start_date, end_date, forecast_horizon, p, q):
    try:
        # Fetch stock data
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            return "Error: No data found. Check the ticker and date range."
        
        # Calculate log returns
        data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        returns = data['Log Returns'].dropna()
        
        # Fit GARCH(p, q) model
        model = arch_model(returns, vol='Garch', p=p, q=q)
        fitted_model = model.fit(disp='off')
        
        # Forecast volatility
        forecast = fitted_model.forecast(horizon=forecast_horizon)
        forecast_vol = np.sqrt(forecast.variance.iloc[-1])
        forecast_dates = pd.date_range(returns.index[-1], periods=forecast_horizon+1, freq='D')[1:]
        
        # Create interactive Plotly figure
        fig = go.Figure()
        
        # Historical volatility
        fig.add_trace(go.Scatter(x=returns.index, y=fitted_model.conditional_volatility, 
                                 mode='lines', name='Historical Volatility',
                                 hoverinfo='x+y'))
        
        # Forecasted volatility
        fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_vol, 
                                 mode='lines', name='Forecasted Volatility',
                                 line=dict(dash='dash', color='red'),
                                 hoverinfo='x+y'))
        
        # Forecast start line
        fig.add_trace(go.Scatter(x=[returns.index[-1], returns.index[-1]],
                                 y=[min(fitted_model.conditional_volatility), max(fitted_model.conditional_volatility)],
                                 mode='lines', name='Forecast Start',
                                 line=dict(dash='dot', color='black')))
        
        # Layout settings
        fig.update_layout(title=f'GARCH({p},{q}) Volatility Forecast for {ticker}',
                          xaxis_title='Date',
                          yaxis_title='Volatility',
                          template='plotly_white')
        
        return fig
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks() as app:
    gr.Markdown("# GARCH Volatility Forecasting")
    with gr.Row():
        ticker = gr.Textbox(label="Stock Ticker (e.g., AAPL)")
        start_date = gr.Textbox(label="Start Date (YYYY-MM-DD)")
        end_date = gr.Textbox(label="End Date (YYYY-MM-DD)")
    with gr.Row():
        p = gr.Slider(minimum=1, maximum=5, step=1, label="ARCH Parameter (p)", value=1)
        q = gr.Slider(minimum=1, maximum=5, step=1, label="GARCH Parameter (q)", value=1)
    forecast_horizon = gr.Slider(minimum=1, maximum=30, step=1, label="Forecast Horizon (Days)", value=10)
    submit = gr.Button("Run GARCH Analysis")
    output = gr.Plot()
    submit.click(fetch_and_plot_garch, inputs=[ticker, start_date, end_date, forecast_horizon, p, q], outputs=output)

app.launch()
