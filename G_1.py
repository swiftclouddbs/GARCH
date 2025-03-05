import gradio as gr
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        
        # Plot historical and forecasted volatility
        plt.figure(figsize=(10,5))
        plt.plot(returns.index, fitted_model.conditional_volatility, label='Historical Volatility')
        plt.axvline(x=returns.index[-1], color='r', linestyle='--', label='Forecast Start')
        plt.plot(pd.date_range(returns.index[-1], periods=forecast_horizon+1, freq='D')[1:], 
                 forecast_vol, label='Forecasted Volatility', linestyle='dashed')
        plt.title(f'GARCH({p},{q}) Volatility Forecast for {ticker}')
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.grid()
        
        # Save and return the plot
        plt.savefig("garch_plot.png")
        return "garch_plot.png"
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
    output = gr.Image()
    submit.click(fetch_and_plot_garch, inputs=[ticker, start_date, end_date, forecast_horizon, p, q], outputs=output)

app.launch()
