# Using Microsoft’s Autogen for Python Tool Calling Agents

Microsoft’s Autogen agent framework is excellent but it was designed to work with OpenAI’s models. Here we use Ollama with the **qwen2.5:14b** and a slightly modified version of a [Microsoft example using Python and Matplotlib](https://microsoft.github.io/autogen/0.2/docs/topics/non-openai-models/local-ollama/).

I experimented with several local models using Ollama with mediocre results but the larger **qwen2.5:14b** model works very well. If you are running on a Mac you will need an Apple Silicon chip with 16G of memory to run this model.

```python
from autogen import AssistantAgent, UserProxyAgent

# Requirements:
# pip install autogen ollama fix_busted_json yfinance matplotlib

config_list = [
 {
    "model": "qwen2.5:14b",   # Choose a model that supports tool calling
    "api_type": "ollama",     # Specify Ollama as the API type
    "client_host": "http://localhost:11434",  # local Ollama server
    "api_key": "fakekey",
    "native_tool_calls": True # Enable native tool calling 
 }
]

# Create the AssistantAgent using the local model config
assistant = AssistantAgent("assistant",
                           llm_config={"config_list": config_list})

# Create the UserProxyAgent; adjust code_execution_config as needed.
user_proxy = UserProxyAgent(
    "user_proxy",
    code_execution_config={"work_dir": "coding", "use_docker": False}
)

# Initiate an automated chat between the agents.
user_proxy.initiate_chat(
    assistant,
    message="Plot a chart of NVDA and TESLA stock price change YTD."
)
```

![KGN query results](images/autogen1.jpg)

Here is a partial listing of the output:

```text
$ python autogen_python_example.py  

user_proxy (to assistant):

Plot a chart of NVDA and TESLA stock price change YTD.

-------------------------------------------------------------
assistant (to user_proxy):

To plot a chart of NVDA (NVIDIA) and Tesla's stock price changes year-to-date, we will need to fetch their historical data from an API or a financial data source such as Yahoo Finance.

Let's use Python with the `yfinance` library to get the data and then plot it using `matplotlib`.

Here is a step-by-step plan:

1. **Install Required Libraries**: If not already installed, install `yfinance` for fetching stock data and `pandas` & `matplotlib` for processing and plotting the data.
2. **Fetch Stock Data**: Use `yfinance` to get the year-to-date historical data for NVDA and TSLA stocks.
3. **Plot Data**: Plot the closing prices of both stocks on a chart.

Let's start by fetching and plotting the stock price data with Python code:

``python
# filename: plot_stock_prices.py

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Fetch year-to-date historical data for NVDA and TSLA
stocks = ['NVDA', 'TSLA']
data = yf.download(stocks, start='2023-01-01')['Close']

# Plot the closing prices of both stocks
plt.figure(figsize=(14,7))
for stock in stocks:
    plt.plot(data.index, data[stock], label=stock)
    
plt.title('Year-to-date Stock Prices for NVDA and TSLA')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
``

You can execute the above Python script to generate the plot. Please make sure you have `yfinance`, `matplotlib`, and `pandas` installed in your environment. If not, install them by running:

``sh
pip install yfinance matplotlib pandas
``

After executing the code, you should see a chart showing the year-to-date closing prices for both NVDA and TSLA stocks.

Once you have executed this script, please inform me of any issues or if everything is plotted correctly.

-------------------------------------------------------------
Replying as user_proxy. Provide feedback to assistant. Press enter to skip and use auto-reply, or type 'exit' to end the conversation:

>>>>>>>> NO HUMAN INPUT RECEIVED.
>>>>>>>> USING AUTO REPLY...
>>>>>>>> EXECUTING CODE BLOCK 0 (inferred language is python)...
```

