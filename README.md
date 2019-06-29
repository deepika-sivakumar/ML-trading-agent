# A Machine Learning Trading Agent: 

### Problem Description: 
To implement a trading agent using an ensemble(**Bootstrapping**) of **Random Forest Learners** that would learn a strategy using past stock price data and **generate stock orders** for each day and give a visualization of the growth of the Portfolio with those predictions. 

### Dataset: 
Stock prices of various symbols from years 2000 – 2012 from the "UCI machine Learning Repository".

### Solving the problem:
* Framed the trading problem as a **Classification Problem** whether to classify the stock symbol as to BUY/SELL/DO_NOTHING for each particular day.
* After assessing various supervised learning methods, I chose to use Random Forest trees with Bootstrapping for this problem.
* **Feature Selection:** Chose the technical indicators Simple moving average(SMA), Bollinger Band Percentage(BBP) and momentum as the features.
* Split data for training and testing, handle missing values and discretization of data.
* Train and test the model
* Evaluate the model and generate data visualization of results.

### Evaluation & Results: 
* **Machine meets Manual Strategy:** I devised an interesting experiment to evaluate my ML model. I created a manual strategy to decide to buy/sell a stock based on researched technical indicator thresholds. The machine learner yielded about more than a **100% increase** in the **portfolio value** compared to the manual strategy.
* I evaluated my model against a **Benchmark** - Starting with same amount of cash, investing in that particular stock symbol on the first day and holding that position. The machine learner yielded a whopping **140% increase** in portfolio value compared to the benchmark.
