<div align="center">
<img align="center" src="https://raw.githubusercontent.com/AI4Finance-Foundation/FinRL/master/figs/FinRL_Tutorials.png">
</div>

## About This Project

This repository provides a collection of hands-on tutorials for the **FinRL** ecosystem, a deep reinforcement learning (DRL) framework designed to automate trading in quantitative finance.

**Mission**: To create hundreds of user-friendly demos that help users apply deep reinforcement learning to financial tasks.

FinRL is a framework that provides a full pipeline for developing and testing DRL-based trading strategies. These tutorials are designed to guide you through the process, from basic introductions to advanced applications and practical implementations.

Note that we provide tutorials for [FinRL-meta](https://github.com/AI4Finance-Foundation/FinRL-Meta/tree/master/tutorials) and [FinRL](https://github.com/AI4Finance-Foundation/FinRL/tree/master/tutorials).


## File Structure

### **1-Introduction**		
**notebooks for beginners, introduction step-by-step**

+ **FinRL_StockTrading_NeurIPS_2018:** first tutorial notebook that trades Dow 30 using 5 DRL algorithms.
+ **FinRL_PortfolioAllocation_NeurIPS_2020:** provides basic settings to do portfolio allocation on Dow 30.
+ **FinRL_StockTrading_Fundamental:** merges fundamental indicators in earnings reports such as 'ROA', 'ROE', 'PE' with technical indicators.

### **2-Advance**
**notebooks for intermediate users**

+ **FinRL_PortfolioAllocation_Explainable_DRL:** this notebook uses an empirical approach to explain the strategies of DRL agents for the portfolio management task. 1) it uses feature weights of a trained DRL agent, 2) histogram of correlation coefficient, 3) Z-statistics to explain the strategies.
+ **FinRL_Compare_ElegantRL_RLlib_Stablebaseline3:** compares popular DRL libraries, namely ElegantRL, RLlib and Stablebaseline3.
+ **FinRL_Ensemble_StockTrading_ICAIF_2020:** uses an ensemble strategy to combine multiple DRL agents to form an adaptive one to improve the robustness.

### **3-Practical**
**notebooks for users to explore paper trading and more financial markets**
+ **FinRL_PaperTrading_Demo:** paper trading using FinRL through Alpaca.
+ **FinRL_MultiCrypto_Trading:** trading top 10 market cap cryptocurrencies.
+ **FinRL_China_A_Share_Market:** trading on China A Share market.

### **4-Optimization**
**notebooks for users interested in hyperparameter optimizations**

### **5-Others** 
**other related notebooks**

## Community and Contribution

This is an open-source project that thrives on community contributions. We welcome you to get involved!

*   **[Contribution Guidelines](./Contributing.md)**: Learn how to contribute, our guiding principles, and PR guidelines.
*   **File an Issue**: If you see something, [say something](https://guides.github.com/features/issues/)!

## Further Reading

For those who want to dive deeper into the theory and research behind financial reinforcement learning, we maintain curated lists of resources:

*   **[Awesome FinRL](./Awesome_FinRL.md)**: A curated list of awesome deep reinforcement learning strategies, tools, and resources for finance.
*   **[FinRL Papers](./FinRL_papers.md)**: A list of academic papers from the AI4Finance community and the Columbia research team.
