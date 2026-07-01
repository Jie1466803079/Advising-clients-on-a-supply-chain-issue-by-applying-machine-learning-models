# Advising Gala Groceries on a Supply-Chain Stock Problem with Machine Learning

> An end-to-end data-science project that advises grocery retailer Gala Groceries on managing product stock levels — spanning exploratory data analysis, data merging and feature engineering, a Random Forest stock-prediction model, and a reusable Python model-training module for the ML engineering team.

`Python` · `scikit-learn` · `Random Forest regression`

## Problem
Gala Groceries wanted to better manage the stock of the items it sells. In the exploratory phase the business question was stated broadly as *"How to better stock the items that they sell"*, which the analysis found too broad to answer from the available data. It was therefore narrowed to a concrete predictive question: *"Can we accurately predict the stock levels of products, based on sales data and sensor data, on an hourly basis in order to more intelligently procure products from our suppliers."*

## Approach
1. **Exploratory data analysis (Step 1).** Loaded a one-week sample of sales data (`sample_sales_data.csv`, 7,829 rows × 9 columns) in Python / Google Colab; computed descriptive statistics and visualised distributions and correlations with pandas and seaborn; concluded the sample (one store, one week) was too small and the business question too broad, and recommended more data, a narrower problem statement, and more features.
2. **Planning (Step 2).** A "Plan of work" presentation framing the data-modeling phase.
3. **Model building & interpretation (Step 3).** Merged three hourly datasets — sales (7,829 rows), sensor stock levels (15,000 rows) and storage temperature (23,890 rows) — by resampling each `timestamp` to the hour and aggregating; corrected data types, filled missing sales `quantity` with 0, engineered day/hour features from the timestamp and one-hot-encoded product `category`, yielding a 10,845-row modeling table with 28 predictor features. Trained a `RandomForestRegressor` to predict `estimated_stock_pct` using 10-fold cross-validation with a 75/25 train/test split and `StandardScaler`, evaluated with mean absolute error (MAE), and interpreted the model's feature importances.
4. **Machine learning production (Step 4).** Packaged the training workflow into a reusable Python module (`load_data`, `create_target_and_predictors`, `train_algorithm_with_cross_validation`, `run`) that trains the Random Forest with k-fold cross-validation (`K = 10`, `SPLIT = 0.75`) and prints per-fold and average MAE metrics for the ML engineering team.

## Results
- **EDA findings (as stated in the notebook):** most products have a unit price below \$10, and most transactions total below \$30; both unit price and quantity correlate with the transaction total, with unit price the stronger driver. The sample was flagged as insufficient (one store, one week).
- **Model:** a Random Forest regressor predicting hourly `estimated_stock_pct`, evaluated by MAE over 10 cross-validation folds. The notebook reports the MAE was consistent across folds — indicating a robust, stable model — but characterises overall accuracy as modest (it notes the target variable averages around 0.51 and describes the accuracy as roughly 50%), and recommends reporting back to the business that more data and further feature engineering are needed. *(The training cell's numeric MAE outputs were not saved in the notebook, so no exact MAE value is stated here.)*
- **Feature importance (as stated in the notebook):** unit price and storage temperature were important predictors of stock, and the hour of day was also important, whereas the product categories were not that important.

## Repository structure
```
Step 1_Exploratory Data Analysis/
    eda.ipynb
    sample_sales_data.csv
Step 2_Data Modeling/
    Plan of work.pptx
Step 3_Model Building and Interpretation/
    modeling.ipynb
    sales.csv
    sensor_stock_levels.csv
    sensor_storage_temperature.csv
Step 4_Machine Learning Production/
    Machine Learning Production.ipynb
```

## Tech
Python · pandas · seaborn · matplotlib · NumPy · scikit-learn (`RandomForestRegressor`, `train_test_split`, `StandardScaler`, `mean_absolute_error`) · Google Colab
