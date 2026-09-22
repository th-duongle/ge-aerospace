# Stock Price Prediction Using Recurrent Neural Networks

### A comparative deep learning study of GE Aerospace, Lockheed Martin, and ExxonMobil

**Python | TensorFlow | Keras | Scikit-learn | Time-Series Forecasting**

## Project Overview

Can a neural network learn patterns in historical stock prices and use them to predict future closing prices? More importantly, how consistent are its predictions across companies operating in different industries?

This project explores these questions by implementing a Recurrent Neural Network (RNN) to forecast closing stock prices for three publicly traded companies:

* **GE Aerospace (NYSE: GE):** Aerospace and defense
* **Lockheed Martin (NYSE: LMT):** Aerospace and defense
* **ExxonMobil (NYSE: XOM):** Oil and gas

The project examines how a common neural network architecture performs when applied separately to three financial time series, with particular attention to prediction error, generalization, and differences in observed price behavior.

The original project was developed in February 2025.

## Research Objectives

1. Develop an RNN architecture capable of learning sequential patterns in historical stock prices.
2. Evaluate prediction performance using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).
3. Compare forecasting results for two companies in the aerospace and defense sector and one in the energy sector.
4. Investigate potential limitations of recurrent neural networks when applied to financial time-series forecasting.

## Dataset and Preprocessing

Historical daily stock-price data was used for each company.

The source datasets contain the following attributes:

| Feature | Description             |
| ------- | ----------------------- |
| Date    | Trading date            |
| Open    | Opening stock price     |
| High    | Highest trading price   |
| Low     | Lowest trading price    |
| Close   | Closing stock price     |
| Volume  | Number of shares traded |

The implemented model uses **closing prices as its sole input feature and prediction target**.

### Data preparation

The preprocessing pipeline consists of:

1. Converting the Date column into a datetime index.
2. Normalizing closing prices using Scikit-learn's MinMaxScaler.
3. Transforming the normalized data into overlapping sequences of 25 observations.
4. Splitting the sequences chronologically into 80% training data and 20% evaluation data.
5. Converting model predictions back into their original dollar-denominated scale for interpretation.

Each input sequence contains 25 consecutive closing-price observations, and its corresponding target is the closing price immediately following that sequence.

Separate instances of the same neural network architecture were trained for each company.

## Model Architecture

A stacked Simple Recurrent Neural Network was implemented using TensorFlow and Keras.

The architecture consists of:

| Layer     | Configuration               |
| --------- | --------------------------- |
| SimpleRNN | 50 units; returns sequences |
| Dropout   | 40%                         |
| SimpleRNN | 50 units                    |
| Dropout   | 40%                         |
| Dense     | 1 output unit               |

**Training configuration**

* Optimizer: Adam
* Loss function: Mean Squared Error (MSE)
* Epochs: 50
* Batch size: 64
* Input sequence length: 25 trading observations

The recurrent layers process sequential information, while dropout regularization is intended to reduce overfitting.

The final dense layer produces a single continuous value representing the predicted closing price.

## Results

### Closing-Price Predictions

The original report presents the following forecasts for March 13, 2025. The actual closing prices were subsequently supplied for comparison.

| Company         | Predicted Close | Actual Close | Absolute Error | Percentage Error |
| --------------- | --------------: | -----------: | -------------: | ---------------: |
| GE Aerospace    |         $190.96 |      $192.49 |          $1.53 |            0.79% |
| Lockheed Martin |         $455.21 |      $467.87 |         $12.66 |            2.71% |
| ExxonMobil      |         $107.01 |      $108.65 |          $1.64 |            1.51% |

*Percentage error is calculated as the absolute difference between predicted and actual prices divided by the actual closing price.*

The comparison provides an illustrative assessment of the three reported forecasts. The notebook's final prediction routine uses the last test input sequence rather than explicitly constructing a new sequence after the final observation. Accordingly, the forecast dates and alignment with the supplied actual prices should be independently verified before these figures are treated as a confirmed out-of-sample forecasting evaluation.

### Model Evaluation Metrics

The original notebook reports the following evaluation metrics:

| Company         |   RMSE |    MAE |
| --------------- | -----: | -----: |
| GE Aerospace    |  $3.41 |  $3.41 |
| Lockheed Martin | $34.30 | $24.93 |
| ExxonMobil      |  $3.21 |  $3.21 |

RMSE penalizes larger prediction errors more heavily, while MAE represents the average absolute difference between predicted and observed prices.

These metrics describe performance on the notebook's evaluation sequences and are distinct from the individual closing-price errors presented above.

Because the companies have different stock-price levels, their dollar-denominated errors should not be interpreted as a fully normalized comparison of forecasting performance.

## Key Findings

### 1. GE Aerospace

The model's reported closing-price forecast was $190.96, compared with a supplied actual price of $192.49.

The evaluation produced an RMSE and MAE of $3.41.

The training and validation loss curves exhibit fluctuations, suggesting that the model's learning behavior was not entirely stable. The plotted predictions also deviate from the observed price trajectory.

These observations illustrate the difficulty of producing consistent forecasts from a limited financial time series.

### 2. Lockheed Martin

Lockheed Martin exhibited the largest reported dollar-denominated evaluation errors, with an RMSE of $34.30 and an MAE of $24.93.

Its reported closing-price prediction was $455.21, compared with a supplied actual price of $467.87.

The original prediction plot shows a relatively flat-to-declining forecast while the observed prices increase.

This discrepancy suggests that the fitted model did not adequately capture the upward movement in the displayed evaluation period.

### 3. ExxonMobil

ExxonMobil produced an RMSE and MAE of $3.21.

Its reported closing-price prediction was $107.01, compared with a supplied actual price of $108.65.

The prediction plot shows the model following the general upward direction of the observed prices, although the predicted values remain below the actual values.

This suggests that the model captured some directional movement within the displayed evaluation period, but the available evidence is insufficient to establish whether this behavior generalizes to other periods or market conditions.

## Limitations

Several methodological limitations should be considered when interpreting the results.

**Limited historical observations:** The project uses a relatively small financial time series. Such datasets provide limited information for training neural networks and assessing their performance across different market conditions.

**Single-feature forecasting:** Only closing prices are used as model inputs. Trading volume, market conditions, and macroeconomic indicators are not incorporated into the implemented model.

**Data leakage:** The MinMaxScaler is fitted to the full dataset before the chronological split. Consequently, information about the evaluation-period price range influences preprocessing.

**Evaluation design:** The held-out evaluation data is also used to monitor validation loss during training. An independent test set would provide a more rigorous estimate of generalization performance.

**Forecast alignment:** The final prediction routine uses the last test sequence. It does not explicitly construct the input required to forecast the trading day after the final available observation.

**Limited reproducibility:** The project does not establish fixed random seeds or document repeated training runs. The reported metrics therefore describe individual model executions rather than the distribution of performance across runs.

These limitations constrain the conclusions that can be drawn from the reported results.

## Future Improvements

The following improvements could be explored in subsequent iterations:

* Incorporate trading volume and additional market indicators as predictive features.
* Compare SimpleRNN performance against LSTM and GRU architectures.
* Implement a training-only normalization procedure to eliminate preprocessing leakage.
* Introduce independent validation and test periods using walk-forward evaluation.
* Compare neural network forecasts with simple benchmarks, such as predicting that the next closing price will equal the previous closing price.
* Evaluate forecasting performance across multiple historical periods and market conditions.
* Incorporate macroeconomic indicators and investigate whether they improve out-of-sample forecasting performance.

These are proposed extensions and were not implemented in the original project.

## Technologies Used

| Technology         | Purpose                                       |
| ------------------ | --------------------------------------------- |
| Python             | Data processing and model implementation      |
| Pandas             | Financial data manipulation                   |
| NumPy              | Numerical computing and sequence construction |
| TensorFlow / Keras | Recurrent neural network development          |
| Scikit-learn       | Data normalization and model evaluation       |
| Matplotlib         | Training-loss and prediction visualization    |
| Jupyter Notebook   | Interactive development and documentation     |

## Disclaimer

This project was developed for educational and research purposes. Its predictions are not financial advice and should not be used as the sole basis for investment decisions.

Stock prices are influenced by numerous factors that may not be represented in historical closing-price data.
