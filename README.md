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
