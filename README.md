# Rough Bergomi Neural Network Calibration

## File Functions

OptionsData.py => Fetches and cleans MSFT call option data from yfinance, stores in csv file (yfinance_data.csv)

RoughBergomi.py => Implements a numerical approximation of the price and implied volatility of a European call option under the rough Bergomi asset dynamics.

SyntheticRBergomi.py => Generates ~750,000 points of data mapping the 6 rough Bergomi parameters, strike and maturiy to an implied volatility using the functions implemented in the RoughBergomi.py file. Saves data to rbergomi_dataset_final.csv

NeuralNetwork.py => Trains a neural network on the rbergomi_dataset_final.csv file to approximate the numerical approximation at a significantly higher speed. The network takes in 8 parameters and outputs an implied volatility, and then saves the trained network to rbergomi_model.pth.

Calibration.py => Uses the trained network to calibrate to the MSFT option implied volatility surface, obtaining 6 parameters for the rough Bergomi model that reflect the dynamics of the underlying asset. This can then in turn be used to price exotic options.

