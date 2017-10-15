'''
Bayesian Classification
Author: Christian M. Adams
--------------------------
1. Handle Data: Load the data from CSV file and split it into training and test datasets.
2. Summarize Data: summarize the properties in the training dataset so that we can calculate probabilities and make predictions.
3. Make a Prediction: Use the summaries of the dataset to generate a single prediction.
4. Make Predictions: Generate predictions given a test dataset and a summarized training dataset.
5. Evaluate Accuracy: Evaluate the accuracy of predictions made for a test dataset as the percentage correct out of all predictions made.
6. Tie it Together: Use all of the code elements to present a complete and standalone implementation of the Naive Bayes algorithm.


'''

import csv
import random
import math
