#!/bin/python3
from collections import defaultdict
from surprise import Dataset, evaluate, Reader, SVDpp, SVD


# Predict top movies for a user
def predict_top_movies(predictions, k=5):
    top_movies = defaultdict(list)
    for user, item, truth, prediction, _ in predictions:
        top_movies[user].append((item, prediction))
    return None


# Define the format
reader = Reader(line_format='user item rating timestamp', sep=',')
# Load the data from the file using the reader format
data = Dataset.load_from_file('./ml-latest-small/ratings.csv', reader=reader)
# Split MovieLens data into 3 folds
data.split(n_folds=3)
# Use the SVD algorithm for prediction
method = SVD()
# Evaluate the results with RMSE and MAE
evaluate(method, data, measures=['RMSE', 'MAE'])
# Train with the dataset
method.fit(data.build_full_trainset())
# Predict the rating this user will rate this movie
user = 196
item = 303
print(method.predict(str(user), str(item)))
