#!/bin/python3
from collections import defaultdict
from surprise import Dataset, evaluate, Reader, SVD


# Predict the top movies (top 5 by default) for a set of users
def predict_top_movies(my_users, predictions, k=5):
    # Set up an empy dictionary
    top_movies = defaultdict(list)
    # Read through the predicted ratings and store them in the dictionary
    for user, item, truth, prediction, _ in predictions:
        # Only store predictions for the requested users
        if user in my_users:
            top_movies[user].append((item, prediction))
    # Organize the ratings so that the top rated movies are first
    for user, ratings in top_movies.items():
        ratings.sort(key=lambda x: x[1], reverse=True)
        top_movies[user] = ratings[:k]
    return top_movies


# Define the format
reader = Reader(line_format='user item rating timestamp', sep=',')
# Load the data from the file using the reader format
data = Dataset.load_from_file('./ml-latest-small/ratings.csv', reader=reader)
# Split MovieLens data into 3 folds
data.split(n_folds=2)
# Use the SVD algorithm for prediction
method = SVD()
# Evaluate the results with RMSE and MAE
evaluate(method, data, measures=['RMSE', 'MAE'])
# Train with the dataset
trainset = data.build_full_trainset()
method.fit(trainset)
# Test on an anti testset
predictions = method.test(trainset.build_anti_testset())
# User range
users = ["196", "197"]
# Predict the top movies for the selected users
top_movies = predict_top_movies(users, predictions)
for user_id, ratings in top_movies.items():
    print(user_id, [iid for (iid, _) in ratings])
