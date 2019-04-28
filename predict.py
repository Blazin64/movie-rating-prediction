#!/bin/python3
import csv
from collections import defaultdict
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate


# Convert movie IDs to titles and also get genres
def get_movie_titles(movie_ids):
    # Set up an empty dictionary
    movie_dict = defaultdict(list)
    # Open the movies CSV file.
    with open('./ml-latest-small/movies.csv', "r") as csvfile:
        reader = csv.reader(csvfile)
        # Read each row
        for row in reader:
            # Store the movie title if the ID matches
            if row[0] in movie_ids:
                movie_dict[row[0]] = [row[1], row[2]]
    return movie_dict


# Get the top movies (top 5 by default) for a set of users
def get_top_movies(my_users, predictions, k=5):
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


# Reassurance that the script is actually running.
print("\nNow training on the MovieLens latest small dataset.")
print("Please wait...\n")
# Define the file's format
reader = Reader(line_format='user item rating timestamp', sep=',')
# Load the data from the ratings.csv file
data = Dataset.load_from_file('./ml-latest-small/ratings.csv', reader=reader)
# Use the SVD algorithm for prediction
method = SVD()
# Use 2-fold cross validation and evaluate the results with RMSE and MAE
cross_validate(method, data, measures=['RMSE', 'MAE'], cv=8, verbose=True)
# Train with the dataset
trainset = data.build_full_trainset()
method.fit(trainset)
print("\nNow predicting ratings for movies not rated yet.")
print("Please wait...\n")
# Test on an anti testset
predictions = method.test(trainset.build_anti_testset())
# Get a user selection
print("What user IDs would you like to predict ratings for?")
print("If entering multiple values, press enter before entering another.")
print("Press enter twice when done.")
print("Valid user IDs are 1-610.")
users = []
counter = 0
while True:
    choice = input("Choice: ")
    if counter >= 1 and choice == "":
        break
    else:
        users.append(choice)
        counter += 1
# Get the predicted top movies for the selected users
top_movies = get_top_movies(users, predictions)
# Collect the movie IDs for the users' top movies
movie_ids = []
for user_id, ratings in top_movies.items():
    for movie_id, _ in ratings:
        movie_ids.append(movie_id)
# Grab the titles and genres for the movie IDs
movie_dict = get_movie_titles(movie_ids)
# Print out the results
counter = 1
print("\nPredicted top 5 movies for your selected users:\n")
for user_id, ratings in top_movies.items():
    if counter == 6:
        counter = 1
    print("User ID: " + user_id)
    for movie_id, prediction in ratings:
        current = movie_dict[movie_id]
        print("\t#" + str(counter) + ". " + current[0])
        print("\t\t" + current[1])
        print("\t\tPredicted rating: {:0.3f}".format(prediction) + "\n")
        counter += 1
