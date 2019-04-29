#!/bin/python3
from collections import defaultdict
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
from tkinter import Tk, Button, Label, Entry, scrolledtext, DISABLED, END, NORMAL
import csv
import numpy
import os
import psutil
import random
import time


class MovieRatings:
    def __init__(self, master):
        # Set up the GUI
        self.master = master
        self.text = scrolledtext.ScrolledText(master, width=100, height=30)
        self.text.grid(row=6, column=0,)
        self.msg1 = Label(master, text="1. Train with the dataset")
        self.msg1.grid(row=0, column=0)
        self.button = Button(master, text='Train Now', width=25, command=self.runner)
        self.button.grid(row=1, column=0)
        self.msg3 = Label(master, text="2. Enter user IDs you would like to display predictions for (1-610).")
        self.msg3.grid(row=2, column=0)
        self.msg4 = Label(master, text="Separate values with commas. No spaces.")
        self.msg4.grid(row=3, column=0)
        self.userentry = Entry(master, width=40)
        self.userentry.grid(row=4, column=0)
        self.button2 = Button(master, text='Display Estimates', width=25, command=self.runner2)
        self.button2.grid(row=5, column=0)
        self.button2.configure(state=DISABLED)
        master.mainloop()

    # Used to print to the UI text box
    def printer(self, string):
        self.text.insert(END, string + "\n")
        self.master.update()

    # Enable a button
    def enableButton(self, button):
        button.configure(state=NORMAL)
        self.master.update()

    # Dsable a button
    def disableButton(self, button):
        button.configure(state=DISABLED)
        self.master.update()

    # Change the text on a button.
    def updateButton(self, string, button):
        button["text"] = string
        self.master.update()

    # Train on the dataset and predict ratings
    def runner(self):
        self.updateButton('Training now...', self.button)
        self.disableButton(self.button)
        results, trainset = self.trainer()
        self.updateButton('Training done!', self.button)
        self.predictions = self.predictor(results, trainset)
        self.enableButton(self.button2)

    # Display top 10 predictions for selected users
    def runner2(self):
        users = self.userentry.get().split(",")
        self.finder(self.predictions, users)

    # Get top 10 predictions
    def finder(self, predictions, users):
        top_movies = self.get_top_movies(users, predictions, 10)
        # Collect the movie IDs for the users' top movies
        movie_ids = []
        for user_id, ratings in top_movies.items():
            for movie_id, _ in ratings:
                movie_ids.append(movie_id)
        # Grab the titles and genres for the movie IDs
        movie_dict = self.get_movie_titles(movie_ids)
        # Print out the results
        counter = 1
        self.printer("\nPredicted top 10 movies for your selected users:\n")
        for user_id, ratings in top_movies.items():
            if counter == 11:
                counter = 1
            self.printer("User ID: " + user_id)
            for movie_id, prediction in ratings:
                current = movie_dict[movie_id]
                self.printer("\t#" + str(counter) + ". " + current[0])
                self.printer("\t\t" + current[1])
                self.printer("\t\tPredicted rating: {:0.3f}".format(prediction) + "\n")
                counter += 1

    # Predict ratings.
    def predictor(self, results, trainset):
        self.printer("\nNow predicting ratings for movies each user hasn't rated.")
        self.printer("Please wait...\n")
        # Test on an anti testset (effectively predicts how users will rate
        # movies they haven't watched yet).
        start = time.time()
        predictions = results.test(trainset.build_anti_testset())
        end = time.time()
        spent = end - start
        self.printer("Prediction time: {:0.3f} seconds".format(spent))
        process = psutil.Process(os.getpid())
        self.printer("Memory used:")
        self.printer("{:0.5f}".format(process.memory_info().rss/1048576.0) + " MB Physical")
        self.printer("{:0.5f}".format(process.memory_info().vms/1048576.0) + " MB Virtual")
        return predictions

    # Train with the dataset
    def trainer(self):
        # Set the random seed that numpy (used internally by Surprise) will use.
        my_seed = random.randint(0, 2**32)
        random.seed(my_seed)
        numpy.random.seed(my_seed)
        # Reassurance that the script is actually running.
        self.printer("\nNow training on the MovieLens latest small dataset. (8 folds used)")
        self.printer("Please wait...\n")
        # Define the file's format
        reader = Reader(line_format='user item rating timestamp', sep=',')
        # Load the data from the ratings.csv file
        data = Dataset.load_from_file('./ml-latest-small/ratings.csv', reader=reader)
        # Use the SVD algorithm for prediction
        method = SVD()
        start = time.time()
        # Use 8-fold cross validation and evaluate the results with RMSE and MAE
        measurements = cross_validate(method, data, measures=['RMSE', 'MAE'],
                                      cv=8, verbose=False, n_jobs=-2, return_train_measures=True)
        # Print the random seed used for fold assignments
        self.printer("Random seed used for fold assignment: {}\n".format(my_seed))
        # Show the stats
        meanFitTime = numpy.mean(measurements["fit_time"])
        meanTestTime = numpy.mean(measurements["test_time"])
        meanTestMAE = numpy.mean(measurements["test_mae"])
        meanTestRMSE = numpy.mean(measurements["test_rmse"])
        meanTrainMAE = numpy.mean(measurements["train_mae"])
        meanTrainRMSE = numpy.mean(measurements["train_rmse"])
        self.printer("Mean fit time per fold: {:0.5f} seconds".format(meanFitTime))
        self.printer("Mean test time per fold: {:0.5f} seconds".format(meanTestTime))
        self.printer("Mean train MAE per fold: {:0.5f}".format(meanTrainMAE))
        self.printer("Mean train RMSE per fold: {:0.5f}".format(meanTrainRMSE))
        self.printer("Mean test MAE per fold: {:0.5f}".format(meanTestMAE))
        self.printer("Mean test RMSE per fold: {:0.5f}\n".format(meanTestRMSE))
        # Train with the dataset
        trainset = data.build_full_trainset()
        method.fit(trainset)
        end = time.time()
        spent = end - start
        self.printer("Training and testing time: {:0.3f} seconds\n".format(spent))
        process = psutil.Process(os.getpid())
        self.printer("Memory used:")
        self.printer("{:0.5f}".format(process.memory_info().rss/1048576.0) + " MB Physical")
        self.printer("{:0.5f}".format(process.memory_info().vms/1048576.0) + " MB Virtual")
        return method, trainset

    # Get the top movies (top 5 by default) for a set of users
    def get_top_movies(self, users, predictions, k):
        # Set up an empy dictionary
        top_movies = defaultdict(list)
        # Read through the predicted ratings and store them in the dictionary
        for user, item, truth, prediction, _ in predictions:
            # Only store predictions for the requested users
            if user in users:
                top_movies[user].append((item, prediction))
        # Organize the ratings so that the top rated movies are first
        for user, ratings in top_movies.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            top_movies[user] = ratings[:k]
        return top_movies

    # Convert movie IDs to titles and also get genres
    def get_movie_titles(self, movie_ids):
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


master = Tk()
MovieRatings(master)
master.mainloop()
