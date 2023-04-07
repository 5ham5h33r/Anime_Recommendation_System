from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import streamlit as st

# Load the anime data and ratings data
# ratings_path = 'animelist.csv'
data_path = 'processed_anime.csv'
# anime_ratings = pd.read_csv(ratings_path)
anime_data = pd.read_csv(data_path)


genres_str = anime_data['Genres'].str.split(',').astype(str)

# Initialize the TfidfVectorizer with various parameters
tfv = TfidfVectorizer(min_df=3, max_features=None,
                      strip_accents='unicode', analyzer='word',
                      token_pattern=r'\w{1,}',
                      ngram_range=(1, 3),
                      stop_words='english')

# Use the TfidfVectorizer to transform the genres_str into a sparse matrix
tfv_matrix = tfv.fit_transform(genres_str)

# Compute the sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

# Create a Pandas Series object where the index is the anime names and the values are the indices in anime_data
indices = pd.Series(anime_data.index, index=anime_data['Name'])
indices = indices.drop_duplicates()

# Define the give_rec function to recommend anime based on similarity to input title


def give_rec(title, sig=sig):
    # Get the index corresponding to anime title
    idx = indices[title]

    # Get the pairwsie similarity scores
    sig_scores = list(enumerate(sig[idx]))

    # Sort the anime based on similarity scores
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of top 10 most similar anime excluding the input anime
    anime_indices = [i[0] for i in sig_scores[1:11]]

    # Create dataframe of top 10 recommended anime
    top_anime = pd.DataFrame({
        'Anime name': anime_data['Name'].iloc[anime_indices].values,
        'Rating': anime_data['Score'].iloc[anime_indices].values
    })

    return top_anime


# Set up the Streamlit app
st.title('Anime Recommender System')
options = anime_data['Name'].tolist()
options.append('Type name here..')
# Create a text input box for the user to enter an anime title
user_input = st.selectbox(
    'Enter the name of an anime you like:', options=options, index=len(options)-1)
if user_input == options[-1]:
    pass
# When the user submits an input, call the give_rec function and display the output
elif user_input:
    try:
        recommendations = give_rec(user_input)
        st.write(f"Recommended anime similar to {user_input}:")
        st.table(recommendations)
    except KeyError:
        st.write(f"Sorry, {user_input} is not in our database.")
