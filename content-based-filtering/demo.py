# Shamelessly copied from
# https://medium.com/@james_aka_yale/the-4-recommendation-engines-that-can-predict-your-movie-tastes-bbec857b8223

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function that get movie recommendations based on the cosine similarity score of movie genres
def genre_recommendations(title, indices, cosine_sim, titles):

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]

def main():

    # Reading ratings file
    # Ignore the timestamp column
    ratings = pd.read_csv('data/ratings.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id', 'rating'])

    # Reading users file
    users = pd.read_csv('data/users.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])

    # Reading movies file
    movies = pd.read_csv('data/movies.csv', sep='\t', encoding='latin-1', usecols=['movie_id', 'title', 'genres'])
    # tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0)
    tfidf_matrix = tf.fit_transform(movies['genres'])

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Build a 1-dimensional array with movie titles
    titles = movies['title']
    indices = pd.Series(movies.index, index=movies['title'])
    title = 'Good Will Hunting (1997)'
    print('Input: ' + title)
    movie_recommendations = genre_recommendations(title, indices, cosine_sim, titles).head(20)
    print(movie_recommendations)

if __name__ == '__main__':
    main()
