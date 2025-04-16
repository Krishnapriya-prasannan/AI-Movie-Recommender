import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===================== Load and Preprocess =====================
# Load ratings and tags
ratings = pd.read_csv('ratings.csv').head(1000)  # limit for faster testing
tags = pd.read_csv('tags.csv').head(1000)        # limit for faster testing

# Fill missing tags
tags['tag'] = tags['tag'].fillna('').astype(str)

# Group tags by movieId
movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()

# Create dummy titles for each movie
movie_tags['title'] = 'Movie ' + movie_tags['movieId'].astype(str)

# Prepare movie list for joining later
movies = movie_tags[['movieId', 'title']]

# Merge ratings with titles
ratings_with_titles = pd.merge(ratings, movies, on='movieId', how='inner')

# ===================== Collaborative Filtering (SVD) =====================
# Prepare Surprise dataset
reader = Reader(rating_scale=(0.5, 5.0))
data_surprise = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Train/test split and SVD model
trainset, testset = train_test_split(data_surprise, test_size=0.2)
model = SVD()
model.fit(trainset)

# Predict example
prediction = model.predict(uid=1, iid=1)
print(f"\nPredicted rating for user 1 on movie 1: {prediction.est:.2f}")

# Top-N recommendation function
def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

# Generate top-5 recommendations for user 1
test_predictions = model.test(testset)
top_n = get_top_n(test_predictions, n=5)

print("\nTop 5 Movie Recommendations for User 1:")
for iid, rating in top_n[1]:
    title = movies[movies['movieId'] == iid]['title'].values
    title = title[0] if len(title) > 0 else f"Movie {iid}"
    print(f"{title} (Predicted Rating: {rating:.2f})")

# ===================== Content-Based Filtering (TF-IDF) =====================
print("\nContent-Based Recommendations for Movie 1:")

# TF-IDF on movie tags
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movie_tags['tag'])

# Recommend similar movies to Movie with ID = 1
target_movie_id = 1

if target_movie_id in movie_tags['movieId'].values:
    idx = movie_tags[movie_tags['movieId'] == target_movie_id].index[0]
    
    # Compute similarity of target movie only
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    # Get top 5 similar movie indices (excluding the movie itself)
    similar_indices = sim_scores.argsort()[-6:-1][::-1]
    
    for i in similar_indices:
        title = movie_tags.iloc[i]['title']
        score = sim_scores[i]
        print(f"{title} (Similarity Score: {score:.2f})")
else:
    print("Movie with ID 1 not found in tag data.")
