# Step 1: Import libraries
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors

# Step 2: Load dataset
df = pd.read_csv("movies.csv")  # Ensure this CSV has columns: title, genre, rating

# Step 3: Preprocess genres
df["genre"] = df["genre"].apply(lambda x: x.split("|"))  # Split if pipe-separated
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(df["genre"])

# Step 4: Convert rating (if it's categorical like 'R', 'PG') into numeric
rating_map = {"G": 1, "PG": 2, "PG-13": 3, "R": 4, "NC-17": 5}
# Try converting, fallback to float if mapping not needed
try:
    df["rating"] = df["rating"].map(rating_map)
except:
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")  # NaNs for bad data

# Step 5: Combine features (genre + rating)
features = pd.DataFrame(genre_encoded, columns=mlb.classes_)
features["rating"] = df["rating"]

# Drop rows with missing (NaN) values after conversion
features = features.dropna()
df = df.loc[features.index]  # keep only aligned rows

# Step 6: Fit KNN model
knn = NearestNeighbors(n_neighbors=6, algorithm="auto", metric="cosine")
knn.fit(features)


# Step 7: Recommend function
def recommend_movie(movie_name):
    pass  # Placeholder for recommendation logic
