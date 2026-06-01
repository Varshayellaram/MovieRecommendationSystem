# 🎬 CineMatch — AI-Powered Movie Recommendation System

A content-based movie recommendation system that suggests similar movies based on genres, keywords, cast, director, and plot overview. Built with Python, scikit-learn, and Streamlit.

---

## 📌 Table of Contents

1. [What is this project?](#what-is-this-project)
2. [How it works — Simple explanation](#how-it-works--simple-explanation)
3. [Project Structure](#project-structure)
4. [Complete Workflow](#complete-workflow)
5. [File-by-File Explanation](#file-by-file-explanation)
6. [Technologies Used](#technologies-used)
7. [APIs Used](#apis-used)
8. [Dataset](#dataset)
9. [How to Run](#how-to-run)
10. [Features of the UI](#features-of-the-ui)
11. [Interview Talking Points](#interview-talking-points)

---

## What is this project?

CineMatch is a **Movie Recommendation System** — you give it a movie name, and it tells you 15 other movies you might enjoy.

For example:
- Input: `The Dark Knight`
- Output: `Batman Begins`, `The Dark Knight Rises`, `Inception`, `Man of Steel`, ...

It uses **Content-Based Filtering** — meaning it recommends movies that are *similar in content* (same genre, similar plot, same actors/director) rather than what other users watched (that would be Collaborative Filtering).

---

## How it works — Simple explanation

Imagine every movie as a bag of words describing it — its genre, plot keywords, cast names, director name. Now imagine converting that bag of words into a point in space (a vector). Movies that are similar will have their points close together in that space.

That's exactly what this system does:

```
Movie → Tags (genres + keywords + cast + director + overview)
     → Vector (5000 numbers)
     → Compare with all other movie vectors
     → Find the closest ones
     → Return top 15
```

The "closeness" is measured using **Cosine Similarity** — it measures the angle between two vectors. Smaller angle = more similar movies.

---

## Project Structure

```
📁 Movie Recommendation System/
└── 📁 data/
    ├── 📓 Recommender_system.ipynb   ← ML model (data processing + training)
    ├── 🐍 app.py                     ← Streamlit web app (UI)
    ├── 📄 tmdb_5000_movies.csv       ← Movies dataset
    ├── 📄 tmdb_5000_credits.csv      ← Cast & crew dataset
    └── 📁 artificats/
        ├── 🥒 movie_list.pkl         ← Saved processed movie data
        └── 🥒 similarity.pkl         ← Saved similarity matrix
```

---

## Complete Workflow

The project has two parts that work together:

```
┌─────────────────────────────────────────────────────────────┐
│              PART 1: Recommender_system.ipynb               │
│                    (Run once to train)                      │
│                                                             │
│  CSV Files → Data Cleaning → Feature Engineering           │
│           → Vectorization → Cosine Similarity              │
│           → Save .pkl files                                 │
└─────────────────────────┬───────────────────────────────────┘
                          │ saves
                          ▼
                   artificats/
                   ├── movie_list.pkl
                   └── similarity.pkl
                          │ loads
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    PART 2: app.py                           │
│                  (Streamlit Web App)                        │
│                                                             │
│  Load .pkl → User selects movie → recommend() function     │
│           → Fetch posters from TMDB API                    │
│           → Display 15 recommendation cards                │
└─────────────────────────────────────────────────────────────┘
```

**Key point:** The notebook does all the heavy computation (takes ~1-2 minutes). The result is saved as `.pkl` files. The app just loads those files and runs instantly — no recomputation needed.

---

## File-by-File Explanation

### 📓 Recommender_system.ipynb

This is the **brain** of the project. It runs once to build the recommendation model.

#### Step 1 — Import Libraries
```python
import pandas as pd       # for data manipulation
import numpy as np        # for numerical operations
import ast                # to parse JSON strings in CSV columns
import nltk               # for text processing (stemming)
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

#### Step 2 — Load Datasets
```python
movies  = pd.read_csv('tmdb_5000_movies.csv')   # 4803 movies, 20 columns
credits = pd.read_csv('tmdb_5000_credits.csv')  # 4803 movies, 4 columns
```

Two separate CSV files:
- `tmdb_5000_movies.csv` — contains movie details like genres, keywords, overview, budget, revenue
- `tmdb_5000_credits.csv` — contains cast and crew information

#### Step 3 — Merge Datasets
```python
movies = movies.merge(credits, on='title')
```
Both files have a `title` column. We join them so each row has both movie details AND cast/crew info. Result: 4809 rows × 23 columns.

#### Step 4 — Select Useful Columns
```python
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
```
We only need these 7 columns. The rest (budget, revenue, runtime, etc.) are dropped.

#### Step 5 — Handle Missing Values
```python
movies.dropna(inplace=True)
```
3 movies had missing overviews — they are removed. Final count: 4806 movies.

#### Step 6 — Parse JSON Columns
The `genres`, `keywords`, `cast`, and `crew` columns contain JSON strings like:
```
"[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]"
```
We need to extract just the names.

**`convert()` — for genres and keywords:**
```python
def convert(text):
    l = []
    for i in ast.literal_eval(text):   # parse the JSON string
        l.append(i['name'])            # extract just the name
    return l
# Result: ["Action", "Adventure", "Fantasy"]
```

**`convert_cast()` — for cast (top 3 actors only):**
```python
def convert_cast(text):
    l = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:                # only take top 3 actors
            l.append(i['name'])
        else:
            break
        counter += 1
    return l
# Result: ["Sam Worthington", "Zoe Saldana", "Sigourney Weaver"]
```

**`convert_crew()` — for crew (director only):**
```python
def convert_crew(text):
    l = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':     # only extract the director
            l.append(i['name'])
    return l
# Result: ["James Cameron"]
```

#### Step 7 — Remove Spaces from Names
```python
def remove_space(word):
    l = []
    for i in word:
        l.append(i.replace(" ", ""))   # "Sam Worthington" → "SamWorthington"
    return l
```
**Why?** If we keep spaces, "Sam Worthington" becomes two separate words "Sam" and "Worthington". The vectorizer would treat them as unrelated words. By joining them, "SamWorthington" is treated as one unique token representing this specific actor.

#### Step 8 — Split Overview into Words
```python
movies['overview'] = movies['overview'].apply(lambda x: x.split())
# "A great movie" → ["A", "great", "movie"]
```

#### Step 9 — Create Tags Column
```python
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
```
All features are combined into one list of words per movie. This is the "bag of words" for each movie.

```python
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))  # list → string
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())    # lowercase
```

Example tags for Avatar:
```
"in the 22nd century a paraplegic marine action adventure fantasy sciencefiction 
cultureclash future spacewar samworthington zoesaldana jamescameron"
```

#### Step 10 — Stemming
```python
ps = PorterStemmer()

def stems(text):
    l = []
    for i in text.split():
        l.append(ps.stem(i))    # "dancing" → "danc", "loved" → "love"
    return " ".join(l)

new_df['tags'] = new_df['tags'].apply(stems)
```
**Why stemming?** Words like "dance", "dancing", "danced" all mean the same thing. Stemming reduces them to their root form "danc" so the model treats them as the same word. This improves similarity matching.

#### Step 11 — Vectorization
```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new_df['tags']).toarray()
```
- `max_features=5000` — keep only the 5000 most common words
- `stop_words='english'` — ignore common words like "the", "a", "is" (they don't help with similarity)
- Each movie becomes a vector of 5000 numbers (word counts)

Result: a matrix of shape `(4806, 5000)` — 4806 movies, each described by 5000 numbers.

#### Step 12 — Cosine Similarity
```python
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vector)
```
- Computes similarity between every pair of movies
- Result: a `(4806, 4806)` matrix
- `similarity[0][1]` = how similar movie 0 is to movie 1 (value between 0 and 1)
- 1 = identical, 0 = completely different

**Why Cosine Similarity and not Euclidean distance?**
Cosine similarity measures the *angle* between vectors, not the distance. This is better for text because a longer movie description shouldn't be considered more "different" just because it has more words.

#### Step 13 — Recommend Function
```python
def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]   # find movie index
    distances = sorted(
        list(enumerate(similarity[index])),              # get similarity scores
        reverse=True,
        key=lambda x: x[1]
    )
    for i in distances[1:6]:                             # skip index 0 (itself)
        print(new_df.iloc[i[0]].title)
```

#### Step 14 — Save Model (Pickle)
```python
import pickle
pickle.dump(new_df, open('artificats/movie_list.pkl', 'wb'))
pickle.dump(similarity, open('artificats/similarity.pkl', 'wb'))
```
Saves the processed dataframe and similarity matrix to disk so the app can load them instantly without recomputing.

---

### 🐍 app.py

This is the **face** of the project — the web application users interact with.

#### Step 1 — Load Saved Model
```python
@st.cache_resource
def load_data():
    movies     = pickle.load(open('artificats/movie_list.pkl', 'rb'))
    similarity = pickle.load(open('artificats/similarity.pkl', 'rb'))
    return movies, similarity
```
`@st.cache_resource` means Streamlit loads the data once and keeps it in memory. Every time a user interacts with the app, it doesn't reload from disk — it uses the cached version. This makes the app fast.

#### Step 2 — Fetch Movie Posters from TMDB API
```python
@st.cache_data(show_spinner=False)
def fetch_movie_details(movie_id):
    r = requests.get(
        f"https://api.themoviedb.org/3/movie/{movie_id}",
        params={
            "api_key": TMDB_API_KEY,
            "append_to_response": "credits"   # get details + credits in 1 call
        }
    ).json()
    poster = f"https://image.tmdb.org/t/p/w342{r['poster_path']}"
    ...
```
`@st.cache_data` caches the API response — if the same movie is requested again, it returns the cached result instead of making another API call.

#### Step 3 — Recommend Function (same logic, returns 15)
```python
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    results = []
    for i in distances[1:16]:    # top 15 (skip index 0 = the movie itself)
        row = movies.iloc[i[0]]
        results.append({'title': row['title'], 'movie_id': row['movie_id'], 'score': round(i[1]*100, 1)})
    return results
```

#### Step 4 — Parallel Poster Fetching
```python
with ThreadPoolExecutor(max_workers=16) as ex:
    future_sel     = ex.submit(fetch_movie_details, sel_movie_id)
    future_posters = [ex.submit(fetch_poster_only, r['movie_id']) for r in recommendations]
```
Instead of fetching 16 posters one by one (slow), all 16 are fetched simultaneously using threads. This is 16x faster.

#### Step 5 — Display UI
- **Sidebar** — dataset stats, algorithm explanation
- **Hero section** — selected movie poster, rating, genres, cast, overview
- **Recommendations grid** — 3 rows × 5 columns = 15 movie cards with rank badge and % match score

---

## Technologies Used

| Technology | Purpose | Why we used it |
|---|---|---|
| **Python 3.11** | Core programming language | Industry standard for ML/data science |
| **Pandas** | Data loading, merging, cleaning | Best library for tabular data manipulation |
| **NumPy** | Numerical operations | Fast array operations used internally by sklearn |
| **scikit-learn** | CountVectorizer + Cosine Similarity | Industry-standard ML library, easy to use |
| **NLTK** | PorterStemmer for text normalization | Reduces words to root form for better matching |
| **Streamlit** | Web application framework | Build data apps in pure Python, no HTML/JS needed |
| **Requests** | HTTP calls to TMDB API | Standard Python library for API calls |
| **Pickle** | Save/load Python objects to disk | Saves trained model so app loads instantly |
| **concurrent.futures** | Parallel API calls | Fetch 16 posters simultaneously instead of one by one |

---

## APIs Used

### TMDB API (The Movie Database)

**What is it?**
TMDB is a community-built movie database (like IMDb but with a free API). It provides movie details, posters, cast information, ratings, and more.

**Why we use it:**
Our dataset only has movie IDs and titles. TMDB API gives us the actual poster images and detailed information to display in the UI.

**How to get a free API key:**
1. Go to [themoviedb.org](https://www.themoviedb.org/signup) and create a free account
2. Go to Settings → API → Create → Developer
3. Fill in the form (Application URL: `http://localhost`, Type: Personal)
4. Your API key appears immediately — no subscription needed

**Endpoints we use:**

```
GET https://api.themoviedb.org/3/movie/{movie_id}
    ?api_key=YOUR_KEY
    &append_to_response=credits
```
Returns: poster path, rating, genres, overview, runtime, budget, revenue, cast, director — all in one call.

**Poster URL format:**
```
https://image.tmdb.org/t/p/w342{poster_path}
```
- `w342` = image width 342px (we use this for speed; `w500` is higher quality)

**`append_to_response` optimization:**
Instead of making 3 separate API calls (movie details + credits + external IDs), we use `append_to_response=credits` to get everything in 1 call. This reduces API calls by 3x and makes the app significantly faster.

---

## Dataset

**Source:** [TMDB 5000 Movie Dataset on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

### tmdb_5000_movies.csv
- **4803 rows** (movies), **20 columns**
- Key columns used: `id` (renamed to `movie_id`), `title`, `overview`, `genres`, `keywords`
- Other columns (budget, revenue, runtime, etc.) are dropped as they don't help with content similarity

### tmdb_5000_credits.csv
- **4803 rows**, **4 columns**: `movie_id`, `title`, `cast`, `crew`
- `cast` — JSON list of all actors with their character names
- `crew` — JSON list of all crew members with their job titles

### After merging and cleaning:
- **4806 movies** (3 dropped due to missing overview)
- **7 columns** used: `movie_id`, `title`, `overview`, `genres`, `keywords`, `cast`, `crew`

---

## How to Run

### Prerequisites
Make sure you have Python 3.x installed. Then install required packages:

```bash
pip install streamlit pandas numpy scikit-learn nltk requests
```

### Step 1 — Run the Notebook (one time only)
Open `Recommender_system.ipynb` in Jupyter and run all cells top to bottom.
This will:
- Process the datasets
- Build the similarity matrix
- Save `artificats/movie_list.pkl` and `artificats/similarity.pkl`

> ⚠️ The `.pkl` files are already present in the `artificats/` folder, so you can skip this step if they exist.

### Step 2 — Run the Streamlit App
```bash
python -m streamlit run app.py
```
The app opens automatically at `http://localhost:8501` in your browser.

> 💡 For movie posters to load, connect to a network that can reach `api.themoviedb.org` (use mobile hotspot if your main network blocks it).

### Step 3 — Use the App
1. Type or select a movie name in the search dropdown
2. Click **"🎯 Find Similar Movies"**
3. View the selected movie's details in the hero section
4. Browse 15 recommended movies in the grid below

---

## Features of the UI

| Feature | Description |
|---|---|
| 🔍 **Smart Search** | Searchable dropdown of all 4806 movies — no typos possible |
| 🔥 **Featured Movies** | Homepage shows 10 popular movies before any search |
| 🎥 **Hero Section** | Selected movie shown with poster, rating, genres, cast, overview |
| 🃏 **15 Recommendation Cards** | 3 rows × 5 columns grid with movie posters |
| 🏅 **Rank Badges** | Each card shows #1 to #15 ranking |
| 📊 **Match Score** | Each card shows similarity percentage (e.g. "▲ 34.2% match") |
| 📚 **Sidebar** | Dataset stats + step-by-step algorithm explanation |
| ⚡ **Fast Loading** | Parallel API calls + Streamlit caching for speed |
| 🌙 **Dark Theme** | Cinema-inspired dark UI with red accent colors |

---

## Interview Talking Points

**Q: What type of recommendation system is this?**
> Content-Based Filtering. It recommends movies similar in content (genres, plot, cast) to what the user selected. It does NOT use other users' behavior (that would be Collaborative Filtering).

**Q: Why CountVectorizer and not TF-IDF?**
> TF-IDF penalizes words that appear frequently across documents. But in our case, if "action" appears in many movies, that's actually useful information — it means those movies are similar. CountVectorizer just counts word frequency without penalizing, which works better here.

**Q: Why Cosine Similarity and not Euclidean distance?**
> Cosine similarity measures the angle between vectors, not the magnitude. A movie with a longer description would have larger vector values, making it seem "far" from shorter descriptions in Euclidean space even if the content is similar. Cosine similarity normalizes for this.

**Q: Why do you stem the words?**
> Words like "love", "loved", "loving" all convey the same meaning. PorterStemmer reduces them all to "love" so the model treats them as the same feature. Without stemming, "love" and "loving" would be counted as different words, reducing similarity accuracy.

**Q: Why save as pickle files?**
> Computing the cosine similarity matrix for 4806 movies takes ~30-60 seconds. If the app recomputed it every time a user made a request, it would be unusably slow. By saving it once and loading it, the app responds in milliseconds.

**Q: How does `append_to_response` help?**
> TMDB API allows combining multiple endpoints into one HTTP request. Instead of 3 calls (movie details + credits + external IDs), we make 1 call. For 16 movies, this reduces API calls from 48 to 16 — 3x fewer network requests.

**Q: What is `@st.cache_resource` vs `@st.cache_data`?**
> `@st.cache_resource` is for heavy objects shared across all users (like the loaded pickle files — loaded once, reused forever). `@st.cache_data` is for function results that can be serialized (like API responses — cached per unique input).

---

## Limitations & Future Improvements

| Limitation | Possible Improvement |
|---|---|
| Only content-based (no user behavior) | Add Collaborative Filtering using user ratings |
| Dataset is from 2017 (no recent movies) | Connect to live TMDB API to fetch latest movies |
| Recommendations based on text similarity only | Add weighted scoring (popularity, ratings) |
| No user accounts | Add login + watch history for personalized recommendations |
| English movies only | Add multilingual support |

---

*Built with ❤️ using Python, scikit-learn, and Streamlit · Powered by TMDB API*
