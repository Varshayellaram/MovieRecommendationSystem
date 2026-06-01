import streamlit as st
import pickle
import requests
import os
from concurrent.futures import ThreadPoolExecutor

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch – Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0a14 0%, #0f0f1a 50%, #12121f 100%);
    color: #e0e0e0;
}
[data-testid="stHeader"]  { background: transparent; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a14 0%, #0f0f1a 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #ccc; }

.main-title {
    text-align: center; font-size: 3.2rem; font-weight: 800;
    background: linear-gradient(90deg, #e50914 0%, #ff6b6b 50%, #ffd700 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -1px; margin-bottom: 0.2rem;
}
.sub-title {
    text-align: center; color: #666; font-size: 0.95rem;
    margin-bottom: 1.5rem; letter-spacing: 2px; text-transform: uppercase;
}

/* Movie card */
.movie-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 8px; text-align: center;
    position: relative; overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    height: 100%;
}
.movie-card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 12px 32px rgba(229,9,20,0.35);
    border-color: #e50914;
}
.movie-card img { width: 100%; border-radius: 8px; display: block; }
.movie-card-placeholder {
    width: 100%; aspect-ratio: 2/3;
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 8px; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 6px;
}
.card-title {
    color: #fff; font-weight: 600; font-size: 0.8rem;
    margin-top: 7px; line-height: 1.3;
    display: -webkit-box; -webkit-line-clamp: 2;
    -webkit-box-orient: vertical; overflow: hidden;
}
.match-score { color: #e50914; font-size: 0.72rem; font-weight: 700; margin-top: 3px; }
.rank-badge {
    position: absolute; top: 10px; left: 10px;
    background: rgba(229,9,20,0.9); color: #fff;
    font-size: 0.68rem; font-weight: 700;
    border-radius: 5px; padding: 2px 6px;
}

/* Section header */
.section-header {
    font-size: 1.1rem; font-weight: 700; color: #fff;
    margin: 1.5rem 0 1rem 0; display: flex; align-items: center; gap: 8px;
}
.section-header::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(229,9,20,0.5), transparent);
    margin-left: 8px;
}

/* Trending card */
.trending-card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 6px; text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.trending-card:hover { transform: translateY(-4px); border-color: rgba(255,215,0,0.4); }

/* Stat card */
.stat-card {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 12px; text-align: center; margin-bottom: 10px;
}
.stat-number { font-size: 1.6rem; font-weight: 800; color: #e50914; }
.stat-label  { font-size: 0.75rem; color: #888; margin-top: 2px; }

/* Button */
div.stButton > button {
    background: linear-gradient(90deg, #e50914, #b00610);
    color: white; border: none; border-radius: 8px;
    padding: 0.6rem 2rem; font-size: 0.95rem; font-weight: 700;
    width: 100%; transition: opacity 0.2s, transform 0.1s; letter-spacing: 0.5px;
}
div.stButton > button:hover { opacity: 0.88; transform: scale(1.01); }

hr { border-color: rgba(255,255,255,0.07); }
.stSelectbox label { color: #bbb !important; font-size: 0.9rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)

@st.cache_resource
def load_data():
    mv  = pickle.load(open(os.path.join(BASE_DIR, 'artificats', 'movie_list.pkl'), 'rb'))
    sim = pickle.load(open(os.path.join(BASE_DIR, 'artificats', 'similarity.pkl'), 'rb'))
    return mv, sim

movies, similarity = load_data()
movie_list = movies['title'].values

# ── TMDB helpers ───────────────────────────────────────────────────────────────
TMDB_API_KEY = "c2824263b18c1ceea51d5ca16b463bc5"
TMDB_BASE    = "https://api.themoviedb.org/3"

@st.cache_data(show_spinner=False)
def fetch_movie_details(movie_id):
    """Single API call using append_to_response for speed."""
    try:
        r = requests.get(
            f"{TMDB_BASE}/movie/{movie_id}",
            params={
                "api_key": TMDB_API_KEY,
                "language": "en-US",
                "append_to_response": "credits"
            },
            timeout=8
        ).json()
        poster   = f"https://image.tmdb.org/t/p/w342{r['poster_path']}" if r.get('poster_path') else None
        rating   = round(r.get('vote_average', 0), 1)
        genres   = [g['name'] for g in r.get('genres', [])[:3]]
        overview = r.get('overview', '')
        year     = r.get('release_date', '')[:4]
        credits  = r.get('credits', {})
        cast     = [p['name'] for p in credits.get('cast', [])[:4]]
        return {
            'poster': poster, 'rating': rating,
            'genres': genres, 'overview': overview,
            'cast': cast, 'year': year,
        }
    except Exception:
        return {'poster': None, 'rating': 0, 'genres': [], 'overview': '', 'cast': [], 'year': ''}

def fetch_poster_only(movie_id):
    d = fetch_movie_details(movie_id)
    return d['poster']

# ── Recommend ──────────────────────────────────────────────────────────────────
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1]
    )
    results = []
    for i in distances[1:16]:
        row = movies.iloc[i[0]]
        results.append({
            'title':    row['title'],
            'movie_id': int(row['movie_id']),
            'score':    round(i[1] * 100, 1),
        })
    return results

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.markdown("---")
    st.markdown("### 📊 Dataset Stats")
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{len(movies):,}</div>
        <div class="stat-label">Movies in Database</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">15</div>
        <div class="stat-label">Recommendations per Search</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">5K</div>
        <div class="stat-label">Feature Vectors</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🧠 How It Works")
    st.markdown("""
    <div style="color:#999; font-size:0.82rem; line-height:1.7;">
    1. <b style="color:#ccc;">Feature Extraction</b><br>
       Combines genres, keywords, cast, director & overview.<br><br>
    2. <b style="color:#ccc;">Text Vectorization</b><br>
       CountVectorizer → 5000-dim vectors.<br><br>
    3. <b style="color:#ccc;">Cosine Similarity</b><br>
       Closer angle = more similar movie.<br><br>
    4. <b style="color:#ccc;">Ranking</b><br>
       Top 15 results ranked by similarity score.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        '<div style="color:#444; font-size:0.75rem; text-align:center;">'
        'Powered by TMDB · Built with Streamlit</div>',
        unsafe_allow_html=True
    )

# ── Main UI ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎬 CineMatch</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Content-Based Movie Recommender</div>', unsafe_allow_html=True)
st.markdown("---")

col_l, col_m, col_r = st.columns([1, 2, 1])
with col_m:
    selected_movie = st.selectbox(
        "🔍 Search a movie to get recommendations",
        movie_list,
        index=None,
        placeholder="Type a movie name...",
    )
    recommend_btn = st.button("🎯 Find Similar Movies")

st.markdown("---")

# ── Featured section ───────────────────────────────────────────────────────────
if not recommend_btn or selected_movie is None:
    st.markdown('<div class="section-header">🔥 Featured Movies</div>', unsafe_allow_html=True)
    featured_titles = [
        "The Dark Knight", "Inception", "Interstellar", "Avatar",
        "The Avengers", "Titanic", "Jurassic World", "Spectre",
        "The Hobbit: The Battle of the Five Armies", "Skyfall"
    ]
    featured = [t for t in featured_titles if t in movie_list]
    with st.spinner("Loading featured movies..."):
        featured_ids = [int(movies[movies['title'] == t]['movie_id'].values[0]) for t in featured]
        with ThreadPoolExecutor(max_workers=10) as ex:
            featured_posters = list(ex.map(fetch_poster_only, featured_ids))

    cols = st.columns(len(featured))
    for i, (title, poster) in enumerate(zip(featured, featured_posters)):
        with cols[i]:
            if poster:
                st.markdown(f"""
                <div class="trending-card">
                    <img src="{poster}" style="width:100%; border-radius:7px;"/>
                    <div style="color:#ddd; font-size:0.72rem; font-weight:600;
                        margin-top:5px; line-height:1.3;">{title}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="trending-card">
                    <div style="aspect-ratio:2/3; background:#1a1a2e; border-radius:7px;
                        display:flex; align-items:center; justify-content:center;
                        font-size:2rem;">🎬</div>
                    <div style="color:#ddd; font-size:0.72rem; font-weight:600;
                        margin-top:5px;">{title}</div>
                </div>""", unsafe_allow_html=True)

    if selected_movie is None and recommend_btn:
        st.warning("⚠️ Please select a movie first.")

# ── Recommendations ────────────────────────────────────────────────────────────
if recommend_btn and selected_movie:

    with st.spinner("🎬 Finding best matches..."):
        recommendations = recommend(selected_movie)
        sel_movie_id    = int(movies[movies['title'] == selected_movie]['movie_id'].values[0])

        # Fetch selected movie details + all 15 posters in parallel
        with ThreadPoolExecutor(max_workers=16) as ex:
            future_sel     = ex.submit(fetch_movie_details, sel_movie_id)
            future_posters = [ex.submit(fetch_poster_only, r['movie_id']) for r in recommendations]

        sel_details = future_sel.result()
        posters     = [f.result() for f in future_posters]

    # ── Selected movie hero ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎥 You Selected</div>', unsafe_allow_html=True)

    hero_col1, hero_col2 = st.columns([1, 4])
    with hero_col1:
        if sel_details['poster']:
            st.image(sel_details['poster'], width=180)
        else:
            st.markdown("""
            <div style="width:180px; height:270px; background:linear-gradient(135deg,#1a1a2e,#16213e);
                border-radius:10px; display:flex; align-items:center;
                justify-content:center; font-size:4rem;">🎬</div>
            """, unsafe_allow_html=True)

    with hero_col2:
        year_str = f" ({sel_details['year']})" if sel_details['year'] else ""
        st.markdown(f"### {selected_movie}{year_str}")

        # Rating + genre badges
        badges = ""
        if sel_details['rating']:
            badges += (f'<span style="background:rgba(245,197,24,0.15);border:1px solid '
                       f'rgba(245,197,24,0.35);color:#f5c518;border-radius:20px;'
                       f'padding:3px 11px;font-size:0.78rem;margin:2px;">⭐ {sel_details["rating"]} / 10</span>')
        for g in sel_details['genres']:
            badges += (f'<span style="background:rgba(229,9,20,0.12);border:1px solid '
                       f'rgba(229,9,20,0.3);color:#ff6b6b;border-radius:20px;'
                       f'padding:3px 11px;font-size:0.78rem;margin:2px;">{g}</span>')
        if badges:
            st.markdown(badges, unsafe_allow_html=True)
            st.markdown("")

        if sel_details['overview']:
            st.markdown(
                f'<p style="color:#aaa; font-size:0.88rem; line-height:1.6;">'
                f'{sel_details["overview"]}</p>', unsafe_allow_html=True
            )
        if sel_details['cast']:
            st.markdown(
                f'<p style="color:#777; font-size:0.8rem;">🎭 '
                f'{" · ".join(sel_details["cast"])}</p>', unsafe_allow_html=True
            )

    st.markdown("---")

    # ── Recommendations grid ──────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-header">✨ Because you liked '
        f'<span style="color:#e50914;">{selected_movie}</span>, you might enjoy</div>',
        unsafe_allow_html=True
    )

    for row in range(3):
        cols = st.columns(5)
        for col in range(5):
            idx        = row * 5 + col
            rec        = recommendations[idx]
            poster_url = posters[idx]
            rank       = idx + 1
            with cols[col]:
                if poster_url:
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="rank-badge">#{rank}</div>
                        <img src="{poster_url}" alt="{rec['title']}"/>
                        <div class="card-title">{rec['title']}</div>
                        <div class="match-score">▲ {rec['score']}% match</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="rank-badge">#{rank}</div>
                        <div class="movie-card-placeholder">
                            <span style="font-size:2.5rem;">🎬</span>
                            <span style="color:#444; font-size:0.65rem;">No Poster</span>
                        </div>
                        <div class="card-title">{rec['title']}</div>
                        <div class="match-score">▲ {rec['score']}% match</div>
                    </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#333; font-size:0.78rem; padding-bottom:1rem;">'
    '🎬 CineMatch &nbsp;·&nbsp; Content-Based Filtering &nbsp;·&nbsp; '
    'TMDB Dataset &nbsp;·&nbsp; Built with Streamlit'
    '</div>',
    unsafe_allow_html=True
)
