# We'll generate a synthetic dataset for 12 weeks pre (no AI Overviews) and 12 weeks post (30% AI-exposed),
# guided by public analyses about which query categories tend to trigger AI Overviews.
# We'll create up to ~120k total rows (60k pre + 60k post) to keep it fast locally.


import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import math

rng = np.random.default_rng(42)

# ---- Configuration ----
total_pre = 60000
total_post = 60000

weeks_pre = 12
weeks_post = 12

end_date = datetime(2025, 10, 12)  # recent Sunday before current date
start_post = end_date - timedelta(weeks=weeks_post-1)
start_pre = start_post - timedelta(weeks=weeks_pre)

# Categories and their AI Overview propensity (drawn from industry reports)
# Higher for Health, Science, Society (informational) and boosted in Travel/Restaurants post March 2025 updates
categories = {
    "Health": {"ai_propensity": 0.55, "commercial_intent": 0.1},
    "Science": {"ai_propensity": 0.5, "commercial_intent": 0.05},
    "Society": {"ai_propensity": 0.45, "commercial_intent": 0.05},
    "HowTo": {"ai_propensity": 0.4, "commercial_intent": 0.05},
    "Travel": {"ai_propensity": 0.35, "commercial_intent": 0.2},
    "Restaurants": {"ai_propensity": 0.35, "commercial_intent": 0.25},
    "Shopping": {"ai_propensity": 0.25, "commercial_intent": 0.8},
    "Finance": {"ai_propensity": 0.2, "commercial_intent": 0.9},
    "Entertainment": {"ai_propensity": 0.3, "commercial_intent": 0.15},
    "Tech": {"ai_propensity": 0.25, "commercial_intent": 0.2},
}

# Category prior distribution (roughly realistic; informational dominates)
cat_probs = {
    "Health": 0.11,
    "Science": 0.08,
    "Society": 0.08,
    "HowTo": 0.14,
    "Travel": 0.12,
    "Restaurants": 0.08,
    "Shopping": 0.16,
    "Finance": 0.08,
    "Entertainment": 0.1,
    "Tech": 0.05,
}
cat_list = list(cat_probs.keys())
cat_weights = np.array([cat_probs[c] for c in cat_list])

# Query templates per category (mix short & long-tail)
templates = {
    "Health": [
        "is {cond} contagious",
        "how to treat {cond} at home",
        "{cond} symptoms in adults",
        "can I {action} while on {medication}",
        "best foods for {cond} recovery",
    ],
    "Science": [
        "explain {concept} like i am five",
        "what is the difference between {concept} and {concept2}",
        "latest research on {concept} 2025",
        "how does {concept} work",
        "impact of {concept} on climate change",
    ],
    "Society": [
        "what is {policy} and how it affects {group}",
        "pros and cons of {policy}",
        "statistics on {topic} 2025",
        "should {group} support {policy}",
        "history of {topic} in india",
    ],
    "HowTo": [
        "how to fix {thing} that won't {verb}",
        "step by step guide to {task}",
        "best way to {task} without tools",
        "quick tips to {task} at home",
        "how do i {task} safely",
    ],
    "Travel": [
        "3 day itinerary for {place}",
        "best time to visit {place} with kids",
        "budget hotels near {place}",
        "public transport pass for {place}",
        "is {place} safe at night",
    ],
    "Restaurants": [
        "best {cuisine} restaurants near me",
        "top rated cafes in {place}",
        "late night food delivery {place}",
        "kid friendly restaurants in {place}",
        "vegetarian {cuisine} spots around {place}",
    ],
    "Shopping": [
        "best {product} under {price}",
        "{brand} vs {brand2} {product} comparison",
        "is {product} worth it 2025",
        "top {product} for {usecase}",
        "where to buy {product} online",
    ],
    "Finance": [
        "best credit card for {usecase} in india",
        "how to file {tax} online",
        "fixed deposit rates {place} 2025",
        "is {investment} safe now",
        "{insurance} premium calculator",
    ],
    "Entertainment": [
        "cast of {movie}",
        "review of {movie} no spoilers",
        "best {genre} shows on {ott}",
        "soundtrack list for {movie}",
        "upcoming releases {genre} 2025",
    ],
    "Tech": [
        "how to set up {device}",
        "compare {phone} vs {phone2}",
        "troubleshoot {device} {issue}",
        "is {software} free for students",
        "{framework} tutorial step by step",
    ],
}

# Filler vocab
conds = ["dengue", "flu", "covid", "migraine", "dehydration", "food poisoning"]
actions = ["exercise", "fast", "drink coffee", "fly", "drive"]
medications = ["ibuprofen", "paracetamol", "metformin", "amoxicillin"]
concepts = ["quantum entanglement", "photosynthesis", "CRISPR", "dark matter", "machine learning"]
concepts2 = ["classical mechanics", "cellular respiration", "gene editing", "dark energy", "deep learning"]
policies = ["data privacy bill", "universal basic income", "net neutrality", "carbon tax"]
groups = ["students", "small businesses", "seniors", "freelancers"]
topics = ["inflation", "unemployment", "digital payments", "renewable energy"]
things = ["washing machine", "android phone", "laptop", "door lock"]
verbs = ["start", "charge", "connect", "open"]
tasks = ["clean windows", "reset wifi router", "descale coffee machine", "organize photos"]
places = ["Bangalore", "Goa", "Prague", "Tokyo", "New York", "Paris", "Delhi", "Pune"]
cuisines = ["Italian", "South Indian", "Thai", "Mexican", "Japanese"]
products = ["wireless earbuds", "mechanical keyboard", "running shoes", "air purifier", "4K monitor"]
prices = ["2000", "5000", "10000", "20000"]
brands = ["Sony", "Bose", "Samsung", "LG", "OnePlus", "Xiaomi", "JBL"]
product_use = ["gaming", "running", "photo editing", "office work"]
taxes = ["ITR-1", "GST", "TDS"]
investments = ["gold ETF", "Nifty 50 index fund", "crypto", "PPF"]
insurances = ["term insurance", "health insurance", "car insurance"]
movies = ["Dune 2", "Jawan", "Oppenheimer", "Kalki 2898 AD", "Top Gun Maverick"]
genres = ["thriller", "comedy", "sci-fi", "drama"]
otts = ["Netflix", "Prime Video", "Hotstar"]
devices = ["router", "MacBook", "Windows PC", "iPhone", "Android"]
issues = ["won't boot", "keeps restarting", "wifi not connecting", "battery draining fast"]
software = ["MATLAB", "Figma", "Notion", "VS Code"]
frameworks = ["PyTorch", "TensorFlow", "React", "Django"]
phones = ["iPhone 16", "Pixel 9", "Samsung S25", "OnePlus 13"]

def sample_query(cat):
    t = random.choice(templates[cat])
    return t.format(
        cond=random.choice(conds),
        action=random.choice(actions),
        medication=random.choice(medications),
        concept=random.choice(concepts),
        concept2=random.choice(concepts2),
        policy=random.choice(policies),
        group=random.choice(groups),
        topic=random.choice(topics),
        thing=random.choice(things),
        verb=random.choice(verbs),
        task=random.choice(tasks),
        place=random.choice(places),
        cuisine=random.choice(cuisines),
        product=random.choice(products),
        price=random.choice(prices),
        brand=random.choice(brands),
        brand2=random.choice(brands),
        usecase=random.choice(product_use),
        tax=random.choice(taxes),
        investment=random.choice(investments),
        insurance=random.choice(insurances),
        movie=random.choice(movies),
        genre=random.choice(genres),
        ott=random.choice(otts),
        device=random.choice(devices),
        issue=random.choice(issues),
        software=random.choice(software),
        framework=random.choice(frameworks),
        phone=random.choice(phones),
        phone2=random.choice(phones),
    )

# Helper to generate weekly dates
def weekly_dates(start, weeks, total_rows):
    # distribute rows across days within each week, skew to weekdays
    rows_per_week = total_rows // weeks
    extra = total_rows % weeks
    dates = []
    current = start
    for w in range(weeks):
        week_rows = rows_per_week + (1 if w < extra else 0)
        # For this week, distribute across 7 days with weekday weights
        day_weights = np.array([0.12,0.14,0.16,0.16,0.16,0.15,0.11])  # Mon..Sun
        day_weights = day_weights / day_weights.sum()
        day_counts = rng.multinomial(week_rows, day_weights)
        # start is the Sunday for simplicity? We'll align Monday start
        week_start = current + timedelta(days=w*7)
        # assume Monday-start week for mapping weights; compute Monday of that week
        monday = week_start - timedelta(days=week_start.weekday())
        for i, cnt in enumerate(day_counts):
            for _ in range(cnt):
                dates.append(monday + timedelta(days=i))
    rng.shuffle(dates)
    return dates

def generate_split(total_rows, period_label, weeks, start_date, ai_target_share_post=0.3):
    categories_arr = rng.choice(cat_list, size=total_rows, p=cat_weights)
    dates = weekly_dates(start_date, weeks, total_rows)
    long_tail = rng.random(total_rows) < 0.55  # 55% long-tail
    # Base AI propensity by category, boosted for long-tail queries
    base_ai = np.array([categories[c]["ai_propensity"] for c in categories_arr])
    long_tail_boost = np.where(long_tail, 0.15, 0.0)
    ai_prob = np.clip(base_ai + long_tail_boost, 0, 0.9)
    # In pre period, AI Overviews are essentially 0 in this simulation
    if period_label == "pre":
        is_ai = np.zeros(total_rows, dtype=bool)
    else:
        # scale ai_prob so that marginal share is ~ ai_target_share_post
        # compute scalar s such that mean(sigmoidish) hits target; use simple ratio with clipping
        # We'll sample with probability s * ai_prob then clip to [0,1]
        s = min(1.0, ai_target_share_post / float(ai_prob.mean()))
        is_ai = rng.random(total_rows) < (ai_prob * s)
    
    # Commercial intent influences ads viewed/clicked & CPC
    comm_intent = np.array([categories[c]["commercial_intent"] for c in categories_arr])
    
    # Ads viewed ~ Poisson with lambda depending on commercial intent and period
    # Assume slight decline post (esp. if AI) due to less scrolling
    base_lambda = 0.8 + 2.8 * comm_intent  # between ~0.9 and ~3.4
    if period_label == "post":
        base_lambda = base_lambda * (0.95 - 0.25 * is_ai.astype(float))  # up to 25% lower if AI shown
    
    ads_viewed = rng.poisson(lam=np.clip(base_lambda, 0.05, None))
    
    # CTR depends on category and declines with AI Overviews
    base_ctr = 0.01 + 0.05 * comm_intent  # 1% to 6%
    if period_label == "post":
        base_ctr = base_ctr * (0.98 - 0.35 * is_ai.astype(float))  # up to 35% lower if AI shown
    
    # Sample ad clicks as Binomial(ads_viewed, ctr) but cap by ads_viewed
    ads_clicked = np.array([rng.binomial(ads_viewed[i], min(0.95, max(0, base_ctr[i]))) for i in range(total_rows)])
    
    # CPC depends on category; finance highest
    cpc = 0.15 + 2.5 * comm_intent  # ~$0.15 to ~$2.4
    # Slight CPC erosion post due to lower competition on AI SERPs
    if period_label == "post":
        cpc = cpc * (0.99 - 0.08 * is_ai.astype(float))
    
    revenue = ads_clicked * cpc
    
    # Time spent on search page (seconds): lognormal base, reduced with AI Overviews
    # Base mean by intent (more commercial tends to browse more ads/cards)
    mu = 2.9 + 0.5 * comm_intent  # controls lognormal mean
    sigma = 0.55
    time_spent = rng.lognormal(mean=mu, sigma=sigma, size=total_rows)  # often tens of seconds
    if period_label == "post":
        time_spent = time_spent * (0.96 - 0.2 * is_ai.astype(float))  # up to ~20% lower if AI
    
    # Actions taken
    actions = []
    for i in range(total_rows):
        if period_label == "post" and is_ai[i]:
            choices = ["ai_expand", "no_click", "visit_ads", "visit_publisher", "refine_query"]
            probs = [0.32, 0.28, 0.16, 0.14, 0.10]
        else:
            choices = ["visit_publisher", "visit_ads", "refine_query", "no_click"]
            probs = [0.42, 0.22, 0.2, 0.16] if period_label=="pre" else [0.38,0.20,0.24,0.18]
        actions.append(rng.choice(choices, p=probs))
    
    # Build queries
    queries = [sample_query(c) for c in categories_arr]
    
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "week_period": period_label,
        "category": categories_arr,
        "query": queries,
        "long_tail": long_tail,
        "ai_overview": is_ai,
        "actions_taken": actions,
        "ads_viewed": ads_viewed,
        "ads_clicked": ads_clicked,
        "cpc_usd": np.round(cpc, 2),
        "google_revenue_usd": np.round(revenue, 2),
        "time_spent_search_sec": np.round(time_spent, 1),
    })
    return df

pre_df = generate_split(total_pre, "pre", weeks_pre, start_pre)
post_df = generate_split(total_post, "post", weeks_post, start_post)

df = pd.concat([pre_df, post_df], ignore_index=True)
df.sort_values("date", inplace=True)

# Sanity checks
summary = {
    "rows_total": len(df),
    "rows_pre": (df["week_period"]=="pre").sum(),
    "rows_post": (df["week_period"]=="post").sum(),
    "post_ai_share": float(df.loc[df["week_period"]=="post","ai_overview"].mean()),
    "avg_time_pre": float(df.loc[df["week_period"]=="pre","time_spent_search_sec"].mean()),
    "avg_time_post_overall": float(df.loc[df["week_period"]=="post","time_spent_search_sec"].mean()),
    "avg_time_post_ai": float(df.loc[(df["week_period"]=="post") & (df["ai_overview"]==True),"time_spent_search_sec"].mean()),
    "avg_time_post_non_ai": float(df.loc[(df["week_period"]=="post") & (df["ai_overview"]==False),"time_spent_search_sec"].mean()),
    "revenue_pre": float(df.loc[df["week_period"]=="pre","google_revenue_usd"].sum()),
    "revenue_post": float(df.loc[df["week_period"]=="post","google_revenue_usd"].sum()),
}

# check first few rows
df.head()


from pathlib import Path

DATA_DIR = Path("data")
FILE_NAME = "synthetic_search_ai_overviews_24w.csv"
OUTPUT_PATH = DATA_DIR / FILE_NAME


# Save to CSV
print(f"Checking if directory '{DATA_DIR}' exists...")
try:
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Directory '{DATA_DIR}' ensured/created successfully.")
    
    # 2. Save the DataFrame to the CSV file inside the created folder.
    #    index=False prevents pandas from writing the DataFrame index as a column.
    df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"DataFrame successfully saved to: {OUTPUT_PATH.resolve()}")

except Exception as e:
    print(f"An error occurred while saving the file: {e}")

df.head()
