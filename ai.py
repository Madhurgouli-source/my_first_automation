"""
AI-Powered News Aggregator (AI/Finance Focus)

What it does
------------
- Pulls news from RSS feeds (no API keys needed)
- Runs sentiment analysis on titles & summaries (NLTK VADER)
- Classifies each item: Positive / Neutral / Negative
- Filters by topic (default: "ai" or "stocks") using simple keyword matching
- Saves results to CSV and prints a concise console digest

Usage
-----
python ai_news_system.py --topic ai --limit 100 --since_days 3 --out ai_news.csv
python ai_news_system.py --topic stocks --limit 200 --since_days 2 --out stock_news.csv
"""

import argparse
import datetime as dt
import re
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import feedparser
import pandas as pd

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Ensure the VADER lexicon is available
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

# ---- Configuration ----
FEEDS = [
    # Tech / AI
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "https://www.theverge.com/rss/index.xml",
    "https://www.technologyreview.com/topnews.rss",
    # Finance / Markets
    "https://www.investing.com/rss/news_301.rss",
    "https://www.moneycontrol.com/rss/latestnews.xml",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.ft.com/?format=rss",
]

AI_KEYWORDS = [
    "artificial intelligence", "ai", "machine learning", "ml",
    "deep learning", "genai", "generative ai", "llm", "openai",
    "chatgpt", "google ai", "anthropic", "llama", "transformer",
]

STOCKS_KEYWORDS = [
    "stock", "stocks", "equity", "market", "nifty", "sensex",
    "nasdaq", "dow", "earnings", "ipo", "listing", "dividend",
    "revenue", "guidance", "share", "shares", "buyback", "fpo",
]

@dataclass
class NewsItem:
    published: str
    source: str
    title: str
    summary: str
    link: str
    topic: str
    compound: float
    label: str
    keywords: str

def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())

def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    text = (text or "").lower()
    tokens = re.findall(r"[a-zA-Z]{4,}", text)
    stop = set("""with from that this will would there about your into over after
                  before where when which their have been were https http com www
                  they them then than herein among between while whose those these
                  including india tech market stock share news""".split())
    freq: Dict[str, int] = {}
    for tok in tokens:
        if tok not in stop:
            freq[tok] = freq.get(tok, 0) + 1
    keys = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [k for k, _ in keys[:top_n]]

def topic_match(text: str, topic: str) -> bool:
    text_l = (text or "").lower()
    if topic == "ai":
        return any(k in text_l for k in AI_KEYWORDS)
    if topic == "stocks":
        return any(k in text_l for k in STOCKS_KEYWORDS)
    return True

def read_feed(url: str) -> List[Dict[str, Any]]:
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries:
        published = ""
        if "published_parsed" in e and e.published_parsed:
            published = dt.datetime(*e.published_parsed[:6]).isoformat()
        elif "updated_parsed" in e and e.updated_parsed:
            published = dt.datetime(*e.updated_parsed[:6]).isoformat()

        items.append({
            "title": normalize_text(getattr(e, "title", "")),
            "summary": normalize_text(getattr(e, "summary", "")),
            "link": getattr(e, "link", ""),
            "published": published,
            "source": normalize_text(getattr(feed.feed, "title", url)),
        })
    return items

def within_days(iso_str: str, days: int) -> bool:
    if not iso_str:
        return True  # keep if unknown
    try:
        t = dt.datetime.fromisoformat(iso_str)
        return (dt.datetime.now() - t) <= dt.timedelta(days=days)
    except Exception:
        return True

def label_from_compound(c: float) -> str:
    if c >= 0.05:
        return "Positive"
    if c <= -0.05:
        return "Negative"
    return "Neutral"

def main():
    parser = argparse.ArgumentParser(description="AI-Powered News Aggregator (Finance/AI Focus)")
    parser.add_argument("--topic", choices=["ai", "stocks"], default="ai",
                        help="Filter news for 'ai' or 'stocks'")
    parser.add_argument("--limit", type=int, default=200, help="Max total articles across feeds")
    parser.add_argument("--since_days", type=int, default=3, help="Keep items published within N days")
    parser.add_argument("--out", type=str, default="news_out.csv", help="Output CSV path")

    args = parser.parse_args()

    sia = SentimentIntensityAnalyzer()

    all_rows: List[NewsItem] = []
    total = 0
    for url in FEEDS:
        try:
            for e in read_feed(url):
                if total >= args.limit:
                    break
                blob = f"{e['title']} {e['summary']}"
                if not topic_match(blob, args.topic):
                    continue
                if not within_days(e["published"], args.since_days):
                    continue

                sc = sia.polarity_scores(blob)["compound"]
                keys = extract_keywords(blob, top_n=5)
                all_rows.append(NewsItem(
                    published=e["published"],
                    source=e["source"],
                    title=e["title"],
                    summary=e["summary"],
                    link=e["link"],
                    topic=args.topic,
                    compound=sc,
                    label=label_from_compound(sc),
                    keywords=", ".join(keys),
                ))
                total += 1
        except Exception as ex:
            print(f"[WARN] Failed {url}: {ex}", file=sys.stderr)

    if not all_rows:
        print("No news items collected. Try increasing --limit or --since_days.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame([asdict(x) for x in all_rows])
    df.sort_values(["published"], ascending=[False], inplace=True)
    df.to_csv(args.out, index=False, encoding="utf-8")

    # Console digest (top 10)
    head = df.head(10)
    print("\n=== Top Headlines ===")
    for _, r in head.iterrows():
        print(f"[{r['label']:^8}] {r['title']}  ({r['source']})")
        if r['link']:
            print(f" -> {r['link']}")
    print(f"\nSaved {len(df)} items to {args.out}")
    print("Done.")
    
if __name__ == "__main__":
    print("✅ Script started...")
    try:
        main()
        print("✅ Script finished successfully.")
    except Exception as e:
        import traceback
        print("❌ Error:", e)
        traceback.print_exc()
 
 
