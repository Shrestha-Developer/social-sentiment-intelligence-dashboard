import argparse
import csv
import os
from pathlib import Path


def fetch_posts(query, limit):
    try:
        from x_twitter_scraper import XTwitterScraper
    except ImportError as error:
        message = "Install the optional client with: pip install x-twitter-scraper==0.4.1"
        raise SystemExit(message) from error

    client = XTwitterScraper(api_key=os.environ.get("X_TWITTER_SCRAPER_API_KEY"))
    page = client.x.tweets.search(q=query, limit=limit)
    return page.tweets


def csv_row(post, campaign):
    created_at = post.created_at or ""
    return {
        "text": post.text,
        "platform": "Twitter/X",
        "campaign": campaign,
        "date": created_at[:10],
    }


def write_csv(posts, output_path, campaign):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "platform", "campaign", "date"])
        writer.writeheader()
        for post in posts:
            writer.writerow(csv_row(post, campaign))
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Export recent X search results as a dashboard-ready CSV."
    )
    parser.add_argument("query", help="X search query, for example: product feedback")
    parser.add_argument("--limit", type=int, default=50, help="Number of posts to fetch")
    parser.add_argument("--campaign", default="Xquik Search", help="Campaign label")
    parser.add_argument(
        "--output",
        default="data/xquik_comments.csv",
        help="CSV path to upload in the dashboard",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    count = write_csv(fetch_posts(args.query, args.limit), output_path, args.campaign)
    print(f"Wrote {count} posts to {output_path}")


if __name__ == "__main__":
    main()
