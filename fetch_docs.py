"""Step 1: download the documents.

Downloads the introduction section of four Wikipedia articles as plain text
and saves them in docs/, together with docs/sources.json, which records where
each one came from.

You only need to run this if you want to refresh the documents. The copies in
docs/ are committed to the repository so that results are reproducible even if
Wikipedia changes the articles later.

Usage:  python3 fetch_docs.py
"""

import http.client
import json
import time
import urllib.parse
from pathlib import Path

DOCS_DIR = Path(__file__).parent / "docs"
API_HOST = "en.wikipedia.org"
API_PATH = "/w/api.php"

# file name we save to -> Wikipedia article title
ARTICLES = {
    "mercury": "Mercury (planet)",
    "venus": "Venus",
    "mars": "Mars",
    "jupiter": "Jupiter",
}


def fetch_intro(title):
    """Ask the Wikipedia API for one article's intro section as plain text."""
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts|revisions",  # the text, plus which version of the page it is
        "exintro": 1,                  # intro section only
        "explaintext": 1,              # plain text, no HTML
        "rvprop": "ids|timestamp",
    }
    # We use http.client rather than the more common urllib.request because
    # urllib always sends a "Connection: close" header, and the Wikipedia API
    # was observed to drop most requests that carry it.
    connection = http.client.HTTPSConnection(API_HOST, timeout=30)
    try:
        connection.request(
            "GET",
            API_PATH + "?" + urllib.parse.urlencode(params),
            # Wikipedia asks scripts to identify themselves with a User-Agent header.
            headers={"User-Agent": "rag-take-home/1.0"},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"Wikipedia returned HTTP {response.status} for {title!r}")
        data = json.load(response)
    finally:
        connection.close()

    page = next(iter(data["query"]["pages"].values()))
    revision = page["revisions"][0]
    return page["extract"].strip(), revision["revid"], revision["timestamp"]


def main():
    DOCS_DIR.mkdir(exist_ok=True)
    sources = []

    for doc_id, title in ARTICLES.items():
        text, revision_id, timestamp = fetch_intro(title)
        (DOCS_DIR / f"{doc_id}.txt").write_text(text + "\n", encoding="utf-8")

        sources.append({
            "doc_id": doc_id,
            "title": title,
            "file": f"docs/{doc_id}.txt",
            "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
            # permanent link to the exact version we downloaded
            "permalink": f"https://en.wikipedia.org/w/index.php?oldid={revision_id}",
            "revision_timestamp": timestamp,
            "section": "introduction only",
            "license": "CC BY-SA 4.0",
        })
        print(f"saved docs/{doc_id}.txt  ({len(text.split())} words)")
        time.sleep(1)  # be polite: do not send requests back to back

    (DOCS_DIR / "sources.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    print("saved docs/sources.json")


if __name__ == "__main__":
    main()
