"""Download the benchmark corpus: Wikipedia articles on shipping and logistics.

This is the fixed document set the evaluation questions in eval/ were written
against, and the "sample documents" the web app offers. It is already
committed in data/benchmark/, so you only need this to refresh it.

Wikipedia is used because the text is real, freely licensed (CC BY-SA 4.0) and
stable enough that the evaluation set stays valid. The app itself works on any
documents you give it.

    python scripts/fetch_corpus.py
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "benchmark"
API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = (
    "hybrid-rag-service/1.0 "
    "(https://github.com/0yman/hybrid-rag-service) python-httpx"
)

TOPICS = [
    "Containerization",
    "Intermodal container",
    "Twenty-foot equivalent unit",
    "Container ship",
    "Port",
    "Port of Alexandria",
    "Suez Canal",
    "Bill of lading",
    "Incoterms",
    "Demurrage",
    "Bulk carrier",
    "Gantry crane",
    "International Maritime Organization",
    "SOLAS Convention",
    "Port state control",
    "Draft (hull)",
    "Freight transport",
    "Supply chain management",
    "Logistics",
    "Warehouse",
    "Just-in-time manufacturing",
    "Cargo",
]

# Reference lists and navigation boilerplate add tokens without adding facts,
# and they pollute BM25 statistics. Cut the article at the first of them.
_TAIL_SECTIONS = re.compile(
    r"\n==+\s*(See also|References|Further reading|External links|Notes|"
    r"Bibliography|Sources|Citations)\s*==+",
    re.IGNORECASE,
)


def clean(text: str) -> str:
    match = _TAIL_SECTIONS.search(text)
    if match:
        text = text[: match.start()]
    # Turn MediaWiki "=== Heading ===" into markdown so titles survive chunking.
    text = re.sub(r"^=====\s*(.+?)\s*=====$", r"#### \1", text, flags=re.MULTILINE)
    text = re.sub(r"^====\s*(.+?)\s*====$", r"### \1", text, flags=re.MULTILINE)
    text = re.sub(r"^===\s*(.+?)\s*===$", r"### \1", text, flags=re.MULTILINE)
    text = re.sub(r"^==\s*(.+?)\s*==$", r"## \1", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return f"{slug}.md"


def fetch(client: httpx.Client, title: str) -> str | None:
    response = client.get(
        API,
        params={
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "redirects": "1",
            "format": "json",
            "titles": title,
        },
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id == "-1" or "extract" not in page:
            return None
        return page["extract"]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--delay", type=float, default=0.3, help="Politeness delay between requests")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    # Wikimedia rejects requests without a User-Agent that identifies the
    # client and offers a way to contact its owner.
    headers = {"User-Agent": USER_AGENT}

    written = 0
    with httpx.Client(timeout=30.0, headers=headers, follow_redirects=True) as client:
        for title in TOPICS:
            try:
                extract = fetch(client, title)
            except httpx.HTTPError as exc:
                print(f"  !! {title}: {exc}", file=sys.stderr)
                continue
            if not extract:
                print(f"  -- {title}: no extract returned", file=sys.stderr)
                continue

            body = clean(extract)
            path = args.out / slugify(title)
            path.write_text(
                f"# {title}\n\n{body}\n\n"
                f"_Source: https://en.wikipedia.org/wiki/{title.replace(' ', '_')} "
                f"(CC BY-SA 4.0)_\n",
                encoding="utf-8",
            )
            print(f"  ok {path.name:<42} {len(body):>7,} chars")
            written += 1
            time.sleep(args.delay)

    print(f"\nWrote {written}/{len(TOPICS)} documents to {args.out}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
