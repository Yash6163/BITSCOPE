import requests
from bs4 import BeautifulSoup
import json
import os
import re

url = "https://www.coindesk.com/"
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")
articles = []

for a in soup.find_all("a", href=True):
    title = a.get_text(strip=True)
    link = a["href"]

    # Extract date from URL (e.g., /2026/04/17/)
    date_match = re.search(r'/(\d{4}/\d{2}/\d{2})/', link)
    if title and len(title) > 30 and date_match:
        articles.append({
            "date": date_match.group(1).replace("/", "-"),
            "title": title,
            "link": "https://www.coindesk.com" + link if link.startswith("/") else link
        })

os.makedirs("../data/raw", exist_ok=True)
with open("../data/raw/news.json", "w") as f:
    json.dump(articles, f, indent=4)

print(f"✅ Saved {len(articles)} articles with dates!")