import urllib.request
import ssl
import os

def fetch_openphish(url="https://openphish.com/feed.txt"):
    print(f"Fetching phishing URLs from {url}...")
    context = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    phish_urls = []
    try:
        with urllib.request.urlopen(req, context=context) as response:
            data = response.read().decode('utf-8')
            for line in data.split('\n'):
                line = line.strip()
                if line:
                    phish_urls.append(line)
    except Exception as e:
        print(f"Failed to fetch OpenPhish: {e}")
    print(f"Fetched {len(phish_urls)} phishing URLs.")
    return phish_urls

def main():
    fetched_urls = list(set(fetch_openphish())) # deduplicate latest fetch
    if len(fetched_urls) == 0:
        print("No phishing URLs found in the current feed, aborting.")
        return
        
    # Read existing URLs to prevent duplication
    existing_urls = set()
    if os.path.exists("phishing_dataset.txt"):
        with open("phishing_dataset.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_urls.add(line)
                    
    # Filter for distinctly new URLs only
    new_urls = [u for u in fetched_urls if u not in existing_urls]
        
    if len(new_urls) == 0:
        print(f"Checked feed against your {len(existing_urls)} known URLs. No new URLs found to append.")
    else:
        print(f"Discovered {len(new_urls)} entirely new Phishing URLs! Appending to phishing_dataset.txt...")
        with open("phishing_dataset.txt", "a", encoding="utf-8") as f:
            for u in new_urls:
                f.write(u + "\n")
        print(f"Successfully augmented! You now have a total of {len(existing_urls) + len(new_urls)} malicious targets.")
        
    # Touch benign_dataset.txt if it doesn't already exist so the scraper doesn't crash
    if not os.path.exists("benign_dataset.txt"):
        with open("benign_dataset.txt", "w", encoding="utf-8") as f:
            pass
            
    print("NOTE: benign_dataset.txt is left completely untouched for your manual curation.")

if __name__ == "__main__":
    main()
