import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

def check_url(url):
    try:
        # A lightweight HEAD request natively saves massive networking overhead
        req = urllib.request.Request(url, method="HEAD", headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'})
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status < 400:
                return url, True
            else:
                return url, False
    except urllib.error.HTTPError as e:
        # Some security servers specifically block HEAD requests, if so we fall back to a raw GET request
        if e.code == 405 or e.code == 403:
            try:
                get_req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'})
                with urllib.request.urlopen(get_req, timeout=5) as get_resp:
                    if get_resp.status < 400:
                        return url, True
            except Exception:
                pass
        return url, False
    except Exception:
        # Handles DNS failures, severe SSL certificate issues, or Hard Timeouts
        return url, False

def curate_benign_list(filepath):
    print(f"Reading target network strings from: {filepath}")
    if not os.path.exists(filepath):
        print(f"Error: Could not find physically {filepath}")
        return
        
    with open(filepath, 'r') as f:
        urls = [line.strip() for line in f if line.strip() and line.startswith('http')]
        
    print(f"Loaded {len(urls)} strict URLs. Initiating extreme-speed Parallel Domain Sweep...")
    
    alive_urls = []
    dead_urls = []
    
    # Utilizing 20 parallel workers allows us to sweep hundreds of websites in seconds
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(check_url, url): url for url in urls}
        for i, future in enumerate(as_completed(futures), 1):
            url, is_alive = future.result()
            if is_alive:
                alive_urls.append(url)
            else:
                dead_urls.append(url)
            
            # Print dynamically every 10 completions so the terminal isn't heavily spammed
            if i % 10 == 0 or i == len(urls):
                print(f"Sweep Progress: [{i}/{len(urls)}] | Detected {len(dead_urls)} Dead/Dropped Domains...")
                
    print("\n===============================")
    print(f"TOTAL ALIVE / ONLINE: {len(alive_urls)}")
    print(f"TOTAL DEAD / 404 / DROPPED: {len(dead_urls)}")
    print("===============================\n")
    
    if len(dead_urls) > 0:
        print("--- Detailed List of Dead Domains ---")
        for bad_url in dead_urls:
            print(f"   [!] DEAD: {bad_url}")
            
        print("\n[Audit Only] Finished Scan. No URLs were forcefully deleted from your text file.")
    else:
        print("\n[Audit Only] Your dataset strictly contains 100% physically available live domains.")

if __name__ == "__main__":
    # Point the network engine directly at the manual dataset
    curate_benign_list("benign_dataset.txt")
