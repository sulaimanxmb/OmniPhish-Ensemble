import asyncio
import os
import random
from tqdm import tqdm
from playwright.async_api import async_playwright

from phish_scraper import check_login_heuristics, sanitize_filename

OUTPUT_DIR_PHISH = "dataset/raw_html/phishing"
OUTPUT_DIR_BENIGN = "dataset/raw_html/benign"

def get_manual_candidates(filepath="benign_dataset.txt"):
    if not os.path.exists(filepath):
        print(f"\n[!] Error: {filepath} not found on your system! Please create it and paste in your safe URLs.")
        return []
        
    with open(filepath, "r", encoding="utf-8") as f:
        # Strip whitespaces and ignore empty lines
        urls = [line.strip() for line in f if line.strip()]
        
    # We heavily randomize the file order so your dataset doesn't train predictably
    random.shuffle(urls)
    return urls

async def balance_scrape(sem, browser, url, target_list, delta):
    async with sem:
        try:
            # Drop context aggressively
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto(url, wait_until="networkidle", timeout=12000)
            html_content = await page.content()
            
            if check_login_heuristics(html_content):
                # Ensure we strictly don't exceed the needed Delta constraint
                if len(target_list) >= delta:
                    await page.close()
                    await context.close()
                    return

                filename = sanitize_filename(url)
                filepath = os.path.join(OUTPUT_DIR_BENIGN, filename)
                
                # Check offline folder to prevent identical overwrites
                if not os.path.exists(filepath):
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    target_list.append(url) # Track our accurate success count
                    
            await page.close()
            await context.close()
        except:
            pass # Keep terminal visuals smooth

async def main():
    phish_count = len(os.listdir(OUTPUT_DIR_PHISH)) if os.path.exists(OUTPUT_DIR_PHISH) else 0
    benign_count = len(os.listdir(OUTPUT_DIR_BENIGN)) if os.path.exists(OUTPUT_DIR_BENIGN) else 0
    
    if benign_count >= phish_count:
        print(f"Dataset is already balanced or Benign is heavier! (Phishing: {phish_count}, Benign: {benign_count})")
        return
        
    delta = phish_count - benign_count
    print(f"Dataset Imbalance Detected: You have {phish_count} Bad targets but only {benign_count} Safe ones.")
    print(f"Goal: Harvest exactly {delta} more Safe Benign profiles.")
    
    raw_candidates = get_manual_candidates("benign_dataset.txt")
    
    # Filter out candidates whose HTML has already been saved to the Benign folder
    candidate_urls = []
    for u in raw_candidates:
        if not os.path.exists(os.path.join(OUTPUT_DIR_BENIGN, sanitize_filename(u))):
            candidate_urls.append(u)
    
    if len(candidate_urls) == 0:
        print("\n[!] Aborting: You have no fresh URLs left in benign_dataset.txt to pull from! Please paste more safe links in there.")
        return
        
    if len(candidate_urls) < delta:
        print(f"\n[?] Warning: You need {delta} more successful HTMLs to achieve balance, but you only provided {len(candidate_urls)} unscraped URLs inside your text file.")
        print(f"[?] The script will harvest all {len(candidate_urls)} of them now, but you'll have to add more later to fully close the gap!")
        delta = len(candidate_urls) # Floor the delta so the progress bar works
    
    successes = []
    
    async with async_playwright() as p:
        print("\nDeploying Local-List Balancing Orchestrator...")
        
        with tqdm(total=delta, desc="Harvesting Benign HTMLs", unit="file") as pbar:
            # Process in 10-URL chunks to iteratively fill the bucket safely and recycle RAM
            for i in range(0, len(candidate_urls), 10):
                if len(successes) >= delta:
                    break
                    
                # Launch pristine Firefox binary per chunk to prevent macOS memory swapping
                browser = await p.firefox.launch(headless=True)
                sem = asyncio.Semaphore(3) # 3-tab limits to save MAC Ram
                
                chunk = candidate_urls[i:i+10]
                tasks = [balance_scrape(sem, browser, url, successes, delta) for url in chunk]
                await asyncio.gather(*tasks)
                
                current_total = len(successes)
                if current_total > pbar.n:
                    diff = current_total - pbar.n
                    pbar.update(min(diff, delta - pbar.n))
                    
                # Gracefully assassinate the browser locally to flush Mac Memory Swaps
                try:
                    await browser.close()
                except Exception:
                    pass
        
    print(f"\nFinished! Added {len(successes)} clean safe HTML pages directly into dataset/raw_html/benign/.")

if __name__ == "__main__":
    asyncio.run(main())
