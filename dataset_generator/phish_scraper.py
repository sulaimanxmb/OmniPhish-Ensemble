import asyncio
import os
import re
import time
import urllib.request
import ssl
from tqdm import tqdm
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from urllib.parse import urlparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR_PHISH = os.path.join(ROOT_DIR, "dataset", "raw_html", "phishing")
OUTPUT_DIR_BENIGN = os.path.join(ROOT_DIR, "dataset", "raw_html", "benign")
os.makedirs(OUTPUT_DIR_PHISH, exist_ok=True)
os.makedirs(OUTPUT_DIR_BENIGN, exist_ok=True)

def fetch_openphish(url="https://openphish.com/feed.txt"):
    print(f"Fetching live phishing URLs directly from {url}...")
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
    return list(set(phish_urls))

def sanitize_filename(url):
    parsed = urlparse(url)
    filename = f"{parsed.netloc}{parsed.path}"
    filename = re.sub(r'[^a-zA-Z0-9_\-]', '_', filename)
    return filename + ".html"

def check_login_heuristics(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    password_inputs = soup.find_all('input', type=lambda t: t and t.lower() == 'password')
    if password_inputs:
        return True

    login_keywords = ['login', 'sign in', 'signin', 'log in', 'authenticate']
    for element in soup.find_all(['button', 'a', 'input']):
        text = element.get_text()
        if element.name == 'input' and element.get('value'):
            text += " " + element.get('value')
        
        if text:
            text = text.lower()
            if any(keyword in text for keyword in login_keywords):
                return True

    return False

async def bounded_scrape_url(sem, browser, url, label, pbar):
    async with sem:
        try:
            # Intentionally spawn a perfectly isolated Context per URL
            # This completely drops cache, cookies, and history from RAM after every hit
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
                viewport={'width': 1280, 'height': 800}
            )
            page = await context.new_page()
            # 15 second timeout to rapidly abort dead links
            await page.goto(url, wait_until="networkidle", timeout=15000)
            html_content = await page.content()
            is_login_page = check_login_heuristics(html_content)
            
            if is_login_page:
                filename = sanitize_filename(url)
                output_dir = OUTPUT_DIR_PHISH if label == 1 else OUTPUT_DIR_BENIGN
                    
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                    
            await page.close()
            await context.close() # Instantly flush this page's memory bloat!
        except Exception:
            # We suppress exception printouts so they don't break the progress bar visual
            pass
        finally:
            pbar.update(1)

async def main():
    # 1. Extract live targets exclusively from OpenPhish
    live_phish = fetch_openphish()

    # 2. Filter out URLs we have already successfully converted into HTML payloads by checking physical folder contents
    targets_phish = []
    for u in live_phish:
        if not os.path.exists(os.path.join(OUTPUT_DIR_PHISH, sanitize_filename(u))):
            targets_phish.append(u)
    
    dataset = [(url, 1) for url in targets_phish]
    
    if len(dataset) == 0:
        print("No new unique URLs found! Your dataset directories already possess HTMLs for completely identical targets.")
        return

    start_time = time.time()

    async with async_playwright() as p:
        print(f"Starting async scraper analyzing exactly {len(dataset)} new un-scraped targets...")
        
        chunk_size = 50 # Flush browser RAM completely every 50 sites!
        
        with tqdm(total=len(dataset), desc="Scraping HTMLs", unit="site") as pbar:
            for i in range(0, len(dataset), chunk_size):
                chunk = dataset[i:i+chunk_size]
                
                # Launch a mathematically pristine Firefox binary
                browser = await p.firefox.launch(headless=True)
                sem = asyncio.Semaphore(3)
                
                tasks = [bounded_scrape_url(sem, browser, url, label, pbar) for url, label in chunk]
                await asyncio.gather(*tasks)
                
                # Gracefully assassinate the browser locally to flush Mac Memory Swaps
                try:
                    await browser.close()
                except Exception:
                    pass
        
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    print(f"\nExecution Complete! Total time taken: {minutes} minutes and {seconds} seconds.")

if __name__ == "__main__":
    asyncio.run(main())
