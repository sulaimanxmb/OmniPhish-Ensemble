import asyncio
import os
import re
import time
from tqdm import tqdm
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from urllib.parse import urlparse

OUTPUT_DIR_BENIGN = "dataset/raw_html/benign"
os.makedirs(OUTPUT_DIR_BENIGN, exist_ok=True)

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

async def bounded_benign_scrape(sem, browser, url, pbar):
    async with sem:
        try:
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
                viewport={'width': 1280, 'height': 800}
            )
            page = await context.new_page()
            
            # Navigate to the Benign target
            await page.goto(url, wait_until="networkidle", timeout=15000)
            
            # First, evaluate the baseline page without executing complex physics
            html_content = await page.content()
            is_login_page = check_login_heuristics(html_content)
            
            if not is_login_page:
                # --- The Dynamic SPA "Click" Fallback Strategy ---
                try:
                    login_button = page.locator("button, a").filter(has_text=re.compile(r"(?i)\b(log in|login|sign in|signin)\b"))
                    
                    if await login_button.count() > 0:
                        await login_button.first.click(timeout=3000)
                        await page.wait_for_timeout(2000)
                        
                        # Re-extract the HTML after allowing the modal Javascript to animate
                        html_content = await page.content()
                        is_login_page = check_login_heuristics(html_content)
                except Exception:
                    pass
            
            if is_login_page:
                filename = sanitize_filename(url)
                filepath = os.path.join(OUTPUT_DIR_BENIGN, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                    
            await page.close()
            await context.close()
        except Exception:
            pass
        finally:
            pbar.update(1)

async def main():
    manual_benign = []
    if os.path.exists("benign_dataset.txt"):
        with open("benign_dataset.txt", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    manual_benign.append(line.strip())
    else:
        print("Error: benign_dataset.txt missing from directory!")
        return

    # Filter out URLs we have perfectly preserved physically
    targets_benign = []
    for u in manual_benign:
        if not os.path.exists(os.path.join(OUTPUT_DIR_BENIGN, sanitize_filename(u))):
            targets_benign.append(u)
    
    if len(targets_benign) == 0:
        print("No new unique Benign URLs found! Directory is saturated.")
        return

    start_time = time.time()

    async with async_playwright() as p:
        print(f"Starting Dedicated Benign SPA UI scraper analyzing {len(targets_benign)} enterprise targets...")
        
        chunk_size = 50 # Flush browser RAM completely every 50 loops!
        
        with tqdm(total=len(targets_benign), desc="Scraping Safe HTMLs", unit="site") as pbar:
            for i in range(0, len(targets_benign), chunk_size):
                chunk = targets_benign[i:i+chunk_size]
                
                # Launch untouched pristine Firefox sequence
                browser = await p.firefox.launch(headless=True)
                sem = asyncio.Semaphore(3)
                
                tasks = [bounded_benign_scrape(sem, browser, url, pbar) for url in chunk]
                await asyncio.gather(*tasks)
                
                # Exterminate browser safely to execute Memory Recapture Loop
                try:
                    await browser.close()
                except Exception:
                    pass
        
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    print(f"\nExecution Complete! Total time processed: {minutes} minutes and {seconds} seconds.")

if __name__ == "__main__":
    asyncio.run(main())
