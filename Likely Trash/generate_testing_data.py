import asyncio
import time
import urllib.request
import ssl
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

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

async def bounded_check_url(sem, browser, url):
    async with sem:
        try:
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
                viewport={'width': 1280, 'height': 800}
            )
            page = await context.new_page()
            # Fast 10 second timeout since we are just scanning
            await page.goto(url, wait_until="networkidle", timeout=10000)
            html_content = await page.content()
            is_login_page = check_login_heuristics(html_content)
            
            if is_login_page:
                # Print the URL clearly in bright green!
                print(f"\033[92m[FOUND LOGIN PHISH] {url}\033[0m")
                    
            await page.close()
            await context.close()
        except Exception:
            # Silently ignore dead links or timeouts to keep the terminal output clean
            pass

async def main():
    live_phish = fetch_openphish()

    if len(live_phish) == 0:
        print("No URLs fetched from OpenPhish!")
        return

    print(f"Starting asynchronous scanner across {len(live_phish)} targets to find live credential harvesters...\n")

    async with async_playwright() as p:
        chunk_size = 50
        for i in range(0, len(live_phish), chunk_size):
            chunk = live_phish[i:i+chunk_size]
            
            # Use Firefox for better stealth
            browser = await p.firefox.launch(headless=True)
            # Concurrency limit to prevent overwhelming network
            sem = asyncio.Semaphore(5)
            
            tasks = [bounded_check_url(sem, browser, url) for url in chunk]
            await asyncio.gather(*tasks)
            
            try:
                await browser.close()
            except Exception:
                pass
                
    print("\nScanning complete. Use these URLs with predict.py!")

if __name__ == "__main__":
    asyncio.run(main())
