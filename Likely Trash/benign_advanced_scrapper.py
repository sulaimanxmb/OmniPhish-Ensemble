import asyncio
import os
import re
import time
import random
from tqdm import tqdm
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from urllib.parse import urlparse

OUTPUT_DIR_BENIGN = "dataset/raw_html/benign"
os.makedirs(OUTPUT_DIR_BENIGN, exist_ok=True)

# ANSI colors for terminal logging
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def sanitize_filename(url):
    parsed = urlparse(url)
    filename = f"{parsed.netloc}{parsed.path}"
    filename = re.sub(r'[^a-zA-Z0-9_\-]', '_', filename)
    return filename + ".html"

async def check_login_heuristics_playwright(page):
    """
    Evaluates the page directly using Playwright locators which pierce Shadow DOMs natively.
    Looks for forms, emails, usernames, passwords, autocomplete hints, aria-labels, and OAuth/SSO buttons.
    """
    try:
        # Check for standard input types
        has_password = await page.locator("input[type='password' i]").count() > 0
        has_email = await page.locator("input[type='email' i]").count() > 0
        
        # Extended to look for 'mobile' in IDs, Names, and Placeholders
        has_username = await page.locator(
            "input[name*='user' i], input[id*='user' i], input[name*='email' i], input[id*='email' i], "
            "input[type='tel' i], input[name*='phone' i], input[id*='phone' i], "
            "input[name*='mobile' i], input[id*='mobile' i], input[type='number' i], "
            "input[placeholder*='mobile' i], input[placeholder*='phone' i], input[placeholder*='number' i], input[placeholder*='email' i]"
        ).count() > 0
        
        # Improvement 5: Detect autocomplete attributes (React/Angular apps often use these instead of type="password")
        has_autocomplete_auth = await page.locator(
            "input[autocomplete='username' i], input[autocomplete='current-password' i], "
            "input[autocomplete='new-password' i], input[autocomplete='email' i]"
        ).count() > 0
        
        # Improvement 6: Detect aria-label on inputs (modern SPA frameworks use these for accessibility)
        has_aria_auth = await page.locator(
            "input[aria-label*='email' i], input[aria-label*='password' i], "
            "input[aria-label*='username' i], input[aria-label*='phone' i], "
            "input[aria-label*='mobile' i], input[aria-label*='login' i], "
            "input[aria-label*='sign in' i]"
        ).count() > 0
        
        # Check if the page actually has a button or a clear heading meant for logging in / continuing
        has_auth_context = await page.locator("button, a, input[type='submit'], div[role='button'], h1, h2, h3, h4, h5, h6, span, p").filter(has_text=re.compile(r"\b(log in|login|sign in|signin|continue|next|submit|sign up|signup|otp)\b", re.IGNORECASE)).count() > 0
        
        # Improvement 7: Detect OAuth/SSO buttons ("Sign in with Google/Apple/Microsoft" = authentication page)
        has_oauth = await page.locator("button, a, div[role='button']").filter(has_text=re.compile(
            r"\b(sign in with|log in with|continue with|login with)\s+(google|apple|microsoft|facebook|github|twitter|sso)\b", re.IGNORECASE
        )).count() > 0
        
        # Merge new detections into existing flags
        if has_autocomplete_auth or has_aria_auth:
            has_username = True
        if has_oauth:
            has_auth_context = True
        
        # Strict evaluation: A page with only a username/phone field MUST have auth context to be valid (prevents search bar false positives)
        if has_password or (has_email and has_username) or (has_username and has_auth_context):
            return True
            
        # Check standard forms
        has_form = await page.locator("form").count() > 0
        if has_form and (has_email or (has_username and has_auth_context)):
            return True
        
        # OAuth/SSO pages may have no form or inputs at all, just buttons
        if has_oauth:
            return True
            
        # Check iframes for login fields (including new autocomplete and aria-label selectors)
        for frame in page.frames:
            if await frame.locator(
                "input[type='password' i], input[type='email' i], input[type='tel' i], "
                "input[name*='phone' i], input[id*='phone' i], input[placeholder*='mobile' i], "
                "input[autocomplete='username' i], input[autocomplete='current-password' i], "
                "input[aria-label*='email' i], input[aria-label*='password' i]"
            ).count() > 0:
                return True
                
        return False
    except Exception:
        return False

async def extract_full_html(page):
    """Extracts HTML from main page and deeply from all child iframes"""
    html = await page.content()
    for frame in page.frames:
        if frame != page.main_frame:
            try:
                frame_html = await frame.content()
                # Append iframe content as a pseudo-block so neural network can read it
                html += f"\n<!-- IFRAME CONTENT FROM {frame.url} -->\n<div>{frame_html}</div>"
            except Exception:
                pass
    return html

async def process_single_url(browser, url):
    """Returns True if successfully saved, False otherwise"""
    max_retries = 3
    
    for attempt in range(max_retries):
        context = None
        page = None
        try:
            # Create a brand new isolated context for every single attempt
            # Improvement 2: Add locale and timezone to match IP geolocation (prevents Cloudflare fingerprint mismatch)
            # Improvement 4: Add extra HTTP headers that real browsers always send (missing ones trigger Cloudflare)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
                viewport={'width': 1280, 'height': 800},
                java_script_enabled=True,
                bypass_csp=True,
                locale='en-US',
                timezone_id='Asia/Kolkata',
                extra_http_headers={
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none',
                    'Sec-Fetch-User': '?1',
                    'Upgrade-Insecure-Requests': '1'
                }
            )
            page = await context.new_page()
            
            # Feature 2 & 21: Increased timeout and retry loop
            print(f"\n[{attempt+1}/{max_retries}] Navigating to {url}...")
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            
            # Improvement 3: Increased delay to 5-8 seconds for Cloudflare JS challenge resolution
            await page.wait_for_timeout(random.randint(5000, 8000))
            
            # Feature 4 & 23: Wait for selectors dynamically rather than relying purely on networkidle
            try:
                await page.wait_for_selector("input, button, form", timeout=5000)
            except PlaywrightTimeoutError:
                pass # Continue anyway, some pages are weird
                
            # Initial check
            is_login = await check_login_heuristics_playwright(page)
            
            if not is_login:
                # Features 9, 10, 18: Try ALL matching login buttons, handle hovers, and interact
                login_buttons = await page.locator("button, a, div[role='button']").filter(has_text=re.compile(r"\b(log in|login|sign in|signin)\b", re.IGNORECASE)).all()
                
                for btn in login_buttons:
                    try:
                        # Feature 7: Scroll into view
                        await btn.scroll_into_view_if_needed(timeout=2000)
                        
                        # Feature 15: Random delay before hover
                        await page.wait_for_timeout(random.randint(500, 1500))
                        
                        # Feature 10: Hover over element to trigger potential JS dropdowns
                        await btn.hover(timeout=2000)
                        await page.wait_for_timeout(random.randint(500, 1000))
                        
                        # Click the button
                        await btn.click(timeout=3000)
                        
                        # Feature 8 & 24: Delay after clicking to allow navigation or DOM update
                        await page.wait_for_timeout(random.randint(2000, 4000))
                        
                        # Check again!
                        is_login = await check_login_heuristics_playwright(page)
                        if is_login:
                            break # We successfully found the login form, stop clicking buttons
                    except Exception:
                        # Continue to next button
                        continue
                        
            if is_login:
                # Feature 5 & 11: Re-fetch full HTML including all iframes
                final_html = await extract_full_html(page)
                
                filename = sanitize_filename(url)
                filepath = os.path.join(OUTPUT_DIR_BENIGN, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(final_html)
                print(f"{GREEN}[SUCCESS] Saved valid login HTML for {url}{RESET}")
                return True
            else:
                print(f"{YELLOW}[NO LOGIN FORM FOUND] {url} on attempt {attempt+1}{RESET}")
                
        except PlaywrightTimeoutError:
            print(f"{RED}[TIMEOUT] {url} on attempt {attempt+1}{RESET}")
        except Exception as e:
            # Feature 19: Verbose exception handling instead of silent except blocks
            print(f"{RED}[ERROR] {url} -> {str(e)}{RESET}")
        finally:
            if page: await page.close()
            if context: await context.close()
            
    return False

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

    # Filter out already scraped URLs
    targets_benign = []
    for u in manual_benign:
        if not os.path.exists(os.path.join(OUTPUT_DIR_BENIGN, sanitize_filename(u))):
            targets_benign.append(u)
    
    if len(targets_benign) == 0:
        print("No new unique Benign URLs found! Directory is saturated.")
        return

    print(f"Starting ADVANCED Headful Benign Scraper analyzing {len(targets_benign)} enterprise targets...")
    
    failed_urls = []
    
    async with async_playwright() as p:
        # Feature 1: Headful Mode (headless=False)
        # Using Chromium by default as requested
        # Improvement 1: Use Firefox instead of Chromium — Cloudflare heavily fingerprints Chromium for automation markers
        browser = await p.firefox.launch(
            headless=False
        )
        
        # Feature 14: Strict Concurrency limit of 1 to preserve Apple Silicon RAM and prevent crashing
        chunk_size = 5 # Flush browser every 5 URLs
        
        with tqdm(total=len(targets_benign), desc="Scraping SPA HTMLs", unit="site") as pbar:
            for i in range(0, len(targets_benign), chunk_size):
                chunk = targets_benign[i:i+chunk_size]
                
                for url in chunk:
                    # Executing sequentially with strict concurrency limit = 1
                    success = await process_single_url(browser, url)
                    if not success:
                        # Feature 20: Log failed URLs for reprocessing
                        failed_urls.append(url)
                    pbar.update(1)
                
                # Exterminate browser safely to execute Memory Recapture Loop
                try:
                    await browser.close()
                except Exception:
                    pass
                    
                # Re-launch fresh browser for next chunk to clear cookies/cache
                if i + chunk_size < len(targets_benign):
                    browser = await p.firefox.launch(
                        headless=False
                    )
                    
        try:
            await browser.close()
        except Exception:
            pass

    if failed_urls:
        print(f"\n{RED}--- FINAL FAILED URLS ({len(failed_urls)}) ---{RESET}")
        for f_url in failed_urls:
            print(f"{RED}{f_url}{RESET}")
        print(f"{YELLOW}These URLs exhausted all 3 retries or hit unbreakable CAPTCHAs.{RESET}")

if __name__ == "__main__":
    asyncio.run(main())
