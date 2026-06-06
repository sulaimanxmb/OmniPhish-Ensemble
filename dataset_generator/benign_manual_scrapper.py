import asyncio
import os
import re
from playwright.async_api import async_playwright
from urllib.parse import urlparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR_BENIGN = os.path.join(ROOT_DIR, "dataset", "raw_html", "benign")
LOG_FILE = os.path.join(ROOT_DIR, "dataset", ".scraped_log.txt")
TOP_1M_FILE = os.path.join(ROOT_DIR, "top-1m.txt")
os.makedirs(OUTPUT_DIR_BENIGN, exist_ok=True)

# ANSI colors for terminal logging
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'

def sanitize_filename(url):
    parsed = urlparse(url)
    filename = f"{parsed.netloc}{parsed.path}"
    filename = re.sub(r'[^a-zA-Z0-9_\-]', '_', filename)
    return filename + ".html"

# Removed check_login_heuristics_playwright as we are now fully manual

async def extract_full_html(page):
    html = await page.content()
    for frame in page.frames:
        if frame != page.main_frame:
            try:
                frame_html = await frame.content()
                html += f"\n<!-- IFRAME CONTENT FROM {frame.url} -->\n<div>{frame_html}</div>"
            except Exception:
                pass
    return html

async def main():
    seen_urls = set()
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                seen_urls.add(line.strip())

    manual_benign = []
    global_lines = []
    if os.path.exists(TOP_1M_FILE):
        with open(TOP_1M_FILE, "r", encoding="utf-8") as f:
            global_lines = f.readlines()
            for line_raw in global_lines:
                line = line_raw.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) == 2:
                        domain = parts[1].strip()
                        url = f"https://{domain}"
                        if url not in seen_urls:
                            manual_benign.append((line_raw, url))
    else:
        print(f"Error: {TOP_1M_FILE} missing!")
        return

    targets_benign = []
    for line_raw, u in manual_benign:
        if not os.path.exists(os.path.join(OUTPUT_DIR_BENIGN, sanitize_filename(u))):
            targets_benign.append((line_raw, u))
    
    if len(targets_benign) == 0:
        print("No new unique Benign URLs found! All URLs in text file have already been successfully saved.")
        return

    print("\n" + "="*70)
    print(f"{CYAN}🤖 CYBORG SCRAPER ACTIVATED: {len(targets_benign)} Targets Remaining{RESET}")
    print("="*70)
    print(f"{YELLOW}INSTRUCTIONS:{RESET}")
    print(" 1. The browser will open the target URL automatically.")
    print(" 2. You have FULL manual control. Click around, bypass CAPTCHAs, or navigate to the Login page.")
    print(" 3. I will inject a RED 'SAVE LOGIN HTML & NEXT' button in the top right corner of the page.")
    print(" 4. When you find the login page, simply CLICK THAT BUTTON. It will auto-save and move to the next URL.")
    print(" 5. If the website is dead or has no login, simply CLOSE the browser window (click the 'X') to skip it.")
    print("="*70 + "\n")
    
    async with async_playwright() as p:
        for index, (line_raw, url) in enumerate(targets_benign):
            print(f"{CYAN}[{index+1}/{len(targets_benign)}] Loading: {url}{RESET} (Waiting for you...)")
            
            browser = await p.firefox.launch(
                headless=False
            )
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
            
            # Inject a floating button into every page/tab the user navigates to
            await context.add_init_script("""
                window.addEventListener('DOMContentLoaded', () => {
                    if (window.top !== window.self) return; // Only add to main window, not iframes
                    const btn = document.createElement('button');
                    btn.innerHTML = '📥 SAVE LOGIN HTML & NEXT';
                    btn.style.position = 'fixed';
                    btn.style.bottom = '20px';
                    btn.style.right = '20px';
                    btn.style.zIndex = '2147483647'; // Max z-index to stay on top
                    btn.style.padding = '15px 20px';
                    btn.style.backgroundColor = '#e74c3c';
                    btn.style.color = '#ffffff';
                    btn.style.fontWeight = 'bold';
                    btn.style.fontSize = '16px';
                    btn.style.border = '3px solid white';
                    btn.style.borderRadius = '8px';
                    btn.style.cursor = 'pointer';
                    btn.style.boxShadow = '0 4px 6px rgba(0,0,0,0.3)';
                    btn.onclick = (e) => { 
                        e.preventDefault();
                        window.__AICS_SAVE_TRIGGERED = true; 
                        btn.innerHTML = '⏳ SAVING...'; 
                        btn.style.backgroundColor = '#f39c12';
                    };
                    document.body.appendChild(btn);
                });
            """)
            
            page = await context.new_page()
            
            try:
                # We give a massive timeout of 2 minutes for the initial load just in case of slow internet
                await page.goto(url, wait_until="domcontentloaded", timeout=120000)
            except Exception as e:
                print(f"{YELLOW}Initial load took too long, but you can still use the browser!{RESET}")
            
            # The Infinite "Cyborg" Loop
            while True:
                try:
                    # Check if the user closed ALL windows manually
                    if len(context.pages) == 0:
                        print(f"{RED}[SKIPPED] You closed all browser windows. Moving to next URL.{RESET}\n")
                        break
                        
                    is_login_found = False
                    target_page = None
                    
                    # Scan EVERY open tab dynamically to see if the user clicked the injected button
                    for tab in context.pages:
                        if not tab.is_closed():
                            try:
                                triggered = await tab.evaluate("window.__AICS_SAVE_TRIGGERED")
                                if triggered:
                                    is_login_found = True
                                    target_page = tab
                                    break
                            except Exception:
                                pass
                    
                    if is_login_found:
                        # Add a tiny delay to ensure any SPA animations finish rendering
                        await target_page.wait_for_timeout(1000)
                        
                        final_html = await extract_full_html(target_page)
                        filename = sanitize_filename(url)
                        filepath = os.path.join(OUTPUT_DIR_BENIGN, filename)
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(final_html)
                            
                        print(f"{GREEN}[SUCCESS] You clicked the SAVE button! Saved HTML.{RESET}\n")
                        break
                        
                    # Scan the screen once every second
                    await page.wait_for_timeout(1000)
                    
                except Exception:
                    # Exception occurs if the user closes the window EXACTLY during the heuristic check
                    print(f"{RED}[SKIPPED] Browser was closed. Moving to next URL.{RESET}\n")
                    break
                    
            # Record that we have processed this URL (whether saved or skipped)
            with open(LOG_FILE, "a", encoding="utf-8") as log:
                log.write(url + "\n")
                
            # Physically shrink the top-1m.txt file
            if line_raw in global_lines:
                global_lines.remove(line_raw)
                with open(TOP_1M_FILE, "w", encoding="utf-8") as f:
                    f.writelines(global_lines)
                    
            try:
                await browser.close()
            except Exception:
                pass
                
    print(f"\n{CYAN}Manual Scraping Session Complete!{RESET}")

if __name__ == "__main__":
    asyncio.run(main())
