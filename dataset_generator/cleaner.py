import os
import hashlib
import bs4
from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = [os.path.join(ROOT_DIR, 'dataset/raw_html/benign'), os.path.join(ROOT_DIR, 'dataset/raw_html/phishing')]

def inspect_file(filepath):
    issues = []
    file_hash = None
    
    try:
        size = os.path.getsize(filepath)
        if size < 800:
            issues.append("Extremely small file size (< 800 Bytes). Likely an empty page or fatal network drop.")
            
        if size > 3 * 1024 * 1024:
            issues.append("Extremely large file size (> 3MB). Likely bloated Javascript payload that will crash PyTorch RAM.")
            return filepath, issues, None
            
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        file_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
        # 1. Structural Checks
        stripped = content.strip()
        if stripped.startswith('{') or stripped.startswith('['):
            issues.append("Severe DOM Failure: File looks like raw JSON server data instead of physical HTML.")
            
        # 2. DOM Neural Extraction (BeautifulSoup Parsing)
        soup = BeautifulSoup(content, 'html.parser')
        if not soup.find('body'):
            issues.append("Structural Error: Missing standard <body> tag entirely. Not a valid webpage.")
            
        # 3. Blockers, Captchas, and Errors
        clean_soup = BeautifulSoup(content, 'html.parser') 
        for element in clean_soup(["script", "style"]):
            element.decompose()
        visible_text = clean_soup.get_text(separator=' ').lower()
        
        blockers = [
            "403 forbidden", 
            "please enable js and disable any ad blocker", 
            "checking if the site connection is secure", 
            "are you human", 
            "incapsula",
            "pardon our interruption",
            "just a moment...",
            "enable javascript and cookies"
        ]
        for b in blockers:
            if b in visible_text:
                issues.append(f"Security Blocker Detected: Contains known textual blockade ('{b}')")
                
        dead_indicators = [
            "404 not found",
            "page not found",
            "account suspended",
            "this account has been suspended"
        ]
        for d in dead_indicators:
            if d in visible_text:
                issues.append(f"Dead Host Detected: Page contains domain suspension or 404 message ('{d}')")
            
        # 4. Machine Learning Applicability
        inputs = soup.find_all('input')
        if len(inputs) == 0:
            issues.append("Irrelevant Target: Contains exactly zero <input> fields. Cannot be a credential harvester / login page.")

    except Exception as e:
        issues.append(f"Python processing exception occurred: {e}")
        
    return filepath, issues, file_hash

def main():
    all_files = []
    for d in DIRS:
        if os.path.exists(d):
            all_files.extend([os.path.join(d, f) for f in os.listdir(d) if f.endswith('.html')])
            
    if not all_files:
        print("No HTML files found in the dataset directories!")
        return
        
    print(f"Starting Deep HTML Structural Inspection & Deduplication on exactly {len(all_files)} offline files...")
    
    faulty_files = {}
    seen_hashes = set()
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(inspect_file, path): path for path in all_files}
        
        for future in tqdm(as_completed(futures), total=len(all_files), desc="Inspecting HTML Payloads"):
            filepath, issues, file_hash = future.result()
            
            if file_hash:
                if file_hash in seen_hashes:
                    issues.append("Exact Template Clone: This file is a 100% mathematical duplicate of another dataset file.")
                else:
                    seen_hashes.add(file_hash)
                    
            if issues:
                faulty_files[filepath] = issues
                
    if not faulty_files:
        print("\n[+] Flawless Dataset! 100% of your targets passed the raw structural inspection.")
    if faulty_files:
        print(f"\n[!] WARNING: {len(faulty_files)} files contain severe anomalies (Duplicates, Blockers, Json) that may poison the Neural Network weights!")
        report_file = "dataset_inspector_report.txt"
        print(f"Writing detailed diagnostic report to '{report_file}'...")
        
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"--- FAULTY HTML INSPECTION REPORT ---\nTotal Checked: {len(all_files)}\nTotal Faulty: {len(faulty_files)}\n\n")
            
            for path, issues in faulty_files.items():
                f.write(f"FILE: {path}\n")
                for issue in issues:
                    f.write(f"  - {issue}\n")
                f.write("\n")
                
                # Explicitly print every single faulty file directly to the console
                print(f"FILE: {path.split('/')[-1]}")
                for issue in issues:
                    print(f"   -> {issue}")
                    
                try:
                    os.remove(path)
                    print(f"   [DELETED] Successfully removed from disk.")
                except Exception as e:
                    print(f"   [ERROR DELETING] {e}")
             
    print(f"\n" + "-"*50)
    print(f"INSPECTION AND CLEANUP COMPLETE")
    print(f"Total Files Scanned  : {len(all_files)}")
    print(f"Total Defective Files DELETED: {len(faulty_files)}")
    print(f"Total Good Files Remaining   : {len(all_files) - len(faulty_files)}")
    print("-" * 50)

    # ---------------------------------------------------------
    # FEED SYNCHRONIZATION (Remove valid scraped URLs from text files)
    # ---------------------------------------------------------
    import re
    from urllib.parse import urlparse
    
    def sanitize_filename_local(url):
        parsed = urlparse(url)
        filename = f"{parsed.netloc}{parsed.path}"
        filename = re.sub(r'[^a-zA-Z0-9_\-]', '_', filename)
        return filename + ".html"
        
    benign_txt = os.path.join(ROOT_DIR, "top-1m.txt")
    if os.path.exists(benign_txt):
        print(f"\nSynchronizing '{benign_txt}' with the verified dataset...")
        with open(benign_txt, "r", encoding="utf-8") as f:
            original_lines = [line.strip() for line in f if line.strip()]
            
        remaining_lines = []
        removed_count = 0
        for line in original_lines:
            # Parse Tranco format (1,google.com) or fallback to raw URL
            parts = line.split(',')
            if len(parts) == 2:
                domain = parts[1].strip()
                url_to_check = f"https://{domain}"
            else:
                url_to_check = line
                
            expected_file = os.path.join(ROOT_DIR, 'dataset/raw_html/benign', sanitize_filename_local(url_to_check))
            if os.path.exists(expected_file):
                # File survived the cleaner! It's valid. Remove URL from the scrape feed.
                removed_count += 1
            else:
                # File was deleted by cleaner (or never scraped). Keep the URL in the text feed to retry.
                remaining_lines.append(line)
                
        if removed_count > 0:
            with open(benign_txt, "w", encoding="utf-8") as f:
                for line in remaining_lines:
                    f.write(line + "\n")
            print(f"[+] Removed {removed_count} completed/valid URLs from {benign_txt}.")
            print(f"[*] {len(remaining_lines)} pending URLs remain in the file queue.")
        else:
            print(f"[*] No valid, matched URLs were found to remove from {benign_txt}.")

if __name__ == "__main__":
    main()
