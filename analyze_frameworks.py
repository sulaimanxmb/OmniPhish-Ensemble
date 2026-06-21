import os
from collections import Counter
from tqdm import tqdm
import re

def analyze_frameworks():
    dataset_dirs = ['Dataset/raw_html/phishing', 'Dataset/raw_html/benign']
    
    counts = Counter({
        'React / Next.js (SPA)': 0,
        'Vue / Nuxt.js (SPA)': 0,
        'Angular (SPA)': 0,
        'Other Modern SPAs (Webpack/Vite)': 0,
        'CMS & Server-Rendered (WordPress/PHP)': 0,
        'Traditional / Static HTML': 0
    })
    
    total_files = 0
    
    for d in dataset_dirs:
        if not os.path.exists(d):
            print(f"[!] Warning: Directory {d} not found.")
            continue
            
        for filename in tqdm(os.listdir(d), desc=f"Scanning {d}"):
            if not filename.endswith('.html'):
                continue
                
            filepath = os.path.join(d, filename)
            total_files += 1
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().lower()
                    
                    # React / Next.js
                    if re.search(r'(data-reactroot|id="root"|__next_data__|_next/static|id="__next"|react-dom|login-react|data-react-helmet)', content):
                        counts['React / Next.js (SPA)'] += 1
                    
                    # Vue / Nuxt.js
                    elif re.search(r'(data-v-[a-z0-9]+|__vue__|__nuxt__|vue\.min\.js|nuxt-link)', content):
                        counts['Vue / Nuxt.js (SPA)'] += 1
                        
                    # Angular
                    elif re.search(r'(ng-version|ng-app|_ngcontent|angular\.min\.js|app-root)', content):
                        counts['Angular (SPA)'] += 1
                        
                    # Other Modern SPAs / Builders
                    elif re.search(r'(webpackjsonp|chunk\.js|bundle\.js|vite|id="app"|_app|type="module")', content):
                        counts['Other Modern SPAs (Webpack/Vite)'] += 1
                        
                    # CMS & Server-Rendered (WordPress, PHP, Joomla, ASP.NET)
                    elif re.search(r'(wp-content|wp-includes|xmlrpc\.php|joomla|drupal|asp\.net|\.php)', content):
                        counts['CMS & Server-Rendered (WordPress/PHP)'] += 1
                        
                    # Static / Unknown
                    else:
                        counts['Traditional / Static HTML'] += 1
                        
            except Exception as e:
                pass

    print("\n" + "="*60)
    print("📊 DATASET FRAMEWORK DISTRIBUTION (DEEP SIGNATURE MATCHING)")
    print("="*60)
    print(f"Total HTML Files Scanned: {total_files}\n")
    
    for framework, count in counts.items():
        percentage = (count / total_files) * 100 if total_files > 0 else 0
        print(f"{framework:<40}: {count} files ({percentage:.1f}%)")
    print("="*60)
    print("Provide me these exact numbers, and I will put them into the LaTeX table!")

if __name__ == "__main__":
    analyze_frameworks()
