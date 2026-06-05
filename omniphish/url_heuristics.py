import re
from urllib.parse import urlparse

# Target highly-phished brands for homograph / typosquatting detection
TOP_BRANDS = [
    "google", "microsoft", "facebook", "amazon", "apple", 
    "netflix", "paypal", "instagram", "linkedin", "twitter",
    "chase", "bankofamerica", "wellsfargo", "github", "dropbox"
]

# TLDs heavily abused by phishing kits due to low cost
SUSPICIOUS_TLDS = [
    ".xyz", ".top", ".pw", ".cc", ".tk", ".ml", ".ga", ".cf", ".gq",
    ".site", ".online", ".club", ".biz", ".info", ".workers.dev", ".herokuapp.com"
]

def is_suspicious_action(action_url):
    """
    Evaluates if a form action URL matches known phishing drop patterns.
    """
    if not action_url:
        return 1 # Empty action relies on hidden JS (suspicious)
        
    action = action_url.lower().strip()
    if action in ['#', 'javascript:void(0)', 'javascript:;']:
        return 1
        
    # Raw IP addresses
    if re.search(r'https?://[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+', action):
        return 1
        
    # Common phishing drop scripts
    if re.search(r'(login|post|action|send|mail|submit)\.php$', action):
        return 1
        
    # Known free hosting or worker tunnels
    if re.search(r'(ngrok\.io|000webhost|herokuapp|firebaseapp|workers\.dev)', action):
        return 1
        
    return 0


def levenshtein_distance(s1, s2):
    """Calculates the minimum edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def extract_url_heuristics(url_or_domain):
    """
    Extracts 2 numerical features from a URL or raw domain string:
    1. Levenshtein Suspicion Score (0 to 1) - High if it looks like a typo of a brand.
    2. Suspicious Domain Flag (0 or 1) - High if raw IP or suspicious TLD.
    """
    if not url_or_domain:
        return 0.0, 1.0 # Empty URL is highly suspicious in context
        
    # Standardize to domain
    if url_or_domain.startswith("http"):
        parsed = urlparse(url_or_domain)
        domain = parsed.netloc
    else:
        domain = url_or_domain
        
    domain = domain.lower().strip()
    
    # 1. Suspicious Domain Flag
    suspicious_domain_flag = 0.0
    
    # Is it a raw IP address? (e.g., 192.168.1.1)
    if re.search(r'^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+', domain):
        suspicious_domain_flag = 1.0
        
    # Does it have a known abused TLD?
    for tld in SUSPICIOUS_TLDS:
        if domain.endswith(tld):
            suspicious_domain_flag = 1.0
            break
            
    # 2. Levenshtein Brand Distance Score
    # We want to catch things like "g00gle" (distance 2 from "google")
    # But we don't want to penalize the actual brand "google.com" (distance 0)
    
    levenshtein_score = 0.0
    
    # Extract the main words from the domain (ignoring dots, hyphens)
    domain_words = re.split(r'[\.\-]', domain)
    
    for word in domain_words:
        if len(word) < 4: 
            continue # Too short to evaluate reliably against brands
            
        for brand in TOP_BRANDS:
            if word == brand:
                # It's an EXACT match of the brand name.
                # If the domain is actually legit (e.g. google.com), this is fine.
                # But if it's google-secure-login.xyz, the TLD check above catches it!
                continue
                
            dist = levenshtein_distance(word, brand)
            
            # If distance is 1 or 2, it's highly likely a typosquat! (e.g., rnicrosoft, f4cebook)
            # We scale the score: distance 1 is very suspicious (score 1.0)
            # Distance 2 is quite suspicious (score 0.8)
            if dist == 1:
                levenshtein_score = max(levenshtein_score, 1.0)
            elif dist == 2 and len(brand) > 5: # Only penalize distance 2 for longer brands
                levenshtein_score = max(levenshtein_score, 0.8)
                
    return levenshtein_score, suspicious_domain_flag

if __name__ == "__main__":
    # Test cases
    test_urls = [
        "https://accounts.google.com",         # Legit
        "http://accounts-g00gle.xyz/login",    # Phish (Typosquat + XYZ)
        "https://rnicrosoft-update.com",       # Phish (Typosquat)
        "192.168.1.100",                       # Phish (Raw IP)
        "https://www.paypal-secure-auth.top"   # Phish (Exact brand but .top TLD)
    ]
    
    for url in test_urls:
        score, flag = extract_url_heuristics(url)
        print(f"URL: {url} -> Levenshtein: {score}, Suspicious TLD/IP: {flag}")
