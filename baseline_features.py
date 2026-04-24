import re
from bs4 import BeautifulSoup
import numpy as np

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

def extract_manual_features(html_content):
    """
    Extracts traditional, manual heuristic features from raw HTML
    for the baseline Random Forest model.
    Returns a 1D numpy array of exactly 8 features.
    """
    # Fallback for empty/failed fetches
    if not html_content or not isinstance(html_content, str):
        return np.zeros(8)

    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 1. HTML Length
    html_length = len(html_content)
    
    # 2. Number of scripts
    num_scripts = len(soup.find_all('script'))
    
    # 3. Number of iframes (Often used to embed malicious login forms)
    num_iframes = len(soup.find_all('iframe'))
    
    # 4. Number of forms
    num_forms = len(soup.find_all('form'))
    
    # 5. Presence of password input (Critical heuristic)
    has_password = 1 if soup.find('input', type=lambda t: t and t.lower() == 'password') else 0
    
    # 6. Number of links
    num_links = len(soup.find_all('a'))
    
    # 7. Number of hidden inputs (Often used to route stolen data secretly)
    num_hidden_inputs = len(soup.find_all('input', type=lambda t: t and t.lower() == 'hidden'))
    
    # 8. Suspicious Form Action
    suspicious_form_action = 0
    for form in soup.find_all('form'):
        if is_suspicious_action(form.get('action', '')):
            suspicious_form_action = 1
            break
            
    features = [
        html_length,
        num_scripts,
        num_iframes,
        num_forms,
        has_password,
        num_links,
        num_hidden_inputs,
        suspicious_form_action
    ]
    
    return np.array(features, dtype=np.float32)

if __name__ == "__main__":
    test_html = "<html><body><form><input type='password'></form><script></script></body></html>"
    print("Test Extraction:")
    print(extract_manual_features(test_html))
