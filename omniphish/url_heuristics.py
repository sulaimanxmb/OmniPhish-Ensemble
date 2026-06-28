import re

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

if __name__ == "__main__":
    # Test cases
    test_actions = ["login.php", "https://192.168.1.1/submit", "https://legit.com/auth"]
    for act in test_actions:
        print(f"Action: {act} -> Suspicious: {is_suspicious_action(act)}")
