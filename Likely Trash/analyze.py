from bs4 import BeautifulSoup
import os

files = [
    "Dataset/raw_html/benign/www_yelp_com_login.html",
    "Dataset/raw_html/benign/www_ulta_com_guest_login.html",
    "Dataset/raw_html/benign/www_bcbs_com_login.html",
    "Dataset/raw_html/benign/login_newrelic_com_login.html",
    "Dataset/raw_html/benign/www_homedepot_com_auth_view_signin.html",
    "Dataset/raw_html/phishing/torombolorombi-ux_github_io_torombolorombi098_.html",
    "Dataset/raw_html/phishing/www_roblox_com_bn_games_9096377689_Killstreak-Sword-Fighting.html",
    "Dataset/raw_html/phishing/senseihx4_github_io_Netflix-clone-_.html",
    "Dataset/raw_html/phishing/gorgeous-puffpuff-d1509c_netlify_app_.html",
    "Dataset/raw_html/phishing/docusignsecuredfile0_weebly_com_.html"
]

for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
        soup = BeautifulSoup(content, 'html.parser')
        forms = len(soup.find_all('form'))
        passwords = len(soup.find_all('input', type=lambda x: x and x.lower() == 'password'))
        if passwords == 0:
            passwords = len([i for i in soup.find_all('input') if i.get('type') == 'password' or i.get('name') == 'password'])
        inputs = len(soup.find_all('input'))
        scripts = len(soup.find_all('script'))
        print(f"File: {os.path.basename(f)}")
        print(f"Size: {len(content)} bytes")
        print(f"Forms: {forms}, Inputs: {inputs}, Passwords: {passwords}, Scripts: {scripts}")
        print("-" * 40)
    except Exception as e:
        print(f"Error reading {f}: {e}")
