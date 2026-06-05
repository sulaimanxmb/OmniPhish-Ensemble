import os
import re
from bs4 import BeautifulSoup

def clean_html(html_content):
    """
    Remove <style>, <svg>, and base64 encoded strings.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove unwanted tags
    for tag in soup(['style', 'svg', 'noscript', 'meta']):
        tag.decompose()
        
    # Remove base64 encoded strings
    cleaned_html = str(soup)
    cleaned_html = re.sub(r'data:image/[^;]+;base64,[a-zA-Z0-9+/=]+', '', cleaned_html)
    cleaned_html = re.sub(r'data:application/[^;]+;base64,[a-zA-Z0-9+/=]+', '', cleaned_html)
    
    return cleaned_html

def extract_codebert_tags(html_content):
    """
    Isolates only <script>, <form>, and <a> tags.
    Returns a string of extracted elements.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    extracted_text = []
    
    # Extract <form>
    for form in soup.find_all('form'):
        extracted_text.append(str(form))
        
    # Extract <script>
    for script in soup.find_all('script'):
        if script.string:
            extracted_text.append(f"<script>{script.string}</script>")
        elif script.get('src'):
            extracted_text.append(f"<script src='{script.get('src')}'></script>")
            
    # Extract <a>
    for a in soup.find_all('a'):
        href = a.get('href', '')
        text = a.get_text(strip=True)
        extracted_text.append(f"<a href='{href}'>{text}</a>")
        
    return "\n".join(extracted_text)

if __name__ == "__main__":
    # Test script if needed
    sample_html = "<html><head><style>.css{ color: red; }</style></head><body><form action='/login'><input type='password'></form><svg width='10'></svg></body></html>"
    
    cleaned = clean_html(sample_html)
    print("Cleaned HTML length:", len(cleaned))
    
    extracted = extract_codebert_tags(cleaned)
    print("Extracted elements:\n", extracted)
