import asyncio
import os
import argparse
import pickle
import numpy as np
from playwright.async_api import async_playwright

from phish_scraper import check_login_heuristics
from baseline_features import extract_manual_features

async def fetch_html(url):
    print(f"Fetching live target URL: {url}")
    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            viewport={'width': 1280, 'height': 800}
        )
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=15000)
            html_content = await page.content()
            
            if not check_login_heuristics(html_content):
                print("\n[!] Warning: No login form or password input detected on this page!")
                
            return html_content
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            return None
        finally:
            await browser.close()

def predict(url):
    html_content = asyncio.run(fetch_html(url))
    if html_content is None:
        return
        
    print("Extracting Manual Heuristics for Baseline Model...")
    features = extract_manual_features(html_content)
    
    if not os.path.exists("weights/baseline_rf.pkl"):
        print("Error: Baseline model has not been trained yet. Please run baseline_trainer.py locally first.")
        return
        
    with open("weights/baseline_rf.pkl", "rb") as f:
        rf_model = pickle.load(f)
        
    # Reshape for single prediction
    X = features.reshape(1, -1)
    
    prediction = rf_model.predict(X)[0]
    prob = rf_model.predict_proba(X)[0][1]
    
    print("\n" + "="*60)
    print(f"🎯 Target: {url}")
    print("="*60)
    
    # Feature map for reasoning output
    feature_names = [
        "HTML Length", "Num Scripts", "Num Iframes", "Num Forms", 
        "Has Password Input", "Num Links", "Num Hidden Inputs",
        "Suspicious Form Action"
    ]
    
    if prediction == 1:
        print(f"🛑 BASELINE PREDICTION: PHISHING / MALICIOUS (Confidence: {prob*100:.2f}%)")
    else:
        print(f"✅ BASELINE PREDICTION: BENIGN / SAFE (Confidence: {(1 - prob)*100:.2f}%)")
        
    print("\n🔍 Extracted Heuristic Values (Why it made this decision):")
    for name, val in zip(feature_names, features):
        print(f"  • {name}: {int(val)}")
        
    print("\n[NOTE] This is the Traditional ML Baseline. Compare its accuracy to the Deep Learning Stacking Ensemble!")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict if a given valid URL is a phishing operation using Random Forest Baseline.")
    parser.add_argument("url", type=str, help="The URL to test (must explicitly include HTTP/HTTPS)")
    args = parser.parse_args()
    
    predict(args.url)
