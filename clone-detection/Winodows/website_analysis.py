import asyncio
import requests
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from io import BytesIO
from datetime import datetime
from pyppeteer import launch
import torch
from torchvision import models, transforms

# ✅ Path to System-Installed Google Chrome (Windows)
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

# ✅ Load Pre-Trained ResNet Model for Deep Feature Extraction
resnet = models.resnet50(pretrained=True)
resnet.eval()

# ✅ Function to Preprocess Images for Deep Learning Similarity
def preprocess_image(img):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(img).unsqueeze(0)

# ✅ Extract Features Using ResNet
def extract_features(img):
    img_tensor = preprocess_image(img)
    with torch.no_grad():
        features = resnet(img_tensor)
    return features

# ✅ Compute Deep Learning-Based Image Similarity
def deep_learning_similarity(img1, img2):
    features1 = extract_features(img1)
    features2 = extract_features(img2)
    cosine_sim = torch.nn.functional.cosine_similarity(features1, features2).item()
    return cosine_sim

# ✅ Function to Fetch Website HTML Content using Pyppeteer (Windows Chrome)
async def fetch_html(url):
    try:
        browser = await launch(
            headless=True,
            executablePath=CHROME_PATH,  # ✅ Use system-installed Chrome
            args=["--no-sandbox"]
        )
        page = await browser.newPage()
        await page.goto(url, timeout=60000)
        html_content = await page.content()
        await browser.close()
        return html_content
    except Exception as e:
        print(f"⚠️ Error fetching HTML for {url}: {e}")
        return None

# ✅ Function to Capture Full Website Screenshot (Windows)
async def fetch_screenshot(url, output_filename):
    try:
        browser = await launch(
            headless=True,
            executablePath=CHROME_PATH,  # ✅ Use system-installed Chrome
            args=["--no-sandbox"]
        )
        page = await browser.newPage()
        await page.setViewport({"width": 1920, "height": 1080})
        await page.goto(url, timeout=60000, waitUntil="networkidle2")
        await page.screenshot(path=output_filename, fullPage=True)
        await browser.close()
    except Exception as e:
        print(f"⚠️ Error fetching screenshot for {url}: {e}")

# ✅ Function to Calculate Jaccard Similarity for Text Comparison
def jaccard_similarity(text1, text2):
    set1, set2 = set(text1.split()), set(text2.split())
    return len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0

# ✅ Function to Fetch WHOIS Information
def get_whois_info(url):
    try:
        domain = url.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]
        response = requests.get(
            f"https://www.whoisxmlapi.com/whoisserver/WhoisService?apiKey=YOUR_API_KEY&domainName={domain}&outputFormat=json"
        )
        if response.status_code == 200:
            whois_data = response.json()
            registrar = whois_data.get("WhoisRecord", {}).get("registrarName", "Unknown")
            creation_date = whois_data.get("WhoisRecord", {}).get("createdDate", None)
            return registrar, creation_date
        else:
            return None, None
    except Exception as e:
        print(f"⚠️ WHOIS lookup failed: {e}")
        return None, None

# ✅ Function to Compare Text Content Similarity using TF-IDF
def text_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

# ✅ Function to Compare Images Using SSIM & Deep Learning Features
def compare_images(image1_path, image2_path):
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise ValueError("Error loading one or both images.")

    img1 = cv2.resize(img1, (512, 512))
    img2 = cv2.resize(img2, (512, 512))

    # SSIM Similarity
    ssim_value = ssim(img1, img2)

    # Deep Learning Similarity
    deep_sim_value = deep_learning_similarity(img1, img2)

    # Weighted Combination
    combined_score = 0.5 * ssim_value + 0.5 * deep_sim_value
    return combined_score

# ✅ Function to Check for Cloned Websites
async def check_clone(site1, site2):
    if not site1.startswith("http"):
        site1 = "https://" + site1
    if not site2.startswith("http"):
        site2 = "https://" + site2

    # Fetch HTML
    html1, html2 = await fetch_html(site1), await fetch_html(site2)
    if not html1 or not html2:
        return {"error": "Could not fetch HTML from one or both sites"}

    # Compare HTML Structure
    soup1, soup2 = BeautifulSoup(html1, "html.parser"), BeautifulSoup(html2, "html.parser")
    html_similarity = jaccard_similarity(soup1.get_text(), soup2.get_text())

    # Capture Screenshots
    await fetch_screenshot(site1, "site1.png")
    await fetch_screenshot(site2, "site2.png")

    # Compare Screenshots
    image_similarity = compare_images("site1.png", "site2.png")

    # Fetch WHOIS Data
    registrar1, creation_date1 = get_whois_info(site1)
    registrar2, creation_date2 = get_whois_info(site2)
    whois_match = registrar1 == registrar2 and creation_date1 == creation_date2

    # Final Result
    result = {
        "html_similarity": round(html_similarity, 2),
        "image_similarity": round(image_similarity, 2),
        "whois_match": whois_match,
        "clone_detected": html_similarity > 0.7 or image_similarity > 0.70 or whois_match,
    }
    return result