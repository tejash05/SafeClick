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


# ✅ Function to Fetch Website HTML Content using Pyppeteer
async def fetch_html(url):
    try:
        browser = await launch(headless=True, args=["--no-sandbox"])
        page = await browser.newPage()
        await page.goto(url, timeout=60000)
        html_content = await page.content()
        await browser.close()
        return html_content
    except Exception as e:
        print(f"⚠️ Error fetching HTML for {url}: {e}")
        return None


# ✅ Function to Capture Full Website Screenshot with Higher Accuracy
async def fetch_screenshot(url):
    try:
        browser = await launch(headless=True, args=["--no-sandbox"])
        page = await browser.newPage()
        await page.setViewport({"width": 1920, "height": 1080})  # Higher resolution

        # Scroll Down for Lazy Loading
        await page.goto(url, timeout=60000)
        page_height = await page.evaluate("document.body.scrollHeight")
        for i in range(0, page_height, 500):
            await page.evaluate(f"window.scrollTo(0, {i})")
            await asyncio.sleep(0.5)  # Allow time for loading

        # Capture Full Page Screenshot
        screenshot = await page.screenshot(fullPage=True)
        await browser.close()

        # Normalize Image
        img = Image.open(BytesIO(screenshot))
        img = img.resize((1024, 1024))  # Standardized resolution
        img = img.convert("L")  # Convert to grayscale

        # 🔹 Contrast normalization for better comparison
        img_np = np.array(img)
        img_np = cv2.equalizeHist(img_np)  # Improves visual similarity

        return img_np

    except Exception as e:
        print(f"⚠️ Error fetching screenshot for {url}: {e}")
        return None


# ✅ Function to Calculate Jaccard Similarity for Text Comparison
def jaccard_similarity(text1, text2):
    set1, set2 = set(text1.split()), set(text2.split())
    return len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0


# ✅ Function to Compare Images using Improved SSIM
def compare_images(img1, img2):
    if img1 is None or img2 is None:
        return 0

    # Resize images to a standard size before comparison
    img1 = cv2.resize(img1, (512, 512))
    img2 = cv2.resize(img2, (512, 512))

    # Calculate Structural Similarity Index (SSIM)
    return ssim(img1, img2)


# ✅ Function to Fetch WHOIS Information
def get_whois_info(url):
    try:
        domain = url.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]
        response = requests.get(
            f"https://www.whoisxmlapi.com/whoisserver/WhoisService?apiKey=447f1e746b2b4a8f96e58ee10aa3e53e&domainName={domain}&outputFormat=json"
        )

        if response.status_code == 200:
            whois_data = response.json()
            registrar = whois_data.get("WhoisRecord", {}).get("registrarName", "Unknown")
            creation_date = whois_data.get("WhoisRecord", {}).get("createdDate", None)
            nameservers = whois_data.get("WhoisRecord", {}).get("nameServers", {}).get("hostNames", [])

            if creation_date:
                creation_date = datetime.strptime(creation_date[:10], "%Y-%m-%d")
                domain_age = (datetime.now() - creation_date).days / 365
            else:
                domain_age = None

            return registrar, domain_age, nameservers
        else:
            return None, None, None

    except Exception as e:
        print(f"⚠️ WHOIS lookup failed: {e}")
        return None, None, None


# ✅ Function to Compare Text Content Similarity using TF-IDF
def text_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(vectors[0], vectors[1])[0][0]


# ✅ Function to Check for Cloned Websites with Improved Image Similarity
async def check_clone(site1, site2):
    if not site1.startswith("http"):
        site1 = "https://" + site1
    if not site2.startswith("http"):
        site2 = "https://" + site2

    # 🔹 Fetch HTML
    html1, html2 = await fetch_html(site1), await fetch_html(site2)
    if not html1 or not html2:
        return {"error": "Could not fetch HTML from one or both sites"}

    # 🔹 Compare HTML Structure
    soup1, soup2 = BeautifulSoup(html1, "html.parser"), BeautifulSoup(html2, "html.parser")
    html_similarity = jaccard_similarity(soup1.get_text(), soup2.get_text())

    # 🔹 Compare Screenshots (Improved)
    img1, img2 = await fetch_screenshot(site1), await fetch_screenshot(site2)
    img_similarity = compare_images(img1, img2)

    # 🔹 Compare WHOIS Data
    registrar1, age1, nameserver1 = get_whois_info(site1)
    registrar2, age2, nameserver2 = get_whois_info(site2)

    # 🔹 WHOIS Matching Logic
    if site1 == site2:
        whois_match = True
    elif None in (registrar1, registrar2, age1, age2, nameserver1, nameserver2):
        whois_match = False
    else:
        whois_match = (
            registrar1 == registrar2 and abs(age1 - age2) < 1 and set(nameserver1) == set(nameserver2)
        )

    # 🔹 Compare Text Content
    text_sim = text_similarity(soup1.get_text(), soup2.get_text())

    # ✅ Final Result with Enhanced Image Similarity
    result = {
        "html_similarity": round(html_similarity, 2),
        "image_similarity": round(img_similarity, 2),
        "whois_match": "True" if whois_match else "False",
        "text_similarity": round(text_sim, 2),
        "clone_detected": "True"
        if (
            html_similarity > 0.7
            and text_sim > 0.6
            or img_similarity > 0.85  # 🔹 Higher threshold for accuracy
            or whois_match
        )
        else "False",
    }
    return result