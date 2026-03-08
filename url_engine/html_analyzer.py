import requests
from bs4 import BeautifulSoup

def analyze_html(url):
    risk_score = 0

    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, timeout=5, headers=headers)

        if not response or not hasattr(response, "text"):
            return 0

        html = response.text.lower()

        soup = BeautifulSoup(html, "html.parser")

        forms = soup.find_all("form")
        for form in forms:
            if "password" in str(form):
                risk_score += 5

        hidden_fields = soup.find_all("input", {"type": "hidden"})
        if len(hidden_fields) > 3:
            risk_score += 3

        iframes = soup.find_all("iframe")
        if len(iframes) > 0:
            risk_score += 5

        suspicious_words = ["verify", "login", "update", "secure", "bank", "account"]
        for word in suspicious_words:
            if word in html:
                risk_score += 2

        for form in forms:
            action = form.get("action")
            if action and url not in action:
                risk_score += 5

    except Exception as e:
        print("HTML Analyzer Error:", e)
        return 0

    return min(risk_score, 25)