import os, requests

API_KEY = os.getenv("AIzaSyDl9ZLcFVhC956XjWpGQ74MamMsCxbwalA")
URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

def explain(objects):
    if not objects:
        return "I do not see any objects."

    prompt = f"Explain briefly for a blind person: {', '.join(objects)}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    r = requests.post(
        f"{URL}?key={API_KEY}",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=10
    )

    if r.status_code != 200:
        return "Gemini explanation unavailable."

    return r.json()["candidates"][0]["content"]["parts"][0]["text"]
