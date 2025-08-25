import requests
import json

URL = "http://127.0.0.1:5000/compare"  # change to your deployed URL if needed

def main():
    with open("sample.txt", "rb") as f:
        files = {"file": ("sample.txt", f, "text/plain")}
        data = {"audio_text": "I am going to college today."}
        resp = requests.post(URL, data=data, files=files, timeout=60)
    print("Status:", resp.status_code)
    print("Body:", resp.text)

if __name__ == "__main__":
    main()
