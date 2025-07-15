import requests
TOKEN = "7738302353:AAGdFjWI6Wg6ye8eFHnvH7N6zRKarlHUZPY"
text = f'你好'
session = requests.Session()
session.proxies = {
    "http": "http://127.0.0.1:10808",
    "https": "http://127.0.0.1:10808"
}

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
params = {"chat_id": 5436165313, "text": text}
resp = session.get(url, params=params, timeout=10)
print(resp.json())
