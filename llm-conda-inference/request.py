import requests

url = "http://localhost:8000/generate"
payload = {"text": "Who is Satya Nadella?"}
response = requests.post(url, json=payload)

if response.status_code == 200:
    data = response.json()
    generated_text = data["generated_text"]
    print("Generated text:", generated_text)
else:
    print("Request failed with status code:", response.status_code)
