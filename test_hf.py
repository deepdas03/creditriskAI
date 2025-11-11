import requests, json
token = ""
url = "https://router.huggingface.co/hf-inference"
payload = {
  "model":"gpt2",
  "inputs":"Summarize: risk analysis test",
  "parameters": {"max_new_tokens": 50},
  "options": {"wait_for_model": True}
}
r = requests.post(url, headers={"Authorization":f"Bearer {token}"}, json=payload, timeout=60)
print("status:", r.status_code)
print(r.text[:2000])
