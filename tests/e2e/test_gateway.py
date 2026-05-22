import json
import httpx


def test_pipeline():
    url = "http://localhost:4005/v1/chat/completions"
    headers = {
        "X-API-Key": "sk-heyjude-dev",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": "My name is John Doe and I work at Microsoft. You can contact me at john.doe@microsoft.com.",
            }
        ],
    }
    print("Sending request to Hey Jude Gateway...")
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=30.0)
        print(f"Status Code: {response.status_code}")
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Failed to communicate with gateway: {e}")


if __name__ == "__main__":
    test_pipeline()
