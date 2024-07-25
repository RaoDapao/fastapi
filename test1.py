import requests

def test_healthcheck():
    url = "http://192.168.20.180:9504/healthcheck"  # 请根据实际情况调整IP地址和端口
    response = requests.get(url)
    assert response.status_code == 200
    assert response.json() == {"status": "running"}

if __name__ == "__main__":
    test_healthcheck()
    print("Test passed.")
