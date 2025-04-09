import requests

url = "http://localhost:5000/predict"
files = {"file": open("./New Test/0bf809ec-841e-4668-b875-adeb73d60de0___FAM_B.Msls 4057.JPG", "rb")}  # replace image.jpg with your image file path
response = requests.post(url, files=files)
print(response.json())
