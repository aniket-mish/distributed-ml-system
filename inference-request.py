import requests
import json

input_path = "cifar10-input.json"

with open(input_path) as json_file:
    data = json.load(json_file)

response = requests.post(
    url="http://localhost:8080/v1/models/tf-cifar10:predict",
    data=json.dumps(data),
    headers={"Host": "tf-cifar10.kubeflow.example.com"},
)
print(response.text)
