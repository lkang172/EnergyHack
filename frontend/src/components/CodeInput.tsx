import React, { useState } from "react";
import { useEffect } from "react";
// import hljs from "highlight.js";
// import "highlight.js/styles/github-dark.css"; // Import a theme

import "../App.css";

const CodeInput = ({ onChange }) => {
  const [code, setCode] = useState(false);
  useEffect(() => {
    // hljs.highlightAll();
  }, []);
  return (
    <>
      {code && (
        <>
          <pre>
            <code className="language-python">
              {`import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters
input_size = 784  # 28x28 pixels
hidden_size = 128
num_classes = 10  # Digits 0-9
batch_size = 64
learning_rate = 0.01
num_epochs = 5

# Define a simple feedforward neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
`}
            </code>
          </pre>
        </>
      )}

      <textarea onChange={onChange}></textarea>
      <button onClick={() => setCode(true)}>Submit</button>
    </>
  );
};

export default CodeInput;
