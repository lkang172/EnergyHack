import ast

# Step 1: Parse the source code
source_code = """
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, tqdm


class model(nn.Module):
    def __init__(self, lr=0.0001, lrDecay=0.95, **kwargs):
        super(model, self).__init__()

        self.visualModel = None
        self.audioModel = None
        self.fusionModel = None
        self.fcModel = None

        self.loss_fn = nn.CrossEntropyLoss()

    def createVisualModel(self):
        self.visualModel = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.Flatten()
        )

    def createAudioModel(self):
        self.audioModel = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.Flatten()
        )

    def createFusionModel(self):
        pass

    def createFCModel(self):
        self.fcModel = nn.Sequential(
            nn.Linear(29824, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
"""

tree = ast.parse(source_code)

class ArgVisitor(ast.NodeVisitor):
    def visit_ClassDef(self, node):
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            print(f"    Method Call: {node.func.attr}")
        self.generic_visit(node) 


# Step 3: Analyze the AST
visitor = ArgVisitor()
visitor.visit(tree)

layerToInt = {"Linear": 0, "Conv1d": 1, "Conv2d": 2, "Conv3d": 3, "MaxPool1d": 2, "MaxPool2d": 3, "MaxPool3d": 4, "AvgPool1d": 2, "AvgPool2d": 3, "AvgPool3d": 4, 
          "RNN": 5, "LSTM" : 6, "GRU": 7, "ReLU": 8, "Sigmoid" : 9, "Tanh" : 10, "BatchNorm1d": 11, "BatchNorm2d": 12, "LayerNorm": 13,
          "Dropout": 14, "Dropout2d": 15, "Dropout3d": 16, "flatten": 17, "Embedding": 18, "CrossEntropyLoss": 19, "MSELoss": 20, "SmoothL1Loss": 21, 
          "NLLLoss": 22, "HingeEmbeddingLoss": 23, "KLDivLoss": 24, "BCELoss": 25}
