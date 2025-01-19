import ast

source_code = """
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
    

def parse_function(source_code):
    layerToInt = {"Linear": 0, "Conv1d": 1, "Conv2d": 2, "Conv3d": 3, "MaxPool1d": 4, "MaxPool2d": 5, "MaxPool3d": 6, "AvgPool1d": 7, "AvgPool2d": 8, "AvgPool3d": 9, 
            "RNN": 10, "LSTM" : 11, "GRU": 12, "ReLU": 13, "Sigmoid" : 14, "Tanh" : 15, "BatchNorm1d": 16, "BatchNorm2d": 17, "LayerNorm": 18,
            "Dropout": 19, "Dropout2d": 20, "Dropout3d": 21, "flatten": 22, "Embedding": 23, "CrossEntropyLoss": 24, "MSELoss": 25, "SmoothL1Loss": 26, 
            "NLLLoss": 27, "HingeEmbeddingLoss": 28, "KLDivLoss": 29, "BCELoss": 30}

    intToParams = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [], 16: [], 17: [], 
                18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: [], 25: [], 26: [], 27: [], 28: [], 29: [], 30: []}

    tree = ast.parse(source_code)

    class ArgVisitor(ast.NodeVisitor):
        def visit_ClassDef(self, node):
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute):
                print(f"    Method Call: {node.func.attr}")
                layer_name = node.func.attr
                if layer_name in layerToInt:
                    sub_array = []
                    for arg in node.args:
                        sub_array.append(arg.value)
                    for keyword in node.keywords:
                        sub_array.append(keyword.value.value)
                    intToParams[layerToInt[layer_name]].append(sub_array)
            self.generic_visit(node) 


    visitor = ArgVisitor()
    visitor.visit(tree)

    print(intToParams)
