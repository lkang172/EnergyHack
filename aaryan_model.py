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

        self.createVisualModel()
        self.createAudioModel()
        self.createFusionModel()
        self.createFCModel()

        self.visualModel = self.visualModel.cuda()
        self.audioModel = self.audioModel.cuda()
        self.fcModel = self.fcModel.cuda()

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=lrDecay)

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

    def train_network(self, loader, epoch, **kwargs):

        self.train()
        self.scheduler.step(epoch - 1)
        lr = self.optim.param_groups[0]['lr']
        index, top1, loss = 0, 0, 0
        for num, (audioFeatures, visualFeatures, labels) in enumerate(loader, start=1):
            self.zero_grad()

            audioFeatures = torch.unsqueeze(audioFeatures, dim=1)

            audioFeatures = audioFeatures.cuda()
            visualFeatures = visualFeatures.cuda()
            labels = labels.squeeze().cuda()

            audioEmbed = self.audioModel(audioFeatures)
            visualEmbed = self.visualModel(visualFeatures)

            avfusion = torch.cat((audioEmbed, visualEmbed), dim=1)

            fcOutput = self.fcModel(avfusion)

            nloss = self.loss_fn(fcOutput, labels)

            self.optim.zero_grad()
            nloss.backward()
            self.optim.step()

            loss += nloss.detach().cpu().numpy()

            top1 += (fcOutput.argmax(1) == labels).type(torch.float).sum().item()
            index += len(labels)
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                             " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / loader.__len__())) + \
                             " Loss: %.5f, ACC: %2.2f%% \\r" % (loss / (num), 100 * (top1 / index)))
            sys.stderr.flush()
        sys.stdout.write("\\n")

        return loss / num, lr

    def evaluate_network(self, loader, **kwargs):
        self.eval()
        predScores = []

        loss, top1, index, numBatches = 0, 0, 0, 0

        for audioFeatures, visualFeatures, labels in tqdm.tqdm(loader):
            audioFeatures = torch.unsqueeze(audioFeatures, dim=1)
            audioFeatures = audioFeatures.cuda()
            visualFeatures = visualFeatures.cuda()
            labels = labels.squeeze().cuda()

            with torch.no_grad():
                audioEmbed = self.audioModel(audioFeatures)
                visualEmbed = self.visualModel(visualFeatures)

                avfusion = torch.cat((audioEmbed, visualEmbed), dim=1)

                fcOutput = self.fcModel(avfusion)

                nloss = self.loss_fn(fcOutput, labels)

                loss += nloss.detach().cpu().numpy()
                top1 += (fcOutput.argmax(1) == labels).type(torch.float).sum().item()
                index += len(labels)
                numBatches += 1

        print('eval loss ', loss / numBatches)
        print('eval accuracy ', top1 / index)

        return top1 / index

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model." % origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s" % (
                origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)
"""

tree = ast.parse(source_code)

# Step 2: Extract class definitions
class ModelAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.classes = []

    def visit_ClassDef(self, node):
        class_info = {"name": node.name, "methods": [], "inherits": []}

        # Extract base classes (e.g., nn.Module)
        for base in node.bases:
            if isinstance(base, ast.Name):
                class_info["inherits"].append(base.id)
            elif isinstance(base, ast.Attribute):
                class_info["inherits"].append(f"{base.value.id}.{base.attr}")

        # Extract methods within the class
        for body_item in node.body:
            if isinstance(body_item, ast.FunctionDef):
                class_info["methods"].append(body_item.name)

        self.classes.append(class_info)
        self.generic_visit(node)

# Step 3: Analyze the AST
analyzer = ModelAnalyzer()
analyzer.visit(tree)

# Step 4: Print extracted class information
for cls in analyzer.classes:
    print(f"Class: {cls['name']}")
    print(f"Inherits from: {', '.join(cls['inherits'])}")
    print(f"Methods: {', '.join(cls['methods'])}")
    print()
