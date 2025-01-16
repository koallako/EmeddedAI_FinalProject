import torch
import torch.nn as nn
from torch.nn.utils import prune
import copy

class AgePredictor(nn.Module):
    def __init__(self, is_student=False):
        super(AgePredictor, self).__init__()
        
        # Student 모델은 더 작은 architecture 사용
        first_channel = 16 if is_student else 32
        second_channel = 32 if is_student else 64
        third_channel = 64 if is_student else 128
        
        self.features = nn.Sequential(
            nn.Conv2d(3, first_channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(first_channel, second_channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(second_channel, third_channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(third_channel * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

def apply_pruning(model, amount=0.5):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    
    # Make pruning permanent
    for module, name in parameters_to_prune:
        prune.remove(module, name)
    
    return model