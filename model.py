import torch
import torch.nn as nn
from transformers import AutoModel

class MalariaDINOClassifier(nn.Module):
    def __init__(self, model_name="facebook/dinov2-small"):
        super(MalariaDINOClassifier, self).__init__()
        print(f"Loading Foundation Model: {model_name}...")
        self.backbone = AutoModel.from_pretrained(model_name)
        for param in self.backbone.parameters():
            param.requires_grad = False
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        outputs = self.backbone(x)
        features = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(features)
        return logits
