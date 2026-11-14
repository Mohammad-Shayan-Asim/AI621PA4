import os

import torch
from torch import nn
from torchvision.models import efficientnet_b0, efficientnet_b7, EfficientNet_B7_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class Teacher(nn.Module):

    def __init__(self, idx_1k):
        super().__init__()
        self.model = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.model.requires_grad_(False)

        self.register_buffer("idx_1k", torch.tensor(idx_1k))

    @torch.no_grad()
    def forward(self, x):
        return self.model(x)[:, self.idx_1k]


def make_student(num_classes=100):
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def feature_extractor(model, layer):
    # returns {"feat": output of `layer`, "logits": classifier output}; shares parameters with `model`
    ext = create_feature_extractor(model, {layer: "feat", "classifier.1": "logits"})
    ext.train(model.training)  # creation calls train() on the shared layers, so restore the model's mode
    return ext


def load_student(path="checkpoints/student_init.pt", seed=0):

    if not os.path.exists(path):
        torch.manual_seed(seed)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(make_student().state_dict(), path)

    model = make_student()
    model.load_state_dict(torch.load(path))
    return model
