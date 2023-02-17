import clip
from torch import nn


class CLIPWrapper(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model, self.preprocess = clip.load("RN50", device=self.device)
        self.model.eval()

    def forward(self, batch):
        image_features = self.model.encode_image(batch)
        normalized_features = nn.functional.normalize(image_features)
        return normalized_features
