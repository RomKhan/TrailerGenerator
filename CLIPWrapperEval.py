import clip
from torch import nn

class CLIPWrapperEval(nn.Module):
  def __init__(self, device):
    super().__init__()
    self.device = device
    self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
    self.model.eval()

  def forward(self,batch):
    features = self.model.encode_image(batch)
    normalized_features = nn.functional.normalize(features)
    return normalized_features

  def forward_text(self,batch):
    features = self.model.encode_text(batch)
    normalized_features = nn.functional.normalize(features)
    return normalized_features