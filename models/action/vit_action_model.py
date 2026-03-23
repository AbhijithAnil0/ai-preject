import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from collections import deque


class ActionViT(nn.Module):

    def __init__(self, checkpoint=None, pretrained=True, num_classes=3, window_size=5):

        super().__init__()
        
        # Buffer to store probabilities of the last N frames to smooth out predictions
        self.prob_buffer = deque(maxlen=window_size)

        self.labels = [
            "run",
            "stand",
            "walk"
        ]

        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained
        )

        in_features = self.model.head.in_features

        # classifier now matches number of labels
        self.model.head = nn.Linear(in_features, num_classes)

        if checkpoint is not None:
            checkpoint_data = torch.load(checkpoint, map_location="cpu")
            if "state_dict" in checkpoint_data:
                self.model.load_state_dict(checkpoint_data["state_dict"], strict=False)
            else:
                self.model.load_state_dict(checkpoint_data, strict=False)
            print(f"Loaded ActionViT checkpoint: {checkpoint}")


    def forward(self, x):

        return self.model(x)


    def predict(self, tensor, conf_thresh=0.7, use_smoothing=True):

        with torch.no_grad():

            out = self.model(tensor)
            
            # If passing a batch of multiple frames (e.g. clip of shape [N, 3, 224, 224])
            if out.shape[0] > 1:
                probs = F.softmax(out, dim=1).mean(dim=0)
            else:
                probs = F.softmax(out, dim=1)[0]
                
            # Keep a rolling average in real-time if smoothing is enabled
            if use_smoothing:
                self.prob_buffer.append(probs)
                probs = torch.stack(list(self.prob_buffer)).mean(dim=0)
                
            conf, idx = torch.max(probs, dim=0)

            if conf.item() < conf_thresh:
                return "unknown"

            if idx.item() >= len(self.labels):
                return "unknown"

            return self.labels[idx.item()]
    
    def decode(self, outputs):
        classes = ['run', 'stand', 'walk']  # 👈 hardcode
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        return classes[pred]