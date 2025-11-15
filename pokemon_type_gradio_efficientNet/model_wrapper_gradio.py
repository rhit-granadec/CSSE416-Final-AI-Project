import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np

# EfficientNetB3 classifier matching your training architecture
class EfficientNetB3Classifier(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        efficientnet = models.efficientnet_b3(weights='EfficientNet_B3_Weights.DEFAULT')
        
        # Use the feature extractor
        self.features = efficientnet.features
        
        # Add adaptive pooling to get fixed-size output
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier head matches your training code
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1536, 512),  # 1536 is EfficientNet-B3's feature dimension
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _strip_state_dict_prefix(state_dict, prefix):
    """Strip a common prefix from state_dict keys if present."""
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state[k[len(prefix):]] = v
        else:
            new_state[k] = v
    return new_state


class GradioPokemonPredictor:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() and 'cuda' in device else 'cpu')
        self.type_list = [
            "Bug", "Dark", "Dragon", "Electric", "Fairy", "Fighting",
            "Fire", "Flying", "Ghost", "Grass", "Ground", "Ice",
            "Normal", "Poison", "Psychic", "Rock", "Steel", "Water"
        ]

        # Transforms must match training pipeline
        # Note: Check your training code for the exact image size used
        # EfficientNetB3 typically uses 300x300, but adjust if different
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Update this to match your training size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create model and load checkpoint
        self.model = EfficientNetB3Classifier(len(self.type_list))
        self.model.to(self.device)
        self._load_checkpoint(model_path)
        self.model.eval()

    def _load_checkpoint(self, model_path):
        ckpt = torch.load(model_path, map_location=self.device)
        
        # Lightning checkpoint is typically a dict with key 'state_dict'
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        elif isinstance(ckpt, dict) and any(k.startswith('state_dict') for k in ckpt.keys()):
            sd = ckpt.get('state_dict') or ckpt
        else:
            sd = ckpt

        # Strip common prefixes that PyTorch Lightning adds
        sample_key = next(iter(sd.keys())) if sd else None
        if sample_key:
            if sample_key.startswith('model.'):
                sd = _strip_state_dict_prefix(sd, 'model.')
            elif sample_key.startswith('efficientnet.'):
                sd = _strip_state_dict_prefix(sd, 'efficientnet.')

        # Try loading with strict=False to be robust to small naming differences
        try:
            missing_keys, unexpected_keys = self.model.load_state_dict(sd, strict=False)
            if missing_keys:
                print(f"Warning: Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")
        except Exception as e:
            print(f"Error loading state_dict: {e}")
            # As a fallback, try to find inner 'state_dict' keys
            flattened = {}
            for k, v in sd.items():
                if k.startswith('state_dict.'):
                    flattened[k.replace('state_dict.', '')] = v
            if flattened:
                self.model.load_state_dict(flattened, strict=False)
            else:
                raise e

    def _preprocess_image(self, image: Image.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return self.transform(image).unsqueeze(0).to(self.device)

    def predict_bytes(self, image_bytes, threshold=0.5):
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return self.predict_pil(img, threshold=threshold)

    def predict_pil(self, image: Image.Image, threshold=0.5):
        x = self._preprocess_image(image)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        
        # Collect predicted types above threshold
        results = []
        for i, p in enumerate(probs):
            if p >= threshold:
                results.append((self.type_list[i], float(p)))
        
        # Sort by probability desc
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def predict_topk(self, image: Image.Image, k=3):
        x = self._preprocess_image(image)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        
        idx = np.argsort(-probs)[:k]
        return [(self.type_list[i], float(probs[i])) for i in idx]

    def predict_full(self, image: Image.Image):
        x = self._preprocess_image(image)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        
        idx = np.argsort(-probs)
        return [(self.type_list[i], float(probs[i])) for i in idx]