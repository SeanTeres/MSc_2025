import torch
from torch import nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchxrayvision as xrv

# Shamlessly copied from Josh

class AdaptivePoolingLayer(nn.Module):
    def __init__(self, output_sizes=[1, 2, 4]):
        super().__init__()
        self.output_sizes = output_sizes

    def forward(self, x):
        batch_size, channels = x.size(0), x.size(1)
        pooled_features = []

        for size in self.output_sizes:
            pooled = F.adaptive_avg_pool2d(x, (size, size))
            pooled_features.append(pooled.view(batch_size, channels * size * size))

        return torch.cat(pooled_features, dim=1)


class TxrvWrapper(nn.Module):
    # Torch X-ray Vision Wrapper compatible with our MCCV
    # Tested with resnet50-res512-all
    def __init__(self, num_classes, model_name):
        super().__init__()

        xrvmodel = xrv.models.ResNet(weights=model_name)
        xrvmodel = xrvmodel.model
        in_features = xrvmodel.fc.in_features
        self.model = nn.Sequential(
                xrvmodel.conv1,
                xrvmodel.bn1,
                xrvmodel.layer1,
                xrvmodel.layer2,
                xrvmodel.layer3,
                xrvmodel.layer4,
        )

        self.spp = AdaptivePoolingLayer(output_sizes=[1, 2, 4])

        spp_output_size = in_features * (1*1 + 2*2 + 4*4)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(spp_output_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        self.swin_with_adapter = None

    def initialize_weights(self, init_backbone=False):
        print("Randomly initialising weights")

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        if init_backbone:
            for module in self.model.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)

    # text_features is just ignored because this has to match the FMs
    def forward(self, images, text_features=None, return_image_features=False):
        features = self.model(images)

        pooled_features = self.spp(features)

        logits = self.classifier(pooled_features)

        if return_image_features:
            return logits, pooled_features

        return logits
    
    def features(self, images):  # Added for compatibility with tsne (SEAN)
        features = self.model(images)
        # Global average pooling (like original ResNet)
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        pooled = avgpool(features)
        return pooled
        

    def encode_text(self, text_tokens, normalize=False):
        return None

    def freeze_weights(self):
        return

    def print_parameter_count(self):
        def count_parameters(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        print("Parameter counts:")
        print(f"RESNET50 Backbone: {count_parameters(self.model):,}")
        print(f"Classifier: {count_parameters(self.classifier):,}")


def init_torchxrayvision_resnet_model(num_classes, randomly_initialise=False):
    model = TxrvWrapper(num_classes=num_classes, model_name='resnet50-res512-all')

    if randomly_initialise:
        model.initialize_weights(init_backbone=True)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    return model, preprocess