import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import JointMultipleKernelMaximumMeanDiscrepancy

def calculate_jmmd_loss_improved(encoder, src_imgs, tgt_imgs, jmmd_loss_fn, layers="layer 1 only", rescale_loss=True):
    """
    IMPROVED: Calculate JMMD loss with proper gradient flow by reusing intermediate features.
    This fixes the gradient flow issue by ensuring all computations share the same computational graph.
    """
    if layers is None:
        raise ValueError("No layers specified for JMMD loss calculation.")

    # Store intermediate features for both domains
    src_features = []
    tgt_features = []
    
    # Forward pass through base layers (shared computation)
    src_x = encoder.model.conv1(src_imgs)
    src_x = encoder.model.bn1(src_x)
    src_x = encoder.model.relu(src_x)
    src_x = encoder.model.maxpool(src_x)
    
    tgt_x = encoder.model.conv1(tgt_imgs)
    tgt_x = encoder.model.bn1(tgt_x)
    tgt_x = encoder.model.relu(tgt_x)
    tgt_x = encoder.model.maxpool(tgt_x)
    
    # Layer 1 - CRITICAL: Reuse these computations if we need final features later
    if "layer 1" in layers or "layer 2" in layers or "layer 3" in layers:
        src_x = encoder.model.layer1(src_x)
        tgt_x = encoder.model.layer1(tgt_x)
        
        if "layer 1" in layers:
            src_feat1 = torch.flatten(encoder.model.avgpool(src_x), 1)
            tgt_feat1 = torch.flatten(encoder.model.avgpool(tgt_x), 1)
            src_features.append(src_feat1)
            tgt_features.append(tgt_feat1)
    
    # Layer 2 - continues from layer 1 computation
    if "layer 2" in layers or "layer 3" in layers:
        src_x = encoder.model.layer2(src_x)
        tgt_x = encoder.model.layer2(tgt_x)
        
        if "layer 2" in layers:
            src_feat2 = torch.flatten(encoder.model.avgpool(src_x), 1)
            tgt_feat2 = torch.flatten(encoder.model.avgpool(tgt_x), 1)
            src_features.append(src_feat2)
            tgt_features.append(tgt_feat2)
    
    # Layer 3
    if "layer 3" in layers:
        src_x = encoder.model.layer3(src_x)
        tgt_x = encoder.model.layer3(tgt_x)
        src_feat3 = torch.flatten(encoder.model.avgpool(src_x), 1)
        tgt_feat3 = torch.flatten(encoder.model.avgpool(tgt_x), 1)
        src_features.append(src_feat3)
        tgt_features.append(tgt_feat3)

    # Compute JMMD loss with proper gradient flow
    loss_value = jmmd_loss_fn(tuple(src_features), tuple(tgt_features))

    if rescale_loss:
        num_kernels = len(jmmd_loss_fn.kernels[0])
        num_layers = len(src_features)
        return loss_value / (num_kernels * num_layers)
    else:
        return loss_value


def calculate_jmmd_with_feature_reuse(encoder, src_imgs, tgt_imgs, jmmd_loss_fn, 
                                    layers="layer 1 only", rescale_loss=True, 
                                    return_final_features=False):
    """
    OPTIMAL: Calculate JMMD loss AND return final features to avoid double computation.
    Use this when you need both JMMD loss and final features for other losses.
    """
    if layers is None:
        raise ValueError("No layers specified for JMMD loss calculation.")

    src_features = []
    tgt_features = []
    
    # Forward pass through base layers
    src_x = encoder.model.conv1(src_imgs)
    src_x = encoder.model.bn1(src_x)
    src_x = encoder.model.relu(src_x)
    src_x = encoder.model.maxpool(src_x)
    
    tgt_x = encoder.model.conv1(tgt_imgs)
    tgt_x = encoder.model.bn1(tgt_x)
    tgt_x = encoder.model.relu(tgt_x)
    tgt_x = encoder.model.maxpool(tgt_x)
    
    # Layer 1
    src_x = encoder.model.layer1(src_x)
    tgt_x = encoder.model.layer1(tgt_x)
    
    if "layer 1" in layers:
        src_feat1 = torch.flatten(encoder.model.avgpool(src_x), 1)
        tgt_feat1 = torch.flatten(encoder.model.avgpool(tgt_x), 1)
        src_features.append(src_feat1)
        tgt_features.append(tgt_feat1)
    
    # Continue through remaining layers
    src_x = encoder.model.layer2(src_x)
    tgt_x = encoder.model.layer2(tgt_x)
    
    if "layer 2" in layers:
        src_feat2 = torch.flatten(encoder.model.avgpool(src_x), 1)
        tgt_feat2 = torch.flatten(encoder.model.avgpool(tgt_x), 1)
        src_features.append(src_feat2)
        tgt_features.append(tgt_feat2)
    
    src_x = encoder.model.layer3(src_x)
    tgt_x = encoder.model.layer3(tgt_x)
    
    if "layer 3" in layers:
        src_feat3 = torch.flatten(encoder.model.avgpool(src_x), 1)
        tgt_feat3 = torch.flatten(encoder.model.avgpool(tgt_x), 1)
        src_features.append(src_feat3)
        tgt_features.append(tgt_feat3)
    
    # Final layer for complete features if requested
    if return_final_features:
        src_x = encoder.model.layer4(src_x)
        tgt_x = encoder.model.layer4(tgt_x)
        
        # Get final features (same as encoder.features() but reusing computation)
        src_final = torch.flatten(encoder.model.avgpool(src_x), 1)
        tgt_final = torch.flatten(encoder.model.avgpool(tgt_x), 1)

    # Compute JMMD loss
    jmmd_loss = jmmd_loss_fn(tuple(src_features), tuple(tgt_features))

    if rescale_loss:
        num_kernels = len(jmmd_loss_fn.kernels[0])
        num_layers = len(src_features)
        jmmd_loss = jmmd_loss / (num_kernels * num_layers)

    if return_final_features:
        return jmmd_loss, src_final, tgt_final
    else:
        return jmmd_loss


# Hook-based approach (alternative solution)
class IntermediateFeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = []
    
    def register_hooks(self, layer_names):
        """Register forward hooks to capture intermediate features"""
        def get_activation(name):
            def hook(model, input, output):
                # Apply global average pooling and flatten
                pooled = self.model.model.avgpool(output)
                self.features[name] = torch.flatten(pooled, 1)
            return hook
        
        for name in layer_names:
            layer = getattr(self.model.model, name)
            hook = layer.register_forward_hook(get_activation(name))
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """Clean up hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def extract_features(self, imgs):
        """Extract features using hooks"""
        self.features = {}
        _ = self.model.features(imgs)  # Trigger forward pass
        return self.features

def calculate_jmmd_with_hooks(encoder, src_imgs, tgt_imgs, jmmd_loss_fn, 
                            layers="layer 1 only", rescale_loss=True):
    """
    Hook-based approach - cleaner but slightly more complex
    """
    # Map layer strings to actual layer names
    layer_mapping = {
        "layer 1": "layer1",
        "layer 2": "layer2", 
        "layer 3": "layer3"
    }
    
    # Extract relevant layers
    requested_layers = []
    for layer_name in layer_mapping:
        if layer_name in layers:
            requested_layers.append(layer_mapping[layer_name])
    
    # Set up feature extractors
    src_extractor = IntermediateFeatureExtractor(encoder)
    tgt_extractor = IntermediateFeatureExtractor(encoder)
    
    src_extractor.register_hooks(requested_layers)
    tgt_extractor.register_hooks(requested_layers)
    
    try:
        # Extract features (triggers forward pass with hooks)
        src_extractor.extract_features(src_imgs)
        tgt_extractor.extract_features(tgt_imgs)
        
        # Organize features for JMMD
        src_features = [src_extractor.features[layer] for layer in requested_layers]
        tgt_features = [tgt_extractor.features[layer] for layer in requested_layers]
        
        # Compute JMMD loss
        jmmd_loss = jmmd_loss_fn(tuple(src_features), tuple(tgt_features))
        
        if rescale_loss:
            num_kernels = len(jmmd_loss_fn.kernels[0])
            num_layers = len(src_features)
            jmmd_loss = jmmd_loss / (num_kernels * num_layers)
            
        return jmmd_loss
        
    finally:
        # Always clean up hooks
        src_extractor.remove_hooks()
        tgt_extractor.remove_hooks()