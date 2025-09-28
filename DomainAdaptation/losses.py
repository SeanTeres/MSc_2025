import sys
import os
import gc
from typing import List
# Add mbod-data-processor to the Python path
sys.path.append(os.path.abspath("../mbod-data-processor"))

from datasets.hdf_dataset import HDF5Dataset, HDF5Dataset2
from utils import LABEL_SCHEMES, load_config
from data_splits import stratify, get_label_scheme_supports
import numpy as np
import matplotlib.pyplot as plt
import h5py
from datasets.dataloader import get_dataloaders, get_dataloaders_with_files
import torchxrayvision as xrv
import torch
import torch.nn.functional as F
import torch.nn as nn
import wandb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, cohen_kappa_score, roc_auc_score
import seaborn as sns
from sklearn.calibration import calibration_curve
import io
import torchvision.transforms as transforms
import os
import math
import random
from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning import losses, miners

from typing import Sequence, Optional

import torch
from torch import device, nn, Tensor
import torch.nn.functional as F
from itertools import zip_longest, cycle
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

from da_utils import visualize_tsne_with_kaggle_tb, analyze_mined_triplets, reinitialize_weights, ForeverDataIterator
sys.path.append(os.path.abspath("../classification"))

from clf_manager import XRVBasedClassifier
from clf_metrics import compute_binary_clf_metrics



def calculate_jmmd_loss(encoder, mbod_imgs, tbnet_imgs, jmmd_loss_fn, layers=None, rescale_loss=True):
    """
    Calculate JMMD loss ensuring proper gradient flow through the network.
    """
    if layers is None:
        raise ValueError("No layers specified for JMMD loss calculation.")

    features_source = []
    features_target = []
    
    # Forward pass through base layers
    x = encoder.model.conv1(mbod_imgs)
    x = encoder.model.bn1(x)
    x = encoder.model.relu(x)
    x = encoder.model.maxpool(x)
    
    y = encoder.model.conv1(tbnet_imgs)
    y = encoder.model.bn1(y)
    y = encoder.model.relu(y)
    y = encoder.model.maxpool(y)
    
    # Layer 1
    if "layer 1" in layers or "layer 2" in layers or "layer 3" in layers:
        x = encoder.model.layer1(x)  # Maintain computational graph
        y = encoder.model.layer1(y)
        
        if "layer 1" in layers:
            features_source.append(torch.flatten(encoder.model.avgpool(x), 1))
            features_target.append(torch.flatten(encoder.model.avgpool(y), 1))
    
    # Layer 2 - uses output from layer 1
    if "layer 2" in layers:
        x = encoder.model.layer2(x)  # Connected to previous layer's computation
        y = encoder.model.layer2(y)
        features_source.append(torch.flatten(encoder.model.avgpool(x), 1))
        features_target.append(torch.flatten(encoder.model.avgpool(y), 1))
    
    if "layer 3" in layers:
        x = encoder.model.layer3(x)
        y = encoder.model.layer3(y)
        features_source.append(torch.flatten(encoder.model.avgpool(x), 1))
        features_target.append(torch.flatten(encoder.model.avgpool(y), 1))

    # Compute JMMD loss while maintaining gradient flow
    loss_value = jmmd_loss_fn(tuple(features_source), tuple(features_target))

    if rescale_loss:
        num_kernels = len(jmmd_loss_fn.kernels[0])  # Number of kernels in first layer tuple
        num_layers = len(features_source)
        return loss_value / (num_kernels * num_layers)
    else:
        return loss_value


class JointMultipleKernelMaximumMeanDiscrepancy(nn.Module):
    """The Joint Multiple Kernel Maximum Mean Discrepancy (JMMD) used in
    `Deep Transfer Learning with Joint Adaptation Networks (ICML 2017) <https://arxiv.org/abs/1605.06636>`_

    Given source domain :math:`\mathcal{D}_s` of :math:`n_s` labeled points and target domain :math:`\mathcal{D}_t`
    of :math:`n_t` unlabeled points drawn i.i.d. from P and Q respectively, the deep networks will generate
    activations in layers :math:`\mathcal{L}` as :math:`\{(z_i^{s1}, ..., z_i^{s|\mathcal{L}|})\}_{i=1}^{n_s}` and
    :math:`\{(z_i^{t1}, ..., z_i^{t|\mathcal{L}|})\}_{i=1}^{n_t}`. The empirical estimate of
    :math:`\hat{D}_{\mathcal{L}}(P, Q)` is computed as the squared distance between the empirical kernel mean
    embeddings as

    .. math::
        \hat{D}_{\mathcal{L}}(P, Q) &=
        \dfrac{1}{n_s^2} \sum_{i=1}^{n_s}\sum_{j=1}^{n_s} \prod_{l\in\mathcal{L}} k^l(z_i^{sl}, z_j^{sl}) \\
        &+ \dfrac{1}{n_t^2} \sum_{i=1}^{n_t}\sum_{j=1}^{n_t} \prod_{l\in\mathcal{L}} k^l(z_i^{tl}, z_j^{tl}) \\
        &- \dfrac{2}{n_s n_t} \sum_{i=1}^{n_s}\sum_{j=1}^{n_t} \prod_{l\in\mathcal{L}} k^l(z_i^{sl}, z_j^{tl}). \\

    Args:
        kernels (tuple(tuple(torch.nn.Module))): kernel functions, where `kernels[r]` corresponds to kernel :math:`k^{\mathcal{L}[r]}`.
        linear (bool): whether use the linear version of JAN. Default: False
        thetas (list(Theta): use adversarial version JAN if not None. Default: None

    Inputs:
        - z_s (tuple(tensor)): multiple layers' activations from the source domain, :math:`z^s`
        - z_t (tuple(tensor)): multiple layers' activations from the target domain, :math:`z^t`

    Shape:
        - :math:`z^{sl}` and :math:`z^{tl}`: :math:`(minibatch, *)`  where * means any dimension
        - Outputs: scalar

    .. note::
        Activations :math:`z^{sl}` and :math:`z^{tl}` must have the same shape.

    .. note::
        The kernel values will add up when there are multiple kernels for a certain layer.

    Examples::

        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> layer1_kernels = (GaussianKernel(alpha=0.5), GaussianKernel(1.), GaussianKernel(2.))
        >>> layer2_kernels = (GaussianKernel(1.), )
        >>> loss = JointMultipleKernelMaximumMeanDiscrepancy((layer1_kernels, layer2_kernels))
        >>> # layer1 features from source domain and target domain
        >>> z1_s, z1_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> # layer2 features from source domain and target domain
        >>> z2_s, z2_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> output = loss((z1_s, z2_s), (z1_t, z2_t))
    """

    def __init__(self, kernels: Sequence[Sequence[nn.Module]], linear: Optional[bool] = False, thetas: Sequence[nn.Module] = None):
        super(JointMultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear
        if thetas:
            self.thetas = thetas
        else:
            self.thetas = [nn.Identity() for _ in kernels]

    def _update_index_matrix(self, batch_size: int, index_matrix: Optional[torch.Tensor] = None,
                            linear: Optional[bool] = True) -> torch.Tensor:
        r"""
        Update the `index_matrix` which convert `kernel_matrix` to loss.
        If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
        Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
        """
        if index_matrix is None or index_matrix.size(0) != batch_size * 2:
            index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
            if linear:
                for i in range(batch_size):
                    s1, s2 = i, (i + 1) % batch_size
                    t1, t2 = s1 + batch_size, s2 + batch_size
                    index_matrix[s1, s2] = 1. / float(batch_size)
                    index_matrix[t1, t2] = 1. / float(batch_size)
                    index_matrix[s1, t2] = -1. / float(batch_size)
                    index_matrix[s2, t1] = -1. / float(batch_size)
            else:
                for i in range(batch_size):
                    for j in range(batch_size):
                        if i != j:
                            index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                            index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
                for i in range(batch_size):
                    for j in range(batch_size):
                        index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                        index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
        return index_matrix


    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        batch_size = int(z_s[0].size(0))
        self.index_matrix = self._update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s[0].device)   # Changed from original library

        kernel_matrix = torch.ones_like(self.index_matrix)
        for layer_z_s, layer_z_t, layer_kernels, theta in zip(z_s, z_t, self.kernels, self.thetas):
            layer_features = torch.cat([layer_z_s, layer_z_t], dim=0)
            layer_features = theta(layer_features)
            kernel_matrix *= sum(
                [kernel(layer_features) for kernel in layer_kernels])  # Add up the matrix of each kernel

        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)
        return loss
    

class GaussianKernel(nn.Module):
    r"""Gaussian Kernel Matrix

    Gaussian Kernel k is defined by

    .. math::
        k(x_1, x_2) = \exp \left( - \dfrac{\| x_1 - x_2 \|^2}{2\sigma^2} \right)

    where :math:`x_1, x_2 \in R^d` are 1-d tensors.

    Gaussian Kernel Matrix K is defined on input group :math:`X=(x_1, x_2, ..., x_m),`

    .. math::
        K(X)_{i,j} = k(x_i, x_j)

    Also by default, during training this layer keeps running estimates of the
    mean of L2 distances, which are then used to set hyperparameter  :math:`\sigma`.
    Mathematically, the estimation is :math:`\sigma^2 = \dfrac{\alpha}{n^2}\sum_{i,j} \| x_i - x_j \|^2`.
    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and use a fixed :math:`\sigma` instead.

    Args:
        sigma (float, optional): bandwidth :math:`\sigma`. Default: None
        track_running_stats (bool, optional): If ``True``, this module tracks the running mean of :math:`\sigma^2`.
          Otherwise, it won't track such statistics and always uses fix :math:`\sigma^2`. Default: ``True``
        alpha (float, optional): :math:`\alpha` which decides the magnitude of :math:`\sigma^2` when track_running_stats is set to ``True``

    Inputs:
        - X (tensor): input group :math:`X`

    Shape:
        - Inputs: :math:`(minibatch, F)` where F means the dimension of input features.
        - Outputs: :math:`(minibatch, minibatch)`
    """

    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))
    
class CorrelationAlignmentLoss(nn.Module):
    r"""The `Correlation Alignment Loss` in
    `Deep CORAL: Correlation Alignment for Deep Domain Adaptation (ECCV 2016) <https://arxiv.org/pdf/1607.01719.pdf>`_.

    Given source features :math:`f_S` and target features :math:`f_T`, the covariance matrices are given by

    .. math::
        C_S = \frac{1}{n_S-1}(f_S^Tf_S-\frac{1}{n_S}(\textbf{1}^Tf_S)^T(\textbf{1}^Tf_S))
    .. math::
        C_T = \frac{1}{n_T-1}(f_T^Tf_T-\frac{1}{n_T}(\textbf{1}^Tf_T)^T(\textbf{1}^Tf_T))

    where :math:`\textbf{1}` denotes a column vector with all elements equal to 1, :math:`n_S, n_T` denotes number of
    source and target samples, respectively. We use :math:`d` to denote feature dimension, use
    :math:`{\Vert\cdot\Vert}^2_F` to denote the squared matrix `Frobenius norm`. The correlation alignment loss is
    given by

    .. math::
        l_{CORAL} = \frac{1}{4d^2}\Vert C_S-C_T \Vert^2_F

    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - f_s, f_t: :math:`(N, d)` where d means the dimension of input features, :math:`N=n_S=n_T` is mini-batch size.
        - Outputs: scalar.
    """

    def __init__(self):
        super(CorrelationAlignmentLoss, self).__init__()

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        mean_s = f_s.mean(0, keepdim=True)
        mean_t = f_t.mean(0, keepdim=True)
        cent_s = f_s - mean_s
        cent_t = f_t - mean_t
        cov_s = torch.mm(cent_s.t(), cent_s) / (len(f_s) - 1)
        cov_t = torch.mm(cent_t.t(), cent_t) / (len(f_t) - 1)

        mean_diff = (mean_s - mean_t).pow(2).mean()
        cov_diff = (cov_s - cov_t).pow(2).mean()

        return mean_diff + cov_diff
    


class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    r"""The Multiple Kernel Maximum Mean Discrepancy (MK-MMD) used in
    `Learning Transferable Features with Deep Adaptation Networks (ICML 2015) <https://arxiv.org/pdf/1502.02791>`_

    Given source domain :math:`\mathcal{D}_s` of :math:`n_s` labeled points and target domain :math:`\mathcal{D}_t`
    of :math:`n_t` unlabeled points drawn i.i.d. from P and Q respectively, the deep networks will generate
    activations as :math:`\{z_i^s\}_{i=1}^{n_s}` and :math:`\{z_i^t\}_{i=1}^{n_t}`.
    The MK-MMD :math:`D_k (P, Q)` between probability distributions P and Q is defined as

    .. math::
        D_k(P, Q) \triangleq \| E_p [\phi(z^s)] - E_q [\phi(z^t)] \|^2_{\mathcal{H}_k},

    :math:`k` is a kernel function in the function space

    .. math::
        \mathcal{K} \triangleq \{ k=\sum_{u=1}^{m}\beta_{u} k_{u} \}

    where :math:`k_{u}` is a single kernel.

    Using kernel trick, MK-MMD can be computed as

    .. math::
        \hat{D}_k(P, Q) &=
        \dfrac{1}{n_s^2} \sum_{i=1}^{n_s}\sum_{j=1}^{n_s} k(z_i^{s}, z_j^{s})\\
        &+ \dfrac{1}{n_t^2} \sum_{i=1}^{n_t}\sum_{j=1}^{n_t} k(z_i^{t}, z_j^{t})\\
        &- \dfrac{2}{n_s n_t} \sum_{i=1}^{n_s}\sum_{j=1}^{n_t} k(z_i^{s}, z_j^{t}).\\

    Args:
        kernels (tuple(torch.nn.Module)): kernel functions.
        linear (bool): whether use the linear version of DAN. Default: False

    Inputs:
        - z_s (tensor): activations from the source domain, :math:`z^s`
        - z_t (tensor): activations from the target domain, :math:`z^t`

    Shape:
        - Inputs: :math:`(minibatch, *)`  where * means any dimension
        - Outputs: scalar

    .. note::
        Activations :math:`z^{s}` and :math:`z^{t}` must have the same shape.

    .. note::
        The kernel values will add up when there are multiple kernels.

    Examples::

        >>> from tllib.modules.kernels import GaussianKernel
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> kernels = (GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.), GaussianKernel(alpha=2.))
        >>> loss = MultipleKernelMaximumMeanDiscrepancy(kernels)
        >>> # features from source domain and target domain
        >>> z_s, z_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> output = loss(z_s, z_t)
    """

    def __init__(self, kernels: Sequence[nn.Module], linear: Optional[bool] = False):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)


        kernel_matrix = sum([kernel(features) for kernel in self.kernels])  # Add up the matrix of each kernel
        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)

        return loss


def _update_index_matrix(batch_size: int, index_matrix: Optional[torch.Tensor] = None,
                         linear: Optional[bool] = True) -> torch.Tensor:
    r"""
    Update the `index_matrix` which convert `kernel_matrix` to loss.
    If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
    Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
    """
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix





def safe_mean(losses):
    if isinstance(losses, Tensor) and losses.numel() > 0:
        return losses.mean()
    
    return torch.tensor(0.0, device=losses.device if isinstance(losses, torch.Tensor) else device)

if __name__ == "__main__":
    device = torch.device("cpu")
    print("*" * 50)
    print(f"Using device: {device}")
    print("*" * 50)
    defaults = load_config("defaults.yaml")
    wandb_api_key = defaults["WANDB_API_KEY"]["value"]
    random_seed = defaults["RANDOM_SEED"]["value"]

    augmentations_list = transforms.Compose([
    transforms.RandomRotation(degrees=10, expand=False, fill=0),
    # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=0)
    ])


    preprocess = transforms.Compose([
        # transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
        # transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def set_random_seeds(seed=42):
        """Set random seeds for reproducibility across all libraries"""
        import random
        import numpy as np
        import torch
        import os
        
        # Python built-in random
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        # PyTorch backends
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Environment variables for additional libraries
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # For sklearn (if used)
        try:
            from sklearn.utils import check_random_state
            check_random_state(seed)
        except ImportError:
            pass
        
        print(f"🌱 Random seeds set to {seed} for reproducibility")
        return seed

    set_random_seeds(seed=random_seed)  # Set a fixed seed for reproducibility

    try:
        cfg = load_config(defaults["DATA_CONFIG"]["path"])

        experiments_config = load_config('clda_pml.yaml')
        experiments = experiments_config["experiments"]

        for exp_cfg in experiments:
            
            exp_name = exp_cfg["name"]
            print(f"Running experiment: {exp_name}")

            if(exp_cfg.get("resolution", 999) == 512):
                print("Using model at 512x512")

                if exp_cfg.get("pretrained", False):
                    print("Using pretrained model")
                    model = xrv.models.ResNet(weights="resnet50-res512-all")
                else:
                    raise ValueError("Only pretrained model is currently supported. Please set pretrained to True in the config file.")
            else:
                raise ValueError("Only 512x512 currently supported. Please set resolution to 512 in the config file.")
            
            model.tb_clf = XRVBasedClassifier(input_dim=2048, num_classes=1, name="xrv_bin_tb")
            clf_loss_fn = nn.BCEWithLogitsLoss()

            if(exp_cfg["loss_components"]["bin_tb_clf"]):
                optimizer = torch.optim.Adam(
                    list(model.parameters()) + list(model.tb_clf.parameters()),
                    lr=exp_cfg["learning_rate"],
                    weight_decay=exp_cfg["learning_rate"]/10)
            else:
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=exp_cfg["learning_rate"],
                    weight_decay=exp_cfg["learning_rate"]/10)


            

            # Create an HDF5SilicosisDataset instance
            mbod_dataset_merged = HDF5Dataset(
                hdf5_path=cfg["merged_silicosis_output"]["hdf5_file"],
                labels_key="multiclass_stb",  # Main pathology labels, 'lab' for all labels
                images_key="images",
                augmentations=None,
                preprocess=preprocess
            )


            ilo_dataset = HDF5Dataset(
                hdf5_path=cfg["ilo_output"]["hdf5_file"],
                labels_key="profusion_score",  # Main pathology labels, 'lab' for all labels
                images_key="images",
                augmentations=None,
                preprocess=preprocess
            )

            kaggle_tb_dataset = HDF5Dataset(
                hdf5_path=cfg["kaggle_TB"]["outputpath"],
                labels_key="tuberculosis",  # Main pathology labels, 'lab' for all labels
                images_key="images",
                augmentations=None,
                preprocess=preprocess
            )

            combined_dataset_test = HDF5Dataset2(
                hdf5_path=cfg["combined_output"]["hdf5_file"],
                labels_key="multiclass_stb",  # Main pathology labels, 'lab' for all labels
                images_key="images",
                augmentations=None,
                preprocess=preprocess
            )


            if exp_cfg["distance_metric"] == "L2 squared":
                print("Using L2 squared distance metric")
                distance = LpDistance(p=2, power=2)
            elif exp_cfg["distance_metric"] == "Cosine":
                print("Using Cosine distance metric")
                distance = CosineSimilarity()
            elif exp_cfg["distance_metric"] == "L2":
                print("Using L2 distance metric")
                distance = LpDistance(p=2, power=1)
            else:
                raise ValueError(f"Unknown distance metric: {exp_cfg['distance_metric']}")
        

            train_loader_mbod, _, _ = get_dataloaders(
                hdf5_path=cfg["merged_silicosis_output"]["hdf5_file"],
                preprocess=preprocess,
                batch_size=8,
                labels_key="multiclass_stb",
                split_file="/home/sean/MSc_2025/mbod-data-processor/stratified_split_MBOD_mlabel_stb.json",
                augmentations=augmentations_list ,
                oversample=False
            )

            train_loader_tbnet, _, _ = get_dataloaders(
                hdf5_path=cfg["kaggle_TB"]["outputpath"],
                preprocess=preprocess,
                batch_size=8,
                labels_key="tuberculosis",
                split_file="stratified_split_tb_net.json",
                augmentations=augmentations_list,
                oversample=False
            )

            _, val_loader_mbod, test_loader_mbod = get_dataloaders(
                hdf5_path=cfg["merged_silicosis_output"]["hdf5_file"],
                preprocess=preprocess,
                batch_size=8,
                labels_key="multiclass_stb",
                split_file="/home/sean/MSc_2025/mbod-data-processor/stratified_split_MBOD_mlabel_stb.json",
                augmentations=None,
                oversample=False
            )
            _, val_loader_tbnet, test_loader_tbnet = get_dataloaders(
                hdf5_path=cfg["kaggle_TB"]["outputpath"],
                preprocess=preprocess,
                batch_size=8,
                labels_key="tuberculosis",
                split_file="/home/sean/MSc_2025/mbod-data-processor/stratified_split_tb_net.json",
                augmentations=None,
                oversample=False
                )
            
            source_iter = ForeverDataIterator(train_loader_tbnet)
            target_iter = ForeverDataIterator(train_loader_mbod)

            layer1_kernels = tuple(GaussianKernel(alpha=alpha).to(device) for alpha in [0.1, 0.25, 0.5, 1.0, 2.0])
            jmmd_loss_fn = JointMultipleKernelMaximumMeanDiscrepancy((layer1_kernels, )).to(device)

            mmd_kernels = (GaussianKernel(alpha=0.5).to(device), GaussianKernel(alpha=1.0).to(device), GaussianKernel(alpha=2.0).to(device))
            mmd_loss_fn = MultipleKernelMaximumMeanDiscrepancy(mmd_kernels).to(device)

            coral_fn = CorrelationAlignmentLoss()

            model = model.to(device)
            jmmd_loss_fn = jmmd_loss_fn.to(device)
            for i in range(1):
                print(f"Iteration {i+1}")
                mbod_batch = next(target_iter)
                tbnet_batch = next(source_iter)
                mbod_imgs = mbod_batch[0].to(device)
                tbnet_imgs = tbnet_batch[0].to(device)

                jmmd_l1 = calculate_jmmd_loss(model, mbod_imgs, tbnet_imgs, jmmd_loss_fn, layers="layer 1 only")
                # jmmd_l2 = calculate_jmmd_loss(model, mbod_imgs, tbnet_imgs, jmmd_loss_fn, layers="layer 2 only")
                # jmmd_l3 = calculate_jmmd_loss(model, mbod_imgs, tbnet_imgs, jmmd_loss_fn, layers="layer 3 only")

                # jmmd_l1_l2 = calculate_jmmd_loss(model, mbod_imgs, tbnet_imgs, jmmd_loss_fn, layers="layer 1 and layer 2")
                # jmmd_l1_l3 = calculate_jmmd_loss(model, mbod_imgs, tbnet_imgs, jmmd_loss_fn, layers="layer 1 and layer 3")
                # jmmd_l2_l3 = calculate_jmmd_loss(model, mbod_imgs, tbnet_imgs, jmmd_loss_fn, layers="layer 2 and layer 3")

                # jmmd_l1_l2_l3 = calculate_jmmd_loss(model, mbod_imgs, tbnet_imgs, jmmd_loss_fn, layers="layer 1 and layer 2 and layer 3")

                # print(f"JMMD Layer 2 Loss: {jmmd_l2.item()}")
                # print(f"JMMD Layer 3 Loss: {jmmd_l3.item()}")
                # print(f"JMMD Layer 1 and Layer 2 Loss: {jmmd_l1_l2.item()}")
                # print(f"Product: {jmmd_l1.item() * jmmd_l2.item()}")
                # print(f"JMMD Layer 1 and Layer 3 Loss: {jmmd_l1_l3.item()}")
                # print(f"JMMD Layer 2 and Layer 3 Loss: {jmmd_l2_l3.item()}")
                # print(f"JMMD Layer 1, Layer 2 and Layer 3 Loss: {jmmd_l1_l2_l3.item()}")

                mbod_feats = model.features(mbod_imgs)
                tbnet_feats = model.features(tbnet_imgs)

                mmd_loss = mmd_loss_fn(mbod_feats, tbnet_feats)
                coral_loss = coral_fn(mbod_feats, tbnet_feats)
                
                print(f"JMMD Layer 1 Loss: {jmmd_l1.item()}")
                print(f"MMD Loss: {mmd_loss.item()}")
                print(f"CORAL Loss: {coral_loss.item()}")



    except KeyError as e:
        print(f"Missing configuration: {e}")

