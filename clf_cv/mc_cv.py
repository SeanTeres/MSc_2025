# To do:
# Fix auc calculations
# Use Josh's classifier as well


from datetime import datetime
import glob
import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchxrayvision as xrv
import wandb
import numpy as np
import yaml
import torch.nn.functional as F
import sklearn.metrics as skmetrics
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import scipy.stats


from dataloader import get_dataloaders

# Add mbod-data-processor to the Python path
sys.path.append(os.path.abspath("../mbod-data-processor"))
from datasets.hdf_dataset import HDF5Dataset, HDF5Dataset2

# Add codev2 and DomainAdaptation to path
sys.path.append(os.path.abspath("../DomainAdaptation"))
from da_utils import reinitialize_weights, visualize_tsne_with_kaggle_tb

sys.path.append(os.path.abspath("../classification"))
from cross_validation import plot_combined_conf_mat, plot_tb_stratified_binary_cm
import metrics
from clf_manager import BinaryClassifier, MulticlassClassifier, XRVBasedClassifier

def mean_std_var_ci(metric_list):
    arr = np.array(metric_list)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    var = np.var(arr, ddof=1)
    n = len(arr)
    # 95% confidence interval for the mean (t-distribution)
    ci95 = scipy.stats.t.interval(
        0.95, n-1, loc=mean, scale=std/np.sqrt(n)
    ) if n > 1 else (mean, mean)
    return {
        "mean": mean,
        "std": std,
        "var": var,
        "ci95_low": ci95[0],
        "ci95_high": ci95[1]
    }

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

class MulticlassMonteCarloCV:
    def __init__(self, model, cfg, labels_key, num_classes, device, loss_fn, hdf5_path, batch_size, epochs, optimizer_type, learning_rate, weight_decay, use_oversampling, checkpoint_save_target, exp_name, split_file, train_set_name, test_set_name, clf_task_labels_key):
        self.model = model
        self.cfg = cfg
        self.labels_key = labels_key
        self.num_classes = num_classes
        self.device = device
        self.loss_fn = loss_fn
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_oversampling = use_oversampling
        self.checkpoint_save_target = checkpoint_save_target
        self.exp_name = exp_name
        self.split_file = split_file
        self.train_set_name = train_set_name
        self.clf_task_labels_key = clf_task_labels_key
        self.preprocess = transforms.Compose([
        # transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
        # transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.checkpoint_save_target = f"{self.checkpoint_save_target}/{self.exp_name}"

        self.augmentations = transforms.Compose([
        transforms.RandomRotation(degrees=10, expand=False, fill=0),
        # T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        # T.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=0)
        ])

        if not (os.path.isdir(self.checkpoint_save_target)):
            os.makedirs(self.checkpoint_save_target)

    def local_get_dataloaders(self, train_split=None, iteration=None):
        """Get dataloaders with consistent splits"""
        now = datetime.now()
        currentTime = now.strftime("%Y-%m-%d_%H-%M-%S")

        if self.split_file:
            print(f"Loading from split file: {self.split_file}")
            split_file = self.split_file
        else:
            # Create new split file name for this iteration
            print("Creating new split file...")
            split_file = f"data_splits/{self.exp_name}_{currentTime}.json"
            if not os.path.exists("data_splits"):
                os.makedirs("data_splits")

        # Get train loader with augmentations
        train_loader, _, _ = get_dataloaders(
            hdf5_path=self.hdf5_path,
            preprocess=self.preprocess,
            train_split=train_split if train_split else 0.7,
            batch_size=self.batch_size,
            labels_key=self.labels_key,
            split_file=split_file,  # This will create and save the split file
            augmentations=self.augmentations,
            oversample=self.use_oversampling
        )

        # Get val/test loaders using same split file but without augmentations
        _, val_loader, test_loader = get_dataloaders(
            hdf5_path=self.hdf5_path,
            preprocess=self.preprocess,
            train_split=train_split if train_split else 0.7,
            batch_size=1,
            labels_key=self.labels_key,
            split_file=split_file,  # This will use the saved split file
            augmentations=None,
            oversample=False
        )

        return train_loader, val_loader, test_loader
    
    def evaluate_model(self, dataloader, epoch, loss_fn, name="", fold=None):
        print(f"\nEVALUATING on {name}\n")
        metrics_dict = {}
        self.model.eval()

        all_labels = []
        all_probs = []
        all_feats = []
        all_logits = []

        all_original_labels = []
        total_loss = 0.0

        with torch.no_grad():

            for batch_imgs, batch_labels in tqdm(dataloader, desc=f"Evaluating {name} {fold}"):
                all_original_labels.extend(batch_labels.cpu().numpy())
                batch_labels = batch_labels % 4

                batch_imgs, batch_labels = batch_imgs.to(self.device), batch_labels.to(self.device)

                feats = self.model.features(batch_imgs)
                logits = self.model.classifier(feats)
                probs = torch.softmax(logits, dim=1)


                loss_val = loss_fn(logits, batch_labels)
                total_loss += loss_val.item()

                all_labels.extend(batch_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_logits.extend(logits.argmax(dim=1).cpu().numpy())

            avg_loss = total_loss / len(dataloader)
            print(f"Average Loss - {name} {fold}: {avg_loss}")

            all_labels = np.array(all_labels).flatten()
            all_probs = np.array(all_probs)  # keep as 2D
            all_logits = np.array(all_logits).flatten()
            all_original_labels = np.array(all_original_labels).flatten()

            # prob_true, prob_pred = calibration_curve(all_labels, all_probs, n_bins=10)
            # bins = list(range(len(prob_true)))
            # wandb.log({f"{name}/calibration_curve": wandb.plot.line_series(
            #     xs=[bins, bins],
            #     ys=[prob_true, prob_pred],
            #     keys=["True", "Predicted"],
            #     title="Calibration Curve",
            #     xname="Bin"
            # )}, step=epoch)

            # print(all_labels)
            # print(all_probs)

            preds = torch.argmax(torch.tensor(all_probs), dim=1).numpy()
            
            # Compute metrics
            cm = skmetrics.confusion_matrix(all_labels, preds)

            spec_at_09_sens, threshold = metrics.multiclass_specificity_at_sensitivity(all_labels, all_probs, 0.9)

            accuracy = metrics.get_accuracy(cm)
            sensitivity = metrics.get_sensitivity(cm)
            specificity = metrics.get_specificity(cm)
            f1 = metrics.get_f1_score(cm)
            kappa = metrics.get_cohens_kappa(cm)

            if(self.cfg["LABELS_KEY"] == "multiclass_stb" and self.cfg["CLF_TASK_LABELS_KEY"] == "profusion_score"):
                tb_stratified_cm = plot_tb_stratified_binary_cm(all_labels, preds, all_original_labels)

                wandb.log({
                    f"cm/{name}-tb_strat_cm": wandb.Image(tb_stratified_cm) if epoch % 5 == 0 else None
                }, step=epoch)

                if(name == "test"):
                    wandb.log({
                        "cm/test-tb_strat_cm": wandb.Image(tb_stratified_cm)
                    })

                tn, fp, fn, tp = metrics.get_cm_for_class(cm, 0)
                bin_cm = np.array([[tp, fn],
                    [fp, tn]])
                
                bin_acc = metrics.get_accuracy(bin_cm)
                bin_sens = metrics.get_sensitivity(bin_cm)
                bin_spec = metrics.get_specificity(bin_cm)
                bin_f1 = metrics.get_f1_score(bin_cm)
                bin_kappa = metrics.get_cohens_kappa(bin_cm)

                # bin_spec_at_sens, bin_thresh = metrics.specificity_at_sensitivity(all_labels, all_probs, 0.9)


            try:
                auc_score = skmetrics.roc_auc_score(all_labels, all_probs, multi_class='ovr')
                # fpr, tpr, _ = skmetrics.roc_curve(all_labels, all_probs)

            except ValueError as e:
                print(f"Warning: Could not calculate auc")
                auc_score = None
                # fpr, tpr = None, None

            

            metrics_dict["auc"] = auc_score
            metrics_dict["fpr"] = None                            # To do: Implement per-class tpr, fpr etc.
            metrics_dict["tpr"] = None
            metrics_dict["loss"] = total_loss

            metrics_dict["spec_at_09_sens"] = spec_at_09_sens
            metrics_dict["threshold"] = threshold
            metrics_dict["cm"] = cm
            metrics_dict["accuracy"] = accuracy
            metrics_dict["f1"] = f1
            metrics_dict["sensitivity"] = sensitivity
            metrics_dict["specificity"] = specificity
            metrics_dict["kappa"] = kappa

            metrics_dict["bin_acc"] = bin_acc
            metrics_dict["bin_sens"] = bin_sens
            metrics_dict["bin_spec"] = bin_spec
            metrics_dict["bin_f1"] = bin_f1
            metrics_dict["bin_kappa"] = bin_kappa
            # metrics_dict["bin_spec_at_sens"] = bin_spec_at_sens

            cm_img = plot_combined_conf_mat(cm)


        # print(f"labels: {all_labels} \n logits: {all_logits} \n probs: {all_probs}")
        

        return metrics_dict
    
    def train(self, cfg, optimizer, fold):

        best_spec_at_09_sens = -np.inf
        best_model_path = None
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,}")

        train_loader, val_loader, test_loader = self.local_get_dataloaders(train_split=0.7, iteration=fold)


        if self.cfg["LOSS_FUNC"] == "CrossEntropyLoss":
            self.loss_fn = nn.CrossEntropyLoss()
            
            if(self.cfg["WEIGHTED_LOSS"]):
                raise ValueError ("Weighted loss not yet implemented for multiclass classification.")
            
        elif self.cfg["LOSS_FUNC"] == "FocalLoss":
            raise ValueError ("FocalLoss not yet implemented for multiclass classification.")

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )

        for epoch in tqdm(range(cfg["EPOCHS"]), desc=f"Training on {cfg['TRAIN_SET_NAME']}"):
            self.model.train()
            total_loss = 0

            all_original_labels = []

            for batch_imgs, batch_labels in train_loader:
                all_original_labels.extend(batch_labels.cpu().numpy())

                batch_labels = batch_labels % 4
                batch_imgs, batch_labels = batch_imgs.to(self.device), batch_labels.to(self.device)

                optimizer.zero_grad()

                feats = self.model.features(batch_imgs)
                logits = self.model.classifier(feats)
                loss = self.loss_fn(logits, batch_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{cfg['EPOCHS']}, Loss: {total_loss/len(train_loader)}")

            train_dict = self.evaluate_model(train_loader, epoch, self.loss_fn, name="train", fold=fold)

            cm_img = plot_combined_conf_mat(train_dict["cm"])

            
            wandb.log({
                "train/loss": train_dict["loss"],
                "train/auc": train_dict["auc"],
                "train/fpr": train_dict["fpr"],
                "train/tpr": train_dict["tpr"],
                "train/spec_at_09_sens": train_dict["spec_at_09_sens"],
                "train/threshold": train_dict["threshold"],
                "cm/train_cm": wandb.Image(cm_img) if epoch % 5 == 0 else None,
                "train/accuracy": train_dict["accuracy"],
                "train/f1": train_dict["f1"],
                "train/sensitivity": train_dict["sensitivity"],
                "train/specificity": train_dict["specificity"],
                "train/kappa": train_dict["kappa"],

                "train/bin_acc": train_dict["bin_acc"],
                "train/bin_sens": train_dict["bin_sens"],
                "train/bin_spec": train_dict["bin_spec"],
                "train/bin_f1": train_dict["bin_f1"],
                "train/bin_kappa": train_dict["bin_kappa"],
                # "train/bin_spec_at_sens": train_dict["bin_spec_at_sens"]
            }, step=epoch)

            val_dict = self.evaluate_model(val_loader, epoch, self.loss_fn, name="val", fold=fold)
            # Checkpoint logic
            if val_dict["spec_at_09_sens"] > best_spec_at_09_sens:
                best_spec_at_09_sens = val_dict["spec_at_09_sens"]
                # Save model and optimizer state
                model_filename = f"{self.checkpoint_save_target}/model_{self.cfg['RUN_NAME']}_fold{fold}_best.pth"
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'spec_at_09_sens': best_spec_at_09_sens
                }, model_filename)
                best_model_path = model_filename
                print(f"Checkpointed model at {model_filename} (spec@0.9sens={best_spec_at_09_sens:.4f})")

                wandb.log({
                    "cm/best_val_spec_at_sens": wandb.Image(plot_combined_conf_mat(val_dict["cm"]))
                }, step=epoch)

            val_cm_img = plot_combined_conf_mat(val_dict["cm"])

            scheduler.step(val_dict["specificity"])

            wandb.log({
                "val/auc": val_dict["auc"],
                "val/fpr": val_dict["fpr"],
                "val/tpr": val_dict["tpr"],
                "val/loss": val_dict["loss"],
                "val/spec_at_09_sens": val_dict["spec_at_09_sens"],
                "val/threshold": val_dict["threshold"],
                "cm/val_cm": wandb.Image(val_cm_img) if (epoch % 5 == 0) else None,
                "val/accuracy": val_dict["accuracy"],
                "val/f1": val_dict["f1"],
                "val/sensitivity": val_dict["sensitivity"],
                "val/specificity": val_dict["specificity"],
                "val/kappa": val_dict["kappa"],

                "val/bin_acc": val_dict["bin_acc"],
                "val/bin_sens": val_dict["bin_sens"],
                "val/bin_spec": val_dict["bin_spec"],
                "val/bin_f1": val_dict["bin_f1"],
                "val/bin_kappa": val_dict["bin_kappa"],
                # "val/bin_spec_at_sens": val_dict["bin_spec_at_sens"]
            }, step=epoch)
        

        test_dict = self.evaluate_model(test_loader, epoch, self.loss_fn, name="test", fold=fold)

        return test_dict
    
    def save_results_to_excel(experiment_name, fold_metrics, summary_metrics, excel_path="experiment_results.xlsx"):
        import pandas as pd
        """
        Save per-fold and summary metrics to an Excel file.
        """
        # Per-fold DataFrame
        df_folds = pd.DataFrame(fold_metrics)
        df_folds.insert(0, "fold", range(1, len(df_folds)+1))
        df_folds["experiment"] = experiment_name

        # Summary DataFrame (mean, std, var, ci95)
        summary_rows = []
        for metric, stats in summary_metrics.items():
            row = {"metric": metric}
            row.update(stats)
            row["experiment"] = experiment_name
            summary_rows.append(row)
        df_summary = pd.DataFrame(summary_rows)

        # Write to Excel (append if exists)
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a" if os.path.exists(excel_path) else "w") as writer:
            df_folds.to_excel(writer, sheet_name=f"{experiment_name}_folds", index=False)
            df_summary.to_excel(writer, sheet_name=f"{experiment_name}_summary", index=False)
    

    def run_k_folds(self):
        project_name = self.cfg["PROJECT_NAME"]
        exp_name = self.cfg["RUN_NAME"]

        accuracies = []
        sensitivities = []
        kappas = []
        specificities = []

        bin_accuracies = []
        bin_sensitivities = []
        bin_kappas = []
        bin_specificities = []

        wandb.login()

        group_name = f"{self.exp_name}-{wandb.util.generate_id()}"

        for i in range(self.cfg["NUM_FOLDS"]):

            if self.cfg["RESOLUTION"] == 512:
                model = xrv.models.ResNet(weights="resnet50-res512-all")

                if not self.cfg["PRETRAINED"]:
                    # raise ValueError("need to still fix random init.")
                    reinitialize_weights(model)
            else:
                raise ValueError(f"Unsupported resolution: {self.cfg['RESOLUTION']}")

            if(self.cfg["CLF_TYPE"] == "Linear"):
                model.classifier = XRVBasedClassifier(input_dim=2048, num_classes=self.cfg["NUM_CLASSES"], name="bin_XRV-Base")
            elif(self.cfg["CLF_TYPE"] == "MLP"):
                model.classifier = MulticlassClassifier(input_dim=2048, num_classes=self.cfg["NUM_CLASSES"], name="mc_mlp-Base")
            elif(self.cfg["CLF_TYPE"] == "MLP2"):
                model.classifier = MulticlassClassifier(input_dim=2048, dropout_rate=0.1, name="mc_mlp-dout_01")
            else:
                raise ValueError(f"Unsupported classifier type: {self.cfg['CLF_TYPE']}")
            
            model = model.to(self.device)
            self.model = model

            if self.cfg["FREEZE_ENC"]:
                for name, param in self.model.named_parameters():
                # Only freeze parameters that are NOT in the classifier
                    if not name.startswith("classifier"):
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                # Only optimize classifier parameters
                optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=config["LEARNING_RATE"], weight_decay=self.cfg["WEIGHT_DECAY"])
            else:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=config["LEARNING_RATE"], weight_decay=self.cfg["WEIGHT_DECAY"])



            wandb.init(
                project=project_name,
                name=f"{exp_name}-fold-{i+1}",
                group=group_name,
                config={
                    "loss_fn": self.loss_fn.__class__.__name__,
                    "optimizer": self.optimizer_type.__name__,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "labels_key": self.labels_key,
                    "batch_size":self.batch_size,
                    "epochs": self.epochs,
                    "save_dir": self.checkpoint_save_target,
                    "fold": i
                },
            )

            wandb.config.update(self.cfg)

            test_dict = self.train(self.cfg, optimizer, fold=i)


            accuracies.append(test_dict["accuracy"])
            sensitivities.append(test_dict["sensitivity"])
            kappas.append(test_dict["kappa"])
            specificities.append(test_dict["specificity"])

            bin_accuracies.append(test_dict["bin_acc"])
            bin_sensitivities.append(test_dict["bin_sens"])
            bin_kappas.append(test_dict["bin_kappa"])
            bin_specificities.append(test_dict["bin_spec"])

            test_cm_img = plot_combined_conf_mat(test_dict["cm"])



            wandb.log({
                "test/auc": test_dict["auc"],
                "test/fpr": test_dict["fpr"],
                "test/tpr": test_dict["tpr"],
                "test/loss": test_dict["loss"],
                "test/spec_at_09_sens": test_dict["spec_at_09_sens"],
                "test/threshold": test_dict["threshold"],
                "cm/test_cm": wandb.Image(test_cm_img),

                "test/accuracy": test_dict["accuracy"],
                "test/f1": test_dict["f1"],
                "test/sensitivity": test_dict["sensitivity"],
                "test/specificity": test_dict["specificity"],
                "test/kappa": test_dict["kappa"],

                "test/bin_acc": test_dict["bin_acc"],
                "test/bin_sens": test_dict["bin_sens"],
                "test/bin_spec": test_dict["bin_spec"],
                "test/bin_f1": test_dict["bin_f1"],
                "test/bin_kappa": test_dict["bin_kappa"],
                # "test/bin_spec_at_sens": test_dict["bin_spec_at_sens"]
            })

            wandb.finish()

        results = {
            "accuracy": mean_std_var_ci(accuracies),
            "sensitivity": mean_std_var_ci(sensitivities),
            "specificity": mean_std_var_ci(specificities),
            "kappa": mean_std_var_ci(kappas)
        }

        return results


def load_config(config_path="config.yaml"):
    """
    Load the YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.
    Returns:
        A dictionary containing the configuration settings.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    with open(config_path, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Error reading the configuration file: {e}")




if __name__ == "__main__":

    config_files = glob.glob("/home/sean/MSc_2025/clf_cv/mc_configs/*.yaml")

    print(f"Found {len(config_files)} configuration files.")

    for cfg_path in config_files:
        with open(cfg_path, "r") as f:
            config = yaml.safe_load(f)
            print(config)


            # Initialize CUDA device
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
            # Initialize model
            model = xrv.models.ResNet(weights="resnet50-res512-all")
            model.classifier = XRVBasedClassifier(input_dim=2048, num_classes=config["NUM_CLASSES"], name="mc_XRV-Base")
            model = model.to(device)

            hdf5_path = config["DATA_PATH_TBNET"] if config["TRAIN_SET_NAME"] == "TBNET" else config["DATA_PATH_MBOD"]



            cv = MulticlassMonteCarloCV(
                model=model,
                cfg=config,
                labels_key=config["LABELS_KEY"],  # Specify your target label
                num_classes=config["NUM_CLASSES"],  # Multiclass classification
                device=device,
                loss_fn=None,
                hdf5_path=hdf5_path,
                batch_size=config["BATCH_SIZE"],
                epochs=config["EPOCHS"],
                optimizer_type=torch.optim.Adam,
                learning_rate=config["LEARNING_RATE"],
                weight_decay=config["WEIGHT_DECAY"],
                use_oversampling=config["OVERSAMPLE"],
                checkpoint_save_target=config["CHECKPOINT_SAVE_DIR"],
                exp_name=config["RUN_NAME"],
                split_file=None,
                train_set_name=config["TRAIN_SET_NAME"],
                test_set_name=config["TEST_SET_NAME"],
                clf_task_labels_key=config["CLF_TASK_LABELS_KEY"]
            )

            results = cv.run_k_folds()
            print(f"Results for {config['RUN_NAME']}: {results}")