import os
import time
import json
import csv
import pickle
import gc
import torch
import psutil
import platform
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets
# import numpy as np # Duplicate import removed
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
    log_loss, confusion_matrix, mean_absolute_error,
)
from sklearn.model_selection import StratifiedKFold
import mlflow

# Import necessary components from model.py
from lightweight_model import GenericClassifier, mobilenet_v3_large_backbone,vit_small_backbone, efficientformer_l1_backbone,        shufflenetv2_x1_backbone

# Define a list of backbone factory functions to iterate through
backbone_factories = [
        mobilenet_v3_large_backbone,
        vit_small_backbone,
        efficientformer_l1_backbone,
        shufflenetv2_x1_backbone
]

# Configuration (adapt from classification_mlflow.py)
class Config:
    img_size = 224 # Use 224 for ViT compatibility, adjust if needed for other models
    batch_size = 32
    lr = 1e-4
    epochs = 100
    num_classes = None # Will be set dynamically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MLflow setup (adapt from classification_mlflow.py)
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
os.environ["MLFLOW_ENABLE_ROCM_MONITORING"] = "false"
mlflow_dir = os.path.join(os.path.expanduser("~"), "mlruns")
mlflow.set_tracking_uri(f"file:///{mlflow_dir}")
class AugmentedImageDataset(Dataset):
    def __init__(self, subset, tfm):
        self.subset = subset
        self.tfm = tfm
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        # Convert palette images with transparency to RGBA
        if x.mode == 'P' and 'transparency' in x.info:
            x = x.convert('RGBA')
        # Ensure image is RGB before transform, as ToTensor() expects RGB
        if x.mode != 'RGB':
            x = x.convert('RGB')
        return self.tfm(x), y


# --- Training, Validation, and Plotting Functions (Copied from classification_mlflow.py) ---
def save_model(model, backbone_name, fold_num, path="."):
    filename = f"{backbone_name}_fold_{fold_num}_best.pth"
    full_path = os.path.join(path, filename)
    torch.save(model.state_dict(), full_path)
    print(f"Model saved to {full_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues,
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

def plot_roc_curve(y_true, y_probs, class_names, save_path, title="ROC Curve"):
    from sklearn.metrics import roc_curve, auc
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve((np.array(y_true) == i).astype(int), y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_name}: AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve saved to {save_path}")

def plot_precision_recall_curve(y_true, y_probs, class_names, save_path, title="Precision-Recall Curve"):
    from sklearn.metrics import precision_recall_curve, auc
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve((np.array(y_true) == i).astype(int), y_probs[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{class_name}: AUC={pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.savefig(save_path)
    plt.close()
    print(f"Precision-Recall curve saved to {save_path}")

def train_model(model, train_loader, val_loader, config, optimizer, scheduler, backbone_name, fold_num):
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    train_losses = []
    val_losses = []
    epoch_times = []
    memory_usages = []
    
    batch_metrics = {
        'loss': [],
        'grad_norms': [],
        'lr': []
    }
    
    # Log initial memory usage
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        initial_mem = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        mlflow.log_metric("initial_gpu_memory", initial_mem)
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.to(config.device)
            labels = labels.to(config.device)
            optimizer.zero_grad()
            
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            
            # Calculate gradient norms per component
            grad_norms_per_component = {}
            total_norm_sq = 0
            for name, p in model.named_parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_norm_sq += param_norm ** 2
                    
                    component_name = name.split('.')[0] # e.g., 'base' or 'classifier'
                    if component_name == "base": # Further break down base model
                        if len(name.split('.')) > 1:
                           component_name = f"base.{name.split('.')[1]}" # e.g., base.patch_embed, base.blocks
                           if "blocks" in component_name and len(name.split('.')) > 2:
                               component_name = f"base.blocks.{name.split('.')[2]}" # e.g., base.blocks.0

                    if component_name not in grad_norms_per_component:
                        grad_norms_per_component[component_name] = []
                    grad_norms_per_component[component_name].append(param_norm)

            total_norm = total_norm_sq ** 0.5
            
            optimizer.step()
            scheduler.step()
            
            # Record batch metrics
            batch_metrics['loss'].append(loss.item())
            batch_metrics['grad_norms'].append(total_norm) # Overall grad norm
            for comp_name, norms in grad_norms_per_component.items():
                avg_comp_norm = np.mean(norms) if norms else 0
                if f'grad_norm_{comp_name}' not in batch_metrics:
                    batch_metrics[f'grad_norm_{comp_name}'] = []
                batch_metrics[f'grad_norm_{comp_name}'].append(avg_comp_norm)
            batch_metrics['lr'].append(scheduler.get_last_lr()[0])
            
            total_loss += loss.item()
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        epoch_time = time.time() - start_time
        # Log batch statistics
        loggable_batch_metrics = {}
        for key, values in batch_metrics.items():
            if values: # Ensure there are values to average/std
                loggable_batch_metrics[f'{key}_avg'] = np.mean(values)
                loggable_batch_metrics[f'{key}_std'] = np.std(values)
        if 'lr_avg' in loggable_batch_metrics: # LR doesn't need std, just last value or avg
             loggable_batch_metrics['lr_last'] = batch_metrics['lr'][-1]

        mlflow.log_metrics(loggable_batch_metrics, step=epoch)
        
        # Reset batch metrics
        current_keys = list(batch_metrics.keys())
        batch_metrics = {k: [] for k in current_keys} # Reset all collected keys
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        val_acc, val_mae, val_loss = validate_model(model, val_loader, config, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_times.append(epoch_time)
        # Log system metrics
        cpu_percent = psutil.cpu_percent()
        ram_used = psutil.virtual_memory().used / (1024**3)  # GB
        mlflow.log_metrics({
            "cpu_usage": cpu_percent,
            "ram_used": ram_used
        }, step=epoch)

        # Log memory usage
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_used = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            memory_usages.append(mem_used)
            mlflow.log_metric("gpu_memory", mem_used, step=epoch)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, Val MAE={val_mae:.3f}, "
              f"Time={epoch_time:.2f}s, GPU Mem={mem_used if torch.cuda.is_available() else 0:.1f}MB")
        # Log epoch metrics with MLflow
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)
        mlflow.log_metric("val_mae", val_mae, step=epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model, backbone_name, fold_num)
    # Log training summary statistics
    mlflow.log_metrics({
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'min_train_loss': np.min(train_losses),
        'min_val_loss': np.min(val_losses),
        'avg_epoch_time': np.mean(epoch_times),
        'total_training_time': np.sum(epoch_times),
        'max_gpu_memory': np.max(memory_usages) if memory_usages else 0,
        'avg_gpu_memory': np.mean(memory_usages) if memory_usages else 0,
        'final_gpu_memory': memory_usages[-1] if memory_usages else 0
    })
    
    return model, train_losses, val_losses, epoch_times

def validate_model(model, loader, config, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(config.device)
            labels = labels.to(config.device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        acc = 0.0
        mae = 0.0
    else:
        acc = 100.0 * correct / total
        mae = mean_absolute_error(all_labels, all_preds)

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0

    return acc, mae, avg_loss

# -----------------------------------------------------------------------------


def run_experiment(backbone_fn, config, train_loader, val_loader, test_loader, class_names, fold_num):
    """Runs a single experiment with a given backbone for a specific fold."""
    with mlflow.start_run(run_name=f"{backbone_fn.__name__}_fold_{fold_num}_run"):
        # Log system information
        mlflow.log_params({
            "system_os": platform.system(),
            "system_platform": platform.platform(),
            "system_processor": platform.processor(),
            "system_ram_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        })

        # Log dataset statistics
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)
        num_total = train_size + val_size + test_size
        
        # Log parameters specific to this run
        mlflow.log_param("model", backbone_fn.__name__)
        mlflow.log_param("dataset_split_ratio", "80/20 (Train+Val/Test) with 5-fold CV on Train+Val")
        mlflow.log_param("total_samples", num_total)
        mlflow.log_param("train_samples", train_size)
        mlflow.log_param("train_percentage", train_size/num_total)
        mlflow.log_param("val_samples", val_size)
        mlflow.log_param("val_percentage", val_size/num_total)
        mlflow.log_param("test_samples", test_size)
        mlflow.log_param("test_percentage", test_size/num_total)
        
        mlflow.log_param("random_seed", 42)
        mlflow.log_param("fold_number", fold_num)

        # Calculate and log label distributions
        def get_label_counts(loader):
            counts = {name: 0 for name in class_names}
            for _, labels in loader:
                for label in labels:
                    counts[class_names[label]] += 1
            return counts

        train_counts = get_label_counts(train_loader)
        val_counts = get_label_counts(val_loader)
        test_counts = get_label_counts(test_loader)

        mlflow.log_param("train_label_counts", json.dumps(train_counts))
        mlflow.log_param("val_label_counts", json.dumps(val_counts))
        mlflow.log_param("test_label_counts", json.dumps(test_counts))
        
        # Log model and training configuration
        mlflow.log_params({
            "img_size": config.img_size,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "epochs": config.epochs,
            "num_classes": config.num_classes,
            "device": str(config.device),
            "optimizer": "AdamW",
            "lr_scheduler": "CosineAnnealingLR",
            "loss_function": "CrossEntropyLoss"
        })

        # Create model
        model = GenericClassifier(config.num_classes, backbone_fn).to(config.device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param("total_params", total_params)
        mlflow.log_param("trainable_params", trainable_params)
        print(f"Running experiment with {backbone_fn.__name__} | Total parameters: {total_params} | Trainable: {trainable_params}")

        # Log model architecture as text
        mlflow.log_text(str(model), "model_architecture.txt")

        # Optimizer and Scheduler
        print("INFO: Using CosineAnnealingLR scheduler.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(train_loader) * config.epochs,
            eta_min=1e-6
        )
        mlflow.log_param("lr_initial", config.lr)
        mlflow.log_param("scheduler_T_max", len(train_loader) * config.epochs)
        mlflow.log_param("scheduler_eta_min", 1e-6)

        # --- Training Loop ---
        print("--- Starting Training ---")
        trained_model, train_losses, val_losses, epoch_times = train_model(
            model, train_loader, val_loader, config, optimizer, scheduler,
            backbone_name=backbone_fn.__name__, fold_num=fold_num
        )
        print("--- Training Finished ---")
        # ---------------------

        # --- Evaluation ---
        print("--- Starting Evaluation ---")
        test_acc, test_mae, _ = validate_model(trained_model, test_loader, config, criterion=nn.CrossEntropyLoss())
        mlflow.log_metric("test_acc", test_acc)
        mlflow.log_metric("test_mae", test_mae)
        print(f"Final Test Acc: {test_acc:.2f}% | Test MAE: {test_mae:.3f}")
        print("--- Evaluation Finished ---")
        # ------------------

        # --- Plotting and Logging Artifacts ---
        print("--- Logging Artifacts ---")
        all_true, all_pred = [], []
        all_probs = []
        trained_model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(config.device)
                labels = labels.to(config.device)
                logits = trained_model(images)
                probs = F.softmax(logits, dim=1)
                _, preds = torch.max(logits, 1)
                all_true.extend(labels.cpu().numpy())
                all_pred.extend(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)

        # Calculate metrics
        accuracy = accuracy_score(all_true, all_pred)
        f1 = f1_score(all_true, all_pred, average='weighted')
        precision = precision_score(all_true, all_pred, average='weighted')
        recall = recall_score(all_true, all_pred, average='weighted')
        mcc = matthews_corrcoef(all_true, all_pred)
        cohen_kappa = cohen_kappa_score(all_true, all_pred)
        balanced_acc = balanced_accuracy_score(all_true, all_pred)
        logloss = log_loss(all_true, all_probs, labels=list(range(config.num_classes)))
        cm = confusion_matrix(all_true, all_pred)

        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("matthews_corrcoef", mcc)
        mlflow.log_metric("cohen_kappa", cohen_kappa)
        mlflow.log_metric("balanced_accuracy", balanced_acc)
        mlflow.log_metric("log_loss", logloss)

        # Log all metrics as a dictionary to MLflow
        test_metrics = {
            "test_accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "matthews_corrcoef": mcc,
            "cohen_kappa": cohen_kappa,
            "balanced_accuracy": balanced_acc,
            "log_loss": logloss
        }
        mlflow.log_metrics(test_metrics)

        run_id = mlflow.active_run().info.run_id

        # Save metrics to a CSV file
        metrics_file_path = f"test_metrics_{run_id}.csv"
        with open(metrics_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(test_metrics.keys())
            writer.writerow(test_metrics.values())
        mlflow.log_artifact(metrics_file_path)

        roc_img_path = f"roc_curve_{run_id}.png"
        plot_roc_curve(all_true, all_probs, class_names, save_path=roc_img_path, title="ROC Curve")
        mlflow.log_artifact(roc_img_path)

        pr_img_path = f"pr_curve_{run_id}.png"
        plot_precision_recall_curve(all_true, all_probs, class_names, save_path=pr_img_path, title="Precision-Recall Curve")
        mlflow.log_artifact(pr_img_path)

        cm_json_path = f"confusion_matrix_{run_id}.json"
        with open(cm_json_path, "w") as f:
            json.dump(cm.tolist(), f)
        mlflow.log_artifact(cm_json_path)

        cm_img_path = f"confusion_matrix_{run_id}.png"
        plot_confusion_matrix(all_true, all_pred, class_names, save_path=cm_img_path)
        mlflow.log_artifact(cm_img_path)

        cm_pickle_path = f"confusion_matrix_{run_id}.pkl"
        with open(cm_pickle_path, "wb") as f:
            pickle.dump(cm, f)
        mlflow.log_artifact(cm_pickle_path)

        # Measure inference time on the test dataset
        trained_model.eval()
        total_time = 0.0
        total_samples = 0
        with torch.no_grad():
            for images, _ in tqdm(test_loader, desc="Measuring inference time"):
                images = images.to(config.device)
                batch_size = images.size(0)
                start_time = time.time()
                _ = trained_model(images)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                batch_time = end_time - start_time
                total_time += batch_time
                total_samples += batch_size

        avg_inference_time = total_time / total_samples
        print(f"Average inference time per sample: {avg_inference_time:.6f} seconds")
        mlflow.log_metric("Average inference time per sample", avg_inference_time)

        # Log final model with MLflow
        example_input, _ = next(iter(test_loader))
        example_input_np = example_input.cpu().numpy()
        mlflow.pytorch.log_model(
            trained_model.cpu(),
            artifact_path=trained_model.__class__.__name__,
            input_example=example_input_np
        )

        # Optionally, log training statistics
        np.savez("training_stats.npz", train_losses=train_losses, val_losses=val_losses, epoch_times=epoch_times)
        mlflow.log_artifact("training_stats.npz")

        print("--- Artifact Logging Finished ---")
        # --------------------------------------

        mlflow.end_run()

        # --- Memory Clearing ---
        print("Clearing memory...")
        del model
        del optimizer
        del scheduler
        del trained_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Memory cleared.")
        # -----------------------


if __name__ == "__main__":
    root_dir = r"your_path"  # Set correct path
    dataset_name = os.path.basename(root_dir)
    mlflow.set_experiment(f"{dataset_name}_classification_5fold_CV")

    # --- Data Preparation ---
    base_ds = datasets.ImageFolder(root=root_dir)
    class_names = base_ds.classes
    Config.num_classes = len(class_names)

    # Define transformations
    train_tfms = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_tfms = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Split dataset into training+validation and a separate test set (80-20 split)
    n = len(base_ds)
    test_len = int(0.2 * n)
    
    # Use indices to handle the split for reproducibility
    indices = list(range(n))
    np.random.seed(42)
    np.random.shuffle(indices)
    test_indices = indices[:test_len]
    train_val_indices = indices[test_len:]

    test_ss = Subset(base_ds, test_indices)

    # Create the test loader, which is fixed across all folds
    num_workers = 0 if os.name == 'nt' else os.cpu_count()
    test_loader = DataLoader(
        AugmentedImageDataset(test_ss, test_tfms),
        Config.batch_size,
        num_workers=num_workers
    )

    # --- 5-Fold Stratified Cross-Validation ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # We need the labels of the train+val set for stratification
    train_val_labels = [base_ds.targets[i] for i in train_val_indices]

    for fold, (train_idx_split, val_idx_split) in enumerate(skf.split(train_val_indices, train_val_labels)):
        print(f"--- Starting Fold {fold+1}/5 ---")
        
        # Map fold indices back to the original dataset indices
        actual_train_idx = [train_val_indices[i] for i in train_idx_split]
        actual_val_idx = [train_val_indices[i] for i in val_idx_split]

        # Create subsets for the current fold
        train_fold_ss = Subset(base_ds, actual_train_idx)
        val_fold_ss = Subset(base_ds, actual_val_idx)

        # Create data loaders for the current fold
        train_loader = DataLoader(
            AugmentedImageDataset(train_fold_ss, train_tfms),
            Config.batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        val_loader = DataLoader(
            AugmentedImageDataset(val_fold_ss, test_tfms),
            Config.batch_size,
            num_workers=num_workers
        )

        # Run experiment for each backbone on the current fold
        for backbone_fn in backbone_factories:
            run_experiment(
                backbone_fn, Config,
                train_loader, val_loader, test_loader,
                class_names, fold_num=fold+1
            )
            
    print("All experiments finished.")
