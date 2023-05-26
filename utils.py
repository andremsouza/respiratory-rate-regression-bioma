"""Functions for PyTorch utility."""

# %% [markdown]
# # Imports

# %%
import gc

import torch
import sklearn.metrics

import config
import data

# %% [markdown]
# # Functions

# %%


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    n_classes: int = 1,
    verbose: bool = True,
) -> dict:
    """Train a PyTorch model.

    Args:
        model (torch.nn.Module): model to train
        train_loader (torch.utils.data.DataLoader): data loader for training data
        criterion (torch.nn.Module): loss function
        optimizer (torch.optim.Optimizer): optimizer
        device (torch.device): device to train on
        num_epochs (int): number of epochs

    Returns:
        dict: dictionary of metrics
    """
    idx: int = 0
    model.train()  # set model to training mode
    metrics: dict = {
        "loss": [],
    }
    # loop over epochs
    for epoch in range(num_epochs):
        running_loss: float = 0.0
        # initialize inputs and labels tensors
        inputs: torch.Tensor = torch.empty(0)
        labels: torch.Tensor = torch.empty(0)
        # Acumulate labels and predictions for F1 score
        result_labels: torch.Tensor = torch.empty(0)
        result_preds: torch.Tensor = torch.empty(0)
        # loop over data
        for inputs, labels in train_loader:
            # free GPU memory
            torch.cuda.empty_cache()
            # free memory
            gc.collect()
            # for each input, expand into batches of N frames with stride S
            inputs = inputs.squeeze(0)
            labels = labels.squeeze(0)
            while inputs.shape[1] >= config.BATCH_SIZE:
                # get first batch_size frames from inputs
                inputs_first = data.expand_video_into_batches(
                    inputs,
                    batch_size=config.BATCH_SIZE,
                    stride=config.STRIDE,
                    first_only=True,
                    device="cpu",
                ).float()
                # remove stride frames from inputs
                inputs = inputs[:, config.STRIDE :, :, :].float()
                # Increment number of samples
                idx += 1
                # send input and labels to device
                inputs_first = inputs_first.float().to(device)
                labels = labels.float().to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    outputs = model(inputs_first)
                    loss = criterion(outputs, labels)
                    # if verbose:
                    #     print("inputs.shape:", inputs.shape)
                    #     print("inputs_first.shape:", inputs_first.shape)
                    #     print("outputs.shape:", outputs.shape)
                    #     print("labels.shape:", labels.shape)
                    #     print("outputs:", outputs)
                    #     print("labels:", labels)
                    #     print("loss:", loss)
                    # backward + optimize
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.item()
                result_labels = torch.cat((result_labels, labels.detach().cpu()), dim=0)
                result_preds = torch.cat((result_preds, outputs.detach().cpu()), dim=0)
                # running_corrects += int(torch.sum(preds == labels.data))
                # free GPU memory
                torch.cuda.empty_cache()
                # free memory
                del inputs_first
                gc.collect()
        # Calculate epoch metrics
        epoch_loss: float = running_loss / idx  # type: ignore
        # Append epoch metrics
        metrics["loss"].append(epoch_loss)
        # Print epoch metrics
        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Training Loss: " f"{epoch_loss:.4f} ")
        # free GPU memory
        torch.cuda.empty_cache()
        # free memory
        gc.collect()
    return metrics


def test(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    verbose: bool = True,
) -> dict:
    """Test a PyTorch model.

    Args:
        model (torch.nn.Module): model to test
        test_loader (torch.utils.data.DataLoader): data loader for test data
        criterion (torch.nn.Module): loss function
        device (torch.device): device to test on

    Returns:
        dict: dictionary of metrics
    """
    model.eval()  # set model to evaluation mode

    # initialize inputs and labels tensors
    inputs: torch.Tensor = torch.empty(0)
    labels: torch.Tensor = torch.empty(0)

    # Acumulate labels and predictions for F1 score
    result_labels: torch.Tensor = torch.empty(0)
    result_preds: torch.Tensor = torch.empty(0)

    # initialize loss and accuracy
    running_loss: float = 0.0
    running_corrects: int = 0

    # loop over data
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += int(
            torch.sum((outputs >= config.PRED_THRESHOLD).float() == labels.data)
        )
        result_labels = torch.cat((result_labels, labels.detach().cpu()), dim=0)
        result_preds = torch.cat((result_preds, outputs.detach().cpu()), dim=0)
    # Calculate epoch metrics
    epoch_loss: float = running_loss / float(len(test_loader.dataset))  # type: ignore
    epoch_accuracy = sklearn.metrics.accuracy_score(
        y_true=result_labels.cpu().numpy().astype(int),
        y_pred=(result_preds.cpu().numpy() >= config.PRED_THRESHOLD).astype(int),
    )
    epoch_hamming = sklearn.metrics.hamming_loss(
        y_true=result_labels.cpu().numpy().astype(int),
        y_pred=(result_preds.cpu().numpy() >= config.PRED_THRESHOLD).astype(int),
    )
    epoch_precision_macro = sklearn.metrics.precision_score(
        y_true=result_labels.cpu().numpy().astype(int),
        y_pred=(result_preds.cpu().numpy() >= config.PRED_THRESHOLD).astype(int),
        average="macro",
        zero_division="warn",
    )
    epoch_precision_micro = sklearn.metrics.precision_score(
        y_true=result_labels.cpu().numpy().astype(int),
        y_pred=(result_preds.cpu().numpy() >= config.PRED_THRESHOLD).astype(int),
        average="micro",
        zero_division="warn",
    )
    epoch_precision_weighted = sklearn.metrics.precision_score(
        y_true=result_labels.cpu().numpy().astype(int),
        y_pred=(result_preds.cpu().numpy() >= config.PRED_THRESHOLD).astype(int),
        average="weighted",
        zero_division="warn",
    )
    epoch_precision_none = sklearn.metrics.precision_score(
        y_true=result_labels.cpu().numpy().astype(int),
        y_pred=(result_preds.cpu().numpy() >= config.PRED_THRESHOLD).astype(int),
        average=None,
        zero_division="warn",
    )
    epoch_recall_macro = sklearn.metrics.recall_score(
        y_true=result_labels.cpu().numpy().astype(int),
        y_pred=(result_preds.cpu().numpy() >= config.PRED_THRESHOLD).astype(int),
        average="macro",
        zero_division="warn",
    )
    epoch_recall_micro = sklearn.metrics.recall_score(
        y_true=result_labels.cpu().numpy().astype(int),
        y_pred=(result_preds.cpu().numpy() >= config.PRED_THRESHOLD).astype(int),
        average="micro",
        zero_division="warn",
    )
    epoch_recall_weighted = sklearn.metrics.recall_score(
        y_true=result_labels.cpu().numpy().astype(int),
        y_pred=(result_preds.cpu().numpy() >= config.PRED_THRESHOLD).astype(int),
        average="weighted",
        zero_division="warn",
    )
    epoch_recall_none = sklearn.metrics.recall_score(
        y_true=result_labels.cpu().numpy().astype(int),
        y_pred=(result_preds.cpu().numpy() >= config.PRED_THRESHOLD).astype(int),
        average=None,
        zero_division="warn",
    )
    epoch_f1_macro = sklearn.metrics.f1_score(
        y_true=result_labels.cpu().numpy().astype(int),
        y_pred=(result_preds.cpu().numpy() >= config.PRED_THRESHOLD).astype(int),
        average="macro",
        zero_division="warn",
    )
    epoch_f1_micro = sklearn.metrics.f1_score(
        y_true=result_labels.cpu().numpy().astype(int),
        y_pred=(result_preds.cpu().numpy() >= config.PRED_THRESHOLD).astype(int),
        average="micro",
        zero_division="warn",
    )
    epoch_f1_weighted = sklearn.metrics.f1_score(
        y_true=result_labels.cpu().numpy().astype(int),
        y_pred=(result_preds.cpu().numpy() >= config.PRED_THRESHOLD).astype(int),
        average="weighted",
        zero_division="warn",
    )
    epoch_f1_none = sklearn.metrics.f1_score(
        y_true=result_labels.cpu().numpy().astype(int),
        y_pred=(result_preds.cpu().numpy() >= config.PRED_THRESHOLD).astype(int),
        average=None,
        zero_division="warn",
    )
    epoch_roc_auc_macro = sklearn.metrics.roc_auc_score(
        y_true=result_labels.cpu().numpy().astype(int),
        y_score=result_preds.cpu().numpy(),
        average="macro",
        multi_class="ovr",
    )
    epoch_roc_auc_micro = sklearn.metrics.roc_auc_score(
        y_true=result_labels.cpu().numpy().astype(int),
        y_score=result_preds.cpu().numpy(),
        average="micro",
        multi_class="ovr",
    )
    epoch_roc_auc_weighted = sklearn.metrics.roc_auc_score(
        y_true=result_labels.cpu().numpy().astype(int),
        y_score=result_preds.cpu().numpy(),
        average="weighted",
        multi_class="ovr",
    )
    epoch_roc_auc_none = sklearn.metrics.roc_auc_score(
        y_true=result_labels.cpu().numpy().astype(int),
        y_score=result_preds.cpu().numpy(),
        average=None,
        multi_class="ovr",
    )

    if verbose:
        print(
            f"Test Loss: {epoch_loss:.4f} "
            f"Accuracy: {epoch_accuracy:.4f} "
            f"Hamming: {epoch_hamming:.4f} "
            f"\n"
            f"Precision (macro): {epoch_precision_macro} "
            f"Precision (micro): {epoch_precision_micro} "
            f"Precision (weighted): {epoch_precision_weighted} "
            f"Precision (none): {epoch_precision_none} "
            f"\n"
            f"Recall (macro): {epoch_recall_macro} "
            f"Recall (micro): {epoch_recall_micro} "
            f"Recall (weighted): {epoch_recall_weighted} "
            f"Recall (none): {epoch_recall_none} "
            f"\n"
            f"F1 (macro): {epoch_f1_macro} "
            f"F1 (micro): {epoch_f1_micro} "
            f"F1 (weighted): {epoch_f1_weighted} "
            f"F1 (none): {epoch_f1_none} "
            f"\n"
            f"ROC AUC (macro): {epoch_roc_auc_macro:.4f} "
            f"ROC AUC (micro): {epoch_roc_auc_micro:.4f} "
            f"ROC AUC (weighted): {epoch_roc_auc_weighted:.4f} "
            f"ROC AUC (none): {epoch_roc_auc_none} "
        )

    return {
        "loss": epoch_loss,
        "accuracy": epoch_accuracy,
        "hamming_loss": epoch_hamming,
        "precision_macro": epoch_precision_macro,
        "precision_micro": epoch_precision_micro,
        "precision_weighted": epoch_precision_weighted,
        "precision_none": epoch_precision_none,
        "recall_macro": epoch_recall_macro,
        "recall_micro": epoch_recall_micro,
        "recall_weighted": epoch_recall_weighted,
        "recall_none": epoch_recall_none,
        "f1-score_macro": epoch_f1_macro,
        "f1-score_micro": epoch_f1_micro,
        "f1-score_weighted": epoch_f1_weighted,
        "f1-score_none": epoch_f1_none,
        "roc_auc_macro": epoch_roc_auc_macro,
        "roc_auc_micro": epoch_roc_auc_micro,
        "roc_auc_weighted": epoch_roc_auc_weighted,
        "roc_auc_none": epoch_roc_auc_none,
    }


def validate(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    verbose: bool = True,
) -> tuple[float, float, float]:
    """Validate a PyTorch model.

    Args:
        model (torch.nn.Module): model to validate
        val_loader (torch.utils.data.DataLoader): data loader for validation data
        criterion (torch.nn.Module): loss function
        device (torch.device): device to validate on

    Returns:
        float: loss
        float: accuracy
        float: f1 score
    """
    model.eval()  # set model to evaluation mode

    # initialize inputs and labels tensors
    inputs: torch.Tensor = torch.empty(0)
    labels: torch.Tensor = torch.empty(0)

    # initialize loss and accuracy
    running_loss: float = 0.0
    running_corrects: int = 0

    # loop over data
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += int(torch.sum(preds == labels.data))

    epoch_loss: float = running_loss / float(len(val_loader.dataset))  # type: ignore
    epoch_acc: float = float(running_corrects) / float(len(val_loader.dataset))  # type: ignore
    epoch_f1: float = sklearn.metrics.f1_score(
        y_true=labels.cpu().numpy(), y_pred=preds.cpu().numpy(), average="weighted"
    )

    if verbose:
        print(
            f"Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}"
        )

    return epoch_loss, epoch_acc, epoch_f1


def save_model(model: torch.nn.Module, path: str) -> None:
    """Save a PyTorch model.

    Args:
        model (torch.nn.Module): model to save
        path (str): path to save model to

    Returns:
        None
    """
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, path: str) -> None:
    """Load a PyTorch model.

    Args:
        model (torch.nn.Module): model to load
        path (str): path to load model from

    Returns:
        None
    """
    model.load_state_dict(torch.load(path))


# %% [markdown]
# # Main

# %%
if __name__ == "__main__":
    pass

# %%
