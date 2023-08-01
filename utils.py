"""Functions for PyTorch utility."""

# %% [markdown]
# # Imports

# %%
import copy
import datetime
import gc
import warnings

import numpy as np
import torch
import torchvision

import config
import data

# %% [markdown]
# # Functions

# %%


def fit(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler
    | torch.optim.lr_scheduler.ReduceLROnPlateau
    | None = None,
    epochs: int = 10000,
    patience: int = 5,
    device: torch.device = torch.device("cpu"),
    save_best: str = "memory",
    verbose: bool = True,
) -> tuple[dict | None, dict]:
    """Fit a PyTorch model.

    Args:
        model (torch.nn.Module): PyTorch model.
        train_loader (torch.utils.data.DataLoader): PyTorch data loader for training.
        test_loader (torch.utils.data.DataLoader): PyTorch data loader for testing.
        loss_fn (torch.nn.Module): PyTorch loss function.
        optimizer (torch.optim.Optimizer): PyTorch optimizer.
        scheduler
                (torch.optim.lr_scheduler.LRScheduler
                    | torch.optim.lr_scheduler.ReduceLROnPlateau
                    | None, optional
                ):
            PyTorch learning rate scheduler. Defaults to None.
        epochs (int, optional): Number of epochs. Defaults to 10000.
        patience (int, optional): Early stopping patience. Defaults to 5.
        device (torch.device, optional): Device to use. Defaults to torch.device("cpu").
        save_best (str, optional): Save best model to "memory" or "disk". Defaults to "memory".
        verbose (bool, optional): Print training progress. Defaults to True.

    Returns:
        tuple[dict | None, dict]: Best model weights (if saved in memory),
            training metrics history.
    """
    # Get optimizer learning rate
    initial_lr = optimizer.param_groups[0]["lr"]
    # If save_best is "disk", set file name
    if save_best == "disk":
        filepath: str = (
            f"{config.MODELS_DIRECTORY}{model.__class__.__name__}_{initial_lr}_best.pt"
        )
    # initialize scheduler
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience // 2, verbose=verbose
        )
    # initialize auto_augment
    auto_augment: torchvision.transforms.AutoAugment | None = (
        torchvision.transforms.AutoAugment()
    )
    if verbose:
        # print training info
        print(f"Start time: {datetime.datetime.now()}")
        print(f"Training {model.__class__.__name__} on {device}.")
        print(f"Training data: {len(train_loader.dataset)}")  # type: ignore
        print(f"Validation data: {len(test_loader.dataset)}")  # type: ignore
        print(f"Loss function: {loss_fn.__class__.__name__}")
        print(f"Optimizer: {optimizer.__class__.__name__}")
        if scheduler is not None:
            print(f"Learning rate scheduler: {scheduler.__class__.__name__}")
        print(f"Patience: {patience}")
        print(f"Epochs: {epochs}")
        print(f"Device: {device}")
    # initialize model
    model.to(device)
    # initialize metrics
    idx: int = 0
    metrics: dict = {
        "train_loss": [],
        "train_loss_sqrt": [],
        "val_loss": [],
        "val_loss_sqrt": [],
    }
    best_val_loss: float = float("inf")
    best_model_wts: dict | None = None
    patience_counter: int = 0
    # loop over epochs
    for epoch in range(epochs):
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} | Time: {datetime.datetime.now()}")
        epoch_training_loss: float = 0.0
        idx = 0
        # Train
        # # initialize inputs and labels tensors
        # inputs: torch.Tensor = torch.empty(0)
        # labels: torch.Tensor = torch.empty(0)
        # # Acumulate labels and predictions for metrics
        # result_labels: torch.Tensor = torch.empty(0)
        # result_preds: torch.Tensor = torch.empty(0)
        model, epoch_training_loss, idx = train(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            verbose,
            auto_augment,
            idx,
            epoch,
        )
        # Append average epoch loss to metrics
        metrics["train_loss"].append(epoch_training_loss / idx)
        metrics["train_loss_sqrt"].append(np.sqrt(epoch_training_loss / idx))
        # free GPU and RAM memory
        torch.cuda.empty_cache()
        gc.collect()
        # Evaluate
        # TODO: Review validation code (output always the same)
        epoch_val_loss: float = 0.0
        idx = 0
        epoch_val_loss, idx, sample_labels, sample_preds = evaluate(
            model,
            test_loader,
            loss_fn,
            device,
            verbose,
            idx,
            epoch,
        )
        # Append average epoch loss to metrics
        metrics["val_loss"].append(epoch_val_loss / idx)
        metrics["val_loss_sqrt"].append(np.sqrt(epoch_val_loss / idx))
        # Print epoch metrics
        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train loss: {metrics['train_loss'][-1]:.4f} "
                f"({metrics['train_loss_sqrt'][-1]:.4f}) | "
                f"Val loss: {metrics['val_loss'][-1]:.4f} | "
                f"({metrics['val_loss_sqrt'][-1]:.4f})"
                f"Sample label {sample_labels} | "
                f"Sample prediction {sample_preds} | "
                f"Sample loss {loss_fn(sample_preds, sample_labels).item():.4f}"
                f"({np.sqrt(loss_fn(sample_preds, sample_labels).item()):.4f})"
            )
        # free GPU memory
        torch.cuda.empty_cache()
        # free memory
        gc.collect()

        # Early stopping
        if metrics["val_loss"][-1] < best_val_loss:
            patience_counter = 0
            best_val_loss = metrics["val_loss"][-1]
            print(
                f"Val loss improved to {best_val_loss:.4f} "
                f"({torch.tensor(best_val_loss).sqrt():.4f})"
            )
            if save_best == "memory":
                # copy model
                best_model_wts = copy.deepcopy(model.state_dict())
            elif save_best == "disk":
                # save model
                torch.save(model.state_dict(), filepath)
            else:
                # warn and do nothing
                warnings.warn(
                    f"save_best={save_best} is not a valid option. "
                    f"Model will not be saved."
                )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(
                        f"Early stopping. "
                        f"Val loss did not improve for {patience} epochs."
                    )
                break
        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss / idx)
            else:
                scheduler.step()

    # Print final metrics
    if verbose:
        print(f"Training complete. " f"Best val loss: {best_val_loss:.4f}")
        print(f"End time: {datetime.datetime.now()}")

    return best_model_wts, metrics


def evaluate(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device | str = torch.device("cpu"),
    verbose: bool = True,
    idx: int = 0,
    epoch: int = 0,
    random_sample_threshold: float = 0.1,
) -> tuple[float, int, torch.Tensor, torch.Tensor]:
    """Evaluate model on test set.

    Args:
        model (torch.nn.Module): Model to evaluate.
        test_loader (torch.utils.data.DataLoader): Test set data loader.
        loss_fn (torch.nn.Module): Loss function.
        device (torch.device | str, optional): Device to use. Defaults to torch.device("cpu").
        verbose (bool, optional): Print metrics. Defaults to True.
        idx (int, optional): Number of samples evaluated. Defaults to 0.
        epoch (int, optional): Current epoch. Defaults to 0.
        random_sample_threshold (float, optional): Probability of randomly sampling a batch. Defaults to 0.1.

    Returns:
        float: Average loss.
        int: Number of samples evaluated.
        torch.Tensor: Sample Labels.
        torch.Tensor: Sample Predictions.
    """
    epoch_val_loss: float = 0.0
    sample_labels: torch.Tensor = torch.empty(0)
    sample_preds: torch.Tensor = torch.empty(0)
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            # free GPU and RAM memory
            torch.cuda.empty_cache()
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
                ).float()
                # remove stride frames from inputs
                inputs = inputs[:, config.STRIDE :, :, :].float()
                # Increment number of samples
                idx += 1

                # send input and labels to device
                inputs_first = inputs_first.float().to(device)
                labels = labels.float().to(device)
                # forward pass
                outputs = model(inputs_first)

                # if first batch, initialize sample_labels and sample_preds
                if idx == 1:
                    sample_labels = labels
                    sample_preds = outputs
                else:
                    # randomly select a sample to keep
                    if np.random.rand() < random_sample_threshold:
                        sample_labels = labels.to("cpu").detach().float()
                        sample_preds = outputs.to("cpu").detach().float()
                # backward pass
                loss = loss_fn(outputs, labels)
                # Gather data and report
                epoch_val_loss += loss.item()
                if verbose and idx % 1000 == 999:
                    print(
                        f"Validation | "
                        f"Time {datetime.datetime.now()} | "
                        f"Epoch {epoch+1:5d} | "
                        f"Batch {i+1:5d} | "
                        f"Avg batch loss {epoch_val_loss / (i+1):.4f} | "
                        f"({np.sqrt(epoch_val_loss / (i+1)):.4f}) | "
                        f"Avg sample loss {epoch_val_loss / idx:.4f} |"
                        f"({np.sqrt(epoch_val_loss / idx):.4f}) | "
                        f"Sample label {sample_labels} | "
                        f"Sample prediction {sample_preds} | "
                        f"Sample loss {loss_fn(sample_preds, sample_labels).item():.4f} "
                        f"({torch.tensor(loss_fn(sample_preds, sample_labels).item()).sqrt():.4f})"
                    )
                    # free GPU and RAM memory
                del inputs_first
                torch.cuda.empty_cache()
                gc.collect()
    return epoch_val_loss, idx, sample_labels, sample_preds


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = torch.device("cpu"),
    verbose: bool = True,
    auto_augment: torchvision.transforms.AutoAugment | None = None,
    idx: int = 0,
    epoch: int = 0,
) -> tuple[torch.nn.Module, float, int]:
    """Train loop for a single epoch.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (torch.utils.data.DataLoader): Training set data loader.
        loss_fn (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device | str, optional): Device to use. Defaults to torch.device("cpu").
        verbose (bool, optional): Print metrics. Defaults to True.
        auto_augment (torchvision.transforms.AutoAugment | None, optional): Autoaugment transform. Defaults to None.
        idx (int, optional): Number of samples evaluated. Defaults to 0.
        epoch (int, optional): Current epoch. Defaults to 0.

    Returns:
        torch.nn.Module: Trained model.
        float: Average loss.
        int: Number of samples evaluated.
    """
    epoch_training_loss: float = 0.0
    model.train()  # set model to training mode
    for i, (inputs, labels) in enumerate(train_loader):
        # free GPU and RAM memory
        torch.cuda.empty_cache()
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
            ).float()
            # remove stride frames from inputs
            inputs = inputs[:, config.STRIDE :, :, :].float()

            # Autoaugment and send to device
            inputs_augmented: list[torch.Tensor] | torch.Tensor = [
                inputs_first.to(torch.uint8)
            ]
            if auto_augment is not None:
                # squeeze batch dimension
                inputs_first = inputs_first.squeeze(0)
                # invert dimensions to use torchvision transforms
                # permute dimensions
                inputs_first = inputs_first.permute(1, 0, 2, 3)
                # create N augmented sequences and stack them
                for _ in range(config.AUTOAUGMENT_N):
                    inputs_augmented.append(  # type: ignore
                        auto_augment(inputs_first.to(torch.uint8))
                        .permute(1, 0, 2, 3)
                        .unsqueeze(0)
                    )
                    # add batch dimension
                inputs_augmented = torch.cat(inputs_augmented)  # type: ignore
                # permute back
                # inputs_augmented = inputs_augmented.permute(0, 2, 1, 3, 4)
                # send to device
                inputs_augmented = inputs_augmented.float().to(device)
            else:
                # send to device
                inputs_augmented = inputs_first.float().to(device)

                # Increment number of samples
                # idx += inputs_augmented.shape[0]
            idx += 1

            # expand labels to comply with inputs_augmented shape
            labels = labels.expand(inputs_augmented.shape[0], -1)
            # send labels to device
            labels = labels.float().to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass
            outputs = model(inputs_augmented)

            # backward pass
            loss = loss_fn(outputs, labels)
            loss.backward()
            # optimize
            optimizer.step()

            # Gather data and report
            epoch_training_loss += loss.item()
            if verbose and idx % 1000 == 999:
                print(
                    f"Training | "
                    f"Time {datetime.datetime.now()} | "
                    f"Epoch {epoch+1:5d} | "
                    f"Batch {i+1:5d} | "
                    f"Avg batch loss {epoch_training_loss / (i+1):.4f} "
                    f"({np.sqrt(epoch_training_loss / (i+1)):.4f}) | "
                    f"Avg sample loss {epoch_training_loss / idx:.4f} "
                    f"({np.sqrt(epoch_training_loss / idx):.4f}) | "
                    f"Sample label {labels} | "
                    f"Sample prediction {outputs} | "
                    f"Sample loss {loss.item():.4f} "
                    f"({torch.tensor(loss.item()).sqrt():.4f})"
                )
                # free GPU and RAM memory
            del inputs_first
            torch.cuda.empty_cache()
            gc.collect()
    return model, epoch_training_loss, idx


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
