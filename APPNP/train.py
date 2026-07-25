import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from networks import Net
from torch.optim import Adam
from torch_geometric import datasets
from tqdm import tqdm
from utils import coarsen_multiple_subgraphs, load_and_coarsen

DATASET_ROOT = Path(__file__).resolve().parent / "dataset"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument(
        "--split-type",
        "--split_type",
        dest="split_type",
        choices=("fixed", "random", "few"),
        default="fixed",
    )
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--early-stopping",
        "--early_stopping",
        dest="early_stopping",
        type=int,
        default=10,
    )
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--weight-decay",
        "--weight_decay",
        dest="weight_decay",
        type=float,
        default=0.0005,
    )
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--normalize-features",
        "--normalize_features",
        dest="normalize_features",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--coarsening-ratio",
        "--coarsening_ratio",
        dest="coarsening_ratio",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--coarsening-method",
        "--coarsening_method",
        dest="coarsening_method",
        type=str,
        default="variation_neighborhoods",
    )
    parser.add_argument("--plot-path", type=Path)
    return parser.parse_args()


def load_dataset(name):
    canonical_names = {
        "cora": "Cora",
        "citeseer": "CiteSeer",
        "pubmed": "PubMed",
        "cora_ml": "Cora_ML",
        "dblp": "DBLP",
        "cs": "CS",
        "physics": "Physics",
    }
    canonical_name = canonical_names.get(name.lower())
    if canonical_name is None:
        supported = ", ".join(canonical_names.values())
        raise ValueError(f"Unsupported dataset {name!r}. Choose one of: {supported}.")

    if canonical_name in {"Cora_ML", "DBLP"}:
        root = DATASET_ROOT / canonical_name.lower()
        return datasets.CitationFull(str(root), name=canonical_name)
    if canonical_name in {"CS", "Physics"}:
        root = DATASET_ROOT / canonical_name
        return datasets.Coauthor(str(root), name=canonical_name)

    root = DATASET_ROOT / canonical_name.lower()
    return datasets.Planetoid(str(root), name=canonical_name)


def train(
    args,
    model,
    coarsen_features,
    coarsen_train_labels,
    coarsen_train_mask,
    coarsen_val_labels,
    coarsen_val_mask,
    coarsen_edge,
    optimizer,
):
    history = {"train_loss": [], "val_accuracy": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for _ in tqdm(range(args.epochs), desc="Training epochs"):
        model.train()
        optimizer.zero_grad()
        output = model(coarsen_features, coarsen_edge)
        loss = F.nll_loss(
            output[coarsen_train_mask], coarsen_train_labels[coarsen_train_mask]
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output = model(coarsen_features, coarsen_edge)
            val_loss = F.nll_loss(
                output[coarsen_val_mask], coarsen_val_labels[coarsen_val_mask]
            )
            predictions = output.argmax(dim=1)
            val_accuracy = (
                predictions[coarsen_val_mask]
                .eq(coarsen_val_labels[coarsen_val_mask])
                .float()
                .mean()
            )

        current_val_loss = val_loss.item()
        history["train_loss"].append(loss.item())
        history["val_accuracy"].append(val_accuracy.item())
        history["val_loss"].append(current_val_loss)

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            args.early_stopping > 0
            and epochs_without_improvement >= args.early_stopping
        ):
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")
    model.load_state_dict(best_state)
    return history


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index).argmax(dim=1)
    return predictions[data.test_mask].eq(data.y[data.test_mask]).float().mean().item()


def plot_history(history, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    axis.plot(history["train_loss"], label="Train loss")
    axis.plot(history["val_accuracy"], label="Validation accuracy")
    axis.plot(history["val_loss"], label="Validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Metric")
    axis.legend()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    if not 0 <= args.coarsening_ratio < 1:
        raise ValueError("--coarsening-ratio must be in [0, 1).")
    if args.runs < 1:
        raise ValueError("--runs must be at least 1.")

    dataset = load_dataset(args.dataset)
    if len(dataset) != 1:
        raise ValueError(
            "Only node-classification datasets containing one graph are supported."
        )

    data = dataset[0]
    args.num_features = data.x.size(1)
    args.num_classes = int(data.y.max().item()) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subgraphs, coarsening_matrices, coarsened_subgraphs = coarsen_multiple_subgraphs(
        data,
        args.coarsening_ratio,
        args.coarsening_method,
    )

    test_accuracies = []
    last_history = None
    for run in range(args.runs):
        run_data = data.clone()
        coarsened_data = load_and_coarsen(
            run_data,
            subgraphs,
            coarsening_matrices,
            coarsened_subgraphs,
            args.split_type,
        )
        (
            run_data,
            coarsen_features,
            coarsen_train_labels,
            coarsen_train_mask,
            coarsen_val_labels,
            coarsen_val_mask,
            coarsen_edge,
        ) = (value.to(device) for value in coarsened_data)

        if args.normalize_features:
            coarsen_features = F.normalize(coarsen_features, p=1)
            run_data.x = F.normalize(run_data.x, p=1)

        model = Net(args).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        last_history = train(
            args,
            model,
            coarsen_features,
            coarsen_train_labels,
            coarsen_train_mask,
            coarsen_val_labels,
            coarsen_val_mask,
            coarsen_edge,
            optimizer,
        )
        test_accuracy = evaluate(model, run_data)
        test_accuracies.append(test_accuracy)
        print(f"run {run + 1:02d}: test accuracy {test_accuracy:.4f}")

    print(
        f"average accuracy: {np.mean(test_accuracies):.4f} "
        f"+/- {np.std(test_accuracies):.4f}"
    )
    if args.plot_path is not None:
        plot_history(last_history, args.plot_path)


if __name__ == "__main__":
    main()
