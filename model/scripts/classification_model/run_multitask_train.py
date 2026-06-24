from pathlib import Path
import argparse

from model.chebi_model import train_model


REPO_ROOT = Path(__file__).resolve().parents[3]  # Up to project root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CCS + ChEBI multitask model.")
    parser.add_argument("--train-input", default=str(REPO_ROOT / "predictions" / "base" / "train_split.csv"), help="Training CSV with fingerprints, adduct, mz and CCS.")
    parser.add_argument("--val-input", default=str(REPO_ROOT / "predictions" / "base" / "val_split.csv"), help="Validation CSV with the same schema as the training split.")
    parser.add_argument("--test-input", default=str(REPO_ROOT / "predictions" / "base" / "test_split.csv"), help="Test CSV with the same schema as the training split.")
    parser.add_argument("--ontology-input",default=str(REPO_ROOT / "data" / "model" / "final_covered_ccs_fingerprints_multilabel_filtered.csv"), help="Multilabel ontology CSV generated from the ChEBI preprocessing pipeline.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "predictions" / "ontology_model"), help="Output directory for the multitask model artifacts.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-ontology", type=float, default=0.1)
    parser.add_argument("--ontology-threshold", type=float, default=0.5)
    parser.add_argument("--row-id-column", default="row_id")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Training device.")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=None, help="Neurons per shared hidden layer (defaults to 1024 256 64).")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for the shared trunk.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        train_csv=args.train_input,
        output_dir=args.output_dir,
        val_csv=args.val_input,
        test_csv=args.test_input,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_ontology=args.lambda_ontology,
        ontology_threshold=args.ontology_threshold,
        ontology_csv=args.ontology_input,
        row_id_column=args.row_id_column,
        device=args.device,
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout,
    )
