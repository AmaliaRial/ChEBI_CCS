import argparse
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )


def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "models" not in data:
        raise ValueError("Config must contain 'models' mapping.")
    if "external_root" not in data:
        data["external_root"] = ""
    return data


def _resolve_repo_path(external_root: str, repo_path: str) -> Path:
    repo_candidate = Path(repo_path)
    if repo_candidate.is_absolute() or repo_path == ".":
        return repo_candidate
    return Path(external_root) / repo_path


def _build_extra_wrapper_args(info: Dict[str, Any]) -> List[str]:
    extra_args: List[str] = []
    wrapper_args = info.get("wrapper_args", {})
    if not isinstance(wrapper_args, dict):
        raise ValueError("wrapper_args must be a dict.")

    for key, value in wrapper_args.items():
        arg_name = "--{}".format(str(key).replace("_", "-"))
        if isinstance(value, bool):
            if value:
                extra_args.append(arg_name)
        elif value is not None:
            extra_args.extend([arg_name, str(value)])

    return extra_args


def _run_model(
    wrapper_path: Path,
    conda_env: str,
    input_csv: Path,
    output_csv: Path,
    repo_path: Path,
    extra_wrapper_args: List[str],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        str(wrapper_path),
        "--input",
        str(input_csv),
        "--output",
        str(output_csv),
        "--repo",
        str(repo_path),
    ] + extra_wrapper_args
    logging.info("Running: %s", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CCS benchmark.")
    parser.add_argument("--input", required=True, help="Input CSV with smiles/adduct/ccs.")
    parser.add_argument("--output-dir", default="predictions", help="Output directory.")
    parser.add_argument("--config", default="configs/benchmark_models.yaml", help="Config YAML.")
    parser.add_argument("--external-root", default=None, help="Override external_root.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = _load_config(config_path)
    external_root = args.external_root or config.get("external_root", "")

    wrapper_root = Path("scripts/wrappers")
    input_csv = Path(args.input)
    output_dir = Path(args.output_dir)

    enabled_models = [
        (name, info)
        for name, info in config["models"].items()
        if isinstance(info, dict) and info.get("enabled")
    ]
    if not enabled_models:
        raise SystemExit("No enabled models. Set enabled: true in config.")

    _setup_logging(Path("logs/benchmark_run.log"))

    for name, info in enabled_models:
        wrapper_name = info.get("wrapper", name)
        conda_env = info.get("conda_env")
        repo_path = _resolve_repo_path(external_root, info.get("repo_path", ""))
        wrapper_path = wrapper_root / "{}.py".format(wrapper_name)
        output_csv = output_dir / name / "predictions.csv"
        extra_wrapper_args = _build_extra_wrapper_args(info)

        if not conda_env:
            raise ValueError("Missing conda_env for '{}'".format(name))
        if not wrapper_path.exists():
            raise FileNotFoundError("Wrapper not found: {}".format(wrapper_path))

        _run_model(wrapper_path, conda_env, input_csv, output_csv, repo_path, extra_wrapper_args)


if __name__ == "__main__":
    main()

