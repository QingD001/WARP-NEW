"""Generate comparable configs without starting paid experiments.

Run: python scripts/make_method_ablations.py --config configs/paper/hotpotqa.yaml --output-dir configs/ablations/hotpotqa
"""
import argparse
from copy import deepcopy
from itertools import product
from pathlib import Path
import yaml


def generate(config, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for objective, selection, steps in product(
        ("evidence_recall", "complete_evidence", "mixed"), ("independent", "conditional"), (1, 2)):
        variant = deepcopy(config)
        variant["warp"].update(benefit_objective=objective, selection_mode=selection, retrieval_steps=steps)
        # Partition ablations multiply cost and do not identify these three effects.
        variant["experiment"]["partition_ablations"]["modes"] = []
        tag = f"{objective}-{selection}-{steps}step"
        # Graph indexes may be shared: builder keys protect the graph content/config.
        # Run output/checkpoints must remain separate for each configuration.
        path = output_dir / f"{tag}.yaml"
        path.write_text(yaml.safe_dump(variant, allow_unicode=True, sort_keys=False), encoding="utf-8")
        paths.append(str(path))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for path in generate(config, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
