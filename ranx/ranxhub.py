import yaml

from .data_structures import Run
from .io import save_lz4


def save(run: Run, runcard_path: str, path: str) -> None:
    with open(runcard_path, "r") as f:
        runcard = yaml.load(f, Loader=yaml.Loader)

    content = {"metadata": runcard, "run": run.to_dict()}

    with open(runcard_path, "w") as f:
        runcard = f.write(yaml.dump(runcard, sort_keys=False))

    save_lz4(content, path)
