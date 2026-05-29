import json
from pathlib import Path

import yaml


def load_structured_file(path: str, what: str) -> object:
    """Load a YAML or JSON config file, raising clear errors on failure.

    `what` names the config for error messages (e.g. "custom recognizers").
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{what} config not found: {path}")

    raw = file_path.read_text()
    suffix = file_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            return yaml.safe_load(raw)
        except yaml.YAMLError as e:
            raise ValueError(f"{what} config is not valid YAML ({path}): {e}") from e
    if suffix == ".json":
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"{what} config is not valid JSON ({path}): {e}") from e
    raise ValueError(
        f"{what} config has unsupported extension {file_path.suffix!r} ({path}); "
        "use .yaml, .yml, or .json"
    )
