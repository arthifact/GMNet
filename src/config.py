"""config.toml, read by every step."""
from pathlib import Path
import tomllib


def load(path='config.toml'):
    """The config as a dict, plus path (the file) and run (outputs/<name>)."""
    config = tomllib.loads(Path(path).read_text())
    config['path'], config['run'] = Path(path), Path('outputs') / config['name']
    return config
