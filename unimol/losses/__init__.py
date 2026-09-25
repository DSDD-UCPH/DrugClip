from pathlib import Path
import importlib

for file in sorted(Path(__file__).parent.glob("*.py")):
    stem = file.stem
    if not stem or stem.startswith("_") or not stem.isidentifier():
        continue
    importlib.import_module("unimol.losses." + stem)
