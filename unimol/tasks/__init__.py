from pathlib import Path
import importlib

# Import task modules so @register_task side effects run. Skip __init__,
# private modules (_*), and stray names like ".py" that would yield
# "unimol.tasks." and crash importlib.
for file in sorted(Path(__file__).parent.glob("*.py")):
    stem = file.stem
    if not stem or stem.startswith("_") or not stem.isidentifier():
        continue
    importlib.import_module("unimol.tasks." + stem)
