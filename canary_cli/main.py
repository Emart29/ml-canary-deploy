import sys
from pathlib import Path

# Ensure the project root is importable when invoked as the installed `canary` script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.main import cli

__all__ = ["cli"]

if __name__ == "__main__":
    cli()
