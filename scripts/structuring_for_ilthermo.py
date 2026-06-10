"""Structure all raw ILThermo crawler CSV inputs into cleaned CSV outputs.

This batch entrypoint iterates over every supported ILThermo property and
dispatches to the local processor exported by each scripts/ilthermo module.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
ILTHERMO_SCRIPTS_ROOT = PROJECT_ROOT / "scripts" / "ilthermo"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(ILTHERMO_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(ILTHERMO_SCRIPTS_ROOT))

from raw_prep import PROPERTY_SPECS, disable_rdkit_logs, raw_root, structured_root

from _runner import get_processor_by_input_filename


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the ILThermo batch structuring entrypoint."""

    parser = argparse.ArgumentParser(
        description="Structure the raw ILThermo crawler CSV inputs into cleaned CSV files for every supported property."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=raw_root(PROJECT_ROOT) / "ILThermo",
        help="Path to the directory that contains the crawler CSV inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=structured_root(PROJECT_ROOT) / "ILThermo",
        help="Path to the directory where the structured ILThermo CSV files will be written.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=[spec.input_filename for spec in PROPERTY_SPECS],
        help="Optional subset of crawler CSV filenames to process instead of the full supported set.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the selected ILThermo property processors against the requested files."""

    disable_rdkit_logs()
    args = parse_args()
    for input_name in args.files:
        spec, processor = get_processor_by_input_filename(input_name)
        processor(
            args.input_dir / spec.input_filename,
            args.output_dir / spec.output_filename,
        )


if __name__ == "__main__":
    main()