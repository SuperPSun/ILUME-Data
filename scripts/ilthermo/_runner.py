"""Run one ILThermo property structuring job through a property-local processor.

Each single-property script in this directory imports this module and calls
main_for() with the relevant ILThermo property slug and processor.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from collections.abc import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raw_prep import ILThermoPropertySpec, disable_rdkit_logs, get_spec_by_input_filename, get_spec_by_slug, raw_root, structured_root


def parse_args(property_slug: str) -> argparse.Namespace:
    """Parse CLI arguments for one ILThermo property processor."""

    spec = get_spec_by_slug(property_slug)
    parser = argparse.ArgumentParser(
        description=f"Structure the ILThermo raw text file for the {property_slug} property and write a cleaned CSV."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=raw_root(PROJECT_ROOT) / "ILThermo",
        help=f"Path to the directory that contains {spec.input_filename}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=structured_root(PROJECT_ROOT) / "ILThermo",
        help=f"Path to the directory where {spec.output_filename} will be written.",
    )
    return parser.parse_args()


ILThermoProcessor = Callable[[Path, Path], object]


def _module_name_for_slug(property_slug: str) -> str:
    """Map a property slug to its local module name."""

    return f"structuring_ilt_{property_slug}"


def get_processor_by_slug(property_slug: str) -> tuple[ILThermoPropertySpec, ILThermoProcessor]:
    """Import and return the processor callable for one property slug."""

    spec = get_spec_by_slug(property_slug)
    module = importlib.import_module(_module_name_for_slug(property_slug))
    processor = getattr(module, "process_file", None)
    if processor is None:
        raise AttributeError(f"Module {_module_name_for_slug(property_slug)} does not export process_file")
    return spec, processor


def get_processor_by_input_filename(input_filename: str) -> tuple[ILThermoPropertySpec, ILThermoProcessor]:
    """Resolve a raw input filename to its property spec and processor."""

    spec = get_spec_by_input_filename(input_filename)
    _, processor = get_processor_by_slug(spec.slug)
    return spec, processor


def main_for(property_slug: str, processor: ILThermoProcessor | None = None) -> None:
    """Execute one ILThermo property processor from its dedicated wrapper script."""

    disable_rdkit_logs()
    args = parse_args(property_slug)
    spec = get_spec_by_slug(property_slug)
    if processor is None:
        _, processor = get_processor_by_slug(property_slug)
    processor(args.input_dir / spec.input_filename, args.output_dir / spec.output_filename)