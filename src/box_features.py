"""Extract scalar structural fingerprints from ionic-liquid PDB boxes.

The input PDB files contain one orthorhombic, periodic snapshot with explicit
hydrogens and POS/NEG residues.  This module deliberately treats the result as
an instantaneous finite-box fingerprint, not as a trajectory average.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial import cKDTree

FEATURE_VERSION = "3d-box-v1"
RDF_DR_A = 0.1
RDF_MAX_A = 20.0
RDF_KEYS = ("cc", "ca", "aa", "pp", "pn", "nn")
RDF_METRICS = (
    "peak1_r_A",
    "peak1_g",
    "minimum1_r_A",
    "coordination1",
    "fwhm_A",
    "excess_area_A",
)
HB_DIRECTIONS = ("cat_to_cat", "cat_to_an", "an_to_cat", "an_to_an")


def _feature_columns() -> list[str]:
    columns = [f"rdf_{key}_{metric}" for key in RDF_KEYS for metric in RDF_METRICS]
    columns.extend(
        f"hb_{direction}_{kind}_count"
        for direction in HB_DIRECTIONS
        for kind in ("strong", "weak_ch")
    )
    columns.extend(
        [
            "hb_total_count",
            "hb_total_per_ion_pair",
            "hb_strict_total_count",
            "hb_loose_total_count",
            "hb_donor_utilization_fraction",
            "hb_acceptor_occupancy_fraction",
            "hb_HA_distance_mean_A",
            "hb_HA_distance_std_A",
            "hb_DHA_angle_mean_deg",
            "hb_DHA_angle_std_deg",
            "hb_network_mean_degree",
            "hb_network_isolated_fraction",
            "hb_network_largest_component_fraction",
            "domain_polar_heavy_fraction",
            "domain_nonpolar_heavy_fraction",
            "scc_prepeak_q_A^-1",
            "scc_prepeak_height",
            "scc_prepeak_fwhm_A^-1",
            "scc_prepeak_area_A^-1",
            "scc_domain_spacing_A",
            "szz_peak_q_A^-1",
            "szz_peak_height",
            "szz_peak_fwhm_A^-1",
            "szz_peak_area_A^-1",
        ]
    )
    return columns


FEATURE_COLUMNS = tuple(_feature_columns())
QC_COLUMNS = (
    "qc_status",
    "qc_flags",
    "qc_error",
    "qc_pdb_present",
    "qc_topology_ok",
    "qc_cation_mol2_match",
    "qc_anion_mol2_match",
    "qc_box_a_A",
    "qc_box_b_A",
    "qc_box_c_A",
    "qc_box_volume_A^3",
    "qc_cation_count",
    "qc_anion_count",
    "qc_atom_count",
    "qc_polar_heavy_count",
    "qc_nonpolar_heavy_count",
)


@dataclass(frozen=True)
class Residue:
    """One POS or NEG residue from a PDB snapshot."""

    side: str
    residue_id: str
    atom_names: tuple[str, ...]
    elements: tuple[str, ...]
    positions: np.ndarray


@dataclass(frozen=True)
class PDBBox:
    """Parsed orthorhombic PDB box."""

    lengths: np.ndarray
    residues: tuple[Residue, ...]
    atom_count: int


@dataclass(frozen=True)
class Mol2Info:
    """Minimal MOL2 metadata used only for topology cross-checking."""

    path: Path
    atom_count: int
    net_charge: float


@dataclass(frozen=True)
class ReferenceTopology:
    """SMILES-derived atom annotations in RDKit atom order."""

    graph: nx.Graph
    elements: tuple[str, ...]
    masses: np.ndarray
    acceptor: np.ndarray
    polar_heavy: np.ndarray
    donor_by_hydrogen: dict[int, tuple[int, str]]
    net_charge: int
    mol2_match: bool | None


@dataclass(frozen=True)
class ResidueTemplate:
    """Topology annotations reordered to match atoms in a PDB residue."""

    elements: tuple[str, ...]
    masses: np.ndarray
    acceptor: np.ndarray
    polar_heavy: np.ndarray
    donor_by_hydrogen: dict[int, tuple[int, str]]
    topology_ok: bool
    mol2_match: bool | None


@dataclass(frozen=True)
class RDFResult:
    r: np.ndarray
    g: np.ndarray
    density_around_reference: float


_FEATURE_FACTORY: Any | None = None


def _feature_factory() -> Any:
    global _FEATURE_FACTORY
    if _FEATURE_FACTORY is None:
        path = str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
        _FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(path)
    return _FEATURE_FACTORY


def empty_feature_row() -> dict[str, Any]:
    """Return a stable empty output schema for failed boxes."""

    row: dict[str, Any] = {column: math.nan for column in FEATURE_COLUMNS}
    row.update({column: math.nan for column in QC_COLUMNS})
    row["qc_status"] = "failed"
    row["qc_flags"] = ""
    row["qc_error"] = ""
    row["qc_pdb_present"] = False
    return row


def element_from_pdb_name(atom_name: str, element_field: str = "") -> str:
    """Recover an element when the PDB element column is empty."""

    field = element_field.strip()
    if field:
        return field[0].upper() + field[1:].lower()
    name = "".join(character for character in atom_name.strip() if character.isalpha())
    if not name:
        raise ValueError(f"cannot infer element from atom name {atom_name!r}")
    upper = name.upper()
    if upper.startswith("CL"):
        return "Cl"
    if upper.startswith("BR"):
        return "Br"
    return upper[0]


def parse_pdb_box(path: Path) -> PDBBox:
    """Parse one explicit-hydrogen, orthorhombic POS/NEG PDB snapshot."""

    lengths: np.ndarray | None = None
    angles: tuple[float, float, float] | None = None
    model_count = 0
    end_model_count = 0
    records: dict[tuple[str, str, str], list[tuple[str, str, np.ndarray]]] = {}

    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("CRYST1"):
                lengths = np.asarray(
                    [float(line[6:15]), float(line[15:24]), float(line[24:33])], dtype=float
                )
                angles = (float(line[33:40]), float(line[40:47]), float(line[47:54]))
            elif line.startswith("MODEL"):
                model_count += 1
            elif line.startswith("ENDMDL"):
                end_model_count += 1
            elif line.startswith(("ATOM  ", "HETATM")):
                residue_name = line[17:20].strip().upper()
                if residue_name not in {"POS", "NEG"}:
                    continue
                side = "cat" if residue_name == "POS" else "an"
                key = (side, line[21:22], line[22:26].strip())
                atom_name = line[12:16].strip()
                element = element_from_pdb_name(atom_name, line[76:78] if len(line) >= 78 else "")
                position = np.asarray(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float
                )
                records.setdefault(key, []).append((atom_name, element, position))

    if lengths is None or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("missing or invalid CRYST1 lengths")
    if angles != (90.0, 90.0, 90.0):
        raise ValueError(f"only orthorhombic boxes are supported, got angles={angles}")
    if model_count != 1 or end_model_count != 1:
        raise ValueError(f"expected one MODEL/ENDMDL, got {model_count}/{end_model_count}")
    if not records:
        raise ValueError("PDB contains no POS/NEG atoms")

    residues: list[Residue] = []
    for (side, chain, sequence), atoms in records.items():
        residues.append(
            Residue(
                side=side,
                residue_id=f"{side}:{chain}:{sequence}",
                atom_names=tuple(atom[0] for atom in atoms),
                elements=tuple(atom[1] for atom in atoms),
                positions=np.vstack([atom[2] for atom in atoms]),
            )
        )
    cation_count = sum(residue.side == "cat" for residue in residues)
    anion_count = sum(residue.side == "an" for residue in residues)
    if cation_count != anion_count:
        raise ValueError(f"cation/anion count mismatch: {cation_count}/{anion_count}")
    return PDBBox(lengths=lengths, residues=tuple(residues), atom_count=sum(len(r.elements) for r in residues))


def parse_mol2_info(path: Path) -> Mol2Info:
    """Read atom count and summed partial charge from a MOL2 file."""

    atom_count = 0
    charges: list[float] = []
    section = ""
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("@<TRIPOS>"):
                section = line.strip()
                continue
            if section == "@<TRIPOS>ATOM" and line.strip():
                fields = line.split()
                if len(fields) >= 9:
                    atom_count += 1
                    charges.append(float(fields[8]))
    if atom_count == 0:
        raise ValueError(f"MOL2 contains no atoms: {path}")
    return Mol2Info(path=Path(path), atom_count=atom_count, net_charge=float(sum(charges)))


def build_reference_topology(smiles: str, mol2_info: Mol2Info | None = None) -> ReferenceTopology:
    """Build donor, acceptor and polar annotations from an ion SMILES."""

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    molecule = Chem.AddHs(molecule)
    graph = nx.Graph()
    elements = tuple(atom.GetSymbol() for atom in molecule.GetAtoms())
    periodic_table = Chem.GetPeriodicTable()
    masses = np.asarray([periodic_table.GetAtomicWeight(element) for element in elements], dtype=float)
    for atom in molecule.GetAtoms():
        graph.add_node(atom.GetIdx(), element=atom.GetSymbol())
    for bond in molecule.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    acceptor_ids: set[int] = set()
    for feature in _feature_factory().GetFeaturesForMol(molecule):
        if feature.GetFamily() == "Acceptor":
            acceptor_ids.update(feature.GetAtomIds())
    net_charge = sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
    if net_charge < 0:
        acceptor_ids.update(
            atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetSymbol() in {"F", "Cl", "Br"}
        )
    acceptor = np.asarray([index in acceptor_ids for index in range(molecule.GetNumAtoms())], dtype=bool)

    positive_ring_atoms: set[int] = set()
    ring_info = molecule.GetRingInfo()
    positive_atoms = {atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetFormalCharge() > 0}
    for ring in ring_info.AtomRings():
        if positive_atoms.intersection(ring):
            positive_ring_atoms.update(ring)

    donor_by_hydrogen: dict[int, tuple[int, str]] = {}
    for atom in molecule.GetAtoms():
        if atom.GetSymbol() == "H" or not any(neighbor.GetSymbol() == "H" for neighbor in atom.GetNeighbors()):
            continue
        if atom.GetSymbol() in {"N", "O", "S"}:
            donor_kind = "strong"
        elif atom.GetSymbol() == "C" and (
            atom.GetIdx() in positive_ring_atoms
            or any(neighbor.GetFormalCharge() > 0 for neighbor in atom.GetNeighbors())
        ):
            donor_kind = "weak_ch"
        else:
            continue
        for hydrogen in atom.GetNeighbors():
            if hydrogen.GetSymbol() == "H":
                donor_by_hydrogen[hydrogen.GetIdx()] = (atom.GetIdx(), donor_kind)

    perfluoro_atoms: set[int] = set()
    for atom in molecule.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        fluorines = [neighbor for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == "F"]
        if len(fluorines) >= 2:
            perfluoro_atoms.add(atom.GetIdx())
            perfluoro_atoms.update(neighbor.GetIdx() for neighbor in fluorines)

    polar_ids: set[int] = set()
    for atom in molecule.GetAtoms():
        standalone_anionic_halogen = (
            atom.GetSymbol() in {"F", "Cl", "Br"} and atom.GetDegree() == 0 and net_charge < 0
        )
        if atom.GetFormalCharge() or atom.GetSymbol() in {"N", "O", "S", "P", "B"} or standalone_anionic_halogen:
            polar_ids.add(atom.GetIdx())
    polar_ids.update(positive_ring_atoms)
    for atom_index in tuple(polar_ids):
        polar_ids.update(neighbor.GetIdx() for neighbor in molecule.GetAtomWithIdx(atom_index).GetNeighbors())
    polar_ids.difference_update(perfluoro_atoms)
    polar_heavy = np.asarray(
        [
            element != "H" and index in polar_ids
            for index, element in enumerate(elements)
        ],
        dtype=bool,
    )

    mol2_match: bool | None = None
    if mol2_info is not None:
        mol2_match = mol2_info.atom_count == molecule.GetNumAtoms() and abs(mol2_info.net_charge - net_charge) <= 0.15
    return ReferenceTopology(
        graph=graph,
        elements=elements,
        masses=masses,
        acceptor=acceptor,
        polar_heavy=polar_heavy,
        donor_by_hydrogen=donor_by_hydrogen,
        net_charge=net_charge,
        mol2_match=mol2_match,
    )


def infer_coordinate_graph(elements: tuple[str, ...], positions: np.ndarray) -> nx.Graph:
    """Infer residue connectivity from covalent distances."""

    graph = nx.Graph()
    for index, element in enumerate(elements):
        graph.add_node(index, element=element)
    periodic_table = Chem.GetPeriodicTable()
    hydrogen_indices = [index for index, element in enumerate(elements) if element == "H"]
    heavy_indices = [index for index, element in enumerate(elements) if element != "H"]

    for offset, left in enumerate(heavy_indices):
        radius_left = periodic_table.GetRcovalent(elements[left])
        for right in heavy_indices[offset + 1 :]:
            radius_right = periodic_table.GetRcovalent(elements[right])
            if elements[left] in {"F", "Cl", "Br"} and elements[right] in {"F", "Cl", "Br"}:
                continue
            cutoff = 1.15 * (radius_left + radius_right) + 0.10
            distance = float(np.linalg.norm(positions[left] - positions[right]))
            if 0.4 < distance <= cutoff:
                graph.add_edge(left, right)
    for hydrogen in hydrogen_indices:
        if not heavy_indices:
            continue
        distances = np.linalg.norm(positions[heavy_indices] - positions[hydrogen], axis=1)
        nearest_offset = int(np.argmin(distances))
        nearest = heavy_indices[nearest_offset]
        cutoff = 1.55 if elements[nearest] == "S" else 1.35
        if distances[nearest_offset] <= cutoff:
            graph.add_edge(hydrogen, nearest)
    return graph


def map_topology_to_residue(reference: ReferenceTopology, residue: Residue) -> ResidueTemplate:
    """Map a reference molecular graph onto PDB atom order."""

    reference_heavy = [index for index, element in enumerate(reference.elements) if element != "H"]
    reference_heavy_graph = reference.graph.subgraph(reference_heavy)
    matcher = nx.algorithms.isomorphism.categorical_node_match("element", "")
    element_orders = [residue.elements]
    pdb_counts = Counter(element for element in residue.elements if element != "H")
    reference_counts = Counter(element for element in reference.elements if element != "H")
    missing = list((reference_counts - pdb_counts).elements())
    excess = list((pdb_counts - reference_counts).elements())
    if len(missing) == len(excess) == 1:
        for index, element in enumerate(residue.elements):
            if element != excess[0]:
                continue
            corrected = list(residue.elements)
            corrected[index] = missing[0]
            element_orders.append(tuple(corrected))

    mapping: dict[int, int] | None = None
    coordinate_graph: nx.Graph | None = None
    mapped_elements = residue.elements
    for candidate_elements in element_orders:
        candidate_graph = infer_coordinate_graph(candidate_elements, residue.positions)
        pdb_heavy = [index for index, element in enumerate(candidate_elements) if element != "H"]
        coordinate_heavy_graph = candidate_graph.subgraph(pdb_heavy)
        graph_matcher = nx.algorithms.isomorphism.GraphMatcher(
            coordinate_heavy_graph, reference_heavy_graph, node_match=matcher
        )
        candidate_mapping = next(graph_matcher.isomorphisms_iter(), None)
        if candidate_mapping is not None:
            mapping = candidate_mapping
            coordinate_graph = candidate_graph
            mapped_elements = candidate_elements
            break
    if mapping is None:
        return fallback_residue_template(residue, reference.mol2_match)
    assert coordinate_graph is not None
    pdb_to_reference = mapping
    reference_to_pdb = {reference_index: pdb_index for pdb_index, reference_index in mapping.items()}
    periodic_table = Chem.GetPeriodicTable()
    masses = np.asarray([periodic_table.GetAtomicWeight(element) for element in mapped_elements], dtype=float)
    acceptor = np.zeros(len(residue.elements), dtype=bool)
    polar_heavy = np.zeros(len(residue.elements), dtype=bool)
    for pdb_index, reference_index in pdb_to_reference.items():
        acceptor[pdb_index] = reference.acceptor[reference_index]
        polar_heavy[pdb_index] = reference.polar_heavy[reference_index]
    donor_kind_by_reference_heavy = {
        donor: kind for donor, kind in reference.donor_by_hydrogen.values()
    }
    donor_kind_by_pdb_heavy = {
        reference_to_pdb[donor]: kind
        for donor, kind in donor_kind_by_reference_heavy.items()
        if donor in reference_to_pdb
    }
    donor_by_hydrogen: dict[int, tuple[int, str]] = {}
    for pdb_index, element in enumerate(mapped_elements):
        if element != "H" or coordinate_graph.degree(pdb_index) != 1:
            continue
        donor = next(iter(coordinate_graph.neighbors(pdb_index)))
        if donor in donor_kind_by_pdb_heavy:
            donor_by_hydrogen[pdb_index] = (donor, donor_kind_by_pdb_heavy[donor])
    return ResidueTemplate(
        elements=mapped_elements,
        masses=masses,
        acceptor=acceptor,
        polar_heavy=polar_heavy,
        donor_by_hydrogen=donor_by_hydrogen,
        topology_ok=True,
        mol2_match=reference.mol2_match,
    )


def fallback_residue_template(residue: Residue, mol2_match: bool | None = None) -> ResidueTemplate:
    """Return a conservative geometry-based template and mark it as degraded."""

    graph = infer_coordinate_graph(residue.elements, residue.positions)
    periodic_table = Chem.GetPeriodicTable()
    masses = np.asarray([periodic_table.GetAtomicWeight(element) for element in residue.elements], dtype=float)
    acceptor = np.asarray(
        [
            element in {"O", "S", "F", "Cl", "Br"} or (residue.side == "an" and element == "N")
            for element in residue.elements
        ],
        dtype=bool,
    )
    polar_heavy = np.asarray(
        [element != "H" and element not in {"C"} for element in residue.elements], dtype=bool
    )
    donor_by_hydrogen: dict[int, tuple[int, str]] = {}
    for index, element in enumerate(residue.elements):
        if element != "H" or graph.degree(index) != 1:
            continue
        donor = next(iter(graph.neighbors(index)))
        donor_element = residue.elements[donor]
        if donor_element in {"N", "O", "S"}:
            donor_by_hydrogen[index] = (donor, "strong")
    return ResidueTemplate(
        elements=residue.elements,
        masses=masses,
        acceptor=acceptor,
        polar_heavy=polar_heavy,
        donor_by_hydrogen=donor_by_hydrogen,
        topology_ok=False,
        mol2_match=mol2_match,
    )


def minimum_image(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Apply the minimum-image convention to orthorhombic displacements."""

    return np.asarray(delta, dtype=float) - np.asarray(box, dtype=float) * np.round(
        np.asarray(delta, dtype=float) / np.asarray(box, dtype=float)
    )


def wrapped_center_of_mass(positions: np.ndarray, masses: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Compute a whole-residue mass center and wrap only the resulting center."""

    center = np.average(np.asarray(positions, dtype=float), axis=0, weights=np.asarray(masses, dtype=float))
    return np.mod(center, box)


def compute_rdf(
    points_a: np.ndarray,
    points_b: np.ndarray,
    box: np.ndarray,
    *,
    same: bool,
    groups_a: np.ndarray | None = None,
    groups_b: np.ndarray | None = None,
    dr: float = RDF_DR_A,
    r_max: float = RDF_MAX_A,
) -> RDFResult:
    """Compute a periodic RDF while optionally excluding within-group pairs."""

    box = np.asarray(box, dtype=float)
    if r_max > float(np.min(box)) / 2.0:
        raise ValueError("r_max exceeds half the shortest box length")
    points_a = np.mod(np.asarray(points_a, dtype=float), box)
    points_b = points_a if same else np.mod(np.asarray(points_b, dtype=float), box)
    edges = np.arange(0.0, r_max + dr * 0.5, dr)
    upper_edges = edges[1:]
    centers = 0.5 * (edges[:-1] + edges[1:])
    if len(points_a) == 0 or len(points_b) == 0:
        return RDFResult(centers, np.full_like(centers, np.nan), math.nan)

    tree_a = cKDTree(points_a, boxsize=box)
    tree_b = tree_a if same else cKDTree(points_b, boxsize=box)
    counts = tree_a.count_neighbors(tree_b, upper_edges, cumulative=False).astype(float)
    volume = float(np.prod(box))

    if same:
        counts[0] -= len(points_a)
        counts /= 2.0
        eligible = len(points_a) * (len(points_a) - 1) / 2.0
        if groups_a is not None:
            groups_a = np.asarray(groups_a)
            excluded = 0.0
            for group in np.unique(groups_a):
                indices = np.flatnonzero(groups_a == group)
                excluded += len(indices) * (len(indices) - 1) / 2.0
                if len(indices) > 1:
                    delta = minimum_image(points_a[indices, None, :] - points_a[None, indices, :], box)
                    distances = np.linalg.norm(delta, axis=2)[np.triu_indices(len(indices), 1)]
                    counts -= np.histogram(distances, bins=edges)[0]
            eligible -= excluded
        density = 2.0 * eligible / (len(points_a) * volume) if len(points_a) else math.nan
    else:
        eligible = float(len(points_a) * len(points_b))
        if groups_a is not None and groups_b is not None:
            groups_a = np.asarray(groups_a)
            groups_b = np.asarray(groups_b)
            excluded = 0.0
            for group in np.intersect1d(np.unique(groups_a), np.unique(groups_b)):
                left = np.flatnonzero(groups_a == group)
                right = np.flatnonzero(groups_b == group)
                excluded += float(len(left) * len(right))
                if len(left) and len(right):
                    delta = minimum_image(points_a[left, None, :] - points_b[None, right, :], box)
                    distances = np.linalg.norm(delta, axis=2).ravel()
                    counts -= np.histogram(distances, bins=edges)[0]
            eligible -= excluded
        density = eligible / (len(points_a) * volume) if len(points_a) else math.nan

    shell_volumes = 4.0 * math.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)
    expected = eligible * shell_volumes / volume
    g = np.divide(counts, expected, out=np.full_like(counts, np.nan), where=expected > 0)
    g[np.abs(g) < 1e-12] = 0.0
    return RDFResult(centers, g, density)


def extract_rdf_features(result: RDFResult) -> dict[str, float]:
    """Extract the first peak, valley, coordination, FWHM and excess area."""

    empty = {metric: math.nan for metric in RDF_METRICS}
    finite = np.isfinite(result.g)
    if finite.sum() < 11:
        return empty
    g = np.where(finite, result.g, 0.0)
    smooth = savgol_filter(g, 11, 3, mode="interp")
    candidates, _ = find_peaks(smooth, prominence=0.1, height=1.1)
    candidates = candidates[result.r[candidates] >= 1.5]
    if not len(candidates):
        return empty
    peak = int(candidates[0])
    minima, _ = find_peaks(-smooth[peak + 1 :])
    if not len(minima):
        return empty
    valley = int(peak + 1 + minima[0])
    peak_height = float(smooth[peak])
    half_level = 1.0 + (peak_height - 1.0) / 2.0

    left = peak
    while left > 0 and smooth[left] > half_level:
        left -= 1
    right = peak
    while right < len(smooth) - 1 and smooth[right] > half_level:
        right += 1
    fwhm = float(result.r[right] - result.r[left]) if left < peak < right else math.nan

    area_left = peak
    while area_left > 0 and smooth[area_left] > 1.0:
        area_left -= 1
    area_slice = slice(area_left, valley + 1)
    excess = float(np.trapezoid(np.maximum(smooth[area_slice] - 1.0, 0.0), result.r[area_slice]))
    integration_slice = slice(0, valley + 1)
    coordination = float(
        4.0
        * math.pi
        * result.density_around_reference
        * np.trapezoid(
            np.maximum(result.g[integration_slice], 0.0) * result.r[integration_slice] ** 2,
            result.r[integration_slice],
        )
    )
    return {
        "peak1_r_A": float(result.r[peak]),
        "peak1_g": peak_height,
        "minimum1_r_A": float(result.r[valley]),
        "coordination1": coordination,
        "fwhm_A": fwhm,
        "excess_area_A": excess,
    }


def _collect_box_sites(
    box_data: PDBBox,
    cation_template: ResidueTemplate,
    anion_template: ResidueTemplate,
) -> dict[str, Any]:
    centers: dict[str, list[np.ndarray]] = {"cat": [], "an": []}
    polar_positions: list[np.ndarray] = []
    nonpolar_positions: list[np.ndarray] = []
    polar_groups: list[int] = []
    nonpolar_groups: list[int] = []
    acceptor_positions: list[np.ndarray] = []
    acceptor_sides: list[str] = []
    acceptor_groups: list[int] = []
    donor_h_positions: list[np.ndarray] = []
    donor_d_positions: list[np.ndarray] = []
    donor_sides: list[str] = []
    donor_groups: list[int] = []
    donor_kinds: list[str] = []

    for group_id, residue in enumerate(box_data.residues):
        template = cation_template if residue.side == "cat" else anion_template
        if len(template.masses) != len(residue.elements):
            raise ValueError(f"atom count changed within {residue.side} residues")
        centers[residue.side].append(wrapped_center_of_mass(residue.positions, template.masses, box_data.lengths))
        wrapped = np.mod(residue.positions, box_data.lengths)
        heavy = np.asarray([element != "H" for element in template.elements], dtype=bool)
        polar = heavy & template.polar_heavy
        nonpolar = heavy & ~template.polar_heavy
        polar_positions.extend(wrapped[polar])
        polar_groups.extend([group_id] * int(polar.sum()))
        nonpolar_positions.extend(wrapped[nonpolar])
        nonpolar_groups.extend([group_id] * int(nonpolar.sum()))
        for atom_index in np.flatnonzero(template.acceptor & heavy):
            acceptor_positions.append(wrapped[atom_index])
            acceptor_sides.append(residue.side)
            acceptor_groups.append(group_id)
        for hydrogen, (donor, kind) in template.donor_by_hydrogen.items():
            donor_h_positions.append(wrapped[hydrogen])
            donor_d_positions.append(wrapped[donor])
            donor_sides.append(residue.side)
            donor_groups.append(group_id)
            donor_kinds.append(kind)

    def positions(values: list[np.ndarray]) -> np.ndarray:
        return np.vstack(values) if values else np.empty((0, 3), dtype=float)

    return {
        "cat_centers": positions(centers["cat"]),
        "an_centers": positions(centers["an"]),
        "polar_positions": positions(polar_positions),
        "nonpolar_positions": positions(nonpolar_positions),
        "polar_groups": np.asarray(polar_groups, dtype=int),
        "nonpolar_groups": np.asarray(nonpolar_groups, dtype=int),
        "acceptor_positions": positions(acceptor_positions),
        "acceptor_sides": np.asarray(acceptor_sides, dtype=object),
        "acceptor_groups": np.asarray(acceptor_groups, dtype=int),
        "donor_h_positions": positions(donor_h_positions),
        "donor_d_positions": positions(donor_d_positions),
        "donor_sides": np.asarray(donor_sides, dtype=object),
        "donor_groups": np.asarray(donor_groups, dtype=int),
        "donor_kinds": np.asarray(donor_kinds, dtype=object),
    }


def compute_hydrogen_bond_features(sites: dict[str, Any], box: np.ndarray, ion_count: int) -> dict[str, float]:
    """Compute fixed-criterion intermolecular hydrogen-bond and network features."""

    result = {
        f"hb_{direction}_{kind}_count": 0.0
        for direction in HB_DIRECTIONS
        for kind in ("strong", "weak_ch")
    }
    result.update(
        {
            "hb_total_count": 0.0,
            "hb_total_per_ion_pair": 0.0,
            "hb_strict_total_count": 0.0,
            "hb_loose_total_count": 0.0,
            "hb_donor_utilization_fraction": 0.0,
            "hb_acceptor_occupancy_fraction": 0.0,
            "hb_HA_distance_mean_A": math.nan,
            "hb_HA_distance_std_A": math.nan,
            "hb_DHA_angle_mean_deg": math.nan,
            "hb_DHA_angle_std_deg": math.nan,
            "hb_network_mean_degree": 0.0,
            "hb_network_isolated_fraction": 1.0 if ion_count else math.nan,
            "hb_network_largest_component_fraction": 1.0 / ion_count if ion_count else math.nan,
        }
    )
    hydrogens = sites["donor_h_positions"]
    acceptors = sites["acceptor_positions"]
    if len(hydrogens) == 0 or len(acceptors) == 0:
        return result

    tree_h = cKDTree(np.mod(hydrogens, box), boxsize=box)
    tree_a = cKDTree(np.mod(acceptors, box), boxsize=box)
    candidates = tree_h.sparse_distance_matrix(tree_a, 2.8, output_type="coo_matrix")
    primary_records: list[tuple[int, int, float, float]] = []
    strict_count = 0
    loose_count = 0
    graph = nx.Graph()
    graph.add_nodes_from(range(ion_count))

    for hydrogen_index, acceptor_index, ha_distance in zip(
        candidates.row, candidates.col, candidates.data, strict=True
    ):
        if sites["donor_groups"][hydrogen_index] == sites["acceptor_groups"][acceptor_index]:
            continue
        donor_position = sites["donor_d_positions"][hydrogen_index]
        hydrogen_position = hydrogens[hydrogen_index]
        acceptor_position = acceptors[acceptor_index]
        hd = minimum_image(donor_position - hydrogen_position, box)
        ha = minimum_image(acceptor_position - hydrogen_position, box)
        denominator = float(np.linalg.norm(hd) * np.linalg.norm(ha))
        if denominator <= 0:
            continue
        cosine = float(np.clip(np.dot(hd, ha) / denominator, -1.0, 1.0))
        angle = math.degrees(math.acos(cosine))
        da_distance = float(np.linalg.norm(minimum_image(acceptor_position - donor_position, box)))
        is_loose = ha_distance <= 2.8 and da_distance <= 3.5 and angle >= 120.0
        is_primary = ha_distance <= 2.5 and da_distance <= 3.5 and angle >= 150.0
        is_strict = ha_distance <= 2.2 and da_distance <= 3.2 and angle >= 160.0
        loose_count += int(is_loose)
        strict_count += int(is_strict)
        if not is_primary:
            continue
        primary_records.append((int(hydrogen_index), int(acceptor_index), float(ha_distance), angle))
        donor_side = sites["donor_sides"][hydrogen_index]
        acceptor_side = sites["acceptor_sides"][acceptor_index]
        direction = f"{donor_side}_to_{acceptor_side}"
        kind = sites["donor_kinds"][hydrogen_index]
        result[f"hb_{direction}_{kind}_count"] += 1.0
        graph.add_edge(
            int(sites["donor_groups"][hydrogen_index]), int(sites["acceptor_groups"][acceptor_index])
        )

    result["hb_total_count"] = float(len(primary_records))
    result["hb_total_per_ion_pair"] = float(len(primary_records) / (ion_count / 2.0)) if ion_count else math.nan
    result["hb_strict_total_count"] = float(strict_count)
    result["hb_loose_total_count"] = float(loose_count)
    if primary_records:
        used_donors = {record[0] for record in primary_records}
        used_acceptors = {record[1] for record in primary_records}
        distances = np.asarray([record[2] for record in primary_records])
        angles = np.asarray([record[3] for record in primary_records])
        result["hb_donor_utilization_fraction"] = len(used_donors) / len(hydrogens)
        result["hb_acceptor_occupancy_fraction"] = len(used_acceptors) / len(acceptors)
        result["hb_HA_distance_mean_A"] = float(distances.mean())
        result["hb_HA_distance_std_A"] = float(distances.std(ddof=0))
        result["hb_DHA_angle_mean_deg"] = float(angles.mean())
        result["hb_DHA_angle_std_deg"] = float(angles.std(ddof=0))
    if ion_count:
        degrees = np.asarray([graph.degree(node) for node in graph.nodes], dtype=float)
        components = list(nx.connected_components(graph))
        result["hb_network_mean_degree"] = float(degrees.mean())
        result["hb_network_isolated_fraction"] = float(np.mean(degrees == 0))
        result["hb_network_largest_component_fraction"] = max(map(len, components)) / ion_count
    return result


def cic_grid(points: np.ndarray, weights: np.ndarray, box: np.ndarray, size: int = 64) -> np.ndarray:
    """Deposit weighted points on a periodic grid using cloud-in-cell."""

    grid = np.zeros((size, size, size), dtype=float)
    if len(points) == 0:
        return grid
    scaled = np.mod(points, box) / box * size
    base = np.floor(scaled).astype(int)
    fraction = scaled - base
    for dx in (0, 1):
        wx = fraction[:, 0] if dx else 1.0 - fraction[:, 0]
        for dy in (0, 1):
            wy = fraction[:, 1] if dy else 1.0 - fraction[:, 1]
            for dz in (0, 1):
                wz = fraction[:, 2] if dz else 1.0 - fraction[:, 2]
                indices = (base + np.asarray([dx, dy, dz])) % size
                np.add.at(grid, (indices[:, 0], indices[:, 1], indices[:, 2]), weights * wx * wy * wz)
    return grid


def radial_structure_factor(
    field: np.ndarray,
    box: np.ndarray,
    normalization: float,
    *,
    dq: float = 0.05,
    q_max: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Radially average a CIC density field in reciprocal space."""

    size = field.shape[0]
    transform = np.fft.fftn(field)
    q_axes = [2.0 * math.pi * np.fft.fftfreq(size, d=float(length) / size) for length in box]
    qx, qy, qz = np.meshgrid(*q_axes, indexing="ij")
    q_magnitude = np.sqrt(qx * qx + qy * qy + qz * qz)
    spacing = box / size
    window_amplitude = (
        np.sinc(qx * spacing[0] / (2.0 * math.pi)) ** 2
        * np.sinc(qy * spacing[1] / (2.0 * math.pi)) ** 2
        * np.sinc(qz * spacing[2] / (2.0 * math.pi)) ** 2
    )
    power = np.abs(transform) ** 2 / max(normalization, np.finfo(float).tiny)
    power = np.divide(
        power,
        window_amplitude**2,
        out=np.full_like(power, np.nan),
        where=window_amplitude > 1e-8,
    )
    bin_count = int(math.ceil(q_max / dq))
    q = (np.arange(bin_count) + 0.5) * dq
    indices = np.floor(q_magnitude.ravel() / dq).astype(int)
    valid = (q_magnitude.ravel() > 0) & (indices < bin_count) & np.isfinite(power.ravel())
    modes = np.bincount(indices[valid], minlength=bin_count).astype(int)
    sums = np.bincount(indices[valid], weights=power.ravel()[valid], minlength=bin_count)
    spectrum = np.divide(sums, modes, out=np.full(bin_count, np.nan), where=modes > 0)
    return q, spectrum, modes


def extract_spectrum_peak(
    q: np.ndarray,
    spectrum: np.ndarray,
    modes: np.ndarray,
    q_min: float,
    q_max: float,
) -> dict[str, float]:
    """Extract the first reproducible structure-factor peak in a q window."""

    empty = {"q": math.nan, "height": math.nan, "fwhm": math.nan, "area": math.nan}
    valid = np.isfinite(spectrum) & (modes >= 6)
    window = valid & (q >= q_min) & (q <= q_max)
    indices = np.flatnonzero(window)
    if len(indices) < 3:
        return empty
    first, last = int(indices[0]), int(indices[-1])
    x = q[first : last + 1]
    raw = spectrum[first : last + 1]
    finite = np.isfinite(raw) & (modes[first : last + 1] >= 6)
    if finite.sum() < 3:
        return empty
    y = np.interp(x, x[finite], raw[finite])
    if len(y) >= 5:
        y = savgol_filter(y, 5, 2, mode="interp")
    peaks, _ = find_peaks(y, prominence=0.1)
    if not len(peaks):
        return empty
    peak = int(peaks[0])
    height = float(y[peak])
    half_level = 1.0 + (height - 1.0) / 2.0
    left = peak
    while left > 0 and y[left] > half_level:
        left -= 1
    right = peak
    while right < len(y) - 1 and y[right] > half_level:
        right += 1
    fwhm = float(x[right] - x[left]) if left < peak < right else math.nan
    area = float(np.trapezoid(np.maximum(y - 1.0, 0.0), x))
    return {"q": float(x[peak]), "height": height, "fwhm": fwhm, "area": area}


def compute_structure_factor_features(
    sites: dict[str, Any], box: np.ndarray
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Compute polar/nonpolar S_CC and ion-center charge S_ZZ spectra."""

    polar = sites["polar_positions"]
    nonpolar = sites["nonpolar_positions"]
    output = {
        "scc_prepeak_q_A^-1": math.nan,
        "scc_prepeak_height": math.nan,
        "scc_prepeak_fwhm_A^-1": math.nan,
        "scc_prepeak_area_A^-1": math.nan,
        "scc_domain_spacing_A": math.nan,
        "szz_peak_q_A^-1": math.nan,
        "szz_peak_height": math.nan,
        "szz_peak_fwhm_A^-1": math.nan,
        "szz_peak_area_A^-1": math.nan,
    }
    curves: dict[str, np.ndarray] = {}
    if len(polar) and len(nonpolar):
        total = len(polar) + len(nonpolar)
        x_p = len(polar) / total
        x_n = len(nonpolar) / total
        points = np.vstack([polar, nonpolar])
        weights = np.concatenate([np.full(len(polar), x_n), np.full(len(nonpolar), -x_p)])
        field = cic_grid(points, weights, box)
        q, scc, modes = radial_structure_factor(field, box, total * x_p * x_n)
        curves.update({"q": q, "scc": scc, "scc_modes": modes})
        peak = extract_spectrum_peak(q, scc, modes, max(2.0 * math.pi / max(box), 0.15), 1.0)
        output.update(
            {
                "scc_prepeak_q_A^-1": peak["q"],
                "scc_prepeak_height": peak["height"],
                "scc_prepeak_fwhm_A^-1": peak["fwhm"],
                "scc_prepeak_area_A^-1": peak["area"],
                "scc_domain_spacing_A": 2.0 * math.pi / peak["q"] if math.isfinite(peak["q"]) else math.nan,
            }
        )

    cations = sites["cat_centers"]
    anions = sites["an_centers"]
    ion_points = np.vstack([cations, anions])
    ion_weights = np.concatenate([np.ones(len(cations)), -np.ones(len(anions))])
    charge_field = cic_grid(ion_points, ion_weights, box)
    q_z, szz, modes_z = radial_structure_factor(charge_field, box, len(ion_points))
    curves.update({"q_z": q_z, "szz": szz, "szz_modes": modes_z})
    peak_z = extract_spectrum_peak(q_z, szz, modes_z, 0.2, 1.5)
    output.update(
        {
            "szz_peak_q_A^-1": peak_z["q"],
            "szz_peak_height": peak_z["height"],
            "szz_peak_fwhm_A^-1": peak_z["fwhm"],
            "szz_peak_area_A^-1": peak_z["area"],
        }
    )
    return output, curves


def extract_box_features(
    pdb_path: Path,
    cation_smiles: str,
    anion_smiles: str,
    *,
    cation_mol2: Mol2Info | None = None,
    anion_mol2: Mol2Info | None = None,
    topology_cache: dict[tuple[str, str], tuple[ReferenceTopology, ResidueTemplate]] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Extract all v1 scalar features and audit curves for one PDB box."""

    row = empty_feature_row()
    flags: set[str] = set()
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        row["qc_flags"] = "missing_pdb"
        return row, {}
    row["qc_pdb_present"] = True
    box_data = parse_pdb_box(pdb_path)
    first_cation = next(residue for residue in box_data.residues if residue.side == "cat")
    first_anion = next(residue for residue in box_data.residues if residue.side == "an")
    cache = topology_cache if topology_cache is not None else {}

    def template_for(side: str, smiles: str, residue: Residue, mol2: Mol2Info | None) -> tuple[ReferenceTopology, ResidueTemplate]:
        key = (side, smiles, residue.elements)
        if key not in cache:
            reference = build_reference_topology(smiles, mol2)
            cache[key] = (reference, map_topology_to_residue(reference, residue))
        return cache[key]

    cation_reference, cation_template = template_for("cat", cation_smiles, first_cation, cation_mol2)
    anion_reference, anion_template = template_for("an", anion_smiles, first_anion, anion_mol2)
    if not cation_template.topology_ok or not anion_template.topology_ok:
        flags.add("topology_fallback")
    for residue in box_data.residues:
        template = cation_template if residue.side == "cat" else anion_template
        if len(template.elements) == len(residue.elements) and template.elements != residue.elements:
            flags.add("pdb_atom_name_element_mismatch")
    row.update(
        {
            "qc_topology_ok": cation_template.topology_ok and anion_template.topology_ok,
            "qc_cation_mol2_match": cation_reference.mol2_match,
            "qc_anion_mol2_match": anion_reference.mol2_match,
            "qc_box_a_A": float(box_data.lengths[0]),
            "qc_box_b_A": float(box_data.lengths[1]),
            "qc_box_c_A": float(box_data.lengths[2]),
            "qc_box_volume_A^3": float(np.prod(box_data.lengths)),
            "qc_cation_count": sum(residue.side == "cat" for residue in box_data.residues),
            "qc_anion_count": sum(residue.side == "an" for residue in box_data.residues),
            "qc_atom_count": box_data.atom_count,
        }
    )
    sites = _collect_box_sites(box_data, cation_template, anion_template)
    polar_count = len(sites["polar_positions"])
    nonpolar_count = len(sites["nonpolar_positions"])
    heavy_count = polar_count + nonpolar_count
    row["qc_polar_heavy_count"] = polar_count
    row["qc_nonpolar_heavy_count"] = nonpolar_count
    row["domain_polar_heavy_fraction"] = polar_count / heavy_count if heavy_count else math.nan
    row["domain_nonpolar_heavy_fraction"] = nonpolar_count / heavy_count if heavy_count else math.nan

    rdf_inputs = {
        "cc": (sites["cat_centers"], sites["cat_centers"], True, None, None),
        "ca": (sites["cat_centers"], sites["an_centers"], False, None, None),
        "aa": (sites["an_centers"], sites["an_centers"], True, None, None),
        "pp": (
            sites["polar_positions"], sites["polar_positions"], True,
            sites["polar_groups"], sites["polar_groups"],
        ),
        "pn": (
            sites["polar_positions"], sites["nonpolar_positions"], False,
            sites["polar_groups"], sites["nonpolar_groups"],
        ),
        "nn": (
            sites["nonpolar_positions"], sites["nonpolar_positions"], True,
            sites["nonpolar_groups"], sites["nonpolar_groups"],
        ),
    }
    curves: dict[str, np.ndarray] = {}
    for key, (points_a, points_b, same, groups_a, groups_b) in rdf_inputs.items():
        rdf = compute_rdf(
            points_a, points_b, box_data.lengths, same=same, groups_a=groups_a, groups_b=groups_b
        )
        curves[f"rdf_{key}"] = rdf.g.astype(np.float32)
        features = extract_rdf_features(rdf)
        row.update({f"rdf_{key}_{name}": value for name, value in features.items()})
        if not math.isfinite(features["peak1_r_A"]):
            flags.add(f"rdf_{key}_peak_missing")
    curves["rdf_r_A"] = rdf.r.astype(np.float32)

    row.update(compute_hydrogen_bond_features(sites, box_data.lengths, len(box_data.residues)))
    structure_features, structure_curves = compute_structure_factor_features(sites, box_data.lengths)
    row.update(structure_features)
    curves.update({key: value.astype(np.float32) for key, value in structure_curves.items()})
    if not math.isfinite(row["scc_prepeak_q_A^-1"]):
        flags.add("scc_prepeak_missing")
    if not math.isfinite(row["szz_peak_q_A^-1"]):
        flags.add("szz_peak_missing")
    if nonpolar_count == 0:
        flags.add("no_nonpolar_atoms")

    row["qc_flags"] = "|".join(sorted(flags))
    partial_flags = {
        "topology_fallback",
        "no_nonpolar_atoms",
        "rdf_cc_peak_missing",
        "rdf_ca_peak_missing",
        "rdf_aa_peak_missing",
    }
    row["qc_status"] = "partial" if flags.intersection(partial_flags) else "ok"
    return row, curves
