#!/usr/bin/env python3
"""Sweep ROCS thresholds across backends and query files.

This script reruns the ROC-based threshold determination (Youden's Index)
for multiple ROCS backends (RDKit/CDPKit/OpenEye) and multiple "query" inputs
(SDF and, for OpenEye, .sq query files).

It saves per-(backend, query) artifacts in an output directory:
  - combined_figure.png (actives/decoys overlap + ROC + metrics)
  - analysis_summary.txt
  - molecule_scores.csv / roc_analysis.csv / threshold_metrics.csv

It also writes an aggregated summary:
  - summary.csv
  - summary.md
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from rdkit import Chem

from threshold_analysis import (
    compare_thresholds,
    create_visualizations,
    generate_report,
    load_datasets,
    perform_roc_analysis,
    save_results,
)

try:
    from config import (
        MAX_CONFORMERS,
        MAX_HEAVY_ATOMS,
        MAX_ISOMERS,
        MAX_ROTATABLE_BONDS,
        ROCS_THRESHOLD,
    )

    DEFAULT_RDKIT_THRESHOLD = float(ROCS_THRESHOLD)
except ImportError:
    MAX_CONFORMERS = 50
    MAX_ISOMERS = 4
    MAX_HEAVY_ATOMS = 45
    MAX_ROTATABLE_BONDS = 15
    DEFAULT_RDKIT_THRESHOLD = 0.9


@dataclass(frozen=True)
class SweepItem:
    backend: str
    query_path: Path
    query_label: str
    current_threshold: float


def _canonicalize_smiles(smiles: Iterable[str]) -> tuple[list[str], int]:
    """Canonicalize SMILES and drop invalid ones."""
    canonical: list[str] = []
    dropped = 0
    for smi in smiles:
        if not isinstance(smi, str) or not smi.strip():
            dropped += 1
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            dropped += 1
            continue
        canonical.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return canonical, dropped


def _default_query_files(root: Path) -> list[Path]:
    query_dir = root / "rocs_rl_ccr/rdkit_cdpkit"
    return [
        query_dir / "CCR2_reference_ligands.sdf",
        query_dir / "supermol_123.sdf",
        query_dir / "model1.sq",
        query_dir / "model2.sq",
        query_dir / "model3-4_v1.sq",
    ]


def _select_items(
    *,
    backends: list[str],
    queries: list[Path],
    cdpkit_threshold: float,
    openeye_threshold: float,
) -> list[SweepItem]:
    items: list[SweepItem] = []
    for backend in backends:
        for query_path in queries:
            ext = query_path.suffix.lower()
            if backend in {"rdkit", "cdpkit"} and ext != ".sdf":
                continue
            if backend == "openeye" and ext not in {".sdf", ".sq"}:
                continue

            if backend == "rdkit":
                current = DEFAULT_RDKIT_THRESHOLD
            elif backend == "cdpkit":
                current = cdpkit_threshold
            else:
                current = openeye_threshold

            label = f"{query_path.stem}{query_path.suffix}"
            items.append(
                SweepItem(
                    backend=backend,
                    query_path=query_path,
                    query_label=label,
                    current_threshold=float(current),
                )
            )
    return items


def _init_scorer(backend: str, query_path: Path, *, n_jobs: int):
    if backend == "rdkit":
        from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
        from drugex.training.scorers.rocs_rdkit import RDKitROCSScorer

        num_threads = 1 if n_jobs == 1 else 0
        return RDKitROCSScorer(
            conformer_generator=RDKitConformerGenerator(
                max_conformers=MAX_CONFORMERS,
                max_isomers=MAX_ISOMERS,
                max_heavy_atoms=MAX_HEAVY_ATOMS,
                max_rotatable_bonds=MAX_ROTATABLE_BONDS,
                num_threads=num_threads,
                show_progress=False,
            ),
            references=str(query_path),
            score_type="TanimotoCombo",
            use_colors=True,
            show_progress=False,
            n_jobs=n_jobs,
        )

    if backend == "cdpkit":
        from drugex.training.scorers.conformer_generators import CDPKitConformerGenerator
        from drugex.training.scorers.rocs_cdpkit import CDPKitROCSScorer

        return CDPKitROCSScorer(
            conformer_generator=CDPKitConformerGenerator(
                max_conformers=MAX_CONFORMERS,
                max_isomers=MAX_ISOMERS,
                max_heavy_atoms=MAX_HEAVY_ATOMS,
                max_rotatable_bonds=MAX_ROTATABLE_BONDS,
                show_progress=False,
            ),
            references=str(query_path),
            show_progress=False,
            n_jobs=n_jobs,
        )

    if backend == "openeye":
        from drugex.training.scorers.conformer_generators import OmegaConformerGenerator
        from drugex.training.scorers.rocs_openeye import OpenEyeROCSScorer

        rocs_binary = shutil.which("rocs")
        if not rocs_binary:
            raise FileNotFoundError("OpenEye ROCS binary ('rocs') not found in PATH")

        return OpenEyeROCSScorer(
            conformer_generator=OmegaConformerGenerator(
                max_conformers=MAX_CONFORMERS,
                max_centers=2,
                max_heavy_atoms=MAX_HEAVY_ATOMS,
                max_rotatable_bonds=MAX_ROTATABLE_BONDS,
                use_gpu=False,
                show_progress=False,
            ),
            references={"query": str(query_path)},
            score_type="TanimotoCombo",
            shape_only=False,
            optimize=True,
            color_optimize=True,
            rocs_binary=rocs_binary,
            show_progress=False,
        )

    raise ValueError(f"Unknown backend: {backend}")


def _score_1d(scorer, smiles: list[str]) -> np.ndarray:
    scores = scorer.getScores(smiles)
    scores = np.asarray(scores)
    if scores.ndim == 1:
        return scores.astype(float)
    if scores.ndim == 2 and scores.shape[1] == 1:
        return scores[:, 0].astype(float)
    raise ValueError(f"Expected 1 score per molecule, got shape: {scores.shape}")


def _overlap_stats(actives_scores: np.ndarray, decoys_scores: np.ndarray) -> Dict[str, float]:
    overlap_min = float(max(decoys_scores.min(), actives_scores.min()))
    overlap_max = float(min(decoys_scores.max(), actives_scores.max()))
    if overlap_max <= overlap_min:
        return {
            "overlap_min": overlap_min,
            "overlap_max": overlap_max,
            "overlap_actives_frac": 0.0,
            "overlap_decoys_frac": 0.0,
        }

    overlap_actives = float(
        np.mean((actives_scores >= overlap_min) & (actives_scores <= overlap_max))
    )
    overlap_decoys = float(
        np.mean((decoys_scores >= overlap_min) & (decoys_scores <= overlap_max))
    )
    return {
        "overlap_min": overlap_min,
        "overlap_max": overlap_max,
        "overlap_actives_frac": overlap_actives,
        "overlap_decoys_frac": overlap_decoys,
    }


def run_sweep(
    *,
    actives_csv: Path,
    decoys_csv: Path,
    reference_ligands_sdf: Path,
    items: list[SweepItem],
    output_dir: Path,
    n_jobs: int,
) -> pd.DataFrame:
    actives_raw, decoys_raw, _, ref_smiles = load_datasets(
        str(actives_csv), str(decoys_csv), str(reference_ligands_sdf)
    )
    actives, actives_dropped = _canonicalize_smiles(actives_raw)
    decoys, decoys_dropped = _canonicalize_smiles(decoys_raw)

    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for item in items:
        out = output_dir / item.backend / item.query_path.stem
        out.mkdir(parents=True, exist_ok=True)

        scorer = _init_scorer(item.backend, item.query_path, n_jobs=n_jobs)

        actives_scores = _score_1d(scorer, actives)
        decoys_scores = _score_1d(scorer, decoys)
        ref_scores = _score_1d(scorer, ref_smiles)

        roc_data = perform_roc_analysis(actives_scores, decoys_scores)
        threshold_df = compare_thresholds(
            roc_data["y_true"], roc_data["y_scores"], current_threshold=item.current_threshold
        )

        create_visualizations(
            actives_scores,
            decoys_scores,
            ref_scores,
            roc_data,
            threshold_df,
            item.current_threshold,
            show_plots=False,
            save_path=str(out / "combined_figure.png"),
        )

        report_text = generate_report(
            roc_data,
            threshold_df,
            item.current_threshold,
            actives_scores,
            decoys_scores,
            ref_scores,
        )

        save_results(
            str(out),
            actives_scores,
            decoys_scores,
            ref_scores,
            actives,
            decoys,
            ref_smiles,
            roc_data,
            threshold_df,
            report_text,
        )

        metadata = {
            "backend": item.backend,
            "query_path": str(item.query_path),
            "reference_ligands_sdf": str(reference_ligands_sdf),
            "n_jobs": n_jobs,
            "max_conformers": MAX_CONFORMERS,
            "max_isomers": MAX_ISOMERS,
            "max_heavy_atoms": MAX_HEAVY_ATOMS,
            "max_rotatable_bonds": MAX_ROTATABLE_BONDS,
            "actives_raw_n": len(actives_raw),
            "actives_used_n": len(actives),
            "actives_dropped_n": actives_dropped,
            "decoys_raw_n": len(decoys_raw),
            "decoys_used_n": len(decoys),
            "decoys_dropped_n": decoys_dropped,
        }
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        overlap = _overlap_stats(actives_scores, decoys_scores)
        rows.append(
            {
                "backend": item.backend,
                "query": item.query_label,
                "query_path": str(item.query_path),
                "current_threshold": item.current_threshold,
                "optimal_threshold": float(roc_data["optimal_threshold"]),
                "roc_auc": float(roc_data["roc_auc"]),
                "pr_auc": float(roc_data["pr_auc"]),
                "optimal_tpr": float(roc_data["optimal_tpr"]),
                "optimal_fpr": float(roc_data["optimal_fpr"]),
                "actives_used_n": len(actives),
                "decoys_used_n": len(decoys),
                "actives_mean": float(np.mean(actives_scores)),
                "decoys_mean": float(np.mean(decoys_scores)),
                "actives>=current_frac": float(np.mean(actives_scores >= item.current_threshold)),
                "decoys>=current_frac": float(np.mean(decoys_scores >= item.current_threshold)),
                "actives>=optimal_frac": float(np.mean(actives_scores >= roc_data["optimal_threshold"])),
                "decoys>=optimal_frac": float(np.mean(decoys_scores >= roc_data["optimal_threshold"])),
                **overlap,
                "artifacts_dir": str(out),
                "figure_path": str(out / "combined_figure.png"),
            }
        )

    df = pd.DataFrame(rows).sort_values(["backend", "query"]).reset_index(drop=True)
    df.to_csv(output_dir / "summary.csv", index=False)

    md = df[
        [
            "backend",
            "query",
            "current_threshold",
            "optimal_threshold",
            "roc_auc",
            "pr_auc",
            "actives>=current_frac",
            "decoys>=current_frac",
            "actives>=optimal_frac",
            "decoys>=optimal_frac",
            "figure_path",
        ]
    ].copy()
    md["figure_path"] = md["figure_path"].apply(
        lambda p: str(Path(p).relative_to(output_dir))
    )
    (output_dir / "summary.md").write_text(md.to_markdown(index=False) + "\n")

    return df


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends",
        default="rdkit,cdpkit,openeye",
        help="Comma-separated list: rdkit,cdpkit,openeye (default: %(default)s)",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=None,
        help=(
            "Query files as either a single comma-separated string or a space-separated list "
            "(default: built-in CCR2 tutorial queries)"
        ),
    )
    parser.add_argument(
        "--actives-csv",
        type=Path,
        default=root / "rocs_rl_ccr/rdkit_cdpkit/actives_decoys/actives_ccr2_N75.csv",
    )
    parser.add_argument(
        "--decoys-csv",
        type=Path,
        default=root / "rocs_rl_ccr/rdkit_cdpkit/actives_decoys/decoys_ccr2_N500.csv",
    )
    parser.add_argument(
        "--reference-ligands-sdf",
        type=Path,
        default=root / "rocs_rl_ccr/rdkit_cdpkit/CCR2_reference_ligands.sdf",
        help="SDF used for the reference-score sanity check and plot markers",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "threshold_sweep_results",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel scoring workers where supported (default: %(default)s)",
    )
    parser.add_argument(
        "--cdpkit-threshold",
        type=float,
        default=0.9,
        help="Current ROCS threshold to compare against for CDPKit (default: %(default)s)",
    )
    parser.add_argument(
        "--openeye-threshold",
        type=float,
        default=0.9,
        help="Current ROCS threshold to compare against for OpenEye (default: %(default)s)",
    )
    return parser.parse_args(argv)


def _available_backends(requested: list[str]) -> list[str]:
    available: list[str] = []
    for backend in requested:
        if backend == "rdkit":
            available.append(backend)
            continue

        if backend == "cdpkit":
            try:
                import CDPL  # noqa: F401
                from drugex.training.scorers.rocs_cdpkit import CDPKitROCSScorer  # noqa: F401
            except Exception:
                continue
            available.append(backend)
            continue

        if backend == "openeye":
            try:
                from openeye import oechem  # noqa: F401
                from drugex.training.scorers.rocs_openeye import OpenEyeROCSScorer  # noqa: F401
            except Exception:
                continue
            if shutil.which("rocs") is None:
                continue
            available.append(backend)
            continue
    return available


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    root = Path(__file__).resolve().parent

    requested_backends = [b.strip().lower() for b in args.backends.split(",") if b.strip()]
    backends = _available_backends(requested_backends)
    if not backends:
        raise RuntimeError(
            f"No requested backends are available. Requested={requested_backends}"
        )

    if args.queries:
        raw_queries: list[str] = []
        for token in args.queries:
            raw_queries.extend([part.strip() for part in token.split(",") if part.strip()])
        queries = [Path(q) for q in raw_queries]
    else:
        queries = _default_query_files(root)

    missing = [q for q in queries if not q.exists()]
    if missing:
        raise FileNotFoundError(f"Missing query file(s): {[str(p) for p in missing]}")

    items = _select_items(
        backends=backends,
        queries=queries,
        cdpkit_threshold=args.cdpkit_threshold,
        openeye_threshold=args.openeye_threshold,
    )

    df = run_sweep(
        actives_csv=args.actives_csv,
        decoys_csv=args.decoys_csv,
        reference_ligands_sdf=args.reference_ligands_sdf,
        items=items,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
    )

    cols = ["backend", "query", "current_threshold", "optimal_threshold", "roc_auc", "pr_auc"]
    print(df[cols].to_string(index=False))
    print(f"\nWrote: {args.output_dir}/summary.csv")
    print(f"Wrote: {args.output_dir}/summary.md")


if __name__ == "__main__":
    main()
