"""Export an interactive PaCMAP visualization from custom MAEST/tempo kNN pairs."""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np

from djprojectexploration.interactive_umap_precomputed import (
    PROJECT_ROOT,
    _align_to_reference,
    _build_html,
    _build_plot,
    _build_similarity_payload,
    _component_matrices,
    _combined_distance,
    _load_combined_records_and_features,
    _render_layout_gif,
)


def _neighbor_pairs_from_distance(D: np.ndarray, *, n_neighbors: int) -> np.ndarray:
    distances = np.asarray(D, dtype=np.float32)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError(f"Expected square distance matrix, got shape {distances.shape}.")

    n = distances.shape[0]
    k = min(max(1, int(n_neighbors)), n - 1)
    pairs = np.empty((n * k, 2), dtype=np.int32)

    row = 0
    for i in range(n):
        order = np.argsort(distances[i], kind="stable")
        order = order[order != i]
        for j in order[:k]:
            pairs[row, 0] = i
            pairs[row, 1] = int(j)
            row += 1

    return pairs


def _compute_pacmap_knn_layouts(
    *,
    X_reference: np.ndarray,
    D_maest: np.ndarray,
    D_tempo: np.ndarray,
    weight_values: list[float],
    n_neighbors: int,
    mn_ratio: float,
    fp_ratio: float,
    random_state: int,
    distance: str,
    align: bool,
) -> dict[str, list[list[float]]]:
    try:
        import pacmap
    except ImportError as exc:
        raise ImportError("PaCMAP is not installed. Install with: uv add pacmap") from exc

    X = np.asarray(X_reference, dtype=np.float32)
    n = X.shape[0]
    effective_neighbors = min(max(1, int(n_neighbors)), n - 1)

    layouts: dict[str, list[list[float]]] = {}
    previous: np.ndarray | None = None
    previous_init: np.ndarray | str = "pca"

    for maest_weight in weight_values:
        D = _combined_distance(D_maest, D_tempo, maest_weight=maest_weight)
        pair_neighbors = _neighbor_pairs_from_distance(D, n_neighbors=effective_neighbors)
        reducer = pacmap.PaCMAP(
            n_components=2,
            n_neighbors=effective_neighbors,
            MN_ratio=float(mn_ratio),
            FP_ratio=float(fp_ratio),
            pair_neighbors=pair_neighbors,
            distance=str(distance),
            random_state=int(random_state),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Warning: random state is set to.*")
            logging.getLogger("pacmap").setLevel(logging.ERROR)
            coords = np.asarray(reducer.fit_transform(X, init=previous_init), dtype=np.float32)
        if align and previous is not None:
            coords = _align_to_reference(previous, coords)
        previous = coords
        previous_init = coords
        key = f"{maest_weight:.1f}"
        layouts[key] = [[float(x), float(y)] for x, y in coords]

    return layouts


def export_interactive_pacmap_knn(
    *,
    project_root: Path = PROJECT_ROOT,
    mix_slugs: list[str] | None = None,
    output_file: Path | None = None,
    maest_dir: Path | None = None,
    chroma_dir: Path | None = None,
    tempo_dir: Path | None = None,
    random_state: int = 7777,
    n_neighbors: int = 10,
    mn_ratio: float = 0.5,
    fp_ratio: float = 1.5,
    distance: str = "angular",
    default_maest_weight: float = 1.0,
    align_layouts: bool = True,
    temperature: float = 0.08,
    tempo_bandwidth: float = 0.06,
    tempo_decay: float = 0.5,
    tempo_allow_octave: bool = True,
    tempo_octave_penalty: float = 0.5,
    tempo_similarity_shape: str = "gaussian",
    tempo_softflat_sharpness: float = 8.0,
    tempo_use_confidence: bool = False,
    snippet_seconds: float = 8.0,
    snippet_middle_fraction: float = 0.66,
    snippet_hop_seconds: float = 0.25,
    snippet_cache_overwrite: bool = False,
    top_k_rows: int = 25,
    background_links_per_song: int = 3,
    click_links_per_song: int = 5,
    bpm_color_scale_pct: float = 0.10,
    gif_output_file: Path | None = None,
    gif_renderer: str = "matplotlib",
    gif_fps: int = 8,
    gif_tween_frames: int = 6,
    gif_hold_frames: int = 4,
    open_browser: bool = False,
) -> Path:
    project_root = project_root.expanduser().resolve()
    mix_slugs = mix_slugs or ["aries-mix", "ara-mix"]
    maest_dir = (maest_dir or project_root / "data" / "maest_embeddings").expanduser().resolve()
    chroma_dir = (chroma_dir or project_root / "data" / "chroma_embeddings").expanduser().resolve()
    tempo_dir = (tempo_dir or project_root / "data" / "tempo_embeddings").expanduser().resolve()
    dataset_tag = "__".join(
        str((project_root / "music" / slug / f"{slug.replace('-', '_')}_tracks.csv").stem)
        for slug in mix_slugs
    )
    output_file = output_file or (
        project_root / "data" / "exports" / f"{dataset_tag}_interactive_pacmap_knn_maest_tempo.html"
    )
    output_file = output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    records, features = _load_combined_records_and_features(
        project_root=project_root,
        mix_slugs=mix_slugs,
        maest_dir=maest_dir,
        chroma_dir=chroma_dir,
        tempo_dir=tempo_dir,
        html_output_dir=output_file.parent,
        snippet_seconds=snippet_seconds,
        snippet_middle_fraction=snippet_middle_fraction,
        snippet_hop_seconds=snippet_hop_seconds,
        snippet_cache_overwrite=snippet_cache_overwrite,
    )
    matrices = _component_matrices(
        features,
        tempo_bandwidth=tempo_bandwidth,
        tempo_decay=tempo_decay,
        tempo_allow_octave=tempo_allow_octave,
        tempo_octave_penalty=tempo_octave_penalty,
        tempo_similarity_shape=tempo_similarity_shape,
        tempo_softflat_sharpness=tempo_softflat_sharpness,
        tempo_use_confidence=tempo_use_confidence,
    )

    weight_values = [round(i / 10.0, 1) for i in range(0, 11)]
    layouts = _compute_pacmap_knn_layouts(
        X_reference=features.maest,
        D_maest=matrices["maest_distance"],
        D_tempo=matrices["tempo_distance"],
        weight_values=weight_values,
        n_neighbors=n_neighbors,
        mn_ratio=mn_ratio,
        fp_ratio=fp_ratio,
        random_state=random_state,
        distance=distance,
        align=align_layouts,
    )
    initial_key = f"{round(float(np.clip(default_maest_weight, 0.0, 1.0)) * 10.0) / 10.0:.1f}"
    initial_coords = np.asarray(layouts[initial_key], dtype=np.float32)
    similarity_payload = _build_similarity_payload(
        records=records,
        maest_similarity=matrices["maest_similarity"],
        tempo_similarity=matrices["tempo_similarity"],
        temperature=temperature,
    )

    mix_title = " + ".join(mix_slugs)
    title = f"{mix_title}: Interactive PaCMAP from Custom MAEST/Tempo kNN"
    plot_div_id = f"{dataset_tag}_pacmap_knn_maest_tempo".replace("-", "_")
    plot_html = _build_plot(
        records,
        initial_coords,
        plot_div_id=plot_div_id,
        title=title,
        xaxis_title="PaCMAP-1 (custom kNN from weighted MAEST/tempo distance)",
        yaxis_title="PaCMAP-2 (custom kNN from weighted MAEST/tempo distance)",
    )
    html = _build_html(
        plot_html=plot_html,
        records=records,
        layouts=layouts,
        similarity_payload=similarity_payload,
        plot_div_id=plot_div_id,
        title=title,
        default_maest_weight=float(np.clip(default_maest_weight, 0.0, 1.0)),
        top_k_rows=top_k_rows,
        background_links_per_song=background_links_per_song,
        click_links_per_song=click_links_per_song,
        bpm_color_scale_pct=bpm_color_scale_pct,
        temperature=temperature,
        method_name="PaCMAP",
        layout_description="Layout interpolates between precomputed 10% PaCMAP runs with custom kNN pairs.",
    )
    output_file.write_text(html, encoding="utf-8")
    if gif_output_file is not None:
        _render_layout_gif(
            records=records,
            layouts=layouts,
            output_file=gif_output_file,
            title=title,
            renderer=gif_renderer,
            fps=gif_fps,
            tween_frames=gif_tween_frames,
            hold_frames=gif_hold_frames,
        )

    print(f"Loaded aligned tracks: {len(records)}")
    print(f"Computed PaCMAP kNN layouts: {len(layouts)} weight splits ({', '.join(layouts.keys())})")
    print(f"Neighbor pairs per layout: {len(records) * min(max(1, int(n_neighbors)), len(records) - 1)}")
    print(f"Standalone HTML saved to: {output_file}")
    if gif_output_file is not None:
        print(f"Animated GIF saved to: {gif_output_file.expanduser().resolve()}")
    if open_browser:
        import webbrowser

        webbrowser.open_new_tab(output_file.as_uri())
    return output_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an interactive PaCMAP HTML from custom MAEST/tempo kNN pairs."
    )
    parser.add_argument(
        "--mix-slug",
        action="append",
        dest="mix_slugs",
        help="Mix slug to include, e.g. aries-mix. May be passed multiple times. Defaults to aries-mix + ara-mix.",
    )
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--random-state", type=int, default=7777)
    parser.add_argument("--n-neighbors", type=int, default=10)
    parser.add_argument("--mn-ratio", type=float, default=0.5)
    parser.add_argument("--fp-ratio", type=float, default=1.5)
    parser.add_argument("--distance", default="angular", choices=["angular", "euclidean", "manhattan", "hamming", "dot"])
    parser.add_argument("--default-maest-weight", type=float, default=1.0)
    parser.add_argument("--no-align-layouts", action="store_true")
    parser.add_argument("--gif-output", type=Path, default=None, help="Optional path for an animated GIF export.")
    parser.add_argument("--gif-renderer", choices=["matplotlib", "plotly"], default="matplotlib")
    parser.add_argument("--gif-fps", type=int, default=8)
    parser.add_argument("--gif-tween-frames", type=int, default=6)
    parser.add_argument("--gif-hold-frames", type=int, default=4)
    parser.add_argument("--open", action="store_true", help="Open the exported HTML in a browser.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    export_interactive_pacmap_knn(
        mix_slugs=args.mix_slugs,
        output_file=args.output_file,
        random_state=args.random_state,
        n_neighbors=args.n_neighbors,
        mn_ratio=args.mn_ratio,
        fp_ratio=args.fp_ratio,
        distance=args.distance,
        default_maest_weight=args.default_maest_weight,
        align_layouts=not args.no_align_layouts,
        gif_output_file=args.gif_output,
        gif_renderer=args.gif_renderer,
        gif_fps=args.gif_fps,
        gif_tween_frames=args.gif_tween_frames,
        gif_hold_frames=args.gif_hold_frames,
        open_browser=bool(args.open),
    )


if __name__ == "__main__":
    main()
