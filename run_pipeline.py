"""
run_pipeline.py

High-Throughput Deep Learning Cell Segmentation & Spatial Analysis Pipeline
Master entry point: Orchestrates CUDA setup, optical preprocessing, DeepCell Mesmer inference,
WSI spatial tile stitching (NMS), GMM single-cell immunophenotyping,
spatial microenvironment graph modeling, Moran's I autocorrelation,
interactive HTML5 QC generation, and CytoMap spatial export.
"""

import os
import sys
import time
import json
import yaml
import argparse
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.gpu_utils import setup_cuda_environment, print_hardware_summary
from src.inference import DeepCellSegmentationPipeline
from src.illumination_correction import OpticalPreprocessor
from src.feature_extraction import extract_single_cell_features
from src.tile_stitcher import WSITileStitcher
from src.phenotyping import AutomatedCellPhenotyper
from src.spatial_analysis import SpatialNeighborhoodAnalyzer
from src.spatial_statistics import compute_morans_i, compute_ripleys_l, compute_microenvironment_diversity
from src.visualize_qc import generate_interactive_qc_dashboard
from src.cytomap_exporter import CytoMapExporter
from src.benchmark import PipelineProfiler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DeepCellPipeline")


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads YAML configuration file."""
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found at '{config_path}'. Using defaults.")
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="High-Throughput Deep Learning Cell Segmentation & Spatial Analysis Pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline_config.yaml",
        help="Path to pipeline YAML configuration file."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/tiles_export",
        help="Directory containing multichannel image tiles exported from QuPath."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/segmentation_output",
        help="Directory to save segmentation masks and feature tables."
    )
    parser.add_argument(
        "--sample-name",
        type=str,
        default="Sample_01",
        help="Identifier for the sample/tissue slide."
    )
    parser.add_argument(
        "--arcsinh",
        action="store_true",
        help="Apply arcsinh transformation to marker intensities for CytoMap."
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable automatic HTML QC dashboard generation."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Initialize high-precision profiler
    profiler = PipelineProfiler()
    profiler.start()

    print_hardware_summary(config)

    # 1. Setup CUDA / GPU Acceleration
    hw_cfg = config.get("hardware", {})
    t0 = time.perf_counter()
    setup_cuda_environment(
        gpu_id=hw_cfg.get("gpu_id", 0),
        allow_growth=hw_cfg.get("allow_growth", True),
        memory_limit_mb=hw_cfg.get("memory_limit_mb", None),
        mixed_precision=hw_cfg.get("mixed_precision", False)
    )
    profiler.record_stage("CUDA & Hardware Init", time.perf_counter() - t0)

    # 2. Check for QuPath Tile Metadata
    metadata_path = os.path.join(args.input_dir, "tiles_metadata.json")
    tile_offset_map = {}
    if os.path.exists(metadata_path):
        logger.info(f"Loading QuPath tile metadata from: {metadata_path}")
        with open(metadata_path, "r") as f:
            meta = json.load(f)
            for t in meta.get("tiles", []):
                tile_offset_map[t["tile_id"]] = (t.get("x", 0), t.get("y", 0))

    # 3. Optical Preprocessing Engine (Flat-field correction & glass exclusion)
    preprocessor = OpticalPreprocessor(rolling_ball_radius=50, min_tissue_fraction=0.03)

    # 4. Initialize Segmentation Engine
    t0 = time.perf_counter()
    pipeline = DeepCellSegmentationPipeline(config)
    profiler.record_stage("Model Weights Loading", time.perf_counter() - t0)

    # 5. Run Deep Learning Segmentation Inference
    t0 = time.perf_counter()
    tile_results = pipeline.process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    infer_time = time.perf_counter() - t0
    profiler.record_stage("Deep Learning Inference", infer_time)

    if not tile_results:
        logger.warning(
            f"No tiles were processed in '{args.input_dir}'. "
            f"Run synthetic generator: python sample_data/generate_synthetic_data.py"
        )
        return

    # 6. Extract Single-Cell Morphological & Intensity Features
    t0 = time.perf_counter()
    feature_dfs: List[pd.DataFrame] = []
    marker_names = config.get("features", {}).get("markers", None)
    pixel_size_um = config.get("imaging", {}).get("microns_per_pixel", 0.5)

    for res in tile_results:
        base_name = res["base_name"]
        offsets = tile_offset_map.get(base_name, (0, 0))

        df_tile = extract_single_cell_features(
            whole_cell_mask=res["whole_cell_mask"],
            nuclear_mask=res["nuclear_mask"],
            multichannel_image=res["full_image"],
            marker_names=marker_names,
            pixel_size_um=pixel_size_um,
            tile_offset_x=offsets[0],
            tile_offset_y=offsets[1]
        )

        if not df_tile.empty:
            df_tile["Tile_Source"] = base_name
            feature_dfs.append(df_tile)

    if not feature_dfs:
        logger.warning("No cells detected across any tiles.")
        return

    raw_combined_df = pd.concat(feature_dfs, ignore_index=True)
    profiler.record_stage("Feature Extraction", time.perf_counter() - t0)

    # 7. WSI Spatial Tile Stitching & Boundary Deduplication (Spatial NMS)
    stitching_cfg = config.get("stitching", {})
    if stitching_cfg.get("enabled", True):
        t0 = time.perf_counter()
        stitcher = WSITileStitcher(
            nms_distance_threshold_um=stitching_cfg.get("nms_distance_threshold_um", 4.0),
            pixel_size_um=pixel_size_um
        )
        combined_df = stitcher.deduplicate_cells_across_tiles(raw_combined_df)
        profiler.record_stage("WSI Boundary Stitching", time.perf_counter() - t0)
    else:
        combined_df = raw_combined_df
        combined_df["CellID"] = np.arange(1, len(combined_df) + 1)

    # 8. Automated GMM Immunophenotyping & Infiltration Scoring
    t0 = time.perf_counter()
    phenotyper = AutomatedCellPhenotyper()
    combined_df = phenotyper.classify_lineages(
        combined_df,
        tumor_marker="Mean_PanCK",
        immune_marker="Mean_CD45",
        prolif_marker="Mean_Ki67"
    )
    coords = combined_df[["XPos", "YPos"]].values
    infil_metrics = phenotyper.compute_infiltration_metrics(
        df=combined_df,
        coords=coords,
        radius_um=40.0,
        pixel_size_um=pixel_size_um
    )
    profiler.record_stage("GMM Phenotyping", time.perf_counter() - t0)

    # 9. Spatial Microenvironment & Cellular Neighborhood (CN) Modeling
    spatial_cfg = config.get("spatial", {})
    if spatial_cfg.get("enabled", True):
        t0 = time.perf_counter()
        analyzer = SpatialNeighborhoodAnalyzer(
            k_neighbors=spatial_cfg.get("k_neighbors", 20),
            radius_um=spatial_cfg.get("radius_um", 50.0),
            pixel_size_um=pixel_size_um
        )
        combined_df = analyzer.compute_cellular_neighborhoods(
            df=combined_df,
            n_clusters=spatial_cfg.get("n_neighborhood_clusters", 5)
        )
        profiler.record_stage("Spatial Graph Analytics", time.perf_counter() - t0)

    # 10. Spatial Point-Pattern Statistics: Moran's I Autocorrelation
    t0 = time.perf_counter()
    morans_results = {}
    marker_cols = [c for c in combined_df.columns if c.startswith("Mean_")]
    for m_col in marker_cols:
        marker_name = m_col.replace("Mean_", "")
        morans_results[marker_name] = compute_morans_i(
            values=combined_df[m_col].values,
            coords=coords,
            distance_threshold=60.0
        )
    profiler.record_stage("Moran's I Autocorrelation", time.perf_counter() - t0)

    # Register volume to profiler
    first_tile_shape = tile_results[0]["full_image"].shape
    profiler.register_tiles(
        num_tiles=len(tile_results),
        tile_height=first_tile_shape[0],
        tile_width=first_tile_shape[1],
        cells=len(combined_df)
    )

    # 11. Save Comprehensive Single-Cell Measurement Matrix
    all_measurements_csv = os.path.join(args.output_dir, f"{args.sample_name}_all_measurements.csv")
    combined_df.to_csv(all_measurements_csv, index=False)
    logger.info(f"Saved complete cell measurements matrix ({len(combined_df)} cells) to: {all_measurements_csv}")

    # 12. Format & Export for CytoMap
    cytomap_cfg = config.get("cytomap", {})
    exporter = CytoMapExporter(
        x_col=cytomap_cfg.get("x_coord_col", "XPos"),
        y_col=cytomap_cfg.get("y_coord_col", "YPos"),
        z_col=cytomap_cfg.get("z_coord_col", "ZPos"),
        cell_id_col=cytomap_cfg.get("cell_id_col", "CellID")
    )
    cytomap_csv = os.path.join(args.output_dir, f"{args.sample_name}_cytomap_input.csv")
    exporter.export(
        df=combined_df,
        output_filepath=cytomap_csv,
        sample_name=args.sample_name,
        apply_arcsinh=args.arcsinh
    )

    # 13. Generate Interactive HTML Spatial QC Dashboard
    dashboard_path = None
    if not args.no_dashboard and config.get("reporting", {}).get("generate_dashboard", True):
        t0 = time.perf_counter()
        dashboard_filename = config.get("reporting", {}).get("dashboard_filename", "spatial_qc_report.html")
        dashboard_path = os.path.join(args.output_dir, dashboard_filename)
        generate_interactive_qc_dashboard(
            df=combined_df,
            output_html_path=dashboard_path,
            sample_name=args.sample_name,
            spatial_stats={"morans_i": morans_results},
            infiltration_stats=infil_metrics
        )
        profiler.record_stage("HTML Dashboard Gen", time.perf_counter() - t0)

    # 14. Print Benchmarking & Performance Report
    benchmark_metrics = profiler.print_benchmark_report()

    print("\n" + "=" * 65)
    print(f" PIPELINE COMPLETED SUCCESSFULLY FOR: {args.sample_name}")
    print("=" * 65)
    print(f" Total Unique Cells      : {len(combined_df):,}")
    print(f" Infiltration Score      : {infil_metrics.get('infiltration_score', 0.0):.1%}")
    print(f" Immune-to-Tumor Ratio   : {infil_metrics.get('immune_to_tumor_ratio', 0.0):.2f}")
    print(f" Segmentation Masks      : {args.output_dir}")
    print(f" Comprehensive CSV       : {all_measurements_csv}")
    print(f" CytoMap Import File     : {cytomap_csv}")
    if dashboard_path:
        print(f" Interactive QC Report   : {dashboard_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
