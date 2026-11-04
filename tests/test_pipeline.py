"""
test_pipeline.py

Comprehensive unit and integration tests for the DeepCell cell segmentation,
optical preprocessing, Spatial NMS, GMM phenotyping, and spatial statistics pipeline.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
import tifffile

from src.gpu_utils import setup_cuda_environment
from src.inference import DeepCellSegmentationPipeline
from src.illumination_correction import OpticalPreprocessor
from src.feature_extraction import extract_single_cell_features
from src.tile_stitcher import WSITileStitcher
from src.phenotyping import AutomatedCellPhenotyper
from src.spatial_analysis import SpatialNeighborhoodAnalyzer
from src.spatial_statistics import compute_morans_i, compute_ripleys_l, extract_delaunay_contact_network
from src.visualize_qc import generate_interactive_qc_dashboard
from src.cytomap_exporter import CytoMapExporter
from src.benchmark import PipelineProfiler


class TestDeepCellPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.tile_path = os.path.join(self.test_dir, "test_tile_x0_y0.tif")

        # Generate mock 2-channel tile (Nuclear + Membrane)
        h, w = 128, 128
        nuc = np.zeros((h, w), dtype=np.uint16)
        mem = np.zeros((h, w), dtype=np.uint16)

        yy, xx = np.ogrid[:h, :w]
        dist_sq = (xx - 64)**2 + (yy - 64)**2
        nuc[dist_sq <= 16] = 3000
        mem[(dist_sq <= 49) & (dist_sq > 16)] = 2500

        mock_multichannel = np.stack([nuc, mem], axis=-1)
        tifffile.imwrite(self.tile_path, mock_multichannel)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_cuda_environment_setup(self):
        """Verifies CUDA initialization runs without unhandled exceptions."""
        status = setup_cuda_environment(gpu_id=0, allow_growth=True)
        self.assertIsInstance(status, dict)
        self.assertIn("cuda_available", status)

    def test_optical_preprocessing(self):
        """Verifies tissue foreground detection and rolling-ball background subtraction."""
        pre = OpticalPreprocessor(rolling_ball_radius=20, min_tissue_fraction=0.01)
        raw = tifffile.imread(self.tile_path)
        proc_img, is_valid, coverage = pre.preprocess_multichannel_tile(raw, nuclear_idx=0, membrane_idx=1)

        self.assertTrue(is_valid)
        self.assertGreater(coverage, 0.0)
        self.assertEqual(proc_img.shape, raw.shape)
        self.assertTrue(np.all(proc_img >= 0.0) and np.all(proc_img <= 1.0))

    def test_tile_preparation(self):
        """Verifies tile reading and channel mapping."""
        cfg = {"imaging": {"nuclear_channel_index": 0, "membrane_channel_index": 1}}
        pipeline = DeepCellSegmentationPipeline(cfg)
        mesmer_in, full_img = pipeline.prepare_tile_input(self.tile_path)

        self.assertEqual(mesmer_in.shape, (128, 128, 2))
        self.assertEqual(full_img.shape, (128, 128, 2))

    def test_feature_extraction(self):
        """Verifies single-cell morphology and intensity extraction."""
        cell_mask = np.zeros((128, 128), dtype=np.uint32)
        nuc_mask = np.zeros((128, 128), dtype=np.uint32)

        yy, xx = np.ogrid[:128, :128]
        dist_sq = (xx - 64)**2 + (yy - 64)**2
        cell_mask[dist_sq <= 49] = 1
        nuc_mask[dist_sq <= 16] = 1

        mock_img = np.zeros((128, 128, 2), dtype=np.float32)
        mock_img[dist_sq <= 16, 0] = 3000
        mock_img[dist_sq <= 49, 1] = 2500

        df = extract_single_cell_features(
            whole_cell_mask=cell_mask,
            nuclear_mask=nuc_mask,
            multichannel_image=mock_img,
            marker_names=["DAPI", "PanCK"],
            pixel_size_um=0.5,
            tile_offset_x=100,
            tile_offset_y=200
        )

        self.assertFalse(df.empty)
        self.assertEqual(len(df), 1)
        self.assertIn("CellID", df.columns)
        self.assertIn("XPos", df.columns)
        self.assertIn("YPos", df.columns)
        self.assertIn("Cell_Area_Pixels", df.columns)
        self.assertIn("Mean_DAPI", df.columns)
        self.assertIn("Mean_PanCK", df.columns)
        self.assertAlmostEqual(df["XPos"].iloc[0], 164.0, delta=1.0)
        self.assertAlmostEqual(df["YPos"].iloc[0], 264.0, delta=1.0)

    def test_spatial_nms_tile_stitching(self):
        """Verifies removal of boundary cell duplicates across adjacent tiles."""
        df_duplicate_cells = pd.DataFrame({
            "CellID": [1, 2, 3],
            "Tile_Source": ["Tile_A", "Tile_B", "Tile_B"],
            "XPos": [100.0, 101.0, 500.0],
            "YPos": [100.0, 101.0, 500.0],
            "Cell_Area_Pixels": [150, 120, 200]
        })

        stitcher = WSITileStitcher(nms_distance_threshold_um=4.0, pixel_size_um=0.5)
        dedup_df = stitcher.deduplicate_cells_across_tiles(df_duplicate_cells)

        self.assertEqual(len(dedup_df), 2)
        self.assertTrue((dedup_df["Tile_Source"] == "Tile_A").any())
        self.assertEqual(dedup_df["CellID"].tolist(), [1, 2])

    def test_gmm_phenotyping_and_infiltration(self):
        """Verifies probabilistic lineage gating and infiltration quantification."""
        df = pd.DataFrame({
            "CellID": [1, 2, 3, 4],
            "XPos": [10.0, 20.0, 100.0, 110.0],
            "YPos": [10.0, 20.0, 100.0, 110.0],
            "Mean_PanCK": [2500.0, 2800.0, 100.0, 120.0],  # Cells 1 & 2 are tumor
            "Mean_CD45": [50.0, 60.0, 3200.0, 2900.0],     # Cells 3 & 4 are immune
            "Mean_Ki67": [1500.0, 200.0, 100.0, 1800.0]
        })

        phenotyper = AutomatedCellPhenotyper()
        res_df = phenotyper.classify_lineages(df)

        self.assertIn("Cell_Phenotype", res_df.columns)
        self.assertTrue((res_df["Cell_Phenotype"].str.contains("Tumor")).any())
        self.assertTrue((res_df["Cell_Phenotype"].str.contains("Immune")).any())

        coords = res_df[["XPos", "YPos"]].values
        infil = phenotyper.compute_infiltration_metrics(res_df, coords, radius_um=50.0, pixel_size_um=0.5)
        self.assertIn("infiltration_score", infil)
        self.assertIn("immune_to_tumor_ratio", infil)

    def test_morans_i_and_spatial_stats(self):
        """Verifies Moran's I spatial autocorrelation and Delaunay contact graph."""
        np.random.seed(42)
        coords = np.array([
            [0.0, 0.0], [5.0, 5.0], [10.0, 0.0],
            [100.0, 100.0], [105.0, 105.0], [110.0, 100.0]
        ])
        # Spatially clustered values: high at cluster 1, low at cluster 2
        vals = np.array([1000.0, 950.0, 980.0, 50.0, 60.0, 45.0])

        res = compute_morans_i(vals, coords, distance_threshold=20.0, permutations=49)
        self.assertIn("morans_i", res)
        self.assertGreater(res["morans_i"], 0.0)  # Positive spatial clustering

        edges, lengths = extract_delaunay_contact_network(coords, max_edge_length_um=25.0)
        self.assertIsInstance(edges, np.ndarray)

    def test_interactive_dashboard_generation(self):
        """Verifies self-contained HTML QC report generation."""
        df = pd.DataFrame({
            "CellID": [1, 2, 3],
            "XPos": [50.0, 150.0, 250.0],
            "YPos": [50.0, 150.0, 250.0],
            "Cell_Area_um2": [50.0, 65.0, 70.0],
            "Nuclear_to_Cell_Ratio": [0.35, 0.40, 0.38],
            "Cell_Phenotype": ["Epithelial_Tumor", "Leukocyte_Immune", "Stroma_Unlabeled"],
            "Mean_DAPI": [800.0, 1200.0, 950.0],
            "Mean_PanCK": [400.0, 1100.0, 750.0],
            "Mean_CD45": [100.0, 3200.0, 200.0]
        })

        out_html = os.path.join(self.test_dir, "test_report.html")
        generate_interactive_qc_dashboard(df, out_html, sample_name="DashboardTest")

        self.assertTrue(os.path.exists(out_html))
        with open(out_html, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Spatial Biology & Cell Segmentation", content)
        self.assertIn("DashboardTest", content)

    def test_cytomap_export(self):
        """Verifies CytoMap table formatting and metadata output."""
        df = pd.DataFrame({
            "CellID": [1, 2],
            "XPos": [10.5, 20.5],
            "YPos": [30.0, 40.0],
            "Mean_DAPI": [500.0, 1200.0],
            "Mean_PanCK": [200.0, 800.0],
            "Cell_Area_Pixels": [120, 150]
        })

        exporter = CytoMapExporter()
        out_csv = os.path.join(self.test_dir, "cytomap_test.csv")
        exporter.export(df, out_csv, sample_name="TestSample", apply_arcsinh=True)

        self.assertTrue(os.path.exists(out_csv))
        loaded = pd.read_csv(out_csv)
        self.assertEqual(len(loaded), 2)
        self.assertIn("CellID", loaded.columns)
        self.assertIn("XPos", loaded.columns)
        self.assertIn("SampleName", loaded.columns)

    def test_evaluation_metrics_pq_and_aji(self):
        """Verifies Panoptic Quality (PQ) and Aggregated Jaccard Index (AJI) benchmarks."""
        from src.evaluation_metrics import compute_panoptic_quality, compute_aggregated_jaccard_index

        gt = np.zeros((100, 100), dtype=np.uint32)
        pred = np.zeros((100, 100), dtype=np.uint32)

        # Ground truth cell at (50, 50) radius 10
        yy, xx = np.ogrid[:100, :100]
        gt[(xx - 50)**2 + (yy - 50)**2 <= 100] = 1

        # Prediction overlapping (52, 50) radius 10
        pred[(xx - 52)**2 + (yy - 50)**2 <= 100] = 1

        pq_res = compute_panoptic_quality(gt, pred, iou_threshold=0.5)
        aji = compute_aggregated_jaccard_index(gt, pred)

        self.assertIn("pq", pq_res)
        self.assertIn("sq", pq_res)
        self.assertIn("rq", pq_res)
        self.assertGreater(pq_res["pq"], 0.5)
        self.assertGreater(aji, 0.5)
        self.assertEqual(pq_res["tp"], 1)


if __name__ == "__main__":
    unittest.main()
