"""
app.py

Production-grade FastAPI inference microservice for the DeepCell Bioimaging Platform.
Exposes GPU-accelerated endpoints for instance segmentation, single-cell feature extraction,
and spatial point-pattern analytics.
"""

import os
import io
import time
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import tifffile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from src.gpu_utils import setup_cuda_environment
from src.inference import DeepCellSegmentationPipeline
from src.feature_extraction import extract_single_cell_features
from src.phenotyping import AutomatedCellPhenotyper
from src.spatial_analysis import SpatialNeighborhoodAnalyzer
from src.spatial_statistics import compute_morans_i, compute_microenvironment_diversity
from api.schemas import GPUStatusResponse, SegmentationResponse, CellMeasurement, SpatialAnalysisRequest, SpatialAnalysisResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DeepCellAPI")

app = FastAPI(
    title="DeepCell Bioimaging & Spatial Biology API",
    description="Enterprise GPU-accelerated cell segmentation and spatial microenvironment analytics service.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for web portals and UI dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model container
pipeline_engine: DeepCellSegmentationPipeline = None
gpu_config_status: Dict[str, Any] = {}


@app.on_event("startup")
async def startup_event():
    """Initializes CUDA hardware environment and loads DeepCell model into VRAM."""
    global pipeline_engine, gpu_config_status
    logger.info("Initializing DeepCell API Service and CUDA hardware...")
    gpu_config_status = setup_cuda_environment(gpu_id=0, allow_growth=True)
    cfg = {
        "imaging": {"nuclear_channel_index": 0, "membrane_channel_index": 1, "microns_per_pixel": 0.5},
        "inference": {"compartment": "both", "batch_size": 2}
    }
    pipeline_engine = DeepCellSegmentationPipeline(cfg)
    logger.info("DeepCell Model Engine ready for inference requests.")


@app.get("/health", response_model=GPUStatusResponse, tags=["System Health"])
async def health_check():
    """Returns GPU accelerator status, device visibility, and VRAM memory telemetry."""
    return GPUStatusResponse(
        cuda_available=gpu_config_status.get("cuda_available", False),
        gpu_count=gpu_config_status.get("gpu_count", 0),
        active_device=gpu_config_status.get("device_name", "CPU"),
        memory_limit_mb=gpu_config_status.get("memory_limit_mb"),
        memory_growth_enabled=gpu_config_status.get("memory_growth", True)
    )


@app.post("/v1/segment", response_model=SegmentationResponse, tags=["Inference"])
async def segment_tile(
    file: UploadFile = File(..., description="Multichannel TIFF image tile (e.g. DAPI + PanCK)"),
    sample_name: str = Form("Sample_01"),
    nuclear_channel_idx: int = Form(0),
    membrane_channel_idx: int = Form(1),
    pixel_size_um: float = Form(0.5)
):
    """
    Executes DeepCell Mesmer deep learning segmentation on an uploaded TIFF tile,
    classifies biological phenotypes, and extracts morphology/intensity features.
    """
    t_start = time.perf_counter()

    try:
        contents = await file.read()
        raw_image = tifffile.imread(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    # Standardize channels
    if raw_image.ndim == 2:
        multichannel = np.stack([raw_image, raw_image], axis=-1)
    elif raw_image.ndim == 3 and raw_image.shape[0] < raw_image.shape[2]:
        multichannel = np.moveaxis(raw_image, 0, -1)
    else:
        multichannel = raw_image

    n_dim = multichannel.shape[-1]
    n_idx = min(nuclear_channel_idx, n_dim - 1)
    m_idx = min(membrane_channel_idx, n_dim - 1)
    mesmer_input = np.stack([multichannel[..., n_idx], multichannel[..., m_idx]], axis=-1).astype(np.float32)

    # Inference
    whole_cell_mask, nuclear_mask = pipeline_engine.predict_tile(mesmer_input)

    # Feature quantification
    df = extract_single_cell_features(
        whole_cell_mask=whole_cell_mask,
        nuclear_mask=nuclear_mask,
        multichannel_image=multichannel,
        marker_names=["DAPI", "PanCK", "CD45", "Ki67"][:n_dim],
        pixel_size_um=pixel_size_um
    )

    if df.empty:
        return SegmentationResponse(
            sample_name=sample_name,
            total_cells=0,
            mean_cell_area_um2=0.0,
            mean_nc_ratio=0.0,
            processing_time_sec=time.perf_counter() - t_start,
            cells=[],
            morphology_summary={}
        )

    # Phenotyping
    phenotyper = AutomatedCellPhenotyper()
    df = phenotyper.classify_lineages(df)

    cells_out: List[CellMeasurement] = []
    marker_cols = [c for c in df.columns if c.startswith("Mean_")]

    for _, row in df.iterrows():
        intensities = {c.replace("Mean_", ""): float(row[c]) for c in marker_cols}
        cells_out.append(CellMeasurement(
            cell_id=int(row["CellID"]),
            x_pos=float(row["XPos"]),
            y_pos=float(row["YPos"]),
            z_pos=float(row.get("ZPos", 0.0)),
            cell_area_um2=float(row["Cell_Area_um2"]),
            nuclear_area_um2=float(row["Nucleus_Area_um2"]),
            nuclear_to_cell_ratio=float(row["Nuclear_to_Cell_Ratio"]),
            cell_phenotype=str(row.get("Cell_Phenotype", "Unlabeled")),
            marker_intensities=intensities
        ))

    elapsed = time.perf_counter() - t_start
    return SegmentationResponse(
        sample_name=sample_name,
        total_cells=len(cells_out),
        mean_cell_area_um2=float(df["Cell_Area_um2"].mean()),
        mean_nc_ratio=float(df["Nuclear_to_Cell_Ratio"].mean()),
        processing_time_sec=float(elapsed),
        cells=cells_out,
        morphology_summary={
            "mean_cell_area_um2": float(df["Cell_Area_um2"].mean()),
            "std_cell_area_um2": float(df["Cell_Area_um2"].std() if len(df) > 1 else 0.0),
            "mean_perimeter": float(df["Cell_Perimeter"].mean())
        }
    )


@app.post("/v1/spatial/analyze", response_model=SpatialAnalysisResponse, tags=["Spatial Biology"])
async def analyze_spatial_patterns(req: SpatialAnalysisRequest):
    """
    Computes Cellular Neighborhood (CN) motifs, Global Moran's I spatial autocorrelation,
    and Shannon/Simpson microenvironment diversity on single-cell coordinate matrices.
    """
    if not req.cells:
        raise HTTPException(status_code=400, detail="Empty cell dataset provided.")

    df = pd.DataFrame(req.cells)
    if "XPos" not in df.columns or "YPos" not in df.columns:
        raise HTTPException(status_code=400, detail="Missing required spatial columns: 'XPos' and 'YPos'")

    analyzer = SpatialNeighborhoodAnalyzer(
        k_neighbors=req.k_neighbors,
        radius_um=req.radius_um,
        pixel_size_um=req.pixel_size_um
    )
    df = analyzer.compute_cellular_neighborhoods(df, n_clusters=5)

    coords = df[["XPos", "YPos"]].values
    morans_results = {}
    marker_cols = [c for c in df.columns if c.startswith("Mean_")]

    for col in marker_cols:
        m_name = col.replace("Mean_", "")
        morans_results[m_name] = compute_morans_i(df[col].values, coords, distance_threshold=60.0)

    diversity = compute_microenvironment_diversity(df["Cellular_Neighborhood_ID"].values)

    return SpatialAnalysisResponse(
        sample_name=req.sample_name,
        total_cells_analyzed=len(df),
        morans_i_statistics=morans_results,
        neighborhood_clusters=int(df["Cellular_Neighborhood_ID"].nunique()),
        shannon_entropy=diversity["shannon_entropy"],
        simpson_diversity=diversity["simpson_diversity"]
    )

