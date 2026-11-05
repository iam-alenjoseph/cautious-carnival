"""
schemas.py

Pydantic schemas for the DeepCell Bioimaging & Spatial Biology REST API.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class GPUStatusResponse(BaseModel):
    cuda_available: bool
    gpu_count: int
    active_device: str
    memory_limit_mb: Optional[int]
    memory_growth_enabled: bool


class CellMeasurement(BaseModel):
    cell_id: int
    x_pos: float
    y_pos: float
    z_pos: float = 0.0
    cell_area_um2: float
    nuclear_area_um2: float
    nuclear_to_cell_ratio: float
    cell_phenotype: Optional[str] = "Unlabeled"
    cellular_neighborhood_id: Optional[int] = 0
    marker_intensities: Dict[str, float] = Field(default_factory=dict)


class SegmentationResponse(BaseModel):
    sample_name: str
    total_cells: int
    mean_cell_area_um2: float
    mean_nc_ratio: float
    processing_time_sec: float
    cells: List[CellMeasurement]
    morphology_summary: Dict[str, float]


class SpatialAnalysisRequest(BaseModel):
    sample_name: str = "Sample_01"
    k_neighbors: int = 20
    radius_um: float = 50.0
    pixel_size_um: float = 0.5
    cells: List[Dict[str, Any]]


class SpatialAnalysisResponse(BaseModel):
    sample_name: str
    total_cells_analyzed: int
    morans_i_statistics: Dict[str, Dict[str, float]]
    neighborhood_clusters: int
    shannon_entropy: float
    simpson_diversity: float

