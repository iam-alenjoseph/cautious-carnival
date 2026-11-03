"""
benchmark.py

Performance profiling & throughput benchmarking for the DeepCell pipeline.
Measures tile processing velocity, GigaPixels/hour throughput, and GPU utilization.
"""

import time
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class PipelineProfiler:
    """Instruments bioimaging execution stages to measure throughput and latency."""

    def __init__(self):
        self.start_time = None
        self.stage_times: Dict[str, float] = {}
        self.tile_count = 0
        self.total_pixels = 0
        self.total_cells = 0

    def start(self) -> None:
        """Starts overall execution stopwatch."""
        self.start_time = time.perf_counter()

    def record_stage(self, stage_name: str, duration_sec: float) -> None:
        """Logs elapsed duration for a specific pipeline stage."""
        self.stage_times[stage_name] = self.stage_times.get(stage_name, 0.0) + duration_sec

    def register_tiles(self, num_tiles: int, tile_height: int, tile_width: int, cells: int) -> None:
        """Registers volume metrics for throughput computations."""
        self.tile_count += num_tiles
        self.total_pixels += num_tiles * tile_height * tile_width
        self.total_cells += cells

    def print_benchmark_report(self) -> Dict[str, Any]:
        """Calculates and outputs high-level throughput analytics."""
        total_wall_time = time.perf_counter() - self.start_time if self.start_time else 1.0
        gigapixels = self.total_pixels / 1e9
        gigapixels_per_hour = (gigapixels / max(total_wall_time, 0.001)) * 3600.0
        cells_per_second = self.total_cells / max(total_wall_time, 0.001)

        print("\n" + "=" * 65)
        print(" PIPELINE THROUGHPUT & HARDWARE BENCHMARK REPORT")
        print("=" * 65)
        print(f" Total Wall-Clock Time   : {total_wall_time:.2f} seconds")
        print(f" Image Tiles Processed   : {self.tile_count}")
        print(f" Total Pixels Segmented  : {self.total_pixels:,} ({gigapixels:.3f} GP)")
        print(f" Total Cells Identified  : {self.total_cells:,}")
        print(f" Processing Speed (FPS)  : {self.tile_count / max(total_wall_time, 0.001):.2f} tiles/sec")
        print(f" Cellular Throughput     : {cells_per_second:.1f} cells/sec")
        print(f" High-Throughput Metric  : {gigapixels_per_hour:.2f} GigaPixels / hour")
        print("-" * 65)
        for stage, duration in self.stage_times.items():
            pct = (duration / max(total_wall_time, 0.001)) * 100
            print(f"  - {stage:<25}: {duration:.2f}s ({pct:.1f}%)")
        print("=" * 65 + "\n")

        return {
            "wall_time_sec": total_wall_time,
            "tile_count": self.tile_count,
            "total_pixels": self.total_pixels,
            "total_cells": self.total_cells,
            "gigapixels_per_hour": gigapixels_per_hour,
            "cells_per_sec": cells_per_second
        }

