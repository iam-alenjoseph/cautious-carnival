# Enterprise Deep Learning Bioimaging & Spatial Biology Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%20%7C%203.10-3776AB.svg?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-11.8%20%7C%2012.x-76B900.svg?style=for-the-badge&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.10-FF6F00.svg?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/DeepCell-Mesmer-79529F.svg?style=for-the-badge" />
  <img src="https://img.shields.io/badge/FastAPI-Production%20REST-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-GPU%20Ready-2496ED.svg?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/QuPath-Headless%20Groovy-2E7D32.svg?style=for-the-badge" />
  <img src="https://img.shields.io/badge/CytoMap-Spatial%20Biology-D32F2F.svg?style=for-the-badge" />
</p>

An end-to-end deep learning platform for multiplexed digital pathology, combining automated cell segmentation, immunophenotyping, and spatial biology analysis.

Built for CODEX/PhenoCycler, multiplex immunofluorescence (mIF), IMC, and MIBI datasets.

---

## Features

- **Image Preprocessing:** Background subtraction, illumination correction, tissue detection, and intensity normalization.
- **Cell Segmentation:** ResNet50-FPN Mesmer architecture for nuclear and whole-cell segmentation.
- **GPU Optimization:** CUDA memory management and CPU fallback.
- **Tile Stitching:** KD-Tree-based spatial matching and boundary deduplication (Spatial NMS).
- **Immunophenotyping:** GMM-based marker classification and tumor-immune infiltration scoring.
- **Spatial Analysis:** Moran's I autocorrelation, Ripley's L, Delaunay triangulation, and cellular neighborhoods.
- **Evaluation:** Panoptic Quality (PQ), AJI, SQ, RQ, Dice, and F1 metrics.
- **QuPath Integration:** Automated tile export and segmentation mask import scripts.
- **REST API:** FastAPI endpoints with OpenAPI Swagger documentation.
- **Visualization:** Interactive HTML5/WebGL quality-control dashboard with spatial map.
- **Deployment:** Docker, Docker Compose, and automated GitHub Actions CI/CD.

---

## Installation

### Conda Environment (Recommended)

```bash
git clone https://github.com/iam-alenjoseph/cautious-carnival.git
cd cautious-carnival

conda env create -f environment.yml
conda activate deepcell-pipeline
```

### Alternatively with Pip

```bash
pip install -r requirements.txt
```

---

## Usage

### Hardware Information
```bash
python cli.py hardware
```

### Generate Synthetic Data
```bash
python cli.py synthetic
```

### Run Full Pipeline
```bash
python cli.py run \
    --input-dir data/tiles_export \
    --output-dir data/segmentation_output \
    --sample-name Biopsy_01 \
    --arcsinh
```

### Evaluate Segmentation Against Ground Truth
```bash
python cli.py evaluate \
    --ground-truth path/to/gt_mask.tif \
    --prediction path/to/pred_mask.tif
```

### Start REST API Microservice
```bash
python cli.py serve --host 0.0.0.0 --port 8000
```
Interactive OpenAPI Swagger documentation: `http://localhost:8000/docs`

---

## Docker Deployment

```bash
docker compose up -d --build
```

---

## Testing

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
