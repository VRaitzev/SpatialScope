# SpatialScope

**A web platform for multimodal spatial transcriptomics: whole-slide histology + per-cell boundaries + spatial gene expression (10x Genomics Xenium).**

> **Status: working engineering platform, models not yet trained on real data.**
> The architecture is implemented end to end: slide ingestion, cell-graph construction, a GATv2 expression predictor, interpretability, training and inference, a REST API and a web UI. No model has been trained or evaluated on real data yet, and the only slide available so far is a single demo Xenium sample, so **the platform makes no accuracy claims**. It was built as a BSc thesis project (Tambov State Technical University, 2026) and is the foundation for the research direction described below.

---

## Motivation

Spatial transcriptomics gives two different trade-offs:

- **Visium** measures the full transcriptome (~33,000 genes) but at low resolution: one spot covers roughly 10 to 20 cells.
- **Xenium** gives true single-cell resolution with cell boundaries, but only for a panel of about 300 to 500 genes.

A central question is how much of a cell's molecular profile can be recovered from its morphology and spatial context. SpatialScope is the engineering backbone for studying that question on Xenium data: it registers a slide as one multimodal object, builds a graph of cells, predicts gene expression with a graph neural network, and explains each prediction.

## What the platform does

- **Slide registration.** A Xenium slide is registered from three files: the whole-slide image (`.tif` / `.ome.tif`), the per-cell table with cell boundaries (`.parquet`), and the expression matrix (`.hdf5`).
- **Slide viewing and ROI selection.** Browse a slide with its metadata and select a region of interest.
- **Spatial inference.** For a selected region, extract per-cell features, build the cell graph, and predict gene expression, aggregated to spots.
- **Interpretability.** Per-cell importance maps for each prediction (see below).
- **Reports.** Results are stored per slide, can be reopened later, and exported as JSON.
- **Background training.** Retrain the spatial model on regions of a registered slide and poll the job status.
- **Model presets.** Combine feature experts and a spatial model into named presets and switch the active one.
- **Accounts and roles.** JWT authentication, role field on users, audit log of actions.

## How it works

```
Xenium slide
  WSI (.ome.tif)  +  cell table with boundaries (.parquet)  +  expression matrix (.hdf5)
                              |
                              v
        Cell boundaries from the Xenium data (per region / patch)
                              |
                              v
          Plug-in expert models (per-cell features, concatenated)
        EfficientNet-B0 embedding | Geometric morphology | Radiomics
                              |
                              v
        Cell graph: Delaunay triangulation, nodes = cells, self-loops
                              |
                              v
        GATv2 predictor (2 layers, PyTorch Geometric) -> expression per cell
                              |
                              v
        Aggregation of cells to spots (radius search) + node-level interpretation
```

**Expert models (default presets).**
- *EfficientNet-B0* (ImageNet-pretrained) cell embeddings from the histology patch.
- *Geometric morphology*: cell area, perimeter, aspect ratio, solidity, mean distance to nearest neighbors, local density.
- *Radiomics* features per cell (PyRadiomics).
- Two presets are defined: "EfficientNet + Geometry + Radiomics + GNN" and "Radiomics + GNN".

**Spatial model.** A two-layer GATv2 network (4 attention heads in the first layer) predicts expression for every cell from its expert features and its neighbors. The default configuration predicts a small immune-marker gene set (CD3D, CD68, CD4, CD8A, FOXP3, PDCD1, CTLA4).

**Interpretability.** For the GNN, node importance combines the gradient of the target gene with respect to the input features (weight 0.7) and the mean attention weights of the first GAT layer (weight 0.3), normalized per patch. For the EfficientNet expert, a Grad-CAM-style saliency map is produced. Feature-level attributions currently use feature indices and are not yet mapped to named biological features.

**Training.** Mean squared error against the per-cell Xenium panel expression, Adam optimizer, checkpoints saved per preset, run in a background thread. Job status is kept in memory.

**Code layout.**
- `backend/app`: FastAPI service (loaders, expert models, spatial model, interpretability, pipelines, orchestrator, preset registry, SQLAlchemy persistence with SQLite). Components sit behind abstract interfaces so that models can be swapped without changing the pipeline.
- `frontend/`: Vue 3 + Vite + Tailwind CSS (login and registration, dashboard, slide viewer, results).
- `docs/`: full project description and UML diagrams (PlantUML sources in `docs/uml/`).

## API overview

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/users/register` | Create a user (currently unauthenticated, see Limitations) |
| POST | `/api/users/login` | Log in, receive a JWT |
| POST / GET | `/api/projects` | Create / list projects |
| POST | `/api/slides/register` | Register a Xenium slide |
| GET | `/api/slides/{slide_id}` | Slide metadata |
| POST | `/api/inference/run` | Run inference on a region |
| POST | `/api/training/start` | Start a background training job |
| GET | `/api/training/status/{job_id}` | Training job status |
| GET | `/api/presets` | List model presets |
| POST | `/api/presets/activate` | Activate a preset |
| GET | `/api/reports/slide/{slide_id}` | Reports for a slide |
| GET | `/api/reports/{report_id}` | A single report |
| GET | `/api/reports/{report_id}/export/json` | Export a report as JSON |

## Getting started

**Requirements:** Python 3.12 or newer, [uv](https://docs.astral.sh/uv/), Node.js. A GPU is recommended (the project pins PyTorch CUDA 12.8 wheels).

**Backend**

```bash
cd backend
uv sync
export SECRET_KEY="a-long-random-string"
uv run python -m app.main        # http://localhost:8000
```

**Frontend**

```bash
cd frontend
npm install
npm run dev                      # http://localhost:5173
```

The frontend reads the API address from `VITE_API_BASE_URL` (default `http://localhost:8000/api`). Slide files are referenced by path on the machine running the backend.

## Research direction

The next step is to train and validate the pipeline on public paired histology and spatial-expression data, and to make interpretability a property of every stage rather than an add-on:

1. Augment rare tumor subtypes in the image domain.
2. Embed each cell or patch with a foundation model (DINOv2, UNI, CONCH).
3. Interpret the embedding with two parallel mechanisms: a concept bottleneck model, and a modern Hopfield associative memory for rare classes.
4. Build an enriched cell graph whose nodes carry interpretable feature vectors instead of raw embeddings.
5. Predict spatial gene expression with a GNN, with the prediction traceable through every earlier stage.

The first validation target is the GNN stage alone, trained on a graph enriched only with concept-bottleneck features, to test whether concept-level features are competitive with raw embeddings before adding the other stages. See the accompanying research proposal for the motivation, related work and plan.

## Limitations

- No model is trained or evaluated on real data yet. Training currently has no held-out validation split, so reported losses would be training losses only.
- Only one demo Xenium slide is available.
- The user registration endpoint is open and accepts a role field. It is meant for local development and must be restricted before any shared deployment. The default `SECRET_KEY` in the code is a placeholder and must be overridden.
- Only Xenium data is wired into the current pipeline. Adapters for DINOv2 embeddings, Cellpose segmentation, a Visium loader and an image registration service exist in the code but are not part of the default presets.
- There is no automated test suite yet. `backend/back/` contains earlier iterations of the project and is not used.

## Author

Viacheslav Raitsev. B.Sc. Information Systems and Technologies, Tambov State Technical University (2026).
