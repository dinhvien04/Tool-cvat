# 9Router Vision — CVAT Serverless Nuclio Detector

This directory contains the lightweight Nuclio function definition for integrating local **9Router** vision models (Gemini 3.8 Flash, Claude Sonnet, etc.) into CVAT Community as an AI Tools 2D Bounding Box Detector.

---

## Features

- **No heavy local dependencies**: No PyTorch, CUDA, YOLO, or SAM weights. Lightweight Python runtime (< 200MB).
- **Exact CVAT label contract**: Exposes the 13 rectangular bounding box candidate labels from the 31-label CVAT project schema.
- **Docker Desktop networking**: Automatically connects to 9Router on the Windows host via `http://host.docker.internal:20128`.
- **Configurable threshold**: Filters detections dynamically based on CVAT's UI threshold slider.

---

## Directory Layout

```
serverless/ninerouter-vision/nuclio/
├── function.yaml       # Nuclio detector configuration and label specifications
├── main.py             # Nuclio handler entry point
├── model_handler.py    # Inference bridge connecting CVAT requests to 9Router
├── requirements.txt    # Python package dependencies
└── README.md           # Documentation
```

---

## Environment Variables

| Variable | Default (Container) | Description |
|---|---|---|
| `NINEROUTER_URL` | `http://host.docker.internal:20128` | URL of local 9Router service on host |
| `NINEROUTER_KEY` | *(empty)* | Optional API key if 9Router requires authorization |
| `VISION_MODEL` | `ag/gemini-3.8-flash-high` | Target vision model |
| `NINEROUTER_TIMEOUT` | `60.0` | Request timeout in seconds |
| `MAX_IMAGE_SIZE` | `1600` | Maximum dimension for image sent to API |

---

## Deployment Instructions

Use the automated deploy script from the repository root:

```powershell
.\scripts\phase2_deploy.ps1
```

Or deploy directly via `nuctl`:

```bash
nuctl deploy ninerouter-vision \
  --project-name cvat \
  --path serverless/ninerouter-vision/nuclio \
  --file serverless/ninerouter-vision/nuclio/function.yaml \
  --platform local \
  --platform-config '{"attributes": {"network": "cvat_cvat"}}'
```
