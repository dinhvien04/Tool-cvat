# 9Router Vision Box+Mask — CVAT Serverless Unified Detector

This directory contains the lightweight Nuclio function definition for integrating local **9Router** vision models (Gemini 3.8 Flash, Claude Sonnet, etc.) into CVAT Community as a unified Bounding Box and Instance Segmentation Detector (`type: any`).

---

## Features

- **Unified Output**: Returns BOTH CVAT rectangle shapes (`type: rectangle`) and CVAT native mask shapes (`type: mask`) with shared `group_id` for paired instances.
- **Flexible CVAT Mapping**: Labels declared with `"type": "any"`, allowing mapping to either rectangle or mask labels in CVAT tasks.
- **Convert Masks to Polygons**: Compatible with CVAT's UI toggle to convert masks into polygons.
- **Zero Heavy ML**: Pure Pillow geometry rasterization (< 200MB Python base image, no PyTorch, no CUDA, no SAM weights).
- **Configurable Threshold**: Filters detections dynamically based on CVAT's UI threshold slider.
- **Docker Desktop Networking**: Connects to 9Router on the Windows host via `http://host.docker.internal:20128`.

---

## Directory Layout

```
serverless/ninerouter-vision-box-mask/nuclio/
├── function.yaml       # Nuclio detector configuration and label specifications (type: any)
├── main.py             # Nuclio handler entry point
├── model_handler.py    # Inference bridge connecting CVAT requests to 9Router (mode: box_and_mask)
├── requirements.txt    # Python package dependencies
└── README.md           # Documentation
```

---

## Environment Variables

| Variable | Default (Container) | Description |
|---|---|---|
| `NINEROUTER_URL` | `http://host.docker.internal:20128` | URL of local 9Router service on host |
| `NINEROUTER_KEY` | *(empty)* | Optional API key if 9Router requires authorization |
| `VISION_MODEL` | *(dynamic)* | Target vision model resolved dynamically from 9Router |
| `NINEROUTER_TIMEOUT` | `60.0` | Request timeout in seconds |
| `MAX_IMAGE_SIZE` | `1600` | Maximum dimension for image sent to API |

---

## Deployment Instructions

Use the automated Phase 3 deployment script from the repository root:

```powershell
.\scripts\phase3_deploy.ps1 -Target box-mask
```
