# 9Router Vision 31-Label Multi-Shape Nuclio Detector

Full 31-label autonomous driving and road-scene AI detector for CVAT, powered by 9Router vision models (Gemini 3.8/3.7 Flash).

## Geometry Routing Architecture

1. **Instance Objects (14 labels)**:
   - `pedestrian`, `rider`, `car`, `truck`, `bus`, `train`, `motorcycle`, `bicycle`, `traffic light`, `traffic sign`, `pole`, `person`, `traffic_light`, `traffic_sign`.
   - Emits paired `rectangle` + `mask` with matching `group_id`.
   - Never fabricates mask from bounding box.

2. **Semantic Regions (10 labels)**:
   - `area/alternative`, `area/drivable`, `road`, `sidewalk`, `building`, `wall`, `fence`, `vegetation`, `terrain`, `sky`.
   - Emits pure `mask` with polygon boundary points (`group_id = None`).
   - Rectangular bounding boxes are strictly suppressed.
   - Compatible with CVAT's "Convert masks to polygons".

3. **Lane Markings (7 labels)**:
   - `lane/crosswalk`: 2D surface patch -> `polygon` or `mask`.
   - `lane/single white`, `lane/single yellow`, `lane/double white`, `lane/double yellow`, `lane/road curb`, `lane/single other`:
     - Thin ribbons (aspect ratio $\ge 2.5$) -> extracted medial centerline `polyline`.
     - Non-ribbon fallback -> `polygon` or `mask`.

## Deployment

Deploy via PowerShell:
```powershell
.\scripts\phase3b_deploy.ps1
```
