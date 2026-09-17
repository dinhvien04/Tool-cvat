# Hướng Dẫn Toàn Diện Phase 3B: Full 31-Label Multi-Shape Annotation (CVAT x 9Router)

Tài liệu này cung cấp hướng dẫn chuyên sâu về kiến trúc, lược đồ 31 nhãn đường phố chuẩn, cơ chế phân tuyến đa hình học (multi-shape routing), thuật toán hình học phi-ML và quy trình triển khai cho **Phase 3B: Full 31-Label Multi-Shape Annotation** thuộc hệ thống trợ lý gán nhãn AI tích hợp giữa **CVAT Community** và **9Router**.

---

## 1. Mục Tiêu & Triết Lý Thiết Kế

### 1.1 Mục tiêu cốt lõi
- Mở rộng hệ thống phát hiện từ 13 đối tượng (Phase 3) thành bộ dò hoàn chỉnh **31 nhãn đường phố (autonomous driving & road scene)** phù hợp với các bộ dữ liệu thực tế như BDD100K, Cityscapes, Mapillary.
- Tự động phân tuyến thông minh (multi-shape routing) các phát hiện từ mô hình VLM (`ag/gemini-3.8-flash-high`, `ag/gemini-3.7-flash-high`,...) sang hình học CVAT tương ứng:
  1. **Đối tượng thực thể (Instance Objects - 14 nhãn)**: Tạo đồng thời cặp `rectangle` + `mask`, liên kết chặt chẽ qua `group_id`. Tuyệt đối không bịa đặt (fabricate) mask từ bounding box.
  2. **Vùng ngữ nghĩa (Semantic Regions - 10 nhãn)**: Tạo `mask` kèm vector tọa độ đa giác `points` (cho phép tính năng "Convert masks to polygons" của CVAT hoạt động), triệt tiêu bounding box, không gom nhóm (`group_id=None`).
  3. **Vạch kẻ đường (Lane Markings - 7 nhãn)**: Tạo đường mảnh `polyline` / `polygon` / `mask`, triệt tiêu bounding box, không gom nhóm (`group_id=None`).
- Cho phép suy luận toàn cảnh một lần duy nhất (**single scene inference**) với chi phí token tối ưu và độ trễ hợp lý (~20-25s cho ảnh 1280x720).

### 1.2 Chỉ thị An Toàn & Zero Local Heavy ML (Bảo toàn tuyệt đối)
- **KHÔNG** cài đặt bất kỳ thư viện học sâu nặng nào: PyTorch, Torchvision, TensorFlow, Ultralytics, SAM, SAM2, Detectron2, CUDA.
- **KHÔNG** tải hay lưu trữ bất kỳ file trọng số mô hình cục bộ nào.
- **KHÔNG** dùng OpenCV (`cv2`).
- Toàn bộ thuật toán trích xuất đường tim (centerline extraction), tính tỉ số trục PCA qua hiệp phương sai không gian (spatial covariance), đơn giản hóa Douglas-Peucker và chuyển đổi mask đều viết bằng **Pure Python Standard Library** và **Pillow** (`PIL.Image`, `PIL.ImageDraw`).
- Giữ nguyên vẹn các Phase 1, Phase 2, Phase 3 trước đó, không gây xung đột hay phá vỡ tính tương thích ngược.

---

## 2. Bảng Danh Mục 31 Nhãn Chuẩn (Exact Master Label Schema)

Hệ thống bảo toàn tuyệt đối 31 nhãn định danh duy nhất, chia làm 3 nhóm hình học:

| STT | Nhãn (Label) | Nhóm (Group) | Kiểu Hình Học Phát Ra (Output Shapes) | Hành Vi Gom Nhóm (Grouping) |
|:---:|---|:---:|---|:---:|
| 1 | `pedestrian` | Instance | `rectangle` + `mask` | Cùng `group_id` |
| 2 | `rider` | Instance | `rectangle` + `mask` | Cùng `group_id` |
| 3 | `car` | Instance | `rectangle` + `mask` | Cùng `group_id` |
| 4 | `truck` | Instance | `rectangle` + `mask` | Cùng `group_id` |
| 5 | `bus` | Instance | `rectangle` + `mask` | Cùng `group_id` |
| 6 | `train` | Instance | `rectangle` + `mask` | Cùng `group_id` |
| 7 | `motorcycle` | Instance | `rectangle` + `mask` | Cùng `group_id` |
| 8 | `bicycle` | Instance | `rectangle` + `mask` | Cùng `group_id` |
| 9 | `traffic light` | Instance | `rectangle` + `mask` | Cùng `group_id` |
| 10 | `traffic sign` | Instance | `rectangle` + `mask` | Cùng `group_id` |
| 11 | `pole` | Instance | `rectangle` + `mask` | Cùng `group_id` |
| 12 | `person` | Instance | `rectangle` + `mask` | Cùng `group_id` |
| 13 | `traffic_light` | Instance | `rectangle` + `mask` | Cùng `group_id` |
| 14 | `traffic_sign` | Instance | `rectangle` + `mask` | Cùng `group_id` |
| 15 | `area/alternative` | Region | `mask` (+ `points`) | Ungrouped (`None`) |
| 16 | `area/drivable` | Region | `mask` (+ `points`) | Ungrouped (`None`) |
| 17 | `road` | Region | `mask` (+ `points`) | Ungrouped (`None`) |
| 18 | `sidewalk` | Region | `mask` (+ `points`) | Ungrouped (`None`) |
| 19 | `building` | Region | `mask` (+ `points`) | Ungrouped (`None`) |
| 20 | `wall` | Region | `mask` (+ `points`) | Ungrouped (`None`) |
| 21 | `fence` | Region | `mask` (+ `points`) | Ungrouped (`None`) |
| 22 | `vegetation` | Region | `mask` (+ `points`) | Ungrouped (`None`) |
| 23 | `terrain` | Region | `mask` (+ `points`) | Ungrouped (`None`) |
| 24 | `sky` | Region | `mask` (+ `points`) | Ungrouped (`None`) |
| 25 | `lane/crosswalk` | Lane/Surface | `polygon` / `mask` | Ungrouped (`None`) |
| 26 | `lane/double white` | Lane | `polyline` / `polygon` / `mask` | Ungrouped (`None`) |
| 27 | `lane/double yellow` | Lane | `polyline` / `polygon` / `mask` | Ungrouped (`None`) |
| 28 | `lane/road curb` | Lane | `polyline` / `polygon` / `mask` | Ungrouped (`None`) |
| 29 | `lane/single other` | Lane | `polyline` / `polygon` / `mask` | Ungrouped (`None`) |
| 30 | `lane/single white` | Lane | `polyline` / `polygon` / `mask` | Ungrouped (`None`) |
| 31 | `lane/single yellow` | Lane | `polyline` / `polygon` / `mask` | Ungrouped (`None`) |

### Quy tắc xử lý các cặp nhãn trùng lặp/dễ gây nhầm lẫn:
- **`pedestrian` vs `person`**: Giữ tách biệt 100%. Nếu task CVAT chỉ có `person`, model trả về `person`. Nếu task có cả hai, `pedestrian` ưu tiên người đi bộ trên đường/vỉa hè; `person` dùng cho đối tượng tổng quát.
- **`traffic light` vs `traffic_light`**: Không gộp nhãn. Trình phân tích `check_cvat_task.py` tự động phát hiện task dùng ký tự khoảng trắng hay gạch dưới để ánh xạ chuẩn xác.
- **`traffic sign` vs `traffic_sign`**: Tương tự, ánh xạ 1:1 theo đúng định danh của task CVAT.
- **`lane/crosswalk`**: Được mô hình hóa như một vùng bề mặt 2D (`polygon`/`mask`), không ép thành polyline 1D mảnh.

---

## 3. Kiến Trúc Phân Tuyến Đa Hình Học (Multi-Shape Routing Engine)

Hệ thống xử lý dòng suy luận từ 9Router VLM thành các cấu trúc chuẩn hóa cho CVAT thông qua `app/service.py`:

```
                    +--------------------------------+
                    |  Input Image / ROI Sub-region  |
                    +--------------------------------+
                                   |
                                   v
                    +--------------------------------+
                    | 9Router VLM Vision Inference   |
                    | (ag/gemini-3.8-flash-high)     |
                    +--------------------------------+
                                   |
                                   v
                    +--------------------------------+
                    |  Raw Multi-Shape Detections    |
                    |  (labels, boxes, contours)     |
                    +--------------------------------+
                                   |
            +----------------------+----------------------+
            |                      |                      |
            v                      v                      v
   [INSTANCE OBJECTS]     [SEMANTIC REGIONS]       [LANE MARKINGS]
   (car, person, bus...)  (road, sky, building)  (lane/single white...)
            |                      |                      |
    Generate BOTH:           Suppress Bounding      Suppress Bounding
    - Rectangle BBox         Box completely!        Box completely!
    - Native CVAT Mask             |                      |
            |                Emit Mask with         Extract Centerline:
    Link via group_id        'points' vector        - Polyline (if thin)
            |                (ungrouped)            - Polygon / Mask
            |                      |                      |
            +----------------------+----------------------+
                                   |
                                   v
                    +--------------------------------+
                    | CVAT DetectionResultConverter  |
                    | [{"type": "rectangle/mask/...} |
                    +--------------------------------+
```

### 3.1 Cấu trúc phản hồi gửi về CVAT Serverless
- **Rectangle**:
  ```json
  {
    "type": "rectangle",
    "label": "car",
    "confidence": 0.98,
    "points": [120, 240, 350, 480],
    "group_id": 1
  }
  ```
- **Instance Mask** (đi kèm Rectangle cùng `group_id`):
  ```json
  {
    "type": "mask",
    "label": "car",
    "confidence": 0.98,
    "mask": [0, 1, 1, 0, ..., 120, 240, 350, 480],
    "points": [120, 240, 350, 240, 350, 480, 120, 480],
    "group_id": 1
  }
  ```
- **Semantic Region**:
  ```json
  {
    "type": "mask",
    "label": "road",
    "confidence": 0.99,
    "mask": [1, 1, ..., 0, 360, 1280, 720],
    "points": [0, 360, 1280, 360, 1280, 720, 0, 720],
    "group_id": null
  }
  ```
- **Lane Polyline**:
  ```json
  {
    "type": "polyline",
    "label": "lane/single white",
    "confidence": 0.92,
    "points": [450, 720, 520, 550, 580, 420],
    "group_id": null
  }
  ```

---

## 4. Thuật Toán Trích Xuất Tâm Vạch Kẻ Phi-ML (`core/line_geometry.py`)

Để xử lý các vạch kẻ đường dạng ruy-băng mảnh (thin ribbons) thành đường tim `polyline` mượt mà mà không phụ thuộc vào OpenCV hay SciPy, chúng tôi đã phát triển module `core/line_geometry.py` với các thuật toán thuần túy:

### 4.1 Tính tỉ số khung hình định hướng (Oriented Aspect Ratio via Spatial Covariance PCA)
- Tính ma trận hiệp phương sai 2D từ tọa độ các đỉnh của đa giác:
  $$\text{Cov}(X, Y) = \begin{bmatrix} \sigma_{xx} & \sigma_{xy} \\ \sigma_{xy} & \sigma_{yy} \end{bmatrix}$$
- Giải trị riêng (eigenvalues) $\lambda_1 \ge \lambda_2 \ge 0$ bằng phương trình bậc hai giải tích:
  $$\text{Trace} = \sigma_{xx} + \sigma_{yy}, \quad \text{Det} = \sigma_{xx}\sigma_{yy} - \sigma_{xy}^2$$
  $$\lambda_{1,2} = \frac{\text{Trace} \pm \sqrt{\text{Trace}^2 - 4\text{Det}}}{2}$$
- Tỉ số trục (Aspect Ratio) $AR = \sqrt{\frac{\lambda_1}{\lambda_2 + \epsilon}}$. Nếu $AR \ge 2.5$, đối tượng được xác nhận là vạch kẻ mảnh kéo dài.

### 4.2 Lấy mẫu cặp đối xứng trung vị (Medial Pair Resampling)
- Tách đường bao đa giác thành hai nhánh biên trái và biên phải dựa trên phép chiếu lên vector riêng chính $\mathbf{v}_1$.
- Nội suy đều $N$ điểm dọc theo hai nhánh biên và lấy trung điểm của từng cặp tương ứng để tái tạo đường tim chính xác.

### 4.3 Đơn giản hóa Douglas-Peucker
- Khử các điểm dao động răng cưa và giảm tải số lượng đỉnh cho CVAT canvas với ngưỡng dung sai $\epsilon = 2.0$ pixel.

---

## 5. Kiểm Tra Tương Thích Task Trực Tiếp Qua Django ORM (`scripts/check_cvat_task.py`)

Công cụ `scripts/check_cvat_task.py` cho phép thanh tra trực tiếp nhãn của task CVAT bên trong container `cvat_server` mà không cần token xác thực web:

```powershell
python scripts\check_cvat_task.py --task-id 14
```

**Kết quả kiểm tra Task 14 ('easy_semantic'):**
```
============================================================================
 CVAT Task Compatibility Report - Task #14: 'easy_semantic'
============================================================================
Overall Status:       STRICT_SUBSET
Details:              Task is a strict subset of 31-label master schema (5/31 active labels).
Matched 31-Labels:    5 / 31
Foreign Labels:       0
Active Groups:        Instances: 0 | Regions: 5 | Lanes: 0

Label Mapping & Routing Table:
----------------------------------------------------------------------------
Task Label             | Model Label            | Group      | Shapes        
----------------------------------------------------------------------------
road                   | road                   | region     | mask, polygon 
sidewalk               | sidewalk               | region     | mask, polygon 
building               | building               | region     | mask, polygon 
vegetation             | vegetation             | region     | mask, polygon 
sky                    | sky                    | region     | mask, polygon 
----------------------------------------------------------------------------
============================================================================
```

---

## 6. Triển Khai & Kiểm Thử Tự Động Hóa Qua PowerShell

### 6.1 Kiểm tra tiền triển khai (Pre-flight Check)
```powershell
.\scripts\phase3b_preflight.ps1 -Target 31 -TaskId 14
```
*Đảm bảo 10/10 tiêu chí kiểm tra: Docker, CVAT compose stack, 9Router connectivity, Vision model, WSL/nuctl, Task schema đều đạt trạng thái PASS.*

### 6.2 Triển khai hàm Nuclio 31-nhãn
```powershell
.\scripts\phase3b_deploy.ps1 -Target 31
```
*Tập lệnh tự động đồng bộ mã nguồn, cấu hình network `cvat_cvat`, cài đặt timeout 120.0s, giải quyết lỗi đường dẫn WSL vsock, và triển khai hàm `ninerouter-vision-31`.*

### 6.3 Kiểm thử khói đầu-cuối (Smoke Test)
```powershell
.\scripts\phase3b_smoke_test.ps1 -Target 31
```
Hoặc kiểm thử trên frame ảnh đường phố thực tế của Task 14:
```powershell
.\scripts\phase3b_smoke_test.ps1 -Target 31 -ImagePath task14_frame0.jpg
```

**Kết quả kiểm thử thực tế trên frame đường phố Task 14:**
```
Found direct HTTP port: http://localhost:2381
Sending POST request to http://localhost:2381...
HTTP Response: 200 (22515 ms)
Detections returned: 18
  -> [INST] [car] type=rectangle conf=0.98 group_id=1
  -> [INST] [car] type=mask conf=0.98 group_id=1
  -> [INST] [car] type=rectangle conf=0.98 group_id=2
  -> [INST] [car] type=mask conf=0.98 group_id=2
  -> [INST] [car] type=rectangle conf=0.88 group_id=3
  -> [INST] [car] type=mask conf=0.88 group_id=3
  -> [INST] [car] type=rectangle conf=0.86 group_id=4
  -> [INST] [car] type=mask conf=0.86 group_id=4
  -> [INST] [car] type=rectangle conf=0.82 group_id=5
  -> [INST] [car] type=mask conf=0.82 group_id=5
  -> [REGION] [sky] type=mask conf=0.99
  -> [REGION] [terrain] type=mask conf=0.95
  -> [REGION] [terrain] type=mask conf=0.95
  -> [REGION] [wall] type=mask conf=0.92
  -> [REGION] [road] type=mask conf=0.99
  -> [LANE] [lane/single yellow] type=polyline conf=0.95
  -> [LANE] [lane/single white] type=polygon conf=0.9
  -> [LANE] [lane/single white] type=polyline conf=0.92

Breakdown:
  Instances: 10 | Regions: 5 | Lanes: 3
  Shapes: Rectangles: 5 | Masks: 10 | Polylines: 2 | Polygons: 1
Smoke test passed for ninerouter-vision-31!
```

---

## 7. Hướng Dẫn Thao Tác Chi Tiết Trên Giao Diện CVAT Web UI

Sau khi hoàn tất triển khai, người dùng có thể thực hiện gán nhãn tự động ngay trên trình duyệt:

1. **Đăng nhập CVAT**:
   - Truy cập `http://localhost:18080` (hoặc cổng Traefik của CVAT).
   - Mở Task mục tiêu (ví dụ Task #14 `easy_semantic` hoặc Task #4 `Day2-CCTV`).
   - Nhấn mở một Job để vào màn hình gán nhãn (Annotation Canvas).

2. **Kích hoạt công cụ AI Tools**:
   - Trên thanh công cụ bên trái (Left Toolbar), nhấn chọn biểu tượng **AI Tools** (hình chiếc đũa thần / robot).
   - Chọn tab **Detectors**.
   - Trong danh sách mô hình xổ xuống, chọn:
     **`9Router Vision 31-Label Multi-Shape Detector`** (hoặc `ninerouter-vision-31`).

3. **Ánh xạ nhãn (Label Matching)**:
   - CVAT sẽ hiển thị bảng đối chiếu nhãn giữa Task và Model.
   - Nhờ khai báo `"type": "any"` trong `function.yaml`, mọi nhãn của Task khớp với 31-nhãn chuẩn sẽ tự động ánh xạ xanh (green match) 100%.
   - Nếu Task chỉ có 5 nhãn (như Task 14: `road`, `sidewalk`, `building`, `vegetation`, `sky`), các nhãn này tự động kết nối mà không cần thao tác thủ công.

4. **Tùy chọn chuyển đổi Mask sang Polygon ("Convert masks to polygons")**:
   - Bật tích chọn checkbox **"Convert masks to polygons"** (`conv_mask_to_poly`).
   - Nhờ trường `"points"` đi kèm trong payload của bộ dò, CVAT sẽ trực tiếp chuyển đổi các mặt nạ phân đoạn thành polygon vector đa giác có thể kéo chỉnh sửa từng đỉnh rất thuận tiện.

5. **Chạy gán nhãn tự động**:
   - Thiết lập **Threshold** (khuyến nghị: `0.3` - `0.5`).
   - Nhấn **Annotate** (hoặc chọn khoảng frame nếu chạy cho video/chuỗi ảnh).
   - Sau ~20 giây, kết quả xuất hiện trên Canvas:
     - Các xe hơi, người đi bộ hiển thị cả hình chữ nhật bounding box và mặt nạ mask được nhóm chung `group_id`.
     - Bầu trời (`sky`), mặt đường (`road`), thảm thực vật (`vegetation`) phủ kín dưới dạng polygon ngữ nghĩa sạch sẽ, không có hộp chữ nhật thừa.
     - Vạch kẻ đường hiển thị dưới dạng đường tim `polyline` hoặc `polygon` thanh mảnh.

---

## 8. Quản Trị Hệ Thống & Khắc Phục Sự Cố (Troubleshooting)

### 8.1 Lỗi Timeout VLM khi xử lý ảnh phức tạp (Request timed out after 60s)
- **Nguyên nhân**: Khi phát hiện toàn diện 31 nhãn trên ảnh độ phân giải cao, mô hình lý luận (Reasoning Model) của 9Router có thể mất 45-65 giây.
- **Giải pháp**: Cấu hình `NINEROUTER_TIMEOUT=120.0` trong `function.yaml`, `app/config.py` và truyền cờ `--env "NINEROUTER_TIMEOUT=120.0"` khi triển khai qua nuctl.

### 8.2 Lỗi kết nối WSL vsock (`Wsl/Service/0x8007274c`)
- **Nguyên nhân**: Quá trình kết nối vsock giữa Windows host và WSL Ubuntu bị gián đoạn hoặc treo tạm thời.
- **Giải pháp**:
  - Không sử dụng lệnh gọi con `wslpath`. Script `phase3b_deploy.ps1` sử dụng giải thuật chuyển đổi chuỗi đường dẫn PowerShell thuần túy.
  - Nếu gặp thông báo này, chỉ cần chạy lệnh giải phóng phiên:
    ```powershell
    wsl -t Ubuntu
    ```
  - Thao tác này an toàn 100%, không ảnh hưởng tới Docker daemon hay dữ liệu CVAT.

### 8.3 An toàn dữ liệu tuyệt đối (Zero Volume Impact)
- Khi cần gỡ bỏ hàm Nuclio để nâng cấp hoặc cài lại, **CHỈ** sử dụng script an toàn:
  ```powershell
  .\scripts\phase3b_remove.ps1 -Target 31
  ```
- Tuyệt đối **KHÔNG** chạy `docker compose down -v` vì sẽ xóa toàn bộ cơ sở dữ liệu PostgreSQL và các tác vụ đã gán nhãn của CVAT.
