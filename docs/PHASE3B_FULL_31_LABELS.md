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

## 8. Cơ Chế Tự Thích Ứng Từ Sửa Đổi Của Con Người (Correction-Aware Memory & Adaptive Prompting)

Bộ dò 31 nhãn chuẩn (`ninerouter-vision-31`) được tích hợp cơ chế tự thích ứng trực tiếp từ các chỉnh sửa thực tế của chuyên viên gán nhãn trên giao diện CVAT, giúp chất lượng phát hiện ngày càng chuẩn xác theo từng dự án mà **không cần huấn luyện lại mô hình (retraining)** và **không đòi hỏi GPU/thư viện ML nặng nề**.

```
                +-----------------------------------------------------+
                |                    CVAT WEB UI                      |
                |  (Chuyên viên gán nhãn review & sửa đổi hình học)   |
                +-----------------------------------------------------+
                                           |
                                           | [Webhook: update:job]
                                           | Hoặc CLI: main.py feedback-sync
                                           v
+-----------------------+       +-------------------------------------+
| ninerouter-vision-31  |       |        app/cvat_sync.py             |
| Dual-Dispatch Handler | ----> |  - HMAC-SHA256 Webhook Verification |
+-----------------------+       |  - Tải ảnh frame & Annotation JSON  |
                                +-------------------------------------+
                                           |
                                           v
                                +-------------------------------------+
                                |         app/feedback.py             |
                                |  - SHA-256 Image Hash reconciliation|
                                |  - CorrectionDiffEngine (Bipartite) |
                                |  - Phân loại 9 dạng sửa đổi         |
                                |  - Lưu .tool-cvat/feedback.sqlite3  |
                                |  - Tự động sinh quy tắc (SQL count) |
                                +-------------------------------------+
                                           |
                                           v
                                +-------------------------------------+
                                |        app/retrieval.py             |
                                |  - Trích xuất 2-4 mẫu ít lỗi nhất   |
                                |  - Bơm quy tắc vào Vision Prompt    |
                                +-------------------------------------+
                                           |
                                           v
                                +-------------------------------------+
                                |        app/service.py               |
                                |  (Suy luận lần sau thông minh hơn)  |
                                +-------------------------------------+
```

### 8.1 Vân tay định danh ảnh chuẩn hóa (Canonical Visual Fingerprinting)
Do CVAT Serverless Lambda Manager chỉ gửi payload ẩn danh dạng `{"image": "<base64>", "threshold": 0.5}` mà không kèm `task_id` hay `job_id`, việc liên kết giữa dự đoán gốc của AI và các frame được tải về sau này dựa trên hàm băm hình ảnh chuẩn hóa duy nhất (`compute_visual_fingerprint` / `compute_image_hash`):
- **Độc lập định dạng container**: Thay vì băm chuỗi byte nén JPEG/PNG (vốn thay đổi theo thuật toán nén và metadata của thư viện), hàm giải mã ảnh sang bộ đệm pixel RGB thô không nén (`im.convert("RGB").tobytes()`), kết hợp với tiêu đề kích thước chiều rộng x chiều cao (`{w}x{h}_`).
- **Đồng nhất 100%**: Ảnh dưới dạng đối tượng `PIL.Image`, chuỗi byte PNG, chuỗi byte JPEG, file lưu trên ổ đĩa, hay frame tải về từ CVAT REST API `/api/jobs/{id}/data?type=frame` đều tạo ra chuỗi hash SHA-256 hoàn toàn giống nhau.
- Khi detector chạy lần đầu, kết quả AI được lưu vào bảng `predictions` với khóa chính là `image_hash`.
- Khi người dùng hoặc webhook kích hoạt đồng bộ (Sync Job), frame được tải về, giải mã và băm lại, đảm bảo tìm thấy đúng dự đoán nền (baseline) của AI.

### 8.2 Cơ sở dữ liệu phản hồi dùng chung & Cấu hình Bind Mount
Hệ thống sử dụng **duy nhất một cơ sở dữ liệu SQLite** lưu trữ liên tục tại host `.tool-cvat/feedback.sqlite3` và thư mục ảnh crop `.tool-cvat/examples/`:
- **Docker Bind Mount**: Khi triển khai detector `ninerouter-vision-31`, thư mục host `.tool-cvat` được gắn kết (bind mount) trực tiếp vào container tại `/opt/nuclio/feedback`:
  ```yaml
  spec:
    volumes:
      - volume:
          name: feedback-data
          hostPath:
            path: 'D:/tool-cvat/.tool-cvat'
        volumeMount:
          name: feedback-data
          mountPath: '/opt/nuclio/feedback'
  ```
- **Biến môi trường đồng bộ**:
  - `FEEDBACK_DATA_DIR=/opt/nuclio/feedback`
  - `FEEDBACK_DB_PATH=/opt/nuclio/feedback/feedback.sqlite3`
  - Cả container Nuclio và CLI trên máy host (Windows PowerShell / Bash) cùng đọc và ghi chung một file SQLite (bật chế độ WAL `PRAGMA journal_mode=WAL`).
- **An toàn dữ liệu**: Database và ảnh crop hoàn toàn tồn tại độc lập trên host, không bị xóa khi rebuild hoặc xóa container.

### 8.3 Động cơ so khớp song ánh IoU (CorrectionDiffEngine) & 9 dạng chỉnh sửa
Để tránh lỗi phân loại nhầm (ví dụ: AI đoán `car`, người dùng sửa nhãn thành `truck` tại cùng vị trí nếu so khớp theo tên nhãn sẽ bị tính thành 1 lần xóa xe + 1 lần thêm xe tải), `CorrectionDiffEngine` tính ma trận chỉ số giao trên hợp (IoU) hình học trước:
1. **`RELABEL`**: Hai hình có IoU >= 0.60 nhưng mang nhãn khác nhau.
2. **`BOX_MOVE`**: Cùng nhãn, hộp chữ nhật bounding box có tâm dịch chuyển > 5px.
3. **`BOX_RESIZE`**: Cùng nhãn, kích thước chiều rộng hoặc chiều cao thay đổi > 15%.
4. **`MASK_EDIT`**: Cùng nhãn, đường bao đa giác mặt nạ thực thể (`mask`) được chỉnh sửa.
5. **`REGION_EDIT`**: Đường bao mặt nạ vùng ngữ nghĩa (như `road`, `sky`, `sidewalk`) được chỉnh sửa.
6. **`LANE_EDIT`**: Đường tim hoặc đa giác của vạch kẻ đường được nắn lại.
7. **`DELETE_FALSE_POSITIVE`**: Đối tượng AI sinh ra nhưng người dùng xóa bỏ hoàn toàn (báo động giả).
8. **`ADD_MISSING`**: Đối tượng người dùng vẽ thêm mới mà AI bỏ sót.
9. **`NO_CHANGE`**: Đối tượng AI sinh ra được người dùng giữ nguyên vẹn (IoU >= 0.92, không dịch chuyển tâm/kích thước).

### 8.4 Tự động suy diễn quy tắc & Quản lý Crop an toàn
- **Hạn mức lưu trữ an toàn (Storage Quotas)**:
  - Tối đa 50MB dung lượng bộ nhớ đệm hình ảnh crop.
  - Tối đa 150 ảnh crop tổng cộng, mỗi nhãn không vượt quá 10 crop.
  - Kích thước ảnh crop tối đa 512px để bảo vệ bộ nhớ và dung lượng ổ đĩa.
  - Tuyệt đối không lưu trữ khóa API hay chuỗi Base64 nguyên ảnh vào CSDL.
- **Suy diễn quy tắc thống kê**:
  - Hệ thống sử dụng truy vấn SQL tổng hợp: `HAVING COUNT(*) >= min_rule_samples` (mặc định ngưỡng >= 3 mẫu lặp lại).
  - Khi một lỗi xảy ra từ 3 lần trở lên, quy tắc tự động kích hoạt để hướng dẫn mô hình.

### 8.5 Học qua vài mẫu trực quan đa thể thức (Real Multimodal Few-Shot)
Thay vì chỉ gửi hướng dẫn bằng chữ trừu tượng, hệ thống triển khai cơ chế **Multimodal Interleaved Few-Shot**:
1. `CorrectionRetrievalEngine` chọn tối đa 2-3 ví dụ sửa đổi tiêu biểu nhất (`max_examples_per_pass = 2..3`).
2. Trích xuất ảnh crop đã được người dùng chỉnh sửa, mã hóa thành Base64 Data URL (`data:image/jpeg;base64,...`).
3. Chuẩn bị định dạng kết quả cấu trúc JSON mong đợi (`expected_output` chứa mảng `objects` với nhãn và tọa độ chuẩn hóa).
4. Tạo mảng tin nhắn người dùng (user content array) lồng ghép đan xen:
   - `text`: Mô tả ví dụ sửa đổi (ví dụ: `Reviewer relabeled truck -> bus`)
   - `image_url`: Ảnh crop vùng đối tượng thực tế
   - `text`: Cấu trúc JSON mong đợi
   - `text`: Ảnh mục tiêu cần gán nhãn
   - `image_url`: Toàn bộ ảnh mục tiêu
5. **Đo lường & Giám sát (Instrumentation)**: `AnnotationResult` ghi nhận chính xác `visual_examples_used` và `text_rules_used`.

### 8.6 Xác thực chữ ký Webhook HMAC-SHA256 bảo mật
Container `ninerouter-vision-31` hỗ trợ tiếp nhận sự kiện Webhook từ CVAT (`update:job`, `create:job`):
- **Xác thực chữ ký HMAC-SHA256**: Xác thực trực tiếp trên chuỗi byte thô của body request (`raw_body`), không parse lại JSON rồi dump để tránh sai lệch ký tự/khoảng trắng.
- **Tra cứu Header không phân biệt hoa thường**: Hỗ trợ `X-Signature-256`, `x-signature-256`, `X-CVAT-Signature`, `x-cvat-signature`, `X-Hub-Signature-256`.
- **An toàn khi không cấu hình Secret**: Nếu `CVAT_WEBHOOK_SECRET` không được thiết lập, hệ thống ghi log cảnh báo và cho phép chạy chế độ dev nội bộ; khi đã cấu hình secret, mọi request thiếu chữ ký hoặc chữ ký không hợp lệ bị từ chối với mã lỗi 401.

### 8.7 Bộ công cụ dòng lệnh quản trị (CLI Feedback Commands)
Người dùng có thể giám sát và điều khiển toàn bộ bộ nhớ sửa đổi thông qua `main.py`:

```bash
# 1. Cấu hình hoặc kiểm tra Webhook tự động trong CVAT
python main.py feedback-webhook-setup --url http://localhost:18080 --target-url http://nuclio-nuclio-ninerouter-vision-31:8080 --secret toolcvat_webhook_secret_2026

# 2. Liệt kê các Webhook đang đăng ký trong CVAT
python main.py feedback-webhook-setup --url http://localhost:18080 --list

# 3. Đồng bộ thủ công kết quả chỉnh sửa từ CVAT Job (kèm token nếu CVAT yêu cầu đăng nhập)
python main.py feedback-sync --job 15 --url http://localhost:18080

# 4. Xem bảng thống kê hiệu chỉnh chi tiết của toàn bộ 31 nhãn và các quy tắc đang hoạt động
python main.py feedback-stats

# 5. Liệt kê danh sách các bản ghi hiệu chỉnh gần nhất
python main.py feedback-list --limit 20 --label car

# 6. Tắt/bật tính năng ghi nhớ và gợi ý thích ứng
python main.py feedback-disable
python main.py feedback-enable

# 7. Dọn dẹp bộ nhớ đệm
python main.py feedback-clear --crops   # Xóa các file ảnh crop
python main.py feedback-clear --rules   # Xóa các quy tắc đã suy diễn
python main.py feedback-clear --all     # Xóa toàn bộ dữ liệu phản hồi

# 8. So sánh trực tiếp hiệu quả trước và sau khi áp dụng gợi ý thích ứng
python main.py feedback-eval --image test.jpg
```

---

## 9. Quản Trị Hệ Thống & Khắc Phục Sự Cố (Troubleshooting)

### 9.1 Lỗi Timeout VLM khi xử lý ảnh phức tạp (Request timed out after 60s)
- **Nguyên nhân**: Khi phát hiện toàn diện 31 nhãn trên ảnh độ phân giải cao, mô hình lý luận (Reasoning Model) của 9Router có thể mất 45-65 giây.
- **Giải pháp**: Cấu hình `NINEROUTER_TIMEOUT=120.0` trong `function.yaml`, `app/config.py` và truyền cờ `--env "NINEROUTER_TIMEOUT=120.0"` khi triển khai qua nuctl.

### 9.2 Lỗi kết nối WSL vsock (`Wsl/Service/0x8007274c`)
- **Nguyên nhân**: Quá trình kết nối vsock giữa Windows host và WSL Ubuntu bị gián đoạn hoặc treo tạm thời.
- **Giải pháp**:
  - Không sử dụng lệnh gọi con `wslpath`. Script `phase3b_deploy.ps1` sử dụng giải thuật chuyển đổi chuỗi đường dẫn PowerShell thuần túy.
  - Nếu gặp thông báo này, chỉ cần chạy lệnh giải phóng phiên:
    ```powershell
    wsl -t Ubuntu
    ```
  - Thao tác này an toàn 100%, không ảnh hưởng tới Docker daemon hay dữ liệu CVAT.

### 9.3 An toàn dữ liệu tuyệt đối (Zero Volume Impact)
- Khi cần gỡ bỏ hàm Nuclio để nâng cấp hoặc cài lại, **CHỈ** sử dụng script an toàn:
  ```powershell
  .\scripts\phase3b_remove.ps1 -Target 31
  ```
- Tuyệt đối **KHÔNG** chạy `docker compose down -v` vì sẽ xóa toàn bộ cơ sở dữ liệu PostgreSQL và các tác vụ đã gán nhãn của CVAT.
