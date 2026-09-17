# Hướng Dẫn Chi Tiết Phase 3: Box + Instance Mask / Segmentation (CVAT x 9Router)

Tài liệu này cung cấp hướng dẫn toàn diện về kiến trúc, hợp đồng dữ liệu, thuật toán hình học và quy trình triển khai cho **Phase 3: Bounding Box + Instance Mask / Segmentation** thuộc dự án tích hợp **CVAT Community** với cổng AI **9Router**.

---

## 1. Mục Tiêu & Triết Lý Thiết Kế

### 1.1 Mục tiêu cốt lõi
- Cho phép **một lần suy luận thị giác duy nhất** (single vision inference call) qua các mô hình VLM từ 9Router (`ag/gemini-3.8-flash-high`, `ag/gemini-3.7-flash-high`, `ag/claude-sonnet-4-6`,...) có khả năng đồng thời tạo ra cả:
  1. **Bounding box 2D** (`box_2d` chuẩn hóa `[ymin, xmin, ymax, xmax]`).
  2. **Đường bao phân đoạn thực thể** (`mask` dạng polygon contour các cặp đỉnh `[x, y]`).
- Tích hợp trực tiếp vào CVAT Community dưới mục **AI Tools -> Detectors**, hỗ trợ gán nhãn tự động cho cả dạng hình chữ nhật (`rectangle`) và mặt nạ phân đoạn (`mask` / `polygon`).

### 1.2 Chỉ thị Zero Local Heavy ML (Bảo toàn tuyệt đối)
- **KHÔNG** cài đặt bất kỳ thư viện học sâu nặng nào: PyTorch, Torchvision, TensorFlow, ONNX, Ultralytics, CUDA.
- **KHÔNG** tải hay lưu trữ bất kỳ file trọng số mô hình lớn nào: SAM (Segment Anything), SAM2, YOLO weights.
- **KHÔNG** dùng OpenCV (`cv2`).
- Toàn bộ các thao tác xử lý hình học vector, chuẩn hóa/ngược chuẩn hóa tọa độ, rasterization từ đa giác sang mặt nạ nhị phân và mã hóa 1D flat list chuẩn CVAT được thực hiện hoàn toàn bằng **Python Standard Library** và **Pillow** (`PIL.Image`, `PIL.ImageDraw`).
- Dung lượng Docker image của hàm Nuclio giữ ở mức siêu nhẹ (< 200MB trên nền `python:3.11-slim`), tiết kiệm tối đa CPU/RAM/VRAM của máy trạm.

---

## 2. Kết Quả Giải Quyết 4 Câu Hỏi Trọng Yếu (Gatekeeper Verification)

Trước khi đi vào triển khai chi tiết, các nghiên cứu và thực nghiệm sâu trên mã nguồn CVAT Server (v2.75.1) và cổng 9Router nội bộ (`http://127.0.0.1:20128`) đã trả lời dứt khoát 4 câu hỏi kiến trúc:

| Câu hỏi kiến trúc | Kết quả | Chi tiết kỹ thuật đã xác minh |
|---|:---:|---|
| **A. CVAT AI Tools có nhận cả `rectangle` và `mask` cho cùng một đối tượng/nhãn trong một phản hồi không?** | **CÓ (YES)** | Tại `cvat/apps/lambda_manager/views.py` (`DetectionResultConverter`), hệ thống duyệt qua từng phần tử trong mảng kết quả của detector. Mỗi phần tử được chuyển đổi dựa trên trường `type` riêng (`"rectangle"`, `"mask"`). Tất cả đều được thêm vào `data["shapes"]` và lưu vào DB dưới dạng các hàng `LabeledShape`. Cả hai shape được liên kết chặt chẽ với nhau thông qua thuộc tính `group: anno["group_id"]`. |
| **B. `function.yaml` có thể khai báo nhãn để cho phép mixed shape types không?** | **CÓ (YES)** | Khai báo nhãn với `"type": "any"` trong `function.yaml` cho phép ánh xạ nhãn của model tới bất kỳ nhãn nào của task trong CVAT. Đặc biệt, khi model khai báo `type: "any"` hoặc `type: "mask"`, CVAT Frontend (`detector-runner.tsx`) tự động hiển thị công tắc **"Convert masks to polygons"** (`conv_mask_to_poly`), cho phép người dùng tùy chọn chuyển đổi mask thành polygon ngay khi gán nhãn. |
| **C. CVAT UI Canvas có hiển thị đồng thời cả Bounding Box và Mask/Polygon không?** | **CÓ (YES)** | Canvas của CVAT hỗ trợ hiển thị đa lớp (multi-layer rendering): vẽ đồng thời bounding box dạng vector và mask dạng bitmap hòa trộn màu (alpha-blended) trên cùng một frame mà không bị xung đột hiển thị. |
| **D. Mô hình thực tế trên 9Router (Gemini 3.8/3.7 Flash) có trả về polygon contour ổn định không?** | **CÓ (YES)** | Thử nghiệm thực tế gửi ảnh 1280x720 tới `http://127.0.0.1:20128` với model `ag/gemini-3.8-flash-high` đã trả về JSON hợp lệ 100%, chứa đồng thời `box_2d` và các đường bao `mask` khép kín từ 8 đến 24 đỉnh, bao quanh chính xác từng xe hơi, người đi bộ và đèn giao thông. |

---

## 3. Quy Ước Tọa Độ & Bộ Chuyển Đổi Hình Học (`core/geometry.py`)

### 3.1 Sự khác biệt trọng yếu về thứ tự tọa độ
Một trong những nguyên nhân phổ biến gây sai lệch vị trí mặt nạ trên ảnh không vuông (như 1920x1080 hay 1280x720) là việc nhầm lẫn thứ tự tọa độ giữa Bounding Box và Polygon:
- **Gemini `box_2d`**: Quy ước `[ymin, xmin, ymax, xmax]` — **y (chiều dọc) đi trước**, x (chiều ngang) đi sau.
- **Gemini `mask` (Polygon Contour)**: Quy ước các điểm `[[x1, y1], [x2, y2], ...]`, trong đó mỗi điểm là `[x, y]` — **x (chiều ngang) đi trước**, y (chiều dọc) đi sau.
- Cả hai đều được chuẩn hóa trong không gian số nguyên `[0, 1000]`.

### 3.2 Định dạng Mask phẳng 1D nguyên bản của CVAT (Native 1D Flattened Mask)
CVAT không nhận ảnh mặt nạ đầy đủ kích thước toàn khung hình qua API serverless (nhằm tiết kiệm băng thông và bộ nhớ). Thay vào đó, CVAT quy định định dạng nén tối ưu:
1. Xác định hộp bao chặt (tight bounding box) `[xmin, ymin, xmax, ymax]` của các pixel tiền cảnh (foreground = 1).
2. Cắt (crop) vùng mặt nạ theo kích thước hộp bao: chiều rộng `crop_w = xmax - xmin + 1`, chiều cao `crop_h = ymax - ymin + 1`.
3. Trải phẳng (flatten) ma trận 2D của vùng cắt theo thứ tự từng hàng (row-major order) thành mảng 1D gồm các giá trị nhị phân `0` và `1`. Độ dài mảng là `crop_w * crop_h`.
4. Nối thêm đúng 4 tọa độ hộp bao `[xmin, ymin, xmax, ymax]` vào cuối mảng.
5. Cấu trúc danh sách hoàn chỉnh gửi về CVAT:
   ```python
   [p0, p1, p2, ..., pN, xmin, ymin, xmax, ymax]
   # Trong đó len(mask) == (xmax - xmin + 1) * (ymax - ymin + 1) + 4
   ```

### 3.3 Các hàm nòng cốt trong `core/geometry.py`
- `calculate_polygon_area(pixel_points)`: Tính diện tích đa giác theo công thức Shoelace (Gauss Area Formula). Dùng để loại bỏ các đa giác suy biến, thẳng hàng hoặc có diện tích < 0.5 pixel.
- `denormalize_contour(contour, width, height)`: Chuyển đổi các cặp đỉnh `[x, y]` từ `[0, 1000]` sang pixel ảnh thực tế `[0, width - 1]` và `[0, height - 1]`, có kiểm tra và kẹp biên an toàn.
- `normalize_contour(pixel_points, width, height)`: Chuẩn hóa ngược từ pixel sang `[0, 1000]`.
- `contour_to_bounding_box(pixel_points, width, height)`: Tính hộp chữ nhật bao chặt bao quanh tập đỉnh.
- `rasterize_polygon_to_mask(pixel_points, width, height)`: Vẽ đa giác lấp đầy (fill=1, background=0) trên ảnh grayscale mode `'L'` bằng Pillow.
- `mask_to_cvat_flat_list(mask_image, bbox)`: Cắt theo hộp bao và trải phẳng thành mảng 1D kèm 4 số nguyên tọa độ cuối.
- `cvat_mask_to_binary_image(cvat_mask, width, height)`: Tái tạo ảnh nhị phân hoàn chỉnh từ mảng 1D phẳng của CVAT. Dùng để kiểm thử round-trip pixel fidelity và hiển thị trực quan.
- `polygon_to_cvat_mask(contour, width, height, ...)`: Pipeline đầu-cuối tích hợp đầy đủ các bước trên, trả về shape dictionary hoàn chỉnh cho CVAT.
- `overlay_mask_on_image(image, mask_image, color, alpha)`: Hòa trộn lớp phủ màu bán trong suốt lên ảnh RGB gốc.

---

## 4. Kiến Trúc 3 Bộ Dò Serverless Trong Thư Mục `serverless/`

Hệ thống cung cấp 3 bộ dò Nuclio chuyên biệt, đáp ứng mọi kịch bản sử dụng trong CVAT:

```
serverless/
├── ninerouter-vision/           # [Bộ dò 1] Bounding Box thuần (Phase 2)
│   └── nuclio/
│       ├── function.yaml        # Declares type: "rectangle"
│       ├── model_handler.py     # mode="box"
│       └── main.py
├── ninerouter-vision-mask/      # [Bộ dò 2] Instance Segmentation thuần (Phase 3)
│   └── nuclio/
│       ├── function.yaml        # Declares type: "mask"
│       ├── model_handler.py     # mode="mask"
│       └── main.py
└── ninerouter-vision-box-mask/  # [Bộ dò 3] Unified Box + Mask (Phase 3)
    └── nuclio/
        ├── function.yaml        # Declares type: "any"
        ├── model_handler.py     # mode="box_and_mask"
        └── main.py
```

### So sánh chi tiết 3 bộ dò:

| Tiêu chí | `ninerouter-vision` | `ninerouter-vision-mask` | `ninerouter-vision-box-mask` |
|---|---|---|---|
| **Loại nhãn khai báo** | `"type": "rectangle"` | `"type": "mask"` | `"type": "any"` |
| **Phân loại trong CVAT** | Detector | Detector / Segmenter | Detector / Segmenter |
| **Dạng shape trả về** | `rectangle` | `mask` | Cả `rectangle` VÀ `mask` |
| **Liên kết thực thể** | Độc lập | Độc lập | Liên kết qua `group_id` giống nhau |
| **Hỗ trợ Convert to Poly** | Không | Có | Có |
| **Khuyến nghị sử dụng** | Khi task chỉ yêu cầu gán nhãn hộp 2D nhanh | Khi task chỉ thu thập mặt nạ phân đoạn pixel | Khi muốn thu thập đồng thời cả Hộp bao và Mặt nạ chi tiết |

---

## 5. Hướng Dẫn Sử Dụng CLI & Pipeline Cục Bộ

Pipeline dòng lệnh (`main.py`) đã được nâng cấp để hỗ trợ tham số `--mode`:

### 5.1 Các chế độ chạy CLI
```bash
# 1. Chế độ mặc định: Tạo đồng thời Bounding Box và Instance Mask
python main.py --image test.jpg --mode box_and_mask

# 2. Chế độ chỉ tạo Bounding Box thuần
python main.py --image test.jpg --mode box

# 3. Chế độ chỉ tạo Instance Mask / Polygon
python main.py --image test.jpg --mode mask

# 4. Chỉ định model cụ thể từ 9Router và thư mục xuất
python main.py --image test.jpg --model ag/gemini-3.8-flash-high --mode box_and_mask -o output
```

### 5.2 Các tệp kết quả được tạo ra trong thư mục `output/`
- `result_bbox.jpg`: Ảnh gốc được vẽ Bounding Box kèm lớp phủ mặt nạ màu bán trong suốt (semi-transparent alpha tint ~33% opacity) bao quanh từng đối tượng.
- `predictions.json`: Danh sách đối tượng phát hiện chi tiết, bao gồm `box_2d`, `pixel_box`, `mask` (tọa độ chuẩn hóa), `pixel_polygon` (tọa độ pixel thực tế), `group_id` và `cvat_annotations`.
- `raw_response.txt`: Toàn văn chuỗi JSON gốc trả về từ mô hình VLM.
- `run_report.json`: Báo cáo thực thi hệ thống (thời gian suy luận, độ trễ API, kích thước ảnh gốc/ảnh gửi, phân bổ nhãn).

---

## 6. Bộ Script Tự Động Hóa Triển Khai PowerShell (`scripts/`)

Để đảm bảo an toàn tuyệt đối cho hệ thống CVAT đang chạy (không xóa volume, không mất dữ liệu task/job/database), toàn bộ quy trình được đóng gói thành các script PowerShell chuẩn hóa:

### 6.1 Kiểm tra tiền khả thi: `scripts\phase3_preflight.ps1`
Thực hiện 9 bước chẩn đoán an toàn (read-only):
```powershell
# Kiểm tra toàn bộ cả 3 bộ dò
.\scripts\phase3_preflight.ps1 -Target all

# Hoặc kiểm tra riêng bộ dò Box+Mask
.\scripts\phase3_preflight.ps1 -Target box-mask
```
Nội dung kiểm tra:
1. Trạng thái Docker daemon.
2. Trạng thái Docker Compose CLI.
3. Vị trí thư mục cài đặt CVAT (`CVAT_ROOT`).
4. Stack serverless của CVAT (`docker-compose.serverless.yml`).
5. Kết nối tới cổng 9Router trên máy host (`http://127.0.0.1:20128/api/health`).
6. Khả năng kết nối từ trong container Docker ra host (`http://host.docker.internal:20128/api/health`).
7. Truy vấn và chọn model thị giác thực tế trên 9Router (`/v1/models`).
8. Tương thích phiên bản giữa `nuctl` CLI và Nuclio Dashboard.
9. Kiểm tra tính hợp lệ của cú pháp `function.yaml` cho từng target.

### 6.2 Triển khai an toàn: `scripts\phase3_deploy.ps1`
Tự động đồng bộ mã nguồn nhẹ vào context build và deploy qua `nuctl`:
```powershell
# Triển khai bộ dò Unified Box+Mask
.\scripts\phase3_deploy.ps1 -Target box-mask

# Triển khai bộ dò Mask thuần
.\scripts\phase3_deploy.ps1 -Target mask

# Triển khai toàn bộ cả 3 bộ dò
.\scripts\phase3_deploy.ps1 -Target all
```
Đặc điểm an toàn:
- Đồng bộ sạch các thư mục `app/`, `core/`, `config/` vào từng thư mục hàm Nuclio trước khi build.
- **Bảo mật API Key tuyệt đối**: Trên WSL, truyền key qua biến môi trường an toàn `WSLENV` và đường ống Base64 trong bộ nhớ; tự động dọn dẹp biến môi trường ngay sau khi kết thúc lệnh. Trên Windows native, tự động ẩn key (`***`) trong toàn bộ output log.
- Gắn đúng vào Docker network của CVAT (`cvat_cvat`).

### 6.3 Kiểm thử chức năng sau triển khai: `scripts\phase3_smoke_test.ps1`
Gửi ảnh mẫu mã hóa Base64 trực tiếp tới container Nuclio đang chạy để xác nhận định dạng phản hồi:
```powershell
# Smoke test cho bộ dò Box+Mask
.\scripts\phase3_smoke_test.ps1 -Target box-mask

# Smoke test cho bộ dò Mask
.\scripts\phase3_smoke_test.ps1 -Target mask
```

### 6.4 Gỡ bỏ an toàn: `scripts\phase3_remove.ps1`
Xóa các container Nuclio mà không ảnh hưởng tới dữ liệu CVAT:
```powershell
.\scripts\phase3_remove.ps1 -Target box-mask
```

---

## 7. Các Biện Pháp Bảo Mật & Phòng Chống Tấn Công Từ Chối Dịch Vụ (DoS)

Trong quá trình đánh giá độc lập bởi Agent G (Security Reviewer), hệ thống đã được gia cố toàn diện:

1. **Giới hạn kích thước Request Body (32MB Guard):**
   - Cả `main.py` của các hàm Nuclio và cấu hình trigger trong `function.yaml` (`maxRequestBodySize: 33554432`) đều từ chối ngay lập tức các yêu cầu vượt quá 32MB với mã lỗi `413 Payload Too Large`.
2. **Ngăn chặn Decompression Bomb:**
   - Cấu hình `Image.MAX_IMAGE_PIXELS = 89_478_485` (~89 megapixels) bảo vệ dịch vụ khỏi các file ảnh nén độc hại gây cạn kiệt RAM máy chủ.
3. **Giới hạn số lượng đỉnh đa giác (Vertex Bomb Protection):**
   - Giới hạn tối đa 10,000 đỉnh cho mỗi polygon contour trong `core/geometry.py` và `app/parser.py`. Các contour có số đỉnh bất thường bị từ chối sớm (hoặc kẹp an toàn) để tránh tấn công làm nghẽn CPU khi rasterize.
4. **Giới hạn số lượng đối tượng trong một ảnh:**
   - Giới hạn tối đa 500 đối tượng/khung hình nhằm ngăn chặn các phản hồi giả mạo tràn ngập bộ nhớ.
5. **Chống rò rỉ API Key qua Error Traces:**
   - Hàm `_sanitize_error_text()` và bộ lọc lỗi trong Nuclio `main.py` tự động quét và thay thế chuỗi API key bằng `***` trước khi log ra màn hình hoặc trả về cho client.
6. **Chính sách Confidence trung thực:**
   - Không bao giờ gán điểm số giả tạo `1.0` khi mô hình không trả về confidence score.

---

## 8. Kết Quả Kiểm Thử Tự Động (Test Suite Results)

Toàn bộ hệ thống được bảo vệ bởi bộ test tự động gồm **245 bài kiểm thử**:

```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\tool-cvat
collected 245 items

tests\test_cli.py ..............                                         [  5%]
tests\test_client.py .......................                             [ 15%]
tests\test_config.py .....                                               [ 17%]
tests\test_cvat_detector_contract.py ...                                 [ 18%]
tests\test_geometry.py ...............................                   [ 31%]
tests\test_image_ops.py ......                                           [ 33%]
tests\test_nuclio_handler.py ...........                                 [ 37%]
tests\test_parser.py .........................                           [ 48%]
tests\test_phase2_config.py ....                                         [ 49%]
tests\test_phase3_nuclio_handlers.py ....................                [ 57%]
tests\test_phase3_parser.py ........                                     [ 61%]
tests\test_phase3_security_reliability.py .....................          [ 69%]
tests\test_phase3_service.py .......                                     [ 72%]
tests\test_pipeline.py ............                                      [ 77%]
tests\test_security_reliability.py .......................               [ 86%]
tests\test_service.py .......                                            [ 89%]
tests\test_vision_contract.py .........................                  [100%]

============================= 245 passed in 16.12s =============================
```

- **Tỷ lệ vượt qua**: 100% (245 passed, 0 failed, 0 errors).
- **Hồi quy**: 0 lỗi hồi quy (toàn bộ các test của Phase 1 và Phase 2 giữ nguyên vẹn 100%).
