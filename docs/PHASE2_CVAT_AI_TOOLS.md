# Hướng Dẫn Tích Hợp CVAT Serverless AI Tools Với 9Router Vision

Tài liệu này hướng dẫn chi tiết cách kích hoạt và sử dụng mô hình thị giác AI (Gemini 3.8 Flash, Claude Sonnet,...) từ **9Router** trực tiếp bên trong giao diện **CVAT Community** (mục **AI Tools -> Detectors**).

---

## 1. Giới Thiệu & Kiến Trúc Tổng Quan

### 1.1 Tại sao giải pháp này siêu nhẹ (Zero Local Heavy ML)?
- **Không cài PyTorch, TensorFlow, CUDA, ONNX hay Ultralytics** trên máy tính cá nhân.
- **Không tải các file trọng số mô hình nặng nhiều Gigabytes** (như SAM, SAM2 hay YOLO weights).
- Hàm Nuclio chỉ sử dụng Python thuần với thư viện nhẹ (`Pillow`, `requests`, `pyyaml`) để gửi ảnh qua cổng mạng nội bộ tới **9Router** đang chạy trên máy tính (`http://host.docker.internal:20128`).
- Thời gian build và khởi động container chỉ tính bằng giây, dung lượng image nhỏ và tiết kiệm tối đa RAM/VRAM của máy.

### 1.2 Sơ đồ luồng hoạt động (Architecture Flow)

```
┌───────────────────────────────────────────────────────────────────────────┐
│                           Trình duyệt (Browser)                           │
│                 Giao diện CVAT UI: http://localhost:18080                 │
│              Người dùng mở Task -> Chọn AI Tools -> Detectors             │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ (1) Gửi ảnh Base64 & Threshold
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                     CVAT Serverless (Nuclio Engine)                       │
│           Container: nuclio-cvat-ninerouter-vision (mạng: cvat_cvat)      │
│                                                                           │
│  - Nhận Base64 JPEG từ CVAT                                               │
│  - Tạo bản sao co giãn giữ nguyên tỷ lệ (tối đa 1600px)                   │
│  - Tạo prompt cấu trúc JSON với 13 nhãn hình chữ nhật                     │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ (2) HTTP POST /v1/chat/completions
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                    9Router Gateway (Host Windows)                         │
│                    http://host.docker.internal:20128                      │
│                                                                           │
│  - Điều phối API Key an toàn                                              │
│  - Chuyển tiếp tới mô hình Vision (Gemini 3.8 Flash / Claude Sonnet)      │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ (3) Nhận tọa độ chuẩn hóa [ymin, xmin, ymax, xmax]
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│           Hàm Nuclio ninerouter-vision (Chuyển đổi & Lọc ngưỡng)          │
│                                                                           │
│  - Chuyển đổi box [0, 1000] sang pixel ảnh gốc: [xtl, ytl, xbr, ybr]      │
│  - Lọc bỏ nhãn có độ tin cậy thấp hơn Threshold do người dùng chỉnh       │
│  - Trả về danh sách rectangle annotations chuẩn format CVAT               │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ (4) Render Bounding Boxes trên ảnh
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                    CVAT UI hiển thị ngay các Bounding Box                 │
│                Người dùng kiểm tra, chỉnh sửa và Lưu nhãn                 │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Hợp Đồng Nhãn (Label Contract: 13 Nhãn Bbox vs 31 Nhãn Toàn Dự Án)

Toàn bộ dự án CVAT quản lý tập dữ liệu xe tự hành và giao thông đô thị gồm **31 nhãn** (định nghĩa tại `config/labels.yaml`). Tuy nhiên:
- **13 nhãn** là các đối tượng dạng hình chữ nhật rời rạc (Bounding Box candidates) được khai báo vào detector Nuclio.
- **18 nhãn còn lại** là các thực thể dạng vùng phân đoạn lớn (polygon/polyline/tag như: `lane/crosswalk`, `lane/double white`, `road`, `sidewalk`, `building`, `sky`,...) không phù hợp để mô hình Vision dự đoán hộp chữ nhật bao quanh toàn khung hình.

### 13 Nhãn chữ nhật được tích hợp vào Detector:
1. `pedestrian` (Người đi bộ)
2. `rider` (Người điều khiển xe máy/xe đạp)
3. `car` (Ô tô con)
4. `truck` (Xe tải)
5. `bus` (Xe buýt)
6. `train` (Tàu hỏa)
7. `motorcycle` (Xe máy)
8. `bicycle` (Xe đạp)
9. `traffic light` (Đèn giao thông - có dấu cách)
10. `traffic sign` (Biển báo giao thông - có dấu cách)
11. `person` (Người - danh mục tổng quát)
12. `traffic_light` (Đèn giao thông - gạch dưới)
13. `traffic_sign` (Biển báo giao thông - gạch dưới)

> **Lưu ý bảo toàn sự khác biệt nhãn (Intentional Ambiguities):**
> Hệ thống giữ nguyên sự phân biệt giữa `pedestrian` và `person`, giữa `traffic light` và `traffic_light`, giữa `traffic sign` và `traffic_sign` theo đúng schema CVAT gốc. Tuyệt đối không gộp hay chuẩn hóa làm sai lệch dữ liệu huấn luyện của dự án.

---

## 3. Điều Kiện Cần Chuẩn Bị (Prerequisites)

1. **Docker Desktop trên Windows:** Đang chạy với backend WSL2.
2. **9Router Service:** Đang chạy trên máy chủ Windows tại cổng `20128` (`http://127.0.0.1:20128`).
3. **Cài đặt CVAT:** Đã có thư mục cài đặt CVAT (mặc định tại `D:\cvat` hoặc thông qua biến môi trường `CVAT_ROOT`).
4. **Nuclio CLI (`nuctl`):** Đã cài đặt trên Windows hoặc trong môi trường WSL Ubuntu (`wsl -d Ubuntu nuctl version`).

---

## 4. Hướng Dẫn Triển Khai Nhanh Bằng PowerShell

Bộ kịch bản tự động hóa an toàn đã được đóng gói trong thư mục `scripts/`.

### Bước 1: Kiểm tra môi trường (Diagnostics)
Chạy script kiểm tra không ghi đè (read-only diagnostics):
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\phase2_preflight.ps1
```
Script sẽ kiểm tra tự động 8 tiêu chí:
- Trạng thái Docker Desktop daemon
- Docker Compose CLI v2
- Đường dẫn cài đặt CVAT (`D:\cvat`)
- File compose serverless của CVAT
- Sức khỏe 9Router trên host (`http://127.0.0.1:20128/api/health`)
- Kết nối container vào host (`http://host.docker.internal:20128`)
- Khả năng sẵn sàng của mô hình thị giác (`ag/gemini-3.8-flash-high`)
- Công cụ dòng lệnh `nuctl`

Khi kết quả trả về `PASS: 8, WARN: 0, FAIL: 0`, môi trường của bạn đã sẵn sàng 100%!

### Bước 2: Triển khai hàm Nuclio Detector
Chạy lệnh triển khai an toàn:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\phase2_deploy.ps1
```
Script sẽ tự động:
1. Đảm bảo stack CVAT Serverless đang hoạt động (`docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d`).
2. Đồng bộ mã nguồn nhẹ vào context build.
3. Sử dụng `nuctl deploy` để build image nhẹ `cvat.custom.ninerouter.vision` và kết nối vào mạng `cvat_cvat`.
4. Cấu hình biến môi trường `NINEROUTER_URL=http://host.docker.internal:20128` và `VISION_MODEL=ag/gemini-3.8-flash-high`.

> **Cam kết an toàn:** Script **KHÔNG BAO GIỜ** thực hiện `docker compose down -v` và không bao giờ xóa dữ liệu database, volume hay các task đã tạo trong CVAT!

### Bước 3: Chạy thử nghiệm tự động (Smoke Test)
Kiểm tra chức năng dự đoán trực tiếp bằng lệnh:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\phase2_smoke_test.ps1 -ImagePath "test.jpg" -Threshold 0.5
```
Nếu hàm hoạt động đúng, script sẽ in ra danh sách các bounding box tìm thấy và thông báo:
```
======================================================================
 Smoke Test PASSED! (x valid detections verified)
======================================================================
```

---

## 5. Hướng Dẫn Thao Tác Trực Quan Trên Giao Diện CVAT (UI Guide)

Sau khi triển khai thành công, tính năng AI Detection sẽ xuất hiện ngay lập tức trong CVAT:

### Bước 1: Mở giao diện CVAT
1. Truy cập trình duyệt web tại địa chỉ: `http://localhost:18080`.
2. Đăng nhập vào tài khoản CVAT của bạn.

### Bước 2: Mở một Công việc Gán nhãn (Job)
1. Chọn **Projects** hoặc **Tasks**.
2. Bấm vào một **Job** chứa các khung hình cần gán nhãn.

### Bước 3: Mở AI Tools (Magic Wand)
1. Ở thanh công cụ bên cạnh trái màn hình (Left Toolbar), tìm và bấm vào biểu tượng **Chiếc đũa thần (Magic Wand / AI Tools)**.
2. Một bảng điều khiển sẽ mở ra với các tab: `Interactors`, `Detectors`, `Trackers`.
3. Bấm chọn tab **Detectors**.

### Bước 4: Chọn mô hình và Ánh xạ nhãn (Label Mapping)
1. Trong danh sách thả xuống **Model**, chọn: **9Router Vision**.
2. Phần **Labels mapping** sẽ tự động liệt kê các nhãn của model:
   - Khớp nhãn `car` của model với nhãn `car` của task.
   - Khớp nhãn `pedestrian` của model với `pedestrian` của task.
   - Khớp các nhãn khác theo nhu cầu gán nhãn của bạn.
3. Kéo thanh trượt **Threshold (Ngưỡng tin cậy)**:
   - Khuyến nghị: `0.5` cho điều kiện thông thường.
   - Chọn `0.7` - `0.8` nếu chỉ muốn giữ lại các đối tượng mô hình cực kỳ tự tin.
   - Chọn `0.3` nếu cần tìm cả những đối tượng nhỏ hoặc bị che khuất một phần.

### Bước 5: Thực hiện gán nhãn tự động
1. Bấm nút **Annotate** (hoặc chọn áp dụng cho ảnh hiện tại / toàn bộ các khung hình tiếp theo).
2. Trong khoảng 1 - 2 giây, các khung chữ nhật bao quanh ô tô, người đi bộ, biển báo,... sẽ xuất hiện với màu sắc tương ứng.
3. Bạn có thể kéo thả viền box để tinh chỉnh tọa độ và bấm **Save** (Ctrl + S) để lưu lại kết quả!

---

## 6. Xử Lý Sự Cố Thường Gặp (Troubleshooting)

### Vấn đề 1: Bảng "Detectors" trong CVAT không thấy mô hình "9Router Vision"
- **Nguyên nhân:** Nuclio container chưa khởi động hoặc chưa liên kết với mạng `cvat_cvat`.
- **Cách khắc phục:**
  1. Kiểm tra trạng thái function:
     ```powershell
     wsl -d Ubuntu nuctl get function --platform local
     ```
  2. Kiểm tra container nuclio:
     ```powershell
     docker ps --filter "name=ninerouter-vision"
     ```
  3. Chạy lại `phase2_deploy.ps1`.

### Vấn đề 2: Lỗi kết nối `Cannot connect to host.docker.internal:20128`
- **Nguyên nhân:** Docker Desktop trên Windows chưa cấu hình DNS host hoặc tường lửa Windows chặn kết nối từ container sang cổng 20128.
- **Cách khắc phục:**
  1. Kiểm tra 9Router có đang lắng nghe trên cổng `0.0.0.0:20128` (hoặc `127.0.0.1:20128`) hay không.
  2. Thêm rule cho phép Inbound trên Windows Firewall cho cổng 20128 nếu cần.
  3. Đảm bảo container chạy trong mạng `cvat_cvat`.

### Vấn đề 3: Thời gian suy luận bị timeout (502 Bad Gateway)
- **Nguyên nhân:** Tải ảnh phân giải quá cao hoặc kết nối tới LLM bị chậm hơn thời gian mặc định.
- **Cách khắc phục:**
  Tăng thời gian timeout khi deploy:
  ```powershell
  powershell -ExecutionPolicy Bypass -File .\scripts\phase2_deploy.ps1 -NineRouterUrl "http://host.docker.internal:20128"
  ```
  Biến môi trường `NINEROUTER_TIMEOUT=60.0` đã được cấu hình mặc định là 60 giây.

---

## 7. Gỡ Bỏ An Toàn (Clean & Safe Removal)

Nếu bạn muốn gỡ bỏ hàm detector khỏi Nuclio mà không ảnh hưởng tới dữ liệu CVAT:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\phase2_remove.ps1
```

Script này chỉ xóa duy nhất container function `ninerouter-vision`. Toàn bộ dữ liệu tác vụ, tài khoản người dùng và cơ sở dữ liệu Postgres của CVAT được giữ nguyên vẹn 100%.
