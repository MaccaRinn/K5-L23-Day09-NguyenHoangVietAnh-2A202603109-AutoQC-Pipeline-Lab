# Day 09 Lab — Nguyen Hoang Viet Anh: 2A202603109 - Nguyen Hoang Viet Anh

## 1. Kết luận trước
Pipeline nhận 224 ảnh raw, giữ 193 ảnh qua intake và loại 31 ảnh. Mô phỏng người duyệt xem 80/138 ảnh trong hàng đợi; sau duyệt còn 0 lỗi bắt buộc nhưng 53/193 ảnh còn cảnh báo (27.46%), vượt ngưỡng Hold 10%, nên kết luận là **Hold** (mã thoát 10). Dataset được xuất ở `submission/export/clean_v1/` dưới dạng bản nháp, chưa được phát hành.

## 2. Lỗi và phân việc
Số dưới đây là số issue/dòng trong `02_error_list_v1` (193 ảnh qua intake; các lỗi intake cũng được tính), không phải số ảnh duy nhất.

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| BAD_CLASS_ID | 6 | Máy | Xóa dòng có ID lớp ngoài phạm vi. |
| BAD_CLASS_NAME | 2 | Máy | Ghi lại names trong data.yaml theo thứ tự lớp chuẩn. |
| BAD_FORMAT | 3 | Máy | Xóa dòng nhãn sai định dạng. |
| BAD_SIZE | 3 | Máy | Loại ảnh không đạt kích thước tối thiểu. |
| BOX_OUT_OF_IMAGE | 8 | Máy | Cắt box về trong ảnh; nếu box nằm hẳn ngoài thì xóa dòng. |
| BOX_ZERO | 4 | Máy | Xóa dòng có chiều rộng hoặc cao bằng 0. |
| CORRUPT_IMAGE | 2 | Máy | Loại ảnh hỏng. |
| DUP_BOX | 7 | Máy | Xóa box trùng cùng lớp. |
| DUP_IMAGE_EXACT / DUP_IMAGE_NEAR | 15 | Máy | Loại ảnh trùng; ngưỡng gần trùng đã calibrate. |
| LEAK_TEST | 11 | Máy | Loại ảnh trùng với tập test/val để tránh rò rỉ. |
| MISSING_LABEL_FILE | 6 | Người | Máy không tự tạo nhãn; chuyển người gán. |
| BOX_HUGE / BOX_TINY | 9 | Người | Cảnh báo hình học, cần xem ảnh để quyết định. |
| EMPTY_LABEL | 11 | Người | Người xác nhận ảnh nền hay bị thiếu nhãn. |
| POSSIBLE_FALSE_POS / POSSIBLE_LOOSE_BOX / POSSIBLE_MISSING / POSSIBLE_WRONG_CLASS | 239 | Người | Ý kiến model chỉ là cảnh báo ngữ nghĩa, có thể sai. |
| CLASS_RARE | 4 | Ghi nhận | Cảnh báo thống kê cấp tập dữ liệu, không gắn với một ảnh nên không vào review queue. |

Hàng đợi xếp `img_0160` đầu tiên (priority 35.692) vì có nhiều cảnh báo quan trọng `POSSIBLE_MISSING` và `POSSIBLE_WRONG_CLASS`; điểm ưu tiên cộng trọng số theo mã lỗi và confidence. Ngân sách là 80 **ảnh**, không phải 80 issue; trong 138 ảnh có cảnh báo, 80 ảnh được mô phỏng duyệt, gồm các ảnh thiếu nhãn được ưu tiên cao.

## 3. Calibration (2 ngưỡng)
| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | 16 | DUP_IMAGE_NEAR: 1.00/0.50 → 1.00/0.83; DUP_IMAGE_EXACT: 1.00/1.00 → 1.00/1.00 | Recall bắt ảnh gần trùng tăng, không giảm precision; tránh 20 vì precision DUP_IMAGE_EXACT còn 0.33. |
| `second_opinion.conf` | 0.25 | 0.25 | POSSIBLE_LOOSE_BOX: 0.29/0.40; POSSIBLE_MISSING: 0.50/1.00; POSSIBLE_WRONG_CLASS: 0.56/0.83 | Giữ 0.25 để ưu tiên recall loose-box (0.40 so với 0.20 ở 0.5); chấp nhận kiểm tra thêm báo động giả. Đây là ngưỡng đã sweep, dù trùng giá trị khởi đầu. |

Điểm có thể đánh lừa: calib chỉ có 39 ảnh và chỉ 5 lỗi BOX lỏng chèn; một lỗi bỏ sót làm recall đổi 0.20. Hơn nữa, phân bố lỗi raw khác calib và calib không có LEAK_TEST, nên không thể xem các số này là hiệu năng chắc chắn trên raw.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
Hai giới hạn quan sát được khi mở ảnh raw cùng overlay nhãn bằng `scripts/viz_labels.py`; cả hai ảnh đều không có dòng tương ứng trong `07_error_list_v3.csv`:
- `img_0048.jpg`: có ít nhất hai người rất nhỏ ở giữa sườn tuyết nhưng không có box person. Checker dựa vào dự đoán model, nên có thể bỏ sót vật nhỏ/xa nếu model không đề xuất chúng.
- `img_0088.jpg`: ảnh có chó trên ván lướt nhưng không có box. Đây là giới hạn phạm vi, không phải lỗi theo guideline hiện tại: pipeline chỉ kiểm bốn lớp `car/person/truck/bus`, nên không thể phát hiện thiếu nhãn của lớp ngoài taxonomy.

Chạy với `--budget 0` còn 6 `MISSING_LABEL_FILE`, kết luận **Rework**, mã thoát 20; `work/exp_nb` không được tạo. Đây là kiểm tra ngân sách duyệt riêng, không thay thế hai ví dụ điểm mù ở trên.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
Train Colab chưa chạy tại thời điểm báo cáo; hiện chưa có kết quả mAP để so sánh raw và clean.

## 6. Quyết định và ghi chú AI
Quyết định hiện tại là **Hold**. Muốn Release cần xử lý đủ cảnh báo còn lại; không hạ ngưỡng chỉ để đạt Release. AI hỗ trợ viết bốn module; code được kiểm bằng 79 test, calibration và lần chạy pipeline. Một nhận định ban đầu của AI chưa chính xác: xem `img_0001` và `img_0030` như điểm mù, nhưng báo cáo pipeline cho thấy cả hai đã được gắn cờ; nhận định đã được sửa.
