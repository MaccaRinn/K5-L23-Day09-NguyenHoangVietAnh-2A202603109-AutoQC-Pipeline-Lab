# Trạng thái Day 09

## Phần lõi

- [x] Điền họ tên và MSHV vào tiêu đề README và báo cáo.
- [x] Ghi ít nhất hai điểm mù có bằng chứng ở mục 4 `REPORT.md`.
- [x] Kiểm thử: `python -m pytest -q` — 79 passed.
- [ ] `python scripts/check_submission.py --no-train`: đã chạy; trên Windows checker còn báo nhầm `pipeline/calibrate.py` do so sánh dấu phân cách đường dẫn. Không sửa file đã viết sẵn để né cảnh báo này.

## Tùy chọn — train Colab (15 điểm riêng)

- [ ] Đặt repo GitHub ở chế độ public và push code, cấu hình, `submission/calibration/`, `submission/reports/`, `submission/export/clean_v1/`.
- [ ] Mở `notebooks/day09_train_compare.ipynb`, chọn T4 GPU, đặt `REPO_URL`, chạy các ô.
- [ ] Tải `results.zip` về, giải nén vào repo để có các file trong `submission/results/`.
- [ ] Chạy `python scripts/summarize_results.py`; điền bảng và phân tích mAP50 so với độ lệch giữa các seed vào mục 5 `REPORT.md`.

## Tùy chọn — CVAT

- [ ] Làm bài thưởng CVAT (+2): lọc box `width < 20`, ghi tên ảnh/số box, đối chiếu với `BOX_TINY` và giải thích thuộc tính `can_review`.

## Lưu ý kiểm tra trên Windows

`check_submission.py` hiện có thể báo nhầm `pipeline/calibrate.py` vì đường dẫn Windows dùng `\`, trong khi danh sách file được bảo vệ dùng `/`. File calibration này là file đề được ghim; `RULES.md` cho phép calibration đọc `data/calib/MANIFEST.csv`. Không sửa file đã viết sẵn để né cảnh báo này.
