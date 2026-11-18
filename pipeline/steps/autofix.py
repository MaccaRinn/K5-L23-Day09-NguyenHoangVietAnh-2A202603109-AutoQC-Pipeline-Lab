"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401


def fix(ds, issues, cfg):
    drop_codes = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
    remove_codes = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}
    by_image = defaultdict(list)
    for issue in issues:
        if issue.severity == ERROR:
            by_image[issue.image].append(issue)

    fixes = []
    removed_images = set()

    for image, image_issues in by_image.items():
        if any(issue.code.startswith("DUP_IMAGE_") or issue.code in remove_codes for issue in image_issues):
            image_path = ds.image_path(image)
            label_path = ds.label_path(image)
            removed = []
            if image_path is not None and image_path.exists():
                image_path.unlink()
                removed.append(image_path.name)
            if label_path.exists():
                label_path.unlink()
                removed.append(label_path.name)
            if removed:
                fixes.append({
                    "image": image,
                    "line": None,
                    "action": "remove_image",
                    "before": ", ".join(removed),
                    "after": "(đã xóa ảnh và nhãn)",
                })
            removed_images.add(image)

    for image, image_issues in by_image.items():
        if image in removed_images:
            continue

        if any(issue.code == "MISSING_LABEL_FILE" for issue in image_issues):
            fixes.append({
                "image": image,
                "line": None,
                "action": "needs_label",
                "before": "(không có file)",
                "after": "chuyển người gán",
            })

        if any(issue.code == "BAD_CLASS_NAME" for issue in image_issues):
            if not ds.yaml_path.exists():
                raise FileNotFoundError(f"Không tìm thấy {ds.yaml_path} để sửa tên lớp")
            data = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
            if not isinstance(data, dict):
                raise ValueError(f"Nội dung {ds.yaml_path} phải là mapping YAML")
            expected_names = {index: name for index, name in enumerate(cfg["classes"])}
            if ds.yaml_names() != expected_names:
                before_names = ds.yaml_names()
                data["names"] = expected_names
                ds.yaml_path.write_text(
                    yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                    encoding="utf-8",
                )
                fixes.append({
                    "image": image,
                    "line": None,
                    "action": "rename_classes",
                    "before": str(before_names),
                    "after": str(expected_names),
                })

        label_path = ds.label_path(image)
        line_issues = [
            issue for issue in image_issues
            if issue.line is not None and (issue.code in drop_codes or issue.code == "BOX_OUT_OF_IMAGE")
        ]
        if not line_issues or not label_path.exists():
            continue

        original_lines = label_path.read_text(encoding="utf-8").splitlines(keepends=True)
        actions = defaultdict(set)
        for issue in line_issues:
            actions[issue.line].add(issue.code)

        updated_lines = []
        line_fixes = []
        for line_number, original in enumerate(original_lines, 1):
            codes = actions.get(line_number, set())
            text = original.rstrip("\r\n")
            newline = original[len(text):]
            if codes & drop_codes:
                line_fixes.append({
                    "image": image,
                    "line": line_number,
                    "action": "drop_line",
                    "before": text,
                    "after": "(đã xóa dòng)",
                })
                continue

            if "BOX_OUT_OF_IMAGE" in codes:
                try:
                    class_id, cx, cy, width, height = parse_yolo_line(text)
                except ValueError:
                    updated_lines.append(original)
                    continue

                x1 = max(0.0, cx - width / 2)
                y1 = max(0.0, cy - height / 2)
                x2 = min(1.0, cx + width / 2)
                y2 = min(1.0, cy + height / 2)
                if x2 <= x1 or y2 <= y1:
                    line_fixes.append({
                        "image": image,
                        "line": line_number,
                        "action": "drop_line",
                        "before": text,
                        "after": "(box nằm ngoài ảnh; đã xóa dòng)",
                    })
                    continue

                clipped = (
                    class_id,
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    x2 - x1,
                    y2 - y1,
                )
                new_text = " ".join(
                    [str(clipped[0])] + [f"{value:.12g}" for value in clipped[1:]]
                )
                if new_text != text:
                    line_fixes.append({
                        "image": image,
                        "line": line_number,
                        "action": "clip_box",
                        "before": text,
                        "after": new_text,
                    })
                    updated_lines.append(new_text + newline)
                else:
                    updated_lines.append(original)
                continue

            updated_lines.append(original)

        if line_fixes:
            label_path.write_text("".join(updated_lines), encoding="utf-8")
            fixes.extend(line_fixes)

    return fixes
