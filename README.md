# Software Requirements Classification (EN/VI) - Transformers

Fine-tuning Transformer cho bài toán phân loại đa nhãn các yêu cầu phần mềm (nhóm NFR) trên dữ liệu tiếng Anh và tiếng Việt.

## Ý tưởng chính
- Chia train/val/test cố định bằng stratified đa nhãn (giữ phân phối nhãn tốt hơn).
- Tinh chỉnh threshold theo từng nhãn trên VAL, sau đó báo cáo trên TEST.
- Dùng cùng split cho cặp EN/VI vì dữ liệu đã căn dòng tương ứng.

---

## Cấu trúc thư mục
- `data/`
  - `Dataset_Full_EN.jsonl`
  - `Dataset_Full_VI.jsonl` (căn dòng theo EN)
  - `PROMISE-relabeled-NICE_EN.jsonl`
  - `PROMISE-relabeled-NICE_VI.jsonl` (căn dòng theo EN)
  - `labelmap_multilabel.json`
  - `splits/` (tạo sau khi chạy `make_splits_stratified.py`)
- `scripts/`
  - `make_splits_stratified.py`
  - `train_multilabel.py`
  - `tune_thresholds_multilabel.py`
  - `eval_multilabel_report.py`
  - `predict_multilabel.py`

---

## Định dạng dữ liệu

Mỗi dòng trong `*.jsonl` là một JSON object:
```json
{"text": "...", "labels": ["Quality (Q)"], "label_ids": [1]}
```

Ghi chú:
- `labels` là bắt buộc, kiểu list tên nhãn.
- `label_ids` là tùy chọn; nếu thiếu sẽ suy ra từ `labelmap_multilabel.json`.
- Key legacy `label` vẫn được chấp nhận (được hiểu như `labels`).

`labelmap_multilabel.json` phải có:
- `label_names` hoặc `label_columns`
- `label2id` và `id2label`

---

## Cài đặt

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

---

## 1) Tạo split cố định (chạy một lần)

Khuyến nghị tạo split trên EN và dùng lại cho VI (dữ liệu đã căn dòng):
```bash
python scripts/make_splits_stratified.py --data_path data/Dataset_Full_EN.jsonl --labelmap_path data/labelmap_multilabel.json --out_split data/splits/split_seed42.json --seed 42 --val_ratio 0.1 --test_ratio 0.1
```

Output:
- `data/splits/split_seed42.json` chứa `train_idx`, `val_idx`, `test_idx`

Tùy chọn xuất tập con ra JSONL:
```bash
python scripts/make_splits_stratified.py --data_path data/Dataset_Full_EN.jsonl --labelmap_path data/labelmap_multilabel.json --out_split data/splits/split_seed42.json --seed 42 --val_ratio 0.1 --test_ratio 0.1 --export_dir data/splits/seed42
```

---

## 2) Train

Ghi chú:
- Thêm `--use_vitokenizer` để tách từ tiếng Việt (pyvi).
- Thêm `--use_pos_weight` để hỗ trợ nhãn hiếm trong dữ liệu mất cân bằng.
- Nếu không có `--split_path`, script sẽ tự chia ngẫu nhiên theo `--seed`, `--val_ratio`, `--test_ratio`.

### Lựa chọn A: XLM-R cho EN và VI (so sánh công bằng)

EN:
```bash
python scripts/train_multilabel.py --data_path data/Dataset_Full_EN.jsonl --labelmap_path data/labelmap_multilabel.json --split_path data/splits/split_seed42.json --model_name xlm-roberta-base --output_dir models/xlmr_en/seed42 --epochs 5 --batch_size 8 --max_length 256 --threshold 0.5
```

VI:
```bash
python scripts/train_multilabel.py --data_path data/Dataset_Full_VI.jsonl --labelmap_path data/labelmap_multilabel.json --split_path data/splits/split_seed42.json --model_name xlm-roberta-base --output_dir models/xlmr_vi/seed42 --epochs 5 --batch_size 8 --max_length 256 --threshold 0.5 --use_vitokenizer
```

### Lựa chọn B: baseline EN (roberta-base)
```bash
python scripts/train_multilabel.py --data_path data/Dataset_Full_EN.jsonl --labelmap_path data/labelmap_multilabel.json --split_path data/splits/split_seed42.json --model_name roberta-base --output_dir models/roberta_en/seed42 --epochs 5 --batch_size 8 --max_length 256 --threshold 0.5
```

### Lựa chọn C: baseline VI (vinai/phobert-base)
```bash
python scripts/train_multilabel.py --data_path data/Dataset_Full_VI.jsonl --labelmap_path data/labelmap_multilabel.json --split_path data/splits/split_seed42.json --model_name vinai/phobert-base --output_dir models/phobert_vi/seed42 --epochs 5 --batch_size 8 --max_length 256 --threshold 0.5 --use_vitokenizer
```

`train_multilabel.py` sẽ ghi `train_config.json` và `labelmap.json` trong mỗi `--output_dir`. Các script tune/eval/predict sẽ tự đọc `max_length` và `use_vitokenizer` từ đây nếu bạn không truyền vào.

---

## 3) Tune threshold trên VAL

```bash
python scripts/tune_thresholds_multilabel.py --model_dir models/xlmr_en/seed42 --data_path data/Dataset_Full_EN.jsonl --labelmap_path data/labelmap_multilabel.json --split_path data/splits/split_seed42.json --split_name val
```

Output:
- `models/xlmr_en/seed42/thresholds.json`

---

## 4) Đánh giá trên TEST

```bash
python scripts/eval_multilabel_report.py --model_dir models/xlmr_en/seed42 --data_path data/Dataset_Full_EN.jsonl --labelmap_path data/labelmap_multilabel.json --split_path data/splits/split_seed42.json --split_name test --thresholds_json models/xlmr_en/seed42/thresholds.json --out_report models/xlmr_en/seed42/report_test.txt --out_metrics models/xlmr_en/seed42/metrics_test.json
```

Mẹo: dùng `--eval_all` để đánh giá toàn bộ file (bỏ qua split).

---

## 5) Dự đoán (inference)

Một câu:
```bash
python scripts/predict_multilabel.py --model_dir models/xlmr_vi/seed42 --text "The system shall refresh the display every 60 seconds." --thresholds_json models/xlmr_vi/seed42/thresholds.json --output_csv predictions.csv --include_active_labels
```

Nhiều câu từ file `.txt` (mỗi dòng 1 requirement):
```bash
python scripts/predict_multilabel.py --model_dir models/xlmr_vi/seed42 --input_txt path/to/requirements.txt --thresholds_json models/xlmr_vi/seed42/thresholds.json --output_csv predictions.csv --include_active_labels
```

Tùy chọn hữu ích:
- `--include_probs`: xuất thêm cột xác suất cho từng nhãn (`__prob`).
- `--include_active_labels`: thêm cột `ActiveLabels`.

CSV đầu ra là UTF-8 có BOM để mở tốt trong Excel.

---
