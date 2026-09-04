from scripts.download_hf_datasets import _quick_validate, uts_vlc_record_to_meta


def test_dataset_filename_path_traversal_is_rejected():
    record = {
        "filename": "../../.env",
        "content": "Điều 1. Nội dung đủ dài để kiểm tra.",
        "type": "law",
        "title": "Luật mẫu",
    }
    assert "invalid_filename" in _quick_validate(record)


def test_dataset_snapshot_is_not_marked_effective_without_verification():
    meta = uts_vlc_record_to_meta({
        "id": "x", "title": "Luật X", "type": "law",
    })
    assert meta["effective_status"] == "unknown"
    assert meta["status_verified_at"] is None

