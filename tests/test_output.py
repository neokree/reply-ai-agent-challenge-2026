import pytest
import os
from pathlib import Path
from utils.output import write_output, sanitize_dataset_name

def test_sanitize_dataset_name():
    assert sanitize_dataset_name("The Truman Show") == "the_truman_show"
    assert sanitize_dataset_name("Brave New World") == "brave_new_world"
    assert sanitize_dataset_name("Deus Ex") == "deus_ex"

def test_write_output_creates_file(tmp_path):
    flagged_ids = ["abc-123", "def-456", "ghi-789"]
    output_path = tmp_path / "output"
    output_path.mkdir()

    filepath = write_output(flagged_ids, "Test Dataset", output_dir=str(output_path))

    assert os.path.exists(filepath)
    with open(filepath) as f:
        lines = f.read().strip().split('\n')
    assert lines == flagged_ids

def test_write_output_empty_list_raises():
    with pytest.raises(ValueError, match="empty"):
        write_output([], "Test")
