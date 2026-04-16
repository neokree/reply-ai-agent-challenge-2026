import os
from pathlib import Path

def sanitize_dataset_name(name: str) -> str:
    """Convert dataset name to safe filename."""
    return name.lower().replace(" ", "_").replace("-", "_")

def write_output(
    flagged_ids: list[str],
    dataset_name: str,
    output_dir: str = None
) -> str:
    """
    Write flagged transaction IDs to output file.

    Args:
        flagged_ids: List of transaction IDs to flag as fraudulent
        dataset_name: Name of the dataset
        output_dir: Output directory (default: ./output)

    Returns:
        Path to the written file
    """
    if not flagged_ids:
        raise ValueError("Cannot write empty list of flagged IDs")

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "output"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)

    filename = f"{sanitize_dataset_name(dataset_name)}.txt"
    filepath = output_dir / filename

    with open(filepath, "w") as f:
        for tx_id in flagged_ids:
            f.write(f"{tx_id}\n")

    return str(filepath)
