from pathlib import Path
import pandas as pd


def load_fd002_data(data_dir):
    """
    Load NASA C-MAPSS FD002 train, test, and ground-truth RUL files
    with robust whitespace and path handling.
    """
    data_path = Path(data_dir)

    columns = (
        ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    train_df = pd.read_csv(
        data_path / "train_FD002.txt",
        sep=r"\s+",
        header=None,
        names=columns,
    ).dropna(axis=1, how="all")

    test_df = pd.read_csv(
        data_path / "test_FD002.txt",
        sep=r"\s+",
        header=None,
        names=columns,
    ).dropna(axis=1, how="all")

    rul_df = pd.read_csv(
        data_path / "RUL_FD002.txt",
        sep=r"\s+",
        header=None,
        names=["RUL_actual"],
    ).dropna(axis=1, how="all")

    return train_df, test_df, rul_df