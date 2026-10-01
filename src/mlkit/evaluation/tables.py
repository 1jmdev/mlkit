"""Experiment tables with aligned text output and CSV export."""

import csv
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any


class Table(Sequence[dict[str, Any]]):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __getitem__(self, index: Any) -> Any:
        return self.records[index]

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.records)

    def __str__(self) -> str:
        if not self.records:
            return ""
        columns = list(self.records[0])
        values = [
            [format_value(record.get(column)) for column in columns] for record in self.records
        ]
        widths = [
            max(len(column), *(len(row[index]) for row in values))
            for index, column in enumerate(columns)
        ]
        return "\n".join(
            "  ".join(value.ljust(width) for value, width in zip(row, widths, strict=True))
            for row in [columns, *values]
        )

    def to_csv(self, path: str | Path) -> None:
        if not self.records:
            raise ValueError("cannot export an empty comparison table")
        with Path(path).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.records[0]))
            writer.writeheader()
            writer.writerows(self.records)


def format_value(value: Any) -> str:
    if value is None:
        return "?"
    return f"{value:.4f}" if isinstance(value, float) else str(value)
