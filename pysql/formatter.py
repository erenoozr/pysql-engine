"""Formats query results as a MySQL CLI style ASCII grid table."""

from __future__ import annotations

from collections.abc import Sequence

from .nodes import Value


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class TabularFormatter:
    """Computes column widths, right-aligns numeric columns and left-aligns the rest.

    Example output::

        +----+------+
        | id | name |
        +----+------+
        |  1 | Ali  |
        +----+------+
    """

    def __init__(self, null_text: str = "NULL") -> None:
        self.null_text = null_text

    def format_value(self, value: Value) -> str:
        """Text representation of a single cell."""
        if value is None:
            return self.null_text
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return repr(round(value, 6))
        return str(value)

    def format(self, columns: Sequence[str], rows: Sequence[Sequence[Value]]) -> str:
        """Render headers and rows as a framed table (no trailing newline)."""
        cells = [[self.format_value(value) for value in row] for row in rows]
        widths = [max([len(header), *(len(row[i]) for row in cells)]) for i, header in enumerate(columns)]
        numeric = [any(_is_number(row[i]) for row in rows)
                   and all(row[i] is None or _is_number(row[i]) for row in rows)
                   for i in range(len(columns))]
        border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

        def line(values: Sequence[str], right_aligned: Sequence[bool]) -> str:
            padded = (value.rjust(width) if right else value.ljust(width)
                      for value, width, right in zip(values, widths, right_aligned))
            return "| " + " | ".join(padded) + " |"

        body = [line(row, numeric) for row in cells]
        return "\n".join([border, line(columns, [False] * len(columns)), border, *body, border])
