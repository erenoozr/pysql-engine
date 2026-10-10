"""Interactive ``pysql>`` shell.

SQL statements end with ``;`` and may span several lines. A leading ``.``
marks a meta command: ``.tables``, ``.schema [table]``, ``.help``,
``.exit`` / ``.quit``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TextIO

from .engine import Database
from .errors import SQLError
from .formatter import TabularFormatter

PROMPT = "pysql> "
CONTINUATION = "   ...> "

HELP_TEXT = """\
Meta commands:
  .tables            List tables
  .schema [table]    Show CREATE TABLE definitions (all tables if none given)
  .help              Show this help
  .exit | .quit      Exit
End SQL statements with ';'; they may span multiple lines."""


class Repl:
    """Read-eval-print loop for SQL and meta commands; I/O is injectable for testing."""

    def __init__(self, database: Database | None = None, formatter: TabularFormatter | None = None,
                 input_func: Callable[[str], str] = input, output: TextIO | None = None) -> None:
        self.database = database or Database()
        self.formatter = formatter or TabularFormatter()
        self._input = input_func
        self._output = output or sys.stdout

    def _print(self, text: str) -> None:
        print(text, file=self._output)

    def run(self) -> None:
        """Read commands until EOF or ``.exit``. Ctrl+C discards a half-typed statement."""
        self._print("PySQL-Engine — type .help for help, .exit to quit")
        buffer: list[str] = []
        while True:
            try:
                line = self._input(CONTINUATION if buffer else PROMPT)
            except EOFError:
                self._print("")
                return
            except KeyboardInterrupt:
                buffer.clear()
                self._print("^C")
                continue
            stripped = line.strip()
            if not buffer and stripped.startswith("."):
                if not self.handle_meta(stripped):
                    return
                continue
            if not stripped and not buffer:
                continue
            buffer.append(line)
            if stripped.endswith(";"):
                self.run_sql("\n".join(buffer))
                buffer.clear()

    def handle_meta(self, command: str) -> bool:
        """Run a meta command; returns ``False`` when the loop should stop."""
        name, *args = command.split()
        name = name.lower()
        if name in (".exit", ".quit"):
            return False
        if name == ".help":
            self._print(HELP_TEXT)
        elif name == ".tables":
            names = sorted(self.database.tables)
            self._print("\n".join(names) if names else "(no tables)")
        elif name == ".schema":
            try:
                tables = [self.database.table(args[0])] if args else [
                    self.database.tables[key] for key in sorted(self.database.tables)]
            except SQLError as error:
                self._print(f"Error: {error}")
                return True
            self._print("\n".join(table.schema_sql() for table in tables) or "(no tables)")
        else:
            self._print(f"Unknown meta command: {name} (type .help)")
        return True

    def run_sql(self, sql: str) -> None:
        """Run SQL text and print the results or the error message; never crashes."""
        try:
            results = self.database.execute_script(sql)
        except SQLError as error:
            self._print(f"{type(error).__name__}: {error}")
            return
        for result in results:
            if result.message is not None:
                self._print(result.message)
            else:
                count = len(result.rows)
                self._print(self.formatter.format(result.columns, result.rows))
                self._print(f"({count} row)" if count == 1 else f"({count} rows)")


def main() -> None:
    """Command-line entry point."""
    Repl().run()
