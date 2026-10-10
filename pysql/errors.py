"""PySQL-Engine exception hierarchy.

Every error derives from ``SQLError`` so callers (e.g. the REPL) can catch
anything the engine raises with a single ``except SQLError``.
"""

from __future__ import annotations


class SQLError(Exception):
    """Base class for all engine errors; optionally carries a source location."""

    def __init__(self, message: str, line: int | None = None, column: int | None = None) -> None:
        location = f" (line {line}, column {column})" if line is not None else ""
        super().__init__(f"{message}{location}")
        self.line = line
        self.column = column


class LexerError(SQLError):
    """Lexical errors such as unknown characters or unterminated strings."""


class ParserError(SQLError):
    """Raised when the token stream does not match the SQL grammar."""


class ExecutionError(SQLError):
    """Syntactically valid queries that are semantically or at runtime invalid."""
