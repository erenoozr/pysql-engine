"""Abstract syntax tree (AST) nodes produced by the parser.

All nodes are immutable (frozen) and hashable, so GROUP BY expressions can
be matched directly as dictionary keys. Identifiers (table/column names)
are lower-cased by the parser: as in PostgreSQL, unquoted names are
case-insensitive.
"""

from __future__ import annotations

from dataclasses import dataclass

Value = int | float | str | bool | None
"""Scalar value types the engine can hold (``None`` = SQL NULL)."""

AGGREGATE_FUNCTIONS: frozenset[str] = frozenset({"COUNT", "SUM", "AVG", "MIN", "MAX"})


# --- Expression AST --------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Literal:
    """Constant value: number, string, boolean or NULL."""

    value: Value


@dataclass(frozen=True, slots=True)
class ColumnRef:
    """Column reference; ``table`` is set when qualified by a table name/alias."""

    name: str
    table: str | None = None


@dataclass(frozen=True, slots=True)
class Star:
    """``*`` or ``table.*``; used in the select list and inside ``COUNT(*)``."""

    table: str | None = None


@dataclass(frozen=True, slots=True)
class UnaryOp:
    """Unary operator: ``NOT``, numeric ``-``, ``IS NULL`` or ``IS NOT NULL``."""

    op: str
    operand: Expr


@dataclass(frozen=True, slots=True)
class BinaryOp:
    """Binary operator: ``AND``, ``OR``, comparison (``=``, ``<`` ...) or arithmetic (``+ - * /``)."""

    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True, slots=True)
class FunctionCall:
    """Aggregate function call; ``argument`` may be ``Star`` only for COUNT."""

    name: str
    argument: Expr | Star


Expr = Literal | ColumnRef | UnaryOp | BinaryOp | FunctionCall


# --- Statement AST ---------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ColumnDef:
    """Column definition inside CREATE TABLE; ``ANY`` when no type is given."""

    name: str
    type_name: str


@dataclass(frozen=True, slots=True)
class CreateTable:
    """``CREATE TABLE name (column [type], ...)``"""

    name: str
    columns: tuple[ColumnDef, ...]


@dataclass(frozen=True, slots=True)
class Insert:
    """``INSERT INTO name [(columns)] VALUES (...), (...)`` — multiple rows supported."""

    table: str
    rows: tuple[tuple[Expr, ...], ...]
    columns: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class Update:
    """``UPDATE name SET column = expr, ... [WHERE condition]``"""

    table: str
    assignments: tuple[tuple[str, Expr], ...]
    where: Expr | None = None


@dataclass(frozen=True, slots=True)
class Delete:
    """``DELETE FROM name [WHERE condition]``"""

    table: str
    where: Expr | None = None


@dataclass(frozen=True, slots=True)
class TableRef:
    """Table in FROM/JOIN; when ``alias`` is set, columns are qualified by it."""

    name: str
    alias: str | None = None

    @property
    def ref_name(self) -> str:
        """Name used to qualify columns (the alias takes precedence)."""
        return self.alias or self.name


@dataclass(frozen=True, slots=True)
class Join:
    """``[INNER | LEFT [OUTER]] JOIN table ON condition``"""

    table: TableRef
    condition: Expr
    left_outer: bool = False


@dataclass(frozen=True, slots=True)
class SelectItem:
    """A single entry of the select list and its optional alias."""

    expr: Expr | Star
    alias: str | None = None


@dataclass(frozen=True, slots=True)
class OrderItem:
    """ORDER BY entry: an expression, a select alias or a 1-based column position."""

    expr: Expr
    descending: bool = False


@dataclass(frozen=True, slots=True)
class Select:
    """A complete SELECT statement."""

    items: tuple[SelectItem, ...]
    from_table: TableRef
    joins: tuple[Join, ...] = ()
    where: Expr | None = None
    group_by: tuple[Expr, ...] = ()
    having: Expr | None = None
    order_by: tuple[OrderItem, ...] = ()
    limit: int | None = None
    offset: int | None = None
    distinct: bool = False


Statement = CreateTable | Insert | Update | Delete | Select


def render(expr: Expr | Star) -> str:
    """Turn an expression back into SQL text (for column headers and error messages)."""
    if isinstance(expr, Star):
        return f"{expr.table}.*" if expr.table else "*"
    if isinstance(expr, Literal):
        value = expr.value
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, str):
            return "'" + value.replace("'", "''") + "'"
        return str(value)
    if isinstance(expr, ColumnRef):
        return f"{expr.table}.{expr.name}" if expr.table else expr.name
    if isinstance(expr, UnaryOp):
        if expr.op == "NOT":
            return f"NOT {render(expr.operand)}"
        if expr.op == "-":
            return f"-{render(expr.operand)}"
        return f"{render(expr.operand)} {expr.op}"
    if isinstance(expr, BinaryOp):
        return f"{render(expr.left)} {expr.op} {render(expr.right)}"
    return f"{expr.name}({render(expr.argument)})"
