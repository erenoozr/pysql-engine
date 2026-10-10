"""In-memory storage (``Database``/``Table``) and the AST execution engine.

SELECT pipeline: FROM → INNER/LEFT JOIN (hash join / nested loop) → WHERE →
GROUP BY + aggregates → HAVING → projection → DISTINCT → ORDER BY → OFFSET/LIMIT.

NULL handling follows SQL three-valued logic: a comparison involving NULL
yields NULL, and WHERE/HAVING/ON keep only rows that evaluate to TRUE.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

from .errors import ExecutionError, ParserError
from .nodes import (
    BinaryOp, ColumnRef, CreateTable, Delete, Expr, FunctionCall, Insert, Join, Literal, Select,
    Star, Statement, UnaryOp, Update, Value, render,
)
from .parser import parse

Row = tuple[Value, ...]
Leaf = Callable[[Expr], Value]
"""Callback that evaluates ``ColumnRef``/``FunctionCall`` leaves in a given context."""

COLUMN_TYPES: dict[str, str] = {
    "INT": "INTEGER", "INTEGER": "INTEGER",
    "FLOAT": "REAL", "REAL": "REAL", "DOUBLE": "REAL",
    "TEXT": "TEXT", "VARCHAR": "TEXT", "STRING": "TEXT",
    "BOOL": "BOOLEAN", "BOOLEAN": "BOOLEAN",
    "ANY": "ANY",
}
"""Accepted type names → canonical type."""

_COMPARATORS: dict[str, Callable[[Value, Value], bool]] = {
    "=": operator.eq, "!=": operator.ne,
    "<": operator.lt, ">": operator.gt, "<=": operator.le, ">=": operator.ge,
}
_ARITHMETIC: dict[str, Callable[[Value, Value], Value]] = {
    "+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv,
}


# --- Storage -------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Column:
    """Table column: name and canonical type (``INTEGER``, ``REAL``, ``TEXT``, ``BOOLEAN``, ``ANY``)."""

    name: str
    type: str

    def coerce(self, value: Value) -> Value:
        """Validate a value against the column type; integers become floats in ``REAL`` columns.

        Raises:
            ExecutionError: If the value does not match the type.
        """
        if value is None or self.type == "ANY":
            return value
        if self.type == "INTEGER" and isinstance(value, int) and not isinstance(value, bool):
            return value
        if self.type == "REAL" and _is_number(value):
            return float(value)  # type: ignore[arg-type]
        if self.type == "TEXT" and isinstance(value, str):
            return value
        if self.type == "BOOLEAN" and isinstance(value, bool):
            return value
        raise ExecutionError(
            f"Column '{self.name}' is {self.type}; value {render(Literal(value))} is not allowed"
        )


class Table:
    """Table that keeps its rows in memory as a list of tuples."""

    def __init__(self, name: str, columns: Sequence[Column]) -> None:
        self.name = name
        self.columns: tuple[Column, ...] = tuple(columns)
        self.rows: list[Row] = []

    def column_index(self, name: str) -> int:
        """Position of a column.

        Raises:
            ExecutionError: If the table has no such column.
        """
        for index, column in enumerate(self.columns):
            if column.name == name:
                return index
        raise ExecutionError(f"Unknown column in table '{self.name}': '{name}'")

    def insert_many(self, rows: Sequence[Sequence[Value]]) -> int:
        """Insert rows atomically: if any row is invalid, none are inserted.

        Returns:
            Number of inserted rows.
        """
        prepared = [self._prepare(row) for row in rows]
        self.rows.extend(prepared)
        return len(prepared)

    def _prepare(self, values: Sequence[Value]) -> Row:
        if len(values) != len(self.columns):
            raise ExecutionError(
                f"Table '{self.name}' expects {len(self.columns)} values, got {len(values)}"
            )
        return tuple(column.coerce(value) for column, value in zip(self.columns, values))

    def schema_sql(self) -> str:
        """``CREATE TABLE`` statement that recreates this table."""
        definitions = ", ".join(f"{column.name} {column.type}" for column in self.columns)
        return f"CREATE TABLE {self.name} ({definitions});"


@dataclass(frozen=True, slots=True)
class Result:
    """Statement result: columns/rows for SELECT, ``message`` for DDL/DML."""

    columns: tuple[str, ...] = ()
    rows: tuple[Row, ...] = ()
    message: str | None = None


class Database:
    """Table catalog and SQL entry point."""

    def __init__(self) -> None:
        self.tables: dict[str, Table] = {}

    def execute(self, sql: str) -> Result:
        """Run SQL text and return the result of the last statement.

        Raises:
            SQLError: Any lexer, parser or execution error.
        """
        results = self.execute_script(sql)
        if not results:
            raise ParserError("No SQL statement to execute")
        return results[-1]

    def execute_script(self, sql: str) -> list[Result]:
        """Run all ``;``-separated statements in order.

        The whole text is parsed first, so a syntax error means no statement runs.
        """
        return [self.execute_statement(statement) for statement in parse(sql)]

    def execute_statement(self, statement: Statement) -> Result:
        """Execute a single, already parsed AST statement."""
        if isinstance(statement, CreateTable):
            return self._create_table(statement)
        if isinstance(statement, Insert):
            return self._insert(statement)
        if isinstance(statement, Update):
            return self._update(statement)
        if isinstance(statement, Delete):
            return self._delete(statement)
        return _SelectExecutor(self, statement).run()

    def table(self, name: str) -> Table:
        """Return the named table (case-insensitive)."""
        try:
            return self.tables[name.lower()]
        except KeyError:
            raise ExecutionError(f"Table not found: '{name}'") from None

    def _create_table(self, statement: CreateTable) -> Result:
        if statement.name in self.tables:
            raise ExecutionError(f"Table already exists: '{statement.name}'")
        seen: set[str] = set()
        columns: list[Column] = []
        for definition in statement.columns:
            if definition.name in seen:
                raise ExecutionError(f"Duplicate column name: '{definition.name}'")
            seen.add(definition.name)
            try:
                column_type = COLUMN_TYPES[definition.type_name]
            except KeyError:
                raise ExecutionError(f"Unsupported column type: '{definition.type_name}'") from None
            columns.append(Column(definition.name, column_type))
        self.tables[statement.name] = Table(statement.name, columns)
        return Result(message=f"Table '{statement.name}' created")

    def _insert(self, statement: Insert) -> Result:
        table = self.table(statement.table)
        rows = [[_evaluate(expr, _constant_leaf, {}) for expr in row] for row in statement.rows]
        if statement.columns is not None:
            positions = [table.column_index(name) for name in statement.columns]
            if len(set(positions)) != len(positions):
                raise ExecutionError("Duplicate column in INSERT column list")
            for values in rows:
                if len(values) != len(positions):
                    raise ExecutionError(
                        f"{len(positions)} columns specified but {len(values)} values given")
            rows = [_spread(values, positions, len(table.columns)) for values in rows]
        return Result(message=f"{_rows(table.insert_many(rows))} inserted")

    def _row_filter(self, table: Table, where: Expr | None) -> tuple[_Scope, Callable[[Row], bool]]:
        """Table scope and WHERE filter for UPDATE/DELETE (all rows when there is no WHERE)."""
        scope = _Scope([(table.name, column.name) for column in table.columns])
        if where is None:
            return scope, lambda row: True
        scope.validate(where)
        if _contains_aggregate(where):
            raise ExecutionError("Aggregate functions are not allowed in WHERE")
        return scope, lambda row: _predicate(_evaluate(where, scope.row_leaf(row), {}))

    def _update(self, statement: Update) -> Result:
        table = self.table(statement.table)
        scope, matches = self._row_filter(table, statement.where)
        targets: list[tuple[int, Expr]] = []
        for name, expr in statement.assignments:
            scope.validate(expr)
            if _contains_aggregate(expr):
                raise ExecutionError("Aggregate functions are not allowed in SET")
            targets.append((table.column_index(name), expr))
        if len({index for index, _ in targets}) != len(targets):
            raise ExecutionError("The same column is assigned more than once in SET")

        # Compute every new row from the old values first, then write them all at once: atomic.
        new_rows: list[Row] = []
        changed = 0
        for row in table.rows:
            if not matches(row):
                new_rows.append(row)
                continue
            values = list(row)
            for index, expr in targets:
                values[index] = table.columns[index].coerce(_evaluate(expr, scope.row_leaf(row), {}))
            new_rows.append(tuple(values))
            changed += 1
        table.rows = new_rows
        return Result(message=f"{_rows(changed)} updated")

    def _delete(self, statement: Delete) -> Result:
        table = self.table(statement.table)
        _, matches = self._row_filter(table, statement.where)
        kept = [row for row in table.rows if not matches(row)]
        removed = len(table.rows) - len(kept)
        table.rows = kept
        return Result(message=f"{_rows(removed)} deleted")


def _rows(count: int) -> str:
    """``"1 row"`` / ``"3 rows"``."""
    return f"{count} row" if count == 1 else f"{count} rows"


def _spread(values: Sequence[Value], positions: Sequence[int], width: int) -> list[Value]:
    """Expand INSERT values given with a column list into a full row; missing columns become NULL."""
    row: list[Value] = [None] * width
    for position, value in zip(positions, values):
        row[position] = value
    return row


# --- Expression evaluation ---------------------------------------------------------

def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _comparable(left: Value, right: Value) -> bool:
    return ((_is_number(left) and _is_number(right))
            or (isinstance(left, str) and isinstance(right, str))
            or (isinstance(left, bool) and isinstance(right, bool)))


def _logical(value: Value) -> bool | None:
    """Validate a value used in a boolean context (NULL → ``None``)."""
    if value is None or isinstance(value, bool):
        return value
    raise ExecutionError(f"Expected a boolean expression, got value: {render(Literal(value))}")


def _predicate(value: Value) -> bool:
    """WHERE/HAVING/ON filter: only TRUE passes."""
    return _logical(value) is True


def _unary(op: str, value: Value) -> Value:
    if op == "IS NULL":
        return value is None
    if op == "IS NOT NULL":
        return value is not None
    if op == "NOT":
        logical = _logical(value)
        return None if logical is None else not logical
    if value is None:
        return None
    if _is_number(value):
        return -value  # type: ignore[operator]
    raise ExecutionError(f"Operator '-' cannot be applied to a non-numeric value: {render(Literal(value))}")


def _binary(op: str, left: Value, right: Value) -> Value:
    if op in ("AND", "OR"):
        a, b = _logical(left), _logical(right)
        dominant = op == "OR"  # TRUE decides OR, FALSE decides AND
        if a is dominant or b is dominant:
            return dominant
        return None if a is None or b is None else not dominant
    if left is None or right is None:
        return None
    if op in _ARITHMETIC:
        if not (_is_number(left) and _is_number(right)):
            raise ExecutionError(
                f"Arithmetic requires numbers: {render(Literal(left))} {op} {render(Literal(right))}")
        if op == "/" and right == 0:
            raise ExecutionError("Division by zero")
        if op == "/" and isinstance(left, int) and isinstance(right, int):
            quotient = abs(left) // abs(right)  # SQL integer division truncates toward zero
            return quotient if (left < 0) == (right < 0) else -quotient
        return _ARITHMETIC[op](left, right)
    if not _comparable(left, right):
        if op in ("=", "!="):
            return op == "!="  # values of different types are never equal
        raise ExecutionError(
            f"Incomparable types: {render(Literal(left))} {op} {render(Literal(right))}"
        )
    return _COMPARATORS[op](left, right)


def _evaluate(expr: Expr, leaf: Leaf, known: Mapping[Expr, Value]) -> Value:
    """Evaluate an expression.

    Args:
        expr: Expression to evaluate.
        leaf: Context function that resolves ``ColumnRef`` and ``FunctionCall`` nodes.
        known: Precomputed subexpressions (e.g. GROUP BY keys).
    """
    if expr in known:
        return known[expr]
    if isinstance(expr, Literal):
        return expr.value
    if isinstance(expr, UnaryOp):
        return _unary(expr.op, _evaluate(expr.operand, leaf, known))
    if isinstance(expr, BinaryOp):
        return _binary(expr.op, _evaluate(expr.left, leaf, known), _evaluate(expr.right, leaf, known))
    return leaf(expr)


def _constant_leaf(expr: Expr) -> Value:
    raise ExecutionError(f"INSERT values must be constants: {render(expr)}")


def _walk(expr: Expr | Star) -> Iterator[Expr | Star]:
    """Visit every node of the expression tree (itself included) in pre-order."""
    yield expr
    if isinstance(expr, UnaryOp):
        yield from _walk(expr.operand)
    elif isinstance(expr, BinaryOp):
        yield from _walk(expr.left)
        yield from _walk(expr.right)
    elif isinstance(expr, FunctionCall):
        yield from _walk(expr.argument)


def _contains_aggregate(expr: Expr) -> bool:
    return any(isinstance(node, FunctionCall) for node in _walk(expr))


def _aggregate(call: FunctionCall, rows: Sequence[Row], scope: _Scope) -> Value:
    """Compute an aggregate over a group of rows; NULL values are ignored."""
    if isinstance(call.argument, Star):
        return len(rows)
    argument = call.argument
    values = [value for row in rows
              if (value := _evaluate(argument, scope.row_leaf(row), {})) is not None]
    if call.name == "COUNT":
        return len(values)
    if not values:
        return None
    if call.name in ("SUM", "AVG"):
        if not all(_is_number(value) for value in values):
            raise ExecutionError(f"{call.name} works only with numeric values: {render(call)}")
        total = sum(values)  # type: ignore[arg-type]
        return total if call.name == "SUM" else total / len(values)
    try:
        return min(values) if call.name == "MIN" else max(values)  # type: ignore[type-var]
    except TypeError:
        raise ExecutionError(f"{render(call)}: incomparable mixed types") from None


class _Scope:
    """Maps every position of a combined row to ``(table reference, column name)``."""

    def __init__(self, bindings: Sequence[tuple[str, str]]) -> None:
        self.bindings: tuple[tuple[str, str], ...] = tuple(bindings)
        self._cache: dict[ColumnRef, int] = {}

    def find(self, ref: ColumnRef) -> list[int]:
        """All positions matching the reference (a list, to detect ambiguity)."""
        return [index for index, (table, column) in enumerate(self.bindings)
                if column == ref.name and ref.table in (None, table)]

    def resolve(self, ref: ColumnRef) -> int:
        """Resolve the reference to a single position.

        Raises:
            ExecutionError: If the column does not exist or exists in several tables (ambiguous).
        """
        if ref in self._cache:
            return self._cache[ref]
        matches = self.find(ref)
        if not matches:
            raise ExecutionError(f"Unknown column: '{render(ref)}'")
        if len(matches) > 1:
            raise ExecutionError(f"Ambiguous column name: '{ref.name}' (use a table prefix)")
        self._cache[ref] = matches[0]
        return matches[0]

    def validate(self, expr: Expr) -> None:
        """Check that every column reference in the expression resolves (even on empty tables)."""
        for node in _walk(expr):
            if isinstance(node, ColumnRef):
                self.resolve(node)

    def row_leaf(self, row: Row) -> Leaf:
        """Single-row context: reads columns from the row and rejects aggregates."""
        def leaf(expr: Expr) -> Value:
            if isinstance(expr, ColumnRef):
                return row[self.resolve(expr)]
            raise ExecutionError(f"Aggregate function not allowed in this context: {render(expr)}")
        return leaf


# --- SELECT executor ----------------------------------------------------------------

OrderTarget = int | Expr
"""A projection position (via alias/number) or an expression to evaluate directly."""


class _SelectExecutor:
    """Executes a single ``Select`` node."""

    def __init__(self, database: Database, statement: Select) -> None:
        self._db = database
        self._stmt = statement

    def run(self) -> Result:
        stmt = self._stmt
        rows, scope = self._source_rows()

        if stmt.where is not None:
            scope.validate(stmt.where)
            if _contains_aggregate(stmt.where):
                raise ExecutionError("Aggregate functions are not allowed in WHERE (use HAVING)")
            rows = [row for row in rows if _predicate(_evaluate(stmt.where, scope.row_leaf(row), {}))]

        projections, aliases = self._projections(scope)
        order = self._order_plan(projections, aliases)
        expressions = [expr for _, expr in projections]
        expressions += [target for target, _ in order if not isinstance(target, int)]
        if stmt.having is not None:
            expressions.append(stmt.having)
        for expr in [*expressions, *stmt.group_by]:
            scope.validate(expr)
        if any(_contains_aggregate(expr) for expr in stmt.group_by):
            raise ExecutionError("Aggregate functions are not allowed in GROUP BY")

        grouped = bool(stmt.group_by) or any(_contains_aggregate(expr) for expr in expressions)
        units = self._groups(rows, scope, expressions) if grouped else (
            (scope.row_leaf(row), {}) for row in rows)

        entries: list[tuple[Row, Row]] = []
        for leaf, known in units:
            if stmt.having is not None and not _predicate(_evaluate(stmt.having, leaf, known)):
                continue
            values = tuple(_evaluate(expr, leaf, known) for _, expr in projections)
            keys = tuple(values[target] if isinstance(target, int) else _evaluate(target, leaf, known)
                         for target, _ in order)
            entries.append((values, keys))

        if stmt.distinct:
            # shortcut: duplicates keep the first row's ORDER BY key; ordering by an expression outside the select list is unspecified.
            unique: dict[Row, Row] = {}
            for values, keys in entries:
                unique.setdefault(values, keys)
            entries = list(unique.items())
        self._sort(entries, [descending for _, descending in order])
        start = stmt.offset or 0
        entries = entries[start:] if stmt.limit is None else entries[start:start + stmt.limit]
        return Result(tuple(header for header, _ in projections), tuple(values for values, _ in entries))

    # --- FROM / JOIN ---------------------------------------------------------

    def _source_rows(self) -> tuple[list[Row], _Scope]:
        base = self._stmt.from_table
        table = self._db.table(base.name)
        bindings = [(base.ref_name, column.name) for column in table.columns]
        rows = list(table.rows)
        used = {base.ref_name}
        for join in self._stmt.joins:
            if join.table.ref_name in used:
                raise ExecutionError(
                    f"Table reference used more than once: '{join.table.ref_name}' (add an alias)"
                )
            used.add(join.table.ref_name)
            rows, bindings = self._join(rows, bindings, join)
        return rows, _Scope(bindings)

    def _join(self, left_rows: list[Row], left_bindings: list[tuple[str, str]],
              join: Join) -> tuple[list[Row], list[tuple[str, str]]]:
        right = self._db.table(join.table.name)
        right_bindings = [(join.table.ref_name, column.name) for column in right.columns]
        bindings = left_bindings + right_bindings
        scope = _Scope(bindings)
        scope.validate(join.condition)
        if _contains_aggregate(join.condition):
            raise ExecutionError("Aggregate functions are not allowed in ON")

        keys = _equi_join_keys(join.condition, _Scope(left_bindings), _Scope(right_bindings))
        if keys is not None:
            # Hash join: index the right table by key, probe from the left. NULL never matches.
            left_index, right_index = keys
            buckets: dict[Value, list[Row]] = {}
            for row in right.rows:
                if row[right_index] is not None:
                    buckets.setdefault(row[right_index], []).append(row)
            def matches(left: Row) -> list[Row]:
                key = left[left_index]
                return [] if key is None else buckets.get(key, [])
        else:
            def matches(left: Row) -> list[Row]:
                return [right_row for right_row in right.rows
                        if _predicate(_evaluate(join.condition, scope.row_leaf(left + right_row), {}))]

        padding: Row = (None,) * len(right.columns)
        joined: list[Row] = []
        for left in left_rows:
            found = matches(left)
            joined.extend(left + right_row for right_row in found)
            if join.left_outer and not found:
                joined.append(left + padding)  # LEFT JOIN: unmatched left rows are kept, padded with NULLs
        return joined, bindings

    # --- GROUP BY ------------------------------------------------------------

    def _groups(self, rows: list[Row], scope: _Scope,
                expressions: list[Expr]) -> Iterator[tuple[Leaf, dict[Expr, Value]]]:
        group_by = self._stmt.group_by
        # Qualified and unqualified spellings of a column (u.name / name) map to the same key.
        key_positions = {scope.resolve(expr): position for position, expr in enumerate(group_by)
                         if isinstance(expr, ColumnRef)}
        for expr in expressions:
            self._check_grouped(expr, scope, key_positions)

        groups: dict[Row, list[Row]] = {}
        for row in rows:
            key = tuple(_evaluate(expr, scope.row_leaf(row), {}) for expr in group_by)
            groups.setdefault(key, []).append(row)
        if not group_by and not groups:
            groups[()] = []  # an aggregate without GROUP BY yields one row even for empty input

        for key, members in groups.items():
            yield self._group_leaf(key, members, scope, key_positions), dict(zip(group_by, key))

    def _check_grouped(self, expr: Expr, scope: _Scope, key_positions: dict[int, int]) -> None:
        """In a grouped query, check that every column is in GROUP BY or inside an aggregate."""
        if expr in self._stmt.group_by or isinstance(expr, Literal):
            return
        if isinstance(expr, FunctionCall):
            if any(isinstance(node, FunctionCall) for node in _walk(expr.argument)):
                raise ExecutionError(f"Nested aggregate functions are not allowed: {render(expr)}")
            return
        if isinstance(expr, ColumnRef):
            if scope.resolve(expr) not in key_positions:
                raise ExecutionError(
                    f"Column '{render(expr)}' must appear in GROUP BY or be used in an aggregate function"
                )
            return
        if isinstance(expr, UnaryOp):
            self._check_grouped(expr.operand, scope, key_positions)
        else:
            self._check_grouped(expr.left, scope, key_positions)
            self._check_grouped(expr.right, scope, key_positions)

    @staticmethod
    def _group_leaf(key: Row, members: list[Row], scope: _Scope, key_positions: dict[int, int]) -> Leaf:
        def leaf(expr: Expr) -> Value:
            if isinstance(expr, FunctionCall):
                return _aggregate(expr, members, scope)
            if isinstance(expr, ColumnRef) and scope.resolve(expr) in key_positions:
                return key[key_positions[scope.resolve(expr)]]
            raise ExecutionError(f"'{render(expr)}' cannot be used in a grouped query")
        return leaf

    # --- Projection / ORDER BY -----------------------------------------------

    def _projections(self, scope: _Scope) -> tuple[list[tuple[str, Expr]], dict[str, int]]:
        """Expand the select list into ``(header, expression)`` pairs; ``*`` is expanded."""
        projections: list[tuple[str, Expr]] = []
        aliases: dict[str, int] = {}
        for item in self._stmt.items:
            if isinstance(item.expr, Star):
                star = item.expr
                expanded = [(column, ColumnRef(column, table)) for table, column in scope.bindings
                            if star.table in (None, table)]
                if not expanded:
                    raise ExecutionError(f"Unknown table reference: '{render(star)}'")
                projections.extend(expanded)
                continue
            header = item.alias or (item.expr.name if isinstance(item.expr, ColumnRef) else render(item.expr))
            if item.alias:
                aliases[item.alias] = len(projections)
            projections.append((header, item.expr))
        return projections, aliases

    def _order_plan(self, projections: list[tuple[str, Expr]],
                    aliases: dict[str, int]) -> list[tuple[OrderTarget, bool]]:
        plan: list[tuple[OrderTarget, bool]] = []
        for item in self._stmt.order_by:
            expr = item.expr
            target: OrderTarget = expr
            if isinstance(expr, Literal) and isinstance(expr.value, int) and not isinstance(expr.value, bool):
                if not 1 <= expr.value <= len(projections):
                    raise ExecutionError(f"ORDER BY position out of range: {expr.value}")
                target = expr.value - 1
            elif isinstance(expr, ColumnRef) and expr.table is None and expr.name in aliases:
                target = aliases[expr.name]
            plan.append((target, item.descending))
        return plan

    @staticmethod
    def _sort(entries: list[tuple[Row, Row]], descending: list[bool]) -> None:
        """Stable multi-key sort; NULL is treated as the smallest value."""
        try:
            for index in reversed(range(len(descending))):
                entries.sort(key=lambda entry: (entry[1][index] is not None, entry[1][index]),
                             reverse=descending[index])
        except TypeError:
            raise ExecutionError("ORDER BY: incomparable mixed types") from None


def _equi_join_keys(condition: Expr, left: _Scope, right: _Scope) -> tuple[int, int] | None:
    """Find hash-join key positions for a condition of the form ``ON left.a = right.b``."""
    if not (isinstance(condition, BinaryOp) and condition.op == "="
            and isinstance(condition.left, ColumnRef) and isinstance(condition.right, ColumnRef)):
        return None
    for left_ref, right_ref in ((condition.left, condition.right), (condition.right, condition.left)):
        left_matches, right_matches = left.find(left_ref), right.find(right_ref)
        if len(left_matches) == 1 and len(right_matches) == 1:
            return left_matches[0], right_matches[0]
    return None
