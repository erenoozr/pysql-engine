"""Recursive descent parser that builds an AST from the token stream.

Expression grammar (lowest to highest precedence)::

    expression := or_expr
    or_expr    := and_expr ("OR" and_expr)*
    and_expr   := not_expr ("AND" not_expr)*
    not_expr   := "NOT" not_expr | comparison
    comparison := additive (("=" | "!=" | "<" | ">" | "<=" | ">=") additive
                            | "IS" ["NOT"] "NULL")?
    additive   := term (("+" | "-") term)*
    term       := unary (("*" | "/") unary)*
    unary      := "-" unary | primary
    primary    := NUMBER | STRING | NULL | TRUE | FALSE
                | "(" expression ")"
                | IDENT "(" ("*" | expression) ")"      -- aggregate
                | IDENT ("." IDENT)?                     -- column
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

from .errors import ParserError
from .lexer import Lexer, Token, TokenType
from .nodes import (
    AGGREGATE_FUNCTIONS, BinaryOp, ColumnDef, ColumnRef, CreateTable, Delete, Expr, FunctionCall,
    Insert, Join, Literal, OrderItem, Select, SelectItem, Star, Statement, TableRef, UnaryOp, Update,
)

_COMPARISON_OPERATORS: frozenset[str] = frozenset({"=", "!=", "<", ">", "<=", ">="})

T = TypeVar("T")


def parse(sql: str) -> list[Statement]:
    """Parse SQL text (one or more ``;``-separated statements) into a list of ASTs."""
    return Parser(Lexer(sql).tokenize()).parse()


class Parser:
    """Consumes ``Lexer`` output and produces ``Statement`` nodes."""

    def __init__(self, tokens: Sequence[Token]) -> None:
        if not tokens or tokens[-1].type is not TokenType.EOF:
            raise ValueError("Token list must end with EOF")
        self._tokens = list(tokens)
        self._pos = 0

    def parse(self) -> list[Statement]:
        """Parse the whole token stream.

        Raises:
            ParserError: When the stream does not match the grammar; the message includes the location.
        """
        statements: list[Statement] = []
        while not self._check(TokenType.EOF):
            if self._match(TokenType.SEMICOLON):
                continue
            statements.append(self._statement())
            if not self._check(TokenType.EOF):
                self._expect(TokenType.SEMICOLON, "';' or end of input")
        return statements

    # --- Token helpers --------------------------------------------------------

    def _peek(self, offset: int = 0) -> Token:
        return self._tokens[min(self._pos + offset, len(self._tokens) - 1)]

    def _advance(self) -> Token:
        token = self._peek()
        if token.type is not TokenType.EOF:
            self._pos += 1
        return token

    def _check(self, token_type: TokenType) -> bool:
        return self._peek().type is token_type

    def _match(self, token_type: TokenType) -> bool:
        if self._check(token_type):
            self._advance()
            return True
        return False

    def _check_keyword(self, *words: str) -> bool:
        token = self._peek()
        return token.type is TokenType.KEYWORD and token.value in words

    def _match_keyword(self, word: str) -> bool:
        if self._check_keyword(word):
            self._advance()
            return True
        return False

    def _expect_keyword(self, word: str) -> None:
        if not self._match_keyword(word):
            raise self._error(f"Expected '{word}'")

    def _expect(self, token_type: TokenType, description: str) -> Token:
        if not self._check(token_type):
            raise self._error(f"Expected {description}")
        return self._advance()

    def _error(self, message: str) -> ParserError:
        token = self._peek()
        return ParserError(f"{message}; found {token.describe()}", token.line, token.column)

    def _identifier(self, what: str) -> str:
        return str(self._expect(TokenType.IDENTIFIER, what).value).lower()

    def _optional_alias(self) -> str | None:
        if self._match_keyword("AS"):
            return self._identifier("alias")
        if self._check(TokenType.IDENTIFIER):
            return self._identifier("alias")
        return None

    def _comma_list(self, parse_item: Callable[[], T]) -> tuple[T, ...]:
        items = [parse_item()]
        while self._match(TokenType.COMMA):
            items.append(parse_item())
        return tuple(items)

    # --- Statements -----------------------------------------------------------

    def _statement(self) -> Statement:
        if self._check_keyword("SELECT"):
            return self._select()
        if self._check_keyword("INSERT"):
            return self._insert()
        if self._check_keyword("CREATE"):
            return self._create_table()
        if self._check_keyword("UPDATE"):
            return self._update()
        if self._check_keyword("DELETE"):
            return self._delete()
        raise self._error("Expected SELECT, INSERT, UPDATE, DELETE or CREATE TABLE")

    def _create_table(self) -> CreateTable:
        self._expect_keyword("CREATE")
        self._expect_keyword("TABLE")
        name = self._identifier("table name")
        self._expect(TokenType.LPAREN, "'('")
        columns = self._comma_list(self._column_def)
        self._expect(TokenType.RPAREN, "')'")
        return CreateTable(name, columns)

    def _column_def(self) -> ColumnDef:
        name = self._identifier("column name")
        type_name = str(self._advance().value).upper() if self._check(TokenType.IDENTIFIER) else "ANY"
        return ColumnDef(name, type_name)

    def _insert(self) -> Insert:
        self._expect_keyword("INSERT")
        self._expect_keyword("INTO")
        table = self._identifier("table name")
        columns: tuple[str, ...] | None = None
        if self._match(TokenType.LPAREN):
            columns = self._comma_list(lambda: self._identifier("column name"))
            self._expect(TokenType.RPAREN, "')'")
        self._expect_keyword("VALUES")
        return Insert(table, self._comma_list(self._value_tuple), columns)

    def _update(self) -> Update:
        self._expect_keyword("UPDATE")
        table = self._identifier("table name")
        self._expect_keyword("SET")
        assignments = self._comma_list(self._assignment)
        where = self._expression() if self._match_keyword("WHERE") else None
        return Update(table, assignments, where)

    def _assignment(self) -> tuple[str, Expr]:
        column = self._identifier("column name")
        token = self._peek()
        if token.type is not TokenType.OPERATOR or token.value != "=":
            raise self._error("Expected '='")
        self._advance()
        return column, self._expression()

    def _delete(self) -> Delete:
        self._expect_keyword("DELETE")
        self._expect_keyword("FROM")
        table = self._identifier("table name")
        where = self._expression() if self._match_keyword("WHERE") else None
        return Delete(table, where)

    def _value_tuple(self) -> tuple[Expr, ...]:
        self._expect(TokenType.LPAREN, "'('")
        values = self._comma_list(self._expression)
        self._expect(TokenType.RPAREN, "')'")
        return values

    def _select(self) -> Select:
        self._expect_keyword("SELECT")
        distinct = self._match_keyword("DISTINCT")
        items = self._comma_list(self._select_item)
        self._expect_keyword("FROM")
        from_table = self._table_ref()
        joins: list[Join] = []
        while self._check_keyword("INNER", "LEFT", "JOIN"):
            left_outer = self._match_keyword("LEFT")
            if left_outer:
                self._match_keyword("OUTER")
            else:
                self._match_keyword("INNER")
            self._expect_keyword("JOIN")
            table = self._table_ref()
            self._expect_keyword("ON")
            joins.append(Join(table, self._expression(), left_outer))
        where = self._expression() if self._match_keyword("WHERE") else None
        group_by: tuple[Expr, ...] = ()
        if self._match_keyword("GROUP"):
            self._expect_keyword("BY")
            group_by = self._comma_list(self._expression)
        having = self._expression() if self._match_keyword("HAVING") else None
        order_by: tuple[OrderItem, ...] = ()
        if self._match_keyword("ORDER"):
            self._expect_keyword("BY")
            order_by = self._comma_list(self._order_item)
        limit = self._count("LIMIT") if self._match_keyword("LIMIT") else None
        offset = self._count("OFFSET") if self._match_keyword("OFFSET") else None
        return Select(items, from_table, tuple(joins), where, group_by, having, order_by,
                      limit, offset, distinct)

    def _count(self, clause: str) -> int:
        token = self._peek()
        if token.type is not TokenType.NUMBER or not isinstance(token.value, int):
            raise self._error(f"Expected a non-negative integer for {clause}")
        self._advance()
        return token.value

    def _select_item(self) -> SelectItem:
        if self._match(TokenType.STAR):
            return SelectItem(Star())
        if (self._check(TokenType.IDENTIFIER) and self._peek(1).type is TokenType.DOT
                and self._peek(2).type is TokenType.STAR):
            table = self._identifier("table name")
            self._advance()
            self._advance()
            return SelectItem(Star(table))
        expr = self._expression()
        return SelectItem(expr, self._optional_alias())

    def _table_ref(self) -> TableRef:
        name = self._identifier("table name")
        return TableRef(name, self._optional_alias())

    def _order_item(self) -> OrderItem:
        expr = self._expression()
        if self._match_keyword("DESC"):
            return OrderItem(expr, descending=True)
        self._match_keyword("ASC")
        return OrderItem(expr)

    # --- Expressions ----------------------------------------------------------

    def _expression(self) -> Expr:
        return self._or()

    def _or(self) -> Expr:
        left = self._and()
        while self._match_keyword("OR"):
            left = BinaryOp("OR", left, self._and())
        return left

    def _and(self) -> Expr:
        left = self._not()
        while self._match_keyword("AND"):
            left = BinaryOp("AND", left, self._not())
        return left

    def _not(self) -> Expr:
        if self._match_keyword("NOT"):
            return UnaryOp("NOT", self._not())
        return self._comparison()

    def _comparison(self) -> Expr:
        left = self._additive()
        token = self._peek()
        if token.type is TokenType.OPERATOR and token.value in _COMPARISON_OPERATORS:
            self._advance()
            return BinaryOp(str(token.value), left, self._additive())
        if self._match_keyword("IS"):
            negated = self._match_keyword("NOT")
            self._expect_keyword("NULL")
            return UnaryOp("IS NOT NULL" if negated else "IS NULL", left)
        return left

    def _additive(self) -> Expr:
        left = self._term()
        while self._peek().type is TokenType.OPERATOR and self._peek().value in ("+", "-"):
            op = str(self._advance().value)
            left = BinaryOp(op, left, self._term())
        return left

    def _term(self) -> Expr:
        left = self._unary()
        while self._check(TokenType.STAR) or (self._peek().type is TokenType.OPERATOR
                                              and self._peek().value == "/"):
            op = str(self._advance().value)
            left = BinaryOp(op, left, self._unary())
        return left

    def _unary(self) -> Expr:
        token = self._peek()
        if token.type is TokenType.OPERATOR and token.value == "-":
            self._advance()
            operand = self._unary()
            if (isinstance(operand, Literal) and isinstance(operand.value, (int, float))
                    and not isinstance(operand.value, bool)):
                return Literal(-operand.value)
            return UnaryOp("-", operand)
        return self._primary()

    def _primary(self) -> Expr:
        token = self._peek()
        if token.type in (TokenType.NUMBER, TokenType.STRING):
            self._advance()
            return Literal(token.value)
        if self._match_keyword("NULL"):
            return Literal(None)
        if self._match_keyword("TRUE"):
            return Literal(True)
        if self._match_keyword("FALSE"):
            return Literal(False)
        if self._match(TokenType.LPAREN):
            expr = self._expression()
            self._expect(TokenType.RPAREN, "')'")
            return expr
        if token.type is TokenType.IDENTIFIER:
            if self._peek(1).type is TokenType.LPAREN:
                return self._function_call()
            name = self._identifier("column name")
            if self._match(TokenType.DOT):
                return ColumnRef(self._identifier("column name"), table=name)
            return ColumnRef(name)
        raise self._error("Expected an expression")

    def _function_call(self) -> FunctionCall:
        token = self._advance()
        name = str(token.value).upper()
        if name not in AGGREGATE_FUNCTIONS:
            raise ParserError(f"Unknown function: '{token.value}'", token.line, token.column)
        self._expect(TokenType.LPAREN, "'('")
        argument: Expr | Star
        if self._check(TokenType.STAR):
            if name != "COUNT":
                raise self._error(f"'*' can only be used with COUNT, not {name}")
            self._advance()
            argument = Star()
        else:
            argument = self._expression()
        self._expect(TokenType.RPAREN, "')'")
        return FunctionCall(name, argument)
