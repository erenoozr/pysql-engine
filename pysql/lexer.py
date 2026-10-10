"""Deterministic lexer that turns SQL source text into a token stream."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import LexerError


class TokenType(Enum):
    """Token kinds the lexer can produce."""

    KEYWORD = "KEYWORD"
    IDENTIFIER = "IDENTIFIER"
    NUMBER = "NUMBER"
    STRING = "STRING"
    OPERATOR = "OPERATOR"
    LPAREN = "("
    RPAREN = ")"
    COMMA = ","
    DOT = "."
    STAR = "*"
    SEMICOLON = ";"
    EOF = "EOF"


KEYWORDS: frozenset[str] = frozenset({
    "SELECT", "FROM", "WHERE", "INSERT", "INTO", "VALUES", "CREATE", "TABLE",
    "INNER", "JOIN", "ON", "GROUP", "BY", "HAVING", "ORDER", "ASC", "DESC",
    "LIMIT", "OFFSET", "AND", "OR", "NOT", "AS", "NULL", "TRUE", "FALSE", "IS",
    "DISTINCT", "LEFT", "OUTER", "UPDATE", "SET", "DELETE",
})

_PUNCTUATION: dict[str, TokenType] = {
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    ",": TokenType.COMMA,
    ".": TokenType.DOT,
    "*": TokenType.STAR,
    ";": TokenType.SEMICOLON,
}
_TWO_CHAR_OPERATORS: frozenset[str] = frozenset({">=", "<=", "!=", "<>"})
_ONE_CHAR_OPERATORS: frozenset[str] = frozenset({"=", "<", ">", "-", "+", "/"})


@dataclass(frozen=True, slots=True)
class Token:
    """A single lexical unit and its (1-based) position in the source.

    ``value`` is the upper-cased text for keywords, an ``int``/``float`` for
    numbers and the unescaped content for strings.
    """

    type: TokenType
    value: str | int | float
    line: int
    column: int

    def describe(self) -> str:
        """Human-readable form used in error messages."""
        return "end of input" if self.type is TokenType.EOF else f"'{self.value}'"


class Lexer:
    """Splits SQL text into tokens in a single left-to-right pass."""

    def __init__(self, source: str) -> None:
        self._source = source
        self._pos = 0
        self._line = 1
        self._column = 1

    def tokenize(self) -> list[Token]:
        """Tokenize the whole source; the list always ends with ``EOF``.

        Raises:
            LexerError: Unknown character, unterminated string or malformed number.
        """
        tokens: list[Token] = []
        while True:
            self._skip_whitespace_and_comments()
            if self._pos >= len(self._source):
                tokens.append(Token(TokenType.EOF, "", self._line, self._column))
                return tokens
            tokens.append(self._next_token())

    def _peek(self, offset: int = 0) -> str:
        index = self._pos + offset
        return self._source[index] if index < len(self._source) else ""

    def _advance(self, count: int = 1) -> None:
        for _ in range(count):
            if self._source[self._pos] == "\n":
                self._line += 1
                self._column = 1
            else:
                self._column += 1
            self._pos += 1

    def _skip_whitespace_and_comments(self) -> None:
        while self._pos < len(self._source):
            if self._peek().isspace():
                self._advance()
            elif self._peek() == "-" and self._peek(1) == "-":
                while self._pos < len(self._source) and self._peek() != "\n":
                    self._advance()
            else:
                return

    def _next_token(self) -> Token:
        char = self._peek()
        line, column = self._line, self._column
        if char.isalpha() or char == "_":
            return self._read_word(line, column)
        if "0" <= char <= "9":
            return self._read_number(line, column)
        if char == "'":
            return self._read_string(line, column)
        pair = char + self._peek(1)
        if pair in _TWO_CHAR_OPERATORS:
            self._advance(2)
            return Token(TokenType.OPERATOR, "!=" if pair == "<>" else pair, line, column)
        if char in _ONE_CHAR_OPERATORS:
            self._advance()
            return Token(TokenType.OPERATOR, char, line, column)
        if char in _PUNCTUATION:
            self._advance()
            return Token(_PUNCTUATION[char], char, line, column)
        raise LexerError(f"Unrecognized character: {char!r}", line, column)

    def _read_word(self, line: int, column: int) -> Token:
        start = self._pos
        while self._peek().isalnum() or self._peek() == "_":
            self._advance()
        word = self._source[start:self._pos]
        if word.upper() in KEYWORDS:
            return Token(TokenType.KEYWORD, word.upper(), line, column)
        return Token(TokenType.IDENTIFIER, word, line, column)

    def _read_number(self, line: int, column: int) -> Token:
        start = self._pos
        while "0" <= self._peek() <= "9":
            self._advance()
        is_float = self._peek() == "." and "0" <= self._peek(1) <= "9"
        if is_float:
            self._advance()
            while "0" <= self._peek() <= "9":
                self._advance()
        if self._peek().isalpha() or self._peek() == "_":
            raise LexerError(f"Invalid number: {self._source[start:self._pos + 1]!r}", line, column)
        text = self._source[start:self._pos]
        return Token(TokenType.NUMBER, float(text) if is_float else int(text), line, column)

    def _read_string(self, line: int, column: int) -> Token:
        self._advance()  # opening quote
        chars: list[str] = []
        while True:
            if self._pos >= len(self._source):
                raise LexerError("Unterminated string literal", line, column)
            char = self._peek()
            if char == "'":
                if self._peek(1) == "'":  # SQL escape: '' -> '
                    chars.append("'")
                    self._advance(2)
                    continue
                self._advance()
                return Token(TokenType.STRING, "".join(chars), line, column)
            chars.append(char)
            self._advance()
