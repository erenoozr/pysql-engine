"""End-to-end test suite for PySQL-Engine (``unittest`` only).

Run from the project root: ``python -m unittest -v``
"""

from __future__ import annotations

import io
import unittest
from collections.abc import Iterator

from pysql import (
    Database, ExecutionError, Lexer, LexerError, ParserError, SQLError, TabularFormatter, TokenType, parse,
)
from pysql.nodes import BinaryOp, ColumnRef, CreateTable, FunctionCall, Insert, Literal, Select, Star
from pysql.repl import Repl

FIXTURE_SQL = """
CREATE TABLE users (id INT, name TEXT, age INT, city TEXT);
INSERT INTO users VALUES
    (1, 'Ali', 30, 'Istanbul'),
    (2, 'Ayse', 25, 'Ankara'),
    (3, 'Mehmet', 35, 'Istanbul'),
    (4, 'Zeynep', 28, 'Izmir'),
    (5, 'Can', NULL, 'Ankara');
CREATE TABLE orders (id INT, user_id INT, product TEXT, amount REAL);
INSERT INTO orders VALUES
    (1, 1, 'Laptop', 1500), (2, 1, 'Mouse', 25.5), (3, 2, 'Phone', 800),
    (4, 3, 'Monitor', 300), (5, 3, 'Keyboard', 75), (6, 99, 'Orphan', 10);
"""


class FixtureMixin(unittest.TestCase):
    """Sets up a fresh ``users``/``orders`` database for every test."""

    def setUp(self) -> None:
        self.db = Database()
        self.db.execute_script(FIXTURE_SQL)

    def rows(self, sql: str) -> list[tuple[object, ...]]:
        return list(self.db.execute(sql).rows)


# --- Stage 1: Lexer -------------------------------------------------------------------

class LexerTests(unittest.TestCase):

    def test_tokenizes_all_categories(self) -> None:
        tokens = Lexer("SELECT a, t.b FROM t WHERE x >= 3.14 AND y != 'it''s' OR z <= 42;").tokenize()
        pairs = [(token.type, token.value) for token in tokens]
        self.assertEqual(pairs, [
            (TokenType.KEYWORD, "SELECT"), (TokenType.IDENTIFIER, "a"), (TokenType.COMMA, ","),
            (TokenType.IDENTIFIER, "t"), (TokenType.DOT, "."), (TokenType.IDENTIFIER, "b"),
            (TokenType.KEYWORD, "FROM"), (TokenType.IDENTIFIER, "t"), (TokenType.KEYWORD, "WHERE"),
            (TokenType.IDENTIFIER, "x"), (TokenType.OPERATOR, ">="), (TokenType.NUMBER, 3.14),
            (TokenType.KEYWORD, "AND"), (TokenType.IDENTIFIER, "y"), (TokenType.OPERATOR, "!="),
            (TokenType.STRING, "it's"), (TokenType.KEYWORD, "OR"), (TokenType.IDENTIFIER, "z"),
            (TokenType.OPERATOR, "<="), (TokenType.NUMBER, 42), (TokenType.SEMICOLON, ";"),
            (TokenType.EOF, ""),
        ])

    def test_keywords_are_case_insensitive_and_numbers_typed(self) -> None:
        tokens = Lexer("select From 7 7.5").tokenize()
        self.assertEqual([t.value for t in tokens[:2]], ["SELECT", "FROM"])
        self.assertIsInstance(tokens[2].value, int)
        self.assertIsInstance(tokens[3].value, float)

    def test_all_operators_and_alias_not_equal(self) -> None:
        tokens = Lexer("= != <> > < >= <= ( ) *").tokenize()
        self.assertEqual([t.value for t in tokens[:-1]], ["=", "!=", "!=", ">", "<", ">=", "<=", "(", ")", "*"])

    def test_comments_and_positions(self) -> None:
        tokens = Lexer("-- comment\n  SELECT").tokenize()
        self.assertEqual((tokens[0].value, tokens[0].line, tokens[0].column), ("SELECT", 2, 3))

    def test_unknown_character_reports_position(self) -> None:
        with self.assertRaises(LexerError) as ctx:
            Lexer("SELECT @ FROM t").tokenize()
        self.assertEqual((ctx.exception.line, ctx.exception.column), (1, 8))
        self.assertIn("'@'", str(ctx.exception))

    def test_unterminated_string(self) -> None:
        with self.assertRaisesRegex(LexerError, "Unterminated"):
            Lexer("SELECT 'abc").tokenize()

    def test_malformed_number(self) -> None:
        with self.assertRaisesRegex(LexerError, "Invalid number"):
            Lexer("SELECT 42abc FROM t").tokenize()

    def test_lone_bang_is_error(self) -> None:
        with self.assertRaises(LexerError):
            Lexer("a ! b").tokenize()


# --- Stage 2: Parser ------------------------------------------------------------------

class ParserTests(unittest.TestCase):

    def where(self, condition: str) -> object:
        statement = parse(f"SELECT a FROM t WHERE {condition}")[0]
        assert isinstance(statement, Select)
        return statement.where

    def test_and_binds_tighter_than_or(self) -> None:
        tree = self.where("a = 1 OR b = 2 AND c = 3")
        self.assertIsInstance(tree, BinaryOp)
        assert isinstance(tree, BinaryOp)
        self.assertEqual(tree.op, "OR")
        self.assertEqual(tree.right, BinaryOp("AND", BinaryOp("=", ColumnRef("b"), Literal(2)),
                                              BinaryOp("=", ColumnRef("c"), Literal(3))))

    def test_parentheses_override_precedence(self) -> None:
        tree = self.where("(a = 1 OR b = 2) AND c = 3")
        assert isinstance(tree, BinaryOp)
        self.assertEqual(tree.op, "AND")
        self.assertEqual(tree.left.op, "OR")  # type: ignore[union-attr]

    def test_full_select_shape(self) -> None:
        statement = parse(
            "SELECT u.name AS n, COUNT(*) FROM Users u INNER JOIN orders o ON u.id = o.user_id "
            "WHERE o.amount > -5 GROUP BY u.name HAVING COUNT(*) > 1 ORDER BY n DESC, 2 LIMIT 3;"
        )[0]
        assert isinstance(statement, Select)
        self.assertEqual(statement.from_table.name, "users")
        self.assertEqual(statement.from_table.alias, "u")
        self.assertEqual(statement.items[0].alias, "n")
        self.assertEqual(statement.items[1].expr, FunctionCall("COUNT", Star()))
        self.assertEqual(len(statement.joins), 1)
        self.assertEqual(statement.where, BinaryOp(">", ColumnRef("amount", "o"), Literal(-5)))
        self.assertEqual(statement.group_by, (ColumnRef("name", "u"),))
        self.assertIsNotNone(statement.having)
        self.assertEqual([item.descending for item in statement.order_by], [True, False])
        self.assertEqual(statement.limit, 3)

    def test_create_and_multi_row_insert(self) -> None:
        create, insert = parse("CREATE TABLE t (a INT, b); INSERT INTO t VALUES (1, 'x'), (-2, NULL)")
        assert isinstance(create, CreateTable) and isinstance(insert, Insert)
        self.assertEqual([(c.name, c.type_name) for c in create.columns], [("a", "INT"), ("b", "ANY")])
        self.assertEqual(insert.rows, ((Literal(1), Literal("x")), (Literal(-2), Literal(None))))

    def test_table_star(self) -> None:
        statement = parse("SELECT t.* FROM t")[0]
        assert isinstance(statement, Select)
        self.assertEqual(statement.items[0].expr, Star("t"))

    def test_syntax_errors(self) -> None:
        bad_queries = [
            "SELECT a",                          # missing FROM
            "SELECT FROM t",                     # empty column list
            "SELECT a FROM t WHERE",             # missing condition
            "SELECT a FROM t WHERE (a = 1",      # unclosed parenthesis
            "SELECT a FROM t LIMIT 1.5",         # non-integer LIMIT
            "SELECT a FROM t x y",               # extra token
            "SELECT LOWER(a) FROM t",            # unknown function
            "SELECT SUM(*) FROM t",              # * only with COUNT
            "INSERT INTO t (1)",                 # missing VALUES
            "CREATE TABLE t ()",                 # no columns
            "DROP TABLE t",                      # unsupported statement
            "SELECT a FROM t WHERE a IS 5",      # IS only with NULL
            "UPDATE t SET a 1",                  # missing '='
            "DELETE t",                          # missing FROM
            "SELECT a FROM t OFFSET x",          # non-numeric OFFSET
        ]
        for sql in bad_queries:
            with self.subTest(sql=sql), self.assertRaises(ParserError):
                parse(sql)

    def test_error_message_has_location(self) -> None:
        with self.assertRaises(ParserError) as ctx:
            parse("SELECT a\nFROM")
        self.assertEqual(ctx.exception.line, 2)
        self.assertIn("end of input", str(ctx.exception))


# --- Stage 3: Storage -------------------------------------------------------------------

class StorageTests(FixtureMixin):

    def test_create_and_insert_counts(self) -> None:
        self.assertEqual(len(self.db.table("users").rows), 5)
        self.assertEqual(self.db.execute("INSERT INTO users VALUES (6, 'Ece', 22, 'Bursa'), "
                                         "(7, 'Ege', 40, 'Bursa')").message, "2 rows inserted")
        self.assertEqual(self.rows("SELECT COUNT(*) FROM users"), [(7,)])

    def test_real_column_coerces_int(self) -> None:
        self.assertEqual(self.rows("SELECT amount FROM orders WHERE id = 1"), [(1500.0,)])
        self.assertIsInstance(self.db.table("orders").rows[0][3], float)

    def test_type_mismatch_rejected(self) -> None:
        with self.assertRaisesRegex(ExecutionError, "INTEGER"):
            self.db.execute("INSERT INTO users VALUES ('x', 'A', 1, 'B')")

    def test_insert_is_atomic(self) -> None:
        with self.assertRaises(ExecutionError):
            self.db.execute("INSERT INTO users VALUES (10, 'A', 1, 'B'), (11, 'B', 'age', 'C')")
        self.assertEqual(len(self.db.table("users").rows), 5)

    def test_value_count_mismatch(self) -> None:
        with self.assertRaisesRegex(ExecutionError, "expects 4 values, got 2"):
            self.db.execute("INSERT INTO users VALUES (1, 'A')")

    def test_insert_requires_constants(self) -> None:
        with self.assertRaisesRegex(ExecutionError, "constants"):
            self.db.execute("INSERT INTO users VALUES (id, 'A', 1, 'B')")

    def test_catalog_errors(self) -> None:
        cases = [
            ("CREATE TABLE users (a)", "already exists"),
            ("CREATE TABLE t (a INT, a TEXT)", "Duplicate column"),
            ("CREATE TABLE t (a BLOB)", "Unsupported column type"),
            ("INSERT INTO missing VALUES (1)", "Table not found"),
            ("SELECT * FROM missing", "Table not found"),
        ]
        for sql, message in cases:
            with self.subTest(sql=sql), self.assertRaisesRegex(ExecutionError, message):
                self.db.execute(sql)

    def test_table_names_case_insensitive_and_schema(self) -> None:
        self.assertEqual(len(self.rows("SELECT * FROM USERS")), 5)
        self.assertEqual(self.db.table("Users").schema_sql(),
                         "CREATE TABLE users (id INTEGER, name TEXT, age INTEGER, city TEXT);")

    def test_boolean_and_any_columns(self) -> None:
        self.db.execute("CREATE TABLE flags (k, v BOOL)")
        self.db.execute("INSERT INTO flags VALUES ('a', TRUE), (1, FALSE), (NULL, NULL)")
        self.assertEqual(self.rows("SELECT k FROM flags WHERE v = TRUE"), [("a",)])
        self.assertEqual(self.rows("SELECT k FROM flags WHERE NOT v"), [(1,)])


# --- Stage 3: WHERE ---------------------------------------------------------------------

class WhereTests(FixtureMixin):

    def names(self, condition: str) -> list[object]:
        return [row[0] for row in self.rows(f"SELECT name FROM users WHERE {condition} ORDER BY id")]

    def test_nested_logic(self) -> None:
        self.assertEqual(self.names("(city = 'Istanbul' OR city = 'Izmir') AND age > 28"), ["Ali", "Mehmet"])
        self.assertEqual(self.names("city = 'Istanbul' OR city = 'Izmir' AND age > 28"), ["Ali", "Mehmet"])
        self.assertEqual(self.names("((age >= 25 AND age <= 28) OR (id = 1 AND NOT city != 'Istanbul'))"),
                         ["Ali", "Ayse", "Zeynep"])

    def test_comparison_operators(self) -> None:
        self.assertEqual(self.names("age < 28"), ["Ayse"])
        self.assertEqual(self.names("age <= 28"), ["Ayse", "Zeynep"])
        self.assertEqual(self.names("age != 30"), ["Ayse", "Mehmet", "Zeynep"])
        self.assertEqual(self.names("name > 'Mehmet'"), ["Zeynep"])

    def test_null_three_valued_logic(self) -> None:
        self.assertEqual(self.names("age > 0"), ["Ali", "Ayse", "Mehmet", "Zeynep"])
        self.assertEqual(self.names("NOT (age > 100)"), ["Ali", "Ayse", "Mehmet", "Zeynep"])
        self.assertEqual(self.names("age > 100 OR city = 'Ankara'"), ["Ayse", "Can"])
        self.assertEqual(self.names("age = NULL"), [])

    def test_type_errors(self) -> None:
        with self.assertRaisesRegex(ExecutionError, "Incomparable"):
            self.db.execute("SELECT name FROM users WHERE age > 'thirty'")
        with self.assertRaisesRegex(ExecutionError, "boolean expression"):
            self.db.execute("SELECT name FROM users WHERE age")
        self.assertEqual(self.names("age = 'thirty'"), [])

    def test_unknown_column_detected_on_empty_table(self) -> None:
        self.db.execute("CREATE TABLE empty (a INT)")
        with self.assertRaisesRegex(ExecutionError, "Unknown column"):
            self.db.execute("SELECT a FROM empty WHERE b = 1")


# --- Stage 3: JOIN ----------------------------------------------------------------------

class JoinTests(FixtureMixin):

    def test_hash_join_on_equality(self) -> None:
        result = self.db.execute(
            "SELECT u.name, o.product FROM users u INNER JOIN orders o ON u.id = o.user_id ORDER BY o.id")
        self.assertEqual(result.columns, ("name", "product"))
        self.assertEqual(list(result.rows), [("Ali", "Laptop"), ("Ali", "Mouse"), ("Ayse", "Phone"),
                                             ("Mehmet", "Monitor"), ("Mehmet", "Keyboard")])

    def test_reversed_on_condition_and_plain_join_keyword(self) -> None:
        self.assertEqual(len(self.rows("SELECT * FROM users JOIN orders ON orders.user_id = users.id")), 5)

    def test_nested_loop_join_with_complex_condition(self) -> None:
        rows = self.rows("SELECT name, product FROM users u JOIN orders o "
                         "ON u.id = o.user_id AND o.amount > 100 ORDER BY amount DESC")
        self.assertEqual(rows, [("Ali", "Laptop"), ("Ayse", "Phone"), ("Mehmet", "Monitor")])

    def test_star_expansion_and_table_star(self) -> None:
        result = self.db.execute("SELECT * FROM users u JOIN orders o ON u.id = o.user_id")
        self.assertEqual(result.columns, ("id", "name", "age", "city", "id", "user_id", "product", "amount"))
        self.assertEqual(self.db.execute("SELECT o.* FROM users u JOIN orders o ON u.id = o.user_id").columns,
                         ("id", "user_id", "product", "amount"))

    def test_ambiguous_column(self) -> None:
        with self.assertRaisesRegex(ExecutionError, "Ambiguous"):
            self.db.execute("SELECT id FROM users u JOIN orders o ON u.id = o.user_id")

    def test_self_join_requires_alias(self) -> None:
        with self.assertRaisesRegex(ExecutionError, "alias"):
            self.db.execute("SELECT * FROM users JOIN users ON users.id = users.id")
        rows = self.rows("SELECT a.name, b.name FROM users a JOIN users b ON a.city = b.city "
                         "WHERE a.id < b.id ORDER BY a.id")
        self.assertEqual(rows, [("Ali", "Mehmet"), ("Ayse", "Can")])

    def test_three_table_join(self) -> None:
        self.db.execute("CREATE TABLE cities (name TEXT, region TEXT)")
        self.db.execute("INSERT INTO cities VALUES ('Istanbul', 'Marmara'), ('Ankara', 'Central')")
        rows = self.rows("SELECT u.name, c.region, o.product FROM users u JOIN orders o ON u.id = o.user_id "
                         "JOIN cities c ON c.name = u.city WHERE c.region = 'Marmara' ORDER BY o.id")
        self.assertEqual([row[2] for row in rows], ["Laptop", "Mouse", "Monitor", "Keyboard"])

    def test_null_keys_never_match(self) -> None:
        self.db.execute("INSERT INTO orders VALUES (7, NULL, 'Lost', 1)")
        self.db.execute("INSERT INTO users VALUES (NULL, 'Ghost', 1, 'X')")
        self.assertEqual(len(self.rows("SELECT * FROM users u JOIN orders o ON u.id = o.user_id")), 5)


# --- Stage 3: Aggregate / GROUP BY / HAVING ---------------------------------------------

class AggregateTests(FixtureMixin):

    def test_whole_table_aggregates(self) -> None:
        result = self.db.execute(
            "SELECT COUNT(*), COUNT(age), SUM(age), AVG(age), MIN(age), MAX(age), MIN(name) FROM users")
        self.assertEqual(result.columns, ("COUNT(*)", "COUNT(age)", "SUM(age)", "AVG(age)",
                                          "MIN(age)", "MAX(age)", "MIN(name)"))
        self.assertEqual(list(result.rows), [(5, 4, 118, 29.5, 25, 35, "Ali")])

    def test_group_by_with_aggregates(self) -> None:
        rows = self.rows("SELECT city, COUNT(*) AS n, AVG(age) FROM users GROUP BY city ORDER BY city")
        self.assertEqual(rows, [("Ankara", 2, 25.0), ("Istanbul", 2, 32.5), ("Izmir", 1, 28.0)])

    def test_group_by_having(self) -> None:
        rows = self.rows("SELECT city, COUNT(*) FROM users GROUP BY city HAVING COUNT(*) > 1 ORDER BY city")
        self.assertEqual(rows, [("Ankara", 2), ("Istanbul", 2)])
        rows = self.rows("SELECT city FROM users GROUP BY city HAVING MAX(age) >= 30 AND city != 'Izmir'")
        self.assertEqual(rows, [("Istanbul",)])

    def test_join_group_order_by_alias(self) -> None:
        rows = self.rows("SELECT u.name, SUM(o.amount) AS total, COUNT(o.id) AS cnt FROM users u "
                         "JOIN orders o ON u.id = o.user_id GROUP BY name ORDER BY total DESC")
        self.assertEqual(rows, [("Ali", 1525.5, 2), ("Ayse", 800.0, 1), ("Mehmet", 375.0, 2)])

    def test_order_by_aggregate_not_in_select(self) -> None:
        rows = self.rows("SELECT city FROM users GROUP BY city ORDER BY COUNT(*) DESC, city LIMIT 2")
        self.assertEqual(rows, [("Ankara",), ("Istanbul",)])

    def test_aggregates_on_empty_input(self) -> None:
        self.assertEqual(self.rows("SELECT COUNT(*), SUM(age), MAX(age) FROM users WHERE age > 100"),
                         [(0, None, None)])
        self.assertEqual(self.rows("SELECT city, COUNT(*) FROM users WHERE age > 100 GROUP BY city"), [])

    def test_having_without_group_by(self) -> None:
        self.assertEqual(self.rows("SELECT COUNT(*) FROM users HAVING COUNT(*) > 3"), [(5,)])
        self.assertEqual(self.rows("SELECT COUNT(*) FROM users HAVING COUNT(*) > 30"), [])

    def test_grouping_errors(self) -> None:
        cases = [
            ("SELECT name, COUNT(*) FROM users GROUP BY city", "must appear in GROUP BY"),
            ("SELECT name, COUNT(*) FROM users", "must appear in GROUP BY"),
            ("SELECT name FROM users WHERE COUNT(*) > 1", "not allowed in WHERE"),
            ("SELECT SUM(MAX(age)) FROM users", "Nested aggregate"),
            ("SELECT SUM(name) FROM users", "only with numeric"),
            ("SELECT city FROM users GROUP BY COUNT(*)", "not allowed in GROUP BY"),
        ]
        for sql, message in cases:
            with self.subTest(sql=sql), self.assertRaisesRegex(ExecutionError, message):
                self.db.execute(sql)

    def test_grouping_errors_detected_on_empty_table(self) -> None:
        self.db.execute("CREATE TABLE empty (a INT, b INT)")
        with self.assertRaises(ExecutionError):
            self.db.execute("SELECT a, COUNT(*) FROM empty GROUP BY b")


# --- Stage 3: ORDER BY / LIMIT ----------------------------------------------------------

class OrderLimitTests(FixtureMixin):

    def names(self, sql: str) -> list[object]:
        return [row[0] for row in self.rows(sql)]

    def test_order_desc_nulls_last(self) -> None:
        self.assertEqual(self.names("SELECT name FROM users ORDER BY age DESC"),
                         ["Mehmet", "Ali", "Zeynep", "Ayse", "Can"])
        self.assertEqual(self.names("SELECT name FROM users ORDER BY age ASC")[0], "Can")

    def test_multi_key_mixed_direction(self) -> None:
        self.assertEqual(self.names("SELECT name FROM users ORDER BY city, age DESC"),
                         ["Ayse", "Can", "Mehmet", "Ali", "Zeynep"])

    def test_order_by_position_and_alias(self) -> None:
        self.assertEqual(self.rows("SELECT name, age FROM users WHERE age > 0 ORDER BY 2 LIMIT 1"), [("Ayse", 25)])
        self.assertEqual(self.names("SELECT name AS person FROM users ORDER BY person DESC LIMIT 1"), ["Zeynep"])

    def test_limit(self) -> None:
        self.assertEqual(len(self.rows("SELECT * FROM users LIMIT 2")), 2)
        self.assertEqual(self.rows("SELECT * FROM users LIMIT 0"), [])
        self.assertEqual(len(self.rows("SELECT * FROM users LIMIT 100")), 5)

    def test_order_errors(self) -> None:
        with self.assertRaisesRegex(ExecutionError, "out of range"):
            self.db.execute("SELECT name FROM users ORDER BY 3")
        self.db.execute("CREATE TABLE mix (v)")
        self.db.execute("INSERT INTO mix VALUES (1), ('a')")
        with self.assertRaisesRegex(ExecutionError, "mixed types"):
            self.db.execute("SELECT v FROM mix ORDER BY v")


# --- Extensions: arithmetic, IS NULL, DISTINCT, OFFSET, LEFT JOIN, UPDATE, DELETE -------

class ExtensionTests(FixtureMixin):

    def test_arithmetic_precedence_and_types(self) -> None:
        self.assertEqual(self.rows("SELECT 1 + 2 * 3, (1 + 2) * 3, 7 / 2, -7 / 2, 7.0 / 2, 10 - 4 - 3 "
                                   "FROM users LIMIT 1"), [(7, 9, 3, -3, 3.5, 3)])
        self.assertEqual(self.rows("SELECT name, age * 2 AS doubled FROM users WHERE age + 5 >= 35 ORDER BY doubled"),
                         [("Ali", 60), ("Mehmet", 70)])
        self.assertEqual(self.rows("SELECT age + 1 FROM users WHERE id = 5"), [(None,)])

    def test_arithmetic_errors(self) -> None:
        with self.assertRaisesRegex(ExecutionError, "Division by zero"):
            self.db.execute("SELECT age / 0 FROM users")
        with self.assertRaisesRegex(ExecutionError, "Arithmetic"):
            self.db.execute("SELECT name + 1 FROM users")

    def test_aggregate_arithmetic_with_group_by(self) -> None:
        rows = self.rows("SELECT u.name, SUM(o.amount) / COUNT(*) AS avg_amount FROM users u "
                         "JOIN orders o ON u.id = o.user_id GROUP BY u.name HAVING SUM(o.amount) * 2 > 1000 "
                         "ORDER BY avg_amount DESC")
        self.assertEqual(rows, [("Ayse", 800.0), ("Ali", 762.75)])

    def test_is_null(self) -> None:
        self.assertEqual(self.rows("SELECT name FROM users WHERE age IS NULL"), [("Can",)])
        self.assertEqual(len(self.rows("SELECT name FROM users WHERE age IS NOT NULL")), 4)
        self.assertEqual(len(self.rows("SELECT name FROM users WHERE NOT age IS NULL")), 4)

    def test_distinct_offset(self) -> None:
        self.assertEqual(self.rows("SELECT DISTINCT city FROM users ORDER BY city"),
                         [("Ankara",), ("Istanbul",), ("Izmir",)])
        self.assertEqual(self.rows("SELECT id FROM users ORDER BY id LIMIT 2 OFFSET 1"), [(2,), (3,)])
        self.assertEqual(self.rows("SELECT id FROM users ORDER BY id OFFSET 3"), [(4,), (5,)])

    def test_left_join(self) -> None:
        rows = self.rows("SELECT u.name, o.product FROM users u LEFT JOIN orders o ON u.id = o.user_id "
                         "ORDER BY u.id, o.id")
        self.assertEqual(rows, [("Ali", "Laptop"), ("Ali", "Mouse"), ("Ayse", "Phone"), ("Mehmet", "Monitor"),
                                ("Mehmet", "Keyboard"), ("Zeynep", None), ("Can", None)])
        orphans = self.rows("SELECT u.name FROM users u LEFT OUTER JOIN orders o "
                            "ON u.id = o.user_id AND o.amount > 1000 WHERE o.id IS NULL ORDER BY u.id")
        self.assertEqual(orphans, [("Ayse",), ("Mehmet",), ("Zeynep",), ("Can",)])
        counts = self.rows("SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id "
                           "GROUP BY u.name ORDER BY 2 DESC, 1")
        self.assertEqual(counts[-2:], [("Can", 0), ("Zeynep", 0)])

    def test_insert_with_column_list(self) -> None:
        self.db.execute("INSERT INTO users (name, id) VALUES ('Ece', 6), ('Ege', 7)")
        self.assertEqual(self.rows("SELECT id, name, age, city FROM users WHERE id > 5 ORDER BY id"),
                         [(6, "Ece", None, None), (7, "Ege", None, None)])
        for sql, message in [("INSERT INTO users (id, missing) VALUES (1, 2)", "Unknown column in table"),
                             ("INSERT INTO users (id, id) VALUES (1, 2)", "Duplicate column in INSERT"),
                             ("INSERT INTO users (id, name) VALUES (1)", "2 columns specified")]:
            with self.subTest(sql=sql), self.assertRaisesRegex(ExecutionError, message):
                self.db.execute(sql)

    def test_update(self) -> None:
        result = self.db.execute("UPDATE users SET age = age + 1, city = 'Bursa' WHERE city = 'Istanbul'")
        self.assertEqual(result.message, "2 rows updated")
        self.assertEqual(self.rows("SELECT name, age FROM users WHERE city = 'Bursa' ORDER BY id"),
                         [("Ali", 31), ("Mehmet", 36)])
        self.assertEqual(self.db.execute("UPDATE orders SET amount = 1").message, "6 rows updated")
        self.assertEqual(self.rows("SELECT DISTINCT amount FROM orders"), [(1.0,)])

    def test_update_is_atomic_and_validated(self) -> None:
        with self.assertRaisesRegex(ExecutionError, "INTEGER"):
            self.db.execute("UPDATE users SET age = 'x' WHERE id = 5")
        with self.assertRaises(ExecutionError):
            self.db.execute("UPDATE users SET age = 100 / (id - 3)")  # division by zero at id=3
        self.assertEqual(self.rows("SELECT SUM(age) FROM users"), [(118,)])
        for sql in ["UPDATE users SET missing = 1", "UPDATE users SET age = 1, age = 2",
                    "UPDATE users SET age = COUNT(*)", "UPDATE missing SET a = 1"]:
            with self.subTest(sql=sql), self.assertRaises(ExecutionError):
                self.db.execute(sql)

    def test_delete(self) -> None:
        self.assertEqual(self.db.execute("DELETE FROM orders WHERE amount < 100").message, "3 rows deleted")
        self.assertEqual(self.rows("SELECT id FROM orders ORDER BY id"), [(1,), (3,), (4,)])
        self.assertEqual(self.db.execute("DELETE FROM orders").message, "3 rows deleted")
        self.assertEqual(self.rows("SELECT COUNT(*) FROM orders"), [(0,)])
        with self.assertRaisesRegex(ExecutionError, "Unknown column"):
            self.db.execute("DELETE FROM users WHERE missing = 1")


# --- Error resilience -------------------------------------------------------------------

class ErrorHandlingTests(FixtureMixin):

    def test_every_error_is_sqlerror_and_db_survives(self) -> None:
        broken = ["SELECT # FROM users", "SELECT name FROM", "SELECT nope FROM users",
                  "SELECT * FROM users WHERE age > 'x'", "", "   ;  "]
        for sql in broken:
            with self.subTest(sql=sql), self.assertRaises(SQLError):
                self.db.execute(sql)
        self.assertEqual(self.rows("SELECT COUNT(*) FROM users"), [(5,)])

    def test_script_with_syntax_error_runs_nothing(self) -> None:
        with self.assertRaises(ParserError):
            self.db.execute_script("INSERT INTO users VALUES (9, 'X', 1, 'Y'); SELEC * FROM users")
        self.assertEqual(len(self.db.table("users").rows), 5)

    def test_specific_exception_types(self) -> None:
        self.assertRaises(LexerError, self.db.execute, "SELECT 'open FROM users")
        self.assertRaises(ParserError, self.db.execute, "SELECT name users")
        self.assertRaises(ExecutionError, self.db.execute, "SELECT x.name FROM users")


# --- Stage 4: Formatter & REPL ----------------------------------------------------------

class FormatterTests(unittest.TestCase):

    def test_grid_layout_and_alignment(self) -> None:
        text = TabularFormatter().format(("id", "name", "ok"), [(1, "Ali", True), (10, None, False)])
        self.assertEqual(text, "\n".join([
            "+----+------+-------+",
            "| id | name | ok    |",
            "+----+------+-------+",
            "|  1 | Ali  | true  |",
            "| 10 | NULL | false |",
            "+----+------+-------+",
        ]))

    def test_floats_and_empty_result(self) -> None:
        formatter = TabularFormatter()
        self.assertEqual(formatter.format_value(1 / 3), "0.333333")
        self.assertEqual(formatter.format_value(2.0), "2.0")
        self.assertEqual(formatter.format(("a",), []), "+---+\n| a |\n+---+\n+---+")


class ReplTests(unittest.TestCase):

    def run_repl(self, lines: list[str]) -> str:
        feed: Iterator[str] = iter(lines)

        def fake_input(prompt: str) -> str:
            try:
                return next(feed)
            except StopIteration:
                raise EOFError from None

        output = io.StringIO()
        Repl(input_func=fake_input, output=output).run()
        return output.getvalue()

    def test_session(self) -> None:
        text = self.run_repl([
            ".tables",
            "CREATE TABLE t (a INT, b TEXT);",
            "INSERT INTO t VALUES (1, 'x'),",
            "  (2, 'y');",
            "SELECT * FROM t ORDER BY a DESC;",
            ".tables",
            ".schema t",
            ".schema missing",
            "SELECT * FROM missing;",
            "SELECT @;",
            ".unknown",
            ".exit",
            "SELECT 'never reached';",
        ])
        self.assertIn("(no tables)", text)
        self.assertIn("Table 't' created", text)
        self.assertIn("2 rows inserted", text)
        self.assertIn("| 2 | y |\n| 1 | x |", text)
        self.assertIn("(2 rows)", text)
        self.assertIn("CREATE TABLE t (a INTEGER, b TEXT);", text)
        self.assertIn("Table not found: 'missing'", text)
        self.assertIn("ExecutionError:", text)
        self.assertIn("LexerError:", text)
        self.assertIn("Unknown meta command", text)
        self.assertNotIn("never reached", text)

    def test_eof_exits_cleanly(self) -> None:
        self.assertIn("PySQL-Engine", self.run_repl([]))


if __name__ == "__main__":
    unittest.main()
