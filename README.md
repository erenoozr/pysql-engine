# PySQL-Engine

An in-memory SQL query engine written from scratch in pure Python, using only the standard library (Python 3.10+, no dependencies).

It contains every stage of a small database: a lexer, a recursive descent parser that produces an AST, an execution engine with in-memory storage, and an interactive shell that prints results as tables.

## Features

- **Statements:** `CREATE TABLE`, `INSERT` (multi-row, optional column list), `UPDATE`, `DELETE`, `SELECT`
- **SELECT:** `DISTINCT`, `INNER JOIN` / `LEFT [OUTER] JOIN`, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY ... ASC|DESC` (by expression, alias or column position), `LIMIT`, `OFFSET`
- **Expressions:** `=`, `!=`/`<>`, `<`, `>`, `<=`, `>=`, `AND`, `OR`, `NOT`, `IS [NOT] NULL`, `+ - * /`, parentheses
- **Aggregates:** `COUNT(*)`, `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`
- **Column types:** `INT`, `REAL`, `TEXT`, `BOOL` (or untyped), checked on every write
- **SQL NULL semantics:** three-valued logic; `WHERE`/`HAVING`/`ON` keep only rows that evaluate to TRUE
- **Joins:** hash join for `ON a.x = b.y` conditions, nested loop for anything else
- **Atomic writes:** if one row in an `INSERT` or `UPDATE` fails, nothing is changed
- **Clear errors:** `LexerError`, `ParserError` and `ExecutionError` (all subclasses of `SQLError`), with line and column information for syntax errors

## Usage

### Interactive shell

```
$ python -m pysql
pysql> CREATE TABLE users (id INT, name TEXT, age INT);
Table 'users' created
pysql> INSERT INTO users VALUES (1, 'Ali', 30), (2, 'Ayse', 25), (3, 'Can', NULL);
3 rows inserted
pysql> SELECT name, age FROM users WHERE age IS NOT NULL ORDER BY age DESC;
+------+-----+
| name | age |
+------+-----+
| Ali  |  30 |
| Ayse |  25 |
+------+-----+
(2 rows)
```

Statements end with `;` and can span several lines.

| Meta command      | Description                                   |
|-------------------|-----------------------------------------------|
| `.tables`         | List tables                                   |
| `.schema [table]` | Show `CREATE TABLE` definitions               |
| `.help`           | Show help                                     |
| `.exit` / `.quit` | Leave the shell                               |

### As a library

```python
from pysql import Database

db = Database()
db.execute_script("""
    CREATE TABLE orders (id INT, customer TEXT, amount REAL);
    INSERT INTO orders VALUES (1, 'Ali', 1500), (2, 'Ali', 25.5), (3, 'Ayse', 800);
""")
result = db.execute(
    "SELECT customer, SUM(amount) AS total FROM orders GROUP BY customer ORDER BY total DESC"
)
print(result.columns)  # ('customer', 'total')
print(result.rows)     # (('Ali', 1525.5), ('Ayse', 800.0))
```

## Project layout

```
pysql/
  lexer.py      Tokenizer
  parser.py     Recursive descent parser -> AST
  nodes.py      AST node definitions
  engine.py     Database, Table and query execution
  formatter.py  ASCII table output
  repl.py       Interactive shell
  errors.py     Exception hierarchy
tests/
  test_pysql.py Test suite (unittest)
```

## Running the tests

```
python -m unittest -v
```

## Limitations

- No indexes, transactions or persistence: all data lives in memory.
- No subqueries, `ALTER TABLE` or `DROP TABLE`.
- With `DISTINCT`, ordering by an expression that is not in the select list may give an unspecified order.
- Table alignment assumes single-width characters, so wide CJK text can misalign columns.
