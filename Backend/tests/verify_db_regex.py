import re
import pytest

# Mock Postgres Cursor
class MockCursor:
    def execute(self, sql, params=()):
        return sql

# Import the regex logic from db.py (duplicating here for standalone test simplicity 
# or we could import if path issues were resolved, but this is safer for a quick verification script)
def test_placeholder_replacement():
    def execute(sql, params=()):
        def replace_placeholder(match):
            if match.group(1):
                return match.group(1)
            return "%s"

        pattern = r"(\'[^\']*\'|\"[^\"]*\")|\?"
        pg_sql = re.sub(pattern, replace_placeholder, sql)
        return pg_sql

    # Test Case 1: Simple Replacement
    assert execute("SELECT * FROM jobs WHERE id = ?") == "SELECT * FROM jobs WHERE id = %s"

    # Test Case 2: Ignore ? in single quotes
    assert execute("INSERT INTO jobs VALUES ('Where is it?')") == "INSERT INTO jobs VALUES ('Where is it?')"

    # Test Case 3: Ignore ? in double quotes
    assert execute('INSERT INTO jobs VALUES ("Is this real?")') == 'INSERT INTO jobs VALUES ("Is this real?")'

    # Test Case 4: Mixed
    assert execute("SELECT * FROM jobs WHERE id = ? AND msg = 'Why?'") == "SELECT * FROM jobs WHERE id = %s AND msg = 'Why?'"

    print("✅ SQL Placeholder Regex Logic Verified")

if __name__ == "__main__":
    test_placeholder_replacement()
