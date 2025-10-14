import sqlite3
import json
from typing import Dict, Any, List, Optional
import ollama
from functools import wraps
import re
from contextlib import contextmanager
from textwrap import dedent # for multi-line string literals

class DatabaseError(Exception):
    """Custom exception for database operations"""
    pass


def _create_sample_data(cursor):  # Helper function to create sample data
    """Create sample data for tables"""
    sample_data = {
        'example': [
            ('Example 1', 10.5),
            ('Example 2', 25.0)
        ],
        'users': [
            ('Bob', 'bob@example.com'),
            ('Susan', 'susan@test.net')
        ],
        'products': [
            ('Laptop', 1200.00),
            ('Keyboard', 75.50)
        ]
    }

    for table, data in sample_data.items():
        for record in data:
            if table == 'example':
                cursor.execute(
                    "INSERT INTO example (name, value) VALUES (?, ?) ON CONFLICT DO NOTHING",
                    record
                )
            elif table == 'users':
                cursor.execute(
                    "INSERT INTO users (name, email) VALUES (?, ?) ON CONFLICT DO NOTHING",
                    record
                )
            elif table == 'products':
                cursor.execute(
                    "INSERT INTO products (product_name, price) VALUES (?, ?) ON CONFLICT DO NOTHING",
                    record
                )


class SQLiteTool:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super(SQLiteTool, cls).__new__(cls)
        return cls._instance

    def __init__(self, default_db: str = "test.db"):
        if hasattr(self, 'default_db'):  # Skip initialization if already done
            return
        self.default_db = default_db
        self._initialize_database()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.default_db)
        try:
            yield conn
        finally:
            conn.close()

    def _initialize_database(self):
        """Initialize database with tables"""
        tables = {
            'example': """
                CREATE TABLE IF NOT EXISTS example (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    value REAL
                );
            """,
            'users': """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    email TEXT UNIQUE
                );
            """,
            'products': """
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY,
                    product_name TEXT,
                    price REAL
                );
            """
        }

        with self.get_connection() as conn:
            cursor = conn.cursor()
            for table_sql in tables.values():
                cursor.execute(table_sql)
            conn.commit()
            _create_sample_data(cursor)
            conn.commit()

    def get_tables(self) -> List[str]:
        """Get list of tables in the database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return [table[0] for table in cursor.fetchall()]

    def get_table_schema(self, table_name: str) -> List[tuple]:
        """Get schema for a specific table"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name});")
            return cursor.fetchall()

    def execute_query(self, query: str) -> List[tuple]:
        """Execute a SQL query and return results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query)
                return cursor.fetchall()
            except sqlite3.Error as e:
                raise DatabaseError(f"Query execution failed: {str(e)}")

class OllamaFunctionCaller:
    def __init__(self, model: str = "llama3.2:latest"):
        self.model = model
        self.sqlite_tool = SQLiteTool()
        self.function_definitions = self._get_function_definitions()

    def _get_function_definitions(self) -> Dict:
        return {
            "query_database": {
                "description": "Execute a SQL query on the database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The SQL query to execute"
                        }
                    },
                    "required": ["query"]
                }
            },
            "list_tables": {
                "description": "List all tables in the database",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }

    def _generate_prompt(self, user_input: str) -> str:
        prompt = dedent(f"""
            You are a SQL assistant. Based on the user's request, generate a JSON response that calls the appropriate function.
            Available functions: {json.dumps(self.function_definitions, indent=2)}

            User request: {user_input}

            Respond with a JSON object containing:
            - "function": The function name to call
            - "parameters": The parameters for the function

            Response:
        """).strip()
        return prompt

    def _parse_ollama_response(self, response: str) -> Dict[str, Any]:
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON found in response")
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {str(e)}")

    def process_request(self, user_input: str) -> Any:
        try:
            response = ollama.generate(model=self.model, prompt=self._generate_prompt(user_input))
            function_call = self._parse_ollama_response(response.response)

            if function_call["function"] == "query_database":
                return self.sqlite_tool.execute_query(function_call["parameters"]["query"])
            elif function_call["function"] == "list_tables":
                return self.sqlite_tool.get_tables()
            else:
                raise ValueError(f"Unknown function: {function_call['function']}")
        except Exception as e:
            raise RuntimeError(f"Request processing failed: {str(e)}")

def main():
    function_caller = OllamaFunctionCaller()
    queries = [
        "Show me all tables in the database",
        "Get all users from the users table",
        "What are the top 5 products by price?"
    ]

    for query in queries:
        try:
            print(f"\nQuery: {query}")
            result = function_caller.process_request(query)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()