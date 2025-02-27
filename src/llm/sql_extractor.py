import re


class SQLQueryExtractor:
    """Extracts SQL queries from model responses."""

    @staticmethod
    def extract_sql_query(response_text: str) -> str:
        match = re.search(r"SELECT.*?;", response_text, re.DOTALL | re.IGNORECASE)
        return match.group(0).strip() if match else response_text.strip()
