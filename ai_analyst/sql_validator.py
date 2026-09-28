import re


# ============================================================
# ALLOWED PROJECT TABLES
# ============================================================

ALLOWED_TABLES = {
    "streaming.transaction_events",
    "streaming.transaction_risk_events",
    "streaming.event_queue",
}


# ============================================================
# FORBIDDEN SQL OPERATIONS
# ============================================================

FORBIDDEN_KEYWORDS = [
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "GRANT",
    "REVOKE",
]


# ============================================================
# SQL VALIDATOR
# ============================================================

def validate_sql(sql):
    """
    Validate SQL before execution.

    Only SELECT queries against approved project tables
    are allowed.

    Returns:
        True if SQL passes validation.

    Raises:
        ValueError if SQL is unsafe or malformed.
    """

    if not sql:
        raise ValueError(
            "Generated SQL is empty."
        )

    sql = sql.strip()

    # --------------------------------------------------
    # 1. Only SELECT statements are allowed
    # --------------------------------------------------

    if not sql.upper().startswith("SELECT"):

        raise ValueError(
            "Only SELECT statements are allowed."
        )

    # --------------------------------------------------
    # 2. Remove trailing semicolon
    # --------------------------------------------------

    sql_without_semicolon = (
        sql.rstrip(";").strip()
    )

    # --------------------------------------------------
    # 3. Prevent multiple SQL statements
    # --------------------------------------------------

    if ";" in sql_without_semicolon:

        raise ValueError(
            "Multiple SQL statements are not allowed."
        )

    # --------------------------------------------------
    # 4. Block dangerous SQL keywords
    # --------------------------------------------------

    upper_sql = sql.upper()

    for keyword in FORBIDDEN_KEYWORDS:

        pattern = rf"\b{re.escape(keyword)}\b"

        if re.search(
            pattern,
            upper_sql
        ):

            raise ValueError(
                f"Forbidden SQL keyword detected: {keyword}"
            )

    # --------------------------------------------------
    # 5. Require an approved project table
    # --------------------------------------------------

    normalized_sql = re.sub(
        r"\s+",
        " ",
        sql.lower()
    )

    known_table_found = any(
        table in normalized_sql
        for table in ALLOWED_TABLES
    )

    if not known_table_found:

        raise ValueError(
            "Generated SQL does not reference "
            "an allowed project table."
        )

    # --------------------------------------------------
    # 6. Block database system/catalog access
    # --------------------------------------------------

    forbidden_sources = [
        "information_schema",
        "pg_catalog",
        "pg_tables",
        "pg_class",
        "pg_attribute",
    ]

    for source in forbidden_sources:

        if source in normalized_sql:

            raise ValueError(
                "Access to database system metadata "
                f"is not allowed: {source}"
            )

    # --------------------------------------------------
    # 7. Return successful validation
    # --------------------------------------------------

    return True