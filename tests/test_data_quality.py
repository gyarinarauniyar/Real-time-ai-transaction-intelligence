import pandas as pd

from src.data_quality.validate_transactions import (
    check_schema,
    check_missing_values,
    check_duplicates,
    check_amounts,
    check_currency,
    check_timestamps,
    check_categories,
)


INPUT_PATH = "data/synthetic/transactions.csv"


def load_test_data():
    return pd.read_csv(INPUT_PATH)


def test_schema_passes():

    df = load_test_data()

    result = check_schema(df)

    assert result["status"] == "PASS"


def test_missing_values_detected():

    df = load_test_data()

    df.loc[0, "amount"] = None

    result = check_missing_values(df)

    assert result["status"] == "FAIL"


def test_duplicate_transaction_detected():

    df = load_test_data()

    df.loc[1, "transaction_id"] = df.loc[0, "transaction_id"]

    result = check_duplicates(df)

    assert result["status"] == "FAIL"


def test_negative_amount_detected():

    df = load_test_data()

    df.loc[0, "amount"] = -100

    result = check_amounts(df)

    assert result["status"] == "FAIL"


def test_zero_amount_detected():

    df = load_test_data()

    df.loc[0, "amount"] = 0

    result = check_amounts(df)

    assert result["status"] == "FAIL"


def test_invalid_currency_detected():

    df = load_test_data()

    df.loc[0, "currency"] = "USD"

    result = check_currency(df)

    assert result["status"] == "FAIL"


def test_invalid_timestamp_detected():

    df = load_test_data()

    df.loc[0, "transaction_timestamp"] = "NOT_A_TIMESTAMP"

    result = check_timestamps(df)

    assert result["status"] == "FAIL"


def test_invalid_payment_method_detected():

    df = load_test_data()

    df.loc[0, "payment_method"] = "cash"

    result = check_categories(df)

    assert result["status"] == "FAIL"


def test_invalid_device_detected():

    df = load_test_data()

    df.loc[0, "device_type"] = "unknown_device"

    result = check_categories(df)

    assert result["status"] == "FAIL"