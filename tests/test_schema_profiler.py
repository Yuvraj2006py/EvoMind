import pandas as pd

from evomind.core.schema_profiler import profile_dataset


def test_profile_dataset_detects_sales_keyword_target():
    df = pd.DataFrame(
        {
            "store_id": [101, 102, 101, 103, 102],
            "promo_flag": [0, 1, 1, 0, 1],
            "week": [1, 1, 2, 2, 3],
            "sales_amount": [120.0, 230.5, 185.2, 199.4, 260.1],
        }
    )

    schema = profile_dataset(df)

    assert schema["target"] == "sales_amount"


def test_profile_dataset_detects_churn_keyword_target():
    df = pd.DataFrame(
        {
            "customer_id": ["A001", "A002", "A003", "A004", "A005"],
            "avg_minutes": [35.5, 12.1, 18.4, 27.9, 9.3],
            "support_tickets": [2, 0, 1, 3, 4],
            "churn_flag": ["yes", "no", "no", "yes", "yes"],
        }
    )

    schema = profile_dataset(df)

    assert schema["target"] == "churn_flag"
