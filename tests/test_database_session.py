from app.database.session import normalize_database_url


def test_postgresql_url_uses_psycopg_driver():
    assert normalize_database_url("postgresql://user:pass@localhost:5432/dbname").startswith("postgresql+psycopg://")
