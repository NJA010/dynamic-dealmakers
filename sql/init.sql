CREATE TABLE IF NOT EXISTS raw_prices (
    id SERIAL PRIMARY KEY,
    scraped_at TIMESTAMP NOT NULL,
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_products (
    id SERIAL PRIMARY KEY,
    scraped_at TIMESTAMP NOT NULL,
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_leaderboards (
    id SERIAL PRIMARY KEY,
    scraped_at TIMESTAMP NOT NULL,
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_stocks (
    id SERIAL PRIMARY KEY,
    scraped_at TIMESTAMP NOT NULL,
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS served_prices(
    product_type CHAR NOT NULL,
    product_uuid CHAR NOT NULL,
    batch_id CHAR NOT NULL,
    price FLOAT NOT NULL
)
