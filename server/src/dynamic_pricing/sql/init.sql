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
    product_type VARCHAR(255) NOT NULL,
    product_uuid VARCHAR(255) NOT NULL,
    batch_id VARCHAR(255) NOT NULL,
    price FLOAT NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    id INTEGER NOT NULL,
    scraped_at TIMESTAMP NOT NULL,
    product_name VARCHAR(255) NOT NULL,
    batch_name VARCHAR(255) NOT NULL,
    batch_id VARCHAR(255) NOT NULL,
    batch_expiry TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS prices (
    id INTEGER NOT NULL,
    scraped_at TIMESTAMP NOT NULL,
    product_name VARCHAR(255) NOT NULL,
    batch_name VARCHAR(255) NOT NULL,
    competitor_name VARCHAR(255) NOT NULL,
    competitor_price NUMERIC NOT NULL
);


