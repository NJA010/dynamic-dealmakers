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

CREATE TABLE IF NOT EXISTS stocks (
    id INTEGER NOT NULL,
    scraped_at TIMESTAMP NOT NULL,
    stock_update_id INTEGER NOT NULL,
    batch_id INTEGER NOT NULL,
    stock_amount INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS stocks (
    id INTEGER NOT NULL,
    scraped_at TIMESTAMP NOT NULL,
    stock_update_id INTEGER NOT NULL,
    batch_id INTEGER NOT NULL,
    stock_amount INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS served_prices (
    product_type VARCHAR(255) NOT NULL,
    product_uuid VARCHAR(255) NOT NULL,
    batch_id VARCHAR(255) NOT NULL,
    price FLOAT NOT NULL
);

CREATE TABLE IF NOT EXISTS stock_changes (
    id SERIAL PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    batch_name VARCHAR(255) NOT NULL,
    batch_id VARCHAR(255) NOT NULL,
    batch_price NUMERIC NOT NULL,
    stock_difference INTEGER NOT NULL
    revenue NUMERIC NOT NULL
);

CREATE TABLE IF NOT EXISTS our_scraped_prices (
    id SERIAL PRIMARY KEY,
    scraped_at TIMESTAMP NOT NULL,
    batch_name VARCHAR(255) NOT NULL,
    batch_id VARCHAR(255) NOT NULL,
    price NUMERIC NOT NULL
    request_source VARCHAR(255) NOT NULL
    status_code INTEGER NOT NULL
);