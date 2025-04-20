CREATE TABLE IF NOT EXISTS agg_sales_by_region AS
SELECT
    dc."Region",
    COUNT(DISTINCT sf."Order ID") AS orders_count,
    SUM(sf."Sales") AS total_sales,
    SUM(sf."Profit") AS total_profit
FROM sales_fact sf
JOIN dim_customer dc ON sf."Customer ID" = dc."Customer ID"
GROUP BY dc."Region";