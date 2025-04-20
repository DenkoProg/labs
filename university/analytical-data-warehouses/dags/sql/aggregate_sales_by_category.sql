CREATE TABLE IF NOT EXISTS agg_sales_by_category AS
SELECT
    dp."Category",
    COUNT(DISTINCT sf."Order ID") AS orders_count,
    SUM(sf."Sales") AS total_sales,
    SUM(sf."Profit") AS total_profit
FROM sales_fact sf
JOIN dim_product dp ON sf."Product ID" = dp."Product ID"
GROUP BY dp."Category";