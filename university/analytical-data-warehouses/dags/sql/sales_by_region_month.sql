DROP TABLE IF EXISTS dwh.sales_by_region_month;

CREATE TABLE dwh.sales_by_region_month AS
SELECT
    c."Region" AS region,
    DATE_TRUNC('month', d."Order Date"::timestamp)::date AS month,
    SUM(f."Sales") AS total_sales,
    SUM(f."Quantity") AS total_quantity,
    SUM(f."Profit") AS total_profit
FROM
    sales_fact f
INNER JOIN dim_customer c ON f."Customer ID" = c."Customer ID"
INNER JOIN dim_date d ON f."Date_ID" = d."Date_ID"
GROUP BY
    region, month
ORDER BY
    month, region;