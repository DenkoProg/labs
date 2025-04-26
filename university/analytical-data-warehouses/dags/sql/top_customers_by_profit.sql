DROP TABLE IF EXISTS dwh.top_customers_by_profit;

CREATE TABLE dwh.top_customers_by_profit AS
SELECT
    c."Customer ID",
    c."Customer Name",
    c."Region",
    SUM(f."Profit") AS total_profit,
    SUM(f."Sales") AS total_sales,
    SUM(f."Quantity") AS total_quantity
FROM
    sales_fact f
INNER JOIN dim_customer c ON f."Customer ID" = c."Customer ID"
GROUP BY
    c."Customer ID", c."Customer Name", c."Region"
ORDER BY
    total_profit DESC
LIMIT 10;