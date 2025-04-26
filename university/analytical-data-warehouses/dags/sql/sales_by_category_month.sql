DROP TABLE IF EXISTS dwh.sales_by_category_month;

CREATE TABLE dwh.sales_by_category_month AS
SELECT
    p."Category" AS category,
    DATE_TRUNC('month', d."Order Date"::timestamp)::date AS month,
    SUM(f."Sales") AS total_sales,
    SUM(f."Quantity") AS total_quantity,
    SUM(f."Profit") AS total_profit
FROM
    sales_fact f
INNER JOIN dim_product p ON f."Product ID" = p."Product ID"
INNER JOIN dim_date d ON f."Date_ID" = d."Date_ID"
GROUP BY
    category, month
ORDER BY
    month, category;