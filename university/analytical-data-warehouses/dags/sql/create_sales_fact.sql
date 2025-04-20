CREATE TABLE IF NOT EXISTS sales_fact (
    "Fact_ID" INTEGER PRIMARY KEY,
    "Order ID" TEXT,
    "Date_ID" INTEGER,
    "Customer ID" TEXT,
    "Product ID" TEXT,
    "ShipMode_ID" INTEGER,
    "Sales" NUMERIC,
    "Quantity" INTEGER,
    "Discount" NUMERIC,
    "Profit" NUMERIC
);