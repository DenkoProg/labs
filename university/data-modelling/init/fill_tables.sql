-- Заповнення таблиці Farm
INSERT INTO Farm (name, address, manager, phone, license)
VALUES
('Green Valley Farm', '123 Greenway, Springfield', 'John Doe', '+380501234567', 'LV12345'),
('Sunny Acres', '456 Sunshine Rd, Sunnyvale', 'Jane Smith', '+380671112233', 'LV23456'),
('Oakwood Farm', '789 Oak St, Forestville', 'Alice Johnson', '+380931234567', 'LV34567'),
('Riverside Fields', '321 River Rd, Riverside', 'Michael Brown', '+380661122334', 'LV45678'),
('Golden Harvest', '654 Harvest Blvd, Farmtown', 'Emily Davis', '+380991223344', 'LV56789');

-- Заповнення таблиці LandPlot
INSERT INTO LandPlot (farm_id, area, soil_type, coordinates, status, rent_cost_per_hectare)
VALUES
(1, 15.5, 'Chernozem', NULL, 'active', 1200.00),
(1, 10.0, 'Sandy', NULL, 'inactive', 900.00),
(2, 20.0, 'Clay', NULL, 'active', 1500.00),
(2, 8.5, 'Loam', NULL, 'active', 1100.00),
(3, 25.0, 'Peaty', NULL, 'inactive', 800.00),
(4, 30.0, 'Chernozem', NULL, 'active', 1600.00),
(5, 18.0, 'Loamy sand', NULL, 'active', 1300.00);

-- Заповнення таблиці Crop
INSERT INTO Crop (name, yield, vegetation_period_days, seed_cost_per_hectare, market_price_per_kg)
VALUES
('Wheat', 6.5, 120, 100.00, 0.30),
('Corn', 8.2, 100, 120.00, 0.25),
('Sunflower', 3.5, 90, 90.00, 0.50),
('Barley', 5.0, 80, 95.00, 0.28),
('Potato', 25.0, 110, 150.00, 0.40);

-- Заповнення таблиці Equipment
INSERT INTO Equipment (farm_id, type, brand, production_year, power, price, maintenance_cost_per_year)
VALUES
(1, 'Tractor', 'John Deere', 2018, 250.00, 80000.00, 5000.00),
(2, 'Harvester', 'Claas', 2015, 400.00, 150000.00, 10000.00),
(3, 'Seeder', 'New Holland', 2020, 100.00, 30000.00, 2000.00),
(4, 'Plough', 'Case IH', 2017, 80.00, 25000.00, 1500.00),
(5, 'Sprayer', 'Amazone', 2019, 120.00, 40000.00, 2500.00);

-- Заповнення таблиці Fertilizer
INSERT INTO Fertilizer (name, type, expiration_date, price_per_kg)
VALUES
('Nitrogen Fertilizer', 'Nitrogen', '2026-12-31', 1.50),
('Phosphate Fertilizer', 'Phosphate', '2025-06-30', 1.80),
('Potassium Fertilizer', 'Potassium', '2027-03-15', 2.00),
('Organic Fertilizer', 'Organic', '2025-09-01', 1.20);

-- Заповнення таблиці Employee
INSERT INTO Employee (farm_id, first_name, last_name, position, hire_date, employment_type, salary, bonus)
VALUES
(1, 'Ivan', 'Petrenko', 'Agronomist', '2020-04-15', 'full-time', 15000.00, 3000.00),
(1, 'Oksana', 'Shevchenko', 'Machine Operator', '2019-05-10', 'full-time', 12000.00, 2000.00),
(2, 'Andriy', 'Kovalchuk', 'Manager', '2018-03-12', 'full-time', 18000.00, 5000.00),
(2, 'Natalia', 'Boyko', 'Technician', '2021-07-01', 'part-time', 8000.00, 1000.00),
(3, 'Oleh', 'Tkachenko', 'Driver', '2022-01-20', 'full-time', 10000.00, 1500.00),
(4, 'Olena', 'Bondarenko', 'Accountant', '2020-09-05', 'full-time', 16000.00, 2500.00),
(5, 'Maksym', 'Melnyk', 'Field Worker', '2023-02-28', 'seasonal', 7000.00, 0.00);

-- Заповнення таблиці Planting
INSERT INTO Planting (plot_id, crop_id, planting_date, harvest_date)
VALUES
(1, 1, '2023-03-15', '2023-07-20'),
(2, 2, '2023-04-10', '2023-08-15'),
(3, 3, '2023-05-01', '2023-08-10'),
(4, 4, '2023-03-25', '2023-07-10'),
(5, 5, '2023-04-05', '2023-08-30'),
(6, 1, '2023-03-20', '2023-07-25'),
(7, 2, '2023-04-15', '2023-08-20');

-- Заповнення таблиці FertilizerUsage
INSERT INTO FertilizerUsage (plot_id, fertilizer_id, date, amount_kg)
VALUES
(1, 1, '2023-03-20', 200.00),
(2, 2, '2023-04-15', 150.00),
(3, 3, '2023-05-05', 180.00),
(4, 4, '2023-03-30', 100.00),
(5, 1, '2023-04-10', 220.00),
(6, 2, '2023-04-25', 170.00),
(7, 3, '2023-05-10', 190.00);