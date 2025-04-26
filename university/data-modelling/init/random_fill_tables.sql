-- 1. Farm (10 записів)
INSERT INTO Farm(name, address, manager, phone, license)
SELECT
  'Farm ' || gs AS name,
  'Address ' || gs || ', Town ' || ((gs % 20) + 1) AS address,
  'Manager ' || ((gs % 50) + 1) AS manager,
  '+380' || (100000000 + (random() * 899999999)::int) AS phone,
  'LIC' || lpad((10000 + gs)::text, 5, '0') AS license
FROM generate_series(1,10) AS gs;


-- 2. LandPlot (200 записів) — farm_id від 1 до 10
INSERT INTO LandPlot(farm_id, area, soil_type, coordinates, status, rent_cost_per_hectare)
SELECT
  (1 + floor(random() * 10)::int) AS farm_id,  -- тільки 1–10
  round((5 + random() * 45)::numeric, 2)                             AS area,
  (ARRAY['Chernozem','Sandy','Clay','Loam','Peaty','Loamy sand','Silt','Alluvial'])
    [1 + floor(random() * 8)::int]                                   AS soil_type,
  NULL                                                              AS coordinates,
  (ARRAY['active','inactive','under_maintenance','planted'])
    [1 + floor(random() * 4)::int]                                   AS status,
  round((300 + random() * 2000)::numeric, 2)                         AS rent_cost_per_hectare
FROM generate_series(1,200) AS gs;


-- 3. Crop (100 записів) — немає зовнішніх ключів зверху
INSERT INTO Crop(name, yield, vegetation_period_days, seed_cost_per_hectare, market_price_per_kg)
SELECT
  'Crop ' || gs                                            AS name,
  round((1 + random() * 15)::numeric, 2)                   AS yield,
  (30 + floor(random() * 150))::int                        AS vegetation_period_days,
  round((50 + random() * 300)::numeric, 2)                  AS seed_cost_per_hectare,
  round((0.1 + random() * 2.0)::numeric, 2)                 AS market_price_per_kg
FROM generate_series(1,100) AS gs;


-- 4. Equipment (50 записів) — farm_id від 1 до 10
INSERT INTO Equipment(farm_id, type, brand, production_year, power, price, maintenance_cost_per_year)
SELECT
  (1 + floor(random() * 10)::int)                                                   AS farm_id,  -- 1–10
  (ARRAY['Tractor','Harvester','Seeder','Plough','Sprayer','Combine','Baler','Cultivator'])
    [1 + floor(random() * 8)::int]                                                   AS type,
  (ARRAY['John Deere','Claas','New Holland','Case IH','Amazone','Kubota','Massey Ferguson','Fendt'])
    [1 + floor(random() * 8)::int]                                                   AS brand,
  (1950 + floor(random() * 75))::int                                                 AS production_year,
  round((50 + random() * 950)::numeric, 2)                                           AS power,
  round((10000 + random() * 190000)::numeric, 2)                                      AS price,
  round((500 + random() * 15000)::numeric, 2)                                         AS maintenance_cost_per_year
FROM generate_series(1,50) AS gs;


-- 5. Fertilizer (50 записів) — немає звернень вгору
INSERT INTO Fertilizer(name, type, expiration_date, price_per_kg)
SELECT
  'Fertilizer ' || gs                                                                  AS name,
  (ARRAY['Nitrogen','Phosphate','Potassium','Organic','Composite','Micronutrient'])
    [1 + floor(random() * 6)::int]                                                      AS type,
  (CURRENT_DATE + (floor(random() * 1000)) * INTERVAL '1 day')::date                   AS expiration_date,
  round((0.5 + random() * 4.5)::numeric, 2)                                             AS price_per_kg
FROM generate_series(1,50) AS gs;


-- 6. Employee (200 записів) — farm_id від 1 до 10
INSERT INTO Employee(farm_id, first_name, last_name, position, hire_date, employment_type, salary, bonus)
SELECT
  (1 + floor(random() * 10)::int)                                                     AS farm_id,  -- 1–10
  (ARRAY['Ivan','Oksana','Andriy','Natalia','Oleh','Olena','Maksym','Kateryna','Taras','Svitlana',
         'Dmytro','Anastasiya','Yaroslav','Iryna','Mykola','Maria'])
    [1 + floor(random() * 16)::int]                                                    AS first_name,
  (ARRAY['Petrenko','Shevchenko','Kovalchuk','Boyko','Tkachenko','Bondarenko','Melnyk','Kovtun',
         'Zaitsev','Moroz','Lysenko','Khmelyk','Hrytsenko','Kravchenko','Yevtushenko','Holub'])
    [1 + floor(random() * 16)::int]                                                    AS last_name,
  (ARRAY['Agronomist','Machine Operator','Manager','Technician','Driver','Accountant',
         'Field Worker','Irrigation Specialist','Mechanic','Seeder Operator'])
    [1 + floor(random() * 10)::int]                                                    AS position,
  (DATE '2010-01-01' + (floor(random() * (365 * 15)))::int)                            AS hire_date,
  (ARRAY['full-time','part-time','seasonal'])[1 + floor(random() * 3)::int]             AS employment_type,
  round((5000 + random() * 20000)::numeric, 2)                                          AS salary,
  round((0 + random() * 7000)::numeric, 2)                                              AS bonus
FROM generate_series(1,200) AS gs;


-- 7. Planting (500 записів) — plot_id від 1 до 200, crop_id від 1 до 100
INSERT INTO Planting(plot_id, crop_id, planting_date, harvest_date)
SELECT
  (1 + floor(random() * 200)::int)                                                     AS plot_id,  -- 1–200
  (1 + floor(random() * 100)::int)                                                     AS crop_id,  -- 1–100
  (DATE '2020-01-01' + (floor(random() * 1500))::int)                                  AS planting_date,
  ((DATE '2020-01-01' + (floor(random() * 1500))::int)
    + (30 + floor(random() * 200)) * INTERVAL '1 day')::date                           AS harvest_date
FROM generate_series(1,500) AS gs;                                                     -- 500 рядків


-- 8. FertilizerUsage (400 записів) — plot_id від 1 до 200, fertilizer_id від 1 до 50
INSERT INTO FertilizerUsage(plot_id, fertilizer_id, date, amount_kg)
SELECT
  (1 + floor(random() * 200)::int)                                                     AS plot_id,        -- 1–200
  (1 + floor(random() * 50)::int)                                                      AS fertilizer_id,  -- 1–50
  (DATE '2020-01-01' + (floor(random() * 1500))::int)                                  AS date,
  round((50 + random() * 450)::numeric, 2)                                              AS amount_kg
FROM generate_series(1,400) AS gs;                                                     -- 400 рядків