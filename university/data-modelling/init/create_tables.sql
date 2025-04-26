-- Оновлення таблиці Farm
CREATE TABLE Farm (
    farm_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    address VARCHAR(150),
    manager VARCHAR(100),
    phone VARCHAR(20),
    license VARCHAR(50)
);

-- Оновлення таблиці LandPlot
CREATE TABLE LandPlot (
    plot_id SERIAL PRIMARY KEY,
    farm_id INT NOT NULL REFERENCES Farm(farm_id),
    area DECIMAL(10,2) CHECK(area > 0),
    soil_type VARCHAR(50),
    coordinates GEOMETRY(POLYGON, 4326),
    status VARCHAR(50),
    rent_cost_per_hectare DECIMAL(10,2) -- Нове поле: орендна плата за гектар
);

-- Оновлення таблиці Crop
CREATE TABLE Crop (
    crop_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    yield DECIMAL(10,2) CHECK(yield >= 0),
    vegetation_period_days INT CHECK(vegetation_period_days > 0),
    seed_cost_per_hectare DECIMAL(10,2),    -- Нове поле: вартість насіння на гектар
    market_price_per_kg DECIMAL(10,2)       -- Нове поле: ринкова ціна урожаю
);

-- Оновлення таблиці Equipment
CREATE TABLE Equipment (
    equipment_id SERIAL PRIMARY KEY,
    farm_id INT NOT NULL REFERENCES Farm(farm_id),
    type VARCHAR(50),
    brand VARCHAR(50),
    production_year INT CHECK(production_year >= 1950),
    power DECIMAL(10,2),
    price DECIMAL(15,2),                   -- Нове поле: вартість техніки
    maintenance_cost_per_year DECIMAL(10,2) -- Нове поле: витрати на обслуговування на рік
);

-- Оновлення таблиці Fertilizer
CREATE TABLE Fertilizer (
    fertilizer_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    type VARCHAR(50),
    expiration_date DATE,
    price_per_kg DECIMAL(10,2) -- Нове поле: вартість 1 кг добрива
);

-- Оновлення таблиці Employee
CREATE TABLE Employee (
    employee_id SERIAL PRIMARY KEY,
    farm_id INT NOT NULL REFERENCES Farm(farm_id),
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    position VARCHAR(50),
    hire_date DATE,
    employment_type VARCHAR(50),
    salary DECIMAL(10,2), -- Нове поле: зарплата
    bonus DECIMAL(10,2)   -- Нове поле: бонуси
);

-- Таблиця Planting
CREATE TABLE Planting (
    planting_id SERIAL PRIMARY KEY,
    plot_id INT NOT NULL REFERENCES LandPlot(plot_id),
    crop_id INT NOT NULL REFERENCES Crop(crop_id),
    planting_date DATE,
    harvest_date DATE
);

-- Таблиця FertilizerUsage
CREATE TABLE FertilizerUsage (
    usage_id SERIAL PRIMARY KEY,
    plot_id INT NOT NULL REFERENCES LandPlot(plot_id),
    fertilizer_id INT NOT NULL REFERENCES Fertilizer(fertilizer_id),
    date DATE,
    amount_kg DECIMAL(10,2) CHECK(amount_kg > 0)
);