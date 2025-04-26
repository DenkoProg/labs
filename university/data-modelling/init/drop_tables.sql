-- Спочатку видаляємо дочірні таблиці, щоб не було помилок FK
DROP TABLE IF EXISTS FertilizerUsage CASCADE;
DROP TABLE IF EXISTS Planting CASCADE;

-- Далі таблиці, які мають посилання на Farm
DROP TABLE IF EXISTS Employee CASCADE;
DROP TABLE IF EXISTS Equipment CASCADE;
DROP TABLE IF EXISTS LandPlot CASCADE;

-- Таблиці без зовнішніх посилань до цього моменту
DROP TABLE IF EXISTS Crop CASCADE;
DROP TABLE IF EXISTS Fertilizer CASCADE;

-- І вкінці — головна таблиця Farm
DROP TABLE IF EXISTS Farm CASCADE;