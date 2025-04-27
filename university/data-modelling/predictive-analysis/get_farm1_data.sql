WITH salary_per_year AS (
  SELECT
    farm_id,
    EXTRACT(YEAR FROM hire_date) AS year,
    SUM(salary + COALESCE(bonus, 0)) AS total_salary
  FROM Employee
  WHERE farm_id = 1
  GROUP BY farm_id, year
),
equipment_per_year AS (
  SELECT
    farm_id,
    EXTRACT(YEAR FROM CURRENT_DATE) AS year,
    SUM(maintenance_cost_per_year) AS total_equipment_maintenance
  FROM Equipment
  WHERE farm_id = 1
  GROUP BY farm_id
),
fertilizer_per_year AS (
  SELECT
    l.farm_id,
    EXTRACT(YEAR FROM fu.date) AS year,
    SUM(fu.amount_kg * f.price_per_kg) AS total_fertilizer_cost
  FROM FertilizerUsage fu
  JOIN Fertilizer f ON fu.fertilizer_id = f.fertilizer_id
  JOIN LandPlot l ON fu.plot_id = l.plot_id
  WHERE l.farm_id = 1
  GROUP BY l.farm_id, year
)

SELECT
  COALESCE(s.farm_id, e.farm_id, f.farm_id) AS farm_id,
  COALESCE(s.year, e.year, f.year) AS year,
  COALESCE(total_salary, 0) AS total_salary,
  COALESCE(total_equipment_maintenance, 0) AS total_equipment_maintenance,
  COALESCE(total_fertilizer_cost, 0) AS total_fertilizer_cost,
  (COALESCE(total_salary, 0) + COALESCE(total_equipment_maintenance, 0) + COALESCE(total_fertilizer_cost, 0)) AS total_expenses
FROM salary_per_year s
FULL OUTER JOIN equipment_per_year e ON s.farm_id = e.farm_id
FULL OUTER JOIN fertilizer_per_year f ON COALESCE(s.farm_id, e.farm_id) = f.farm_id AND COALESCE(s.year, e.year) = f.year
ORDER BY year;