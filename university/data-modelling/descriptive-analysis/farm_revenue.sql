WITH salary_cost AS (
  SELECT farm_id, SUM(salary) AS salary FROM Employee GROUP BY farm_id
),
equipment_cost AS (
  SELECT farm_id, SUM(price) + SUM(maintenance_cost_per_year) AS equipment FROM Equipment GROUP BY farm_id
),
fertilizer_cost AS (
  SELECT l.farm_id, SUM(fu.amount_kg * f.price_per_kg) AS fertilizer
  FROM FertilizerUsage fu
  JOIN Fertilizer f ON fu.fertilizer_id = f.fertilizer_id
  JOIN LandPlot l ON fu.plot_id = l.plot_id
  GROUP BY l.farm_id
),
crop_income AS (
  SELECT l.farm_id, SUM(lp.area * c.yield * 1000 * c.market_price_per_kg) AS income
  FROM Planting p
  JOIN LandPlot l ON p.plot_id = l.plot_id
  JOIN Crop c ON p.crop_id = c.crop_id
  JOIN LandPlot lp ON p.plot_id = lp.plot_id
  GROUP BY l.farm_id
)

SELECT
  f.farm_id,
  f.name,
  COALESCE(crop_income.income, 0) AS total_income,
  COALESCE(salary_cost.salary, 0) + COALESCE(equipment_cost.equipment, 0) + COALESCE(fertilizer_cost.fertilizer, 0) AS total_expenses,
  (COALESCE(crop_income.income, 0) - (COALESCE(salary_cost.salary, 0) + COALESCE(equipment_cost.equipment, 0) + COALESCE(fertilizer_cost.fertilizer, 0))) AS net_profit
FROM Farm f
LEFT JOIN salary_cost ON f.farm_id = salary_cost.farm_id
LEFT JOIN equipment_cost ON f.farm_id = equipment_cost.farm_id
LEFT JOIN fertilizer_cost ON f.farm_id = fertilizer_cost.farm_id
LEFT JOIN crop_income ON f.farm_id = crop_income.farm_id
ORDER BY net_profit DESC;