SELECT
  name AS crop_name,
  ROUND(AVG(yield), 2) AS average_yield
FROM Crop
GROUP BY name
ORDER BY average_yield DESC;