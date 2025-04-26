SELECT
  l.soil_type,
  ROUND(AVG(c.yield), 2) AS average_yield
FROM LandPlot l
JOIN Planting p ON l.plot_id = p.plot_id
JOIN Crop c ON p.crop_id = c.crop_id
GROUP BY l.soil_type
ORDER BY average_yield DESC;