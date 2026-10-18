USE airbnb_project;
-- Row counts per city
SELECT 'listings' AS tbl, city, COUNT(*) AS n FROM raw_listings GROUP BY city
UNION ALL
SELECT 'calendar', city, COUNT(*) FROM raw_calendar GROUP BY city;
-- Check date range and number of unique days per city in calendar
SELECT 
    city, 
    MIN(`date`) AS first_date, 
    MAX(`date`) AS last_date, 
    COUNT(DISTINCT `date`) AS n_days 
FROM raw_calendar 
GROUP BY city;
-- Check for orphan listings (calendar records with no matching listing)
SELECT 
    c.city, 
    COUNT(DISTINCT c.listing_id) AS orphan_listings 
FROM raw_calendar c 
LEFT JOIN raw_listings l ON l.id = c.listing_id AND l.city = c.city 
WHERE l.id IS NULL 
GROUP BY c.city;

-- Check for duplicate listings in the raw_listings table
SELECT city, id, COUNT(*) AS n FROM raw_listings
GROUP BY city, id HAVING COUNT(*) > 1;
-- Check for duplicate rows in raw_calendar (same listing and date)
SELECT city, COUNT(*) AS dup_rows FROM (
  SELECT city, listing_id, `date`, COUNT(*) AS n
  FROM raw_calendar GROUP BY city, listing_id, `date` HAVING COUNT(*) > 1
) t GROUP BY city;
SHOW INDEX FROM raw_calendar;
