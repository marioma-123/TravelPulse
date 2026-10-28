# Data Sources

Phase 2: Data Collection and Storage
Database: MySQL (`airbnb_project`), all raw tables stored as text exactly as received, plus a `city` column where relevant. Cleaning and type conversion happen in Phase 3.

## 1. Summary

| # | Source | Method | Table | Rows | Retrieved |
|---|---|---|---|---|---|
| 1 | Inside Airbnb (listings) | CSV download | `raw_listings` | 41,183 | 17-18 Oct 2026 |
| 2 | Inside Airbnb (calendar) | CSV download | `raw_calendar` | 15,032,960 | 17-18 Oct 2026 |
| 3 | Open-Meteo Historical Weather | REST API | `raw_weather` | 1,472 (4 x 368) | 18-Oct-2026 |
| 4 | World Bank Indicators | REST API | `raw_economic_indicators` | 96 | 18-Oct-2026 |
| 5 | Wikipedia Geosearch | REST API | `raw_landmarks` | 1,640 | 18-Oct-2026 |

## 2. Details

### 2.1 Inside Airbnb
- **URL:** https://insideairbnb.com/get-the-data/
- **Cities:** Amsterdam, Barcelona, Berlin, Columbus
- **Files:** `<City>_listings.csv`, `<City>_calendar.csv`
- **Row counts:**

| City | listings | calendar |
|---|---|---|
| Amsterdam | 10,612 | 3,873,380 |
| Barcelona | 15,236 | 5,562,302 |
| Berlin | 12,376 | 4,517,243 |
| Columbus | 2,959 | 1,080,035 |

- **Calendar date range:** Amsterdam 2026-09-17 to 2027-09-16, Barcelona 2026-09-17 to 2027-09-18, Berlin 2026-09-19 to 2027-09-19, Columbus 2026-09-25 to 2027-09-24. Common period across all cities: 2026-09-25 to 2027-09-16.
- **Notes:**
  - The calendar file has no daily price column (only `available`, `minimum_nights`, `maximum_nights`). Price comes from `raw_listings.price`.
  - Calendar ranges differ slightly by city because the files were scraped on different dates.
  - Quality checks (`SQL/03_checks.sql`): row counts match the source files, no orphan calendar listings, no duplicate listing ids.

### 2.2 Open-Meteo Historical Weather API
- **URL:** https://archive-api.open-meteo.com/v1/archive
- **Variables:** `temperature_2m_max`, `temperature_2m_min`, `precipitation_sum` (daily)
- **Location:** city-center coordinates (Amsterdam 52.3676, 4.9041; Barcelona 41.3874, 2.1686; Berlin 52.5200, 13.4050; Columbus 39.9612, -82.9988)
- **Period:** 2025-09-17 to 2026-09-19 (368 days per city)
- **Important note:** the Airbnb calendar covers future dates (Sep 2026 to Sep 2027), so no actual weather exists for them. We use the weather of the same calendar days one year earlier as a reference ("typical weather"). In Phase 3 it should be joined by month and day (or date + 1 year), not by exact date.

### 2.3 World Bank Indicators API
- **URL:** https://api.worldbank.org/v2/
- **Countries:** NLD, ESP, DEU, USA
- **Indicators:**
  - `NY.GDP.PCAP.CD` GDP per capita (current US$)
  - `ST.INT.ARVL` International tourism, number of arrivals
  - `FP.CPI.TOTL.ZG` Inflation, consumer prices (annual %)
- **Period requested:** 2018 to 2025
- **Notes:** country-level data (not city-level). Latest years are often missing (NaN) because of publication delay. International tourism arrivals is missing for most countries/years (e.g. Spain, USA after 2020);International tourism arrivals was dropped in cleaning (62% missing); see cleaning_log.md.

### 2.4 Wikipedia Geosearch API
- **URL:** https://en.wikipedia.org/w/api.php (`action=query&list=geosearch`)
- - **Method:** 9 search points per city (center + 8 points 5 km away), 10 km radius each, duplicates removed, only pages within 10 km of the center kept.
- **Filtering:** removed pages matching an exclusion keyword list (stations, streets, events, institutions, districts) and the city page itself.
- **Result:** Amsterdam 421, Barcelona 380, Berlin 428, Columbus 411.
- **Columns:** city, name, latitude, longitude, distance_from_center_m, source_url
- **Limitations:** Geosearch returns any article with coordinates, so some non-tourist pages remain (e.g. institutions, neighbourhoods). Suggested use in Phase 3: distance to the nearest landmark and number of landmarks within 1-2 km, which are robust to a little noise.

## 3. Known Limitations
1. Calendar has no daily prices, so seasonality is analyzed through availability.
2. Weather is a reference year, not actual weather for the calendar dates.
3. Economic indicators are national-level and partly incomplete.
4. Landmark list is keyword-filtered and not a curated tourism list.

## 4. How to Rebuild the Database

1. Place the 8 CSV files in `data/raw/` (not tracked by git).
2. Create a `.env` file with `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`.
3. Run `SQL/01_create_database.sql`.
4. Run `notebooks/load_airbnb.ipynb` (loads listings and calendar).
5. Run `SQL/04_indexes.sql` (once, after loading).
6. Run `SQL/03_checks.sql` and compare the results with the table in 2.1.
7. Run `notebooks/06_collect_weather.ipynb`, `07_collect_worldbank.ipynb`, `08_collect_landmarks.ipynb`.
8. Run notebooks `09` to `13` (cleaning), then `14_merge_final.ipynb` to build `listings_final` and `city_daily_final`.