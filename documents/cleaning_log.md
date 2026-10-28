# Cleaning Log (Phase 3)

Input: raw tables in MySQL (`airbnb_project`, all columns stored as text).
Output: files in `data/processed/` (CSV, Parquet) and the table `calendar_clean` in MySQL.
Principle: clean each source separately first, then aggregate cities, then merge. Rows are flagged, not deleted, unless stated.

## Summary

| Source | Notebook | Raw rows | Clean rows | Output |
|---|---|---|---|---|
| Listings | `09_cleaning_listing.ipynb` | 41,183 | 41,183 | `listings_clean.csv` |
| Calendar | `10_cleaning_calendar.ipynb` | 15,032,960 | 15,032,960 | `calendar_clean` (MySQL), `calendar_daily_city.csv`, `calendar_listing_summary.csv` |
| Weather | `11_cleaning_weather.ipynb` | 1,472 | 1,472 | `weather_clean.csv/.parquet` |
| World Bank | `12_cleaning_worldbank.ipynb` | 96 (long) | 32 (wide) | `economy_clean.csv/.parquet` |
| Landmarks | `13_cleaning_landmarks.ipynb` | 1,640 | 1,589 | `landmarks_clean.csv/.parquet` |

## 1. Listings (`09_cleaning_listing.ipynb`)

- Input: `raw_listings`, 41,183 rows x 91 columns (Amsterdam 10,612; Barcelona 15,236; Berlin 12,376; Columbus 2,959).
- Empty strings were converted to NaN.
- Dropped columns (91 -> 64): columns that are 100% empty, plus URLs, host picture/about, `description`, `neighborhood_overview`, `scrape_id`, `source`, `host_profile_id`, and the `price_quote_*` columns (checkin/checkout/total/raw). `price_quote_price_per_night` was also dropped after checking it adds no price that `price` does not have (0 rows with price missing but quote available).
- Type conversion:
  - `price`: `$1,200.00` -> 1200.0 (float).
  - About 40 numeric columns converted with `to_numeric`.
  - `t/f` columns (`host_is_superhost`, `host_has_profile_pic`, `host_identity_verified`, `has_availability`) -> boolean.
  - Date columns (`last_scraped`, `calendar_last_scraped`, `first_review`, `last_review`) -> datetime.
  - Text categories trimmed (`room_type`, `property_type`, `neighbourhood_cleansed`, `neighbourhood_group_cleansed`, `city`).
- Derived columns:
  - `bathrooms` (number extracted from `bathrooms_text`, "half" = 0.5) and `bathroom_shared`.
  - `price_missing`, `has_reviews`, `has_license` (then `license` dropped), `host_tenure_years` (years + months/12).
- Duplicates (city, id): 0 removed. Prices <= 0: 0.
- Missing price (not imputed, flagged with `price_missing`):

| City | Rows | With price | Missing price |
|---|---|---|---|
| Amsterdam | 10,612 | 6,039 | 43.1% |
| Barcelona | 15,236 | 12,610 | 17.2% |
| Berlin | 12,376 | 8,157 | 34.1% |
| Columbus | 2,959 | 2,634 | 11.0% |

  Reason: price is the target variable; filling it would invent values. Missing price is also more common for listings with `has_availability = t` (29%) than `f` (10%).
- Outliers (flagged, not removed):
  - Price: IQR rule (above Q3 + 1.5 x IQR), per city: Amsterdam 333, Barcelona 635, Berlin 434, Columbus 146.
  - `minimum_nights` > 365: Amsterdam 1, Barcelona 4, Berlin 10, Columbus 0.
  - Location more than 0.5 degrees from the city center: 0.
- Checked and kept: `bathrooms` = 0 in 816 rows (487 "0 baths", 329 "0 shared baths"); `bathrooms` >= 6 in 153 rows (average 13 guests, mostly entire homes, plausible); `host_tenure_years` = 0 in 259 rows.
- Output: `listings_clean.csv`, 41,183 rows x 71 columns.

## 2. Calendar (`10_cleaning_calendar.ipynb`)

- Input: `raw_calendar`, 15,032,960 rows. Cleaning was done inside MySQL (too big for pandas) into the table `calendar_clean`.
- Conversions: `date` -> DATE, `available` t/f -> 1/0, `minimum_nights` and `maximum_nights` -> unsigned integers (non-numeric -> NULL).
- Orphans (calendar rows with no listing): 0 removed in every city.
- Duplicates (city, listing_id, date): none.
- Indexes added on (city, listing_id) and (date).
- The calendar has no daily price column, so seasonality is measured with availability.
- Aggregates saved:
  - `calendar_daily_city.csv`: one row per city and date (n_listings, n_available, share_available), 1,463 rows.
  - `calendar_listing_summary.csv`: one row per listing (n_days, days_available, share_available, min/max of minimum_nights), 41,183 rows.
- Note: calendar date ranges differ by city. Compare cities only on the common period 2026-09-25 to 2027-09-16.

## 3. Weather (`11_cleaning_weather.ipynb`)

- Input: `raw_weather`, 1,472 rows (4 cities x 368 days).
- `date` -> datetime; temperature and rain -> numeric.
- Missing values: 0. Duplicates (city, date): 0. Impossible rows (min > max, negative rain): 0.
- Rule kept in code: temperature gaps up to 2 days are interpolated per city; rain is never filled.
- New columns: `temp_mean`, `calendar_date` = `date` + 1 year. Dropped: `note`.
- The weather is from one year earlier (2025-09-17 to 2026-09-19), used as reference weather. Join with the calendar on city + `calendar_date`.
- Output: `weather_clean.csv/.parquet`, 1,472 rows.

## 4. World Bank (`12_cleaning_worldbank.ipynb`)

- Input: `raw_economic_indicators`, 96 rows (4 countries x 3 indicators x 8 years). Duplicates: 0. Missing values: 21.
- Pivoted from long to wide (one row per country and year, 32 rows).
- Share missing: GDP per capita 0%, inflation 3% (USA 2025), international tourism arrivals 62% (only 3 of 8 years per country).
- Decision: dropped `international_tourism_arrivals` (more than 50% missing). No imputation for the rest.
- Added `city` (NLD -> Amsterdam, ESP -> Barcelona, DEU -> Berlin, USA -> Columbus).
- Output: `economy_clean.csv/.parquet`, 32 rows: city, country_code, country, year, gdp_per_capita_usd, inflation_pct.
- For the merge: use one fixed year per city (2025, the latest available). It differs between cities, not between listings.

## 5. Landmarks (`13_cleaning_landmarks.ipynb`)

- Input: `raw_landmarks`, 1,640 rows.
- Coordinates and distance -> numeric; names trimmed.
- Removed: rows without name or coordinates 0; duplicates (city, name) 0; distance outside 0-10,000 m 0.
- Collection was redone (notebook `08`): the first version took only the 500 nearest Wikipedia pages to the city center. In Berlin they all fell within 2.5 km of the center, so listings farther out looked far from any landmark (median distance to the nearest landmark 1.87 km, 0 landmarks within 1 km). New method: 9 search points (center + 8 points 5 km away), 10 km radius each, duplicates removed, only landmarks within 10 km of the center kept.
- Input: `raw_landmarks`, ?? rows.
- Extra exclusion by name: pages containing "hotel", "company", "manufacturing", "government" -> ?? rows removed.
- Output: `landmarks_clean.csv/.parquet`, 2443 rows.

## 6. Next steps ## 7. Notes for ML
- Do not use `estimated_revenue_l365d` or other price-derived columns (`price_missing`, `price_outlier`) as features when predicting price: they leak the target.
- Weather and economy columns are city-level, so they separate cities and days, not listings in the same city.
- Price is missing for 29% of listings (not imputed): training uses listings with a price only.

## 8. Next step
EDA (Phase 4) on `listings_final` and `city_daily_final`.
