# EDA Findings (Phase 4)

Data: `listings_final` (41,183 listings) and `city_daily_final` (1,463 city-days).
Split: 80/20 by `host_id` (random_state 42): **33,300 train / 7,883 test**, no host in both. The split is saved in `data/processed/split_assignment.csv`.
All decisions below were made on the **train** set only. The test set was not used.

## Key insights

### 1. Airbnb prices differ a lot between cities
Listings with a price in train (23,921):

| City | Listings with price | Mean | Median |
|---|---|---|---|
| Amsterdam | 4,863 | 344 | 285 |
| Barcelona | 10,498 | 266 | 195 |
| Berlin | 6,479 | 168 | 131 |
| Columbus | 2,081 | 218 | 164.5 |

Mean is above median in every city: a few very expensive listings pull it up. Use the **median** for "typical price".

### 2. 28% of listings have no price, and most of them look inactive
- Share with a price: Amsterdam 57%, Barcelona 84%, Berlin 66%, Columbus 90%.
- Listings without a price are smaller (2.8 vs 3.7 guests), have far fewer available days (31 vs 204 per year) and fewer reviews (22 vs 80). Rating is similar (4.80 vs 4.71) and room-type mix is close.
- Zero available days: no price 92% (Amsterdam), 80% (Berlin), 62% (Barcelona), 51% (Columbus); with price only 0.1% to 16%.
- Of the 7,504 listings with no price and zero availability, 72% had no review in the last year, so most are probably inactive (an inference, not proven).
- About 2,100 listings (roughly 6% of train, approximate) have no price, zero availability and some recent reviews: they may be fully booked, and a price model never sees them.
- Consequence: a price model describes **listings that are open for booking**, not the whole market.

### 3. The old IQR outlier rule removes real listings
- IQR rule (per city): 1,294 priced listings above the limit (5.4%). Only 240 are above the city 99th percentile, so **1,054 sit between the IQR limit and the 99th percentile**.
- Those 1,054 look real: median 6 guests, 3 bedrooms, 11 reviews, about 115 per person per night.
- The top 1% (240 listings) are mostly large homes (median 12 guests, 4 bedrooms). Only about 7% have more than 500 per person.
- The cheapest 1% (241 listings) are 96% private or shared rooms (median 24.6, 1 guest). Real prices.
- Clear errors are better found with **price per person**: more than 500 per person gives only **32 listings** (0.13%), such as a 2-guest 1-bedroom flat at 11,000 per night. Other thresholds: 300 -> 162, 400 -> 70, 750 -> 15.
- Prices are asking prices: many of the most expensive listings have zero reviews, so nobody paid them.

### 4. Size of the listing is the strongest driver of price
Spearman correlation with log(price), train, priced listings, without the 32 suspect rows (23,889 rows):

| Feature | Amsterdam | Barcelona | Berlin | Columbus | All |
|---|---|---|---|---|---|
| accommodates | 0.49 | 0.65 | 0.59 | 0.70 | 0.53 |
| bedrooms | 0.55 | 0.51 | 0.47 | 0.63 | 0.47 |
| beds | 0.38 | 0.61 | 0.49 | 0.67 | 0.46 |
| bathrooms | 0.33 | 0.28 | 0.25 | 0.45 | 0.28 |

Same direction in all four cities. These four columns measure almost the same thing (size).

### 5. Location matters, but less than size
- `dist_center_km`: -0.11 to -0.19 in every city (closer to the center = higher price).
- `landmarks_2km`: 0.12 to 0.16 inside cities, but 0.20 for all cities together, so part of the pooled value is a difference between cities.
- `dist_nearest_landmark_km`: between 0.00 and -0.13, weak and not stable.
- `availability_365`: between -0.01 and 0.08, no relation to price.

### 6. Reviews and host experience behave differently in each city
- `number_of_reviews`: -0.29 in Amsterdam, +0.36 in Barcelona, +0.21 Berlin, +0.20 Columbus (pooled +0.17 hides this).
- `host_tenure_years`: 0.33 in Barcelona, about 0 elsewhere.
- `review_scores_rating`: 0.30 in Amsterdam, about 0 to 0.09 elsewhere.
- The reasons are not tested. Columbus has only 2,081 priced listings, so its numbers are less stable.

### 7. Room type: entire homes cost about 2 to 3 times a private room
- 75% of listings are entire homes, 24% private rooms, hotel rooms and shared rooms under 1% each (too few for conclusions; Columbus has 2 shared rooms).
- Median entire home vs private room: Amsterdam 331 vs 175, Barcelona 224 vs 76, Berlin 158 vs 85, Columbus 186 vs 67.
- Part of the effect of `accommodates` is really the room type.

### 8. Availability data cannot show peak season
- Share of listings available rises during the first 3 months simply because days close to the scrape date are already booked.
- Sudden drops appear about **6 and 9 months after the scrape date** (Amsterdam 2027-03-16 and 2027-06-14, Berlin 2027-03-19 and 2027-06-17, Barcelona 2027-06-15). These look like booking-window limits set by hosts, not demand. This is the most likely explanation, not proven.
- Weekend vs other days (Jan 5 to Mar 10, 2027): almost no difference (Amsterdam 0.333 vs 0.334, Barcelona 0.661 vs 0.662, Berlin 0.508 vs 0.509), and Columbus shows 0.753 vs 0.768.
- So the calendar is **not enough to name a peak season**, and it shows no clear weekly pattern.

## Missing values (train set)

Checked on train (33,300 listings) and on the listings that will be used for the model (23,889 with a price).

| Columns | Missing | Reason | Decision |
|---|---|---|---|
| `price`, `price_per_person`, `estimated_revenue_l365d` | 28.2% | listings without a price (see insight 2) | never filled; `price` is the target and `estimated_revenue_l365d` leaks it |
| `review_scores_*` (7 columns) | 15.5% | listing has no reviews | fill with the train median and keep the flag `has_reviews` |
| `reviews_per_month` | 15.5% | no reviews | fill with 0 |
| `first_review`, `last_review` | 15.5% | no reviews | use only "days since last review"; missing = the largest value in train |
| `minimum_nights`, `maximum_nights` and the 4 related columns | 6.4% to 8.4% | unknown | train median |
| `neighbourhood_group_cleansed` | 29% to 33% | 100% missing in Amsterdam and Columbus (the column does not exist there) | dropped; use `neighbourhood_cleansed`, which has no missing values |
| `host_location` | 18% to 20% | free text | dropped |
| `beds`, `bedrooms`, `bathrooms` | under 3% | unknown | train median |

Rule: every fill value (median, maximum) is computed on **train only** and then applied to both train and test, so nothing from the test set leaks into training.

## Decisions taken from the EDA

1. **Split:** 80/20 by `host_id`, random_state 42: 33,300 train / 7,883 test, saved in `data/processed/split_assignment.csv`. No host appears in both sets.
2. **Rows used for modelling:** listings with a price and price per person at most 500 (23,889 rows in train). Listings without a price (28%) and 32 suspect listings are excluded. The same filter is applied to test, so test results describe listings with a sensible price that are open for booking.
3. **Target:** `log(price)`.
4. **Outliers:** the IQR rule is replaced by the fixed threshold "price per person > 500" (insight 3).
5. **Features not used:** `estimated_revenue_l365d`, `estimated_occupancy_l365d`, `price_missing`, `price_outlier` (they leak the price), and `neighbourhood_group_cleansed`, `host_location`.

## Notes for the ML team

1. **Outliers:** replace the IQR rule (about 1,294 priced listings removed in this train set) with "price per person > 500" (32 listings). Use a fixed threshold, not limits computed on all data, so test prices do not influence cleaning. The IQR flag `price_outlier` was computed in cleaning on the full data.
2. **Leakage:** do not use `estimated_revenue_l365d`, `estimated_occupancy_l365d`, `price_missing`, `price_outlier`. The current notebook already leaves them out.
3. **Target:** model `log(price)` (the current notebook does).
4. **Who the model is for:** listings open for booking. Listings without a price are mostly inactive (insight 2).
5. **City:** keep `city` as a feature; effects of reviews and host experience differ by city (insight 6). Weather and economy are city-level, so they separate cities, not listings.
6. **Model selection:** choose the model with cross-validation and use the test set once at the end (the current notebook picks the best model by test MAE).
7. **Split:** the ML notebook splits after removing 13,291 rows, so its split is different from `split_assignment.csv`. Agree on one split before comparing results.

## Limitations
- Correlations show direction, not cause.
- Columbus has few listings, so its figures are less stable.
- Prices are asking prices from one scrape.
- The calendar is one snapshot, so seasonality cannot be measured.
- Weather is reference weather from one year earlier and was not analyzed here.
- Test results are measured only on listings with a price and price per person at most 500 that are open for booking; they do not describe inactive listings or extreme prices.