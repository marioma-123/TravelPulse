USE airbnb_project;
-- Run once after loading data
CREATE INDEX idx_listings_city_id ON raw_listings (city, id);
CREATE INDEX idx_calendar_city_listing ON raw_calendar (city, listing_id);
CREATE INDEX idx_calendar_date ON raw_calendar (`date`);
ANALYZE TABLE raw_listings, raw_calendar;

SHOW INDEX FROM raw_listings;
SHOW INDEX FROM raw_calendar;
