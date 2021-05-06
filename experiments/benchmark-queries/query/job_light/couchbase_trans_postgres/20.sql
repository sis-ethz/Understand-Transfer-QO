CREATE INDEX temp_imdb_idx_1 ON imdb_cast_info(movie_id);
CREATE INDEX temp_imdb_idx_2 ON imdb_title(id, production_year);


SELECT COUNT(*) 
FROM imdb_cast_info ci
JOIN imdb_title t USE HASH(BUILD) ON t.id=ci.movie_id 
WHERE t.production_year>1980 AND t.production_year<1995;

DROP INDEX imdb_cast_info.temp_imdb_idx_1;
DROP INDEX imdb_title.temp_imdb_idx_2;