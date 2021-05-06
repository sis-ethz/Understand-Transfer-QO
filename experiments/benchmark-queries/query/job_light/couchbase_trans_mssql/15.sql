CREATE INDEX temp_imdb_idx_1 ON imdb_title(id, production_year);
CREATE INDEX temp_imdb_idx_2 ON imdb_cast_info(movie_id);
CREATE INDEX temp_imdb_idx_3 ON imdb_movie_keyword(movie_id, keyword_id);

SELECT COUNT(*) 
FROM imdb_movie_keyword mk
JOIN imdb_title t USE HASH(BUILD) ON t.id=mk.movie_id
JOIN imdb_cast_info ci USE HASH(PROBE) ON t.id=ci.movie_id
WHERE t.production_year>2010 AND mk.keyword_id=8200;

DROP INDEX imdb_title.temp_imdb_idx_1;
DROP INDEX imdb_cast_info.temp_imdb_idx_2;
DROP INDEX imdb_movie_keyword.temp_imdb_idx_3;