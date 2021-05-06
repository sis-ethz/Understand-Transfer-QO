CREATE INDEX temp_imdb_idx_1 ON imdb_movie_companies(movie_id);
CREATE INDEX temp_imdb_idx_2 ON imdb_title(id);
CREATE INDEX temp_imdb_idx_3 ON imdb_movie_keyword(movie_id, keyword_id);

SELECT COUNT(*) 
FROM imdb_movie_companies mc 
JOIN imdb_title t ON t.id=mc.movie_id
JOIN imdb_movie_keyword mk ON t.id=mk.movie_id
WHERE mk.keyword_id=117;

DROP INDEX imdb_movie_companies.temp_imdb_idx_1;
DROP INDEX imdb_title.temp_imdb_idx_2;
DROP INDEX imdb_movie_keyword.temp_imdb_idx_3;