CREATE INDEX temp_imdb_idx_1 ON imdb_title(id, production_year);
CREATE INDEX temp_imdb_idx_2 ON imdb_movie_info_idx(movie_id, info_type_id);
CREATE INDEX temp_imdb_idx_3 ON imdb_movie_keyword(movie_id);


SELECT COUNT(*) 
FROM imdb_title t
JOIN imdb_movie_info_idx mi_idx USE HASH(BUILD) ON t.id=mi_idx.movie_id
JOIN imdb_movie_keyword mk USE HASH(BUILD) ON t.id=mk.movie_id
WHERE t.production_year>2010 AND mi_idx.info_type_id=101;
 
DROP INDEX imdb_title.temp_imdb_idx_1;
DROP INDEX imdb_movie_info_idx.temp_imdb_idx_2;
DROP INDEX imdb_movie_keyword.temp_imdb_idx_3;