
CREATE INDEX temp_imdb_idx_1 ON imdb_movie_companies(movie_id, company_type_id);
CREATE INDEX temp_imdb_idx_2 ON imdb_title(id);
CREATE INDEX temp_imdb_idx_3 ON imdb_movie_info_idx(movie_id, info_type_id);

SELECT COUNT(*) 
FROM imdb_movie_companies mc 
INNER JOIN imdb_title t ON t.id=mc.movie_id
INNER JOIN imdb_movie_info_idx mi_idx ON t.id=mi_idx.movie_id 
WHERE 
mi_idx.info_type_id=112 AND mc.company_type_id=2;

DROP INDEX imdb_movie_companies.temp_imdb_idx_1;
DROP INDEX imdb_title.temp_imdb_idx_2;
DROP INDEX imdb_movie_info_idx.temp_imdb_idx_3;