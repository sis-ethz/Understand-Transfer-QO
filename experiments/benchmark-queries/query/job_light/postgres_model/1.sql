
/*+
HashJoin(mi_idx t)
HashJoin(mc mi_idx t)
SeqScan(mi_idx)
SeqScan(mc)
*/
SELECT COUNT(*) FROM movie_companies mc,title t,movie_info_idx mi_idx 
WHERE t.id=mc.movie_id AND t.id=mi_idx.movie_id AND mi_idx.info_type_id=112 AND mc.company_type_id=2;