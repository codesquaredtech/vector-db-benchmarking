'''
logger = get_logger()

db = get_vector_database(DATABASE_FOR_BENCHMARKING)
"""
Insert + Delete benchmarking
"""

insert_embeddings(db)

"""
Search benchmarking
"""

search_params = {"certainty": 0.6, "limit": 10000, "num_candidates": 10000}

search_similar_embeddings(
    db,
    search_params,
    num_threads=10,
    num_iterations=100,
)
'''
