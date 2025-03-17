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

search_params = {"certainty": 0.8, "limit": 10, "num_candidates": 100}

search_similar_embeddings(
    db,
    search_params,
    num_threads=50,
    num_iterations=100,
)
'''