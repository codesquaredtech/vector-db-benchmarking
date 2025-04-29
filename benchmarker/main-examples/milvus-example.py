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

search_params = {
    "anns_field": "embedding",
    "metric_type": "COSINE",
    "index_params": {"ef": 10000},
    "limit": None,
    "threshold": 0.6,
    "output_fields": ["id", "image_path"],
}

search_similar_embeddings(
    db,
    search_params,
    num_threads=10,
    num_iterations=100,
)
'''
