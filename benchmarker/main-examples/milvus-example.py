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

search_params = search_params = {
    "anns_field": "embedding",
    "metric_type": "COSINE",
    "index_params": {"ef": 64},
    "limit": None,
    "threshold": 0.8,
    "output_fields": ["id", "image_path"],
}

search_similar_embeddings(
    db,
    search_params,
    num_threads=50,
    num_iterations=100,
)
'''
