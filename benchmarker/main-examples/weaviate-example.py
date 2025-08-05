'''
logger = get_logger()

db = get_vector_database(DATABASE_FOR_BENCHMARKING)
"""
Insert + Delete benchmarking
"""

insert_embeddings(db, NUM_ITERATIONS, COLLECTION_NAME, VECTOR_SIZE, DATABASE_FOR_BENCHMARKING)

"""
Search benchmarking
"""
search_params = get_default_search_params(DATABASE_FOR_BENCHMARKING)
search_similar_embeddings(
    db=db,
    collection_name=COLLECTION_NAME,
    vector_size=VECTOR_SIZE,
    search_params=search_params,
    database_name=DATABASE_FOR_BENCHMARKING,
    num_threads=10,
    num_iterations=NUM_ITERATIONS,
)
'''
