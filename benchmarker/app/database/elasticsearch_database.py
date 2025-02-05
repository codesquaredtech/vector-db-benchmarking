from app.database.vector_database import VectorDatabase
from elasticsearch import Elasticsearch, helpers
from app.logger import get_logger

logger = get_logger()

class ElasticsearchDatabase(VectorDatabase):

    def __init__(self):
        self.client = None


    def connect(self, host="elasticsearch_db", port=9200):

        try:
            self.client = Elasticsearch([{'host': host, 'port': port, 'scheme': 'http'}])
            if not self.client.ping():
                raise ConnectionError(f"Could not connect to Elasticsearch at {host}:{port}")
            
            logger.info(f"Successfully connected to Elasticsearch at {host}:{port}")
                
        except ConnectionError as ce:
            logger.error(f"Error connecting to Elasticsearch at {host}:{port}: {ce}")
            self.client = None
        
        except Exception as e:
            logger.error(f"Unexpected error occured while connecting to Elasticsearch at {host}:{port}: {e}")
            self.client = None


    def drop_collection(self, collection_name: str):

        if self.client is None:
            logger.error("Elasticsearch client is not connected.")
            return

        index_name = collection_name.lower() #Has to be lowercase for es
        if self.client.indices.exists(index=index_name):
            self.client.indices.delete(index=index_name)
            logger.info(f"Index '{collection_name}' deleted successfully.") 
        else:
            logger.info(f"Index '{collection_name}' does not exist. Skipping drop.")


    def create_collection(self, collection_name: str):

        if self.client is None:
            logger.error("Elasticsearch client is not connected.")
            return
        
        index_name = collection_name.lower()

        mapping = {
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 1280
                    },
                    "image_path": {"type": "keyword"}
                }
            }
        }

        if not self.client.indices.exists(index=index_name):
            self.client.indices.create(index=index_name, body=mapping)
            logger.info(f"Index '{collection_name}' created successfully.")
        else:
            logger.warning(f"Index '{collection_name}' already exists. Skipping index creation.")


    def insert(self, collection_name: str, data):

        if self.client is None:
            logger.error("Elasticsearch client is not connected.")
            return
        
        if data.empty:
            logger.warning("Data is empty. Skipping insert.")

        index_name = collection_name.lower()
        rows = data.to_dict(orient="records")
        ids = data.index.tolist()

        actions = [
            {
                "_index": index_name,
                "_id": ids[i],
                "_source": {
                    "embedding": row["embedding"],
                    "image_path": row["image_path"]
                }
            }
            for i, row in enumerate(rows)
        ]

        helpers.bulk(self.client, actions)


    def delete(self, collection_name: str):

        if self.client is None:
            logger.error("Elasticsearch client is not connected.")
            return
        
        self.client.delete_by_query(index=collection_name.lower(), body={"query": {"match_all": {}}})


    def search(self, collection_name: str, embedding: list, params: dict):

        if self.client is None:
            logger.error("Elasticsearch client is not connected.")
            return []

        index_name = collection_name.lower()
        limit = params.get("limit", 16000)
        certainty = params.get("certainty", 0.5)
        results = []


        query = {
            "script_score": {
                "query" : {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.queryVector, 'embedding') + 1.0",
                    "params": {
                        "queryVector": embedding
                    }
                }
            }
        }

        body={
            "size": limit,
            "query": query,
            "_source": {"includes": ["image_path"]}
        }

        try:
            response = self.client.search(index=index_name, body=body)

            results = [
                {
                    **hit["_source"],
                    "score": hit["_score"]
                }
                for hit in response["hits"]["hits"]
                if hit["_score"] >= certainty
            ]

            return results

        except Exception as e:
            logger.error(f"Error occured while searching in Elasticsearch: {e}")
            return []


    def parse_search_results(self, results: list):

        if self.client is None:
            logger.error("Elasticsearch client is not connected.")
            return

        similar_embeddings = []
        
        for result in results:
            image_path = result["image_path"]
            score = result["score"]

            logger.info(f"Image path: {image_path}, Score: {score}")

            if image_path:
                similar_embeddings.append(image_path.split("/")[-1])

        return similar_embeddings