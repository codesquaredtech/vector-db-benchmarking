echo "Removing milvus"
docker compose -f "./milvus/docker-compose.yaml" down -v

echo "Removing weaviate"
docker compose -f "./weaviate/docker-compose.yaml" down -v

echo "Removing pgvector"
docker compose -f "./pgvector/docker-compose.yaml" down -v

#echo "Removing vectorizer"
#docker compose -f "./vectorizer/docker-compose.yaml" down -v

echo "Removing benchmarker"
docker compose -f "./benchmarker/docker-compose.yaml" down -v

echo "Deleting network"
docker network rm vector_db_testing