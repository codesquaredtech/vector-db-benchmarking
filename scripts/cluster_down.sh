
echo "Removing milvus"
docker compose -f "./milvus/docker-compose.yaml" down -v

echo "Removing vectorizer"
docker compose -f "./vectorizer/docker-compose.yaml" down -v

echo "Removing benchmarker"
docker compose -f "./vectorizer/docker-compose.yaml" down -v

echo "Deleting network"
docker network rm vector_db_testing