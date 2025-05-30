echo "Removing milvus"
docker compose -f "./milvus/docker-compose.yaml" down -v

echo "Removing weaviate"
docker compose -f "./weaviate/docker-compose.yaml" down -v

echo "Removing elasticsearch"
docker compose -f "./elasticsearch/docker-compose.yaml" down -v

echo "Removing qdrant"
docker compose -f "./qdrant/docker-compose.yaml" down -v

echo "Removing pgvector"
docker compose -f "./pgvector/docker-compose.yaml" down -v

echo "Removing chroma"
docker compose -f "./chroma/docker-compose.yaml" down -v

echo "Removing vectorizer"
docker compose -f "./vectorizer/docker-compose.yaml" down -v

echo "Removing benchmarker"
docker compose -f "./benchmarker/docker-compose.yaml" down -v

echo "Removing data visualiser"
docker compose -f "./data_visualiser/docker-compose.yaml" down -v

echo "Deleting network"
docker network rm vector_db_testing