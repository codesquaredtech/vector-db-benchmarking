echo "Creating network"
docker network create vector_db_testing

# 1280 for mediapipe, 512 for insightface, 768 for dino
export DATABASE="ELASTICSEARCH"
export COLLECTION_NAME="faces"
export NUM_ITERATIONS=3
export VECTOR_SIZE=512

DB=$(echo "$DATABASE" | tr '[:upper:]' '[:lower:]')

# Service startup functions
start_milvus() {
    echo "Starting Milvus service..."
    docker compose -f "./milvus/docker-compose.yaml" up -d --build
}

start_weaviate() {
    echo "Starting Weaviate service..."
    docker compose -f "./weaviate/docker-compose.yaml" up -d
}

start_elasticsearch() {
    echo "Starting Elasticsearch service..."
    docker compose -f "./elasticsearch/docker-compose.yaml" up -d
}

start_qdrant() {
    echo "Starting Qdrant service..."
    docker compose -f "./qdrant/docker-compose.yaml" up -d
}

start_pgvector() {
    echo "Starting PGVector service..."
    docker compose -f "./pgvector/docker-compose.yaml" up -d
    echo "Sleeping 10 seconds..."
    sleep 10

    echo "Installing dos2unix inside the container..."
    docker exec -it pgvector_db bash -c "apt update && apt install -y dos2unix"

    echo "Converting line endings for init.sh..."
    docker exec -it pgvector_db bash -c "dos2unix /config/init.sh"

    echo "Sleeping 5 seconds..."
    sleep 5

    echo "Configuring PGVector..."
    docker exec -it pgvector_db bash -c "/config/init.sh"
}

start_chroma() {
    echo "Starting Chroma service..."
    docker compose -f "./chroma/docker-compose.yaml" up -d --build
}

# Start the appropriate DB service
echo "Launching database service for: $DATABASE"
case "$DB" in
    milvus)
        start_milvus
        ;;
    weaviate)
        start_weaviate
        ;;
    elasticsearch)
        start_elasticsearch
        ;;
    qdrant)
        start_qdrant
        ;;
    pgvector)
        start_pgvector
        ;;
    chroma)
        start_chroma
        ;;
    *)
        echo "Error: Unsupported database type '$DATABASE'"
        exit 1
        ;;
esac

echo "Sleeping 30 seconds while the vector DB container initializes..."
sleep 30

docker compose -f "./benchmarker/docker-compose.yaml" up -d --build
