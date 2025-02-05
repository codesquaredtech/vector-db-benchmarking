echo "Creating network"
docker network create vector_db_testing

# Functions for running individual services
start_milvus() {
    echo "Starting Milvus service..."
    docker compose -f "./milvus/docker-compose.yaml" up -d
}

start_weavite() {
    echo "Starting Weavite service..."
    docker compose -f "./weaviate/docker-compose.yaml" up -d
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
    echo "Configuring PGVector"
    docker exec -it pgvector_db bash -c "/config/init.sh"
}


# Flags initialization
MV=false
WV=false
PG=false
QD=false
ALL=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -mv)
            MV=true
            ;;
        -wv)
            WV=true
            ;;
        -pg)
            PG=true
            ;;
        -qd)
            QD=true
            ;;
        -all)
            ALL=true
            ;;
        *)
            echo "Invalid flag: $1"
            exit 1
            ;;
    esac
    shift

done

# Check if any service-specific flag is true
if ! $MV && ! $WV && ! $PG && ! $QD; then
    echo "No service-specific flags provided. Running all."
    ALL=true
fi

echo "Running DB sevices"
if $ALL; then
    echo "Running all DB services."
    start_milvus
    start_weavite
    start_pgvector
    start_qdrant
else
    echo "Running individual DB services."
    if $MV; then
        start_milvus
    fi
    if $WV; then
        start_weavite
    fi
    if $PG; then
        start_pgvector
    fi
    if $QD; then
        start_qdrant
    fi
fi

echo "Sleeping 30 seconds while everything is up and running."
sleep 30

echo "Starting vectorizer"
docker compose -f "./vectorizer/docker-compose.yaml" up -d

echo "Starting benchmarker"
docker compose -f "./benchmarker/docker-compose.yaml" up -d