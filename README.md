# vector-db-benchmarking

## Running the Vectoriser
- Position yourself in the root folder (vector-db-benchmarking).
- Take a look at ```/scripts/vectoriser_up.sh``` script. There are two parameters that can be modified:
    - MODEL - Name of the embedder model that will be used for the embedding creation;
    - DIR - Name of the directory where the pictures are stored
- To run the vectoriser component, run the following command: ```./scripts/vectoriser_up.sh```.
- To shut down the vectoriser component, run the following command: ```./scripts/vectorisers_down.sh```.
- #todo - when automatizing vectorisation, somehow make it so that small, medium, large datasets (if we still need them) are separated (each smaller will be a part of the largest dataset); maybe: when there have been X pictures that have been processed, place all of the embeddings in the small dataset folder, same for the medium, large is everything;

## Running the Benchmarker
- Position yourself in the root folder (vector-db-benchmarking).
- Take a look at ```/scripts/benchmarker_up.sh``` script. There are four parameters that can be modified:
    - DATABASE - Name of the database that you want to perform benchmarking on;
    - COLLECTION_NAME - Name of the collection inside of the vector database in which you want to store the embeddings;
    - NUM_ITERATIONS - Number of iterations to be run in the insert/delete part of the benchmarking;
    - VECTOR_SIZE - Vector size depending on the previous embedder model usage;
- To run the benchmarker component, run the following command ```./scripts/benchmarker_up.sh```.
- To shut down the benchmarker component, run the following command: ```./scripts/benchmarker_down.sh```.
- #todo - automatization - this should include another parameter for what is being benchmarked (search/insert+delete)
- #todo - maybe modify the vector_size to be also MODEL and then retrieve the vector size from an if/else
- #todo - ...

## Working with the Vector Databases
Each vector database should be a separate component. After adding all of the necessary files inside of the new vector database directory (e.g., take a look at the  _mivlus_ directory), ```benchmarker_up.sh``` and ```benchmarker_down.sh``` should be complemented with the code to create and run the vector database, as well as to shut it down.

## Working with the Vectorizer Component
Three variables should be changed if needed when running the Vectorizer Component (in the main.py file):
- <b>REFERENT_IMAGE_DIRECTORIES</b> - refers to relative paths of the directories where the images that should be processed are present;
- <b>OUTPUT_FILE_PATH</b> - refers to the relative path of the ```.parquet``` file which will contain all of the extracted embeddings;
- <b>OUTPUT_EMBEDDING_TO_COMPARE_WITH_PATH</b> - refers to the relative path of the ```.csv``` file which will contain embedding and image path (name) of the face that is being searched for in the benchmarker component.

It should be noted that generated files should be placed in the benchmarker component (defined in the next segment).

## Working with the Benchmarker Component
There are a few things to consider when working with the benchmarker component:
- <b>input directory</b> - this is the directory where the previously created embeddings should be put. Additionally, <b>INPUT_FOLDER_PATH</b> in the ```main.py``` file should be modified if new embeddings are inserted.
- <b>app/database directory</b> - this is the directory that contains the vector_database abstract class, whose methods should be implemented by each of the vector databases. An example is present in the ```milvus_database.py``` file. Additionally, ```get_vector_database(db_type)``` function in the ```main.py``` should be updated to include the newly implemented vector database. This is the only part where something new should be implemented in the benchmarker component.
- <b>app/search_data directory</b> - this is the directory where both the ```embedding_compare_with.csv``` file with the embedding and image path of the face that is being searched for, as well as the ```labeled_pictures.csv``` file should be. The first mentioned ```.csv``` file is loaded in the code when the search operation is done. The second ```.csv``` file is of the following structure: the ```picture_name``` column should contain all of the picture names present in the vector data. Other columns should have their header name set to the picture name where the face that is being search for is present (e.g., ```test_1.jpg```), while their content should be 0/1 representing whether that face is present on the corresponding picture or not. See the present ```search_data/*.csv``` files as an example. Note: there is a function in the ```vectorizer/app/utils.py``` file that generates the ```.csv``` file with the above mentioned structure.
- <b>main.py</b> - two main operations are occurring in this file: insertion + deletion benchmarking and search benchmarking. Functions referring to those operations should not be modified. The only code that should be modified is under the ```if __name__ == '__main__':``` line of code, as well as the global variables present at the top of the ```main.py``` file.
- <b>results directory</b> - this is the directory that will include both insertion + deletion, as well as the search benchmarking results. These results will be stored in the ```.csv``` files.
- <b>main-examples directory</b> - this is the directory that should include examples of the code below the ```if __name__ == '__main__':``` line of code for each vector database. Be sure to add this when done with the implementation.

## Working with the Data Visualiser Component
Main point of this component is to generate plots that visualise the received results accordingly. Few things should be considered when working with this component:
- <b>results directory</b> - results from the benchmarking component should be put here.
- <b>main.py</b> - to generate plots, functions that are present in this file should be called (there are examples here already).
- <b>plots directory</b> - all of the generated plots will be placed in this directory.

## Experiment Results
Experiment results from the research paper are present in the ```Experiment Results``` folder.

## Datasets
All of the used data is available <a href="https://drive.google.com/file/d/1Gm9NvwOrOaeJX1mAEBdejqKcR9xcdOO9/view?usp=sharing">here</a>.
