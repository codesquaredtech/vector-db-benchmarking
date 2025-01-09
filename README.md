# vector-db-benchmarking

## Running the Cluster

Position yourself in the root folder (vector-db-benchmarking).
To run the cluster, run ```./scripts/cluster_up.sh```.
Running the cluster consists of running three main components:
- <b>vector databases</b> - depending on which vector database is being benchmarked, others can be commented out in the script;
- <b>vectorizer</b> - this component should be commented out if no new embeddings are being created, but instead benchmarking is the only thing that is being done;
- <b>benchmarker</b> - if there is only need for creating the new embeddings that will later be used by benchmarker, this component can be commented out.

To shut down the cluster, run ```./scripts/cluster_down.sh```.

## Working with the Vector Databases
Each vector database should be a separate component. After adding all of the necessary files inside of the new vector database directory (e.g., take a look at the  _mivlus_ directory), ```cluster_up.sh``` and ```cluster_down.sh``` should be complemented with the code to create and run the vector database, as well as to shut it down.

## Working with the Vectorizer Component
Only two variables should be changed if needed when running the Vectorizer Component (in the main.py file):
- <b>REFERENT_IMAGE_DIRECTORY</b> - refers to relative path of the directory where the images that should be processed are present
- <b>OUTPUT_FILE_PATH</b> - refers to the relative path of the ```.parquet``` file which will contain all of the extracted embeddings

## Working with the Benchmarker Component
There are a few things to consider when working with the benchmarker component:
- <b>input directory</b> - this is the directory where the previously created embeddings should be put. Additionally, <b>INPUT_FILE_PATH</b> in the ```main.py``` file should be modified if new embeddings are inserted.
- <b>app/database directory</b> - this is the directory that contains the vector_database abstract class, whose methods should be implemented by each of the vector databases. An example is present in the ```milvus_database.py``` file. Additionally, ```get_vector_database(db_type)``` function in the ```main.py``` should be updated to include the newly implemented vector database. This is the only part where something new should be implemented in the benchmarker component. 
- <b>app/search_data directory</b> - this is the directory where both the face picture that is being searched for, as well as the ```labeled_pictures.csv``` file should be. The previously mentioned csv file is of the following structure: the ```picture_name``` column should contain all of the picture names present in the <b>REFERENT_IMAGE_DIRECTORY</b>. Other columns should have their header name set to the picture name where the face that is being search for is present (e.g., ```test_1.jpg```), while their content should be 0/1 representing whether that face is present on the corresponding picture or not. See the present ```search_data/labeled_pictures.csv``` file as an example. Note: there is a function in the ```vectorizer/app/utils.py``` file that generates the csv file with the above mentioned structure.
- <b>main.py</b> - two main operations are occurring in this file: insertion + deletion benchmarking and search benchmarking. Functions referring to those operations should not be modified. The only code that should be modified is under the ```if __name__ == '__main__':``` line of code, as well as the global variables present at the top of the ```main.py``` file.
- <b>results directory</b> - this is the directory that will include both insertion + deletion, as well as the search benchmarking results. These results will be stored in the csv files.