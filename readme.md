# KG training Framework for WMD

## Configure

To set up and configure the Neo4j database and TransE model for training, follow these steps:

### Neo4j Configuration
To fetch dataset from Neo4j, ensure that you have a running instance of Neo4j and provide the necessary connection details in `./config/neo4j.config` configuration file. Don't omit the comma at the end of each line. Insert values inside the double quotation.

    {
        "uri": "<NEO4J_DATABASE_URI>",
        "user": "<NEO4J_USERNAME>",
        "password": "<NEO4J_PASSWORD>"
    }

- `uri`: The connection string for the Neo4j database (e.g., bolt://localhost:7687).
- `user`: Your Neo4j username.
- `password`: Your Neo4j password.

### TransE Model Configuration
Define the hyperparameters for training the TransE model in `./config/train.config` configuration file. Don't omit the comma at the end of each line.

    {
        "emb_dim": 100,
        "lr": 0.0004,
        "n_epochs": 1000,
        "b_size": 32768,
        "margin": 0.5
    }

- `emb_dim`: Dimensionality of the entity and relation embeddings.
- `lr`: Learning rate for the optimizer.
- `n_epochs`: Number of training epochs.
- `b_size`: Batch size for training.
- `margin`: Margin value for the ranking loss function.

## Data Loader and Training Script

The script `train.py` loads data from **Neo4j**, a **local TSV file**, or a **remote TSV file**, then trains a model using the retrieved data. The TSV should include only three columns in sequence `HEAD`, `REL`, and `TAIL`. It should not have any header. The first line should contain the first triplet. The entities and relations are automatically converted into indices and stored in `catalog.json` file during training.

### Requirements

- Python 3.x
- Neo4j Database (if using `neo4j` input type)
- Required Python libraries (install via `pip install -r requirements.txt`)

### Usage

Run the script with the appropriate input type and parameters.

#### Command-Line Arguments

| Argument      | Type    | Description |
|--------------|---------|-------------|
| `--input_type` | `str` | Choose input source: `"neo4j"`, `"remote"`, or `"local"`. Required parameter. |
| `--url`      | `str`  | URL of the remote TSV file. Required if `input_type=remote`. |
| `--filename` | `str`  | Path to the local TSV file. Required if `input_type=local`. |

### Examples

#### Load Data from Neo4j

    python script.py --input_type neo4j

#### Load Data from a Remote TSV File

    python script.py --input_type remote --url "https://example.com/data.tsv"

#### Load Data from a Local TSV File

    python script.py --input_type local --filename "./data.tsv"

## Model Inference Script

The `infer.py` script performs inference using a trained model to predict missing entities or relations in a knowledge graph.

### Usage

Run the script with the required arguments to perform inference and save the results to a TSV file.

#### Command-Line Arguments

| Argument       | Type   | Required | Description |
|---------------|--------|----------|-------------|
| `--model`     | `str`  | ✅ | Path to the trained model parameters file |
| `--dataset`   | `str`  | ✅ | Path to the dataset mappings file |
| `--num`       | `int`  | ✅ | Number of top predictions to return |
| `--head`      | `str` or `int`  | ✅ | Head entity name or index (`-1` if unknown) |
| `--rel`       | `str` or `int`  | ✅ | Relation name or index (`-1` if unknown) |
| `--tail`      | `str` or `int`  | ✅ | Tail entity name or index (`-1` if unknown) |
| `--input_type`| `str`  | ✅ | Specify if input is `"idx"` (index) or `"name"` (entity name) |
| `--output`    | `str`  | ✅ | Path to output TSV file |

#### Example Usage

##### Predict missing tail entity

    python infer.py --model trained_model.pth \
                    --dataset dataset_mappings.bin \
                    --num 5 \
                    --head "Paris" \
                    --rel "CapitalOf" \
                    --tail "-1" \
                    --input_type name \
                    --output predictions.tsv

## Upload new inference edges to Neo4j database

Inspect the output tsv file generated by `infer.py` script and delete any unwanted edge(s). Then run-

    python upload.py new_edges.tsv

to upload the edges to Neo4j database. Please note that scores will not be uploaded to neo4j database.