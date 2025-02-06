# pip install neo4j
from neo4j import GraphDatabase
import pandas as pd
import requests
from io import StringIO

read_tsv = lambda x: pd.read_csv(x, low_memory=False, sep='\t', header=None, names=['from', 'rel', 'to'])

def retrieve_from_url(filename):
    print('Reading file...', end='')
    df = read_tsv(filename)
    print('Done\n')
    return df

def retrieve_from_url(tsv_url):
    print('Downloading file...', end='')
    response = requests.get(tsv_url)
    response.raise_for_status()  # Ensure we got a valid response
    df = read_tsv(StringIO(response.text))
    print('Done\n')
    return df

def read_csv(input_file):
    print('Reading file...', end='')
    df = read_tsv(input_file)
    print('Done\n')
    return df

def get_train_config():
    with open('./config/train.config') as f:
        config = eval(f.read())

    emb_dim = config["emb_dim"]
    lr = config["lr"]
    n_epochs = config["n_epochs"]
    b_size = config["b_size"]
    margin = config["margin"]

    return emb_dim, lr, n_epochs, b_size, margin

def get_neo4j_config():
    with open('./config/neo4j.config') as f:
        config = eval(f.read())

    uri = config['uri']
    user = config['user']
    password = config['password']
    return uri, user, password

def upload_to_neo4j(triples, uri, user, password):
    print('Uploading...')
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        for subj, pred, obj in triples.itertuples(index=False, name=None):
            session.run(
                """
                MERGE (subject:Resource {uri: $subj})
                MERGE (object:Resource {uri: $obj})
                MERGE (subject)-[:RELATION {type: $pred}]->(object)
                """,
                subj=str(subj),
                pred=str(pred),
                obj=str(obj),
            )
    driver.close()
    print('Uploaded to Neo4j completed!')

def retrieve_from_neo4j(uri, user, password, limit=None):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    with driver.session() as session:
        query = """
        MATCH (subject:Resource)-[r:RELATION]->(object:Resource)
        RETURN subject.uri AS from, r.type AS rel, object.uri AS to
        """
        if limit:
            query += f" LIMIT {limit}"
        
        result = session.run(query)
        
        data = [record.data() for record in result]
    
    driver.close()
    
    ret = pd.DataFrame(data)
    
    return ret