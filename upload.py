
import sys
from utils import *

uri, user, password = get_neo4j_config()

input_file = sys.argv[1]
df = read_csv(input_file)
print(df.head())

upload_to_neo4j(df, uri, user, password)
print('Script finished!')