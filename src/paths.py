import os
# Data lives in ../data next to src/, unless CFB_DATA is set
DATA_DIR = os.environ.get("CFB_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
