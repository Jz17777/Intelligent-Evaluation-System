from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

PROCESSED_DATA_DIR = Path(__file__).parent.parent/'data'/'processed'
LOGS_DIR = Path(__file__).parent.parent/'logs'
RAW_DATA_DIR = Path(__file__).parent.parent/'data'/'raw'
MODELS_DIR = Path(__file__).parent.parent/'models'

SEQ_LEN = 128
BATCH_SIZE = 128
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 3
BIDIRECTIONAL = True

LEARNING_RATE=1e-3
EPOCHS=100