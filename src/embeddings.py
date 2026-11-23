
import pandas as pd
import numpy as np
import torch

from pathlib import Path
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "url_features.csv"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT / "data" / "processed" / "embeddings"
)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Output files
OUTPUT_FILE = OUTPUT_FOLDER / "url_embeddings.npy"

METADATA_FILE = OUTPUT_FOLDER / "embedding_metadata.csv"


# ==========================================
# 2. MODEL SETTINGS
# ==========================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 32

# None means process the complete dataset
MAX_ROWS = None


# ==========================================
# 3. LOAD DATASET
# ==========================================

print("=" * 60)
print("FULL DATASET TRANSFORMER EMBEDDING GENERATION")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

df = df.dropna(subset=["URL", "label"]).reset_index(drop=True)

if MAX_ROWS is not None:
    df = df.head(MAX_ROWS).copy()

urls = df["URL"].astype(str).tolist()

total_urls = len(urls)

print("\nTotal URLs:", total_urls)


# ==========================================
# 4. SELECT DEVICE
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("\nSelected device:", device)

if device == "cpu":
    print("Running on CPU. Processing may take longer.")


# ==========================================
# 5. LOAD TRANSFORMER MODEL
# ==========================================

print("\nLoading Transformer model...")

model = SentenceTransformer(
    MODEL_NAME,
    device=device
)

embedding_dimension = model.get_sentence_embedding_dimension()

print("Model loaded successfully!")

print("Embedding dimension:", embedding_dimension)


# ==========================================
# 6. CREATE OUTPUT MEMORY MAP
# ==========================================

print("\nPreparing output storage...")

embeddings = np.lib.format.open_memmap(
    OUTPUT_FILE,
    mode="w+",
    dtype="float32",
    shape=(total_urls, embedding_dimension)
)


# ==========================================
# 7. GENERATE EMBEDDINGS IN BATCHES
# ==========================================

print("\nStarting embedding generation...")

total_batches = (total_urls + BATCH_SIZE - 1) // BATCH_SIZE

try:

    for batch_number, start in enumerate(
        range(0, total_urls, BATCH_SIZE), start=1
    ):

        end = min(start + BATCH_SIZE, total_urls)

        batch_urls = urls[start:end]

        batch_embeddings = model.encode(
            batch_urls,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        embeddings[start:end] = batch_embeddings

        # Display progress periodically
        if batch_number % 100 == 0 or batch_number == total_batches:

            percentage = (end / total_urls) * 100

            print(
                f"Batch {batch_number}/{total_batches} | "
                f"Processed {end}/{total_urls} URLs | "
                f"{percentage:.2f}% completed"
            )

    embeddings.flush()

finally:
    embeddings.flush()


# ==========================================
# 8. SAVE METADATA
# ==========================================

print("\nSaving URL metadata...")

df[["URL", "label"]].to_csv(
    METADATA_FILE,
    index=False
)


# ==========================================
# 9. VERIFY EMBEDDINGS
# ==========================================

print("\n" + "=" * 60)
print("EMBEDDING GENERATION COMPLETED!")
print("=" * 60)

print("\nTotal URLs processed:", total_urls)

print("Embedding shape:", embeddings.shape)

print("Embedding dimension:", embedding_dimension)

print("Data type:", embeddings.dtype)

print("\nFirst five embedding values:")

print(embeddings[0][:5])

print("\nSaved embedding file:", OUTPUT_FILE)

print("\nSaved metadata file:", METADATA_FILE)

print("\nALL URL EMBEDDINGS GENERATED SUCCESSFULLY!")
