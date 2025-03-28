from datasets import load_dataset, load_from_disk, Dataset
import matplotlib.pyplot as plt
import os
import chromadb
from chromadb.api.models import Collection
from chromadb.utils import embedding_functions
from chromadb.utils.data_loaders import ImageLoader
from tqdm import tqdm

from utils import pil2base64

DATASET_PATH = "./data/flowers-102-categories"
DATABASE_PATH = "./data/flower.db"
NUM_DATA = 300


def load_local_dataset() -> Dataset:
    if not os.path.exists(DATASET_PATH):
        print("Downloading dataset...")
        download_dataset()

    print("Loading dataset...")
    ds = load_from_disk(DATASET_PATH)
    return ds


def download_dataset() -> None:
    """
    Download a huggingface dataset to disk.
    """
    os.makedirs(DATASET_PATH, exist_ok=True)

    ds = load_dataset("huggan/flowers-102-categories")
    ds = ds["train"]
    ds.save_to_disk(DATASET_PATH)


def save_examples(ds: Dataset, path: str) -> None:
    num_data = 10
    ds = ds.shuffle(seed=42).take(num_data)

    for i in tqdm(range(0, len(ds)), desc="Adding data"):
        image = ds[i]["image"]
        image.save(os.path.join(path, f"flower_{i+1}.png"))


def load_collection() -> Collection:
    print("Loading collection...")
    chroma_client = chromadb.PersistentClient(path=DATABASE_PATH)

    image_loader = ImageLoader()
    embedding_function = embedding_functions.OpenCLIPEmbeddingFunction()

    collection = chroma_client.get_or_create_collection(
        "flower_collection",
        embedding_function=embedding_function,
        data_loader=image_loader
    )

    # Initialize collection if no data
    if collection.count() == 0:
        init_collection(collection)

    return collection


def init_collection(collection: Collection) -> None:
    print("Initializing collection...")

    ds = load_from_disk(DATASET_PATH)
    ds = ds.map(lambda example, idx: {"id": str(idx)}, with_indices=True)  # add feature "id"
    ds = ds.with_format("numpy")  # format numpy array

    if NUM_DATA is not None:
        ds = ds.shuffle(seed=10).take(NUM_DATA)

    # Add data in batches
    batch_size = 100
    for i in tqdm(range(0, len(ds), batch_size), desc="Add data to collection"):
        ids = [ds[i]["id"] for i in range(i, i + batch_size)]  # id should be string
        images = [ds[i]["image"] for i in range(i, i + batch_size)]  # image should be numpy array
        collection.add(ids=ids, images=images)


def do_query(collection: Collection, query_text: str, n_results=5) -> dict:
    print(f"Querying database...")
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["distances"]
    )
    return results


def print_query_result(ds: Dataset, query_text: str, result: dict) -> None:
    print(f"-- Query --")
    print(query_text)

    print(f"-- Results --")
    n_results = len(result["ids"][0])
    for i in range(n_results):
        id = result["ids"][0][i]
        distance = result["distances"][0][i]
        data = ds[int(id)]["image"]

        print(f"id: {id}, distance: {distance:.4f}:")

        ax = plt.subplot(1, n_results, i+1)
        ax.imshow(data)
        ax.axis("off")
        ax.set_title(f"id={id}, \ndis={distance:.4f}")

    plt.suptitle(f"query={query_text}", y=0.8)
    # plt.savefig(f"./results/{query_text}.png")
    plt.show()


def format_query_result(ds: Dataset, query_text: str, result: dict) -> dict:
    """
    Format query result to a dict
    {
        "user_query": str,
        "images": list of Base64 str,
        ...
    }
    """
    print("Formatting query result...")

    result_formatted = {"user_query": query_text, "images": []}

    n_results = len(result["ids"][0])
    for i in range(n_results):
        id = result["ids"][0][i]
        data = ds[int(id)]["image"]
        result_formatted["images"].append(pil2base64(data))

    return result_formatted


if __name__ == "__main__":
    ds = load_local_dataset()
    print(ds)

    # print("Example ds[0]:")
    # print(ds[0])
    # img = ds[0]["image"]
    # plt.imshow(img)
    # plt.axis("off")
    # plt.show()

    collection = load_collection()
    print("collection count:", collection.count())

    query_text = "soft pink flower"
    result = do_query(collection, query_text, 5)
    print(result)

    print_query_result(ds, query_text, result)

    result_formatted = format_query_result(ds, query_text, result)
    print(result_formatted)