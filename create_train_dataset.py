import gzip
import json
import random


def with_rand_docs():
    # Load all documents into memory for negative sampling
    all_docs = {}
    with gzip.open("data/pubmed_2022_tiny.jsonl.gz", "r") as all_file:
        for i, line in enumerate(all_file):
            print(f"Loading {i} documents for negative sampling...", end="\r")
            doc = json.loads(line)
            all_docs[doc["pmid"]] = doc["title"] + " " + doc["abstract"]

    all_ids_list = list(all_docs.keys())  # List of all document ids

    print()

    # Create the train dataset, each entry contains:
    # - 1 query
    # - N positive/relevant documents
    # - N negative/non-relevant documents
    dataset = []
    with open("data/RI_2023_training_data_wContents.jsonl", "r") as pos_file:
        for i, line in enumerate(pos_file):
            print(f"Creating {i} dataset entries...", end="\r")
            old = json.loads(line)
            out = {}

            # key: query
            out["query"] = old["body"]

            # key: pos_docs (length=N)
            out["pos_docs"] = []
            for doc in old["documents"]:
                out["pos_docs"].append({"id": doc["id"], "text": doc["text"]})
            N = len(out["pos_docs"])  # Number of relevant documents
            pos_ids = {
                doc["id"] for doc in out["pos_docs"]
            }  # Set of relevant document ids

            # key: neg_docs (length=N)
            out["neg_docs"] = []
            while len(out["neg_docs"]) < N:
                choice = random.choice(all_ids_list)
                if choice not in pos_ids:
                    out["neg_docs"].append({"id": choice, "text": all_docs[choice]})

            dataset.append(out)

    print()

    # Save the dataset
    with open("data/train_dataset.jsonl", "w") as out_file:
        print(f"Saving the dataset...")
        [out_file.write(json.dumps(entry) + "\n") for entry in dataset]


def with_own_bm25():
    # Load all documents into memory for negative sampling
    all_docs = {}  # doc_id -> doc_title + doc_abstract
    with gzip.open("data/pubmed_2022_small.jsonl.gz", "r") as all_file:
        for i, line in enumerate(all_file):
            print(f"Loading {i} documents for negative sampling...", end="\r")
            doc = json.loads(line)
            all_docs[doc["pmid"]] = doc["title"] + " " + doc["abstract"]

    print()

    # Load BM25 runs over train dataset questions
    bm25_runs = {}  # query_text -> list of doc_ids
    with open("data/BM25_train_own.jsonl", "r") as bm25_file:
        for i, line in enumerate(bm25_file):
            print(f"Loading {i} BM25 runs...", end="\r")
            data = json.loads(line)
            bm25_runs[data["question"]] = [doc["id"] for doc in data["documents"]]

    print()

    # Create the train dataset, each entry contains:
    # - 1 query
    # - N positive/relevant documents
    # - N negative/non-relevant documents
    dataset = []
    with open("data/RI_2023_training_data_wContents.jsonl", "r") as pos_file:
        for i, line in enumerate(pos_file):
            print(f"Creating {i} dataset entries...", end="\r")
            old = json.loads(line)
            out = {}

            # key: query
            out["query"] = old["body"]

            # key: pos_docs (length=N)
            out["pos_docs"] = []
            for doc in old["documents"]:
                out["pos_docs"].append({"id": doc["id"], "text": doc["text"]})
            N = len(out["pos_docs"])  # Number of relevant documents
            pos_ids = {
                doc["id"] for doc in out["pos_docs"]
            }  # Set of relevant document ids

            # key: neg_docs (length=N)
            out["neg_docs"] = []
            while len(out["neg_docs"]) < N:
                choice = bm25_runs[out["query"]].pop()
                if choice not in pos_ids:
                    out["neg_docs"].append({"id": choice, "text": all_docs[choice]})

            dataset.append(out)

    print()

    # Save the dataset
    with open("data/train_dataset_v2_own.jsonl", "w") as out_file:
        print(f"Saving the dataset...")
        [out_file.write(json.dumps(entry) + "\n") for entry in dataset]


def with_alt_bm25():
    # Load all documents into memory for negative sampling
    all_docs = {}  # doc_id -> doc_title + doc_abstract
    with gzip.open("data/pubmed_2022_small.jsonl.gz", "r") as all_file:
        for i, line in enumerate(all_file):
            print(f"Loading {i} documents for negative sampling...", end="\r")
            doc = json.loads(line)
            all_docs[doc["pmid"]] = doc["title"] + " " + doc["abstract"]

    print()

    # Load BM25 runs over train dataset questions
    bm25_runs = []  # query_index -> list of doc_ids
    with open("data/BM25_train_alt.jsonl", "r") as bm25_file:
        for i, line in enumerate(bm25_file):
            print(f"Loading {i} BM25 runs...", end="\r")
            data = json.loads(line)
            bm25_runs.append([str(doc_id) for doc_id in data["documents_pmid"]])

    print()

    # Create the train dataset, each entry contains:
    # - 1 query
    # - N positive/relevant documents
    # - N negative/non-relevant documents
    dataset = []
    with open("data/RI_2023_training_data_wContents.jsonl", "r") as pos_file:
        for i, line in enumerate(pos_file):
            print(f"Creating {i} dataset entries...", end="\r")
            old = json.loads(line)
            out = {}

            # key: query
            out["query"] = old["body"]

            # key: pos_docs (length=N)
            out["pos_docs"] = []
            for doc in old["documents"]:
                out["pos_docs"].append({"id": doc["id"], "text": doc["text"]})
            N = len(out["pos_docs"])  # Number of relevant documents
            pos_ids = {
                doc["id"] for doc in out["pos_docs"]
            }  # Set of relevant document ids

            # key: neg_docs (length=N)
            out["neg_docs"] = []
            while len(out["neg_docs"]) < N:
                choice = bm25_runs[i].pop()
                if choice not in pos_ids:
                    out["neg_docs"].append({"id": choice, "text": all_docs[choice]})

            dataset.append(out)

    print()

    # Save the dataset
    with open("data/train_dataset_v2_alt.jsonl", "w") as out_file:
        print(f"Saving the dataset...")
        [out_file.write(json.dumps(entry) + "\n") for entry in dataset]


if __name__ == "__main__":
    # with_rand_docs()
    # with_own_bm25()
    with_alt_bm25()
