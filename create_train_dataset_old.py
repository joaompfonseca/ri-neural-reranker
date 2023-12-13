import gzip
import json
import random


if __name__ == "__main__":
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
            pos_ids = {doc["id"] for doc in out["pos_docs"]} # Set of relevant document ids
            
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
