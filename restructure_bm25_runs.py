import json

TYPE = "E8B2" # E8B1

if __name__ == "__main__":
    query_to_id = dict()

    with open(f"data/question_{TYPE}_gs.jsonl", "r") as gs:
        for line in gs:
            data = json.loads(line)
            query_to_id[data["query_text"]] = data["query_id"]

    with open(f"data/BM25_{TYPE}_old.jsonl", "r") as old:
        with open(f"data/BM25_{TYPE}.jsonl", "w") as new:
            for line in old:
                new_data = dict()
                old_data = json.loads(line)

                new_data["id"] = query_to_id[old_data["query_text"]]
                new_data["question"] = old_data["query_text"]
                new_data["documents"] = [
                    {"id": doc_id, "score": score}
                    for doc_id, score in old_data["scores"].items()
                ]

                new.write(json.dumps(new_data) + "\n")
