import json
import math
import os


class Evaluator:
    def __init__(self, questions_file, results_file, output_folder):
        self.questions_file = questions_file
        self.results_file = results_file
        self.output_folder = output_folder

        # Load questions
        self.questions = {}
        for q in [json.loads(l) for l in open(self.questions_file)]:
            self.questions[q["query_text"]] = {"docs": q["documents_pmid"]}

        # Load results
        self.results = {}
        for r in [json.loads(l) for l in open(self.results_file)]:
            self.results[r["question"]] = {
                "docs": [doc["id"] for doc in r["documents"]],
            }

        # Create output folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def precision(self, question_docs: list, result_docs: list):
        """
        Fraction of retrieved documents that are relevant
        """
        q_set = set(question_docs)
        r_set = set(result_docs)

        tp = len(q_set.intersection(r_set))  # n docs (relevant, retrieved)
        fp = len(r_set.difference(q_set))  # n docs (not relevant, retrieved)

        if (tp + fp) == 0:
            return 0.0

        return tp / (tp + fp)

    def recall(self, question_docs: list, result_docs: list):
        """
        Fraction of relevant documents that are retrieved
        """
        q_set = set(question_docs)
        r_set = set(result_docs)

        tp = len(q_set.intersection(r_set))  # n docs (relevant, retrieved)
        fn = len(q_set.difference(r_set))  # n docs (relevant, not retrieved)

        if (tp + fn) == 0:
            return 0.0

        return tp / (tp + fn)

    def average_precision(self, question_docs: list, result_docs: list):
        """
        Considering the lists of documents are ordered by the ranking
        - Calculate the precision of the first k = 1, ..., R retrieved documents, where R = len(result_docs).
        - Consider only the precisions where the document k is relevant, for each k.
        - Sum the precision values and divide by Q = len(question_docs), the number of relevant documents.
        """

        R = len(result_docs)
        precisions = []
        for k in range(1, R + 1):
            if k > len(question_docs):
                break  # No more relevant documents
            if result_docs[k - 1] == question_docs[k - 1]:  # Document k is relevant
                precisions += [self.precision(question_docs[:k], result_docs[:k])]

        Q = len(question_docs)
        if Q == 0:
            return 0.0

        return sum(precisions) / Q

    def mean_average_precision(self, question_docs: list, result_docs: list, k: int):
        """
        Considering the lists of documents are ordered by the ranking
        - Calculate the average precision of the first k = 1, ..., R retrieved documents, where R = len(result_docs).
        - Return the average precision values for each k.
        """
        R = len(result_docs)
        ap_values = []
        for k in range(1, R + 1):
            ap_values += [self.average_precision(question_docs, result_docs[:k])]
        return sum(ap_values) / R

    def discounted_cumulative_gain(
        self, question_docs: list, result_docs: list, k: int
    ):
        """
        Considering the lists of documents are ordered by the ranking
        - The relevance of a document is 1 if it is relevant, 0 otherwise.
        - Calculate the DCG of the first k = 1, ..., R retrieved documents, where R = len(result_docs).
        - Return the DCG value for k.
        """
        R = len(result_docs)

        dcg_values = []

        for k in range(1, R + 1):
            relevance_k = 1 if result_docs[k - 1] in question_docs else 0
            if k == 1:
                dcg_values += [relevance_k]
            else:
                dcg_values += [relevance_k / (math.log(k, 2))]

        dcg = [sum(dcg_values[: k + 1]) for k in range(R)]  # DCG is a cumulative sum

        if k > len(dcg):
            return dcg[-1]
        else:
            return dcg[k - 1]

    def normalized_discounted_cumulative_gain(
        self, question_docs: list, result_docs: list, k: int
    ):
        """
        Considering the lists of documents are ordered by the ranking
        - The relevance of a document is 1 if it is relevant, 0 otherwise.
        - Calculate the NDCG of the first k = 1, ..., R retrieved documents, where R = len(result_docs).
        - Return the NDCG value for k.
        """
        dcg = self.discounted_cumulative_gain(question_docs, result_docs, k)
        idcg = self.discounted_cumulative_gain(question_docs, question_docs, k)
        return dcg / idcg

    def mean_reciprocal_rank(self, question_docs: list, result_docs: list, k: int):
        """
        Considering the lists of documents are ordered by the ranking
        - Calculate the reciprocal rank of the first k = 1, ..., R retrieved documents, where R = len(result_docs).
        - Return the reciprocal rank value for k.
        """
        R = len(result_docs)

        for k in range(1, R + 1):
            if result_docs[k - 1] in question_docs:
                return 1 / k

        return 0.0

    def evaluate(self):
        query_header = [
            "query",
            "map@10",  # Mean Average Precision
            "recall@10",  # Recall
            "ndcg@100",  # Normalized Discounted Cumulative Gain
            "mrr@10",  # Mean Reciprocal Rank
        ]

        query_data = []

        for query in self.results.keys():
            d = [query]

            q_docs = self.questions[query]["docs"]
            r_docs = self.results[query]["docs"]

            d += [self.mean_average_precision(q_docs, r_docs, 10)]
            d += [self.recall(q_docs, r_docs[:10])]
            d += [self.normalized_discounted_cumulative_gain(q_docs, r_docs, 100)]
            d += [self.mean_reciprocal_rank(q_docs, r_docs, 10)]

            query_data.append(d)

        global_header = [
            "avg_map@10",
            "avg_recall@10",
            "avg_ndcg@100",
            "avg_mrr@10",
        ]

        global_data = []
        # Average metrics
        for i in range(1, len(query_header)):  # Skip query column
            global_data += [sum([d[i] for d in query_data]) / len(query_data)]

        self.save_metrics(query_header, query_data, global_header, global_data)

    def save_metrics(self, query_header, query_data, global_header, global_data):
        # Per query results in CSV
        path = os.path.join(
            self.output_folder,
            f"{self.results_file.split('/')[-1].split('.')[0]}.csv",
        )
        with open(path, "w") as f:
            f.write(";".join(query_header) + "\n")
            for d in query_data:
                f.write(";".join(map(str, d)) + "\n")

        # Per query results in MD
        path = os.path.join(
            self.output_folder,
            f"{self.results_file.split('/')[-1].split('.')[0]}.md",
        )
        with open(path, "w") as f:
            f.write("| " + " | ".join(query_header) + " |\n")
            f.write("| " + " | ".join(["---"] * len(query_header)) + " |\n")
            for d in query_data:
                f.write("| " + " | ".join(map(str, d)) + " |\n")

        # Global results in CSV
        path = os.path.join(
            self.output_folder,
            f"global_{self.results_file.split('/')[-1].split('.')[0]}.csv",
        )
        with open(path, "w") as f:
            f.write(";".join(global_header) + "\n")
            f.write(";".join(map(str, global_data)) + "\n")

        # Global results in MD
        path = os.path.join(
            self.output_folder,
            f"global_{self.results_file.split('/')[-1].split('.')[0]}.md",
        )
        with open(path, "w") as f:
            f.write("| " + " | ".join(global_header) + " |\n")
            f.write("| " + " | ".join(["---"] * len(global_header)) + " |\n")
            f.write("| " + " | ".join(map(str, global_data)) + " |\n")


if __name__ == "__main__":
    # Evaluate BM25 results
    e = Evaluator(
        "data/question_E8B1_gs.jsonl",
        "data/BM25_E8B1.jsonl",
        "eval/BM25_E8B1",
    )
    e.evaluate()

    e = Evaluator(
        "data/question_E8B2_gs.jsonl",
        "data/BM25_E8B2.jsonl",
        "eval/BM25_E8B2",
    )
    e.evaluate()

    # Evaluate rerank results
    e = Evaluator(
        "data/question_E8B1_gs.jsonl",
        "data/BM25_E8B1_rerank.jsonl",
        "eval/BM25_E8B1",
    )
    e.evaluate()

    e = Evaluator(
        "data/question_E8B2_gs.jsonl",
        "data/BM25_E8B2_rerank.jsonl",
        "eval/BM25_E8B2",
    )
    e.evaluate()
