import json
from collections import Counter

from tasks.quality import QuALITY

def evaluate_answers(data):
    results = []  # Store whether each question is correct or not

    for entry in data:
        for question in entry.get("questions", []):
            # Get the correct answer and attempts
            correct_answer = question.get("answer")
            attempts = question.get("attempts", [])

            if not attempts:
                # If there are no attempts, skip this question
                # print(entry['title'])
                # print(question['statement'])
                # results.append(0)  # Assume incorrect if no data
                continue

            # Count the frequency of each answer index
            answer_counts = Counter(attempt["answer_index"] for attempt in attempts)

            # Determine the majority answer index
            majority_answer_index, _ = answer_counts.most_common(1)[0]

            # Check if the correct answer matches the majority answer
            is_correct = (ord(correct_answer) - ord('A')) == majority_answer_index

            # Append result (convert True/False to 1/0)
            results.append(1 if is_correct else 0)

    return results

def compute_correctness_list_from_quality(quality, split="test"):
    """
    Extracts a binary correctness list (1 for correct, 0 for incorrect) for all questions in the specified split
    using QuALITY's performance stats.

    Parameters:
        quality: QuALITY
            The QuALITY task object loaded with attempts.
        split: str
            The dataset split to compute correctness for ('test', 'val', etc.).

    Returns:
        List[int]: A binary list where each entry corresponds to a question's correctness.
    """
    correctness = []

    # Retrieve the appropriate split documents
    total_documents = len(quality.documents)
    if split == "val":
        documents = quality.documents[:int(0.2 * total_documents)]
    elif split == "test":
        documents = quality.documents[int(0.2 * total_documents):]
    else:
        raise ValueError(f"Unsupported split: {split}")

    # Compute correctness for each question in the split
    for document in documents:
        for question in document.questions:
            try:
                # Use the majority vote attempt to decide correctness
                correctness.append(1 if question.iscorrect() else 0)
            except IndexError:
                # If no attempt exists, consider the question incorrect
                correctness.append(0)
    
    return correctness

import numpy as np
from typing import List, Callable

def paired_bootstrap_test(
    model_a_scores: List[int],
    model_b_scores: List[int],
    metric_fn: Callable[[List[int], List[int]], float],
    num_samples: int = 10000
) -> float:
    """
    Perform a paired bootstrap test based on the methodology in the paper.

    Args:
        model_a_scores (List[int]): Scores from model A on the test set (1 = correct, 0 = incorrect).
        model_b_scores (List[int]): Scores from model B on the test set (1 = correct, 0 = incorrect).
        metric_fn (Callable[[List[int], List[int]], float]): Metric function to compute \(\delta(x)\).
        num_samples (int): Number of bootstrap samples to draw.

    Returns:
        float: Estimated p-value for the null hypothesis (\(\delta(x) <= 0\)).
    """
    assert len(model_a_scores) == len(model_b_scores), "Score lists must be of equal length."
    n = len(model_a_scores)
    
    # Compute the observed metric gain (delta)
    observed_delta = metric_fn(model_a_scores, model_b_scores)
    
    # Generate bootstrap samples and calculate deltas
    bootstrap_deltas = []
    for _ in range(num_samples):
        indices = np.random.choice(n, n, replace=True)  # Sample indices with replacement
        resampled_a = [model_a_scores[i] for i in indices]
        resampled_b = [model_b_scores[i] for i in indices]
        bootstrap_deltas.append(metric_fn(resampled_a, resampled_b))
    
    # Compute the p-value as the proportion of \(\delta(x^i) < 0\)
    p_value = sum(delta < 0 for delta in bootstrap_deltas) / num_samples
    return p_value

def accuracy_diff(a_scores: List[int], b_scores: List[int]) -> float:
    """
    Compute the accuracy difference between two sets of scores.

    Args:
        a_scores (List[int]): Scores from model A.
        b_scores (List[int]): Scores from model B.

    Returns:
        float: Difference in accuracy (\(\text{mean}(a) - \text{mean}(b)\)).
    """
    # print(f"Accuracy score: {np.mean(a_scores)}")
    # print(f"Accuracy score: {np.mean(b_scores)}")
    # print(np.mean(a_scores))
    return np.mean(a_scores) - np.mean(b_scores)

# Example Instance-Level Scores (Replace with your actual data)

rephrase_summ_path = "/share/goyal/mposey/Synthetic_Continued_Pretraining/out/qualityqa-quality-entigraph-gpt-4o-lr5e-06-rr0.1-epochs2-bs16-wd0.01-warmup0.05-MetaLlama38Bh.json"
entigraph_path = "/share/goyal/mposey/Synthetic_Continued_Pretraining/out/qualityqa-checkpoint-2440.json"

quality = QuALITY('all')
quality.load_attempts_json(rephrase_summ_path)
quality.majority_vote(n_samples=1)

rephrase_summ_cpt_scores = compute_correctness_list_from_quality(quality, split="test")

quality.load_attempts_json(entigraph_path)
quality.majority_vote(n_samples=1)

enti_graph_scores = compute_correctness_list_from_quality(quality, split="test")

# with open(rephrase_summ_path, "r") as file:
#     rephrase_summ = json.load(file)

# with open(entigraph_path, "r") as file:
#     entigraph = json.load(file)

# Simulate instance-level scores for three models
np.random.seed(6)  # Set random seed for reproducibility
# enti_graph_scores = evaluate_answers(entigraph)
# rephrase_summ_cpt_scores = evaluate_answers(rephrase_summ)
print(f"Accuracy score: {np.mean(enti_graph_scores):.4f}")
print(f"Accuracy score: {np.mean(rephrase_summ_cpt_scores):.4f}")
# summ_cpt_scores = np.random.choice([1, 0], size=num_questions, p=[0.4524, 0.5476])
print(len(enti_graph_scores))
print(len(rephrase_summ_cpt_scores))
# Perform Pairwise Bootstrap Tests
model_pairs = [("EntiGraph", enti_graph_scores), 
               ("Rephrase Summ CPT", rephrase_summ_cpt_scores)]

results = {}
for i, (name_a, scores_a) in enumerate(model_pairs):
    for name_b, scores_b in model_pairs[i + 1:]:
        p_value = paired_bootstrap_test(
            model_a_scores=scores_a,
            model_b_scores=scores_b,
            metric_fn=accuracy_diff,
            num_samples=10000
        )
        observed_diff = accuracy_diff(scores_a, scores_b)
        results[(name_a, name_b)] = (observed_diff, p_value)

# Print Results
print("Pairwise Bootstrap Test Results:")
print("================================")
for (model_a, model_b), (observed_diff, p_value) in results.items():
    significance = "Significant" if p_value < 0.05 else "Not Significant"
    print(f"{model_a} vs. {model_b}:")
    print(f"  Observed Difference: {observed_diff:.4f}")
    print(f"  P-value: {p_value:.4f} ({significance})")
