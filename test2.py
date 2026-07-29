import numpy as np
from tasks.quality import QuALITY


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

# Example usage:
quality = QuALITY('all')
quality.load_attempts_json('/share/goyal/mposey/Synthetic_Continued_Pretraining/out/qualityqa-quality-entigraph-lr5e-06-rr0.1-epochs2-bs16-wd0.01-warmup0.05-MetaLlama38B.json')
quality.majority_vote(n_samples=1)

# Compute correctness list for the test split
test_correctness = compute_correctness_list_from_quality(quality, split="test")

# Output the number of questions and some sample data
print(f"Total questions in test split: {len(test_correctness)}")
print(f"Sample correctness list: {test_correctness[:10]}")
accuracy = np.mean(test_correctness)
print(f"Sample correctness accuracy: {accuracy}")