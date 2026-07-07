import numpy as np


def save_output(output_path, **values):
    np.savez_compressed(
        output_path,
        **values,
    )

    print("Saved output to:", output_path)


def load_output(output_path):
    loaded = np.load(output_path)

    return {
        key: loaded[key]
        for key in loaded.files
    }


def save_test_outputs(
    output_path,
    scores,
    predictions,
    labels,
    threshold,
):
    save_output(
        output_path,
        scores=np.asarray(scores, dtype=np.float32),
        predictions=np.asarray(predictions, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        threshold=np.asarray(threshold, dtype=np.float32),
    )


def load_test_outputs(output_path):
    results = load_output(output_path)

    return {
        "scores": np.asarray(results["scores"]).reshape(-1),
        "predictions": np.asarray(results["predictions"]).astype(int).reshape(-1),
        "labels": np.asarray(results["labels"]).astype(int).reshape(-1),
        "threshold": float(np.asarray(results["threshold"]).reshape(-1)[0]),
    }

def check_same_labels(first_results, second_results):
    if not np.array_equal(
        first_results["labels"],
        second_results["labels"],
    ):
        raise ValueError(
            "The model outputs are not aligned. "
            "Both models must use the same test-record order."
        )


def get_mutual_correct_indices(
    labels,
    anoformer_predictions,
    tranad_predictions,
    target_label,
):
    return np.where(
        (labels == target_label)
        & (anoformer_predictions == target_label)
        & (tranad_predictions == target_label)
    )[0]


def select_mutual_samples(
    labels,
    anoformer_scores,
    anoformer_predictions,
    tranad_scores,
    tranad_predictions,
    target_label,
    n_samples,
    selection_method = "random",
    seed=42,
    exclude_indices=None,
):
    candidates = get_mutual_correct_indices(
        labels=labels,
        anoformer_predictions=anoformer_predictions,
        tranad_predictions=tranad_predictions,
        target_label=target_label,
    )

    if exclude_indices is not None:
        exclude_indices = set(
            int(index)
            for index in exclude_indices
        )

        candidates = np.asarray([
            index
            for index in candidates
            if int(index) not in exclude_indices
        ])

    if len(candidates) < n_samples:
        raise ValueError(
            f"Only {len(candidates)} mutual correct samples found "
            f"for label {target_label}, but {n_samples} were requested."
        )

    if selection_method == "random":
        rng = np.random.default_rng(seed)

        selected_indices = rng.choice(
            candidates,
            size=n_samples,
            replace=False,
        )

        return selected_indices.astype(int)

    if selection_method == "representative":
        candidate_anoformer_scores = anoformer_scores[candidates]
        candidate_tranad_scores = tranad_scores[candidates]

        anoformer_distance = np.abs(
            (candidate_anoformer_scores - candidate_anoformer_scores.mean())
            / (candidate_anoformer_scores.std() + 1e-12)
        )

        tranad_distance = np.abs(
            (candidate_tranad_scores - candidate_tranad_scores.mean())
            / (candidate_tranad_scores.std() + 1e-12)
        )

        combined_distance = anoformer_distance + tranad_distance

        sorted_positions = np.argsort(combined_distance)

        selected_indices = candidates[
            sorted_positions[:n_samples]
        ]

        return selected_indices.astype(int)

    raise ValueError(
        "selection_method must be either 'random' or 'representative'."
    )



def load_samples_from_dataset(
    sample_path,
    test_dataset,
    device=None,
):
    saved_samples = load_output(sample_path)

    selected_indices = saved_samples["selected_indices"]
    case_types = saved_samples["case_types"]

    samples = []

    for index, case_type in zip(selected_indices, case_types):
        ecg, label = test_dataset[int(index)]

        ecg = ecg.unsqueeze(0)

        if device is not None:
            ecg = ecg.to(device)

        samples.append({
            "index": int(index),
            "case_type": str(case_type),
            "label": int(label.item()),
            "ecg": ecg,
        })

    return samples