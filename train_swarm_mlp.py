import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# WEIGHTED MEAN SWARM MLP
# NumPy-only implementation
# ============================================================

RANDOM_STATE = 20260903

NUM_NODES = 5
LOCAL_EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 128

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PREPROCESSED_DIR = os.path.join(
    BASE_DIR, "preprocessed_data"
)

RESULTS_DIR = os.path.join(
    BASE_DIR, "results"
)

MODELS_DIR = os.path.join(
    BASE_DIR, "models"
)

COMPARISON_DIR = os.path.join(
    BASE_DIR, "comparison"
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(COMPARISON_DIR, exist_ok=True)

rng = np.random.default_rng(RANDOM_STATE)


# ============================================================
# FILE FINDING
# ============================================================

def find_dataset_file(dataset, split, xy):

    files = [
        f for f in os.listdir(PREPROCESSED_DIR)
        if f.lower().endswith(".csv")
    ]

    dataset = dataset.lower()
    split = split.lower()
    xy = xy.lower()

    candidates = []

    for f in files:

        name = f.lower().replace("-", "_")

        if dataset not in name:
            continue

        if split not in name:
            continue

        if xy == "x":
            if (
                "_x_" in name
                or name.startswith("x_")
                or "_x" in name
                or "x_" in name
            ):
                candidates.append(f)

        elif xy == "y":
            if (
                "_y_" in name
                or name.startswith("y_")
                or "_y" in name
                or "y_" in name
            ):
                candidates.append(f)

    if not candidates:

        raise FileNotFoundError(
            f"\nCould not find {xy}_{split} file for "
            f"{dataset}.\n\nAvailable CSV files:\n"
            + "\n".join(files)
        )

    # Prefer the shortest matching filename
    candidates.sort(key=len)

    return os.path.join(
        PREPROCESSED_DIR,
        candidates[0]
    )


def load_dataset(dataset):

    print("\n" + "=" * 70)
    print(f"LOADING {dataset.upper()} DATASET")
    print("=" * 70)

    X_train_path = find_dataset_file(
        dataset, "train", "x"
    )

    y_train_path = find_dataset_file(
        dataset, "train", "y"
    )

    X_test_path = find_dataset_file(
        dataset, "test", "x"
    )

    y_test_path = find_dataset_file(
        dataset, "test", "y"
    )

    print("\nTraining X:", os.path.basename(X_train_path))
    print("Training y:", os.path.basename(y_train_path))
    print("Testing X :", os.path.basename(X_test_path))
    print("Testing y :", os.path.basename(y_test_path))

    X_train = pd.read_csv(
        X_train_path
    )

    y_train = pd.read_csv(
        y_train_path
    )

    X_test = pd.read_csv(
        X_test_path
    )

    y_test = pd.read_csv(
        y_test_path
    )

    # Remove accidental pandas index columns
    X_train = X_train.loc[
        :,
        ~X_train.columns.str.startswith("Unnamed")
    ]

    X_test = X_test.loc[
        :,
        ~X_test.columns.str.startswith("Unnamed")
    ]

    y_train = y_train.loc[
        :,
        ~y_train.columns.str.startswith("Unnamed")
    ]

    y_test = y_test.loc[
        :,
        ~y_test.columns.str.startswith("Unnamed")
    ]

    X_train = X_train.to_numpy(
        dtype=np.float32
    )

    X_test = X_test.to_numpy(
        dtype=np.float32
    )

    y_train = y_train.iloc[:, 0].to_numpy(
        dtype=np.float32
    )

    y_test = y_test.iloc[:, 0].to_numpy(
        dtype=np.float32
    )

    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    print("\nTraining shape:", X_train.shape)
    print("Testing shape :", X_test.shape)

    return (
        X_train,
        y_train,
        X_test,
        y_test
    )


# ============================================================
# MLP FUNCTIONS
# ============================================================

def initialize_parameters(input_size):

    parameters = {

        "W1": (
            rng.normal(
                0,
                np.sqrt(2 / input_size),
                (input_size, 64)
            ).astype(np.float32)
        ),

        "b1": np.zeros(
            (1, 64),
            dtype=np.float32
        ),

        "W2": (
            rng.normal(
                0,
                np.sqrt(2 / 64),
                (64, 32)
            ).astype(np.float32)
        ),

        "b2": np.zeros(
            (1, 32),
            dtype=np.float32
        ),

        "W3": (
            rng.normal(
                0,
                np.sqrt(2 / 32),
                (32, 16)
            ).astype(np.float32)
        ),

        "b3": np.zeros(
            (1, 16),
            dtype=np.float32
        ),

        "W4": (
            rng.normal(
                0,
                np.sqrt(2 / 16),
                (16, 1)
            ).astype(np.float32)
        ),

        "b4": np.zeros(
            (1, 1),
            dtype=np.float32
        )
    }

    return parameters


def relu(x):

    return np.maximum(
        0,
        x
    )


def relu_derivative(x):

    return (
        x > 0
    ).astype(np.float32)


def sigmoid(x):

    x = np.clip(
        x,
        -50,
        50
    )

    return 1 / (
        1 + np.exp(-x)
    )


def forward(X, parameters):

    Z1 = X @ parameters["W1"] + parameters["b1"]
    A1 = relu(Z1)

    Z2 = A1 @ parameters["W2"] + parameters["b2"]
    A2 = relu(Z2)

    Z3 = A2 @ parameters["W3"] + parameters["b3"]
    A3 = relu(Z3)

    Z4 = A3 @ parameters["W4"] + parameters["b4"]
    A4 = sigmoid(Z4)

    cache = {
        "X": X,
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
        "Z3": Z3,
        "A3": A3,
        "A4": A4
    }

    return A4, cache


def train_local_model(
    X,
    y,
    initial_parameters
):

    parameters = {
        key: value.copy()
        for key, value
        in initial_parameters.items()
    }

    n = len(X)

    for epoch in range(
        LOCAL_EPOCHS
    ):

        indices = rng.permutation(n)

        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for start in range(
            0,
            n,
            BATCH_SIZE
        ):

            end = min(
                start + BATCH_SIZE,
                n
            )

            X_batch = X_shuffled[
                start:end
            ]

            y_batch = y_shuffled[
                start:end
            ].reshape(-1, 1)

            # Forward
            predictions, cache = forward(
                X_batch,
                parameters
            )

            batch_n = len(
                X_batch
            )

            # ------------------------------------------------
            # BACKPROPAGATION
            # ------------------------------------------------

            dZ4 = (
                predictions - y_batch
            ) / batch_n

            dW4 = (
                cache["A3"].T @ dZ4
            )

            db4 = np.sum(
                dZ4,
                axis=0,
                keepdims=True
            )

            dA3 = (
                dZ4 @ parameters["W4"].T
            )

            dZ3 = (
                dA3
                * relu_derivative(
                    cache["Z3"]
                )
            )

            dW3 = (
                cache["A2"].T @ dZ3
            )

            db3 = np.sum(
                dZ3,
                axis=0,
                keepdims=True
            )

            dA2 = (
                dZ3 @ parameters["W3"].T
            )

            dZ2 = (
                dA2
                * relu_derivative(
                    cache["Z2"]
                )
            )

            dW2 = (
                cache["A1"].T @ dZ2
            )

            db2 = np.sum(
                dZ2,
                axis=0,
                keepdims=True
            )

            dA1 = (
                dZ2 @ parameters["W2"].T
            )

            dZ1 = (
                dA1
                * relu_derivative(
                    cache["Z1"]
                )
            )

            dW1 = (
                cache["X"].T @ dZ1
            )

            db1 = np.sum(
                dZ1,
                axis=0,
                keepdims=True
            )

            # ------------------------------------------------
            # PARAMETER UPDATE
            # ------------------------------------------------

            parameters["W1"] -= (
                LEARNING_RATE * dW1
            )

            parameters["b1"] -= (
                LEARNING_RATE * db1
            )

            parameters["W2"] -= (
                LEARNING_RATE * dW2
            )

            parameters["b2"] -= (
                LEARNING_RATE * db2
            )

            parameters["W3"] -= (
                LEARNING_RATE * dW3
            )

            parameters["b3"] -= (
                LEARNING_RATE * db3
            )

            parameters["W4"] -= (
                LEARNING_RATE * dW4
            )

            parameters["b4"] -= (
                LEARNING_RATE * db4
            )

    return parameters


# ============================================================
# WEIGHTED MEAN AGGREGATION
# ============================================================

def weighted_mean(
    local_parameters,
    sample_counts
):

    total_samples = sum(
        sample_counts
    )

    aggregation_weights = [
        count / total_samples
        for count in sample_counts
    ]

    print(
        "\nWeighted Mean Aggregation:"
    )

    for i, weight in enumerate(
        aggregation_weights
    ):

        print(
            f"Node {i + 1}: "
            f"{sample_counts[i]} samples "
            f"-> weight = {weight:.4f}"
        )

    global_parameters = {}

    for parameter_name in (
        local_parameters[0].keys()
    ):

        weighted_parameter = np.zeros_like(
            local_parameters[0][
                parameter_name
            ]
        )

        for local_model, weight in zip(
            local_parameters,
            aggregation_weights
        ):

            weighted_parameter += (
                weight
                * local_model[
                    parameter_name
                ]
            )

        global_parameters[
            parameter_name
        ] = weighted_parameter

    return global_parameters


# ============================================================
# METRICS
# ============================================================

def calculate_auc(
    y_true,
    probabilities
):

    y_true = y_true.astype(int)

    positive_scores = probabilities[
        y_true == 1
    ]

    negative_scores = probabilities[
        y_true == 0
    ]

    if (
        len(positive_scores) == 0
        or len(negative_scores) == 0
    ):
        return 0.0

    scores = np.concatenate([
        positive_scores,
        negative_scores
    ])

    ranks = pd.Series(
        scores
    ).rank(
        method="average"
    ).to_numpy()

    positive_ranks = ranks[
        :len(positive_scores)
    ]

    auc = (
        np.sum(positive_ranks)
        - len(positive_scores)
        * (len(positive_scores) + 1)
        / 2
    ) / (
        len(positive_scores)
        * len(negative_scores)
    )

    return float(auc)


def calculate_metrics(
    y_true,
    probabilities
):

    y_true = y_true.astype(int)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    tp = np.sum(
        (y_true == 1)
        & (predictions == 1)
    )

    tn = np.sum(
        (y_true == 0)
        & (predictions == 0)
    )

    fp = np.sum(
        (y_true == 0)
        & (predictions == 1)
    )

    fn = np.sum(
        (y_true == 1)
        & (predictions == 0)
    )

    accuracy = (
        (tp + tn)
        / len(y_true)
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0
    )

    auc = calculate_auc(
        y_true,
        probabilities
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": auc,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    metrics,
    dataset
):

    matrix = np.array([
        [
            metrics["TN"],
            metrics["FP"]
        ],
        [
            metrics["FN"],
            metrics["TP"]
        ]
    ])

    plt.figure(
        figsize=(5.5, 5.5)
    )

    plt.imshow(
        matrix,
        cmap="Blues"
    )

    plt.colorbar()

    plt.title(
    f"Weighted Mean Swarm MLP - "
    f"{dataset.capitalize()} Dataset\n"
    f"Confusion Matrix"
)
    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "Actual Label"
    )

    plt.xticks(
    [0, 1],
    ["Low", "High"]
)

    plt.yticks(
    [0, 1],
    ["Low", "High"]
)

    for i in range(2):

        for j in range(2):

            plt.text(
    j,
    i,
    str(matrix[i, j]),
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold" if i == 0 and j == 0 else "normal"
)

    plt.tight_layout()

    path = os.path.join(
        COMPARISON_DIR,
        f"weighted_mean_swarm_mlp_confusion_matrix_{dataset}.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nConfusion matrix saved:"
    )

    print(path)


# ============================================================
# ACCURACY COMPARISON GRAPH
# ============================================================

def create_comparison_graph(
    swarm_results
):

    existing_path = os.path.join(
        RESULTS_DIR,
        "final_model_comparison.csv"
    )

    if not os.path.exists(
        existing_path
    ):

        print(
            "\nExisting comparison file "
            "not found. Skipping comparison graph."
        )

        return

    existing = pd.read_csv(
        existing_path
    )

    print(
        "\nCreating model comparison graph..."
    )

    # Normalize column names
    existing.columns = [
        c.strip()
        for c in existing.columns
    ]

    # Find required columns
    dataset_col = next(
        (
            c for c in existing.columns
            if c.lower() == "dataset"
        ),
        None
    )

    model_col = next(
        (
            c for c in existing.columns
            if c.lower() == "model"
        ),
        None
    )

    accuracy_col = next(
        (
            c for c in existing.columns
            if c.lower() == "accuracy"
        ),
        None
    )

    if not all([
        dataset_col,
        model_col,
        accuracy_col
    ]):

        print(
            "Could not identify comparison columns."
        )

        return

    old = existing[
        [
            dataset_col,
            model_col,
            accuracy_col
        ]
    ].copy()

    old.columns = [
        "Dataset",
        "Model",
        "Accuracy"
    ]

    new = swarm_results[
        [
            "Dataset",
            "Model",
            "Accuracy"
        ]
    ].copy()

    combined = pd.concat(
        [
            old,
            new
        ],
        ignore_index=True
    )

    comparison_path = os.path.join(
        RESULTS_DIR,
        "final_model_comparison_with_swarm.csv"
    )

    combined.to_csv(
        comparison_path,
        index=False
    )

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    datasets = [
        "mechanistic",
        "probabilistic"
    ]

    models = list(
        combined["Model"].unique()
    )

    # Keep Swarm model at the end
    models = [
        m for m in models
        if m != "Weighted Mean Swarm MLP"
    ]

    models.append(
        "Weighted Mean Swarm MLP"
    )

    x = np.arange(
        len(models)
    )

    width = 0.35

    plt.figure(
        figsize=(13, 7)
    )

    for i, dataset in enumerate(
        datasets
    ):

        values = []

        for model in models:

            rows = combined[
                (
                    combined["Dataset"]
                    .str.lower()
                    == dataset
                )
                &
                (
                    combined["Model"]
                    == model
                )
            ]

            if len(rows) > 0:

                values.append(
                    float(
                        rows.iloc[0][
                            "Accuracy"
                        ]
                    ) * 100
                )

            else:

                values.append(
                    np.nan
                )

        offset = (
            -width / 2
            if i == 0
            else width / 2
        )

        plt.bar(
            x + offset,
            values,
            width,
            label=dataset.capitalize()
        )

    plt.xticks(
        x,
        models,
        rotation=30,
        ha="right"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.title(
        "Accuracy Comparison of ML/DL Models "
        "with Weighted Mean Swarm MLP",
        fontweight="bold"
    )

    plt.legend()

    plt.tight_layout()

    graph_path = os.path.join(
        COMPARISON_DIR,
        "accuracy_comparison_with_weighted_mean_swarm_mlp.png"
    )

    plt.savefig(
        graph_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nComparison graph saved:"
    )

    print(graph_path)

    print(
        "\nComparison table saved:"
    )

    print(comparison_path)


# ============================================================
# TRAIN SWARM MODEL
# ============================================================

def train_swarm(
    dataset
):

    X_train, y_train, X_test, y_test = (
        load_dataset(dataset)
    )

    print(
        "\nCreating "
        f"{NUM_NODES} simulated swarm nodes..."
    )

    # --------------------------------------------------------
    # PARTITION TRAINING DATA
    # --------------------------------------------------------

    indices = rng.permutation(
        len(X_train)
    )

    node_indices = np.array_split(
        indices,
        NUM_NODES
    )

    # --------------------------------------------------------
    # INITIAL GLOBAL MODEL
    # --------------------------------------------------------

    global_parameters = initialize_parameters(
        X_train.shape[1]
    )

    # --------------------------------------------------------
    # LOCAL TRAINING
    # --------------------------------------------------------

    local_parameters = []

    sample_counts = []

    for node_number, node_idx in enumerate(
        node_indices,
        start=1
    ):

        print(
            "\n" + "-" * 65
        )

        print(
            f"TRAINING SWARM NODE {node_number}"
        )

        print(
            "-" * 65
        )

        X_local = X_train[
            node_idx
        ]

        y_local = y_train[
            node_idx
        ]

        print(
            "Local samples:",
            len(X_local)
        )

        parameters = train_local_model(
            X_local,
            y_local,
            global_parameters
        )

        local_parameters.append(
            parameters
        )

        sample_counts.append(
            len(X_local)
        )

        print(
            f"Node {node_number} training complete."
        )

    # --------------------------------------------------------
    # WEIGHTED MEAN
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "PERFORMING WEIGHTED MEAN "
        "SWARM AGGREGATION"
    )

    print(
        "=" * 70
    )

    global_parameters = weighted_mean(
        local_parameters,
        sample_counts
    )

    print(
        "\nGlobal Swarm MLP parameters "
        "successfully aggregated."
    )

    # --------------------------------------------------------
    # FINAL TEST PREDICTION
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "EVALUATING WEIGHTED MEAN SWARM MLP"
    )

    print(
        "=" * 70
    )

    probabilities, _ = forward(
        X_test,
        global_parameters
    )

    probabilities = probabilities.reshape(
        -1
    )

    metrics = calculate_metrics(
        y_test,
        probabilities
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        "\nFINAL TEST RESULTS"
    )

    print(
        f"Accuracy : "
        f"{metrics['Accuracy'] * 100:.4f}%"
    )

    print(
        f"Precision: "
        f"{metrics['Precision'] * 100:.4f}%"
    )

    print(
        f"Recall   : "
        f"{metrics['Recall'] * 100:.4f}%"
    )

    print(
        f"F1 Score : "
        f"{metrics['F1'] * 100:.4f}%"
    )

    print(
        f"ROC-AUC  : "
        f"{metrics['ROC-AUC'] * 100:.4f}%"
    )

    print(
        "\nCONFUSION MATRIX"
    )

    print(
        f"TN = {metrics['TN']}    "
        f"FP = {metrics['FP']}"
    )

    print(
        f"FN = {metrics['FN']}    "
        f"TP = {metrics['TP']}"
    )

    # --------------------------------------------------------
    # SAVE MODEL PARAMETERS
    # --------------------------------------------------------

    model_path = os.path.join(
        MODELS_DIR,
        f"weighted_mean_swarm_mlp_{dataset}.npz"
    )

    np.savez(
        model_path,
        **global_parameters
    )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    predictions_path = os.path.join(
        RESULTS_DIR,
        f"weighted_mean_swarm_mlp_{dataset}_predictions.csv"
    )

    predictions_df = pd.DataFrame({

        "Actual":
            y_test.astype(int),

        "Probability":
            probabilities,

        "Predicted":
            (
                probabilities >= 0.5
            ).astype(int)
    })

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    results = pd.DataFrame([{

        "Model":
            "Weighted Mean Swarm MLP",

        "Dataset":
            dataset,

        "Type":
            "Deep Learning",

        "Accuracy":
            metrics["Accuracy"],

        "Precision":
            metrics["Precision"],

        "Recall":
            metrics["Recall"],

        "F1":
            metrics["F1"],

        "ROC-AUC":
            metrics["ROC-AUC"],

        "TN":
            metrics["TN"],

        "FP":
            metrics["FP"],

        "FN":
            metrics["FN"],

        "TP":
            metrics["TP"]
    }])

    results_path = os.path.join(
        RESULTS_DIR,
        f"weighted_mean_swarm_mlp_{dataset}_results.csv"
    )

    results.to_csv(
        results_path,
        index=False
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    save_confusion_matrix(
        metrics,
        dataset
    )

    print(
        "\nSaved model:"
    )

    print(model_path)

    print(
        "\nSaved predictions:"
    )

    print(predictions_path)

    print(
        "\nSaved metrics:"
    )

    print(results_path)

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n" + "=" * 75
    )

    print(
        "WEIGHTED MEAN SWARM MLP"
    )

    print(
        "NumPy-only implementation"
    )

    print(
        "=" * 75
    )

    all_results = []

    # --------------------------------------------------------
    # MECHANISTIC
    # --------------------------------------------------------

    try:

        result = train_swarm(
            "mechanistic"
        )

        all_results.append(
            result
        )

    except Exception as e:

        print(
            "\nERROR in mechanistic dataset:"
        )

        print(
            repr(e)
        )

    # --------------------------------------------------------
    # PROBABILISTIC
    # --------------------------------------------------------

    try:

        result = train_swarm(
            "probabilistic"
        )

        all_results.append(
            result
        )

    except Exception as e:

        print(
            "\nERROR in probabilistic dataset:"
        )

        print(
            repr(e)
        )

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    if all_results:

        final_results = pd.concat(
            all_results,
            ignore_index=True
        )

        final_path = os.path.join(
            RESULTS_DIR,
            "weighted_mean_swarm_mlp_results.csv"
        )

        final_results.to_csv(
            final_path,
            index=False
        )

        print(
            "\n" + "=" * 75
        )

        print(
            "SWARM MLP TRAINING COMPLETE"
        )

        print(
            "=" * 75
        )

        print(
            "\nFINAL RESULTS:"
        )

        print(
            final_results[
                [
                    "Model",
                    "Dataset",
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "ROC-AUC"
                ]
            ].to_string(
                index=False
            )
        )

        print(
            "\nCombined results saved:"
        )

        print(
            final_path
        )

        # ----------------------------------------------------
        # COMPARISON GRAPH
        # ----------------------------------------------------

        create_comparison_graph(
            final_results
        )

    else:

        print(
            "\nNo Swarm MLP results were generated."
        )