from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import ARTIFACTS_DIR, PERSON_FEATURES, SITUATION_FEATURES
from .features import (
    build_person_features,
    person_feature_matrix,
    prepare_trials,
    situation_matrix,
)
from .kl_net import KLProbabilisticNet, kl_divergence
from .pnn import ProbabilisticNeuralNetwork, ScaledPNN

N_FOLDS = 50  # one fold per participant (leave-one-subject-out)


def _eval_classifier(model, X_test, y_test, average: str) -> dict:
    X = np.asarray(X_test, dtype=float)
    y = np.asarray(y_test)
    pred = model.predict(X)
    proba = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    class_index = {int(c): i for i, c in enumerate(classes)}
    target = np.zeros_like(proba)
    for i, label in enumerate(y):
        target[i, class_index[int(label)]] = 1.0
    mean_kl = float(kl_divergence(target, proba).mean())
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, average=average, zero_division=0)),
        "mean_kl": mean_kl,
    }


def select_best_model(model_metrics: dict) -> str:
    """Pick the model that works best on CV metrics."""
    return max(
        model_metrics.items(),
        key=lambda item: (
            item[1]["accuracy"],
            item[1]["f1"],
            -item[1].get("mean_kl", 0.0),
        ),
    )[0]


def _fit_svm(calibration_cv: int = 3):
    base = SVC(kernel="rbf", C=2.0, gamma="scale", random_state=42)
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=calibration_cv)
    return Pipeline([("scaler", StandardScaler()), ("clf", calibrated)])


def _subsample(Xs, y, max_per_class: int, rng: np.random.Generator):
    idxs = []
    for cls in np.unique(y):
        cls_idx = np.where(np.asarray(y) == cls)[0]
        take = min(max_per_class, len(cls_idx))
        idxs.append(rng.choice(cls_idx, size=take, replace=False))
    chosen = np.concatenate(idxs)
    return Xs[chosen], np.asarray(y)[chosen]


def _aggregate_folds(fold_metrics: list[dict]) -> dict:
    acc = np.array([m["accuracy"] for m in fold_metrics], dtype=float)
    f1 = np.array([m["f1"] for m in fold_metrics], dtype=float)
    kl = np.array([m["mean_kl"] for m in fold_metrics], dtype=float)
    return {
        "accuracy": float(acc.mean()),
        "accuracy_std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
        "f1": float(f1.mean()),
        "f1_std": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
        "mean_kl": float(kl.mean()),
        "mean_kl_std": float(kl.std(ddof=1)) if len(kl) > 1 else 0.0,
        "n_folds": int(len(fold_metrics)),
    }


def _fit_pnn(X_train, y_train, max_per_class: int, rng: np.random.Generator):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    Xp, yp = _subsample(Xs, y_train, max_per_class=max_per_class, rng=rng)
    model = ProbabilisticNeuralNetwork(sigma=0.9).fit(Xp, yp)
    return ScaledPNN(scaler, model)


def _fit_kl(X_train, y_train, epochs: int):
    return KLProbabilisticNet(
        epochs=epochs, lr=0.08, label_smoothing=0.05, seed=42
    ).fit(X_train, y_train)


def cross_validate_models(
    X: np.ndarray,
    y_deck: np.ndarray,
    y_switch: np.ndarray,
    groups: np.ndarray,
    n_folds: int = N_FOLDS,
) -> dict:
    """Leave-one-participant-out CV across all 50 subjects."""
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_folds:
        raise ValueError(
            f"Need at least {n_folds} participants for {n_folds}-fold LOSO CV; "
            f"found {len(unique_groups)}"
        )

    logo = LeaveOneGroupOut()
    # Fast settings inside folds; final models get full budget later
    fold_scores = {
        "deck": {"svm": [], "pnn": [], "kl_pnn": []},
        "switch": {"svm": [], "pnn": [], "kl_pnn": []},
    }

    for fold_i, (train_idx, test_idx) in enumerate(
        logo.split(X, y_deck, groups), start=1
    ):
        rng = np.random.default_rng(42 + fold_i)
        X_tr, X_te = X[train_idx], X[test_idx]
        yd_tr, yd_te = y_deck[train_idx], y_deck[test_idx]
        ys_tr, ys_te = y_switch[train_idx], y_switch[test_idx]

        svm_deck = _fit_svm(calibration_cv=2).fit(X_tr, yd_tr)
        svm_switch = _fit_svm(calibration_cv=2).fit(X_tr, ys_tr)
        pnn_deck = _fit_pnn(X_tr, yd_tr, max_per_class=250, rng=rng)
        pnn_switch = _fit_pnn(X_tr, ys_tr, max_per_class=350, rng=rng)
        kl_deck = _fit_kl(X_tr, yd_tr, epochs=120)
        kl_switch = _fit_kl(X_tr, ys_tr, epochs=100)

        fold_scores["deck"]["svm"].append(
            _eval_classifier(svm_deck, X_te, yd_te, average="macro")
        )
        fold_scores["deck"]["pnn"].append(
            _eval_classifier(pnn_deck, X_te, yd_te, average="macro")
        )
        fold_scores["deck"]["kl_pnn"].append(
            _eval_classifier(kl_deck, X_te, yd_te, average="macro")
        )
        fold_scores["switch"]["svm"].append(
            _eval_classifier(svm_switch, X_te, ys_te, average="binary")
        )
        fold_scores["switch"]["pnn"].append(
            _eval_classifier(pnn_switch, X_te, ys_te, average="binary")
        )
        fold_scores["switch"]["kl_pnn"].append(
            _eval_classifier(kl_switch, X_te, ys_te, average="binary")
        )

        if fold_i % 10 == 0:
            print(f"CV fold {fold_i}/{len(unique_groups)} complete")

    return {
        "deck": {
            name: _aggregate_folds(scores) for name, scores in fold_scores["deck"].items()
        },
        "switch": {
            name: _aggregate_folds(scores)
            for name, scores in fold_scores["switch"].items()
        },
    }


def train_and_save(artifacts_dir: Path | None = None) -> dict:
    artifacts = Path(artifacts_dir or ARTIFACTS_DIR)
    artifacts.mkdir(parents=True, exist_ok=True)

    trials = prepare_trials()
    usable = trials[trials["trial_number"] > 1].copy()

    X = situation_matrix(usable).to_numpy(dtype=float)
    y_deck = usable["deck_choice"].astype(int).to_numpy()
    y_switch = usable["switched_deck"].astype(int).to_numpy()
    groups = usable["participant_id"].to_numpy()
    n_participants = int(len(np.unique(groups)))

    print(f"Running {n_participants}-fold leave-one-participant-out CV…")
    cv_metrics = cross_validate_models(X, y_deck, y_switch, groups, n_folds=n_participants)

    best_deck = select_best_model(cv_metrics["deck"])
    best_switch = select_best_model(cv_metrics["switch"])
    print(f"Best by CV → deck={best_deck}, switch={best_switch}")

    # Refit final serving models on ALL participants' trials
    rng = np.random.default_rng(42)
    svm_deck = _fit_svm(calibration_cv=3).fit(X, y_deck)
    svm_switch = _fit_svm(calibration_cv=3).fit(X, y_switch)
    pnn_deck_wrapped = _fit_pnn(X, y_deck, max_per_class=400, rng=rng)
    pnn_switch_wrapped = _fit_pnn(X, y_switch, max_per_class=600, rng=rng)
    kl_deck = _fit_kl(X, y_deck, epochs=500)
    kl_switch = _fit_kl(X, y_switch, epochs=400)

    metrics = {
        "deck": cv_metrics["deck"],
        "switch": cv_metrics["switch"],
        "loss": "kl_divergence_for_discrete_distributions",
        "cv": {
            "method": "leave_one_participant_out",
            "n_folds": n_participants,
            "n_participants": n_participants,
            "n_trials_evaluated": int(len(usable)),
            "note": (
                "Each fold trains on 49 participants and tests on 1; "
                "every trial is used for testing exactly once. "
                "Serving models are then refit on all data."
            ),
        },
        "n_train_trials": int(len(usable)),
        "n_test_trials": int(len(usable)),
        "situation_features": SITUATION_FEATURES,
    }

    person_df = build_person_features(trials)
    X_person = person_feature_matrix(person_df)
    person_scaler = StandardScaler()
    X_scaled = person_scaler.fit_transform(X_person)

    gmm = GaussianMixture(
        n_components=3,
        covariance_type="full",
        random_state=42,
        n_init=5,
    )
    cluster_ids = gmm.fit_predict(X_scaled)
    person_df = person_df.copy()
    person_df["cluster"] = cluster_ids

    cluster_stats = (
        person_df.groupby("cluster")
        .agg(
            late_advantageous_rate=("late_advantageous_rate", "mean"),
            late_bad_deck_rate=("late_bad_deck_rate", "mean"),
            mean_net_outcome=("mean_net_outcome", "mean"),
            n_participants=("bad_deck_rate", "count"),
        )
        .reset_index()
        .sort_values("late_advantageous_rate", ascending=False)
    )
    tier_labels = ["A", "B", "C"]
    cluster_to_tier = {
        int(row.cluster): tier_labels[i]
        for i, row in enumerate(cluster_stats.itertuples())
    }
    person_df["tier"] = person_df["cluster"].map(cluster_to_tier)

    tiers_summary = []
    for i, row in enumerate(cluster_stats.itertuples()):
        tiers_summary.append(
            {
                "tier": tier_labels[i],
                "cluster": int(row.cluster),
                "n_participants": int(row.n_participants),
                "late_advantageous_rate": float(row.late_advantageous_rate),
                "late_bad_deck_rate": float(row.late_bad_deck_rate),
                "mean_net_outcome": float(row.mean_net_outcome),
            }
        )

    profiles = {}
    for pid, row in person_df.iterrows():
        profiles[pid] = {
            "participant_id": pid,
            "cluster": int(row["cluster"]),
            "tier": row["tier"],
            "switch_rate": float(row["switch_rate"]),
            "bad_deck_rate": float(row["bad_deck_rate"]),
            "late_bad_deck_rate": float(row["late_bad_deck_rate"]),
            "late_advantageous_rate": float(row["late_advantageous_rate"]),
            "mean_net_outcome": float(row["mean_net_outcome"]),
            "final_cumulative": float(row["final_cumulative"]),
            "loss_sensitivity": float(row["loss_sensitivity"]),
            "deck_prefs": {
                "1": float(row["deck1_rate"]),
                "2": float(row["deck2_rate"]),
                "3": float(row["deck3_rate"]),
                "4": float(row["deck4_rate"]),
            },
        }

    joblib.dump(svm_deck, artifacts / "svm_deck.joblib")
    joblib.dump(svm_switch, artifacts / "svm_switch.joblib")
    joblib.dump(pnn_deck_wrapped, artifacts / "pnn_deck.joblib")
    joblib.dump(pnn_switch_wrapped, artifacts / "pnn_switch.joblib")
    joblib.dump(kl_deck, artifacts / "kl_pnn_deck.joblib")
    joblib.dump(kl_switch, artifacts / "kl_pnn_switch.joblib")
    joblib.dump(gmm, artifacts / "gmm.joblib")
    joblib.dump(person_scaler, artifacts / "person_scaler.joblib")

    for stale in (
        "lr_deck.joblib",
        "lr_switch.joblib",
        "rf_deck.joblib",
        "rf_switch.joblib",
    ):
        path = artifacts / stale
        if path.exists():
            path.unlink()

    selection_rule = "50-fold LOSO CV: max_accuracy then max_f1 then min_mean_kl"
    meta = {
        "metrics": metrics,
        "cluster_to_tier": {str(k): v for k, v in cluster_to_tier.items()},
        "tiers_summary": tiers_summary,
        "profiles": profiles,
        "person_features": PERSON_FEATURES,
        "situation_features": SITUATION_FEATURES,
        "default_deck_model": best_deck,
        "default_switch_model": best_switch,
        "best_models": {
            "deck": best_deck,
            "switch": best_switch,
            "selection_rule": selection_rule,
            "deck_metrics": metrics["deck"][best_deck],
            "switch_metrics": metrics["switch"][best_switch],
        },
        "available_models": ["best", "svm", "pnn", "kl_pnn"],
    }
    with open(artifacts / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


if __name__ == "__main__":
    result = train_and_save()
    print(json.dumps(result["metrics"], indent=2))
    print("Best:", result["best_models"])
    print(f"Saved artifacts to {ARTIFACTS_DIR}")
