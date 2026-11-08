from __future__ import annotations

import pandas as pd

from .config import (
    ADVANTAGEOUS_DECKS,
    CSV_PATH,
    DISADVANTAGEOUS_DECKS,
    LATE_TRIAL_START,
    PERSON_FEATURES,
    SITUATION_FEATURES,
)


def load_raw_trials(csv_path=None) -> pd.DataFrame:
    path = csv_path or CSV_PATH
    df = pd.read_csv(path)
    return df


def prepare_trials(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = load_raw_trials()

    out = df.copy()
    out["deck_choice"] = out["deck_choice"].astype(int)
    out["trial_number"] = out["trial_number"].astype(int)
    out["reward"] = out["reward"].astype(float)
    out["loss"] = out["loss"].astype(float)
    out["net_outcome"] = out["net_outcome"].astype(float)
    out["cumulative_net_outcome_before_trial"] = out[
        "cumulative_net_outcome_before_trial"
    ].astype(float)
    out["recent_loss_count_5"] = out["recent_loss_count_5"].astype(float)

    out["previous_choice"] = out["previous_choice"].fillna(0).astype(float)
    out["previous_net_outcome"] = out["previous_net_outcome"].fillna(0).astype(float)
    out["switched_deck"] = out["switched_deck"].fillna(0).astype(float)

    # Lagged switch: whether the *previous* trial was a switch (0 on first two trials)
    out["switched_deck_lag"] = (
        out.groupby("participant_id")["switched_deck"].shift(1).fillna(0)
    )

    out["is_suboptimal"] = out["deck_choice"].isin(DISADVANTAGEOUS_DECKS).astype(int)
    out["is_advantageous"] = out["deck_choice"].isin(ADVANTAGEOUS_DECKS).astype(int)
    out["is_late"] = (out["trial_number"] >= LATE_TRIAL_START).astype(int)
    out["after_loss"] = (out["previous_net_outcome"] < 0).astype(int)
    out["after_win"] = (out["previous_net_outcome"] > 0).astype(int)
    out["trial_block"] = ((out["trial_number"] - 1) // 20) + 1  # 1..5

    return out


def situation_matrix(trials: pd.DataFrame) -> pd.DataFrame:
    return trials[SITUATION_FEATURES].astype(float)


def build_person_features(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, g in trials.groupby("participant_id"):
        late = g[g["is_late"] == 1]
        after_loss = g[g["after_loss"] == 1]
        loss_sensitivity = (
            float(after_loss["switched_deck"].mean()) if len(after_loss) else 0.0
        )
        deck_counts = g["deck_choice"].value_counts(normalize=True)
        rows.append(
            {
                "participant_id": pid,
                "switch_rate": float(g["switched_deck"].mean()),
                "bad_deck_rate": float(g["is_suboptimal"].mean()),
                "late_bad_deck_rate": float(late["is_suboptimal"].mean())
                if len(late)
                else float(g["is_suboptimal"].mean()),
                "late_advantageous_rate": float(late["is_advantageous"].mean())
                if len(late)
                else float(g["is_advantageous"].mean()),
                "mean_net_outcome": float(g["net_outcome"].mean()),
                "final_cumulative": float(
                    g["cumulative_net_outcome_before_trial"].iloc[-1]
                    + g["net_outcome"].iloc[-1]
                ),
                "loss_sensitivity": loss_sensitivity,
                "deck1_rate": float(deck_counts.get(1, 0.0)),
                "deck2_rate": float(deck_counts.get(2, 0.0)),
                "deck3_rate": float(deck_counts.get(3, 0.0)),
                "deck4_rate": float(deck_counts.get(4, 0.0)),
                "n_trials": int(len(g)),
            }
        )
    return pd.DataFrame(rows).set_index("participant_id")


def person_feature_matrix(person_df: pd.DataFrame) -> pd.DataFrame:
    return person_df[PERSON_FEATURES].astype(float)


def bias_overview(trials: pd.DataFrame) -> dict:
    by_block = (
        trials.groupby("trial_block")["is_suboptimal"]
        .mean()
        .reset_index()
        .rename(columns={"is_suboptimal": "bad_deck_rate"})
    )
    after_loss_rate = float(trials.loc[trials["after_loss"] == 1, "is_suboptimal"].mean())
    after_win_rate = float(trials.loc[trials["after_win"] == 1, "is_suboptimal"].mean())
    late_rate = float(trials.loc[trials["is_late"] == 1, "is_suboptimal"].mean())
    early_rate = float(trials.loc[trials["is_late"] == 0, "is_suboptimal"].mean())

    # High-risk situations: combinations with elevated suboptimal rate
    situations = []
    for label, mask in [
        ("after_loss", trials["after_loss"] == 1),
        ("after_win", trials["after_win"] == 1),
        ("late_trials", trials["is_late"] == 1),
        ("early_trials", trials["is_late"] == 0),
        ("recent_losses_ge_2", trials["recent_loss_count_5"] >= 2),
        ("switched_last_trial", trials["switched_deck_lag"] == 1),
    ]:
        subset = trials.loc[mask]
        if len(subset) == 0:
            continue
        situations.append(
            {
                "situation": label,
                "n_trials": int(len(subset)),
                "bad_deck_rate": float(subset["is_suboptimal"].mean()),
            }
        )
    situations.sort(key=lambda x: x["bad_deck_rate"], reverse=True)

    return {
        "by_trial_block": by_block.to_dict(orient="records"),
        "after_loss_bad_deck_rate": after_loss_rate,
        "after_win_bad_deck_rate": after_win_rate,
        "late_bad_deck_rate": late_rate,
        "early_bad_deck_rate": early_rate,
        "overall_bad_deck_rate": float(trials["is_suboptimal"].mean()),
        "high_risk_situations": situations,
    }


def participant_bias(trials: pd.DataFrame, participant_id: str) -> dict:
    g = trials[trials["participant_id"] == participant_id].sort_values("trial_number")
    if g.empty:
        raise KeyError(participant_id)

    timeline = [
        {
            "trial_number": int(row.trial_number),
            "deck_choice": int(row.deck_choice),
            "is_suboptimal": int(row.is_suboptimal),
            "net_outcome": float(row.net_outcome),
            "after_loss": int(row.after_loss),
            "cumulative_before": float(row.cumulative_net_outcome_before_trial),
        }
        for row in g.itertuples()
    ]

    mistakes = g[g["is_suboptimal"] == 1]
    contexts = []
    for row in mistakes.itertuples():
        contexts.append(
            {
                "trial_number": int(row.trial_number),
                "previous_choice": float(row.previous_choice),
                "previous_net_outcome": float(row.previous_net_outcome),
                "recent_loss_count_5": float(row.recent_loss_count_5),
                "after_loss": int(row.after_loss),
                "is_late": int(row.is_late),
                "deck_choice": int(row.deck_choice),
            }
        )

    # Top situations where this person errs
    situation_rates = []
    for label, mask in [
        ("after_loss", g["after_loss"] == 1),
        ("after_win", g["after_win"] == 1),
        ("late_trials", g["is_late"] == 1),
        ("recent_losses_ge_2", g["recent_loss_count_5"] >= 2),
    ]:
        subset = g.loc[mask]
        if len(subset) < 3:
            continue
        situation_rates.append(
            {
                "situation": label,
                "n_trials": int(len(subset)),
                "bad_deck_rate": float(subset["is_suboptimal"].mean()),
            }
        )
    situation_rates.sort(key=lambda x: x["bad_deck_rate"], reverse=True)

    return {
        "participant_id": participant_id,
        "mistake_count": int(g["is_suboptimal"].sum()),
        "bad_deck_rate": float(g["is_suboptimal"].mean()),
        "late_bad_deck_rate": float(g.loc[g["is_late"] == 1, "is_suboptimal"].mean()),
        "timeline": timeline,
        "mistake_contexts": contexts,
        "top_error_situations": situation_rates,
    }
