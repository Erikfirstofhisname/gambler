from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT_DIR / "igt_real_5000_behavioral_trials.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"

DISADVANTAGEOUS_DECKS = {1, 2}
ADVANTAGEOUS_DECKS = {3, 4}
LATE_TRIAL_START = 51
SITUATION_FEATURES = [
    "trial_number",
    "previous_choice",
    "previous_net_outcome",
    "cumulative_net_outcome_before_trial",
    "recent_loss_count_5",
    "switched_deck_lag",
]
PERSON_FEATURES = [
    "switch_rate",
    "bad_deck_rate",
    "late_bad_deck_rate",
    "mean_net_outcome",
    "loss_sensitivity",
    "deck1_rate",
    "deck2_rate",
    "deck3_rate",
    "deck4_rate",
]
