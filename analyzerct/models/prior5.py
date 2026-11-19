"""Prior 5: mean of the student's previous 5 scored logs before the target (the original control).

Only used when rct_data has no prior5 column; students with no earlier log get the remnant's mean correctness."""
from . import positions, take

N = 5


def predict(history, target_ids, fill):
    prev = history.groupby("user_id")["correct"].shift().astype(float)
    avg = prev.groupby(history["user_id"]).rolling(N, min_periods=1).mean().reset_index(level=0, drop=True)
    return take(avg.sort_index().fillna(fill).to_numpy(), positions(history, target_ids))
