import pandas as pd


def build_leaderboard(results):
    df = pd.DataFrame(results)
    df = df.sort_values(by="Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)
    return df
