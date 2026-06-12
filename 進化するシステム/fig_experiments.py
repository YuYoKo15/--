from pathlib import Path
from joblib import Parallel, delayed
import pandas as pd
import matplotlib.pyplot as plt
import json
import time

from rumor_model import (
    RumorSimulation_fast,
    ADVISOR,
    LIAR,
    LIKE_TFT,
    CONDITIONAL_ADVISOR,
)


# =========================
# 共通設定
# =========================

N = 40


def make_task_key(fig_name, T, r, freq, rep, seed):
    return (
        f"{fig_name}"
        f"__T={int(T)}"
        f"__r={int(r)}"
        f"__freq={float(freq):.3f}"
        f"__rep={int(rep)}"
        f"__seed={int(seed)}"
    )


def make_initial_counts(target_strategy, opponent_strategy, target_freq, N=40):
    target_count = int(round(N * target_freq))
    opponent_count = N - target_count

    if target_count < 0 or opponent_count < 0:
        raise ValueError("Invalid frequency")

    return {
        target_strategy: target_count,
        opponent_strategy: opponent_count,
    }


def judge_winner(result, target_name, opponent_name):
    winners = result["winner"]

    if winners == [opponent_name]:
        return opponent_name

    if winners == [target_name]:
        return target_name

    if target_name in winners and opponent_name in winners:
        return "TIE"

    if opponent_name in winners and target_name not in winners:
        return opponent_name

    if target_name in winners and opponent_name not in winners:
        return target_name

    return "OTHER"


def run_one_trial_task(task):
    fig_name = task["fig_name"]
    T = int(task["T"])
    r = int(task["r"])
    g = int(task["g"])
    seed = int(task["seed"])
    rep = int(task["rep"])
    target_freq = float(task["target_freq"])
    max_generations = int(task["max_generations"])

    target_strategy = task["target_strategy"]
    opponent_strategy = task["opponent_strategy"]

    initial_counts = make_initial_counts(
        target_strategy=target_strategy,
        opponent_strategy=opponent_strategy,
        target_freq=target_freq,
        N=N,
    )

    sim = RumorSimulation_fast(
        initial_counts=initial_counts,
        T=T,
        g=g,
        r=r,
        seed=seed,
    )

    result = sim.run_until_convergence(
        max_generations=max_generations,
        record=False,
    )

    target_name = target_strategy.name
    opponent_name = opponent_strategy.name

    outcome = judge_winner(
        result=result,
        target_name=target_name,
        opponent_name=opponent_name,
    )

    return {
        "task_key": task["task_key"],
        "fig_name": fig_name,
        "T": T,
        "r": r,
        "g": g,
        "rep": rep,
        "seed": seed,
        "target_strategy": target_name,
        "opponent_strategy": opponent_name,
        "target_freq": target_freq,
        "winner": outcome,
        "converged": result["converged"],
        "generation": result["generation"],
        "wg": T * g / (N * (N - 1) / 2),
        "r_over_g": r / g if g != 0 else None,
    }


def build_tasks(
    fig_name,
    target_strategy,
    opponent_strategy,
    T_values,
    r_values,
    freq_values,
    g,
    repeat,
    max_generations,
    base_seed,
):
    tasks = []
    task_id = 0

    for T in T_values:
        for r in r_values:
            for freq in freq_values:
                for rep in range(repeat):
                    seed = base_seed + task_id
                    task_key = make_task_key(
                        fig_name=fig_name,
                        T=T,
                        r=r,
                        freq=freq,
                        rep=rep,
                        seed=seed,
                    )

                    tasks.append({
                        "task_id": task_id,
                        "task_key": task_key,
                        "fig_name": fig_name,
                        "target_strategy": target_strategy,
                        "opponent_strategy": opponent_strategy,
                        "T": int(T),
                        "r": int(r),
                        "g": int(g),
                        "target_freq": float(freq),
                        "rep": int(rep),
                        "seed": int(seed),
                        "max_generations": int(max_generations),
                    })

                    task_id += 1

    return tasks


def summarize_trials(df_trials, repeat):
    if df_trials.empty:
        return pd.DataFrame()

    rows = []

    group_cols = [
        "fig_name",
        "T",
        "r",
        "g",
        "target_freq",
        "target_strategy",
        "opponent_strategy",
    ]

    for keys, group in df_trials.groupby(group_cols):
        (
            fig_name,
            T,
            r,
            g,
            target_freq,
            target_strategy,
            opponent_strategy,
        ) = keys

        total = len(group)

        target_win_rate = (group["winner"] == target_strategy).sum() / total
        opponent_win_rate = (group["winner"] == opponent_strategy).sum() / total
        tie_rate = (group["winner"] == "TIE").sum() / total
        other_rate = (group["winner"] == "OTHER").sum() / total

        rows.append({
            "fig_name": fig_name,
            "T": int(T),
            "r": int(r),
            "g": int(g),
            "target_freq": float(target_freq),
            "target_strategy": target_strategy,
            "opponent_strategy": opponent_strategy,
            "wg": int(T) * int(g) / (N * (N - 1) / 2),
            "r_over_g": int(r) / int(g),
            "target_win_rate": target_win_rate,
            "opponent_win_rate": opponent_win_rate,
            "tie_rate": tie_rate,
            "other_rate": other_rate,
            "finished_trials": total,
            "expected_trials": repeat,
            "is_complete": total == repeat,
            "converged_rate": group["converged"].mean(),
            "avg_generation": group["generation"].mean(),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["T", "r", "target_freq"])
        .reset_index(drop=True)
    )


def run_checkpoint_experiment(
    fig_name,
    target_strategy,
    opponent_strategy,
    T_values,
    r_values,
    freq_values,
    g,
    repeat,
    max_generations,
    result_dir,
    n_jobs=6,
    chunk_size=12,
    base_seed=0,
):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = result_dir / "trials_checkpoint.csv"
    summary_path = result_dir / "summary_checkpoint.csv"
    meta_path = result_dir / "config.json"

    config = {
        "fig_name": fig_name,
        "target_strategy": target_strategy.name,
        "opponent_strategy": opponent_strategy.name,
        "T_values": [int(x) for x in T_values],
        "r_values": [int(x) for x in r_values],
        "freq_values": [float(x) for x in freq_values],
        "g": int(g),
        "N": int(N),
        "repeat": int(repeat),
        "max_generations": int(max_generations),
        "base_seed": int(base_seed),
    }

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            old_config = json.load(f)

        if old_config != config:
            raise ValueError(
                f"{meta_path} の既存設定と今回の設定が違います。\n"
                "別実験として回すなら result_dir を変えてください。"
            )
    else:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    all_tasks = build_tasks(
        fig_name=fig_name,
        target_strategy=target_strategy,
        opponent_strategy=opponent_strategy,
        T_values=T_values,
        r_values=r_values,
        freq_values=freq_values,
        g=g,
        repeat=repeat,
        max_generations=max_generations,
        base_seed=base_seed,
    )

    if checkpoint_path.exists():
        df_done = pd.read_csv(checkpoint_path)
        done_keys = set(df_done["task_key"].astype(str))
        print(f"checkpoint loaded: {checkpoint_path}")
    else:
        df_done = pd.DataFrame()
        done_keys = set()
        print("new experiment")

    remaining_tasks = [
        task for task in all_tasks
        if task["task_key"] not in done_keys
    ]

    print("================================")
    print(f"figure         : {fig_name}")
    print(f"result_dir     : {result_dir}")
    print(f"all tasks      : {len(all_tasks)}")
    print(f"done           : {len(done_keys)}")
    print(f"remaining      : {len(remaining_tasks)}")
    print(f"repeat         : {repeat}")
    print(f"max_generations: {max_generations}")
    print(f"n_jobs         : {n_jobs}")
    print(f"chunk_size     : {chunk_size}")
    print("================================")

    start_time = time.time()

    try:
        for start in range(0, len(remaining_tasks), chunk_size):
            chunk = remaining_tasks[start:start + chunk_size]

            chunk_results = Parallel(
                n_jobs=n_jobs,
                backend="loky",
                verbose=0,
            )(
                delayed(run_one_trial_task)(task)
                for task in chunk
            )

            df_chunk = pd.DataFrame(chunk_results)

            if df_done.empty:
                df_done = df_chunk
            else:
                df_done = pd.concat([df_done, df_chunk], ignore_index=True)

            df_done = (
                df_done
                .drop_duplicates(subset=["task_key"], keep="last")
                .sort_values("task_key")
                .reset_index(drop=True)
            )

            df_done.to_csv(checkpoint_path, index=False)

            df_summary = summarize_trials(df_done, repeat=repeat)
            df_summary.to_csv(summary_path, index=False)

            elapsed = time.time() - start_time

            print(
                f"\r{fig_name}: {len(df_done)} / {len(all_tasks)} finished "
                f"({elapsed / 60:.1f} min)",
                end=""
            )

    except KeyboardInterrupt:
        print()
        print("stopped manually")
        print(f"saved until last completed chunk: {checkpoint_path}")

    print()

    df_summary = summarize_trials(df_done, repeat=repeat)
    df_summary.to_csv(summary_path, index=False)

    return df_summary, df_done


def plot_result_scatter(
    df_summary,
    x_col,
    y_col,
    color_col,
    title,
    save_path,
    xlabel=None,
    ylabel=None,
    color_label=None,
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df_summary[x_col],
        df_summary[y_col],
        c=df_summary[color_col],
        s=300,
        vmin=0,
        vmax=1,
    )

    plt.colorbar(label=color_label or color_col)
    plt.xlabel(xlabel or x_col)
    plt.ylabel(ylabel or y_col)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()