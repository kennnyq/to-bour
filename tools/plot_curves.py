"""Render the TO-Bour training figure (light + dark PNG) and a run-log table from TensorBoard logs."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

sys.path.insert(0, str(Path(__file__).parent))
from tbread import run_series  # noqa: E402

LOGS = Path(sys.argv[1])
OUT = Path(sys.argv[2])
RUN = "crawler8"

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df",
                  s1="#2a78d6", s2="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#3a3a38",
                 s1="#3987e5", s2="#d95926"),
}


def ema(values, alpha=0.06):
    out, acc = [], None
    for v in values:
        acc = v if acc is None else alpha * v + (1 - alpha) * acc
        out.append(acc)
    return out


def render(series, theme, path):
    t = THEMES[theme]
    reward = series["Environment/Cumulative Reward"]
    xs = [s / 1e6 for s, _ in reward]
    ys = [v for _, v in reward]
    lessons = {
        "beam width": series.get("Environment/Lesson Number/ground_width", []),
        "target speed": series.get("Environment/Lesson Number/target_velocity", []),
    }

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.18}
    )
    fig.patch.set_facecolor(t["surface"])
    for ax in (ax1, ax2):
        ax.set_facecolor(t["surface"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(t["grid"])
        ax.tick_params(colors=t["ink2"], length=0)
        ax.grid(axis="y", color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)

    ax1.plot(xs, ys, color=t["s1"], linewidth=1, alpha=0.25)
    ax1.plot(xs, ema(ys), color=t["s1"], linewidth=2)
    ax1.set_ylabel("Mean episode reward", color=t["ink2"])
    ax1.yaxis.set_major_locator(MaxNLocator(5))

    for (name, pts), color in zip(lessons.items(), (t["s1"], t["s2"])):
        if not pts:
            continue
        lx = [s / 1e6 for s, _ in pts]
        ly = [v + 1 for _, v in pts]  # lessons are 0-indexed in the logs
        ax2.step(lx, ly, where="post", color=color, linewidth=2, label=name)
        ax2.annotate(name, (lx[-1], ly[-1]), xytext=(4, 0), textcoords="offset points",
                     va="center", color=t["ink2"], fontsize=9)
    ax2.set_ylabel("Lesson", color=t["ink2"])
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_xlabel("Environment steps (millions)", color=t["ink2"])
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}M"))
    leg = ax2.legend(loc="upper left", frameon=False, ncol=2, fontsize=9)
    for text in leg.get_texts():
        text.set_color(t["ink2"])
    ax2.set_xlim(0, xs[-1] * 1.08)

    fig.suptitle("Training run crawler8: reward and curriculum progress", x=0.06, ha="left",
                 color=t["ink"], fontsize=12, fontweight="bold", y=0.98)
    fig.text(0.06, 0.915, "PPO, Unity ML-Agents. Faint line: logged values; solid: smoothed. "
             "Reward scale shifts as lessons advance.", color=t["ink2"], fontsize=9)
    fig.subplots_adjust(left=0.1, right=0.86, top=0.87, bottom=0.1)
    fig.savefig(path, dpi=160, facecolor=t["surface"])
    plt.close(fig)


def run_log():
    rows = []
    for run_dir in sorted(LOGS.glob("crawler*"), key=lambda p: int(p.name[7:] or 1)):
        s = run_series(run_dir)
        reward = s.get("Environment/Cumulative Reward")
        if not reward:
            continue
        params = sorted(t.split("/")[-1].replace("_", " ") for t in s if "Lesson Number" in t)
        rows.append((run_dir.name, reward[-1][0], ", ".join(params) or "none"))
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    series = run_series(LOGS / RUN)
    for theme in THEMES:
        render(series, theme, OUT / f"training-{RUN}-{theme}.png")
    rows = run_log()
    total = sum(r[1] for r in rows)
    print("| Run | Steps | Curriculum |\n| --- | ---: | --- |")
    for name, steps, params in rows:
        print(f"| `{name}` | {steps / 1e6:.1f}M | {params} |")
    print(f"\nTOTAL {total / 1e6:.1f}M across {len(rows)} runs")
