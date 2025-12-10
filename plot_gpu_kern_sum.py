#!/usr/bin/env python
"""
cd /root/autodl-tmp
python plot_gpu_kern_sum.py \
  --csv /root/autodl-tmp/nano-vllm-main/gpu_kern_sum.csv \
  --out-dir ./gpu_plots \
  --topk 20
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_kernel_csv(path: Path) -> pd.DataFrame:
    """
    读取 nsight-systems 的 cuda_gpu_kern_sum.csv。
    前面几行是 NOTICE / Processing 文本，从第 7 行开始才是表头。
    """
    df = pd.read_csv(path, skiprows=6)
    # 去掉列名里的多余空格
    df.columns = [c.strip() for c in df.columns]
    return df


def plot_top_time_percent(df: pd.DataFrame, topk: int, out_dir: Path):
    """
    画按 Time (%) 排序的前 topk kernel 柱状图。
    """
    df_top = df.sort_values("Time (%)", ascending=False).head(topk)

    plt.figure(figsize=(14, max(6, 0.4 * len(df_top))))
    plt.barh(df_top["Name"], df_top["Time (%)"])
    plt.xlabel("Time (%)")
    plt.ylabel("Kernel Name")
    plt.title(f"Top {topk} Kernels by GPU Time Percentage")
    plt.gca().invert_yaxis()  # 让最耗时的在最上面
    # 先自适应，再额外给左边留出更多空间，避免长 kernel 名称被裁剪
    plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.35)

    out_path = out_dir / f"top{topk}_kernels_time_percent.png"
    # bbox_inches="tight" 确保文字完全包含在图片内
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_instances_vs_avg(df: pd.DataFrame, out_dir: Path):
    """
    画 Instances vs Avg (ns) 散点图，点大小按 Time (%) 缩放。
    用 log 坐标，方便看分布。
    """
    plt.figure(figsize=(10, 8))
    x = df["Avg (ns)"]
    y = df["Instances"]
    s = df["Time (%)"] * 5  # 点大小可按需要调大/调小

    plt.scatter(x, y, s=s, alpha=0.7)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Average Duration (ns, log-scale)")
    plt.ylabel("Instances (log-scale)")
    plt.title("Kernel Instances vs Avg Duration (size ~ Time %)")
    plt.grid(True, which="both", ls="--", alpha=0.3)

    out_path = out_dir / "instances_vs_avg_ns.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_cumulative_time(df: pd.DataFrame, out_dir: Path):
    """
    画累计 Time (%) 曲线，看前多少个 kernel 覆盖了多少总时间（帕累托曲线）。
    """
    df_sorted = df.sort_values("Time (%)", ascending=False).reset_index(drop=True)
    cum_time = df_sorted["Time (%)"].cumsum()
    ranks = df_sorted.index + 1

    plt.figure(figsize=(10, 6))
    plt.plot(ranks, cum_time, marker="o", markersize=3)
    plt.xlabel("Kernel Rank (sorted by Time %)")
    plt.ylabel("Cumulative Time (%)")
    plt.title("Cumulative GPU Time vs Kernel Rank")
    plt.grid(True, ls="--", alpha=0.3)

    # 标出 80% / 90% 的大致位置，帮助看 top-k 贡献
    for target in [80, 90]:
        idx = (cum_time >= target).idxmax()
        plt.axhline(target, color="red", ls="--", alpha=0.5)
        plt.axvline(idx + 1, color="red", ls="--", alpha=0.5)
        plt.text(
            idx + 1,
            target,
            f"  {target}% at top-{idx+1}",
            va="bottom",
            ha="left",
            fontsize=8,
            color="red",
        )

    out_path = out_dir / "cumulative_time_percent.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def _kernel_family(name: str) -> str:
    """
    简单根据名称前缀/关键字把 kernel 分组，方便看不同库/组件的占比。
    可以按需再丰富规则。
    """
    n = name.lower()
    if n.startswith("triton_"):
        return "triton"
    if "flash::" in n:
        return "flash-attn"
    if "cutlass::" in n:
        return "cutlass"
    if "cublas" in n or "gemv" in n:
        return "cublas"
    if "ampere_bf16" in n or "ampere_s16816" in n:
        return "ampere_gemm"
    if "at::native" in n or "aten::" in n:
        return "pytorch_at::native"
    if "store_kvcache" in n:
        return "custom_kvcache"
    return "other"


def plot_family_time_breakdown(df: pd.DataFrame, out_dir: Path):
    """
    按 kernel family 聚合 Time (%)，看大头是在 triton / cutlass / flash / cublas /
    PyTorch runtime 还是自定义 kernel 上。
    """
    fam_series = df["Name"].apply(_kernel_family)
    fam_time = df.groupby(fam_series)["Time (%)"].sum().sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(fam_time.index, fam_time.values)
    plt.ylabel("Time (%)")
    plt.title("Kernel Family Time Breakdown (sum of Time %)")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", ls="--", alpha=0.3)

    out_path = out_dir / "family_time_breakdown.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_avg_duration_hist(df: pd.DataFrame, out_dir: Path):
    """
    画 Avg (ns) 的分布（对数坐标），看是否有极端慢的 kernel。
    """
    avg = df["Avg (ns)"].dropna()
    plt.figure(figsize=(10, 6))
    plt.hist(avg, bins=30, log=True)
    plt.xscale("log")
    plt.xlabel("Average Duration (ns, log-scale)")
    plt.ylabel("Count (log-scale)")
    plt.title("Distribution of Kernel Avg Duration")
    plt.grid(True, which="both", ls="--", alpha=0.3)

    out_path = out_dir / "avg_duration_hist.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_total_time_vs_avg(df: pd.DataFrame, out_dir: Path):
    """
    画 Total Time (ns) vs Avg (ns) 散点图，点大小按 Instances 缩放。
    右上/右侧偏上的点通常是单次慢 or 累积时间大的优化热点。
    """
    plt.figure(figsize=(10, 8))
    x = df["Avg (ns)"]
    y = df["Total Time (ns)"]
    s = df["Instances"] * 0.02  # 根据实际数据大小可调整

    plt.scatter(x, y, s=s, alpha=0.7)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Average Duration (ns, log-scale)")
    plt.ylabel("Total Time (ns, log-scale)")
    plt.title("Total Time vs Avg Duration (size ~ Instances)")
    plt.grid(True, which="both", ls="--", alpha=0.3)

    out_path = out_dir / "total_time_vs_avg_ns.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot Nsight Systems cuda_gpu_kern_sum.csv for analysis."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("/root/autodl-tmp/nano-vllm-main/gpu_kern_sum.csv"),
        help="Path to gpu_kern_sum.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./plots"),
        help="Directory to save plots",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Number of top kernels by Time (%%) to plot",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_kernel_csv(args.csv)

    # 转成数值（一般 pandas 会自动转，这里只是保险）
    for col in ["Time (%)", "Total Time (ns)", "Instances", "Avg (ns)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 画图
    plot_top_time_percent(df, args.topk, args.out_dir)
    plot_instances_vs_avg(df, args.out_dir)
    plot_cumulative_time(df, args.out_dir)
    plot_family_time_breakdown(df, args.out_dir)
    plot_avg_duration_hist(df, args.out_dir)
    plot_total_time_vs_avg(df, args.out_dir)


if __name__ == "__main__":
    main()

