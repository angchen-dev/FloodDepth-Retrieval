import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from colour import Color
import seaborn as sns
from collections import Counter
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker


REFERENCE_VEHICLE_HEIGHT_CM = 145.7
NUM_DEPTH_INTERVALS = 10
DEPTH_INTERVAL_WIDTH_CM = REFERENCE_VEHICLE_HEIGHT_CM / NUM_DEPTH_INTERVALS


def map_to_depth_interval(
    depth_cm,
    reference_height_cm=REFERENCE_VEHICLE_HEIGHT_CM,
    num_intervals=NUM_DEPTH_INTERVALS,
):
    """Map a positive reference depth to one of ten equal-width intervals."""
    if depth_cm <= 0:
        return 1

    interval_width = reference_height_cm / num_intervals
    interval_id = int(np.ceil(depth_cm / interval_width - 1e-12))
    return max(1, min(interval_id, num_intervals))


def depth_interval_label(
    interval_id,
    reference_height_cm=REFERENCE_VEHICLE_HEIGHT_CM,
    num_intervals=NUM_DEPTH_INTERVALS,
):
    """Return the depth range in centimeters for an interval identifier."""
    interval_width = reference_height_cm / num_intervals
    lower = (interval_id - 1) * interval_width
    upper = interval_id * interval_width
    return f"{lower:.2f}--{upper:.2f}"


def read_results(json_paths):
    methods = []
    all_res = {}
    for p in json_paths:
        with open(p, "r") as fi:
            tmp_res = json.load(fi)

            for image_name,v in tmp_res.items():
                gt_level = None
                image_name = image_name.replace(" ", "")
                if v is not None:
                    v_keys = list(v.keys())
                    v_keys.remove("gt")
                    if "gt_level" in v_keys:
                        gt_level = v["gt_level"]
                        v_keys.remove("gt_level")
                        

                    m = v_keys[0]
                    if "hdr" in m:
                        tmp_m = m.split("_")[4] + " Wm=" + m.split("_")[7][:3]
                    else:
                        tmp_m = m
    

                    if tmp_m not in methods:
                        methods.append(tmp_m)
                    
                    if image_name not in all_res:
                        all_res[image_name] = {}

                    if "gt" not in all_res[image_name]:
                        all_res[image_name]["gt"] = v["gt"]
                    else:
                        assert all_res[image_name]["gt"] == v["gt"], f"Mismatch at key {image_name}: {all_res[image_name]['gt']} != {v['gt']}, {p}"

                    if gt_level:
                        all_res[image_name]["gt_level"] = gt_level


                    all_res[image_name][tmp_m] = v[m]

    return all_res, methods


def check_method_image_count(all_results_dict, method):
    cnt = 0
    for k,v in all_results_dict.items():
        if method in v:
            cnt += 1
    return cnt



def mae_and_std(estimations, ground_truths):
    errors = [abs(est - gt) for est, gt in zip(estimations, ground_truths)]
    mae = sum(errors) / len(errors)
    std = np.std(errors, ddof=0)  # population std, use ddof=1 for sample std

    mae = round(mae, 2)
    std = round(std, 2)
    return mae, std


def MedAE(estimations, ground_truths):
    errors = [abs(est - gt) for est, gt in zip(estimations, ground_truths)]
    return np.median(errors)



def plot_per_level_boxplot(images_results: dict, methods_color={}):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Collect errors
    data = []
    for _, res in images_results.items():
        interval_id = map_to_depth_interval(res["gt"])
        gt = res["gt"]
        for m, pred in res.items():
            if m in ["gt", "gt_level"]:
                continue
            error = abs(pred - gt)
            data.append({"interval": interval_id, "method": m, "error": error})

    df = pd.DataFrame(data)

    # Use one method order for both the boxes and the legend.
    priority = ["park", "DSC-YOLOv8n", "FloodDepth-GPT"]
    methods = list(dict.fromkeys(df["method"].tolist()))
    methods_ordered = (
        [m for m in priority if m in methods] +
        [m for m in methods if m not in priority]
    )

    # Count unique images per reference-depth interval
    interval_counts = {}
    for _, res in images_results.items():
        interval_id = map_to_depth_interval(res["gt"])
        interval_counts[interval_id] = interval_counts.get(interval_id, 0) + 1

    # Plot boxplot
    fig = plt.figure(figsize=(13.5, 6))
    ax = sns.boxplot(
        x="interval", y="error", hue="method",
        data=df, palette=methods_color, dodge=True,
        order=range(1, NUM_DEPTH_INTERVALS + 1),
        hue_order=methods_ordered
    )

    # Mapping for legend labels
    label_map = {
        "park": "Park et al.",
        "FloodDepth-GPT": "FloodDepth-GPT",
        "20depths Wm=0.2": "Ours",
        "DSC-YOLOv8n": "DSC-YOLOv8n"
    }

    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))

    ordered_handles = [label_to_handle[m] for m in methods_ordered]
    ordered_labels = [label_map.get(m, m) for m in methods_ordered]

    ax.legend(
        ordered_handles,
        ordered_labels,
        title=None,
        prop={'size': 16, 'weight': 'bold'}
    )

    # Modify x-axis labels with counts
    new_xticklabels = []
    for interval_id in range(1, NUM_DEPTH_INTERVALS + 1):
        count = interval_counts.get(interval_id, 0)
        new_xticklabels.append(f"{interval_id}\n(n={count})")
    ax.set_xticks(range(NUM_DEPTH_INTERVALS))
    ax.set_xticklabels(new_xticklabels)

    # Set y-axis ticks (rounded to nearest 10)
    ymax = ax.get_ylim()[1]
    ymax = int((ymax // 10 + 1) * 10)
    yticks = list(range(0, ymax + 1, 10))
    ax.set_yticks(yticks)

    # Font styling
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(16)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(16)

    ax.grid(True, axis="y", alpha=0.3)

    plt.xlabel("Reference flood-depth bin", fontsize=16, fontweight='bold')
    plt.ylabel("Absolute error (cm)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return fig


#-------------------------------
def plot_level_error_with_counts(images_results:dict, methods_color={}, error_type="mae"):
    """
    error_type = mae / medae
    Reference depths are grouped into ten equal intervals spanning the
    145.7-cm height of the reference vehicle. This grouping is independent
    of the T=20 discretization used to construct the retrieval dataset.
    """

    interval_ids = np.asarray([
        map_to_depth_interval(v["gt"]) for _, v in images_results.items()
    ])
    interval_error = {}
    
    interval_ids_freq = dict(Counter(interval_ids))
    
    target_intervals = list(range(1, NUM_DEPTH_INTERVALS + 1))
    for interval_id in target_intervals:
        interval_error[interval_id] = {}

        if interval_id in interval_ids_freq:
            continue
        else:
            interval_ids_freq[interval_id] = 0

    interval_ids_freq = dict(sorted(interval_ids_freq.items()))
    interval_ids = list(interval_ids_freq.keys())
    interval_ids_count = list(interval_ids_freq.values())

    print("Total count:", int(sum(interval_ids_count)))
    for interval_id, count in interval_ids_freq.items():
        print(f"  {depth_interval_label(interval_id)} cm: {int(count)}")
    

    
    # {l:{m:[]}}
    for im, res in images_results.items():
        interval_id = map_to_depth_interval(res["gt"])

        for m, v in res.items():
            if m == "gt_level":
                continue
            else:
                if m not in interval_error[interval_id]:
                     interval_error[interval_id][m] = []
                interval_error[interval_id][m].append(v)

    # {l:{m:mae}}
    for interval_id, m_res in interval_error.items():
        for m, v in m_res.items():
            if m == "gt" or m == "gt_level":
                continue
            else:
                
                if error_type == "mae":
                    err, std = mae_and_std(m_res["gt"], v)
                else:
                    err = MedAE(m_res["gt"], v)
                
                interval_error[interval_id][m] = round(err, 2)
                # print(f"level:{l}, method:{m}, mae:{mae}")

        if len(interval_error[interval_id]) > 0:
            del interval_error[interval_id]["gt"]
            


    #------------------------------------
    methods = sorted({m for d in interval_error.values() for m in d})
    intervals = sorted(interval_error.keys())

    # ==== 排序方法，FloodDepth-GPT 和 park 优先 ====
    priority = ["park", "DSC-YOLOv8n", "FloodDepth-GPT", ]
    methods_ordered = [m for m in priority if m in methods] + [m for m in methods if m not in priority]

    fig, ax1 = plt.subplots(figsize=(13, 6))

    for m in methods_ordered:
        y = [np.nan if m not in interval_error[i] else interval_error[i][m] for i in intervals]
        ax1.plot(intervals, y, marker="o", linewidth=2,
                label=m, color=methods_color.get(m, None))

    ax1.set_xlabel("Flood-depth interval index", fontsize=16, fontweight='bold')
    if error_type == "mae":
        ax1.set_ylabel("MAE (cm)", fontsize=16, fontweight='bold')
    elif error_type == "medae":
        ax1.set_ylabel("MedAE (cm)", fontsize=16, fontweight='bold')
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    bars = ax2.bar(interval_ids, interval_ids_count, width=0.6,
                alpha=0.2, color="gray", label="Image count")
    ax2.set_ylabel("Image count", fontsize=16, fontweight='bold')
    ax2.set_ylim(0, max(interval_ids_count) + 7)
    ax2.set_yticks(range(0, max(interval_ids_count) + 8, 5))

    ax1.set_xticks(interval_ids)
    ax1.set_xticklabels(interval_ids)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)

    # 加粗 tick label
    for lbl in ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels():
        lbl.set_fontweight('bold')


    # 获取 legend 内容
    line_handles, line_labels = ax1.get_legend_handles_labels()
    bar_handles, bar_labels = ax2.get_legend_handles_labels()

    # 定义映射
    label_map = {
        "park": "Park et al.",
        "FloodDepth-GPT": "FloodDepth-GPT",
        "20depths Wm=0.2": "Ours",
        "DSC-YOLOv8n":"DSC-YOLOv8n"
    }

    # 替换
    line_labels = [label_map.get(l, l) for l in line_labels]
    bar_labels  = [label_map.get(l, l) for l in bar_labels]

    # 合并 legend
    ax1.legend(line_handles + bar_handles,
            line_labels + bar_labels,
            loc="best", prop={'size':16, 'weight': 'bold'})

    # plt.title("MAE and Count by Level", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return fig




def plot_mae_pdfs(gt, methods, maxX=90, methods_color={}, remove_thresh=6):
    """
    Plot PDFs of absolute error for different methods, with vertical lines at 90% quantile.
    Vertical lines are always drawn. X-axis ticks are filtered to avoid overlap.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    gt = np.asarray(gt)
    fig, ax = plt.subplots(figsize=(10, 5))

    ymax = 0
    q90_all = {}

    # ---- plotting order ----
    priority = ["park", "DSC-YOLOv8n", "FloodDepth-GPT"]
    methods_ordered = (
        [m for m in priority if m in methods] +
        [m for m in methods if m not in priority]
    )

    # ---- plot PDFs + q90 lines ----
    for method in methods_ordered:
        pred = np.asarray(methods[method])
        mask = np.isfinite(gt) & np.isfinite(pred)
        abs_err = np.abs(pred[mask] - gt[mask])

        if abs_err.size <= 1 or np.nanmax(abs_err) <= 0:
            continue

        xs = np.linspace(0, np.nanmax(abs_err), 200)
        kde = gaussian_kde(abs_err)
        ys = kde(xs)

        label_map = {
            "park": "Park et al.",
            "FloodDepth-GPT": "FloodDepth-GPT",
            "20depths Wm=0.2": "Ours",
            "DSC-YOLOv8n": "DSC-YOLOv8n"
        }

        color = methods_color.get(method, None)
        label = label_map.get(method, method)

        ax.plot(xs, ys, label=label, linewidth=4, color=color)
        ymax = max(ymax, ys.max())

        # ---- 90% quantile line (always draw) ----
        q90 = np.percentile(abs_err, 90)
        q90_all[method] = q90
        ax.axvline(q90, linestyle="--", color=color, alpha=0.6, linewidth=3)

    # ---- x-axis ticks (only affect labels, not lines) ----
    base_ticks = list(np.arange(0, maxX + 1, 10))
    raw_new_ticks = []

    for val in q90_all.values():
        if 0 < val < maxX:
            # remove base ticks too close to q90
            to_remove = [t for t in base_ticks if abs(t - val) < remove_thresh]
            for t in to_remove:
                base_ticks.remove(t)
            raw_new_ticks.append(val)

    # ---- filter new ticks: keep larger if too close ----
    raw_new_ticks = sorted(raw_new_ticks)
    filtered_new_ticks = []

    for v in raw_new_ticks:
        if not filtered_new_ticks:
            filtered_new_ticks.append(v)
        else:
            prev = filtered_new_ticks[-1]
            if abs(v - prev) < remove_thresh:
                filtered_new_ticks[-1] = max(prev, v)
            else:
                filtered_new_ticks.append(v)

    xticks = sorted(set(base_ticks + filtered_new_ticks))
    ax.set_xticks(xticks)

    xticklabels = []
    for t in xticks:
        if any(abs(t - v) < 1e-6 for v in q90_all.values()):
            xticklabels.append(f"{t:.1f}")
        else:
            xticklabels.append(f"{t:.0f}")

    ax.set_xticklabels(xticklabels, fontsize=18, fontweight="bold")

    # ---- cosmetics ----
    ax.set_xlim(0, maxX)
    ax.set_xlabel("Absolute error (cm)", fontsize=18, fontweight="bold")
    ax.set_yticks([])

    ax.legend(frameon=True, prop={'size': 18, 'weight': 'bold'})
    plt.tight_layout()
    plt.show()

    return fig



def assign_colors(all_methods):
    # 固定三种色盲安全颜色


    fixed = {
        "20depths Wm=0.2": "#0072B2",   # Blue (ours)
        "FloodDepth-GPT": "#E69F00",   # Orange
        "DSC-YOLOv8n": "#009E73",      # Green
        "FLOOD-DEPTH-ML": "#D55E00",  # Vermillion
        "park": "#CC79A7",             # Magenta
    }

    # "#E69F00",  # 橙
    # "#56B4E9",  # 天蓝
    # "#009E73",  # 绿
    # "#F0E442",  # 黄
    # "#0072B2",  # 蓝
    # "#D55E00",  # 红棕
    # "#CC79A7",  # 品红
    # "#000000",  # 黑
    
    # 生成额外调色板（husl 保证区分度）
    base_palette = sns.color_palette("husl", n_colors=30)
    pool = [
        "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
        for r,g,b in base_palette
    ]
    random.shuffle(pool)

    colors = {}
    for method in all_methods:
        if method in fixed:
            colors[method] = fixed[method]
        else:
            colors[method] = pool.pop(0) if pool else "#000000"
    return colors
