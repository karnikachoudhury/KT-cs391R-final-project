import matplotlib.pyplot as plt
import os


def parse_training_log(filepath):
    data = {
        "Avg. Reward": [],
        "Weighted ICM Loss": [],
        "Inverse Loss": [],
        "Forward Loss": [],
        "Avg. Intrinsic Reward": [],
        "Successful Epochs": [], # cumulative number of epochs with a success
    }

    with open(filepath, "r") as f:
        lines = f.readlines()

    step = -1
    had_success = False
    successes = 0

    for line in lines:
        line = line.strip()

        # stop when testing section starts
        if line.startswith("Episode"):
            break

        if line.startswith("Success"):
            had_success = True
        
        elif line.startswith("Step"):
            step += 1
            if had_success:
                successes += 1
            data["Successful Epochs"].append(successes)
            had_success = False

        elif "ep_rew_mean" in line:
            data["Avg. Reward"].append(float(line.split("|")[2]))


        # these may not be accurate currently, since we randomly sample all episodes instead of recent ones
        elif line.startswith("icm_loss"):
            data["Weighted ICM Loss"].append(float(line.split(":")[1]))

        elif line.startswith("inv_loss"):
            data["Inverse Loss"].append(float(line.split(":")[1]))

        elif line.startswith("fwd_loss"):
            data["Forward Loss"].append(float(line.split(":")[1]))

        elif line.startswith("r_int_mean"):
            data["Avg. Intrinsic Reward"].append(float(line.split(":")[1]))


    return data


def plot_stat(values, title, ylabel, filename, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure()
    plt.plot(range(len(values)), values, marker="o")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

def plot_stat_across_runs(all_stats, stat_name, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    with_labeled = False
    wo_labeled = False
    for run_name, stats in all_stats.items():
        values = stats.get(stat_name, [])
        if len(values) == 0:
            continue
        # make all ICM runs share the same color/label, and same for non-ICM runs
        if run_name.startswith("With ICM"):
            label = "With ICM" if not with_labeled else None
            with_labeled = True
            plt.plot(range(len(values)), values, color="tab:blue", alpha=0.6, label=label)

        else:
            label = "Without ICM" if not wo_labeled else None
            wo_labeled = True
            plt.plot(range(len(values)), values, color="tab:red", alpha=0.6, label=label)

    plt.title(f"{stat_name} Comparison")
    plt.xlabel("Epoch")
    plt.ylabel(stat_name)
    plt.grid(True)
    plt.legend()

    filepath = os.path.join(output_dir, f"{stat_name}_comparison.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    dir_to_name = {
        "test_with_icm_lam_0.1_1": "With ICM (1)",
        "test_with_icm_lam_0.1_2": "With ICM (2)",
        "test_with_icm_lam_0.1_3": "With ICM (3)",
        "test_no_icm_1": "Without ICM (1)",
        "test_no_icm_2": "Without ICM (2)",
        "test_no_icm_3": "Without ICM (3)",
        ".karnika_messing": "With ICM (idk)",
    }
    input_dirs = ["test_with_icm_lam_0.1_1", "test_with_icm_lam_0.1_2", "test_with_icm_lam_0.1_3", "test_no_icm_1", "test_no_icm_2", "test_no_icm_3", ".karnika_messing"]
    all_stats = {}
    for input_dir in input_dirs:
        name = dir_to_name[input_dir]
        data = parse_training_log(os.path.join(input_dir, "output.txt"))
        all_stats[name] = data
        for stat, values in data.items():
            if len(values) == 0:
                continue
            plot_stat(values, f"{name}: {stat} over Time", stat, f"{stat}.png", output_dir=os.path.join(input_dir, "plots"))

    plot_stat_across_runs(all_stats, "Avg. Reward")
    plot_stat_across_runs(all_stats, "Successful Epochs")