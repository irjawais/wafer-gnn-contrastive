"""Driver: regenerate every table and figure in the paper."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", nargs="*", default=[], help="Skip these experiments.")
    args = parser.parse_args()

    experiments = [
        ("pretrain", "experiments.pretrain_main"),
        ("table4_features", "experiments.feature_ablation"),
        ("table5_per_class", "experiments.per_class_f1"),
        ("table6_baselines", "experiments.run_baselines"),
        ("table7_sota", "experiments.sota_comparison"),
        ("table8_objective_graph", "experiments.objective_and_graph_ablation"),
        ("table9_significance", "experiments.statistical_significance"),
        ("table10_thresholds", "experiments.threshold_sensitivity"),
        ("table11_weights", "experiments.weight_sensitivity"),
        ("table12_compute", "experiments.computational_analysis"),
        ("low_label", "experiments.low_label_experiment"),
        ("robustness", "experiments.robustness"),
        ("ood", "experiments.ood_detection"),
        ("figs_pca_tsne", "experiments.embedding_visualization"),
        ("fig_confusion", "experiments.confusion_matrix_plot"),
    ]

    for name, mod in experiments:
        if name in args.skip:
            print(f"== Skipping {name} ==")
            continue
        print(f"\n========== Running {name} ==========")
        __import__(mod)
        sys.modules[mod].run()


if __name__ == "__main__":
    main()
