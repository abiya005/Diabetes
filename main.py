"""
main.py
Master execution pipeline for the machine learning research study:
"Evaluating the Robustness and Fairness of Diabetes Risk Prediction Models Across Demographic Groups"

Usage:
    python main.py                  # Executes full research pipeline end-to-end
    python main.py --dashboard      # Launches the interactive Streamlit research dashboard
    python main.py --reports-only   # Compiles all PDF research reports from saved results
    python main.py --skip-train     # Skips model training and evaluates existing models
"""

import os
import subprocess
import sys

# Auto-detect and redirect to configured Python 3.13 environment if running in an unconfigured Python alias
PY313_PATH = r"C:\Users\Abiya Biju\AppData\Local\Programs\Python\Python313\python.exe"
if os.path.exists(PY313_PATH) and os.path.normcase(sys.executable) != os.path.normcase(
    PY313_PATH
):
    import importlib.util

    if (
        importlib.util.find_spec("matplotlib") is None
        or importlib.util.find_spec("streamlit") is None
    ):
        result = subprocess.call([PY313_PATH] + sys.argv)
        sys.exit(result)

import tempfile

os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")
import matplotlib

matplotlib.use("Agg")

import argparse
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("MasterPipeline")


def run_step(step_name: str, func, *args, **kwargs):
    logger.info("=" * 65)
    logger.info(f"STARTING STEP: {step_name}")
    logger.info("=" * 65)
    result = func(*args, **kwargs)
    logger.info(f"COMPLETED STEP: {step_name}\n")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Diabetes Risk ML Fairness & Robustness Research Pipeline"
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch interactive Streamlit dashboard",
    )
    parser.add_argument(
        "--port", type=int, default=8501, help="Port for Streamlit dashboard"
    )
    parser.add_argument(
        "--reports-only", action="store_true", help="Generate PDF reports only"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip model training and use saved models",
    )
    args = parser.parse_args()

    if args.dashboard:
        logger.info("Launching Streamlit Research Dashboard...")
        py_exe = sys.executable if "Python313" in sys.executable else PY313_PATH
        cmd = [
            py_exe,
            "-m",
            "streamlit",
            "run",
            "dashboard/app.py",
            "--server.port",
            str(args.port),
            "--server.headless",
            "false",
        ]
        subprocess.run(cmd, check=False)
        return

    if args.reports_only:
        from src.generate_reports import (
            generate_fairness_report,
            generate_final_research_report,
            generate_model_comparison_report,
            generate_preprocessing_report,
        )

        run_step("Preprocessing Report", generate_preprocessing_report)
        run_step("Model Comparison Report", generate_model_comparison_report)
        run_step("Fairness Report", generate_fairness_report)
        run_step("Final Research Report", generate_final_research_report)
        logger.info("All reports compiled successfully in reports/")
        return

    # Full End-to-End Execution Pipeline
    from src.combined_comparison import generate_combined_comparison
    from src.create_notebooks import (
        create_exploration_notebook,
        create_fairness_notebook,
        create_model_training_notebook,
        create_preprocessing_notebook,
        create_robustness_notebook,
    )
    from src.data_loader import (
        generate_dataset_profile,
        load_dataset,
        plot_dataset_visualizations,
    )
    from src.evaluate_models import run_evaluation_pipeline
    from src.fairness_analysis import run_fairness_pipeline
    from src.generate_reports import (
        generate_fairness_report,
        generate_final_research_report,
        generate_model_comparison_report,
        generate_preprocessing_report,
    )
    from src.preprocessing import prepare_data
    from src.robustness_analysis import run_robustness_pipeline
    from src.statistical_analysis import run_statistical_pipeline
    from src.train_models import run_training_pipeline

    # 1. Dataset Profiling & Visualization
    df = run_step("Dataset Ingestion & Harmonization", load_dataset)
    run_step("Dataset Profiling", generate_dataset_profile, df)
    run_step("Exploratory Visualizations", plot_dataset_visualizations, df)

    # 2. Preprocessing & Leakage Prevention
    run_step("Preprocessing Pipeline & Split", prepare_data)

    # 3. Model Training & 5-Fold Cross-Validation
    if not args.skip_train:
        run_step("Model Training & 5-Fold CV", run_training_pipeline)

    # 4. Overall Test Performance Evaluation
    run_step("Test Set Performance Evaluation", run_evaluation_pipeline)

    # 5. Demographic Fairness Evaluation
    run_step("Demographic Fairness Audit", run_fairness_pipeline)

    # 6. Robustness Stress-Testing
    run_step("Robustness Analysis (Missingness, Noise, Shift)", run_robustness_pipeline)

    # 7. Statistical Validation
    run_step("Statistical Validation & Hypothesis Testing", run_statistical_pipeline)

    # 8. Combined Synthesis Matrix
    run_step("Combined Three-Dimensional Matrix", generate_combined_comparison)

    # 9. Jupyter Notebooks Generation
    run_step(
        "Jupyter Notebooks Generation",
        lambda: [
            create_exploration_notebook(),
            create_preprocessing_notebook(),
            create_model_training_notebook(),
            create_fairness_notebook(),
            create_robustness_notebook(),
        ],
    )

    # 10. PDF Research Reports Compilation
    run_step(
        "PDF Research Reports Compilation",
        lambda: [
            generate_preprocessing_report(),
            generate_model_comparison_report(),
            generate_fairness_report(),
            generate_final_research_report(),
        ],
    )

    logger.info("=" * 65)
    logger.info("RESEARCH PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info(
        "Results saved in results/, figures/, reports/, notebooks/, and models/"
    )
    logger.info(
        "Run 'python main.py --dashboard' to launch the interactive research interface."
    )
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
