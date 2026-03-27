import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the pipeline paths."""

    project_root: Path = Path(".")
    notebook_path: Path = Path("main.ipynb")
    train_script: Path = Path("scripts_logs_entreno/train_main.py")
    notebook_gen_script: Path = Path("scripts_logs_entreno/write_notebook.py")
    models_dir: Path = Path("models")
    data_split_dir: Path = Path("data_split")
    results_file: Path = Path("train_main_results.json")


def _run_command(
    cmd: List[str], description: str, success_msg: str, error_msg: str
) -> bool:
    """
    Execute a command with consistent error handling.

    Args:
        cmd: Command and arguments to execute
        description: Description of the command being run
        success_msg: Message to log on success
        error_msg: Message to log on error

    Returns:
        True if command succeeded, False otherwise
    """
    logger.info(f"🔬 {description}")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"✅ {success_msg}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {error_msg}")
        if e.stderr:
            logger.error(f"Error details: {e.stderr.strip()}")
        return False
    except FileNotFoundError as e:
        logger.error(f"❌ Command not found: {cmd[0]}")
        logger.error(f"Error details: {e}")
        return False


def _check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    missing_deps = []

    try:
        import nbconvert
    except ImportError:
        missing_deps.append("nbconvert (install with: pip install nbconvert)")

    # Check if jupyter is available
    try:
        subprocess.run(
            ["jupyter", "--version"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing_deps.append("jupyter (install with: pip install jupyter)")

    if missing_deps:
        logger.error("Missing dependencies:")
        for dep in missing_deps:
            logger.error(f"  - {dep}")
        return False

    return True


def run_notebook(notebook_path: Optional[Path] = None, config: Optional[PipelineConfig] = None) -> bool:
    """
    Execute a Jupyter notebook using nbconvert.

    Args:
        notebook_path: Path to the notebook to execute
        config: Pipeline configuration

    Returns:
        True if notebook executed successfully, False otherwise
    """
    if config is None:
        config = PipelineConfig()

    notebook_path = notebook_path or config.notebook_path

    if not notebook_path.exists():
        logger.error(f"Notebook '{notebook_path}' not found")
        return False

    return _run_command(
        cmd=[
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--inplace",
            "--execute",
            "--ExecutePreprocessor.timeout=0",
            str(notebook_path),
        ],
        description=f"Executing notebook: {notebook_path}",
        success_msg=f"Notebook executed successfully: {notebook_path}",
        error_msg=f"Notebook execution failed: {notebook_path}",
    )


def run_training(config: Optional[PipelineConfig] = None) -> bool:
    """
    Run the training script.

    Args:
        config: Pipeline configuration

    Returns:
        True if training completed successfully, False otherwise
    """
    if config is None:
        config = PipelineConfig()

    train_script = config.train_script

    if not train_script.exists():
        logger.error(f"Training script '{train_script}' not found")
        return False

    return _run_command(
        cmd=[sys.executable, str(train_script)],
        description=f"Running training: {train_script}",
        success_msg="Training completed successfully",
        error_msg="Training failed",
    )


def regenerate_notebook(config: Optional[PipelineConfig] = None) -> bool:
    """
    Regenerate main.ipynb from configuration.

    Args:
        config: Pipeline configuration

    Returns:
        True if notebook regenerated successfully, False otherwise
    """
    if config is None:
        config = PipelineConfig()

    script = config.notebook_gen_script

    if not script.exists():
        logger.error(f"Notebook generation script '{script}' not found")
        return False

    return _run_command(
        cmd=[sys.executable, str(script)],
        description="Regenerating notebook from configuration",
        success_msg="Notebook regenerated successfully",
        error_msg="Notebook generation failed",
    )


def _print_summary(results: List[Tuple[str, bool]]) -> None:
    """Print a summary of pipeline execution results."""
    logger.info("=" * 60)
    logger.info("Pipeline Execution Summary:")
    for task, success in results:
        status = "✅ Success" if success else "❌ Failed"
        logger.info(f"  {task}: {status}")
    logger.info("=" * 60)


def show_status(config: Optional[PipelineConfig] = None) -> None:
    """Display comprehensive project status."""
    if config is None:
        config = PipelineConfig()

    logger.info("=" * 70)
    logger.info("📊 Diabetic Retinopathy Classification - Project Status")
    logger.info("=" * 70)

    # Check data splits
    logger.info("\n📁 Data Splits:")
    if config.data_split_dir.exists():
        for split in ["train", "val", "test"]:
            split_dir = config.data_split_dir / split
            if split_dir.exists():
                num_images = sum(
                    1
                    for f in split_dir.rglob("*")
                    if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg"]
                )
                logger.info(f"  ✓ {split:5s}: {num_images:>6,} images")
            else:
                logger.info(f"  ✗ {split:5s}: not found")
    else:
        logger.info(f"  ✗ {config.data_split_dir}/ directory not found")

    # Check trained models
    logger.info("\n🤖 Trained Models:")
    if config.models_dir.exists():
        models = sorted(config.models_dir.glob("*.pth"))
        if models:
            for model in models:
                size_mb = model.stat().st_size / (1024 * 1024)
                mtime = datetime.fromtimestamp(model.stat().st_mtime)
                logger.info(f"  ✓ {model.name}")
                logger.info(
                    f"    └─ {size_mb:>6.1f} MB | Modified: {mtime:%Y-%m-%d %H:%M:%S}"
                )
        else:
            logger.info("  ⚠ No trained models found")
    else:
        logger.info(f"  ✗ {config.models_dir}/ directory not found")

    # Check training results
    logger.info("\n📈 Training Results:")
    if config.results_file.exists():
        try:
            with open(config.results_file) as f:
                results = json.load(f)
            logger.info(
                f"  ✓ Best Val Accuracy:  {results.get('best_val_accuracy', 0):.4f}"
            )
            logger.info(f"  ✓ Test Accuracy:      {results.get('test_accuracy', 0):.4f}")
            logger.info(f"  ✓ Epochs Trained:     {results.get('epochs_trained', 0)}")
        except json.JSONDecodeError:
            logger.warning("  ⚠ Results file exists but is corrupted")
        except Exception as e:
            logger.error(f"  ✗ Error reading results: {e}")
    else:
        logger.info("  ⚠ No training results found (train model first)")

    # Check available notebooks
    logger.info("\n📓 Available Notebooks:")
    notebooks = {
        "main.ipynb": "Main classification notebook",
        "segmentation.ipynb": "Segmentation analysis",
        "yolov8_segmentation.ipynb": "YOLOv8 segmentation",
        "sam_segmentation.ipynb": "SAM segmentation",
    }
    for nb_name, desc in notebooks.items():
        nb_path = config.project_root / nb_name
        if nb_path.exists():
            size_kb = nb_path.stat().st_size / 1024
            logger.info(f"  ✓ {nb_name:30s} ({size_kb:>7.1f} KB) - {desc}")
        else:
            logger.info(f"  ✗ {nb_name:30s} - Not found")

    logger.info("\n" + "=" * 70)


def open_notebook(notebook_path: Path, config: Optional[PipelineConfig] = None) -> bool:
    """
    Open a notebook in Jupyter for interactive editing.

    Args:
        notebook_path: Path to the notebook to open
        config: Pipeline configuration

    Returns:
        True if notebook opened successfully, False otherwise
    """
    if config is None:
        config = PipelineConfig()

    if not notebook_path.exists():
        logger.error(f"Notebook '{notebook_path}' not found")
        # List available notebooks
        available = list(config.project_root.glob("*.ipynb"))
        if available:
            logger.info("Available notebooks:")
            for nb in available:
                logger.info(f"  - {nb.name}")
        return False

    logger.info(f"📓 Opening notebook: {notebook_path.name}")
    try:
        subprocess.run(["jupyter", "notebook", str(notebook_path)])
        return True
    except FileNotFoundError:
        logger.error("jupyter not found. Install with: pip install jupyter")
        return False
    except Exception as e:
        logger.error(f"Failed to open notebook: {e}")
        return False


def interactive_mode(config: Optional[PipelineConfig] = None) -> int:
    """
    Launch interactive menu-driven mode.

    Args:
        config: Pipeline configuration

    Returns:
        Exit code
    """
    if config is None:
        config = PipelineConfig()

    while True:
        print("\n" + "=" * 70)
        print("🩺 Diabetic Retinopathy Classification Pipeline")
        print("=" * 70)
        print("\n1. Show project status")
        print("2. Train model")
        print("3. Regenerate notebook")
        print("4. Execute notebook")
        print("5. Open notebook in Jupyter")
        print("6. Full pipeline (train → regenerate → execute)")
        print("7. Check dependencies")
        print("0. Exit")
        print()

        try:
            choice = input("Select option [0-7]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            return 0

        if choice == "0":
            print("\n👋 Goodbye!")
            return 0
        elif choice == "1":
            show_status(config)
        elif choice == "2":
            run_training(config)
        elif choice == "3":
            regenerate_notebook(config)
        elif choice == "4":
            nb = input(f"Notebook name [{config.notebook_path}]: ").strip()
            nb_path = Path(nb) if nb else config.notebook_path
            run_notebook(nb_path, config)
        elif choice == "5":
            nb = input(f"Notebook name [{config.notebook_path}]: ").strip()
            nb_path = Path(nb) if nb else config.notebook_path
            open_notebook(nb_path, config)
        elif choice == "6":
            logger.info("🔄 Running full pipeline...")
            results = []
            results.append(("Training", run_training(config)))
            if results[-1][1]:
                results.append(("Regenerate", regenerate_notebook(config)))
                if results[-1][1]:
                    results.append(("Execute", run_notebook(config.notebook_path, config)))
            _print_summary(results)
        elif choice == "7":
            if _check_dependencies():
                logger.info("✅ All dependencies are installed")
            else:
                logger.error("❌ Some dependencies are missing")
        else:
            logger.warning("Invalid option. Try again.")

        input("\nPress Enter to continue...")


def main() -> int:
    """
    Main entry point for the pipeline.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Diabetic Retinopathy Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --status                      # Show project status
  python main.py --interactive                 # Launch interactive mode
  python main.py --train                       # Train the model
  python main.py --regenerate                  # Regenerate main.ipynb
  python main.py --run-notebook                # Execute main.ipynb
  python main.py --open main.ipynb             # Open notebook in Jupyter
  python main.py --all                         # Full pipeline
  python main.py --notebook segmentation.ipynb --run-notebook  # Run specific notebook
        """,
    )

    parser.add_argument(
        "--status",
        "-s",
        action="store_true",
        help="Show comprehensive project status (models, data, results)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Launch interactive menu-driven mode",
    )
    parser.add_argument(
        "--run-notebook",
        action="store_true",
        help="Execute the main.ipynb notebook (or notebook specified with --notebook)",
    )
    parser.add_argument(
        "--open",
        metavar="NOTEBOOK",
        type=Path,
        help="Open a notebook in Jupyter for interactive editing",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run the training pipeline",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate main.ipynb from best configuration",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run full pipeline: train → regenerate → execute notebook",
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        help="Specify a notebook path (use with --run-notebook)",
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check if all required dependencies are installed",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Create configuration
    config = PipelineConfig()

    # Check dependencies if requested
    if args.check_deps:
        if _check_dependencies():
            logger.info("✅ All dependencies are installed")
            return 0
        return 1

    # Show status
    if args.status:
        show_status(config)
        return 0

    # Interactive mode
    if args.interactive:
        return interactive_mode(config)

    # Open notebook for editing
    if args.open:
        return 0 if open_notebook(args.open, config) else 1

    # If no action specified, show help
    if not any([args.run_notebook, args.train, args.regenerate, args.all]):
        parser.print_help()
        return 0

    # Check dependencies before running pipeline tasks
    if not _check_dependencies():
        logger.error("Please install missing dependencies before continuing")
        return 1

    # Execute requested tasks
    results: List[Tuple[str, bool]] = []

    if args.all or args.train:
        results.append(("Training", run_training(config)))

    if args.all or args.regenerate:
        results.append(("Notebook Generation", regenerate_notebook(config)))

    if args.all or args.run_notebook:
        notebook_path = args.notebook if args.notebook else config.notebook_path
        results.append(("Notebook Execution", run_notebook(notebook_path, config)))

    # Print summary if multiple operations
    if len(results) > 1:
        _print_summary(results)

    # Return appropriate exit code
    return 0 if all(success for _, success in results) else 1


if __name__ == "__main__":
    sys.exit(main())
