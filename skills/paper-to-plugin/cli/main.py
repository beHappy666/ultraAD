"""
CLI for paper-to-plugin skill
"""

import argparse
import os

# Adjust path to import from core
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from core.pipeline import PaperToPluginPipeline

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Automate paper analysis and plugin generation for ultraAD."
    )

    parser.add_argument(
        "source",
        type=str,
        help="Paper source (arXiv URL, PDF path, or URL)."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./p2p_output",
        help="Output directory for all generated artifacts."
    )
    parser.add_argument(
        "-t", "--type",
        type=str,
        choices=['arxiv', 'pdf', 'url', 'auto'],
        default='auto',
        help="Type of paper source."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging."
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup config
    config = {
        'test_config': {
            'verbose': args.verbose
        }
    }

    # Initialize and run pipeline
    pipeline = PaperToPluginPipeline(output_dir=args.output_dir, config=config)
    report = pipeline.run(source=args.source, source_type=args.type)

    # Print summary
    print("\n--- Final Summary ---")
    print(f"Status: {report.summary.status}")
    print(f"Total time: {report.summary.execution_time:.2f} seconds")
    print(f"Innovations found: {report.summary.innovations_found}")
    print(f"Plugins generated: {report.summary.plugins_generated}")
    print(f"Reports are available in: {report.output_directory}")

if __name__ == "__main__":
    main()
