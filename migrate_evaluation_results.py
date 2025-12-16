"""
Migration utility to convert old score-based evaluation results
to new A/B verdict format for apples-to-apples comparison.

This adds new columns to old evaluation files so they can be compared
directly with new evaluation results using the same metrics.
"""

import pandas as pd
import openpyxl
from pathlib import Path
import argparse


def migrate_evaluation_file(input_path: str, output_path: str = None):
    """
    Migrate old evaluation Excel file to new format.

    Args:
        input_path: Path to old evaluation_YYYYMMDD_HHMMSS.xlsx
        output_path: Path to save migrated file (optional, defaults to *_migrated.xlsx)
    """
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        output_path = input_file.parent / f"{input_file.stem}_migrated.xlsx"
    else:
        output_path = Path(output_path)

    print(f"Migrating {input_file.name}...")

    # Load both sheets
    metrics_df = pd.read_excel(input_path, sheet_name='metrics')
    predictions_df = pd.read_excel(input_path, sheet_name='predictions')

    print(f"  Loaded {len(metrics_df)} model(s) and {len(predictions_df)} prediction(s)")

    # Check format
    if 'confidence' in predictions_df.columns and 'tier' in predictions_df.columns:
        print("  ⚠ File appears to already be in new format, skipping migration")
        return

    # Add predicted_label column (rename consensus_label)
    if 'consensus_label' in predictions_df.columns:
        predictions_df['predicted_label'] = predictions_df['consensus_label']
    else:
        raise ValueError("Missing 'consensus_label' column in predictions sheet")

    # Convert score to confidence (0-10 scale → 0.0-1.0)
    if 'score_example' in predictions_df.columns:
        # Score interpretation: higher = more likely fake
        # score > 4 means "AI Generated", so map accordingly
        predictions_df['confidence'] = predictions_df['score_example'] / 10.0
    else:
        # If no score, infer from consensus_label
        print("  ⚠ No score_example column found, inferring confidence from predictions")
        predictions_df['confidence'] = predictions_df['predicted_label'].map({
            'AI Generated': 0.75,  # Assume high confidence for AI
            'Real': 0.25  # Assume low confidence for Real (low fake probability)
        })

    # Map verdict token (A=Real, B=Fake)
    predictions_df['verdict_token'] = predictions_df['predicted_label'].map({
        'Real': 'A',
        'AI Generated': 'B'
    })

    # Map to three-tier system based on confidence (fake probability)
    def confidence_to_tier(conf):
        """Convert confidence (fake probability) to tier."""
        if conf >= 0.90:
            return 'Deepfake'
        elif conf >= 0.50:
            return 'Suspicious'
        else:
            return 'Authentic'

    predictions_df['tier'] = predictions_df['confidence'].apply(confidence_to_tier)

    # Rename analysis_example to analysis if present
    if 'analysis_example' in predictions_df.columns:
        predictions_df['analysis'] = predictions_df['analysis_example']

    # Metrics sheet remains unchanged (TP/TN/FP/FN stay the same)

    # Write to new file
    print(f"  Writing migrated file to {output_path.name}...")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        metrics_df.to_excel(writer, sheet_name='metrics', index=False)
        predictions_df.to_excel(writer, sheet_name='predictions', index=False)

    print(f"✓ Migration complete!")
    print(f"  Added columns: predicted_label, confidence, tier, verdict_token")
    print(f"  Output: {output_path}")
    print()


def migrate_all_results(results_dir: str = 'results', overwrite: bool = False):
    """
    Migrate all evaluation files in results directory.

    Args:
        results_dir: Directory containing evaluation files
        overwrite: If True, overwrite original files. If False, create *_migrated.xlsx
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return

    xlsx_files = list(results_path.glob('evaluation_*.xlsx'))
    xlsx_files = [f for f in xlsx_files if '_migrated' not in f.name]

    if not xlsx_files:
        print(f"No evaluation files found in {results_dir}")
        return

    print(f"Found {len(xlsx_files)} evaluation file(s) to migrate:")
    for f in xlsx_files:
        print(f"  - {f.name}")
    print()

    for xlsx_file in xlsx_files:
        try:
            if overwrite:
                # Create temp file, then replace original
                temp_output = xlsx_file.parent / f"{xlsx_file.stem}_temp.xlsx"
                migrate_evaluation_file(str(xlsx_file), str(temp_output))
                temp_output.replace(xlsx_file)
                print(f"  ✓ Overwrote {xlsx_file.name}")
            else:
                # Create new _migrated file
                migrate_evaluation_file(str(xlsx_file))

        except Exception as e:
            print(f"  ✗ Error migrating {xlsx_file.name}: {e}")
            continue

    print()
    print("="*80)
    print("Migration Summary:")
    print(f"  Total files processed: {len(xlsx_files)}")
    print(f"  Output location: {results_path.absolute()}")
    print()
    print("Next steps:")
    print("  1. Open migrated files to verify data integrity")
    print("  2. Run generate_report_updated.py on both old and new files")
    print("  3. Compare metrics (TP/TN/FP/FN should match exactly)")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Migrate old evaluation results to new A/B verdict format'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Migrate a single file (e.g., results/evaluation_20251202_213404.xlsx)'
    )
    parser.add_argument(
        '--dir',
        type=str,
        default='results',
        help='Directory containing evaluation files (default: results)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (only used with --file)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite original files instead of creating *_migrated.xlsx'
    )

    args = parser.parse_args()

    print("="*80)
    print("Evaluation Results Migration Utility")
    print("="*80)
    print()

    if args.file:
        # Migrate single file
        migrate_evaluation_file(args.file, args.output)
    else:
        # Migrate all files in directory
        migrate_all_results(args.dir, args.overwrite)


if __name__ == '__main__':
    main()
