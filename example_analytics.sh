#!/bin/bash
#
# Example Analytics Usage
#
# This script demonstrates how to generate analytics reports comparing
# different evaluation runs from the NexInspect system.
#

set -e

RESULTS_DIR="results"
OUTPUT_DIR="reports"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "NexInspect Analytics Examples"
echo "=========================================="
echo

# Example 1: Compare all evaluation files
echo "Example 1: Comparing all evaluation files..."
python3 generate_analytics_report.py \
  "${RESULTS_DIR}/evaluation_20251202_213404_migrated.xlsx" \
  "${RESULTS_DIR}/evaluation_20251215_144928.xlsx" \
  "${RESULTS_DIR}/evaluation_20251216_041250.xlsx" \
  "${RESULTS_DIR}/evaluation_20251216_095143.xlsx" \
  --labels "Old System (Qwen)" "New VLM Only (Qwen)" "SPAI Standalone" "SPAI + VLM (Qwen)" \
  --output "${OUTPUT_DIR}/full_comparison.html"

echo "✓ Generated: ${OUTPUT_DIR}/full_comparison.html"
echo "✓ CSV data: ${OUTPUT_DIR}/full_comparison.csv"
echo

# Example 2: Compare just SPAI configurations
echo "Example 2: SPAI-only comparison..."
python3 generate_analytics_report.py \
  "${RESULTS_DIR}/evaluation_20251216_041250.xlsx" \
  "${RESULTS_DIR}/evaluation_20251216_095143.xlsx" \
  --labels "SPAI Standalone" "SPAI + VLM" \
  --output "${OUTPUT_DIR}/spai_comparison.html"

echo "✓ Generated: ${OUTPUT_DIR}/spai_comparison.html"
echo

# Example 3: Using wildcards
echo "Example 3: Using wildcards to select files..."
python3 generate_analytics_report.py \
  ${RESULTS_DIR}/evaluation_20251216_*.xlsx \
  --output "${OUTPUT_DIR}/recent_results.html"

echo "✓ Generated: ${OUTPUT_DIR}/recent_results.html"
echo

echo "=========================================="
echo "All reports generated successfully!"
echo "=========================================="
echo
echo "To view reports, open them in your browser:"
echo "  xdg-open ${OUTPUT_DIR}/full_comparison.html"
echo
echo "Or launch interactive dashboard:"
echo "  streamlit run analytics.py"
echo
