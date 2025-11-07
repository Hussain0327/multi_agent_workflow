#!/bin/bash

# Auto-analysis script that runs when eval completes

echo "Waiting for evaluation to complete..."

# Wait for eval process to finish
while ps aux | grep "eval/benchmark.py" | grep -v grep > /dev/null; do
    sleep 60
    echo "Still running... $(date +%H:%M:%S)"
done

echo "Evaluation complete! Running analysis..."

# Run statistical analysis
python3 scripts/run_analysis.py

# Update WEEK2_COMPLETE.md with results
echo "Updating Week 2 report..."

# Check if results exist
if ls eval/results_*.json 1> /dev/null 2>&1; then
    echo "✓ Evaluation results found"
    echo "✓ Analysis complete"
    echo ""
    echo "Next steps:"
    echo "1. Review eval/ANALYSIS_REPORT.md"
    echo "2. Update docs/WEEK2_COMPLETE.md with findings"
    echo "3. git add . && git commit"
else
    echo "✗ No evaluation results found"
fi
