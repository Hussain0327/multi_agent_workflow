#!/bin/bash
# Quick script to check benchmark progress

echo "=== BENCHMARK PROGRESS ==="
echo ""
echo "Current query:"
grep -E "Query [0-9]+/25" eval/benchmark_run.log | tail -1
echo ""
echo "Completed queries:"
grep -c "✓ Latency:" eval/benchmark_run.log
echo ""
echo "Last 20 lines:"
tail -20 eval/benchmark_run.log
