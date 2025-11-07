# Scripts Directory

Utility scripts for training, evaluation, and analysis.

## Training Scripts

**export_langsmith_data.py** - Export training data from LangSmith traces
- Output: models/training_data.json
- 105 training examples, 22 validation examples

**add_training_examples.py** - Add boundary examples to training data
- Adds 20 examples for weak agents (leadgen, operations)
- Run before retraining

**quick_retrain.py** - Retrain ML routing classifier
- Uses models/training_data.json
- 5 epochs training
- Output: models/routing_classifier.pkl

**check_accuracy.py** - Test ML model accuracy on validation set
- Quick accuracy check after training

## Evaluation Scripts

**run_analysis.py** - Run statistical analysis on evaluation results
- Input: eval/results_*.json
- Output: eval/ANALYSIS_REPORT.md
- T-tests, Cohen's d, significance testing

**auto_analyze.sh** - Auto-run analysis when eval completes
- Waits for benchmark to finish
- Runs statistical analysis
- Updates documentation

## Model Inspection

**try_load_model.py** - Test loading the pickled model
- Debug script for model inspection

## Usage

```bash
# Train ML routing classifier
python3 scripts/add_training_examples.py
python3 scripts/quick_retrain.py
python3 scripts/check_accuracy.py

# Run evaluation
python3 eval/benchmark.py --mode both --num-queries 25

# Analyze results (manual)
python3 scripts/run_analysis.py

# Analyze results (auto)
./scripts/auto_analyze.sh &
```
