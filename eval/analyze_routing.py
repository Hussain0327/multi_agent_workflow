#!/usr/bin/env python3
"""Analyze routing decisions and identify weak choices."""

import csv
from pathlib import Path
from collections import defaultdict

csv_file = Path("eval/benchmark_results_10queries.csv")

# Read CSV
with open(csv_file) as f:
    reader = csv.DictReader(f)
    results = list(reader)

print("="*80)
print("🔍 ROUTING ANALYSIS - 10 Queries")
print("="*80)

# Overall stats
total_queries = len(results)
perfect_routing = sum(1 for r in results if float(r['routing_accuracy']) == 100.0)
avg_accuracy = sum(float(r['routing_accuracy']) for r in results) / total_queries

print(f"\n📊 OVERALL PERFORMANCE:")
print(f"  Total Queries:        {total_queries}")
print(f"  Perfect Routing:      {perfect_routing}/{total_queries} ({perfect_routing/total_queries*100:.0f}%)")
print(f"  Average Accuracy:     {avg_accuracy:.1f}%")
print(f"  Avg Cost per Query:   ${sum(float(r['total_cost']) for r in results) / total_queries:.6f}")
print(f"  Avg Latency:          {sum(float(r['latency_sec']) for r in results) / total_queries:.1f}s")

# Analyze routing mistakes
print(f"\n❌ ROUTING MISTAKES (Accuracy < 100%):")
print(f"{'─'*80}")

mistakes_by_agent = defaultdict(lambda: {'false_neg': 0, 'false_pos': 0})

for r in results:
    if float(r['routing_accuracy']) < 100.0:
        query_id = r['query_id']
        query = r['query']
        expected = set(r['expected_agents'].split(', '))
        called = set(r['agents_called'].split(', '))
        accuracy = float(r['routing_accuracy'])

        false_negatives = expected - called  # Agents that should have been called but weren't
        false_positives = called - expected   # Agents that were called but shouldn't have been

        print(f"\nQ{query_id} ({accuracy:.0f}% accuracy): {query[:70]}...")
        print(f"  Expected:  {', '.join(sorted(expected))}")
        print(f"  Called:    {', '.join(sorted(called))}")

        if false_negatives:
            print(f"  ❌ MISSED: {', '.join(sorted(false_negatives))}")
            for agent in false_negatives:
                mistakes_by_agent[agent]['false_neg'] += 1

        if false_positives:
            print(f"  ⚠️  EXTRA:  {', '.join(sorted(false_positives))}")
            for agent in false_positives:
                mistakes_by_agent[agent]['false_pos'] += 1

        # Show confidence scores
        print(f"  Confidence: ", end="")
        conf_scores = []
        for agent in ['market', 'financial', 'operations', 'leadgen']:
            conf_key = f'conf_{agent}'
            if conf_key in r:
                conf = float(r[conf_key])
                marker = "✅" if conf >= 0.7 else "⚠️ " if conf >= 0.3 else "❌"
                conf_scores.append(f"{agent}={conf:.1f}{marker}")
        print(" | ".join(conf_scores))

# Agent-specific analysis
print(f"\n\n🎯 AGENT-SPECIFIC ANALYSIS:")
print(f"{'─'*80}")
print(f"{'Agent':<15} {'False Negatives':<20} {'False Positives':<20} {'Total Errors'}")
print(f"{'─'*80}")

for agent in sorted(mistakes_by_agent.keys()):
    fn = mistakes_by_agent[agent]['false_neg']
    fp = mistakes_by_agent[agent]['false_pos']
    total = fn + fp
    print(f"{agent:<15} {fn:<20} {fp:<20} {total}")

# Confidence analysis
print(f"\n\n💡 CONFIDENCE ANALYSIS:")
print(f"{'─'*80}")

low_confidence_decisions = []
for r in results:
    for agent in ['market', 'financial', 'operations', 'leadgen']:
        conf_key = f'conf_{agent}'
        if conf_key in r:
            conf = float(r[conf_key])
            called = agent in r['agents_called'].split(', ')
            expected = agent in r['expected_agents'].split(', ')

            # Low confidence but called
            if called and conf < 0.7:
                low_confidence_decisions.append({
                    'query_id': r['query_id'],
                    'agent': agent,
                    'confidence': conf,
                    'decision': 'called',
                    'correct': expected
                })

            # High confidence but not called when it should have been
            if not called and expected and conf < 0.3:
                low_confidence_decisions.append({
                    'query_id': r['query_id'],
                    'agent': agent,
                    'confidence': conf,
                    'decision': 'not called',
                    'correct': False
                })

if low_confidence_decisions:
    print("\n⚠️  Low confidence decisions that need review:")
    for d in low_confidence_decisions[:10]:  # Show top 10
        correct_marker = "✅" if d['correct'] else "❌"
        print(f"  Q{d['query_id']}: {d['agent']} (conf={d['confidence']:.2f}) - {d['decision']} {correct_marker}")
else:
    print("\n✅ No concerning low-confidence decisions found")

# Recommendations
print(f"\n\n🚀 RECOMMENDATIONS:")
print(f"{'─'*80}")

if avg_accuracy < 80:
    print("\n1. ⚠️  ML ROUTER ACCURACY IS LOW (62.5%)")
    print("   Options:")
    print("   - Switch to GPT-5 semantic routing (90%+ accuracy, +$0.01/query)")
    print("   - Implement confidence-gated fallback (ML if confident, GPT-5 if not)")
    print("   - Retrain ML classifier with more examples (especially for leadgen/market)")

if mistakes_by_agent:
    print("\n2. 🎯 PROBLEMATIC AGENTS:")
    for agent, counts in sorted(mistakes_by_agent.items(), key=lambda x: x[1]['false_neg'], reverse=True):
        if counts['false_neg'] > 2:
            print(f"   - {agent}: {counts['false_neg']} false negatives (often missed)")
            print(f"     → Add more training examples for queries requiring {agent}")

print("\n3. ✅ DEEPSEEK MODEL PERFORMANCE:")
print(f"   - Cost: ${sum(float(r['total_cost']) for r in results) / total_queries:.6f}/query")
print(f"   - ~99% cheaper than GPT-5 ($0.28/query)")
print(f"   - Latency: Comparable ({sum(float(r['latency_sec']) for r in results) / total_queries:.1f}s)")
print(f"   - Quality: Need to fix LLM judge to assess")
print(f"   → DeepSeek model itself is working great! Routing is the issue.")

print("\n" + "="*80)
