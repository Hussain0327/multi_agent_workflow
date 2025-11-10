#!/usr/bin/env python3
"""Parse benchmark log and create CSV."""

import re
import csv
from pathlib import Path

log_file = Path("eval/benchmark_run.log")
output_csv = Path("eval/benchmark_results_10queries.csv")

# Read log file
log_content = log_file.read_text()

# Split by query sections
query_sections = re.split(r'Query (\d+)/25', log_content)[1:]  # Skip before first query

results = []

for i in range(0, len(query_sections), 2):
    if i + 1 >= len(query_sections):
        break

    query_num = query_sections[i]
    query_content = query_sections[i + 1]

    # Skip if no completion marker
    if "✓ Latency:" not in query_content:
        continue

    result = {"query_id": int(query_num)}

    # Extract query text
    query_match = re.search(r'\[Query \d+\] (.+?)\.\.\.', query_content)
    if query_match:
        result["query"] = query_match.group(1)

    # Extract ML Router decision
    router_match = re.search(r'ML Router: \[([^\]]+)\]', query_content)
    if router_match:
        agents_called = [a.strip().strip("'") for a in router_match.group(1).split(',')]
        result["agents_called"] = ", ".join(agents_called)
        result["num_agents"] = len(agents_called)

    # Extract confidence scores
    confidence_match = re.search(r"Confidence: ({[^}]+})", query_content)
    if confidence_match:
        confidence_str = confidence_match.group(1)
        # Parse confidence scores
        for agent in ["market", "financial", "operations", "leadgen"]:
            conf_match = re.search(rf"'{agent}': ([\d.]+)", confidence_str)
            if conf_match:
                result[f"conf_{agent}"] = float(conf_match.group(1))

    # Extract routing accuracy
    accuracy_match = re.search(r'Routing Accuracy: ([\d.]+)%', query_content)
    if accuracy_match:
        result["routing_accuracy"] = float(accuracy_match.group(1))

    # Extract latency
    latency_match = re.search(r'Latency: ([\d.]+)s', query_content)
    if latency_match:
        result["latency_sec"] = float(latency_match.group(1))

    # Extract cost
    cost_match = re.search(r'Estimated Cost: \$([\d.]+)', query_content)
    if cost_match:
        result["total_cost"] = float(cost_match.group(1))

    # Extract agent costs
    agent_cost_matches = re.findall(r'- (\w+): ([\w-]+) \(\$([\d.]+)\)', query_content)
    for j, (agent_name, model, cost) in enumerate(agent_cost_matches, 1):
        result[f"agent{j}_name"] = agent_name
        result[f"agent{j}_model"] = model
        result[f"agent{j}_cost"] = float(cost)

    results.append(result)

# Load test queries to get expected agents
import json
with open("eval/test_queries.json") as f:
    test_data = json.load(f)
    queries_info = {q["id"]: q for q in test_data["queries"]}

# Add expected agents
for result in results:
    query_id = result["query_id"]
    if query_id in queries_info:
        expected = queries_info[query_id].get("expected_agents", [])
        result["expected_agents"] = ", ".join(expected)
        result["category"] = queries_info[query_id].get("category", "")
        result["complexity"] = queries_info[query_id].get("complexity", "")

# Write CSV
if results:
    # Collect all possible fieldnames from all results
    all_fieldnames = set()
    for r in results:
        all_fieldnames.update(r.keys())
    fieldnames = sorted(all_fieldnames)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Created CSV with {len(results)} queries: {output_csv}")

    # Print summary
    print("\n📊 SUMMARY:")
    print(f"Total queries: {len(results)}")
    print(f"Avg routing accuracy: {sum(r['routing_accuracy'] for r in results) / len(results):.1f}%")
    print(f"Avg cost: ${sum(r['total_cost'] for r in results) / len(results):.6f}")
    print(f"Avg latency: {sum(r['latency_sec'] for r in results) / len(results):.1f}s")
    print(f"\nRouting accuracy by query:")
    for r in results:
        correct = "✅" if r['routing_accuracy'] == 100.0 else "❌"
        print(f"  Q{r['query_id']}: {r['routing_accuracy']:5.1f}% {correct} - {r.get('query', '')[:60]}")
else:
    print("❌ No completed queries found in log")
