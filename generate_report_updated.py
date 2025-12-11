"""
Deepfake Detection Evaluation Report Generator (Updated)
Generates comprehensive performance analysis with visualizations
Includes: HADR, MilitaryConflict, MilitaryShowcase domains + Qwen3 VL 32B results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
os.makedirs('analysis_output', exist_ok=True)

print("="*80)
print("DEEPFAKE DETECTION EVALUATION REPORT GENERATOR")
print("="*80)

# ========== LOAD AND MERGE DATA FROM BOTH EVALUATIONS ==========
print("\nLoading evaluation data...")

# Load first evaluation (3 models)
eval_file_1 = 'results/evaluation_20251126_204434.xlsx'
metrics_df_1 = pd.read_excel(eval_file_1, sheet_name='metrics')
predictions_df_1 = pd.read_excel(eval_file_1, sheet_name='predictions')

# Load second evaluation (Qwen3 VL)
eval_file_2 = 'results/evaluation_20251202_213404.xlsx'
metrics_df_2 = pd.read_excel(eval_file_2, sheet_name='metrics')
predictions_df_2 = pd.read_excel(eval_file_2, sheet_name='predictions')

# Merge metrics
metrics_df = pd.concat([metrics_df_1, metrics_df_2], ignore_index=True)

# Merge predictions
predictions_df = pd.concat([predictions_df_1, predictions_df_2], ignore_index=True)

print(f"✓ Loaded data from both evaluations")
print(f"  Total models: {len(metrics_df)}")
print(f"  Models: {', '.join(metrics_df['model_name'].unique())}")

# ========== EXTRACT METADATA FROM FILENAMES ==========

def extract_category(filename):
    """Extract AIG/AIM/REAL category"""
    if '_AIG_' in filename:
        return 'AIG'
    elif '_AIM_' in filename:
        return 'AIM'
    elif '_REAL_' in filename:
        return 'REAL'
    else:
        return 'UNKNOWN'

def extract_domain(filename):
    """Extract HADR/MilitaryConflict/MilitaryShowcase domain"""
    if 'HADR_' in filename:
        return 'HADR'
    elif 'MilitaryConflict_' in filename:
        return 'MilitaryConflict'
    elif 'MilitaryShowcase_' in filename:
        return 'MilitaryShowcase'
    return 'Unknown'

def extract_scenario(filename):
    """Extract specific scenario type"""
    # Check LIVEFIRE before FIRE to avoid false matches
    if 'LIVEFIRE' in filename:
        return 'LIVEFIRE'
    elif 'FIRE' in filename:
        return 'FIRE'
    elif 'EARTHQUAKE' in filename:
        return 'EARTHQUAKE'
    elif 'FLOOD' in filename:
        return 'FLOOD'
    elif 'STORM' in filename:
        return 'STORM'
    elif 'AIRSTRIKE' in filename:
        return 'AIRSTRIKE'
    elif 'CONVOY' in filename:
        return 'CONVOY'
    elif 'GROUNDBATTLE' in filename:
        return 'GROUNDBATTLE'
    elif 'URBANDAMAGE' in filename:
        return 'URBANDAMAGE'
    elif 'PARADE' in filename:
        return 'PARADE'
    elif 'MARCHING' in filename:
        return 'MARCHING'
    elif 'NAVIGATION' in filename:
        return 'NAVIGATION'
    return 'OTHER'

def extract_time(filename):
    """Extract day/night"""
    if '_DAY_' in filename:
        return 'DAY'
    elif '_NIGHT_' in filename:
        return 'NIGHT'
    return 'UNKNOWN'

predictions_df['category'] = predictions_df['filename'].apply(extract_category)
predictions_df['domain'] = predictions_df['filename'].apply(extract_domain)
predictions_df['scenario'] = predictions_df['filename'].apply(extract_scenario)
predictions_df['time_of_day'] = predictions_df['filename'].apply(extract_time)

# ========== DATASET SUMMARY ==========
print("\n" + "="*80)
print("DATASET SUMMARY")
print("="*80)

unique_images = predictions_df['filename'].nunique()

dataset_summary = {
    'Total Images': unique_images,
    'Real Images': len(predictions_df[predictions_df['category'] == 'REAL']['filename'].unique()),
    'AI-Generated (AIG)': len(predictions_df[predictions_df['category'] == 'AIG']['filename'].unique()),
    'AI-Manipulated (AIM)': len(predictions_df[predictions_df['category'] == 'AIM']['filename'].unique()),
    'Models Evaluated': len(metrics_df),
    'Runs per Image': predictions_df['total_runs'].iloc[0] if len(predictions_df) > 0 else 0,
}

print(f"\nDataset Composition:")
for key, value in dataset_summary.items():
    print(f"  {key}: {value}")

# Category distribution
category_counts = predictions_df.groupby('category')['filename'].nunique()
print(f"\nCategory Distribution:")
for cat, count in category_counts.items():
    print(f"  {cat}: {count} ({count/unique_images*100:.1f}%)")

# Domain distribution
domain_counts = predictions_df.groupby('domain')['filename'].nunique()
print(f"\nDomain Distribution:")
for domain, count in domain_counts.items():
    print(f"  {domain}: {count} ({count/unique_images*100:.1f}%)")

# Scenario distribution
scenario_counts = predictions_df.groupby('scenario')['filename'].nunique()
print(f"\nScenario Distribution:")
for scenario, count in sorted(scenario_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {scenario}: {count}")

# Time distribution
time_counts = predictions_df.groupby('time_of_day')['filename'].nunique()
print(f"\nTime of Day Distribution:")
for time, count in time_counts.items():
    print(f"  {time}: {count}")

# ========== CALCULATE AIG vs AIM METRICS ==========
print("\n" + "="*80)
print("AIG vs AIM PERFORMANCE ANALYSIS (ALL MODELS)")
print("="*80)

models = metrics_df['model_name'].unique()
detailed_results = []

for model in models:
    model_data = predictions_df[predictions_df['model_name'] == model]

    # Overall metrics
    total = len(model_data)
    correct = model_data['correct'].sum()
    accuracy = correct / total if total > 0 else 0

    # AIG metrics
    aig_data = model_data[model_data['category'] == 'AIG']
    aig_total = len(aig_data)
    aig_detected = (aig_data['consensus_label'] == 'AI Generated').sum()
    aig_recall = aig_detected / aig_total if aig_total > 0 else 0

    # AIM metrics
    aim_data = model_data[model_data['category'] == 'AIM']
    aim_total = len(aim_data)
    aim_detected = (aim_data['consensus_label'] == 'AI Generated').sum()
    aim_recall = aim_detected / aim_total if aim_total > 0 else 0

    # REAL metrics
    real_data = model_data[model_data['category'] == 'REAL']
    real_total = len(real_data)
    real_correct = (real_data['consensus_label'] == 'Real').sum()
    real_specificity = real_correct / real_total if real_total > 0 else 0
    false_positives = real_total - real_correct

    # Get confusion matrix values
    model_metrics = metrics_df[metrics_df['model_name'] == model].iloc[0]

    detailed_results.append({
        'Model': model,
        'Overall Accuracy': accuracy,
        'Precision': model_metrics['precision'],
        'Recall': model_metrics['recall'],
        'F1 Score': model_metrics['f1'],
        'TP': int(model_metrics['tp']),
        'TN': int(model_metrics['tn']),
        'FP': int(model_metrics['fp']),
        'FN': int(model_metrics['fn']),
        'AIG Recall': aig_recall,
        'AIG Detected': aig_detected,
        'AIG Total': aig_total,
        'AIM Recall': aim_recall,
        'AIM Detected': aim_detected,
        'AIM Total': aim_total,
        'Real Specificity': real_specificity,
        'False Positives': false_positives,
    })

    print(f"\n{model}")
    print("-" * 80)
    print(f"Overall Accuracy: {accuracy:.2%}")
    print(f"Precision: {model_metrics['precision']:.2%} | Recall: {model_metrics['recall']:.2%} | F1: {model_metrics['f1']:.2%}")
    print(f"\nAIG Detection: {aig_detected}/{aig_total} ({aig_recall:.1%})")
    print(f"AIM Detection: {aim_detected}/{aim_total} ({aim_recall:.1%})")
    print(f"Real Specificity: {real_correct}/{real_total} ({real_specificity:.1%})")
    print(f"False Positives: {false_positives}")

results_df = pd.DataFrame(detailed_results)
# Sort by overall accuracy descending
results_df = results_df.sort_values('Overall Accuracy', ascending=False).reset_index(drop=True)

# ========== VISUALIZATION 1: Model Comparison (4 models now) ==========
print("\nGenerating visualizations...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Deepfake Detection Model Performance Comparison (4 Models)', fontsize=18, fontweight='bold')

# 1. Overall Metrics Comparison
ax1 = axes[0]
metrics_to_plot = ['Overall Accuracy', 'Precision', 'Recall', 'F1 Score']
x = np.arange(len(results_df))
width = 0.2

for i, metric in enumerate(metrics_to_plot):
    ax1.bar(x + i*width, results_df[metric], width, label=metric)

ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels(results_df['Model'], rotation=20, ha='right', fontsize=10)
ax1.legend()
ax1.set_ylim([0, 1.1])
ax1.grid(axis='y', alpha=0.3)

# 2. AIG vs AIM Detection Rates
ax2 = axes[1]
x = np.arange(len(results_df))
width = 0.35

bars1 = ax2.bar(x - width/2, results_df['AIG Recall'], width, label='AIG (Fully AI-Generated)', color='#ff7f0e')
bars2 = ax2.bar(x + width/2, results_df['AIM Recall'], width, label='AIM (AI-Manipulated)', color='#2ca02c')

ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
ax2.set_ylabel('Detection Rate (Recall)', fontsize=12, fontweight='bold')
ax2.set_title('AIG vs AIM Detection Performance', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(results_df['Model'], rotation=20, ha='right', fontsize=10)
ax2.legend()
ax2.set_ylim([0, 1.1])
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontsize=8)

# 3. Category Performance Breakdown (All 3 metrics)
ax3 = axes[2]
categories = ['AIG Recall', 'AIM Recall', 'Real Specificity']
colors = ['#ff7f0e', '#2ca02c', '#1f77b4']

x = np.arange(len(results_df))
width = 0.25

for i, cat in enumerate(categories):
    ax3.bar(x + i*width, results_df[cat], width, label=cat.replace(' Recall', '').replace(' Specificity', ''),
            color=colors[i])

ax3.set_xlabel('Model', fontsize=12, fontweight='bold')
ax3.set_ylabel('Performance Rate', fontsize=12, fontweight='bold')
ax3.set_title('Performance by Image Category', fontsize=14, fontweight='bold')
ax3.set_xticks(x + width)
ax3.set_xticklabels(results_df['Model'], rotation=20, ha='right', fontsize=10)
ax3.legend()
ax3.set_ylim([0, 1.1])
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('analysis_output/1_model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: analysis_output/1_model_comparison.png")
plt.close()

# ========== VISUALIZATION 2: Confusion Matrices for All Models ==========
n_models = len(results_df)
fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
fig.suptitle('Confusion Matrices - All Models', fontsize=18, fontweight='bold')

if n_models == 1:
    axes = [axes]

for idx, model in enumerate(results_df['Model']):
    model_data = results_df[results_df['Model'] == model].iloc[0]

    cm_data = np.array([
        [model_data['TN'], model_data['FP']],
        [model_data['FN'], model_data['TP']]
    ])

    # Calculate percentages
    cm_pct = cm_data / cm_data.sum() * 100

    # Create annotations with both count and percentage
    annot_labels = np.array([
        [f"{cm_data[0,0]}\n({cm_pct[0,0]:.1f}%)", f"{cm_data[0,1]}\n({cm_pct[0,1]:.1f}%)"],
        [f"{cm_data[1,0]}\n({cm_pct[1,0]:.1f}%)", f"{cm_data[1,1]}\n({cm_pct[1,1]:.1f}%)"]
    ])

    sns.heatmap(cm_data, annot=annot_labels, fmt='', cmap='RdYlGn', ax=axes[idx],
                xticklabels=['Predicted\nReal', 'Predicted\nAI'],
                yticklabels=['Actual\nReal', 'Actual\nAI'],
                cbar_kws={'label': 'Count'}, vmin=0, vmax=50)

    axes[idx].set_title(f'{model}\nAccuracy: {model_data["Overall Accuracy"]:.1%}',
                       fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('analysis_output/2_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved: analysis_output/2_confusion_matrices.png")
plt.close()

# ========== VISUALIZATION 3: Dataset Composition ==========
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Dataset Composition Analysis', fontsize=18, fontweight='bold')

# Category distribution
ax1 = axes[0, 0]
category_data = predictions_df.groupby('category')['filename'].nunique()
colors_cat = ['#1f77b4', '#ff7f0e', '#2ca02c']
wedges, texts, autotexts = ax1.pie(category_data.values, labels=category_data.index,
                                     autopct='%1.1f%%', colors=colors_cat, startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax1.set_title('Image Category Distribution\n(AIG/AIM/REAL)', fontsize=12, fontweight='bold')

# Domain distribution
ax2 = axes[0, 1]
domain_data = predictions_df.groupby('domain')['filename'].nunique()
colors_domain = ['#e377c2', '#bcbd22', '#17becf']
wedges, texts, autotexts = ax2.pie(domain_data.values, labels=domain_data.index,
                                     autopct='%1.1f%%', colors=colors_domain, startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax2.set_title('Domain Distribution\n(HADR/MilitaryConflict/MilitaryShowcase)', fontsize=12, fontweight='bold')

# Scenario distribution
ax3 = axes[1, 0]
scenario_data = predictions_df.groupby('scenario')['filename'].nunique().sort_values(ascending=True)
ax3.barh(range(len(scenario_data)), scenario_data.values, color='#8c564b')
ax3.set_yticks(range(len(scenario_data)))
ax3.set_yticklabels(scenario_data.index, fontsize=9)
ax3.set_xlabel('Number of Images', fontsize=10, fontweight='bold')
ax3.set_title('Scenario Distribution', fontsize=12, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# Add value labels
for i, v in enumerate(scenario_data.values):
    ax3.text(v + 0.3, i, str(v), va='center', fontweight='bold', fontsize=9)

# Time of day distribution
ax4 = axes[1, 1]
time_data = predictions_df.groupby('time_of_day')['filename'].nunique()
colors_time = ['#e377c2', '#7f7f7f']
bars = ax4.bar(time_data.index, time_data.values, color=colors_time)
ax4.set_ylabel('Number of Images', fontsize=10, fontweight='bold')
ax4.set_title('Day vs Night Distribution', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('analysis_output/3_dataset_composition.png', dpi=300, bbox_inches='tight')
print("✓ Saved: analysis_output/3_dataset_composition.png")
plt.close()

# ========== VISUALIZATION 4: Performance Summary Table as Image ==========
fig, ax = plt.subplots(figsize=(18, 4 + len(results_df)*0.5))
ax.axis('tight')
ax.axis('off')

# Create table data
table_data = []
for _, row in results_df.iterrows():
    table_data.append([
        row['Model'],
        f"{row['Overall Accuracy']:.1%}",
        f"{row['Precision']:.1%}",
        f"{row['Recall']:.1%}",
        f"{row['F1 Score']:.1%}",
        f"{row['AIG Recall']:.1%}",
        f"{row['AIM Recall']:.1%}",
        f"{row['Real Specificity']:.1%}",
        f"{int(row['TP'])}/{int(row['TN'])}/{int(row['FP'])}/{int(row['FN'])}"
    ])

headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1',
           'AIG Recall', 'AIM Recall', 'Real Spec.', 'TP/TN/FP/FN']

table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center',
                colWidths=[0.18, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(len(headers)):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white')

# Style data rows with alternating colors, highlight best model
for i in range(1, len(table_data) + 1):
    for j in range(len(headers)):
        cell = table[(i, j)]
        if i == 1:  # Best model (first row after sorting)
            cell.set_facecolor('#C6E0B4')  # Light green
        elif i % 2 == 0:
            cell.set_facecolor('#E7E6E6')
        else:
            cell.set_facecolor('#F2F2F2')

plt.title('Model Performance Summary Table', fontsize=16, fontweight='bold', pad=20)
plt.savefig('analysis_output/4_performance_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved: analysis_output/4_performance_table.png")
plt.close()

# ========== VISUALIZATION 5: Detection Rate Comparison ==========
fig, ax = plt.subplots(figsize=(14, 8))

categories = results_df['Model'].tolist()
aig_rates = results_df['AIG Recall'].tolist()
aim_rates = results_df['AIM Recall'].tolist()
real_rates = results_df['Real Specificity'].tolist()

x = np.arange(len(categories))
width = 0.25

bars1 = ax.bar(x - width, aig_rates, width, label='AIG Detection Rate', color='#ff7f0e')
bars2 = ax.bar(x, aim_rates, width, label='AIM Detection Rate', color='#2ca02c')
bars3 = ax.bar(x + width, real_rates, width, label='Real Identification Rate', color='#1f77b4')

ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_ylabel('Rate', fontsize=14, fontweight='bold')
ax.set_title('Detection & Identification Rates by Category', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11, rotation=20, ha='right')
ax.legend(fontsize=11)
ax.set_ylim([0, 1.15])
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('analysis_output/5_detection_rates.png', dpi=300, bbox_inches='tight')
print("✓ Saved: analysis_output/5_detection_rates.png")
plt.close()

# ========== VISUALIZATION 6: Confusion Matrix Definitions (unchanged) ==========
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

confusion_text = """
CONFUSION MATRIX DEFINITIONS (Deepfake Detection Context)

TRUE POSITIVE (TP)
Model correctly identifies an AI-generated/manipulated image as "AI Generated"
→ Successfully caught a fake that could spread misinformation
→ Example: AI-generated flood photo is correctly flagged as fake

TRUE NEGATIVE (TN)
Model correctly identifies an authentic image as "Real"
→ Properly verified genuine disaster footage, allowing it to be trusted
→ Example: Real earthquake photo is correctly identified as authentic

FALSE POSITIVE (FP) ⚠️
Model incorrectly identifies an authentic image as "AI Generated"
→ Falsely discredited legitimate disaster footage - damages trust & credibility
→ Example: Real fire photo wrongly flagged as fake, causing doubt about actual emergencies
→ Impact: Can lead to "crying wolf" syndrome where users ignore warnings

FALSE NEGATIVE (FN) 🚨
Model incorrectly identifies a fake image as "Real"
→ Failed to detect misinformation - allows fakes to spread unchecked
→ Example: AI-generated disaster scene accepted as real, potentially causing panic
→ Impact: Can cause public panic, misallocate resources, or undermine response efforts

TRADE-OFF ANALYSIS
High FP Rate: May discourage legitimate reporting but catches more fakes
High FN Rate: Allows misinformation to spread but doesn't wrongly discredit real images

Optimal balance depends on use case:
• Emergency responders → Prefer low FP (don't want to miss real disasters)
• Social media moderation → Prefer low FN (can't allow fakes to go viral)
"""

ax.text(0.5, 0.5, confusion_text, ha='center', va='center', fontsize=11,
        family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.title('Confusion Matrix Definitions', fontsize=18, fontweight='bold', pad=20)
plt.savefig('analysis_output/6_confusion_definitions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: analysis_output/6_confusion_definitions.png")
plt.close()

# ========== VISUALIZATION 7: Domain Performance Breakdown ==========
print("\nGenerating domain-specific analysis...")

# Calculate per-domain performance for each model
domain_performance = []

for model in models:
    model_data = predictions_df[predictions_df['model_name'] == model]

    for domain in ['HADR', 'MilitaryConflict', 'MilitaryShowcase']:
        domain_data = model_data[model_data['domain'] == domain]
        if len(domain_data) == 0:
            continue

        total = len(domain_data)
        correct = domain_data['correct'].sum()
        accuracy = correct / total if total > 0 else 0

        # Category breakdown within domain
        aig_data = domain_data[domain_data['category'] == 'AIG']
        aig_detected = (aig_data['consensus_label'] == 'AI Generated').sum() if len(aig_data) > 0 else 0
        aig_total = len(aig_data)

        aim_data = domain_data[domain_data['category'] == 'AIM']
        aim_detected = (aim_data['consensus_label'] == 'AI Generated').sum() if len(aim_data) > 0 else 0
        aim_total = len(aim_data)

        real_data = domain_data[domain_data['category'] == 'REAL']
        real_correct = (real_data['consensus_label'] == 'Real').sum() if len(real_data) > 0 else 0
        real_total = len(real_data)

        domain_performance.append({
            'Model': model,
            'Domain': domain,
            'Accuracy': accuracy,
            'AIG Detected': f'{aig_detected}/{aig_total}',
            'AIM Detected': f'{aim_detected}/{aim_total}',
            'Real Correct': f'{real_correct}/{real_total}',
        })

domain_perf_df = pd.DataFrame(domain_performance)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Model Performance by Domain', fontsize=18, fontweight='bold')

domains = ['HADR', 'MilitaryConflict', 'MilitaryShowcase']
for idx, domain in enumerate(domains):
    domain_subset = domain_perf_df[domain_perf_df['Domain'] == domain]

    ax = axes[idx]
    x = np.arange(len(domain_subset))
    bars = ax.bar(x, domain_subset['Accuracy'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(domain_subset)])

    ax.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title(f'{domain}', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(domain_subset['Model'], rotation=45, ha='right', fontsize=9)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, domain_subset['Accuracy'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('analysis_output/7_domain_performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: analysis_output/7_domain_performance.png")
plt.close()

# ========== EXPORT DETAILED RESULTS TO CSV ==========
results_df.to_csv('analysis_output/detailed_results.csv', index=False)
print("✓ Saved: analysis_output/detailed_results.csv")

# Export domain performance
domain_perf_df.to_csv('analysis_output/domain_performance.csv', index=False)
print("✓ Saved: analysis_output/domain_performance.csv")

# Save dataset summary
with open('analysis_output/dataset_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("DATASET SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write("Dataset Composition:\n")
    for key, value in dataset_summary.items():
        f.write(f"  {key}: {value}\n")
    f.write(f"\nCategory Distribution:\n")
    for cat, count in category_counts.items():
        f.write(f"  {cat}: {count} ({count/unique_images*100:.1f}%)\n")
    f.write(f"\nDomain Distribution:\n")
    for domain, count in domain_counts.items():
        f.write(f"  {domain}: {count} ({count/unique_images*100:.1f}%)\n")
    f.write(f"\nScenario Distribution:\n")
    for scenario, count in sorted(scenario_counts.items(), key=lambda x: x[1], reverse=True):
        f.write(f"  {scenario}: {count}\n")
    f.write(f"\nTime of Day Distribution:\n")
    for time, count in time_counts.items():
        f.write(f"  {time}: {count}\n")

print("✓ Saved: analysis_output/dataset_summary.txt")

print("\n" + "="*80)
print("REPORT GENERATION COMPLETE")
print("="*80)
print(f"\nAll outputs saved to: analysis_output/")
print("\nGenerated files:")
print("  1. 1_model_comparison.png - Overall performance metrics (4 models)")
print("  2. 2_confusion_matrices.png - Confusion matrices for all models")
print("  3. 3_dataset_composition.png - Dataset breakdown with domains")
print("  4. 4_performance_table.png - Summary table as image")
print("  5. 5_detection_rates.png - Detection rates by category")
print("  6. 6_confusion_definitions.png - Confusion matrix explanations")
print("  7. 7_domain_performance.png - Performance by domain (NEW)")
print("  8. detailed_results.csv - Raw data export")
print("  9. domain_performance.csv - Domain-specific metrics (NEW)")
print(" 10. dataset_summary.txt - Dataset statistics")
print("\nNote: Reports with corrected domain breakdown (HADR/MilitaryConflict/MilitaryShowcase)")
print("      and including Qwen3 VL 32B results")
