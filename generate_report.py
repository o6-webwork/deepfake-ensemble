"""
Deepfake Detection Evaluation Report Generator
Generates comprehensive performance analysis with visualizations
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

# Read evaluation data
print("Loading evaluation data...")
eval_file = 'results/evaluation_20251126_204434.xlsx'
metrics_df = pd.read_excel(eval_file, sheet_name='metrics')
predictions_df = pd.read_excel(eval_file, sheet_name='predictions')

# Extract category from filename
def extract_category(filename):
    if '_AIG_' in filename:
        return 'AIG'
    elif '_AIM_' in filename:
        return 'AIM'
    elif '_REAL_' in filename:
        return 'REAL'
    else:
        return 'UNKNOWN'

def extract_scenario(filename):
    """Extract disaster scenario type"""
    scenarios = ['EARTHQUAKE', 'FIRE', 'FLOOD', 'STORM', 'AIRSTRIKE', 'CONVOY',
                 'GROUNDBATTLE', 'URBANDAMAGE']
    for scenario in scenarios:
        if scenario in filename:
            return scenario
    return 'OTHER'

def extract_time(filename):
    """Extract day/night"""
    if '_DAY_' in filename:
        return 'DAY'
    elif '_NIGHT_' in filename:
        return 'NIGHT'
    return 'UNKNOWN'

predictions_df['category'] = predictions_df['filename'].apply(extract_category)
predictions_df['scenario'] = predictions_df['filename'].apply(extract_scenario)
predictions_df['time_of_day'] = predictions_df['filename'].apply(extract_time)

# ========== DATASET SUMMARY ==========
print("\n" + "="*80)
print("DATASET SUMMARY")
print("="*80)

dataset_summary = {
    'Total Images': len(predictions_df['filename'].unique()),
    'Real Images': len(predictions_df[predictions_df['category'] == 'REAL']['filename'].unique()),
    'AI-Generated (AIG)': len(predictions_df[predictions_df['category'] == 'AIG']['filename'].unique()),
    'AI-Manipulated (AIM)': len(predictions_df[predictions_df['category'] == 'AIM']['filename'].unique()),
    'Models Evaluated': len(predictions_df['model_name'].unique()),
    'Runs per Image': predictions_df['total_runs'].iloc[0] if len(predictions_df) > 0 else 0,
}

print(f"\nDataset Composition:")
for key, value in dataset_summary.items():
    print(f"  {key}: {value}")

# Category distribution
category_counts = predictions_df.groupby('category')['filename'].nunique()
print(f"\nCategory Distribution:")
for cat, count in category_counts.items():
    print(f"  {cat}: {count} ({count/category_counts.sum()*100:.1f}%)")

# Scenario distribution
scenario_counts = predictions_df.groupby('scenario')['filename'].nunique()
print(f"\nScenario Distribution:")
for scenario, count in scenario_counts.items():
    print(f"  {scenario}: {count}")

# Time distribution
time_counts = predictions_df.groupby('time_of_day')['filename'].nunique()
print(f"\nTime of Day Distribution:")
for time, count in time_counts.items():
    print(f"  {time}: {count}")

# ========== CALCULATE AIG vs AIM METRICS ==========
print("\n" + "="*80)
print("AIG vs AIM PERFORMANCE ANALYSIS")
print("="*80)

models = predictions_df['model_name'].unique()
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

# ========== VISUALIZATION 1: Model Comparison Bar Chart ==========
print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Deepfake Detection Model Performance Comparison', fontsize=18, fontweight='bold')

# 1. Overall Metrics Comparison
ax1 = axes[0, 0]
metrics_to_plot = ['Overall Accuracy', 'Precision', 'Recall', 'F1 Score']
x = np.arange(len(results_df))
width = 0.2

for i, metric in enumerate(metrics_to_plot):
    ax1.bar(x + i*width, results_df[metric], width, label=metric)

ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels(results_df['Model'], rotation=15, ha='right')
ax1.legend()
ax1.set_ylim([0, 1.1])
ax1.grid(axis='y', alpha=0.3)

# 2. AIG vs AIM Detection Rates
ax2 = axes[0, 1]
x = np.arange(len(results_df))
width = 0.35

bars1 = ax2.bar(x - width/2, results_df['AIG Recall'], width, label='AIG (Fully AI-Generated)', color='#ff7f0e')
bars2 = ax2.bar(x + width/2, results_df['AIM Recall'], width, label='AIM (AI-Manipulated)', color='#2ca02c')

ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
ax2.set_ylabel('Detection Rate (Recall)', fontsize=12, fontweight='bold')
ax2.set_title('AIG vs AIM Detection Performance', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(results_df['Model'], rotation=15, ha='right')
ax2.legend()
ax2.set_ylim([0, 1.1])
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontsize=9)

# 3. Confusion Matrix Heatmap for Best Model
ax3 = axes[1, 0]
best_model_idx = results_df['Overall Accuracy'].idxmax()
best_model = results_df.iloc[best_model_idx]

cm_data = np.array([
    [best_model['TN'], best_model['FP']],
    [best_model['FN'], best_model['TP']]
])

sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=['Predicted Real', 'Predicted AI'],
            yticklabels=['Actual Real', 'Actual AI'],
            cbar_kws={'label': 'Count'})
ax3.set_title(f'Confusion Matrix: {best_model["Model"]}\n(Best Overall Accuracy)',
              fontsize=14, fontweight='bold')

# 4. Category Performance Breakdown
ax4 = axes[1, 1]
categories = ['AIG Recall', 'AIM Recall', 'Real Specificity']
colors = ['#ff7f0e', '#2ca02c', '#1f77b4']

x = np.arange(len(results_df))
width = 0.25

for i, cat in enumerate(categories):
    ax4.bar(x + i*width, results_df[cat], width, label=cat.replace(' Recall', '').replace(' Specificity', ''),
            color=colors[i])

ax4.set_xlabel('Model', fontsize=12, fontweight='bold')
ax4.set_ylabel('Performance Rate', fontsize=12, fontweight='bold')
ax4.set_title('Performance by Image Category', fontsize=14, fontweight='bold')
ax4.set_xticks(x + width)
ax4.set_xticklabels(results_df['Model'], rotation=15, ha='right')
ax4.legend()
ax4.set_ylim([0, 1.1])
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('analysis_output/1_model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: analysis_output/1_model_comparison.png")
plt.close()

# ========== VISUALIZATION 2: Confusion Matrices for All Models ==========
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Confusion Matrices - All Models', fontsize=18, fontweight='bold')

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
                       fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('analysis_output/2_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved: analysis_output/2_confusion_matrices.png")
plt.close()

# ========== VISUALIZATION 3: Dataset Composition ==========
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Dataset Composition Analysis', fontsize=18, fontweight='bold')

# Category distribution
ax1 = axes[0]
category_data = predictions_df.groupby('category')['filename'].nunique()
colors_cat = ['#1f77b4', '#ff7f0e', '#2ca02c']
wedges, texts, autotexts = ax1.pie(category_data.values, labels=category_data.index,
                                     autopct='%1.1f%%', colors=colors_cat, startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax1.set_title('Image Category Distribution', fontsize=12, fontweight='bold')

# Scenario distribution
ax2 = axes[1]
scenario_data = predictions_df.groupby('scenario')['filename'].nunique().sort_values(ascending=True)
ax2.barh(range(len(scenario_data)), scenario_data.values, color='#8c564b')
ax2.set_yticks(range(len(scenario_data)))
ax2.set_yticklabels(scenario_data.index)
ax2.set_xlabel('Number of Images', fontsize=10, fontweight='bold')
ax2.set_title('Disaster Scenario Distribution', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for i, v in enumerate(scenario_data.values):
    ax2.text(v + 0.3, i, str(v), va='center', fontweight='bold')

# Time of day distribution
ax3 = axes[2]
time_data = predictions_df.groupby('time_of_day')['filename'].nunique()
colors_time = ['#e377c2', '#7f7f7f']
bars = ax3.bar(time_data.index, time_data.values, color=colors_time)
ax3.set_ylabel('Number of Images', fontsize=10, fontweight='bold')
ax3.set_title('Day vs Night Distribution', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('analysis_output/3_dataset_composition.png', dpi=300, bbox_inches='tight')
print("✓ Saved: analysis_output/3_dataset_composition.png")
plt.close()

# ========== VISUALIZATION 4: Performance Summary Table as Image ==========
fig, ax = plt.subplots(figsize=(16, 6))
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
                colWidths=[0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(len(headers)):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white')

# Style data rows with alternating colors
for i in range(1, len(table_data) + 1):
    for j in range(len(headers)):
        cell = table[(i, j)]
        if i % 2 == 0:
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
ax.set_xticklabels(categories, fontsize=11)
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

# ========== VISUALIZATION 6: Confusion Matrix Definitions ==========
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

# ========== EXPORT DETAILED RESULTS TO CSV ==========
results_df.to_csv('analysis_output/detailed_results.csv', index=False)
print("✓ Saved: analysis_output/detailed_results.csv")

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
        f.write(f"  {cat}: {count} ({count/category_counts.sum()*100:.1f}%)\n")
    f.write(f"\nScenario Distribution:\n")
    for scenario, count in scenario_counts.items():
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
print("  1. 1_model_comparison.png - Overall performance metrics comparison")
print("  2. 2_confusion_matrices.png - Confusion matrices for all models")
print("  3. 3_dataset_composition.png - Dataset breakdown visualizations")
print("  4. 4_performance_table.png - Summary table as image")
print("  5. 5_detection_rates.png - Detection rates by category")
print("  6. 6_confusion_definitions.png - Confusion matrix explanations")
print("  7. detailed_results.csv - Raw data export")
print("  8. dataset_summary.txt - Dataset statistics")
