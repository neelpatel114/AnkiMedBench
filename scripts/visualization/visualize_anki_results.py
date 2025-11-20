#!/usr/bin/env python3
"""
Create visualizations for Anki analysis results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def create_dual_metric_comparison(df_firstaid, df_embeddings, output_dir):
    """Create dual-metric (F1 vs hF1) comparison plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Firstaid models
    level_stats_fa = df_firstaid.groupby('level').agg({
        'f1_score': ['mean', 'std'],
        'hierarchical_f1': ['mean', 'std'],
        'categories': 'first'
    })

    levels = level_stats_fa.index
    f1_mean = level_stats_fa['f1_score']['mean']
    f1_std = level_stats_fa['f1_score']['std']
    hf1_mean = level_stats_fa['hierarchical_f1']['mean']
    hf1_std = level_stats_fa['hierarchical_f1']['std']

    # Plot 1: FirstAid models
    ax1.plot(levels, f1_mean, 'o-', label='Standard F1', linewidth=2, markersize=8, color='#d62728')
    ax1.fill_between(levels, f1_mean - f1_std, f1_mean + f1_std, alpha=0.2, color='#d62728')

    ax1.plot(levels, hf1_mean, 's-', label='Hierarchical F1', linewidth=2, markersize=8, color='#2ca02c')
    ax1.fill_between(levels, hf1_mean - hf1_std, hf1_mean + hf1_std, alpha=0.2, color='#2ca02c')

    ax1.set_xlabel('Hierarchy Level', fontsize=13)
    ax1.set_ylabel('F1 Score', fontsize=13)
    ax1.set_title('Fine-Tuned Models (FirstAid)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # Add category counts
    ax1_twin = ax1.twiny()
    ax1_twin.set_xlim(ax1.get_xlim())
    ax1_twin.set_xticks(levels)
    ax1_twin.set_xticklabels([f"{int(level_stats_fa['categories']['first'][l])}" for l in levels], fontsize=10)
    ax1_twin.set_xlabel('Number of Categories', fontsize=11, color='gray')

    # Plot 2: Embedding models
    level_stats_emb = df_embeddings.groupby('level').agg({
        'f1_score': ['mean', 'std'],
        'hierarchical_f1': ['mean', 'std'],
        'categories': 'first'
    })

    levels_emb = level_stats_emb.index
    f1_mean_emb = level_stats_emb['f1_score']['mean']
    f1_std_emb = level_stats_emb['f1_score']['std']
    hf1_mean_emb = level_stats_emb['hierarchical_f1']['mean']
    hf1_std_emb = level_stats_emb['hierarchical_f1']['std']

    ax2.plot(levels_emb, f1_mean_emb, 'o-', label='Standard F1', linewidth=2, markersize=8, color='#d62728')
    ax2.fill_between(levels_emb, f1_mean_emb - f1_std_emb, f1_mean_emb + f1_std_emb, alpha=0.2, color='#d62728')

    ax2.plot(levels_emb, hf1_mean_emb, 's-', label='Hierarchical F1', linewidth=2, markersize=8, color='#2ca02c')
    ax2.fill_between(levels_emb, hf1_mean_emb - hf1_std_emb, hf1_mean_emb + hf1_std_emb, alpha=0.2, color='#2ca02c')

    ax2.set_xlabel('Hierarchy Level', fontsize=13)
    ax2.set_ylabel('F1 Score', fontsize=13)
    ax2.set_title('Commercial Embeddings', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    # Add category counts
    ax2_twin = ax2.twiny()
    ax2_twin.set_xlim(ax2.get_xlim())
    ax2_twin.set_xticks(levels_emb)
    ax2_twin.set_xticklabels([f"{int(level_stats_emb['categories']['first'][l])}" for l in levels_emb], fontsize=10)
    ax2_twin.set_xlabel('Number of Categories', fontsize=11, color='gray')

    plt.tight_layout()
    plt.savefig(output_dir / 'dual_metric_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'dual_metric_comparison.pdf', bbox_inches='tight')
    print(f"Saved: dual_metric_comparison.png/pdf")
    plt.close()

def create_model_comparison_heatmap(df, output_dir, title, filename):
    """Create heatmap of hierarchical F1 by model and level"""

    # Pivot data
    pivot = df.pivot_table(values='hierarchical_f1', index='model', columns='level', aggfunc='max')

    # Sort by average performance
    pivot['avg'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('avg', ascending=False).drop('avg', axis=1)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                cbar_kws={'label': 'Hierarchical F1'}, ax=ax, linewidths=0.5)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Hierarchy Level', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{filename}.pdf', bbox_inches='tight')
    print(f"Saved: {filename}.png/pdf")
    plt.close()

def create_category_scaling_plot(df_firstaid, df_embeddings, output_dir):
    """Show how performance scales with category count"""

    fig, ax = plt.subplots(figsize=(12, 7))

    # FirstAid models
    level_stats_fa = df_firstaid.groupby('level').agg({
        'categories': 'first',
        'hierarchical_f1': 'mean'
    }).reset_index()

    # Embeddings
    level_stats_emb = df_embeddings.groupby('level').agg({
        'categories': 'first',
        'hierarchical_f1': 'mean'
    }).reset_index()

    ax.plot(level_stats_fa['categories'], level_stats_fa['hierarchical_f1'],
            'o-', label='Fine-Tuned Models', linewidth=2.5, markersize=10, color='#1f77b4')

    ax.plot(level_stats_emb['categories'], level_stats_emb['hierarchical_f1'],
            's-', label='Commercial Embeddings', linewidth=2.5, markersize=10, color='#ff7f0e')

    # Add level labels
    for _, row in level_stats_fa.iterrows():
        ax.annotate(f"L{int(row['level'])}",
                   xy=(row['categories'], row['hierarchical_f1']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

    # Highlight L5 threshold
    ax.axvline(x=608, color='red', linestyle='--', alpha=0.5, linewidth=2, label='L5 Saturation (608 cats)')
    ax.axhline(y=0.417, color='red', linestyle='--', alpha=0.3, linewidth=1.5)

    ax.set_xlabel('Number of Categories', fontsize=13)
    ax.set_ylabel('Hierarchical F1', fontsize=13)
    ax.set_title('Performance vs Category Count (Scaling Analysis)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'category_scaling.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'category_scaling.pdf', bbox_inches='tight')
    print(f"Saved: category_scaling.png/pdf")
    plt.close()

def create_error_distance_plot(df_firstaid, df_embeddings, output_dir):
    """Plot average error distance by level"""

    fig, ax = plt.subplots(figsize=(12, 7))

    # FirstAid
    level_stats_fa = df_firstaid.groupby('level').agg({
        'avg_error_distance': ['mean', 'std']
    }).reset_index()

    # Embeddings
    level_stats_emb = df_embeddings.groupby('level').agg({
        'avg_error_distance': ['mean', 'std']
    }).reset_index()

    levels_fa = level_stats_fa['level']
    error_fa = level_stats_fa['avg_error_distance']['mean']
    error_fa_std = level_stats_fa['avg_error_distance']['std']

    levels_emb = level_stats_emb['level']
    error_emb = level_stats_emb['avg_error_distance']['mean']
    error_emb_std = level_stats_emb['avg_error_distance']['std']

    ax.plot(levels_fa, error_fa, 'o-', label='Fine-Tuned Models',
            linewidth=2.5, markersize=10, color='#1f77b4')
    ax.fill_between(levels_fa, error_fa - error_fa_std, error_fa + error_fa_std,
                     alpha=0.2, color='#1f77b4')

    ax.plot(levels_emb, error_emb, 's-', label='Commercial Embeddings',
            linewidth=2.5, markersize=10, color='#ff7f0e')
    ax.fill_between(levels_emb, error_emb - error_emb_std, error_emb + error_emb_std,
                     alpha=0.2, color='#ff7f0e')

    ax.set_xlabel('Hierarchy Level', fontsize=13)
    ax.set_ylabel('Average Graph Distance (steps)', fontsize=13)
    ax.set_title('Prediction Error Distance in Hierarchy Graph', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'error_distance.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'error_distance.pdf', bbox_inches='tight')
    print(f"Saved: error_distance.png/pdf")
    plt.close()

def create_top_models_comparison(df_firstaid, df_embeddings, output_dir):
    """Compare top 5 models from each group at each level"""

    levels = [3, 4, 5, 6, 7]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, level in enumerate(levels):
        ax = axes[idx]

        # Get top 5 from each group
        top_fa = df_firstaid[df_firstaid['level'] == level].nlargest(5, 'hierarchical_f1')
        top_emb = df_embeddings[df_embeddings['level'] == level].nlargest(5, 'hierarchical_f1')

        # Combine
        combined = pd.concat([
            top_fa[['model', 'variant', 'hierarchical_f1']].assign(group='Fine-Tuned'),
            top_emb[['model', 'variant', 'hierarchical_f1']].assign(group='Commercial')
        ])

        combined['label'] = combined['model'].str.split('/').str[-1] + '\n' + combined['variant']
        combined = combined.sort_values('hierarchical_f1', ascending=True)

        # Plot
        colors = ['#1f77b4' if g == 'Fine-Tuned' else '#ff7f0e' for g in combined['group']]
        bars = ax.barh(range(len(combined)), combined['hierarchical_f1'], color=colors)

        ax.set_yticks(range(len(combined)))
        ax.set_yticklabels(combined['label'], fontsize=8)
        ax.set_xlabel('Hierarchical F1', fontsize=10)
        ax.set_title(f'Level {level} (Top 5 Each)', fontsize=11, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3, axis='x')

        # Add values
        for i, (bar, val) in enumerate(zip(bars, combined['hierarchical_f1'])):
            ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=8)

    # Remove extra subplot
    axes[-1].axis('off')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Fine-Tuned Models'),
        Patch(facecolor='#ff7f0e', label='Commercial Embeddings')
    ]
    axes[-1].legend(handles=legend_elements, loc='center', fontsize=12)

    plt.suptitle('Top 5 Models Comparison by Level', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'top_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'top_models_comparison.pdf', bbox_inches='tight')
    print(f"Saved: top_models_comparison.png/pdf")
    plt.close()

def main():
    # Load data
    results_dir = Path("/Users/npatel/Downloads/Thesis_Anki_Analysis/results")
    df_firstaid = pd.read_csv(results_dir / "anki_analysis_summary_firstaid.csv")
    df_embeddings = pd.read_csv(results_dir / "anki_analysis_summary_embeddings.csv")

    # Create output directory
    viz_dir = results_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    print("Creating visualizations...")
    print("-" * 60)

    # Generate plots
    create_dual_metric_comparison(df_firstaid, df_embeddings, viz_dir)
    create_model_comparison_heatmap(df_firstaid, viz_dir,
                                   'Fine-Tuned Models: Hierarchical F1 Heatmap',
                                   'heatmap_firstaid')
    create_model_comparison_heatmap(df_embeddings, viz_dir,
                                   'Commercial Embeddings: Hierarchical F1 Heatmap',
                                   'heatmap_embeddings')
    create_category_scaling_plot(df_firstaid, df_embeddings, viz_dir)
    create_error_distance_plot(df_firstaid, df_embeddings, viz_dir)
    create_top_models_comparison(df_firstaid, df_embeddings, viz_dir)

    print("-" * 60)
    print(f"All visualizations saved to: {viz_dir}")
    print("\nGenerated files:")
    for f in sorted(viz_dir.glob("*.png")):
        print(f"  - {f.name}")

if __name__ == '__main__':
    main()
