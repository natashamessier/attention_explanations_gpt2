"""
Streamlit app for interactive interpretability visualization.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.json_utils import load_example_json, list_example_files
from pipeline import run_pipeline

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def clean_token_for_display(token: str) -> str:
    """
    Clean token string for display by removing GPT-2 special characters.
    
    Args:
        token: Raw token string from tokenizer
        
    Returns:
        Cleaned token string suitable for display
    """
    # Replace Ġ (U+0120, space marker) with a regular space
    token = token.replace('Ġ', ' ')
    # Remove leading/trailing spaces that might result
    token = token.strip()
    return token

# Page config
st.set_page_config(
    page_title="Attention vs. Explanation",
    page_icon="🔍",
    layout="wide"
)

st.title("Attention vs. Explanation: Interpretability Analysis")
st.markdown("""
**Research Question:** Can attention weights be trusted as explanations for transformer predictions, or are they simply visualizations of information flow?

This project evaluates whether attention reflects **model reasoning** or merely **information flow** by comparing attention-based interpretability with gradient-based attribution methods (Integrated Gradients and SHAP).
""")

# Sidebar for example selection
st.sidebar.header("Example Selection")

# Load available examples
results_dir = "results"
example_files = list_example_files(results_dir)

if not example_files:
    st.sidebar.warning("No examples found. Run pipeline first!")
    st.info("💡 **Tip**: Run `python pipeline.py` or `python quick_viz.py --text 'Your text'` to generate examples.")
    st.stop()

# Example selection
example_names = [Path(f).stem for f in example_files]
selected_example = st.sidebar.selectbox("Select Example", example_names)

# Load selected example
@st.cache_data
def load_example_data(filepath: str):
    """Load and cache example data."""
    return load_example_json(filepath)

example_path = example_files[example_names.index(selected_example)]
data = load_example_data(example_path)

# Display text
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Text:** {data['text'][:100]}...")

# Display actual label if available (from IMDb dataset)
if 'actual_label' in data:
    actual_label = data['actual_label']
    prediction = data['prediction']
    
    # Color code based on match
    if actual_label == prediction:
        match_status = "✓ Match"
        match_color = "green"
    else:
        match_status = "✗ Mismatch"
        match_color = "red"
    
    st.sidebar.markdown(f"**Actual Label:** {actual_label}")
    st.sidebar.markdown(f"**Prediction:** {prediction}")
    st.sidebar.markdown(f"**Status:** :{match_color}[{match_status}]")
else:
    st.sidebar.markdown(f"**Prediction:** {data['prediction']}")

st.sidebar.markdown(f"**Tokens:** {len(data['tokens'])}")

# Cross-example statistics
st.sidebar.markdown("---")
st.sidebar.subheader("Cross-Example Statistics")
st.sidebar.markdown("Aggregate metrics across all loaded examples:")

# Compute metrics for all examples
@st.cache_data
def compute_cross_example_stats(example_files_list):
    """Compute aggregate statistics across all examples."""
    all_correlations_ig = []
    all_correlations_shap = []
    all_jaccard_scores_ig = []
    all_jaccard_scores_shap = []

    for filepath in example_files_list:
        try:
            example_data = load_example_json(filepath)

            # Compute last-layer attention importance
            n_layers = len([k for k in example_data['attentions'].keys() if k.startswith('layer_')])
            last_layer_key = f"layer_{n_layers - 1}"
            head_matrices = []
            for head_key in sorted(example_data['attentions'][last_layer_key].keys()):
                head_matrices.append(np.array(example_data['attentions'][last_layer_key][head_key]))
            attention_matrix = np.mean(head_matrices, axis=0)
            attention_scores = attention_matrix.sum(axis=0)

            # Normalize
            if attention_scores.max() > 0:
                attention_scores = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min())

            ig_scores = np.array(example_data['ig_scores'])
            shap_scores = np.array(example_data['shap_values'])

            # Pearson correlation
            corr_ig = np.corrcoef(attention_scores, ig_scores)[0, 1]
            corr_shap = np.corrcoef(attention_scores, shap_scores)[0, 1]

            all_correlations_ig.append(corr_ig)
            all_correlations_shap.append(corr_shap)

            # Jaccard similarity (top-5 tokens)
            top_k = 5
            top_attn = set(np.argsort(attention_scores)[-top_k:])
            top_ig = set(np.argsort(ig_scores)[-top_k:])
            top_shap = set(np.argsort(shap_scores)[-top_k:])

            jaccard_ig = len(top_attn & top_ig) / len(top_attn | top_ig)
            jaccard_shap = len(top_attn & top_shap) / len(top_attn | top_shap)

            all_jaccard_scores_ig.append(jaccard_ig)
            all_jaccard_scores_shap.append(jaccard_shap)

        except Exception as e:
            # Skip files that can't be loaded
            continue

    return {
        'avg_corr_ig': np.mean(all_correlations_ig) if all_correlations_ig else 0,
        'avg_corr_shap': np.mean(all_correlations_shap) if all_correlations_shap else 0,
        'avg_jaccard_ig': np.mean(all_jaccard_scores_ig) if all_jaccard_scores_ig else 0,
        'avg_jaccard_shap': np.mean(all_jaccard_scores_shap) if all_jaccard_scores_shap else 0,
        'n_examples': len(all_correlations_ig)
    }

# Only compute if multiple examples exist
if len(example_files) > 1:
    stats = compute_cross_example_stats(example_files)

    st.sidebar.metric("Avg Attention-IG Correlation", f"{stats['avg_corr_ig']:.3f}")
    st.sidebar.metric("Avg Attention-SHAP Correlation", f"{stats['avg_corr_shap']:.3f}")
    st.sidebar.metric("Avg Jaccard Similarity (IG)", f"{stats['avg_jaccard_ig']:.3f}")
    st.sidebar.caption(f"Computed across {stats['n_examples']} examples")
else:
    st.sidebar.info("Add more examples to see cross-example statistics")

# Main content area
st.markdown("---")
st.markdown("""
### Sub-Questions Addressed:
1. **Single example:** Which words does the model *look at*? → Explore in Qualitative View
2. **Across layers/heads:** Do deeper layers focus on sentiment vs structure? → Analyze in Core Analysis
3. **Comparison:** When do attention and IG/SHAP agree or disagree? → Compare in Core Analysis
""")

# Main content tabs
tab1, tab2, tab3 = st.tabs([
    "Qualitative / Exploratory: Attention as Information Flow",
    "Core Analysis: Token-Level Attention vs IG/SHAP",
    "Core Analysis: Agreement, Correlation, and Layer-wise Trends"
])

# ============================================================================
# VIEW A (MERGED WITH B): Qualitative / Exploratory View
# Purpose: Visualize the raw attention matrix from the transformer architecture.
# This shows which tokens the model "looks at" when processing each position.
# Addresses Sub-question 1: For a given review, which words does the model look at?
# ============================================================================
with tab1:
    st.header("Qualitative / Exploratory: Attention as Information Flow")
    st.markdown("""
    **Purpose:** This view addresses **Sub-question 1**: *For a given review, which words does the model look at?*

    This is an **exploratory** view showing attention patterns and is not used for quantitative faithfulness claims.
    Attention shows *where information flows* during processing, not necessarily *what drives the prediction*.
    """)

    st.markdown("---")
    st.subheader("Token-Token Attention Heatmap")
    st.markdown("**How to use:** Select different layers and heads to explore attention patterns. Earlier layers often capture syntax, while later layers may focus on semantics.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_layers = len([k for k in data['attentions'].keys() if k.startswith('layer_')])
        selected_layer = st.selectbox(
            "Layer",
            options=list(range(n_layers)),
            index=n_layers - 1,  # Default to last layer
            format_func=lambda x: f"Layer {x}"
        )
    
    with col2:
        n_heads = len([k for k in data['attentions'][f'layer_{selected_layer}'].keys() if k.startswith('head_')])
        head_options = ["average"] + list(range(n_heads))
        selected_head = st.selectbox(
            "Head",
            options=head_options,
            index=0,  # Default to average
            format_func=lambda x: "Average" if x == "average" else f"Head {x}"
        )
    
    # Get attention matrix
    layer_key = f"layer_{selected_layer}"
    if selected_head == "average":
        head_matrices = []
        for head_key in sorted(data['attentions'][layer_key].keys()):
            head_matrices.append(np.array(data['attentions'][layer_key][head_key]))
        attention_matrix = np.mean(head_matrices, axis=0)
    else:
        head_key = f"head_{selected_head}"
        attention_matrix = np.array(data['attentions'][layer_key][head_key])
    
    # Create unique labels with position indices (cleaned for display)
    x_labels = [f"{i}: {clean_token_for_display(token)}" for i, token in enumerate(data['tokens'])]
    y_labels = [f"{i}: {clean_token_for_display(token)}" for i, token in enumerate(data['tokens'])]

    # Create Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=attention_matrix,
        x=x_labels,
        y=y_labels,
        colorscale='Blues',
        colorbar=dict(title="Attention Weight"),
        hovertemplate='From: %{y}<br>To: %{x}<br>Attention: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Attention Heatmap - Layer {selected_layer}, {selected_head if selected_head != 'average' else 'Averaged Heads'}",
        xaxis_title="Attended To",
        yaxis_title="Attending From",
        width=800,
        height=800,
        yaxis=dict(autorange='reversed')
    )
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Interpretation:** Rows show which tokens each word attends to. Darker colors indicate higher attention weights.")

    # ========================================================================
    # Focus on One Word (formerly View B, now merged into View A)
    # ========================================================================
    st.markdown("---")
    st.subheader("Focus on One Word")
    st.markdown("**Purpose:** Select a specific word to see which tokens it attends to most. This shows what contextual information the model gathers when processing that word.")
    st.markdown("**Example insight:** Sentiment words (like 'fantastic') might attend strongly to other sentiment words or to the subject being described.")
    
    # Word selection
    selected_word_idx = st.selectbox(
        "Select Word",
        options=list(range(len(data['tokens']))),
        format_func=lambda x: f"{x}: {clean_token_for_display(data['tokens'][x])}"
    )
    
    selected_word = clean_token_for_display(data['tokens'][selected_word_idx])
    st.markdown(f"**Selected word:** `{selected_word}` (position {selected_word_idx})")
    
    # Get attention from selected word (using last layer, averaged heads)
    n_layers = len([k for k in data['attentions'].keys() if k.startswith('layer_')])
    last_layer_key = f"layer_{n_layers - 1}"
    
    head_matrices = []
    for head_key in sorted(data['attentions'][last_layer_key].keys()):
        head_matrices.append(np.array(data['attentions'][last_layer_key][head_key]))
    attention_matrix = np.mean(head_matrices, axis=0)
    
    # Get attention from selected word to all other tokens
    attention_from_word = attention_matrix[selected_word_idx, :]
    
    # Create unique labels with position indices (cleaned for display)
    token_labels = [f"{i}: {clean_token_for_display(token)}" for i, token in enumerate(data['tokens'])]
    
    # Create bar chart
    fig = go.Figure(data=go.Bar(
        x=token_labels,
        y=attention_from_word,
        marker_color='steelblue',
        hovertemplate='Token: %{x}<br>Attention: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Attention from '{selected_word}' to Other Tokens",
        xaxis_title="Token",
        yaxis_title="Attention Weight",
        height=500
    )
    
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show top tokens
    top_indices = np.argsort(attention_from_word)[-5:][::-1]
    st.markdown("**Top 5 tokens this word attends to:**")
    for i, idx in enumerate(top_indices, 1):
        st.markdown(f"{i}. `{idx}: {clean_token_for_display(data['tokens'][idx])}` (attention: {attention_from_word[idx]:.3f})")

# ============================================================================
# VIEW C (NOW TAB 2): Core Analysis - Token-Level Attention vs IG/SHAP
# Purpose: Compare gradient-based explanations (IG/SHAP) with attention.
# IG and SHAP measure which tokens actually influence the prediction.
# Addresses Sub-question 3: When do attention and IG/SHAP agree or disagree?
# ============================================================================
with tab2:
    st.header("Core Analysis: Token-Level Attention vs IG/SHAP")
    st.markdown("""
    **Purpose:** This view addresses **Sub-question 3**: *When do attention and IG/SHAP agree or disagree?*

    This view provides **primary evidence** for whether attention can be trusted as an explanation.

    **Methods:**
    - **Attention (treated as hypothesized explanation):** Computed using GPT-2 final layer, averaged across heads, measuring total attention *received* by each token, then normalized. This measures *where information flows*, not necessarily *what drives the prediction*.
    - **Integrated Gradients (IG):** Reference gradient-based attribution method showing which tokens influence the prediction.
    - **SHAP:** Reference gradient-based attribution method based on Shapley values.
    """)
    
    # Get attention matrix for hover (last layer, averaged)
    n_layers = len([k for k in data['attentions'].keys() if k.startswith('layer_')])
    last_layer_key = f"layer_{n_layers - 1}"
    
    head_matrices = []
    for head_key in sorted(data['attentions'][last_layer_key].keys()):
        head_matrices.append(np.array(data['attentions'][last_layer_key][head_key]))
    attention_matrix = np.mean(head_matrices, axis=0)

    # Add importance bars comparison (grouped bar chart)
    st.markdown("---")
    st.subheader("Token Importance: Side-by-Side Comparison")
    st.markdown("**Visual comparison** of importance scores across all three methods for each token.")

    # Add interpretation guidance
    with st.expander("💡 What patterns should I look for?"):
        st.markdown("""
        - **Similar bar heights** → Methods agree; attention is informative
        - **Attention high, IG/SHAP low** → Token receives attention but doesn't influence prediction (context gathering)
        - **Attention low, IG/SHAP high** → Important token overlooked by attention (use gradient methods)
        - **All different** → Controversial token; no clear consensus
        """)

    # Compute attention importance scores by summing attention RECEIVED by each token
    # This aggregation sums across all attending positions (axis=0) to measure
    # how much the model collectively "focuses on" each token overall
    attention_importance = attention_matrix.sum(axis=0)
    if attention_importance.max() > 0:
        attention_importance = (attention_importance - attention_importance.min()) / (attention_importance.max() - attention_importance.min())

    # Truncate if too many tokens for readability
    max_tokens_display = 30
    if len(data['tokens']) > max_tokens_display:
        tokens_display = data['tokens'][:max_tokens_display]
        attention_display = attention_importance[:max_tokens_display]
        ig_display = data['ig_scores'][:max_tokens_display]
        shap_display = data['shap_values'][:max_tokens_display]
        st.info(f"📊 Showing first {max_tokens_display} of {len(data['tokens'])} tokens for readability.")
    else:
        tokens_display = data['tokens']
        attention_display = attention_importance
        ig_display = data['ig_scores']
        shap_display = data['shap_values']

    # Create unique labels with position indices (cleaned for display)
    token_labels = [f"{i}: {clean_token_for_display(token)}" for i, token in enumerate(tokens_display)]

    fig_bars = go.Figure()

    # Add bars for each method
    fig_bars.add_trace(go.Bar(
        x=token_labels,
        y=attention_display,
        name='Attention',
        marker_color='#3498db',
        opacity=0.8,
        hovertemplate='<b>%{x}</b><br>Attention: %{y:.3f}<extra></extra>'
    ))

    fig_bars.add_trace(go.Bar(
        x=token_labels,
        y=ig_display,
        name='Integrated Gradients',
        marker_color='#e74c3c',
        opacity=0.8,
        hovertemplate='<b>%{x}</b><br>IG: %{y:.3f}<extra></extra>'
    ))

    fig_bars.add_trace(go.Bar(
        x=token_labels,
        y=shap_display,
        name='SHAP',
        marker_color='#2ecc71',
        opacity=0.8,
        hovertemplate='<b>%{x}</b><br>SHAP: %{y:.3f}<extra></extra>'
    ))

    # Update layout
    fig_bars.update_layout(
        barmode='group',
        xaxis_title='Tokens',
        yaxis_title='Importance Score (Normalized)',
        height=500,
        xaxis=dict(tickangle=45),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )

    st.plotly_chart(fig_bars, use_container_width=True)

    st.markdown("""
    **How to read this chart:**
    - Each token has 3 bars showing its importance according to each method
    - If bars are similar heights → methods agree on this token's importance
    - If bars differ significantly → methods disagree (interesting case!)
    - Look for tokens where attention is high but IG/SHAP are low (or vice versa)
    """)

    # Helps vs Misleads Analysis
    st.markdown("---")
    st.subheader("🔍 When Does Attention Help vs Mislead?")
    st.markdown("**Automated identification** of tokens where methods agree vs. disagree, helping assess attention reliability.")

    # Identify high-discrepancy tokens
    disagreement = np.abs(attention_display - ig_display) + np.abs(attention_display - shap_display)
    high_discrep_threshold = np.percentile(disagreement, 75)

    high_discrep_indices = np.where(disagreement >= high_discrep_threshold)[0]
    high_agreement_indices = np.where(disagreement < np.percentile(disagreement, 25))[0]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**✅ Attention HELPS (Methods Agree)**")
        st.markdown("These tokens show consistent importance across all methods:")
        if len(high_agreement_indices) > 0:
            for idx in high_agreement_indices[:5]:
                st.markdown(f"- **{token_labels[idx]}**: Attn={attention_display[idx]:.2f}, "
                           f"IG={ig_display[idx]:.2f}, SHAP={shap_display[idx]:.2f}")
        else:
            st.markdown("_No tokens with strong agreement found._")
        st.caption("When bars have similar heights, attention is reliable.")

    with col2:
        st.markdown("**⚠️ Attention May MISLEAD (Methods Disagree)**")
        st.markdown("These tokens have large disagreement between methods:")
        if len(high_discrep_indices) > 0:
            for idx in high_discrep_indices[:5]:
                st.markdown(f"- **{token_labels[idx]}**: Attn={attention_display[idx]:.2f}, "
                           f"IG={ig_display[idx]:.2f}, SHAP={shap_display[idx]:.2f}")
        else:
            st.markdown("_No tokens with strong disagreement found._")
        st.caption("When bars differ greatly, be cautious using attention alone.")

    # Comparison table
    st.markdown("---")
    st.subheader("Token Importance: Numerical Comparison")
    
    # Create comparison DataFrame
    import pandas as pd
    # Create unique token labels with position indices (cleaned for display)
    token_labels_df = [f"{i}: {clean_token_for_display(token)}" for i, token in enumerate(data['tokens'])]
    comparison_data = {
        'Token': token_labels_df,
        'IG Score': data['ig_scores'],
        'SHAP Score': data['shap_values'],
        'Attention (avg received)': [attention_matrix[:, i].mean() for i in range(len(data['tokens']))]
    }
    df = pd.DataFrame(comparison_data)
    
    # Normalize attention for comparison
    attn_scores = df['Attention (avg received)'].values
    if attn_scores.max() > 0:
        df['Attention (avg received)'] = (attn_scores - attn_scores.min()) / (attn_scores.max() - attn_scores.min())
    
    st.dataframe(df.style.background_gradient(subset=['IG Score', 'SHAP Score', 'Attention (avg received)'], 
                                               cmap='Blues'), 
                 use_container_width=True)

# ============================================================================
# VIEW D (NOW TAB 3): Core Analysis - Agreement, Correlation, and Layer-wise Trends
# Purpose: Quantify the relationship between attention and gradient-based methods.
# Analyzes: (1) Overall correlation, (2) Layer-wise patterns, (3) Word-type focus.
# Addresses Sub-questions 2 and 3
# ============================================================================
with tab3:
    st.header("Core Analysis: Agreement, Correlation, and Layer-wise Trends")

    # Research framing block
    with st.expander("📋 Research Framing", expanded=False):
        st.markdown("""
        **Research Question:** Can attention weights be trusted as explanations for transformer predictions?

        **Approach:**
        - **Attention** is treated as a *hypothesized explanation* of model behavior
        - **Integrated Gradients (IG)** and **SHAP** are used as *reference gradient-based attribution methods*
        - **Goal:** Identify **when attention helps** (aligns with gradient methods) **vs when it misleads** (diverges from gradient methods)

        **Key Point:** This view provides **primary evidence** for evaluating attention's reliability as an explanatory tool.
        """)

    st.markdown("""
    **Purpose:** This view addresses **Sub-question 2** (*Do deeper layers focus on sentiment vs structure?*) and **Sub-question 3** (*When do attention and IG/SHAP agree or disagree?*).

    Quantitatively measure how well attention weights align with gradient-based importance scores across examples and layers.
    """)
    
    # Compute attention importance for each layer
    n_layers = len([k for k in data['attentions'].keys() if k.startswith('layer_')])
    
    def compute_layer_attention_importance(layer_idx: int) -> np.ndarray:
        """Compute attention importance for a specific layer."""
        layer_key = f"layer_{layer_idx}"
        head_matrices = []
        for head_key in sorted(data['attentions'][layer_key].keys()):
            head_matrices.append(np.array(data['attentions'][layer_key][head_key]))
        attention_matrix = np.mean(head_matrices, axis=0)
        importance = attention_matrix.sum(axis=0)
        if importance.max() > 0:
            importance = (importance - importance.min()) / (importance.max() - importance.min())
        return importance
    
    # Scatter plot: Attention vs IG/SHAP
    st.subheader("Scatter Plot: Attention vs. IG/SHAP")

    st.info("""
    **Attention importance computation:**
    - GPT-2 final layer
    - Averaged across all attention heads
    - Total attention **received** by each token (summed across attending positions)
    - Normalized across tokens
    - **Interpretation:** This measures *where information flows*, not necessarily *what drives the prediction*.
    """)

    method_scatter = st.radio("Compare with:", ["Integrated Gradients", "SHAP"], horizontal=True, key="scatter")
    
    # Use last layer attention
    attention_scores = compute_layer_attention_importance(n_layers - 1)
    comparison_scores = data['ig_scores'] if method_scatter == "Integrated Gradients" else data['shap_values']
    
    # Create unique token labels with position indices for hover (cleaned for display)
    token_labels_hover = [f"{i}: {clean_token_for_display(token)}" for i, token in enumerate(data['tokens'])]
    
    # Create scatter plot
    fig = px.scatter(
        x=attention_scores,
        y=comparison_scores,
        labels={'x': 'Attention Score', 'y': f'{method_scatter} Score'},
        title=f'Attention vs. {method_scatter}',
        hover_data=[token_labels_hover],
        hover_name=token_labels_hover
    )
    
    # Add diagonal line
    max_val = max(attention_scores.max(), comparison_scores.max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(dash='dash', color='red'),
        name='Perfect Correlation'
    ))
    
    st.plotly_chart(fig, use_container_width=True)

    # Compute correlation
    correlation = np.corrcoef(attention_scores, comparison_scores)[0, 1]

    # Check if correlation is undefined (NaN due to constant values)
    if np.isnan(correlation):
        st.warning("⚠️ **Agreement undefined for this example**: Scores do not vary enough to compute correlation.")
    else:
        col_metric1, col_metric2 = st.columns(2)

        with col_metric1:
            st.metric("Per-Example Pearson Correlation ⭐", f"{correlation:.3f}")
            st.caption("PRIMARY METRIC: This specific example")

        with col_metric2:
            # Display dataset-level average if available
            if len(example_files) > 1:
                dataset_avg = stats['avg_corr_ig'] if method_scatter == "Integrated Gradients" else stats['avg_corr_shap']
                st.metric("Dataset-Level Average", f"{dataset_avg:.3f}")
                st.caption(f"Global trend across {stats['n_examples']} examples")
            else:
                st.info("Add more examples to see dataset-level trends")

        st.caption("**Note:** Per-example values show *local behavior* for this review. Dataset averages show *global tendencies* across all examples.")

        # Interpretation guide
        if correlation > 0.7:
            st.success("🟢 **Strong correlation**: Attention patterns largely align with gradient-based methods. "
                       "Attention is relatively reliable for this example.")
        elif correlation > 0.4:
            st.warning("🟡 **Moderate correlation**: Methods partially agree. "
                       "Use attention with caution; validate important tokens with IG/SHAP.")
        else:
            st.error("🔴 **Weak correlation**: Attention and gradient methods substantially disagree. "
                     "Attention may not reflect model reasoning for this example.")

    # Additional similarity metrics
    st.markdown("---")
    st.subheader("Agreement Metrics")
    st.markdown("""
    **Primary Metrics** (used for main claims):
    - **Pearson Correlation** (shown above): Linear relationship between attention and gradient-based scores
    - **Jaccard Similarity (top-5)**: Overlap of top 5 most important tokens

    **Supporting Metrics** (robustness checks):
    - **Spearman Correlation**: Rank-order agreement
    - **Kendall's Tau**: Pairwise rank concordance
    """)

    from scipy.stats import spearmanr, kendalltau

    col1, col2, col3 = st.columns(3)

    # Compute Jaccard similarity for top-5 tokens
    top_k = 5
    top_attn = set(np.argsort(attention_scores)[-top_k:])
    top_comp = set(np.argsort(comparison_scores)[-top_k:])
    jaccard = len(top_attn & top_comp) / len(top_attn | top_comp)

    # Spearman rank correlation
    spearman_corr, _ = spearmanr(attention_scores, comparison_scores)

    # Kendall's tau
    kendall_corr, _ = kendalltau(attention_scores, comparison_scores)

    # Check if metrics are undefined
    jaccard_display = f"{jaccard:.3f}" if not np.isnan(jaccard) else "Undefined"
    spearman_display = f"{spearman_corr:.3f}" if not np.isnan(spearman_corr) else "Undefined"
    kendall_display = f"{kendall_corr:.3f}" if not np.isnan(kendall_corr) else "Undefined"

    with col1:
        st.metric("Jaccard Similarity (top-5) ⭐", jaccard_display)
        st.caption("PRIMARY: Overlap of top 5 tokens (per-example)")
        if len(example_files) > 1:
            dataset_jaccard = stats['avg_jaccard_ig'] if method_scatter == "Integrated Gradients" else stats['avg_jaccard_shap']
            st.caption(f"📊 Dataset avg: {dataset_jaccard:.3f}")

    with col2:
        st.metric("Spearman Correlation", spearman_display)
        st.caption("Supporting: Rank-order agreement")

    with col3:
        st.metric("Kendall's Tau", kendall_display)
        st.caption("Supporting: Pairwise concordance")

    # Layer-wise correlation
    st.subheader("Layer-wise Correlation Analysis")
    st.markdown("**What this shows:** How the correlation between attention and gradient-based methods changes across transformer layers.")
    st.markdown("**Why it matters:** Different layers encode different information. Early layers often process syntax/structure, while later layers typically handle semantics/meaning. If attention in later layers correlates more with IG/SHAP, it might suggest attention becomes more aligned with task-relevant features deeper in the network, though this varies by example.")

    layer_correlations_ig = []
    layer_correlations_shap = []
    
    for layer_idx in range(n_layers):
        layer_attn = compute_layer_attention_importance(layer_idx)
        corr_ig = np.corrcoef(layer_attn, data['ig_scores'])[0, 1]
        corr_shap = np.corrcoef(layer_attn, data['shap_values'])[0, 1]
        layer_correlations_ig.append(corr_ig)
        layer_correlations_shap.append(corr_shap)
    
    # Plot layer-wise correlations
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(n_layers)),
        y=layer_correlations_ig,
        mode='lines+markers',
        name='vs. IG',
        line=dict(color='red')
    ))
    fig.add_trace(go.Scatter(
        x=list(range(n_layers)),
        y=layer_correlations_shap,
        mode='lines+markers',
        name='vs. SHAP',
        line=dict(color='green')
    ))
    
    fig.update_layout(
        title='Attention-IG/SHAP Correlation by Layer',
        xaxis_title='Layer',
        yaxis_title='Correlation Coefficient',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Add interpretation example
    max_corr_layer = np.argmax(layer_correlations_ig)
    min_corr_layer = np.argmin(layer_correlations_ig)

    st.markdown(f"""
**Pattern Analysis:**
- **Highest correlation**: Layer {max_corr_layer} (r={layer_correlations_ig[max_corr_layer]:.3f})
- **Lowest correlation**: Layer {min_corr_layer} (r={layer_correlations_ig[min_corr_layer]:.3f})

{'If attention were task-aligned, earlier layers showing stronger alignment might suggest attention captures syntactic patterns in this example.' if max_corr_layer < 6 else 'If attention were task-aligned, later layers showing stronger alignment might suggest attention aligns with semantic/task-relevant features in this example. However, this pattern is example-dependent.'}
""")

    # Histogram: Attention to sentiment words vs stopwords
    st.subheader("Attention to Sentiment Words vs. Stopwords")

    st.warning("""
    ⚠️ **Methodological Disclaimer:**
    - This analysis uses **lexicon-based sentiment detection** (simple word lists), which is coarse and approximate
    - **GPT-2 uses subword tokenization** that can fragment sentiment words (e.g., "fantastic" → "fant", "astic")
    - Results should be interpreted as **exploratory evidence** about layer-wise attention patterns, not definitive proof of task reasoning
    """)

    st.markdown("**What this shows:** How much attention the model allocates to sentiment-bearing words (like 'fantastic', 'terrible') versus structural stopwords (like 'the', 'and') across different layers.")
    st.markdown("**Why it matters:** If attention is task-aligned for sentiment classification, we might expect deeper layers to focus more on sentiment words. If attention focuses mainly on stopwords, it may suggest attention doesn't reflect the model's reasoning for this task.")
    st.markdown("**Possible pattern (if attention is task-aligned):** Sentiment words might receive more attention in later layers, but this depends on the specific example and model internals.")

    # Simple sentiment/stopword detection
    sentiment_words = ['love', 'loved', 'great', 'good', 'excellent', 'fantastic', 'amazing',
                       'terrible', 'awful', 'bad', 'horrible', 'poor', 'worst']
    stopwords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']

    # Convert to sets for O(1) lookup and use exact matching
    sentiment_words_set = {word.lower() for word in sentiment_words}
    stopwords_set = {word.lower() for word in stopwords}

    # Identify sentiment and stopword tokens using exact matching (after cleaning and stripping punctuation)
    sentiment_indices = [i for i, token in enumerate(data['tokens'])
                         if clean_token_for_display(token).lower().strip('.,!?;:\'"') in sentiment_words_set]
    stopword_indices = [i for i, token in enumerate(data['tokens'])
                       if clean_token_for_display(token).lower().strip('.,!?;:\'"') in stopwords_set]
    
    if sentiment_indices and stopword_indices:
        # Get average attention to sentiment words and stopwords by layer
        sentiment_attention_by_layer = []
        stopword_attention_by_layer = []
        
        for layer_idx in range(n_layers):
            layer_attn = compute_layer_attention_importance(layer_idx)
            sentiment_attn = layer_attn[sentiment_indices].mean() if sentiment_indices else 0
            stopword_attn = layer_attn[stopword_indices].mean() if stopword_indices else 0
            sentiment_attention_by_layer.append(sentiment_attn)
            stopword_attention_by_layer.append(stopword_attn)
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(n_layers)),
            y=sentiment_attention_by_layer,
            name='Sentiment Words',
            marker_color='green'
        ))
        fig.add_trace(go.Bar(
            x=list(range(n_layers)),
            y=stopword_attention_by_layer,
            name='Stopwords',
            marker_color='gray'
        ))
        
        fig.update_layout(
            title='Average Attention to Sentiment Words vs. Stopwords by Layer',
            xaxis_title='Layer',
            yaxis_title='Average Attention',
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough sentiment words or stopwords detected in this example.")

# ============================================================================
# Footer
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### Add New Example")
new_text = st.sidebar.text_area("Enter text to analyze:", height=100)

if st.sidebar.button("Run Pipeline"):
    if not new_text.strip():
        st.sidebar.error("Please enter some text before running the pipeline.")
    else:
        with st.spinner("Processing..."):
            try:
                new_filepath = run_pipeline(new_text, output_dir=results_dir)
                st.sidebar.success(f"Saved to {Path(new_filepath).name}")
                st.sidebar.info("Refresh the page to see the new example.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")

