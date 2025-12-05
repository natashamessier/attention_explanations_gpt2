# Attention ≠ Explanation: Evaluating How Well Attention Reflects Model Reasoning

**Author:** Natasha Messier  
**Course:** DS-5690 Generative AI Models in Theory and Practice  
**Semester:** Fall 2025

## Overview

This project investigates whether transformer attention weights accurately reflect model reasoning by comparing attention-based interpretability methods with gradient-based approaches (Integrated Gradients and SHAP). Using GPT-2 on the IMDb sentiment analysis dataset, we evaluate when attention helps explain model behavior and when it may be misleading.

## Research Question

**Can attention weights be trusted as explanations for transformer predictions, or are they simply visualizations of information flow?**

This project also addresses three sub-questions:

1. **Single example:** Which words does the model *look at*?
2. **Across layers/heads:** Do deeper layers focus on sentiment vs structure?
3. **Comparison:** When do attention and IG/SHAP agree or disagree?

## Methodology

This project applies interpretability techniques and transformer architecture concepts to empirically evaluate the explanatory power of attention mechanisms.

### Theoretical Foundation

The methodology builds directly on course material and current research:

1. **Transformer Architecture** (*Formal Algorithms for Transformers*, Phuong & Hutter): We examine the attention mechanism as defined formally in course readings, extracting raw attention weights from each of GPT-2's 12 layers and 12 heads.

2. **Attention Mechanism Analysis**: Following the self-attention formulation Q·K^T/√d, we extract attention distributions to understand which tokens the model attends to during prediction.

3. **Attribution Methods**: We implement gradient-based interpretability approaches that provide an alternative view of token importance:
   - **Integrated Gradients** (Sundararajan et al., 2017): Satisfies implementation invariance and sensitivity axioms
   - **SHAP** (Lundberg & Lee, 2017): Provides game-theoretic feature attribution based on Shapley values
   
### Interpretability Methods

#### 1. Attention Visualization
- **Description:** Extract and aggregate attention weights from transformer layers
- **Implementation:** Raw attention weights from GPT-2
- **Output:** Token-level attention matrices (by layer and head)

#### 2. Integrated Gradients
- **Description:** Gradient-based attribution method with path integration
- **Implementation:** Captum library
- **Baseline:** Zero embedding vector
- **Steps:** 50 interpolation steps
- **Output:** Token-level importance based on gradient × input

#### 3. SHAP (SHapley Additive exPlanations)
- **Description:** Game-theoretic approach to feature attribution
- **Implementation:** Token masking approach
- **Output:** Token-level Shapley values

### Approach

Our experimental pipeline follows these steps:

1. **Model Selection**: Use pre-trained GPT-2 (124M parameters), a standard transformer architecture studied in course
2. **Dataset**: IMDb movie reviews for sentiment analysis - a well-defined binary classification task
3. **Attention Extraction**: Collect attention weights from all layers/heads during forward pass
4. **Attribution Computation**:
   - Compute Integrated Gradients with 50 interpolation steps from zero baseline
   - Calculate SHAP values using token masking approach
5. **Comparison Analysis**:
   - Quantitative: Measure correlation between attention scores and attribution scores
   - Qualitative: Visual comparison through heatmaps and token highlighting
6. **Evaluation**: Assess when attention aligns with or diverges from gradient-based explanations

## Project Structure

```
final_project_v2/
├── pipeline.py              # Main data pipeline (tokenize → GPT-2 → JSON)
├── quick_viz.py             # Terminal command for quick visualization
├── app.py                   # Streamlit interactive visualization app
├── utils/                   # Simplified utility modules
│   ├── model_utils.py       # GPT-2 model wrapper
│   ├── interpretability.py # IG and SHAP computation
│   ├── json_utils.py       # JSON save/load utilities
│   └── model_info.py       # Model information utilities
├── data/                    # Data loading and preprocessing
│   ├── load_dataset.py     # IMDb dataset loader
│   └── preprocessing.py    # Text cleaning utilities
├── results/                 # Output directory
│   ├── example_*.json      # Processed example data (JSON format)
│   └── quick_viz_*.png     # Quick visualization outputs
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster processing

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd /path/to/final_project_v2
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   # Using venv
   python -m venv venv

   # Activate on macOS/Linux
   source venv/bin/activate

   # Activate on Windows
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - PyTorch
   - HuggingFace Transformers
   - Captum (for Integrated Gradients)
   - SHAP
   - Scientific Python stack (NumPy, Pandas, SciPy)
   - Visualization libraries (Matplotlib, Seaborn, Plotly)
   - Streamlit (for interactive app)
   - Jupyter for notebooks

4. **Verify installation:**
   ```bash
   python -c "import torch; import transformers; import streamlit; print('Setup successful!')"
   ```

## Usage

### Quick Start

**1. Process IMDb examples and save to JSON:**
```bash
# Process 10 IMDb examples (default)
python pipeline.py

# Process 20 IMDb examples
python pipeline.py --imdb --num-examples 20

# Process a single custom text
python pipeline.py --text "This movie was absolutely fantastic!"
```

**2. Quick visualization from terminal:**
```bash
# Visualize an IMDb example
python quick_viz.py --imdb --example-id 1

# Visualize custom text
python quick_viz.py --text "This movie was great!"

# Use existing JSON file
python quick_viz.py --json-file results/example_1.json
```

**3. Interactive Streamlit app:**
```bash
streamlit run app.py
```

The app provides three analysis views:
- **Qualitative / Exploratory:** Attention as information flow (token-token heatmap and focus on one word)
- **Core Analysis (Token-Level):** Token attributions comparing attention vs IG/SHAP
- **Core Analysis (Agreement):** Correlation analysis, layer-wise trends, and agreement metrics

### Data Pipeline

The pipeline follows this simple flow:
1. **Tokenize** input text
2. **Run GPT-2** with `output_attentions=True`
3. **Collect** attentions, IG scores, and SHAP values
4. **Save** to JSON format in `results/`

Each JSON file contains:
- `text`: Original input text
- `tokens`: Tokenized text
- `attentions`: Nested dictionary of attention weights by layer/head
- `ig_scores`: Integrated Gradients importance scores
- `shap_values`: SHAP importance scores
- `prediction`: Sentiment prediction (positive/negative)
- `actual_label`: Ground truth label (for IMDb examples)

### Expected Outputs

After running the pipeline, the `results/` directory will contain:

```
results/
├── example_1.json          # Processed example data
├── example_2.json
├── ...
├── quick_viz_heatmap_2.png   # Quick visualization heatmap
└── quick_viz_bars_2.png      # Quick visualization bar chart
```

The Streamlit app reads from these JSON files to generate interactive visualizations.

### Using as a Library

```python
from utils.model_utils import GPT2Model
from utils.interpretability import compute_ig, compute_shap
from utils.json_utils import load_example_json, save_example_json
from pipeline import run_pipeline

# Process a text example
filepath = run_pipeline("This movie was fantastic!")

# Load results
data = load_example_json(filepath)
print(f"Tokens: {data['tokens']}")
print(f"Prediction: {data['prediction']}")
```

## Model Card

### Model Details

**Model Name:** GPT-2 Small
**Source:** HuggingFace Hub (`gpt2`)
**Developer:** OpenAI
**Model Type:** Autoregressive transformer language model
**Parameters:** 124M
**License:** MIT (open source)

**Architecture:**
- 12 transformer layers
- 12 attention heads per layer
- 768 hidden dimensions
- 1024 max position embeddings
- Vocabulary size: 50,257 tokens

**Training Data:** WebText corpus (40GB of text from URLs shared on Reddit with 3+ karma)

### Intended Uses

**This Project:** Interpretability research comparing attention to gradient-based methods on sentiment analysis

**Appropriate Uses:** Text generation, feature extraction, interpretability research, educational demonstrations

**Out-of-Scope:** Production deployment without evaluation, high-stakes decisions, tasks requiring factual accuracy

### Ethical Considerations and Limitations

**Biases:**
- Training data skews toward English-speaking, Western demographics (Reddit sources)
- May perform poorly on underrepresented groups or non-standard English
- Sentiment classifications may not generalize across cultural contexts

**Limitations:**
- Attention may not faithfully represent model reasoning (core research question)
- Transformer decision-making remains partially opaque despite interpretability methods
- Analysis limited to single task and model

**Mitigation:**
- Compares multiple interpretability methods to avoid over-reliance on attention
- Results presented with appropriate caveats
- Educational context emphasizes critical evaluation

### Model Citation

```
Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019).
Language models are unsupervised multitask learners.
OpenAI blog, 1(8), 9.
```

## Data Card

### Dataset Details

**Dataset Name:** IMDb Movie Review Dataset
**Source:** HuggingFace Datasets (`stanfordnlp/imdb`)
**Task:** Binary sentiment classification (positive/negative)
**Language:** English
**License:** Other - Available for research and educational use

**Dataset Splits:**
- Training: 25,000 reviews
- Test: 25,000 reviews
- Unsupervised: 50,000 reviews (unlabeled)

**Data Collection:**
- Reviews collected from IMDb (Internet Movie Database)
- Only reviews with strong sentiment (≤4 stars = negative, ≥7 stars = positive)
- No more than 30 reviews per movie
- Collected before 2011

### Intended Uses

**This Project:** Evaluating interpretability methods on sentiment classification and analyzing token-level importance

**Appropriate Uses:** Sentiment analysis research, NLP education, benchmarking, interpretability studies

### Data Limitations and Biases

**Biases:**
- Reviewers skew toward English-speaking, Western audiences (demographic and selection bias)
- Popular genres and pre-2011 sentiment patterns may be overrepresented
- Strong sentiment filter (≤4 or ≥7 stars) excludes nuanced opinions

**Limitations:**
- Review length and quality vary significantly
- May contain sarcasm, irony, or spoilers complicating sentiment analysis
- Subjective opinions may perpetuate biases in movie preferences

### Dataset Citation

```
Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011).
Learning word vectors for sentiment analysis.
In Proceedings of the 49th annual meeting of the association for computational linguistics (pp. 142-150).
```

## Critical Analysis

### Impact of This Project

**For the Research Community:**
This project contributes to the ongoing debate about transformer interpretability by providing empirical evidence about attention's explanatory power. While researchers frequently use attention visualizations in papers and presentations, this work quantifies when such visualizations are trustworthy and when they may be misleading. The comparative framework (attention vs. IG vs. SHAP) provides a template for rigorous interpretability evaluation.

**For Practitioners:**
Machine learning engineers and data scientists often rely on attention weights for debugging models and explaining predictions to stakeholders. This project reveals which scenarios warrant trust in attention visualizations and which require additional validation through gradient-based methods. This has practical implications for model auditing, bias detection, and explainable AI systems.

**For Education:**
As an educational tool, this project demonstrates that popular interpretability methods should be questioned rather than accepted at face value. It models critical thinking about AI tools, a core goal of DS-5690, and provides hands-on experience with multiple interpretability frameworks.

### What This Project Reveals

**Key Findings:**

1. **Layer-wise Attention Patterns**: Attention-IG/SHAP correlations remain relatively consistent across layers rather than showing clear hierarchical progression. Attention heatmaps reveal early syntactic focus, but throughout most layers attention concentrates heavily on the first token (likely the BOS token), with later layers also attending to a few additional tokens. This suggests attention may reflect positional biases rather than purely semantic task-relevant features.

2. **Attention-Attribution Misalignment**: Cases where attention and gradient-based methods disagree reveal that attention may track information flow rather than causal importance. A word can receive high attention without contributing to the final prediction.

3. **Method Complementarity**: Different interpretability methods capture different aspects of model behavior. Attention shows *what information is gathered*, while IG/SHAP show *what information influences the prediction*. Neither alone provides complete understanding.

4. **Task Dependency**: The reliability of attention as an explanation varies by example. Some reviews show strong attention-gradient alignment while others show substantial disagreement, highlighting the need to validate attention-based interpretations with gradient methods on a case-by-case basis.

### Limitations

**Scope Constraints:**
- Analysis limited to GPT-2 on sentiment classification; findings may not generalize to other models (BERT, LLaMA) or tasks (question answering, summarization)
- Small-scale study (due to computational constraints); large-scale statistical validation would strengthen conclusions
- Pre-trained model used without fine-tuning; fine-tuned models might show different attention patterns

**Methodological Limitations:**
- Integrated Gradients and SHAP both have their own assumptions and limitations (choice of baseline, computational cost)
- No ground-truth token importance available; we compare methods to each other rather than to known-correct explanations
- Attention aggregation (averaging across heads/layers) loses granular information

### Next Steps

**Extensions:** Test other models (BERT, LLaMA) and tasks (QA, NLI); implement attention interventions; compare to human judgments

**Deeper Analysis:** Identify specialized attention heads; use probing classifiers; test counterfactual examples

**Applications:** Develop hybrid interpretability methods; create best practices guide for practitioners

## Key Papers and References

### Primary References
1. **Jain, S., & Wallace, B. C. (2019).** Attention is not Explanation. *NAACL*.
2. **Wiegreffe, S., & Pinter, Y. (2019).** Attention is not not Explanation. *EMNLP*.
3. **Sundararajan, M., Taly, A., & Yan, Q. (2017).** Axiomatic Attribution for Deep Networks. *ICML*.
4. **Lundberg, S. M., & Lee, S. I. (2017).** A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
5. **Phuong, M., & Hutter, M. (2022).** Formal Algorithms for Transformers.

### Additional Resources
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer) - Interactive visualization
- [LLM Visualizations](https://bbycroft.net/llm) - 3D model visualizations
- [Captum Documentation](https://captum.ai/) - Interpretability library
- [SHAP Documentation](https://shap.readthedocs.io/) - SHAP explanations

## Troubleshooting

### Common Issues

**Issue:** Out of memory errors
```bash
# Solution: Use CPU instead of GPU
python pipeline.py --device cpu
```

**Issue:** Slow execution
```bash
# Solution: Process fewer examples or use shorter texts
python pipeline.py --imdb --num-examples 3
```

**Issue:** Module not found errors
```bash
# Solution: Ensure virtual environment is activated and dependencies installed
pip install -r requirements.txt
```

**Issue:** CUDA not available
```bash
# Solution: Code will automatically fall back to CPU
# No action needed unless you want to install CUDA
```

**Issue:** No examples found in Streamlit app
```bash
# Solution: Run pipeline first to generate JSON files
python pipeline.py --imdb --num-examples 10
```

## Contributing

This is an academic project for DS-5690. For questions or suggestions, please contact:
- **Email:** natasha.messier@vanderbilt.edu

## License

This project is for educational purposes as part of DS-5690 at Vanderbilt University.

**Model License:** GPT-2 is licensed under MIT  
**Dataset License:** IMDb dataset has its own 'Other' license (see HuggingFace)

## Acknowledgments

- **Instructor:** Jesse Spencer-Smith
- **TA:** Shivam Tyagi
- **Course:** DS-5690 Generative AI Models in Theory and Practice
- **Institution:** Vanderbilt University Data Science Institute

## Contact

**Natasha Messier**  
Vanderbilt University  
Data Science Institute  
natasha.messier@vanderbilt.edu

---

*Last Updated: November 2025*
