# Custom LLM Implementations: From Scratch to GPT-2 (124M)

PyTorch implementations of autoregressive decoder-only Transformer language models based on Andrej Karpathy's lectures.

---

## Architecture Comparison

| Specification | `gpt.ipynb` (Small LLM) | `gpt2_124m.ipynb` (GPT-2 Exact) |
|---|---|---|
| **Architecture** | Decoder-Only Transformer | Decoder-Only Transformer (OpenAI GPT-2) |
| **Parameters** | ~10.7M (configurable) | 124.44M |
| **Layers ($N_{\text{layer}}$)** | 4 – 6 | 12 |
| **Attention Heads ($N_{\text{head}}$)** | 4 – 6 | 12 |
| **Embedding Dimension ($d_{\text{model}}$)** | 256 – 384 | 768 |
| **Head Dimension ($d_{\text{k}}$)** | 64 | 64 |
| **Context Window ($T$)** | 256 tokens | 1,024 tokens |
| **Vocabulary Size ($V$)** | 65 (Char) / 50,257 (BPE) | 50,257 (OpenAI BPE) / 50,304 (Padded) |
| **Tokenizer** | Dual (`CharTokenizer` + `tiktoken` BPE) | Subword BPE (`tiktoken` `gpt2`) |
| **Attention Kernel** | Explicit scaled dot-product + `torch.tril` mask | `F.scaled_dot_product_attention` (FlashAttention) |
| **Feed-Forward Activation** | GELU (Standard) | GELU (tanh approximation) |
| **Weight Tying** | Separate embeddings | $W_{\text{wte}} = W_{\text{lm\_head}}$ (saves 38.6M params) |
| **Weight Initialization** | $\mathcal{N}(0, 0.02)$ | $\mathcal{N}(0, 0.02)$ with $1/\sqrt{2N_{\text{layer}}}$ residual scaling |
| **Precision** | Float32 | Bfloat16 / TF32 mixed precision |
| **Optimizer** | AdamW (`lr=3e-4`) | Fused AdamW (`lr=6e-4`, $\beta=(0.9, 0.95)$) with 2D/1D decay split |
| **Learning Rate Schedule** | Constant | Linear Warmup (100 steps) + Cosine Decay |
| **Target Hardware** | Apple Silicon (`mps`), CUDA, CPU | NVIDIA Ampere/Hopper GPU (`cuda`) |

---

## 1. `gpt.ipynb` — Small Decoder-Only Transformer

### Architecture Pipeline

```
Token Indices [B, T]
      │
      ├──► Token Embedding [B, T, C] ────┐
      │                                  ├──► Add ──► x [B, T, C]
      └──► Pos Embedding   [T, C]    ────┘       │
                                                 ▼
                               ┌──────────────────────────────────┐
                               │     Transformer Block (x N)      │
                               │                                  │
                               │  x ──► LayerNorm ──► Causal MHA  │
                               │  │                        │      │
                               │  └──► (+) ◄───────────────┘      │
                               │        │                         │
                               │        ├──► LayerNorm ──► MLP    │
                               │        │                   │     │
                               │        └──► (+) ◄──────────┘     │
                               └─────────────────┬────────────────┘
                                                 │
                                                 ▼
                                            LayerNorm
                                                 │
                                                 ▼
                                        Linear Projection
                                                 │
                                                 ▼
                                         Logits [B, T, V]
```

### Mathematical Definitions

1. **Causal Attention Masking**:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$
   $$M_{i,j} = \begin{cases} 0 & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases}$$

2. **Pre-LayerNorm Residual Block**:
   $$x_1 = x + \text{MultiHeadAttention}(\text{LayerNorm}(x))$$
   $$x_2 = x_1 + \text{MLP}(\text{LayerNorm}(x_1))$$

3. **Inference Sampling (Temperature + Top-K)**:
   $$P(w_i) = \frac{\exp(z_i / \tau)}{\sum_{j \in \text{Top-K}} \exp(z_j / \tau)}$$
   - $\tau$: Temperature scaling factor.
   - Top-K: Truncates logits distribution to top $K$ candidates before softmax.

---

## 2. `gpt2_124m.ipynb` — Production GPT-2 Reproduction

### Key Structural Differences from `gpt.ipynb`

```
┌────────────────────────────────────────────────────────────────────────┐
│                        GPT-2 124M Specifics                            │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Fused QKV:          Single Linear(C, 3C) splits into Q, K, V        │
│ 2. FlashAttention:     O(N) memory attention via SRAM tiling kernel    │
│ 3. Weight Tying:       Token Embedding Matrix == LM Head Matrix        │
│ 4. Residual Scaling:   Linear projection std scaled by 1/sqrt(24)      │
│ 5. Optimizer Tuning:   Weight decay (0.1) on 2D weights only           │
│ 6. Batch Simulation:   Gradient accumulation to reach 65K-524K tokens  │
└────────────────────────────────────────────────────────────────────────┘
```

### Systems Optimizations

1. **FlashAttention**:
   Replaces the $O(T^2)$ materialized attention matrix with `torch.nn.functional.scaled_dot_product_attention`, fusing QK multiplication, softmax, and V reduction directly into GPU SRAM.

2. **TF32 & Bfloat16 Mixed Precision**:
   - `torch.set_float32_matmul_precision('high')`: Activates Tensor Cores for 19-bit TensorFloat-32 matrix multiplications.
   - `torch.autocast(dtype=torch.bfloat16)`: Executes forward pass in 16-bit brain floating point, maintaining 8-bit dynamic range without gradient scaling.

3. **Weight Tying Formulation**:
   $$W_{\text{lm\_head}} \equiv W_{\text{wte}} \in \mathbb{R}^{50257 \times 768}$$
   Reduces model memory footprint by $50,257 \times 768 \times 4 \text{ bytes} \approx 154.4\text{ MB}$.

4. **Cosine Learning Rate Schedule with Warmup**:
   $$\eta_t = \begin{cases} 
   \eta_{\text{max}} \cdot \frac{t + 1}{T_{\text{warmup}}} & \text{if } t < T_{\text{warmup}} \\
   \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{t - T_{\text{warmup}}}{T_{\text{max}} - T_{\text{warmup}}}\pi\right)\right) & \text{if } t \ge T_{\text{warmup}}
   \end{cases}$$
   - $\eta_{\text{max}} = 6 \times 10^{-4}$
   - $\eta_{\text{min}} = 6 \times 10^{-5}$
   - $T_{\text{warmup}} = 100$

5. **Weight Decay Parameter Split**:
   - `dim >= 2` (Linear weights, Embedding tables): `weight_decay = 0.1`
   - `dim < 2` (Biases, LayerNorm scale $\gamma$ and shift $\beta$): `weight_decay = 0.0`

6. **Pretrained Weights Loader**:
   Implements `GPT.from_pretrained('gpt2')` to copy official HuggingFace checkpoint weights directly into the custom module hierarchy for exact numerical validation.

---

## Environment Setup & Execution

### Local (Apple Silicon / macOS)
```bash
# Clone / navigate to repo
git clone <repo_url>
cd llm

# Install dependencies via uv
uv sync

# Run Jupyter kernel
uv run jupyter lab
```

### Google Colab (Pro / GPU)
1. Set runtime: **Runtime** $\to$ **Change runtime type** $\to$ **GPU** (A100, L4, or T4).
2. Install dependencies:
   ```bash
   !pip install tiktoken transformers
   ```
3. Open either notebook:
   - `gpt.ipynb`: Educational build with char/BPE tokenizers and prompt conditioning.
   - `gpt2_124m.ipynb`: Production-grade GPT-2 124M with FlashAttention, gradient accumulation, and pretrained weight loading.
