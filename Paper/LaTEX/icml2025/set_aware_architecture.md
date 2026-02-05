# Architecture of the Set-Aware Geometric Filter (Figure 2)

Below is the text description derived from the paper implementation and Algorithm 1 in `main.tex`.

## Core Structure (left–middle–right panels)

- **Input & low-rank projection (left):** Each candidate update is
  \(\Delta\theta_i=\theta_{t,i}-\theta_t\). Project into a low-rank subspace
  \(U\in\mathbb{R}^{D\times d}\) with \(d\ll D\) to obtain
  \(\mathbf{z}_i=U^\top\Delta\theta_i\). This preserves drift geometry in a compact
  embedding and reduces compute/noise.

- **Set Transformer aggregation (middle):** The set \(\{\mathbf{z}_i\}\) is fed
  to a Set Transformer self-attention block (SAB), which captures global
  covariance and relative geometry across candidates (not pointwise magnitudes).

- **Dual-head outputs (right):**
  1. **Reweighting head** outputs weights \(w_i\) for variance control
     (contraction).
  2. **Correction head** outputs a bias estimate in embedding space
     \(\Delta\phi_{\text{emb}}\), which is mapped back to parameter space via
     \(\Delta\phi=U\Delta\phi_{\text{emb}}\).

## Update Rule (output end)

- Weighted update:
  \[
  \Delta\theta_{\text{weighted}}=\frac{\sum_i w_i\,\Delta\theta_i}{\sum_i w_i},
  \quad
  \theta_{\text{weighted}}=\theta_t+\Delta\theta_{\text{weighted}}.
  \]
- Explicit correction:
  \[
  \theta_{t+1}=\theta_{\text{weighted}}-\eta\,\Delta\phi.
  \]
This is the “active geometric correction” shown in Figure 2.

## Algorithm 1 (text form)

1. Input candidate updates \(\{\Delta\theta_i\}\) and low-rank basis \(U\).
2. Project: \(\mathbf{z}_i\leftarrow U^\top\Delta\theta_i\).
3. Apply set-aware filter: \((w_i,\Delta\phi_{\text{emb}})\leftarrow f_\psi(\{\mathbf{z}_i\})\).
4. Weighted update: \(\Delta\theta_{\text{weighted}}\leftarrow \sum_i w_i\Delta\theta_i / \sum_i w_i\).
5. Back-project correction: \(\Delta\phi\leftarrow U\Delta\phi_{\text{emb}}\).
6. Update: \(\theta_{t+1}\leftarrow \theta_t+\Delta\theta_{\text{weighted}}-\eta\Delta\phi\).

## Training/Proxy Signal (dashed arrow in Figure 2)

The dashed arrow denotes the validation/proxy signal used only during training
to supervise the correction head. At inference, the filter consumes only the
candidate set and does not access clean labels or validation data.
