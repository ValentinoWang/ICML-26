## Epsilon Upper-Bound Note

### Current status in the manuscript

The current appendix already states the bias-robust UUB condition in assumption form:

\[
\|b(e_t)-\Delta\phi_t\|_P \le \epsilon,
\]

and derives the steady-state radius scaling

\[
R^\star \propto \epsilon / c.
\]

What is still missing is a finite-sample upper bound for `epsilon`.

### Safest way to extend the appendix

Do **not** promise a sharp closed-form theorem unless all authors agree on the assumptions. The safe addition is a short proposition that decomposes

\[
\epsilon
=
\|b(\mathcal D)-\widehat{\Delta\phi}(\mathcal D)\|_P
\]

into three terms:

\[
\epsilon
\;\le\;
\underbrace{\inf_{f\in\mathcal F_{\mathrm{ST}}}\|b-f\|_P}_{\text{approximation}}
\;+\;
\underbrace{\|\widehat f_N-f^\star\|_P}_{\text{estimation}}
\;+\;
\underbrace{\|\widehat{\Delta\phi}-\widehat f_N\|_P}_{\text{optimization}}.
\]

Then state informally that:

1. `Approximation error` can be made arbitrarily small on compact domains because Set Transformers are universal approximators for permutation-invariant functions.
2. `Estimation error` decays with set size `N`; under sub-Gaussian noise / bounded embeddings, it is of standard order
   \[
   O\!\left(\sqrt{\frac{\mathfrak C(\mathcal F_{\mathrm{ST}})+\log(1/\delta)}{N}}\right),
   \]
   where `\mathfrak C` is a capacity term such as Rademacher complexity.
3. `Optimization error` is the training residual and vanishes if the filter is fit to sufficient accuracy.

Combining them gives the qualitative finite-sample statement

\[
\epsilon_N
\;=\;
O\!\left(
\epsilon_{\mathrm{approx}}
+
\sqrt{\frac{\mathfrak C(\mathcal F_{\mathrm{ST}})+\log(1/\delta)}{N}}
+
\epsilon_{\mathrm{opt}}
\right).
\]

Substituting into the current UUB result yields

\[
R^\star
=
O\!\left(
\frac{
\epsilon_{\mathrm{approx}}
+
\sqrt{(\mathfrak C(\mathcal F_{\mathrm{ST}})+\log(1/\delta))/N}
+
\epsilon_{\mathrm{opt}}
}{c}
\right).
\]

### Suggested wording for the appendix

Under Assumption~A2 (dense sampling), bounded/sub-Gaussian candidate perturbations, and a permutation-invariant bias functional `b(\mathcal D)` lying in the closure of the Set-Transformer hypothesis class, the residual correction error admits the decomposition

\[
\epsilon_N
\le
\epsilon_{\mathrm{approx}}
+
\epsilon_{\mathrm{stat}}(N,\delta)
+
\epsilon_{\mathrm{opt}},
\]

where `\epsilon_{\mathrm{approx}}` is the function-approximation error, `\epsilon_{\mathrm{opt}}` is the optimization residual, and with probability at least `1-\delta`,

\[
\epsilon_{\mathrm{stat}}(N,\delta)
=
O\!\left(
\sqrt{\frac{\mathfrak C(\mathcal F_{\mathrm{ST}})+\log(1/\delta)}{N}}
\right).
\]

Hence the bias-robust UUB radius scales as

\[
R^\star
=
O\!\left(\frac{\epsilon_N}{c}\right)
=
O\!\left(
\frac{
\epsilon_{\mathrm{approx}}
+
\sqrt{(\mathfrak C(\mathcal F_{\mathrm{ST}})+\log(1/\delta))/N}
+
\epsilon_{\mathrm{opt}}
}{c}
\right).
\]

This formalizes the intuitive claim that larger candidate sets improve bias estimation accuracy and tighten the steady-state radius.

### Discussion with authors

This is likely acceptable for the appendix because it is conservative and aligns with the current theory. What probably requires author agreement is only the exact level of formality:

- `Minimal version`: one proposition plus proof sketch, no explicit complexity constant.
- `Stronger version`: instantiate `\mathfrak C(\mathcal F_{\mathrm{ST}})` using a specific norm-bounded Set-Transformer class.

For rebuttal timing, the minimal version is the practical choice.

### Relevant manuscript locations

- `Paper/LaTEX/icml2026/main.tex` lines 189-191
- `Paper/LaTEX/icml2026/main.tex` lines 798-808
- `Paper/LaTEX/icml2026/main.tex` lines 820-863
