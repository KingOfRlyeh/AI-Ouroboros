# AI-Ouroboros

This repository is a mathematical model/simulation 
exploring feedback dynamics between human-generated content 
and multiple AI models competing on the open internet.


## Conceptual Framing

Let:

- \( q_i \): true human token distribution (Zipfian)
- \( p_{ij}(t) \): token distribution of AI model \( j \) at time \( t \)
- \( H(t) \): cumulative human content
- \( G(t) \): cumulative AI content

AI systems are retrained periodically on a mixture of human and AI outputs, creating a **closed feedback loop**.


## Human Content Model

Human production is modeled as exponentially growing content with a fixed lexical distribution:

\[
h(t) = c\,e^{dt}
\]

\[
q_i = \frac{1}{i^\alpha} \Big/ \sum_{k=1}^{I} \frac{1}{k^\alpha}
\]

This distribution remains *structurally stable* over time.


## AI Content Model

Each AI model \( j \) produces content according to:

\[
g_j(t) = a_j e^{b_j t}
\]

Total AI content:

\[
G(t) = G_0 + \sum_{s=1}^{t} \sum_{j=1}^{J} g_j(s)
\]



## Internet Mixture Distribution

At time \( t \), the effective internet distribution is:

\[
R_i(t) = \frac{
H_0 q_i + \sum_{s=1}^{t} \left[
h(s) q_i + \sum_{j=1}^{J} g_j(s)\, p_{ij}(s-1)
\right]
}{
H(t) + G(t)
}
\]

This distribution represents what future models observe when retraining.



## Retraining Rule

At retraining steps, each AI model constructs a filtered mixture:

\[
M_{ij}(t) = F_j q_i + (1 - F_j) R_i(t)
\]

Where:
- \( F_j \in [0,1] \) controls **human filtering strength**
- \( F_j = 1 \): pure human data
- \( F_j = 0 \): pure feedback loop

Sampling is then constrained via **top-p (nucleus) sampling**:

\[
\sum_{i \in \mathcal{A}} M_{ij}(t) \ge p
\]



## Memory Retention

Model updates blend new samples with prior state:

\[
p_{ij}(t) =
(1-\mu)\,\hat{p}_{ij}(t) + \mu\,p_{ij}(t-1)
\]

where \( \mu \) is the memory fraction.



## Metrics

### Entropy
\[
H[p_j(t)] = -\sum_i p_{ij}(t)\log_2 p_{ij}(t)
\]

### Variance
\[
\mathrm{Var}[p_j(t)] = \frac{1}{I}\sum_i (p_{ij}(t) - \bar p)^2
\]

### Token Loss

A token is considered *lost* if:

\[
p_{ij}(t) < \varepsilon
\]

Measured separately for:
- Human-origin tokens
- Long-tail tokens (lowest \( 20\% \) of \( q_i \))



## Simulation Scenarios

Four regimes are explored:

1. **Baseline**  
   Moderate filtering, frequent retraining, weak memory

2. **Slow Growth / Strong Memory**  
   Reduced growth rates, long retrain period, high \( \mu \)

3. **Diversity Protected**  
   \( F_j = 1 \), high top-p, strong memory

4. **High Feedback Stress**  
   Low \( F_j \), aggressive retraining, weak memory



## Key Observations

- Entropy collapse accelerates once AI content dominates \( R_i(t) \)
- Tail-token extinction precedes core vocabulary loss
- Memory introduces hysteresis that delays collapse
- Human filtering acts as a **structural entropy floor**



## Dependencies

- Python ≥ 3.9
- NumPy
- Matplotlib


