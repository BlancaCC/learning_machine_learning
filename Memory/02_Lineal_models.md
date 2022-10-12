# Lineal models 

The simplest linear model for regression is 

\begin{equation}
y(x,w) = w \cdot x^T
\end{equation}
where $x \in \{1\} \times \mathbb{R}^d$ and 
$w \in \mathbb{R}^{d+1}$.


Significant models other solutions: 

- Splines  
- Transformation of $x$ by *basic functions*

\begin{equation}
    y(x,w) = w \cdot \phi(x)^T
\end{equation}

## Questions related with the relationship between splines and basic functions 

- **Why not continuing using polynomial functions as splines?**
- **Think about relation with splines.** Differences -> create table  with pro and cons
- **What about simulating analytic functions with?**

## Choices for the basic functions  

### Gaussian basic functions 

\begin{equation}
    \phi_j (x) =
    exp \left\{
        - \frac{
     (x - \mu_j) ^2
         }{
            2 s^2
         }
        \right\}
\end{equation}



