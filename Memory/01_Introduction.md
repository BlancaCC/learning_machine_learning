\chapter{Introdution}

# What is Machine learning   

As it is said in the introduction of chapter one in  @BishopPatternRecognition
**machine learning** *is the field of pattern 
recognition that is concerned with the automatic
 discovery of regularities in data and the use of
these regularities to take actions such as 
classifying the data into different categories*.
 
The mathematical formulation of this according to @HastieStatisticalLearing (chapter two) is the following. 

Let $X \in \mathbb{R}^p$  a real valued random input vector, and  $Y \in \mathbb{R}$  a real valued random out-put variable, with joint distribution $Pr(X,Y)$. We seek a function $f(X)$ for predicting $Y$ given values of the input $X$.

This theory requires a *loss function* $L(Y,f(X))$
for penalizing errors in prediction,
by far the most common and convenient is (**why?**) *squared error loss*
$$L(Y,f(X)) = ((Y - f(X))^2$$
This leads us to a criterion for choosing $f$.

The expected (squared) prediction error, EPE.
\begin{equation}
    EPE(f) =
    E[(Y - f(X))^2]
    =
    \int (y - f(x))^2 Pr(d_x, d_y)
\end{equation}

By conditioning on $X$ and using the Law of total expectation (or law of iterated expectations) we can write EPE as

\begin{equation}
EPE(f) =
E_x E_{Y|X}
\left(
    [Y-f(X)]^2 | X
\right)
\end{equation}

and we see that is suffices to minimize EPE pointwise:

\begin{equation}
    f(x) =
    argmin_c E_{Y|X}
    \left(
    [Y-c]^2 | X = x
\right)
\end{equation}.

The solution is
\begin{equation}
    f(x)
    =
    E(Y | X=x).
\end{equation}
(**Why**)

