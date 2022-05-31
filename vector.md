---
author:
- Andreas Maier
bibliography:
- promotion.bib
title: Mostly harmless notes
---

# Vector calculus

## Longitudinal and transversal projection of vectors

A vector $\boldsymbol{a}$ can be split into two parts: the longitudinal part $\boldsymbol{a}_{\parallel}$,
which is parallel to another vector $\boldsymbol{b}$ and the transversal part $\boldsymbol{a}_{\perp}$,
which is perpendicular to $\boldsymbol{b}$. The length of the longitudinal part $a_{\parallel}$
and the transversal part $a_{\perp}$ can be computed from geometry (see figure [\[fig:projection\]](#fig:projection){reference-type="ref" reference="fig:projection"})
$$\begin{aligned}
\frac{a_{\parallel}}{a}=\cos \alpha & \Rightarrow
a_{\parallel}= a \cos\alpha = \frac{a b \cos \alpha}{b} = \frac{\boldsymbol{a}\cdot\boldsymbol{b}}{b},\\
\frac{a_{\perp}}{a}=\sin \alpha & \Rightarrow 
a_{\perp}= a \sin\alpha = \frac{a b \sin \alpha}{b} = \frac{\left\lvert\boldsymbol{a}\times\boldsymbol{b}\right\rvert}{b}. \label{eq:protrans}\end{aligned}$$
But from the Pythagorean theorem we can get another expression for the length of the transversal
part
$$\begin{aligned}
a^2=a_{\parallel}^2+a_{\perp}^2 & \Rightarrow
a_{\perp}^2 = a^2-a_{\parallel}^2= a^2-\frac{(\boldsymbol{a}\cdot\boldsymbol{b})^2}{b^2}.\label{eq:protrans2}\end{aligned}$$
Substituting equation [\[eq:protrans\]](#eq:protrans){reference-type="eqref" reference="eq:protrans"} in equation [\[eq:protrans2\]](#eq:protrans2){reference-type="eqref" reference="eq:protrans2"} we get
$$\begin{aligned}
\frac{(\left\lvert\boldsymbol{a}\times\boldsymbol{b}\right\rvert)^2}{b^2}=a^2-\frac{(\boldsymbol{a}\cdot\boldsymbol{b})^2}{b^2},\end{aligned}$$
which leads us to the following expression for the square of the norm of the cross product
$$\begin{aligned}
\left\lvert\boldsymbol{a}\times\boldsymbol{b}\right\rvert^2 = (a b)^2 - (\boldsymbol{a}\cdot\boldsymbol{b})^2.\end{aligned}$$
This is again Lagrange's identity (see [\[eq:lagrident\]](#eq:lagrident){reference-type="eqref" reference="eq:lagrident"}).

## Vector identities

In this chapter we show the derivation of some vector quantities in cartesian
tensor notation.

### $(\boldsymbol{u}\cdot\nabla) \boldsymbol{v}$ {#vecid01}

For some arbitraty vectors $u_i, v_i$ we can write
$$\begin{aligned}
u_j \frac{\partial}{\partial r_j} v_i &= \overbrace{u_j \frac{\partial}{\partial r_i} v_j - u_j \frac{\partial}{\partial r_i} v_j}^0 
+ u_j\frac{\partial}{\partial r_j} v_i \\
&= u_j \frac{\partial}{\partial r_i} v_j - \delta_{ik}\delta_{jl} u_j \frac{\partial}{\partial r_k} v_l 
+\delta_{il}\delta_{jk} u_j \frac{\partial}{\partial r_k} v_l \\
&= u_j \frac{\partial}{\partial r_i} v_j 
- (\delta_{ik}\delta_{jl}-\delta_{il}\delta_{jk})u_j \frac{\partial}{\partial r_k} v_l \\
&= u_j \frac{\partial}{\partial r_i} v_j - \epsilon_{mij} \epsilon_{mkl} u_j \frac{\partial}{\partial r_k} v_l \\
&= u_j \frac{\partial}{\partial r_i} v_j - \epsilon_{ijm} u_j \epsilon_{mkl} \frac{\partial}{\partial r_k} v_l.\end{aligned}$$
In vector notation this can be expressed like
$$\begin{aligned}
(\boldsymbol{u}\cdot\nabla) \boldsymbol{v} = 
\boldsymbol{u}\cdot (\nabla \boldsymbol{v})-\boldsymbol{u} \times (\nabla \times \boldsymbol{v})
\label{eq:vecid01}\end{aligned}$$

### $(\boldsymbol{v}\cdot\nabla) \boldsymbol{v}$

Inserting $u_j=v_j$ into equation [\[eq:vecid01\]](#eq:vecid01){reference-type="eqref" reference="eq:vecid01"} yields
$$\begin{aligned}
v_j \frac{\partial}{\partial r_j} v_i = v_j \frac{\partial}{\partial r_i} v_j 
- \epsilon_{ijm} v_j \epsilon_{mkl} \frac{\partial}{\partial r_k} v_l\end{aligned}$$
For $v_j \frac{\partial}{\partial r_i} v_j$ we can write
$$\begin{aligned}
v_j \frac{\partial}{\partial r_i} v_j &= \frac{\partial}{\partial r_i} (v_j v_j) - v_j \frac{\partial}{\partial r_i} v_j\end{aligned}$$
and therefore
$$\begin{aligned}
v_j \frac{\partial}{\partial r_i} v_j = \frac{\partial}{\partial r_i}  \left( \frac{1}{2} v_j v_j \right) .\end{aligned}$$
Using this we get for
$$\begin{aligned}
v_j \frac{\partial}{\partial r_j} v_i = \frac{\partial}{\partial r_i}  \left( \frac{1}{2} v_j v_j \right) 
- \epsilon_{ijm} v_j \epsilon_{mkl} \frac{\partial}{\partial r_k} v_l\end{aligned}$$
or in vector notation
$$\begin{aligned}
(\boldsymbol{v}\cdot\nabla) \boldsymbol{v} = 
\frac{1}{2} \nabla \boldsymbol{v}^2-\boldsymbol{v} \times (\nabla \times \boldsymbol{v})
\label{eq:vecid02}\end{aligned}$$

### $\nabla \times  \left( \boldsymbol{u} \times \boldsymbol{v} \right)$

The $i$-th component of the rotation of a cross product of two vectors is
$$\begin{aligned}
 \left[ \nabla \times  \left( \boldsymbol{u} \times \boldsymbol{v} \right)  \right] _i 
&= \epsilon_{ijk} \frac{\partial}{\partial r_j} \epsilon_{klm} u_l v_m \\
&= \epsilon_{kij} \epsilon_{klm} \frac{\partial}{\partial r_j} (u_l v_m) \\
&= (\delta_{il}\delta_{jm}-\delta_{im}\delta_{jl}) \frac{\partial}{\partial r_j} (u_l v_m) \\
&= \frac{\partial}{\partial r_j} (u_i v_j) - \frac{\partial}{\partial r_j} (u_j v_i) \\
&= u_i \frac{\partial}{\partial r_j} v_j + v_j \frac{\partial}{\partial r_j} u_i - u_j \frac{\partial}{\partial r_j} v_i - v_i \frac{\partial}{\partial r_j} u_j\end{aligned}$$
It can be written in vector notation like
$$\begin{aligned}
\nabla \times  \left( \boldsymbol{u} \times \boldsymbol{v} \right)  
= \boldsymbol{u} (\nabla \cdot \boldsymbol{v}) - \boldsymbol{v} (\nabla \cdot \boldsymbol{u})
+ (\boldsymbol{v} \cdot \nabla) \boldsymbol{u} - (\boldsymbol{u} \cdot \nabla) \boldsymbol{v}\end{aligned}$$
