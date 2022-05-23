# Matrices
In the following we want to restrict our discussion to matrices with real entries. 

## Square matrices
Every square matrix can be split into a symmetric and an antisymmetric (skew-symmetric)
part

$$
\newcommand{\lra}[1]{{ \left( #1 \right) }}
A=\underbrace{\frac{1}{2}\lra{A + A^T}}_{\text{symmetric}}
+\underbrace{\frac{1}{2}\lra{A - A^T}}_{\text{antisymmetric}}
$$

## Symmetric matrices
A symmetric matrix is a square matrix, that is equal to it's transpose.

$$
A = A^T
$$

The sum of two symmetric matrices is again a symmetric matrix, but the product of two matrices is
in general not symmetric. The product of two symmetric matrices $A$ and $B$ is symmetric only, if the two matrices commute:

$$
AB = (AB)^T \ \text{if}\ AB = BA
$$

Symmetric matrices with real entries have only real eigenvalues. That is why in principle, every symmetric
matrix is equivalent to a diagonal matrix with its eigenvalues being the entries on the diagonal.  

## Applications of Matrices
### Solving unsolvable equations
We start with a simple equation:

$$
A^2 * 1 = 1 
$$

What is the transformation A, when applied twice, which turns 1 into 1 ? The answer is simple: $A=1$ or $A=-1$. But what is the transformation $B$, when applied twice, which turns 1 into -1 ?

$$
B^2 * 1 = -1 
$$


The answer it turns out, is not so simple. Based on the common rules of multiplication for real numbers, there seems to be no way to solve this equation for $B$. 

However there is a way out of the dilemma. The reader might have noticed, that we didn't call $A$ or $B$ a number or a variable, but a transformation. We could have also called it also an operator. This should be a hint, that maybe simple numbers are not enough to solve such an equation. 

So let's think a bit. Can this equation maybe interpreted in a geometrical way? Let's imagine the line of numbers. What we want is a way, to move or transform the point 1 to the point -1, but doing it with two steps (One step would be simple, this could be reflection at the zero point). If we draw the line on a sheet of paper and look from far away the solution might be become more obvious: The line is embedded on the two dimensional surface of the paper! So we might interpret our source point 1 in fact as a vector (1,0) and our target point -1 as vector (-1,0). Maybe this helps. Let's rewrite our equation in two dimensions

$$
B^2 * \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} -1 \\ 0 \end{pmatrix} 
$$

So what we want is a transformation $B$, which when used twice on the vector (1,0) will turn the vector into the vector (-1,0). Vector (-1,0) is pointing into the opposite direction of vector (1,0), so it is basically rotated by 180 degrees. So we are searching for a transformation which rotates a vector in two steps by 180 degrees. Now it should be obvious that the solution for $B$ must be a transformation, which rotates a vector by 90 degrees, in short $B = R(90^{\circ})$ (or a rotation in the other direction by -90 degrees).

So we found a solution by geometrical intuition, let's try to make it precise. From linear algebra we know, that a transformation which turns a 2D vector into another 2D vector must be a 2x2 matrix. So let's rewrite the equation as a matrix equation

$$
\begin{pmatrix} a & b \\ c & d \end{pmatrix} * \begin{pmatrix} a & b \\ c & d \end{pmatrix} * \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} -1 \\ 0 \end{pmatrix} 
$$

If we compute the matrix product on the left hand side we get

$$
\begin{pmatrix} a^2 + bc & b (a+d) \\ c(a+d) & d^2 + bc \end{pmatrix} \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} -1 \\ 0 \end{pmatrix} 
$$

Further simplifying we get two equations
$$

\begin{align}
a^2+bc = -1 && ac + cd = 0
\end{align}
$$

To solve this equation one might attempt to set $c=0$. But in this case we would end up where we started because the equation left would be $a^2=-1$. So we have to assume $c \neq 0$ to find a sensible solution. So we get

$$
\begin{align}
c = -\frac{a^2+1}{b} && a = -d
\end{align}
$$

From these equation we see that we also have to assume $b \neq 0$. But without loss of generality we can set $a=0$,$d=0$ and are left with

$$
bc = -1
$$

If we restrict ourself to integer numbers, we finally have two solutions for our transformation

$$
\begin{align}
B = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}  && B^* = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}
\end{align}
$$

So what seemed impossible to solve in 1D with simple numbers turned out to have quite simple solutions in 2D in the form of 2x2 matrices.

### Solving polynomial equations
A polynomial equation of order $n$ is an equation that look like

$$
a_nx^n + a_{n-1}x^{n-1} + \cdots + a_1 x + a_0 = 0
$$

Solving these kind of equations has been a hobby of mathematicans since the invention of math. 
But whereas quadratic equations ($n=2$) could be solved since ancient times, finding a 
general solution for cubic ($n=3$) and quartic ($n=4$) equations turned out to be much harder. 
Rafael Bombelli made a crucial step in 1572 when - in a desparate move - he invented complex 
numbers to solve cubic equations, which had been unsolvable up to that time. Had he known matrices
and linear algebra, the invention of "complex numbers" would have been unnecessary.

This is so, because there is a deep connection between polynominal equations and matrices. Actually
it turns out that **any polynomial with degree $n$ is the characteristic polynomial of 
some [companion matrix](https://en.wikipedia.org/wiki/Companion_matrix) of order $n$**. So the problem of solving a polynominal equation is equivalent 
to solving the characteristic equation of the companion matrix. But solving the characteristic 
equation of a matrix means computing the eigenvalues of that matrix. And computing eigenvalues has 
a simple geometric meaning: They give the factor by which an eigenvector 
(a vector which direction is left unchanged by the matrix transformation) is stretched by 
the matrix transformation. The eigenvalues are the scale factors of the linear transformation 
represented by the matrix. This means that solving a polynomial equation is equivalent to computing
the scale factors of a corresponding linear transformation.

Knowning this we can nowadays understand, why some polynominal equations have no solutions.
These are the equations, which correspond to matrices, which don't have eigenvalues, meaning the
transformation doesn't leave any vector unchanged. From linear algebra we know, that these 
transformations describe rotations. And this is the connection to the 2d rotation matrix we found 
in the previous chapter. The polynominal equation 

$$
x^2 = -1
$$

has no solution, because it corresponds to a matrix describing a 90 degree rotation. 

On the other side the [fundamental theorem of algebra](https://en.wikipedia.org/wiki/Fundamental_theorem_of_algebra) is easy to understand, when thinking of polynomials as being represented by matrices. The number of solutions to a polynomial of degree $n$ is the same as the number of eigenvalues of the corresponding companion matrix.

