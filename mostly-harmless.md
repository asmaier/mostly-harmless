# Introduction

This book holds a collection of mostly harmless notes on mathematics and physics. The topics covered are simply all things I found interesting. I mostly cover topics which
you won’t find easily in Wikipedia or a usual text book. Often I also approach things from a different point of view (namely my point of view). My hope is that these notes grow into an interesting addition to the existing literature.

# Mathematics

### Numbers

#### Triangle inequality

##### Number inequality

We start with the obvious inequality that the absolute value of a number
$$\left|a\right|$$ is always greater or equal the number $$a$$ itself

$$\begin{aligned}
|a| \geq a\end{aligned}$$

Using this we can prove that for numbers $$a$$ and $$b$$ we have

$$\begin{aligned}
|a| + |b|  \geq |a+b|\end{aligned}$$

because

$$\begin{aligned}
(|a| + |b|)^2  &\geq (|a+b|)^2 \\
|a|^2  + 2|a||b| + |b|^2 &\geq |a^2 + 2ab + b^2| \\
2|a||b| &\geq 2|ab|\end{aligned}$$

The last line is always true if $$|a| \geq a$$, which completes our
prove.

#### Cauchy-Schwarz inequality

Because
$$\sin^2(\phi) + \cos^2(\phi) = 1$$ we have

$$\begin{aligned}
|a|^2|b|^2 = |a|^2|b|^2 (\sin^2(\phi) + \cos^2(\phi))\end{aligned}$$

With the definition of the scalar product
$$|a\cdot b| = |a||b|\cos(\phi)$$ and the cross product
$$|a\times b| = |a||b|\sin(\phi)$$ of vectors $$a$$ and $$b$$ we can
write this expression as

$$\begin{aligned}
|a|^2|b|^2 = |a\times b|^2 + |a\cdot b|^2 
\label{eq:lagrident}\end{aligned}$$

This is called [Lagrange’s identity](https://en.wikipedia.org/wiki/Lagrange%27s_identity). Since all terms are squared and
therefor positive we immediatelly can derive the inequalities

$$\begin{aligned}
|a|^2|b|^2 &\geq |a\cdot b|^2 \\
|a|^2|b|^2 &\geq |a\times b|^2\end{aligned}$$

The first of these equations is the Cauchy-Schwarz inequality. The
second equations doesn’t seem to have a name in the literature.

The Cauchy-Schwarz inequality comes in many different forms. For example
for $$n$$-dimensional vectors it can be written in cartesian coordinates
like

$$\begin{aligned}
\sum (a_i)^2 \sum (b_i)^2 \geq \left(\sum a_i b_i \right)^2\end{aligned}$$

This can even be generalized to uncountable infinite dimensional
vectors (also called continuous square integrable functions) like

$$\begin{aligned}
\left(\int \left|a(x)\right|^2 dx\right) \cdot \left(\int \left|b(x)\right|^2 dx\right) \geq \left|\int a(x) \cdot b(x)dx\right|^2\end{aligned}$$

### Matrices

In the following we want to restrict our discussion to matrices with
real entries.

#### Square matrices

Every square matrix can be split into a symmetric and an antisymmetric
(skew-symmetric) part

$$\begin{aligned}
A=\underbrace{\frac{1}{2} \left( A + A^T \right) }_{\text{symmetric}}
+\underbrace{\frac{1}{2} \left( A - A^T \right) }_{\text{antisymmetric}}\end{aligned}$$

#### Symmetric matrices

A symmetric matrix is a square matrix, that is equal to it’s transpose.

$$\begin{aligned}
A = A^T\end{aligned}$$

The sum of two symmetric matrices is again a symmetric matrix, but
the product of two matrices is in general not symmetric. The product of
two symmetric matrices $$A$$ and $$B$$ is symmetric only, if the two
matrices commute:

$$\begin{aligned}
AB = (AB)^T \ \text{if}\ AB = BA\end{aligned}$$

Symmetric matrices with real entries have only real eigenvalues. That
is why in principle, every symmetric matrix is equivalent to a diagonal
matrix with its eigenvalues being the entries on the diagonal.

#### Applications of Matrices

##### Solving unsolvable equations

We start with a simple equation:

$$\begin{aligned}
A^2 * 1 = 1 \end{aligned}$$

What is the transformation A, when applied twice, which turns 1 into
1 ? The answer is simple: $$A=1$$ or $$A=-1$$. But what is the
transformation $$B$$, when applied twice, which turns 1 into -1 ?

$$\begin{aligned}
B^2 * 1 = -1 \end{aligned}$$

The answer it turns out, is not so simple. Based on the common rules
of multiplication for real numbers, there seems to be no way to solve
this equation for $$B$$.

However there is a way out of the dilemma. The reader might have
noticed, that we didn’t call $$A$$ or $$B$$ a number or a variable, but
a transformation. We could have also called it also an operator. This
should be a hint, that maybe simple numbers are not enough to solve such
an equation.

So let’s think a bit. Can this equation maybe interpreted in a
geometrical way? Let’s imagine the line of numbers. What we want is a
way, to move or transform the point 1 to the point -1, but doing it with
two steps (One step would be simple, this could be reflection at the
zero point). If we draw the line on a sheet of paper and look from far
away the solution might be become more obvious: The line is embedded on
the two dimensional surface of the paper! So we might interpret our
source point 1 in fact as a vector (1,0) and our target point -1 as
vector (-1,0). Maybe this helps. Let’s rewrite our equation in two
dimensions

$$\begin{aligned}
B^2 * \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} -1 \\ 0 \end{pmatrix} \end{aligned}$$

So what we want is a transformation $$B$$, which when used twice on
the vector (1,0) will turn the vector into the vector (-1,0). Vector
(-1,0) is pointing into the opposite direction of vector (1,0), so it is
basically rotated by 180 degrees. So we are searching for a
transformation which rotates a vector in two steps by 180 degrees. Now
it should be obvious that the solution for $$B$$ must be a
transformation, which rotates a vector by 90 degrees, in short
$$B = R(90^{\circ})$$ (or a rotation in the other direction by -90
degrees).

So we found a solution by geometrical intuition, let’s try to make it
precise. From linear algebra we know, that a transformation which turns
a 2D vector into another 2D vector must be a 2x2 matrix. So let’s
rewrite the equation as a matrix equation

$$\begin{aligned}
\begin{pmatrix} a & b \\ c & d \end{pmatrix} * \begin{pmatrix} a & b \\ c & d \end{pmatrix} * \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} -1 \\ 0 \end{pmatrix} \end{aligned}$$

If we compute the matrix product on the left hand side we get

$$\begin{aligned}
\begin{pmatrix} a^2 + bc & b (a+d) \\ c(a+d) & d^2 + bc \end{pmatrix} \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} -1 \\ 0 \end{pmatrix} \end{aligned}$$

Further simplifying we get two equations

$$\begin{aligned}
a^2+bc = -1 &\ & ac + cd = 0\end{aligned}$$

To solve this equation one might attempt to set $$c=0$$. But in this
case we would end up where we started because the equation left would be
$$a^2=-1$$. So we have to assume $$c \neq 0$$ to find a sensible
solution. So we get

$$\begin{aligned}
c = -\frac{a^2+1}{b} &\ & a = -d\end{aligned}$$

From these equation we see that we also have to assume $$b \neq 0$$.
But without loss of generality we can set $$a=0$$,$$d=0$$ and are left
with

$$\begin{aligned}
bc = -1\end{aligned}$$

If we restrict ourself to integer numbers, we finally have two
solutions for our transformation

$$\begin{aligned}
B = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}  &\ & B^* = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}\end{aligned}$$

So what seemed impossible to solve in 1D with simple numbers turned
out to have quite simple solutions in 2D in the form of 2x2 matrices.

##### Solving polynomial equations

A polynomial equation of order $$n$$ is an equation that look like

$$\begin{aligned}
a_nx^n + a_{n-1}x^{n-1} + \cdots + a_1 x + a_0 = 0\end{aligned}$$

Solving these kind of equations has been a hobby of mathematicans
since the invention of math. But whereas quadratic equations ($$n=2$$)
could be solved since ancient times, finding a general solution for
cubic ($$n=3$$) and quartic ($$n=4$$) equations turned out to be much
harder. Rafael Bombelli made a crucial step in 1572 when - in a
desparate move - he invented complex numbers to solve cubic equations,
which had been unsolvable up to that time. Had he known matrices and
linear algebra, the invention of “complex numbers” would have been
unnecessary.

This is so, because there is a deep connection between polynominal
equations and matrices. Actually it turns out that **any
polynomial with degree $$n$$ is the characteristic polynomial of some
[companion matrix](https://en.wikipedia.org/wiki/Companion_matrix)
of order $$n$$**. So the problem of solving a polynominal equation is
equivalent to solving the characteristic equation of the companion
matrix. But solving the characteristic equation of a matrix means
computing the eigenvalues of that matrix. And computing eigenvalues has
a simple geometric meaning: They give the factor by which an eigenvector
(a vector which direction is left unchanged by the matrix
transformation) is stretched by the matrix transformation. The
eigenvalues are the scale factors of the linear transformation
represented by the matrix. This means that solving a polynomial equation
is equivalent to computing the scale factors of a corresponding linear
transformation.

Knowning this we can nowadays understand, why some polynominal equations
have no solutions. These are the equations, which correspond to
matrices, which don’t have eigenvalues, meaning the transformation
doesn’t leave any vector unchanged. From linear algebra we know, that
these transformations describe rotations. And this is the connection to
the 2d rotation matrix we found in the previous chapter. The polynominal
equation

$$\begin{aligned}
x^2 = -1\end{aligned}$$

has no solution, because it corresponds to a matrix describing a 90
degree rotation.

On the other side the
[fundamental
theorem of algebra](https://en.wikipedia.org/wiki/Fundamental_theorem_of_algebra) is easy to understand, when thinking of polynomials
as being represented by matrices. The number of solutions to a
polynomial of degree $$n$$ is the same as the number of eigenvalues of
the corresponding companion matrix.

### Complex Numbers

Complex numbers are not numbers. They cannot be ordered according to their size. This basic insight makes clear that trying to work with complex numbers like with usual “real numbers” must fail (e.g. division doesn’t work) and in general is also the reason for the big confusion around them.

But complex numbers are also no vectors (in the geometrical sense). The multiplication rule for complex numbers is completely different from the usual scalar product of geometrical vectors. Multiplying two complex numbers yields another complex number, whereas the usual scalar multiplication of geometrical vectors yields a scalar. So although a complex number can be represented as a set of two numbers, this set of two numbers should not(!) be visualized as geometrical vector.

Instead the modern view is that complex numbers are 2D matrices. They represent the group of (antisymmetric) 2D rotation matrices (<https://en.wikipedia.org/wiki/Complex_number#Matrix_representation_of_complex_numbers>). All the features of complex numbers follow naturally from this representation, e.g. multiplying two matrices yields another matrix. Unfortunately most textbooks don’t even mention the matrix representation of complex numbers, although this really makes clear what complex “numbers” are, how they can be extended (e.g. quaternions are 3D rotation matrices) and how they fit into the bigger picture which is <https://en.wikipedia.org/wiki/Group_theory>.

### Tensor calculus

#### Levi-Civita-Symbol

The symbol $$\epsilon_{ijkl\ldots}$$ is called Levi-Civita symbol and defined as
follows:

$$\begin{aligned}
\epsilon_{ijkl\ldots}=
\begin{cases} 	 1 & \text{if $i,j,k,l\ldots$ is an even permutation}, \\
					-1 & \text{if $i,j,k,l\ldots$ is an odd permutation}, \\
					 0 & \text{otherwise (two or more labels are the same)}.
\end{cases}				  \end{aligned}$$

Therefore the Levi-Civita-Symbol will change its sign, if two labels are
exchanged

$$\begin{aligned}
\epsilon_{ijkl\ldots u \ldots v} = -\epsilon_{ijkl\ldots v \ldots u}.\end{aligned}$$

The Levi-Civita-Symbol is not a tensor, but a pseudotensor, because it
transforms like a tensor under rotation, but not under reflection
(Pope 2000).[^1]

##### Levi-Civita-Symbol in 3D

In 3D only 6 of the 27 components of the Levi-Civita-Symbolare are unequal zero

$$\begin{aligned}
\epsilon_{123}=\epsilon_{312}=\epsilon_{231}&=1 \\
\epsilon_{321}=\epsilon_{132}=\epsilon_{213}&=-1\end{aligned}$$

The Levi-Civita-Symbol in 3D is most often used to express components
of a cross product of vectors in cartesian tensor notation

$$\begin{aligned}
 \left[ \boldsymbol{u} \times \boldsymbol{v} \right] _i = \epsilon_{ijk} u_j v_k
=&\ \epsilon_{i11} u_1 v_1 + \epsilon_{i12} u_1 v_2 +\epsilon_{i13} u_1 v_3\\
&+\epsilon_{i21} u_2 v_1 + \epsilon_{i22} u_2 v_2 + \epsilon_{i23} u_2 v_3\\
&+\epsilon_{i31} u_3 v_1 + \epsilon_{i32} u_3 v_2 + \epsilon_{i32} u_3 v_3\\
=&\ \delta_{i3} u_1 v_2 -\delta_{i2} u_1 v_3 -\delta_{i3} u_2 v_1\\
&+\delta_{i1} u_2 v_3 + \delta_{i2} u_3 v_1 - \delta_{i1} u_3 v_2\\
=&\ \delta_{i1} \left( u_2 v_3-u_3 v_2 \right) \\
&+\delta_{i2} \left( u_3 v_1-u_1 v_3 \right) \\
&+\delta_{i3} \left( u_1 v_2-u_2 v_1 \right) \end{aligned}$$

or the components of the curl of a vector field

$$\begin{aligned}
 \left[ \nabla \times \boldsymbol{v} \right] _i = \epsilon_{ijk} \frac{\partial}{\partial r_j} v_k\end{aligned}$$

To express double cross product other more complicated expressions we need
the following important relation between the Kronecker Delta and the
Levi-Civita-Symbol

$$\begin{aligned}
\begin{split}
\epsilon_{ijk}\epsilon_{lmn}=&
\begin{vmatrix}
  \delta_{il} & \delta_{im} & \delta_{in} \\
  \delta_{jl} & \delta_{jm} & \delta_{jn} \\
  \delta_{kl} & \delta_{km} & \delta_{kn}
\end{vmatrix}
\\
=&\ \delta_{il}\delta_{jm}\delta_{kn}
+\delta_{im}\delta_{jn}\delta_{kl}
+\delta_{in}\delta_{jl}\delta_{km}\\
&-\delta_{in}\delta_{jm}\delta_{kl}
-\delta_{il}\delta_{jn}\delta_{km}
-\delta_{im}\delta_{jl}\delta_{kn}
\end{split}\end{aligned}$$

From this relation we can derive the following

$$\begin{aligned}
\begin{split}
\epsilon_{ijk}\epsilon_{imn}
=&\ \delta_{ii}\delta_{jm}\delta_{kn}
+\delta_{im}\delta_{jn}\delta_{ki}
+\delta_{in}\delta_{ji}\delta_{km}\\
&-\delta_{in}\delta_{jm}\delta_{ki}
-\delta_{ii}\delta_{jn}\delta_{km}
-\delta_{im}\delta_{ji}\delta_{kn}\\
=&\ 3\delta_{jm}\delta_{kn}
+\delta_{km}\delta_{jn}
+\delta_{jn}\delta_{km}\\
&-\delta_{kn}\delta_{jm}
-3\delta_{jn}\delta_{km}
-\delta_{jm}\delta_{kn}\\
=&\ \delta_{jm}\delta_{kn}-\delta_{jn}\delta_{km}
\end{split}
\\
\begin{split}
\epsilon_{ijk}\epsilon_{ijn}
=&\ \delta_{jj}\delta_{kn}-\delta_{jn}\delta_{kj}\\
=&\ 3\delta_{kn}-\delta_{kn}\\
=&\ 2\delta_{kn}
\end{split}
\\
\begin{split}
\epsilon_{ijk}\epsilon_{ijk}
=&\ 2\delta_{kk}\\
=&\ 6
\end{split}\end{aligned}$$

#### Properties of second order tensors

A second order tensor can be decomposed into a symmetric and an antisymmetric
part in the following way[^2]

$$\begin{aligned}
T_{ij}=\underbrace{\frac{1}{2} \left( T_{ij}+T_{ji} \right) }_{\text{symmetric}}
+\underbrace{\frac{1}{2} \left( T_{ij}-T_{ji} \right) }_{\text{antisymmetric}}\end{aligned}$$

It can also be decomposed into an isotropic and deviatoric part by subtracting
and adding the trace of the tensor like

$$\begin{aligned}
T_{ij}=\underbrace{\frac{1}{n}\delta_{ij} T_{kk}}_{\text{isotropic}}
+\underbrace{T_{ij}-\frac{1}{n}\delta_{ij} T_{kk}}_{\text{deviatoric, 
tracefree}}\end{aligned}$$

Combining these two relations yields the general decomposition

$$\begin{aligned}
\begin{split}
T_{ij}=&
\overbrace{\frac{1}{n}\delta_{ij} T_{kk}}^{\text{isotropic}}
+
\overbrace{
\underbrace{\frac{1}{2} \left( T_{ij}+T_{ji}-\frac{2}{n}\delta_{ij}
T_{kk} \right) }_{\text{symmetric, tracefree}}
+\frac{1}{2} \left( T_{ij}-T_{ji} \right) }^{\text{deviatoric, tracefree}}\\[-1em]
&\underbrace{\hphantom{\frac{1}{n}\delta_{ij}
T_{kk}+\frac{1}{2} \left( T_{ij}+T_{ji}-\frac{2}{n}\delta_{ij}
T_{kk} \right) }}_{\text{symmetric}}
\hphantom{+}
\underbrace{\hphantom{\frac{1}{2} \left( T_{ij}-T_{ji} \right) }}_{\text{
antisymmetric}}
\end{split}\end{aligned}$$

An interesting relation can be found when computing the contraction of a
unsymmetric tensor $$U_{ij} \neq U_{ji}$$ with a symmetric tensor
$$V_{ij}=V_{ji}$$

$$\begin{aligned}
U_{ij}V_{ij}
= \frac{1}{2} U_{ij}V_{ij} + \frac{1}{2} U_{ji}V_{ji} 
= \frac{1}{2} U_{ij}V_{ij} + \frac{1}{2} U_{ji}V_{ij}
= \frac{1}{2}  \left( U_{ij}+U_{ji} \right)  V_{ij}
\label{eq:uscontr}\end{aligned}$$

In analogy one finds for the contraction of an unsymmetric tensor $$U_{ij}$$ with an
antisymmetric tensor $$W_{ij}=-W_{ji}$$

$$\begin{aligned}
U_{ij}W_{ij} = \frac{1}{2}  \left( U_{ij}-U_{ji} \right)  W_{ij}
\label{eq:uascontr}\end{aligned}$$

### Vector calculus

#### Longitudinal and transversal projection of vectors

A vector $$\boldsymbol{a}$$ can be split into two parts: the longitudinal part $$\boldsymbol{a}_{\parallel}$$,
which is parallel to another vector $$\boldsymbol{b}$$ and the transversal part $$\boldsymbol{a}_{\perp}$$,
which is perpendicular to $$\boldsymbol{b}$$. The length of the longitudinal part $$a_{\parallel}$$
and the transversal part $$a_{\perp}$$ can be computed from geometry (see figure <a href="#fig:projection" data-reference-type="ref" data-reference="fig:projection">[fig:projection]</a>)

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

Substituting equation <a href="#eq:protrans" data-reference-type="eqref" data-reference="eq:protrans">[eq:protrans]</a> in equation <a href="#eq:protrans2" data-reference-type="eqref" data-reference="eq:protrans2">[eq:protrans2]</a> we get

$$\begin{aligned}
\frac{(\left\lvert\boldsymbol{a}\times\boldsymbol{b}\right\rvert)^2}{b^2}=a^2-\frac{(\boldsymbol{a}\cdot\boldsymbol{b})^2}{b^2},\end{aligned}$$

which leads us to the following expression for the square of the norm of the cross product

$$\begin{aligned}
\left\lvert\boldsymbol{a}\times\boldsymbol{b}\right\rvert^2 = (a b)^2 - (\boldsymbol{a}\cdot\boldsymbol{b})^2.\end{aligned}$$

This is again Lagrange’s identity (see <a href="#eq:lagrident" data-reference-type="eqref" data-reference="eq:lagrident">[eq:lagrident]</a>).

#### Vector identities

In this chapter we show the derivation of some vector quantities in cartesian
tensor notation.

##### $$(\boldsymbol{u}\cdot\nabla) \boldsymbol{v}$$ [vecid01]

For some arbitraty vectors $$u_i, v_i$$ we can write

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

##### $$(\boldsymbol{v}\cdot\nabla) \boldsymbol{v}$$ [boldsymbolvcdotnabla-boldsymbolv]

Inserting $$u_j=v_j$$ into equation <a href="#eq:vecid01" data-reference-type="eqref" data-reference="eq:vecid01">[eq:vecid01]</a> yields

$$\begin{aligned}
v_j \frac{\partial}{\partial r_j} v_i = v_j \frac{\partial}{\partial r_i} v_j 
- \epsilon_{ijm} v_j \epsilon_{mkl} \frac{\partial}{\partial r_k} v_l\end{aligned}$$

For $$v_j \frac{\partial}{\partial r_i} v_j$$ we can write

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

##### $$\nabla \times  \left( \boldsymbol{u} \times \boldsymbol{v} \right)$$ [nabla-times-left-boldsymbolu-times-boldsymbolv-right]

The $$i$$-th component of the rotation of a cross product of two vectors is

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

#### Jacobian determinant

##### Definition [jacobi]

If the cartesian coordinates are given as functions of general curvilinear
coordinates $$a_1, a_2, a_3$$, i.e.

$$\begin{aligned}
 x=x(a_1, a_2, a_3),&&  y=y(a_1, a_2, a_3),&& z=z(a_1, a_2, a_3),\end{aligned}$$

it follows for the infinitesimal volume element in curvilinear coordinates

$$\begin{aligned}
dV=\left\lvert J\right\rvert da_1 da_2 da_3.\end{aligned}$$

Thereby the Jacobian determinant $$J$$ is defined as

$$\begin{aligned}
J = \left\lvert\frac{\partial(x,y,z)}{\partial (a_1, a_2,a_3)}\right\rvert=
\begin{vmatrix} \frac{\partial x}{\partial a_1} & 
					 \frac{\partial x}{\partial a_2} &
					 \frac{\partial x}{\partial a_3} \\
					 \frac{\partial y}{\partial a_1} & 
					 \frac{\partial y}{\partial a_2} &
					 \frac{\partial y}{\partial a_3} \\
					 \frac{\partial z}{\partial a_1} & 
					 \frac{\partial z}{\partial a_2} &
					 \frac{\partial z}{\partial a_3} 
\end{vmatrix}=
\sum_{i,j,k} \epsilon_{ijk} \frac{\partial x}{\partial a_i} 
					 				 \frac{\partial y}{\partial a_j} 
					 				 \frac{\partial z}{\partial a_k}.\end{aligned}$$

##### Time derivative of Jacobian determinant [jacdt]

With $$v_i= \frac{dx_i}{dt}$$ it follows for the time derivative of $$J$$

$$\begin{aligned}
\frac{dJ}{dt}&=\frac{d}{dt} \sum_{i,j,k} \epsilon_{ijk} 
\frac{\partial x}{\partial a_i} \frac{\partial y}{\partial a_j} 
\frac{\partial z}{\partial a_k} \\
&=\sum_{i,j,k} \epsilon_{ijk} \left(
\frac{\partial v_x}{\partial a_i} 
\frac{\partial y}{\partial a_j} 
\frac{\partial z}{\partial a_k}+
\frac{\partial x}{\partial a_i} 
\frac{\partial v_y}{\partial a_j} 
\frac{\partial z}{\partial a_k}+
\frac{\partial x}{\partial a_i} 
\frac{\partial y}{\partial a_j} 
\frac{\partial v_z}{\partial a_k} \right).\end{aligned}$$

From $$\frac{\partial v_k}{\partial a_i}=\sum_l \frac{\partial v_k}{\partial x_l}
\frac{\partial x_l}{\partial a_i}$$ $$(k \in x,y,z)$$ we get:

$$\begin{aligned}
\frac{dJ}{dt}=\sum_{i,j,k,l} \epsilon_{ijk}
\left(
\frac{\partial v_x}{\partial x_l}
\frac{\partial x_l}{\partial a_i} 
\frac{\partial y}{\partial a_j} 
\frac{\partial z}{\partial a_k}+
\frac{\partial x}{\partial a_i}
\frac{\partial v_y}{\partial x_l}
\frac{\partial x_l}{\partial a_j} 
\frac{\partial z}{\partial a_k}+
\frac{\partial x}{\partial a_i} 
\frac{\partial y}{\partial a_j} 
\frac{\partial v_z}{\partial x_l}
\frac{\partial x_l}{\partial a_k} \right). \end{aligned}$$

Analysis of the first term in the bracket for

<div class="flalign*" markdown="1">

l&=1:
,  
l&=2:
\_
,  
l&=3:
.

</div>

So for symmetry reasons most of the contributions are zero when summed up. In we
end we see: For the first term there is only a non-vanishing contribution for
$$l=1$$, for the second term there is only one for $$l=2$$ and for the third term
there is only one for $$l=3$$. So we get

$$\begin{aligned}
\frac{dJ}{dt}=\sum_{i,j,k} \epsilon_{ijk} 
\left(\frac{\partial v_x}{\partial x} + \frac{\partial v_y}{\partial y} +
\frac{\partial v_z}{\partial z}\right) 
\frac{\partial x}{\partial a_i} 
\frac{\partial y}{\partial a_j} 
\frac{\partial z}{\partial a_k} = 
(\nabla \cdot \boldsymbol{v}) J.\end{aligned}$$

### Fourier transform

The continous one dimensional Fourier transform in $$k$$-space $$F(k)$$ of some
function in
$$x$$-space $$f(x)$$is defined like

$$\begin{aligned}
F(k)&=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}f(x) e^{-ikx} dx\ \ \text{(Fourier transform)}\end{aligned}$$

Using the Fourier transform on a function twice will produce the
the original function again, but mirrored at the origin.
That’s why one conventionally defines an inverse Fourier transform
[^3],
that will generate the not mirrored original function again, when used on the
Fourier transform of a function

$$\begin{aligned}
f(x)&=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}F(k) e^{ikx} dx\ \ \text{(inverse Fourier transform)}\end{aligned}$$

In three dimensions one defines the Fourier transform like

$$\begin{aligned}
F(\boldsymbol{k})=\frac{1}{(2\pi)^{3/2}}\iiint^{\infty}_{-\infty}f(\boldsymbol{x}) e^{-i\boldsymbol{k}\boldsymbol{x}} dV \\
f(\boldsymbol{x})=\frac{1}{(2\pi)^{3/2}}\iiint^{\infty}_{-\infty}F(\boldsymbol{k}) e^{i\boldsymbol{k}\boldsymbol{x}} dK\end{aligned}$$

In cartesian coordinates the kernel of the Fourier transform
$$e^{-i\boldsymbol{k}\boldsymbol{x}}=e^{-i(k_xx+k_yy+k_zz)}$$
separates and so the three dimensional Fourier transform of a function
which separates in cartesian coordinates $$f(\boldsymbol{x})=a(x)b(y)c(z)$$ is
also separable

$$\begin{aligned}
F(\boldsymbol{k})=A(k_x)B(k_y)C(k_z)=
\frac{1}{(2\pi)^{3/2}}\int_{-\infty}^{\infty}a(x) e^{-ik_xx} dx 
\int_{-\infty}^{\infty}b(y) e^{-ik_yy} dy
\int_{-\infty}^{\infty}c(z) e^{-ik_zz} dz\end{aligned}$$

Thats why we like to use cartesian coordinates when we are using Fourier
transforms.

#### Fourier transform of a delta function

An important result can be derived by computing the inverse Fourier
transform of the Fourier transform of a delta function

$$\begin{aligned}
\delta(x-x_0) &= \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}\delta(x-x_0) e^{-ikx} dx e^{ikx} dk\\
&=\frac{1}{2\pi}\int_{-\infty}^{\infty}e^{-ikx_0} e^{ikx} dk = \frac{1}{2\pi}\int_{-\infty}^{\infty}e^{ik(x-x_0)} dk.\end{aligned}$$

From this we get, that the inverse Fourier transform of a
constant is the delta function

$$\begin{aligned}
\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}e^{ik(x-x_0)} dk = \sqrt{2\pi} \delta(x-x_0).\end{aligned}$$

Taking the complex conjugate of this equation and making use of the fact
that $$\delta^*(x-x_0) = \delta(x-x_0)$$ we get as definition for the delta
function

$$\begin{aligned}
\delta(x-x_0)= \frac{1}{2\pi}\int_{-\infty}^{\infty}e^{\pm ik(x-x_0)} dk\end{aligned}$$

Using this we can derive the astonishing result

$$\begin{aligned}
\int_{-\infty}^{\infty}f(x) dx = \sqrt{2\pi} F(0)\end{aligned}$$

as can be seen from

$$\begin{aligned}
\int_{-\infty}^{\infty}f(x) dx &= \int_{-\infty}^{\infty}\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}F(k) e^{ikx} dk dx 
= \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}F(k) \underbrace{\int_{-\infty}^{\infty}e^{ikx}}_{2\pi \delta(k)} dx dk\\
&= \sqrt{2\pi} \int_{-\infty}^{\infty}F(k) \delta(k) dk = \sqrt{2\pi} F(0).\end{aligned}$$

#### Convolution theorem

The Fourier transform of the product of two function in $$k$$-space is

$$\begin{aligned}
&\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}F(k) G(k) e^{ikx} dk =\\
&= \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}f(x') e^{-ikx'} dx' \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}g(x'') e^{-ikx''} dx''
e^{ikx} dk\\
&=\frac{1}{(2\pi)^{3/2}}\iiint^{\infty}_{-\infty}f(x') e^{-ikx'} g(x'') e^{-ikx''} e^{ikx} dx' dx'' dk\\
&=\frac{1}{(2\pi)^{3/2}}\iint_{-\infty}^{\infty}f(x') g(x'') 
\underbrace{\int_{-\infty}^{\infty}e^{-ik(x'+x''-x)} dk}_{2\pi \delta(x''-(x-x'))} dx' dx'' \\
&=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}f(x') \int_{-\infty}^{\infty}g(x'') \delta(x''-(x-x')) dx'' dx' \\
&=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}f(x') g(x-x') dx' = h(x).\end{aligned}$$

The integral $$h(x)$$ is called convolution of the functions $$f(x)$$ and $$g(x)$$.
So the convolution theorem says that

$$\begin{aligned}
h(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}f(x') g(x-x') dx' = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}F(k) G(k) e^{ikx} dk.\end{aligned}$$

#### Autocorrelation and Wiener-Khinchin Theorem

The autocorrelation of a function is defined as[^4]

$$\begin{aligned}
h_{AC}(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}f^*(x') f(x+x') dx'\end{aligned}$$

The Wiener-Khinchin Theorem states, that

$$\begin{aligned}
\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}f^*(x') f(x+x') dx' = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}\left\lvert F(k)\right\rvert^2 e^{ikx} dk\end{aligned}$$

which can be proved in analogy to the convolution theorem

$$\begin{aligned}
&\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}f^*(x') f(x+x') dx' =\\
&=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}f^*(x') \int_{-\infty}^{\infty}f(x'') \delta(x''-(x+x')) dx'' dx'\\
&=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}f^*(x') \int_{-\infty}^{\infty}f(x'') \frac{1}{2\pi}\int_{-\infty}^{\infty}e^{-ik(x'+x''-x)} dk dx'' dx'\\
&=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}f^*(x') e^{ikx'} dx' \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}f(x'') e^{-ikx''} dx''
e^{ikx} dk\\
&=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}F^*(k) F(k) e^{ikx} dk = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}\left\lvert F(k)\right\rvert^2 e^{ikx} dk.\end{aligned}$$

A special case of the Wiener-Khichnin theorem is Parseval’s theorem

$$\begin{aligned}
\int_{-\infty}^{\infty}\left\lvert f(x)\right\rvert^2 dx = \int_{-\infty}^{\infty}\left\lvert F(k)\right\rvert^2 dk, \end{aligned}$$

which can be obtained from the Wiener-Khichnin theorem for $$x=0$$

$$\begin{aligned}
h_{AC}(0)&= \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}f^*(x') f(x') dx' = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}\left\lvert f(x)\right\rvert^2 dx \\
&= \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}\left\lvert F(k)\right\rvert^2 e^{ik0} dk = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}\left\lvert F(k)\right\rvert^2 dk\end{aligned}$$

### Basic probability theory

#### Definition of probability

An experiment can measure if an event $$A$$ happend or not. If we repeat
an experiment $$n$$ times and we measure that the event $$A$$ happened
$$m$$ times we define the probability that the event $$A$$ happens as

$$\begin{aligned}
P(A) = \lim\limits_{n \to \infty} \frac{m}{n}\end{aligned}$$

If event $$C$$ means that event $$A$$ *or* event $$B$$ can happen
we write $$C = A \cup B$$. If event $$D$$ means that event $$A$$
*and* event $$B$$ happen, we write $$D = A \cap B$$. Classical
probability theory is then based on the following three axioms (called
the Kolmogorov axioms):

1.  Every event $$A$$ has a real non-negative probability $$P(A) \ge 0$$.

2.  The probability that any event from the event space will happen is
    one:

    $$P(A \cup A^c) = 1$$ (where $$A^c$$ is the complement event to
    $$A$$ in the event space)

3.  The probabilities of mutually exclusive events ( $$P(A \cap B) = 0$$ )
    add:

    $$P(A \cup B) = P(A) + P(B)$$

From the last axiom it also follows that in general

$$\begin{aligned}
P(A \cup B) = P(A) + P(B) - P(A \cap B)\end{aligned}$$

Although these axioms seem unavoidable it should be mentioned that
quantum probability theory violates axiom 1 and axiom 3. Axiom 3 is
violated, because measurement of events in quantum mechanics (QM) are
not commutative, meaning the measurement of event A often must influence
the measurement of event B. Axiom 1 is violated since QM must describe
interference effects between events and does this by introducing
[negative
probabilities](https://en.wikipedia.org/wiki/Negative_probability) (To be more precise, the probability wave function of QM
is complex, because in the theory one must basically take the square
root of the negative probabilities). But as Dirac put it: “Negative
probabilities should not be considered nonsense. They are well defined
concepts mathematically, like a negative sum of money. Negative
probabilities should be considered simply as things which do not appear
in experimental results.”

So it is possible to work with things like negative or complex
probabilities. But to be able to derive the central limit theorem it is
necessary that the three axioms of Kolmogorov for classical probability
hold.

#### Random variables

Examples for random variables are e.g. the number on a thrown dice, the
lifetime of a instable radioactive nucleus, the amplitude of
athmospheric noise recorded by normal radio. In general a random
variable $$X$$ takes finite real values $$x$$ where the probability that
$$X$$ takes the value $$x$$ in a given intervall from $$a$$ to $$b$$
depends on the event described by $$X$$. We write

$$\begin{aligned}
P(a < X < b) = \int_a^b f_X(x) dx \end{aligned}$$

where $$f_X(x)$$ is the so called probability density function
characteristic for the event. We have to be careful to distinguish the
random variable $$X$$ from its value $$x$$ it takes after the
measurement. If $$a < x < b$$ then the probability $$P(a < x < b)$$ is
always one, because $$x$$ is just a number between $$a$$ and $$b$$. But
the value of $$P(a < X < b)$$ depends on the form of the probability
density function (Note however that $$P(-\infty < x < \infty)$$ =
$$P(-\infty < X < \infty)$$ according to the second axiom). So a random
variable is - despite its name - actually not a number or value, it is
[a set of
possible values from a random experiment](https://www.mathsisfun.com/data/random-variables.html), where each value has a
probability associated with it. To describe a experiment of throwing a
dice one could write

$$\begin{aligned}
X = \{(1,1/6),(2,1/6),(3,1/6),(4,1/6),(5,1/6),(6,1/6)\}\end{aligned}$$

where the first value of each tuple is the possible outcome $$x_i$$
(called [Random
Variate](https://en.wikipedia.org/wiki/Random_variate)), the second is the corresponding probability $$p_i$$. The
probability is

$$\begin{aligned}
P(a < X < b) = \sum_{a < x_i < b} p_i  \end{aligned}$$

or in case that the possible outcomes are a continous set $$x(t)$$ with
the corresponding
[probability
density function](https://en.wikipedia.org/wiki/Probability_density_function) $$p(t)$$

$$\begin{aligned}
P(a < X < b) = \int_a^b p(t) dt  \end{aligned}$$

In the special case $$a=-\infty$$ the probability $$P$$ only depends on
the upper limit $$b$$

$$\begin{aligned}
P(-\infty < X < b) = P(X < b) = \int_{-\infty}^b p(t) dt = F(b)  \end{aligned}$$

and we call $$F(b)$$ the
[cumulative
distribution function](https://en.wikipedia.org/wiki/Cumulative_distribution_function).

It should be noted that a possible outcome or random variate $$x_i$$ is
in principle a functional $$x[p(t)]$$ over the space of probability
density functions. This can be seen in the way how pseudo random numbers
are generated on a computer. One usually has a
[random
number generator](https://en.wikipedia.org/wiki/Pseudorandom_number_generator) (often also called deterministic random number
generator, because starting from the same seed, it will generate the
same series of random numbers) generating uniformly distributed numbers
between 0 and 1. These uniformly distributed numbers are then
transformed to non-uniform random numbers by methods like
[inverse
transform sampling](https://en.wikipedia.org/wiki/Inverse_transform_sampling) or
[rejection
sampling](https://en.wikipedia.org/wiki/Rejection_sampling), which basically make use of the probability density function
$$p(t)$$. So the scalar real value taken by a random variable generated
by a computer depends on the form of a function, making $$x[p(t)]$$ a
functional and the random variable $$X$$ a set of functionals.

#### Multiple random variables

Assume you throw one red and one blue dice in an experiment. The number
on the red dice would be random variable $$X$$, the number of the blue
dice random variable $$Y$$. Under normal circumstances the number on
each dice would be independent of each other. We say, the random
variables $$X$$ and $$Y$$ are uncorrelated. However, assume that the
dice are magnetic. In that case the number shown on each dice might not
be independent anymore.

To describe such an experiment where we measure a pair of random
variables we use a joint probability and a joint distribution function:

$$\begin{aligned}
P(a < X < b, c < Y < d) = \int_a^b\int_c^d f_{XY}(x,y) dx dy \end{aligned}$$

In analogy we can also describe experiments with $$n$$ random
variables by using a $$n$$-dimensional joint distribution function (In
the limit that $$n$$ would approach an uncountable infinity the
probability would be expressed by an infinite dimensional integral, see
also
[Functional
integration](https://en.wikipedia.org/wiki/Functional_integration))

Quantum mechanics = asymmetric covariance matrix??

### More topics

#### Linear algebra

-   <https://physics.stackexchange.com/questions/35562/is-a-1d-vector-also-a-scalar>

-   <https://math.stackexchange.com/questions/219434/is-a-one-by-one-matrix-just-a-number-scalar>

-   <https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors#Eigenvalues_of_geometric_transformations>

##### Functions of matrices

-   <https://en.wikipedia.org/wiki/Matrix_function>

-   <https://en.wikipedia.org/wiki/Matrix_exponential>

-   <https://en.wikipedia.org/wiki/Logarithm_of_a_matrix>

-   <https://math.stackexchange.com/questions/1149598/how-to-solve-a-non-linear-matrix-equation-over-integer-numbers>

#### Transformations and groups

-   Prove of eulers formula as a solution to 2d wave equation: <http://math.stackexchange.com/a/3512/27609>

-   <https://en.wikipedia.org/wiki/Linear_canonical_transformation>

-   <https://en.wikipedia.org/wiki/Hartley_transform>

-   <https://en.wikipedia.org/wiki/Split-complex_number>

-   <https://en.wikipedia.org/wiki/Dual_number> (<https://math.stackexchange.com/questions/1120720/are-dual-numbers-a-special-case-of-grassmann-number>)

-   <https://en.wikipedia.org/wiki/Grassmann_number> [Grassmann vectors](https://math.stackexchange.com/questions/1108045/relationship-between-levi-civita-symbol-and-grassmann-numbers)

-   <https://en.wikipedia.org/wiki/Quaternion> (<https://math.stackexchange.com/questions/147166/does-my-definition-of-double-complex-noncommutative-numbers-make-any-sense>)

-   <https://math.stackexchange.com/questions/2083950/relationship-between-levi-civita-symbol-and-complex-quaternionic-numbers>

#### Series

-   <http://blog.wolfram.com/2014/08/06/the-abcd-of-divergent-series>

    -   <http://physicsbuzz.physicscentral.com/2014/01/redux-does-1234-112-absolutely-not.html>

    -   <https://www.quora.com/Whats-the-intuition-behind-the-equation-1+2+3+-cdots-tfrac-1-12>

#### Calculus

##### Euler-MacLaurin

-   <https://people.csail.mit.edu/kuat/courses/euler-maclaurin.pdf>

-   <http://www.hep.caltech.edu/~phys199/lectures/lect5_6_ems.pdf>

-   <https://terrytao.wordpress.com/2010/04/10/the-euler-maclaurin-formula-bernoulli-numbers-the-zeta-function-and-real-variable-analytic-continuation>

##### Watsons Triple Integrals

-   <http://mathworld.wolfram.com/WatsonsTripleIntegrals.html>

-   <http://www.inp.nsk.su/~silagadz/Watson_Integral.pdf>

##### Generalized Calculus

-   <https://en.wikipedia.org/wiki/Product_integral>

-   <http://math2.org/math/paper/preface.htm>

-   <http://www.gauge-institute.org/calculus/PowerMeansCalculus.pdf>

##### Finite calculus

-   <https://www.cs.purdue.edu/homes/dgleich/publications/Gleich%202005%20-%20finite%20calculus.pdf>

-   <https://en.wikipedia.org/wiki/Concrete_Mathematics>

##### Iterative roots and fractional iteration

-   <http://reglos.de/lars/ffx.html>

-   <https://mathoverflow.net/questions/17605/how-to-solve-ffx-cosx>

#### Geometry

-   [Cutting a cube along the diagonal](https://www.friedrich-verlag.de/fileadmin/redaktion/sekundarstufe/Mathematik/Der_Mathematikunterricht/Leseproben/Der_Mathematikunterricht_3_13_Leseprobe_2.pdf)

-   <https://en.wikipedia.org/wiki/Visual_calculus>

#### Weird constants and functions

##### Euler-Mascheroni constant

-   <https://en.wikipedia.org/wiki/Euler%E2%80%93Mascheroni_constant#Generalizations>

##### Universal Parabolic constant

-   <https://en.wikipedia.org/wiki/Universal_parabolic_constant>

-   <http://mathworld.wolfram.com/UniversalParabolicConstant.html>

-   <https://mathoverflow.net/questions/37871/is-it-a-coincidence-that-the-universal-parabolic-constant-shows-up-in-the-soluti>

##### Apery’s constant

-   <https://en.wikipedia.org/wiki/Ap%C3%A9ry%27s_constant>

-   <https://math.stackexchange.com/questions/12815/riemann-zeta-function-at-odd-positive-integers/12819#12819>

##### Gauss’s constant

-   <https://en.wikipedia.org/wiki/Gauss%27s_constant>

-   <https://en.wikipedia.org/wiki/Lemniscatic_elliptic_function>

-   <https://en.wikipedia.org/wiki/Particular_values_of_the_Gamma_function>

##### Riemann Zeta function

-   <https://math.stackexchange.com/questions/1792755/connection-between-the-area-of-a-n-sphere-and-the-riemann-zeta-function>

-   <https://suryatejag.wordpress.com/2011/11/24/riemann-functional-equation-and-hamburgers-theorem>

#### Probability

-   <https://en.wikipedia.org/wiki/Secretary_problem>

-   <https://en.wikipedia.org/wiki/Kelly_criterion>

##### Sample size

-   <https://stats.stackexchange.com/questions/192199/derivation-of-formula-for-sample-size-of-finite-population/192601#192601>

-   <https://math.stackexchange.com/questions/926478/how-does-accuracy-of-a-survey-depend-on-sample-size-and-population-size/1357604#1357604>

-   <https://onlinecourses.science.psu.edu/stat414/node/264>

-   <http://www.surveysystem.com/sscalc.htm>

-   <http://research-advisors.com/tools/SampleSize.htm>

# Physics

## Classical mechanics

In the view of classical mechanics the world consists of point particles with constant mass $$m$$,
position $$\boldsymbol{x(t)}$$ and velocity $$\boldsymbol{v(t)}$$. The number of particles is countable
(not an uncountable infinity) and stays constant. The particles don’t split or unite.
The time is a global parameter which is the same for each particle. This description
of the world is sometimes also called point mechanics.

It is amazing how many phenomenoms one can describe with this simple model of the world.
On the other side this short description already hints to the several limitations of this model.
The standard model of physics nowadays based on quantum field theory basically abandons
all the assumptions made by classical mechanics.

## Fluid dynamics

### Introduction

Fluids
[^5]
in fluid dynamics are treated as continuous fields. Each point of the
field represents a fluid element consisting of several point particles.
[^6]
The statistical behaviour of these point particles
defines quantities like density, temperature, pressure and average velocity
for each fluid element. In this sense an fluid can be represented by a number of
contiuous fields (density, velocity, pressure, temperature,...)
which defines at every point a certain statistical quantity of a bunch
of fluid molecules.

#### Substantial derivative

To compute the change with time of a scalar quantity $$A(x,y,z,t)$$ at a fixed
point in
space $$(x,y,z)$$, we get

$$\begin{aligned}
\frac{d}{d t} A=\frac{\partial}{\partial t} A.\end{aligned}$$

However in fluid dynamics we are often interested in the temporal change of a
quantity in a certain local fluid element, which moves with the fluid. This
means
$$A=A(x(t),y(t),z(t),t)$$ and therefore we get for the change

$$\begin{aligned}
\frac{dA}{dt}=\frac{dA}{dx}\frac{dx}{dt}+\frac{dA}{dy}\frac{dy}{dt}+
\frac{dA}{dz}\frac{dz}{dt}+\frac{\partial A}{\partial t}.\end{aligned}$$

So if in general we want to express the total derivative $$\frac{d}{dt}$$ by
quantities at fixed space points, we can make use of the so called substantial
derivative

$$\begin{aligned}
\frac{d}{d t}=\frac{\partial}{\partial t} + v_j \frac{\partial}{\partial r_j}. \label{eq:1}\end{aligned}$$

#### Reynolds transport theorem

If $$A=\int_{V(t)} \alpha(\boldsymbol{r},t) dV$$ is a scalar quantity which is conserved
in a local fluid element moving with the fluid (and therefore having a time
dependent
volume) we can write

$$\begin{aligned}
\frac{d}{dt} A = \frac{d}{dt} \int_{V(t)} \alpha(\boldsymbol{r},t) dV = 0.\end{aligned}$$

But because the boundary of the integral is time dependent, we cannot exchange
integration with the time derivative. Therefore we have to ascribe the
integration over the time dependent volume $$V(t)$$ to the volume $$V_0$$ at time
$$t=0$$. The transformation of the volume element $$dV_0$$ (at time $$t=0$$) to the
the volume element $$dV_0$$ can be described by

$$\begin{aligned}
dV=J dV_0 &&\text{mit } J= \left\lvert\frac{\partial(x,y,z)}{\partial (x_0,
y_0,z_0)}\right\rvert,\end{aligned}$$

where $$J$$ is called Jacobian determinant [^7]. It describes the change of the fluid element if it is
transported with the fluid. So we can express $$\frac{d}{dt} A$$ by

$$\begin{aligned}
\frac{d}{dt} A = \int_{V_0} \frac{d}{dt}(\alpha(\boldsymbol{r},t) J) dV_0
= \int_{V_0} \left(J \frac{d\alpha}{dt}+\alpha \frac{dJ}{dt}\right) dV_0.\end{aligned}$$

Using the substantial derivative and $$\frac{dJ}{dt}=J \frac{\partial}{\partial r_j} v_j$$
[^8] leads to

$$\begin{aligned}
\frac{d}{d t} A = \int_{V_0} \left[\frac{\partial \alpha}{\partial t} + 
(v_j \frac{\partial}{\partial r_j}) \alpha + \alpha (\frac{\partial}{\partial r_j} v_j)\right] J  dV_0.\end{aligned}$$

From this we get the Reynolds transport theorem

$$\begin{aligned}
\frac{d}{d t} A = \int_{V(t)} \left[\frac{\partial}{\partial t} \alpha + 
\frac{\partial}{\partial r_j} (v_j \alpha)\right]  dV. \label{eq:RT}\end{aligned}$$

Because it is valid for arbitrary volumes we can also write it as a generalised
continuity equation

$$\begin{aligned}
\frac{\partial}{\partial t}(\alpha)+ \frac{\partial}{\partial r_j} (v_j \alpha) = 0.
\label{eq:RTdiff}\end{aligned}$$

### General compressible fluid

#### Balance equations

##### Mass equation

The mass inside a local fluid element $$M=\int_V \rho(r_i) dV$$ is conserved.
Therefore we get the balance equation for the mass from the generalised
continuity equation setting $$\alpha=\rho$$ in the form

$$\begin{aligned}
\frac{\partial}{\partial t}\rho + \frac{\partial}{\partial r_j}(v_j \rho) = 0. \label{eq:3}\end{aligned}$$

##### Momentum equation

The momentum $$P_i$$ inside a local fluid element is $$P_i=\int_V \rho(r_i) v_i
dV$$. The change of the momentum with time $$\frac{d}{dt} P_i$$ is equal to the sum
of the forces on the fluid element

$$\begin{aligned}
\frac{d}{d t} P_i=F_i=F_{p,i}+F_{visc,i}+F_{g,i}.\end{aligned}$$

Here we have restricted ourself to a viscous, selfgravitating fluid, so the sum
of forces on each fluid element consists of

-   the thermodynamic pressure on the surface $$A$$ of the fluid element

    $$\begin{aligned}
    F_{p,i}= - \oint_A p n_i dA = - \int_V \frac{\partial}{\partial r_i} p dV,\end{aligned}$$

-   the viscous force meaning the irreversible transfer of momentum due to
    friction between the surfaces of the fluid elements

    $$\begin{aligned}
    F_{visc,i}= \oint_A \sigma'_{ij} n_j dA = \int_V \frac{\partial}{\partial r_j} \sigma'_{ij} dV,\end{aligned}$$

-   the gravitational force, where for a selfgravitationg fluid $$g_i$$ is
    generated by the fluid itself (not only by the local fluid element, because
    gravity is a long-range force)

    $$\begin{aligned}
    F_{g,i}= \int_V \rho g_i dV.\end{aligned}$$

Using the generalized continuity equation <a href="#eq:RTdiff" data-reference-type="eqref" data-reference="eq:RTdiff">[eq:RTdiff]</a> we can express the
temporal change of each component of the momentum of a fluid element
setting $$\alpha=\rho v_i$$ in the form

$$\begin{aligned}
\frac{\partial}{\partial t}(\rho v_i) + \frac{\partial}{\partial r_j}(v_j \rho v_i) = -\frac{\partial}{\partial r_i}p + \frac{\partial}{\partial r_j}\sigma'_{ij}
+\rho g_i.\end{aligned}$$

With the help of the continuity equation <a href="#eq:3" data-reference-type="eqref" data-reference="eq:3">[eq:3]</a> we can write the momentum
equation in the often used form called Euler equation

$$\begin{aligned}
\frac{\partial}{\partial t}(v_i) + v_j \frac{\partial}{\partial r_j}( v_i) = -\frac{1}{\rho}\frac{\partial}{\partial r_i}p +
\frac{1}{\rho}\frac{\partial}{\partial r_j}\sigma'_{ij} + g_i. \label{eq:vel}\end{aligned}$$

##### Kinetic energy equation

If we multiply equation <a href="#eq:vel" data-reference-type="eqref" data-reference="eq:vel">[eq:vel]</a> with the velocity $$v_i$$, use
$$\frac{1}{2}\frac{d x^2}{dt} = x \frac{dx}{dt}$$ and the continuity
equation we get an equation for the kinetic energy of a local fluid element in
conservation form

$$\begin{aligned}
\frac{\partial}{\partial t}(\frac{1}{2}\rho v^2) + \frac{\partial}{\partial r_j}(v_j\frac{1}{2}\rho v^2) =
-v_i\frac{\partial}{\partial r_i}p + v_i\frac{\partial}{\partial r_j}\sigma'_{ij}+v_i\rho g_i\end{aligned}$$

This equation shows us that locally the kinetic energy is not conserved
(otherwise the right-hand side of the equation should be zero).

##### Internal energy equation

If we assume that each fluid element is in thermal equilibrium the first law of
thermodynamics does hold locally and we can write for the internal
energy of a fluid element[^9]

$$\begin{aligned}
E_{int} = \int dE_{int} = \int T dS - \int p dV = \int \rho T s dV - \int p dV\end{aligned}$$

with $$s=\frac{dS}{dm}$$ and $$\rho=\frac{dm}{dV}$$. Because $$T$$ and $$p$$ are
understood as the average temperature and pressure in the fluid element, they can
be moved out of the integral so that

$$\begin{aligned}
E_{int} = \int \rho e_{int} dV = T \int \rho s dV - p \int dV.\end{aligned}$$

with $$e_{int}=\frac{dE_{int}}{dm}$$.
If we take the time derivative of the internal energy we get

$$\begin{aligned}
\begin{split}
\frac{d}{d t} \int \rho e_{int} dV &= \frac{d}{d t} \left( T \int \rho s dV \right) -\frac{d}{d t} \left( p \int
dV \right)  \\
&= T \frac{d}{d t} \int \rho s dV - p \frac{d}{d t} \int dV + \int \rho s dV \frac{d}{d t}T 
- \int dV \frac{d}{d t} p
\end{split}\end{aligned}$$

Because we assume local thermodynamic equilibrium $$\frac{d}{d t} T=0$$ and $$\frac{d}{d t} p=0$$
and the last two terms on the right hand side vanish. The other terms can be
computed by using the Reynolds transport theorem and we get the balance
equation for the internal energy of a fluid element

$$\begin{aligned}
\frac{\partial}{\partial t} \rho e_{int} + \frac{\partial}{\partial r_j} v_j \rho e_{int} = T  \left( \frac{\partial}{\partial t} \rho s +
\frac{\partial}{\partial r_j} v_j \rho s \right)  -p \frac{\partial}{\partial r_j} v_j \label{eq:eint}\end{aligned}$$

##### Global dissipation of kinetic energy

For investigating the conservation of the kinetic energy of the
whole fluid we write using the Reynolds transport theorem <a href="#eq:RT" data-reference-type="eqref" data-reference="eq:RT">[eq:RT]</a>

$$\begin{aligned}
\frac{d}{d t} E_{kin} = \frac{d}{d t} \int_V \frac{1}{2}\rho v^2 dV = 
\int_V \frac{\partial}{\partial t}  \left( \frac{1}{2}\rho v^2 \right)  + 
\frac{\partial}{\partial r_j}  \left( v_j \frac{1}{2}\rho v^2 \right)  dV\end{aligned}$$

The second term on the right hand side can be transformed with Gauss’s theorem
to an integral over the surface of the whole fluid

$$\begin{aligned}
\int_V \frac{\partial}{\partial r_j}  \left( v_j \frac{1}{2}\rho v^2 \right)  dV = 
\oint_A v_j \frac{1}{2}\rho v^2 dA = 0.\end{aligned}$$

The surface integral is zero because the velocity on the boundary of the fluid
$$v_j = 0$$. Therefore we can write

$$\begin{aligned}
\frac{d}{d t} E_{kin}=\frac{\partial}{\partial t} E_{kin}=\int_V \frac{\partial}{\partial t}  \left( \frac{1}{2}\rho v^2 \right)  dV =
\int_V \rho v_i \frac{\partial}{\partial t} v_i + \frac{1}{2} v^2 \frac{\partial}{\partial t}\rho dV \end{aligned}$$

Inserting the Euler equation <a href="#eq:vel" data-reference-type="eqref" data-reference="eq:vel">[eq:vel]</a> and using the continuity equation
<a href="#eq:3" data-reference-type="eqref" data-reference="eq:3">[eq:3]</a> and $$g_i=\frac{\partial}{\partial r_i}\phi$$ we can transform this to

$$\begin{aligned}
\begin{split}
\frac{\partial}{\partial t} E_{kin} =& - \int_V \frac{\partial}{\partial r_j} \left[ v_j \rho
 \left( \frac{1}{2}v^2+\frac{p}{\rho}+\phi \right)  + v_i \sigma'_{ij} \right]  dV\\ 
&+ \int_V p \frac{\partial}{\partial r_j} v_j dV - \int_V \sigma'_{ij} \frac{\partial}{\partial r_j} v_i dV
- \int_V \phi \frac{\partial}{\partial t}\rho dV.
\end{split}\end{aligned}$$

The first term on the right hand side can be transformed with Gauss’s theorem
to a surface integral. Again this surface integral is zero because on the
surface of the fluid the velocity $$v_i,v_j=0$$. So we are left with

$$\begin{aligned}
\frac{\partial}{\partial t} E_{kin} = \int_V p \frac{\partial}{\partial r_j} v_j dV - \int_V \sigma'_{ij}
\frac{\partial}{\partial r_j} v_i dV
- \int_V \phi \frac{\partial}{\partial t}\rho dV,\end{aligned}$$

which is the generalization of equation (16.2) from Landau and Lifschitz (1991)
for general compressible fluids. So we see that the total kinetic energy for an
ideal, compressible fluid is not conserved.

If we substitute the first term on the right hand side with the balance equation
for internal energy <a href="#eq:eint" data-reference-type="eqref" data-reference="eq:eint">[eq:eint]</a> and again makes use of Gauss’s theorem we
get the following expression

$$\begin{aligned}
\frac{\partial}{\partial t} E_{kin} = \frac{\partial}{\partial t} \int_V \rho s dV -\frac{\partial}{\partial t} \int_V \rho e_{int} dV
- \frac{\partial}{\partial t} \int_V\rho \phi dV + \int_V \rho \frac{\partial}{\partial t} \phi dV 
- \int_V \sigma'_{ij} \frac{\partial}{\partial r_j} v_i dV \label{eq:glodis}\end{aligned}$$

One might be tempted to the following conclusion, that

$$\begin{aligned}
\frac{\partial}{\partial t} E_{tot} = 
\frac{\partial}{\partial t} E_{kin}+\frac{\partial}{\partial t} E_{int}+\frac{\partial}{\partial t} E_{pot} = 
T \frac{\partial}{\partial t} S - \int_V \sigma'_{ij} \frac{\partial}{\partial r_j} v_i dV + \int_V \rho \frac{\partial}{\partial t} \phi dV \end{aligned}$$

If then one requires that the total energy should be constant, one would come
to the conclusion

$$\begin{aligned}
\frac{\partial}{\partial t} S = \frac{1}{T}\int_V \sigma'_{ij} \frac{\partial}{\partial r_j} v_i dV 
- \frac{1}{T} \int_V \rho \frac{\partial}{\partial t} \phi dV\end{aligned}$$

This would lead to the statement that the total entropy of an ideal fluid
($$\sigma'_{ij}=0$$) is not constant but dependent on the time derivative of the
potential $$\phi$$. But this is not true!

To get the right answer first notice that the potential energy of a
selfgravitating system is not $$\int \rho \phi dV$$ but $$\frac{1}{2}\int \rho
\phi dV$$. So in the statement above the definition of the total energy was
wrong. The potential of a certain density distribution is the solution of the
poisson equation

$$\begin{aligned}
\phi(x_i,t)=-G \int_V \frac{\rho(x_j,t)}{\left\lvert x_i-x_j\right\rvert} dV_j\end{aligned}$$

and therefore the potential energy of a selfgravitating system can be expressed
as a double integral

$$\begin{aligned}
E_{pot}=-\frac{G}{2}\iint \frac{\rho(x_i,t)\rho(x_j,t)}{\left\lvert x_i-x_j\right\rvert}dV_j dV_i.\end{aligned}$$

If we now take the time derivative of this expression for the potential energy
of a selfgravitating system

$$\begin{aligned}
\begin{split}
\frac{\partial}{\partial t}E_{pot}&=
-\frac{G}{2} \iint \rho_i \frac{\frac{\partial}{\partial t}\rho_j}{\left\lvert x_i-x_j\right\rvert}dV_j dV_i
-\frac{G}{2} \iint \frac{\partial}{\partial t}(\rho_i)\frac{\rho_j}{\left\lvert x_i-x_j\right\rvert}dV_j dV_i\\
&=
-\frac{G}{2} \iint \rho_i \frac{\frac{\partial}{\partial t}\rho_j}{\left\lvert x_i-x_j\right\rvert}dV_j dV_i
-\frac{G}{2} \iint \rho_j \frac{\frac{\partial}{\partial t}\rho_i}{\left\lvert x_i-x_j\right\rvert}dV_i dV_j\\
&=
-\frac{G}{2} \iint \rho_i \frac{\frac{\partial}{\partial t}\rho_j}{\left\lvert x_i-x_j\right\rvert}dV_j dV_i
-\frac{G}{2} \iint \rho_i \frac{\frac{\partial}{\partial t}\rho_j}{\left\lvert x_j-x_i\right\rvert}dV_j dV_i\\
&=
-G \iint \rho_i \frac{\frac{\partial}{\partial t}\rho_j}{\left\lvert x_i-x_j\right\rvert}dV_j dV_i \\
&=
\int \rho \frac{\partial}{\partial t} \phi dV
\end{split}\end{aligned}$$

Here we used the abbreviation $$\rho_i=\rho(x_i,t)$$ and $$\rho_j=\rho(x_j,t)$$.
[^10]

So we see that for a selfgravitating system

$$\begin{aligned}
\int \rho \frac{\partial}{\partial t} \phi dV = \frac{\partial}{\partial t}  \left( \frac{1}{2}\int \rho\phi dV \right) .\end{aligned}$$

If we use this expression in <a href="#eq:glodis" data-reference-type="eqref" data-reference="eq:glodis">[eq:glodis]</a> and use $$E_{pot}= \frac{1}{2}\int
\rho \phi dV$$ we get

$$\begin{aligned}
\frac{\partial}{\partial t} E_{tot} = 
\frac{\partial}{\partial t} E_{kin}+\frac{\partial}{\partial t} E_{int}+\frac{\partial}{\partial t} E_{pot} = 
T \frac{\partial}{\partial t} S - \int_V \sigma'_{ij} \frac{\partial}{\partial r_j} v_i dV\end{aligned}$$

Demanding that the total energy of the whole fluid should be conserved leads to
the right expression for the time evolution of the total entropy

$$\begin{aligned}
\frac{\partial}{\partial t} S = \frac{1}{T}\int_V \sigma'_{ij} \frac{\partial}{\partial r_j} v_i dV\end{aligned}$$

So total entropy is conserved for an ideal fluid, even if we take selfgravity
into account[^11].

##### Local dissipation of kinetic energy

For analyzing the dissipation of kinetic energy in a local fluid element we
have to make the same calculations as we did to get equation <a href="#eq:glodis" data-reference-type="eqref" data-reference="eq:glodis">[eq:glodis]</a>,
but without the assumption of $$v=0$$ on the boundary. Doing this we arrive at

$$\begin{aligned}
\begin{split}
\frac{d}{d t} E_{kin} =& \int_V \left[ \frac{\partial}{\partial r_j} \left( v_i \sigma'_{ij}-v_j p \right)  
+ T \left( \frac{\partial}{\partial t}\rho s + \frac{\partial}{\partial r_j} v_j \rho s \right)  
-  \left( \frac{\partial}{\partial t}\rho e_{int} + \frac{\partial}{\partial r_j} v_j \rho e_{int} \right)  \right. \\
&\left. - \left( \frac{\partial}{\partial t}\rho \phi + \frac{\partial}{\partial r_j} v_j \rho \phi \right) 
+ \rho\frac{\partial}{\partial t}\phi -\sigma'_{ij}\frac{\partial}{\partial r_j}v_i \right] dV.
\end{split}\end{aligned}$$

If we identify the second, third and fourth term on the right hand side with the
total time derivative of the entropy, the internal energy and the potential
energy respectively we get

$$\begin{aligned}
\frac{d}{d t} E_{kin} + \frac{d}{d t} E_{int} + \frac{d}{d t} E_{pot} = 
\int_V \frac{\partial}{\partial r_j} \left( v_i \sigma'_{ij}-v_j p \right)  + T \frac{d}{d t} S
+ \rho\frac{\partial}{\partial t}\phi -\sigma'_{ij}\frac{\partial}{\partial r_j}v_i dV.\end{aligned}$$

If we additionally assume that locally the same entropy equation holds as
globally

$$\begin{aligned}
\frac{d}{d t} S = \int_V \sigma'_{ij}\frac{\partial}{\partial r_j}v_i dV\end{aligned}$$

or in differential form

$$\begin{aligned}
\frac{\partial}{\partial t}\rho s + \frac{\partial}{\partial r_j} v_j \rho s = \sigma'_{ij}\frac{\partial}{\partial r_j}v_i,\end{aligned}$$

we are led to the following balance equation for the total energy
$$e_{tot}=e_k+e_{int}+\phi$$ for a local fluid element

$$\begin{aligned}
\frac{\partial}{\partial t}\rho e_{tot} + \frac{\partial}{\partial r_j} v_j \rho e_{tot} = -\frac{\partial}{\partial r_j} \left( v_j p \right) 
+\frac{\partial}{\partial r_j} \left( v_i \sigma'_{ij} \right)  + \rho \frac{\partial}{\partial t} \phi. \label{eq:etotal}\end{aligned}$$

which is basically the sum of the three balance equations

$$\begin{aligned}
\frac{\partial}{\partial t}\rho \phi + \frac{\partial}{\partial r_j} v_j \rho \phi &=  + \rho \frac{\partial}{\partial t} \phi \\
\frac{\partial}{\partial t}\rho e_{k} + \frac{\partial}{\partial r_j} v_j \rho e_{k} &= -v_j \frac{\partial}{\partial r_j} p
+ v_i \frac{\partial}{\partial r_j} \sigma'_{ij}\\
\frac{\partial}{\partial t}\rho e_{int} + \frac{\partial}{\partial r_j} v_j \rho e_{int} &= -p \frac{\partial}{\partial r_j}v_j
+\sigma'_{ij} \frac{\partial}{\partial r_j}v_i\end{aligned}$$

If we do not include potential energy in the total energy but use instead
$$e = e_k+e_{int}$$ we can write

$$\begin{aligned}
\frac{\partial}{\partial t}\rho e + \frac{\partial}{\partial r_j} v_j \rho e = -\frac{\partial}{\partial r_j} \left( v_j p \right) 
+\frac{\partial}{\partial r_j} \left( v_i \sigma'_{ij} \right)  - v_i \rho \frac{\partial}{\partial r_i} \phi. \label{eq:etotal2}\end{aligned}$$

which can be split into two balance equations

$$\begin{aligned}
\frac{\partial}{\partial t}\rho e_{k} + \frac{\partial}{\partial r_j} v_j \rho e_{k} &= -v_j \frac{\partial}{\partial r_j} p
+ v_i \frac{\partial}{\partial r_j} \sigma'_{ij} - v_i \rho \frac{\partial}{\partial r_i} \phi\\
\frac{\partial}{\partial t}\rho e_{int} + \frac{\partial}{\partial r_j} v_j \rho e_{int} &= -p \frac{\partial}{\partial r_j}v_j
+\sigma'_{ij} \frac{\partial}{\partial r_j}v_i\end{aligned}$$

where gravity is now just treated as a source in the balance equation of the
kinetic energy.

One might recognize, that we didn’t mention selfgravity. In fact for the
derivation of the energy equation <a href="#eq:etotal" data-reference-type="eqref" data-reference="eq:etotal">[eq:etotal]</a> we assumed, that the
potential energy of the local fluid element is not due to selfgravity but due
to some external potential generated by the rest of the fluid. This assumption
is valid in case the local fluid element does not contribute much to the global
potential $$\phi$$. This means, that

$$\begin{aligned}
\phi_{local}= -G \int_{V_{local}}\frac{\rho \left( x_j \right) }{\left\lvert x_i-x_j\right\rvert}dV_j \ll
\phi_{global}= -G \int_{V_{global}}\frac{\rho \left( x_j \right) }{\left\lvert x_i-x_j\right\rvert}dV_j\end{aligned}$$

Because we know from the last chapter that the term $$- v_i \rho \frac{\partial}{\partial r_i} \phi$$
transforms into $$-\frac{1}{2}\frac{\partial}{\partial t}\rho\phi$$ in the selfgravity case we could
implement a correction in <a href="#eq:etotal2" data-reference-type="eqref" data-reference="eq:etotal2">[eq:etotal2]</a>, which accounts for a rising
"selfgravitiness" of a local fluid element like

$$\begin{aligned}
\begin{split}
\frac{\partial}{\partial t}\rho e + \frac{\partial}{\partial r_j} v_j \rho e =& -\frac{\partial}{\partial r_j} \left( v_j p \right) 
+\frac{\partial}{\partial r_j} \left( v_i \sigma'_{ij} \right)  - v_i \rho \frac{\partial}{\partial r_i} \phi\\
&+\frac{\phi_{local}}{\phi_{global}}
 \left( \frac{1}{2}\frac{\partial}{\partial t}\rho\phi-\frac{\partial}{\partial r_j}v_j\rho\phi-\rho\frac{\partial}{\partial t}\phi \right) 
\end{split}\end{aligned}$$

Nevertheless a more practical solution for numerical simulations might be to
refine the grid in such a way that the condition $$\phi_{local} \ll
\phi_{global}$$ holds everywhere in the computational domain.
[^12]

#### Divergence equation

Using

$$\begin{aligned}
v_j \frac{\partial}{\partial r_j}( v_i) = \frac{\partial}{\partial r_j}(v_i v_j) - v_i\frac{\partial}{\partial r_j}v_j\end{aligned}$$

we can express the euler equation <a href="#eq:vel" data-reference-type="eqref" data-reference="eq:vel">[eq:vel]</a> like

$$\begin{aligned}
\frac{\partial}{\partial t}(v_i) +\frac{\partial}{\partial r_j}(v_i v_j) - v_i\frac{\partial}{\partial r_j}v_j = 
-\frac{1}{\rho}\frac{\partial}{\partial r_i}p + \frac{1}{\rho}\frac{\partial}{\partial r_j}\sigma'_{ij} + g_i.\end{aligned}$$

Taking the divergence of this equation yields

$$\begin{aligned}
\frac{\partial}{\partial t} \left( \frac{\partial v_i}{\partial r_i} \right) &+\frac{\partial^2}{\partial r_i \partial r_j}(v_i
v_j)
- \left( \frac{\partial v_i}{\partial r_i} \right)  \left( \frac{\partial v_j}{\partial r_j} \right) -v_i\frac{\partial}{\partial r_i} \left( \frac{\partial v_j}{\partial r_j} \right) =\\
&\frac{1}{\rho^2} \left( \frac{\partial \rho}{\partial r_i} \right)  \left( \frac{\partial p}{\partial r_i} \right) 
-\frac{1}{\rho}\frac{\partial^2}{\partial r_i^2}p
-\frac{1}{\rho^2} \left( \frac{\partial \rho}{\partial r_i} \right)  \left( \frac{\partial}{\partial r_j}\sigma'_{ij} \right) 
+\frac{1}{\rho}\frac{\partial^2}{\partial r_i \partial r_j}\sigma'_{ij}
+\frac{\partial}{\partial r_i}g_i\end{aligned}$$

With the poisson equation <a href="#eq:maxgrav1" data-reference-type="eqref" data-reference="eq:maxgrav1">[eq:maxgrav1]</a> and using the notation from
Truesdell for the divergence $$\theta=\frac{\partial v_i}{\partial r_i}$$
we get an equation for the divergence of an arbitrary fluid

$$\begin{aligned}
\begin{split}
\frac{\partial}{\partial t}\theta-\theta^2-v_i\frac{\partial}{\partial r_i}\theta
+\frac{\partial^2}{\partial r_i \partial r_j}(v_i v_j) =&
\frac{1}{\rho^2} \left( \frac{\partial \rho}{\partial r_i} \right) 
 \left[ \frac{\partial p}{\partial r_i}-\frac{\partial}{\partial r_j}\sigma'_{ij} \right] \\
&-\frac{1}{\rho} \left[ \frac{\partial^2}{\partial r_i^2}p
-\frac{\partial^2}{\partial r_i \partial r_j}\sigma'_{ij} \right] 
-4\pi G \rho
\end{split}\end{aligned}$$

or in vector notation

$$\begin{aligned}
\begin{split}
\frac{\partial}{\partial t}\theta-\theta^2-(\boldsymbol{v}\cdot \nabla)\theta 
+\nabla \left[ \nabla\cdot(v_i v_j) \right] =&
\frac{1}{\rho^2}\nabla \rho \cdot  \left[ \nabla p - \nabla \cdot \sigma'_{ij} \right] \\
&-\frac{1}{\rho} \left[ \Delta p
-\nabla(\nabla \cdot \sigma'_{ij}) \right] 
-4\pi G \rho
\end{split}\end{aligned}$$

#### Vorticity equation

Using the vector identity <a href="#eq:vecid02" data-reference-type="eqref" data-reference="eq:vecid02">[eq:vecid02]</a> we can express the euler
equation <a href="#eq:vel" data-reference-type="eqref" data-reference="eq:vel">[eq:vel]</a> like

$$\begin{aligned}
\frac{\partial}{\partial t}(v_i) + \frac{\partial}{\partial r_i}  \left( \frac{1}{2} v_j v_j \right) 
- \epsilon_{ijm} v_j \epsilon_{mkl} \frac{\partial}{\partial r_k} v_l = -\frac{1}{\rho}\frac{\partial}{\partial r_i}p +
\frac{1}{\rho}\frac{\partial}{\partial r_j}\sigma'_{ij} + g_i.\end{aligned}$$

We can obtain an equation for the vorticity of the flow field by taking the
curl of this form of the euler equation

$$\begin{aligned}
\begin{split}
\epsilon_{ghi}\frac{\partial}{\partial r_h}\frac{\partial}{\partial t}(v_i) 
+\epsilon_{ghi}\frac{\partial}{\partial r_h} \frac{\partial}{\partial r_i} \left( \frac{1}{2} v_j v_j \right) 
-\epsilon_{ghi}\frac{\partial}{\partial r_h} \epsilon_{ijm} v_j \epsilon_{mkl} \frac{\partial}{\partial r_k} v_l =\\
-\epsilon_{ghi}\frac{\partial}{\partial r_h} \left( \frac{1}{\rho}\frac{\partial}{\partial r_i}p \right)  
+\epsilon_{ghi}\frac{\partial}{\partial r_h} \left( \frac{1}{\rho}\frac{\partial}{\partial r_j}\sigma'_{ij} \right)  
+\epsilon_{ghi}\frac{\partial}{\partial r_h} g_i.
\end{split}\end{aligned}$$

If the time and space derivative of the velocity field commute, if
the spatial derivatives of the velocity field commute[^13] and if the
gravitational field can be expressed by $$g_i=-\frac{\partial}{\partial r_i}\phi$$ (means as the
gradient of a potential) we see that the second term on the left hand side and
also the term due to gravity are zero. This is so because these terms are
equivalent to the curl of a gradient of a scalar field which is a zero vector.
So we get

$$\begin{aligned}
\begin{split}
\frac{\partial}{\partial t}\epsilon_{ghi}\frac{\partial}{\partial r_h}v_i
-\epsilon_{ghi}\frac{\partial}{\partial r_h} \epsilon_{ijm} v_j \epsilon_{mkl} \frac{\partial}{\partial r_k} v_l =
&-\epsilon_{ghi} \left[ \frac{1}{\rho}\frac{\partial}{\partial r_h} \left( \frac{\partial}{\partial r_i}p \right)  
+  \left( \frac{\partial}{\partial r_i}p \right)   \left( \frac{\partial}{\partial r_h}\frac{1}{\rho} \right)  \right] \\
&+\epsilon_{ghi} \left[ \frac{1}{\rho}\frac{\partial}{\partial r_h} \left( \frac{\partial}{\partial r_j}\sigma'_{ij} \right) 
+  \left( \frac{\partial}{\partial r_j}\sigma'_{ij} \right)   \left( \frac{\partial}{\partial r_h}\frac{1}{\rho} \right)  \right] 
\end{split}\end{aligned}$$

The first term on the right hand side is zero again, because it is the curl
of a gradient field and so we get as the vorticity equation
[^14]
for some arbitrary
fluid with $$\omega_g=\epsilon_{ghi}\frac{\partial}{\partial r_h}v_i$$

$$\begin{aligned}
\begin{split}
\frac{\partial}{\partial t}\omega_g
-\epsilon_{ghi}\frac{\partial}{\partial r_h} \epsilon_{ijm} v_j \omega_m =
&-\frac{1}{\rho^2} \left[ 
\epsilon_{ghi}  \left( \frac{\partial}{\partial r_h}\rho \right)   \left( \frac{\partial}{\partial r_i}p \right) 
-\epsilon_{ghi}  \left( \frac{\partial}{\partial r_h}\rho \right)   \left( \frac{\partial}{\partial r_j}\sigma'_{ij} \right)  \right] \\
&+\frac{1}{\rho}\epsilon_{ghi}\frac{\partial}{\partial r_h}\frac{\partial}{\partial r_j}\sigma'_{ij}
\end{split}\end{aligned}$$

or in vector notation with $$\boldsymbol{\omega}=\nabla \times \boldsymbol{v}$$

$$\begin{aligned}
\frac{\partial}{\partial t} \boldsymbol{\omega}-\nabla \times (\boldsymbol{v} \times \boldsymbol{\omega}) = 
-\frac{1}{\rho^2} \left[ (\nabla \rho) \times(\nabla p)
- (\nabla \rho) \times (\nabla \cdot \tilde{\sigma}) \right] 
+\frac{1}{\rho}  \left[ \nabla \times (\nabla \cdot \tilde{\sigma}) \right] \end{aligned}$$

#### Summary

##### Balance equations [balance-equations-1]

$$\begin{aligned}
\frac{\partial}{\partial t}\rho + \frac{\partial}{\partial r_j}(v_j \rho) &= 0 \label{eq:mass}\\
\frac{\partial}{\partial t}(\rho v_i) + \frac{\partial}{\partial r_j}(v_j \rho v_i) &= -\frac{\partial}{\partial r_i}p + \frac{\partial}{\partial r_j}\sigma'_{ij}
+\rho g_i 
\label{eq:mom} \\
\frac{\partial}{\partial t}(\rho e) + \frac{\partial}{\partial r_j}(v_j \rho e) &= -\frac{\partial}{\partial r_j}(v_j p) + \frac{\partial}{\partial r_j}(v_i
\sigma'_{ij}) + v_i \rho g_i
\label{eq:etot}\end{aligned}$$

with Newtonian gravity (Poisson Equation):

$$\begin{aligned}
\frac{\partial}{\partial r_j}g_j=4\pi G \rho\end{aligned}$$

and an equation of state dependent on the material of the fluid.
[^15]

##### Global dissipation of kinetic energy $$\mathcal{E}$$ [global-dissipation-of-kinetic-energy-mathcale]

$$\begin{aligned}
\mathcal{E} = \frac{1}{V} \frac{\partial}{\partial t} E_{kin} = 
\frac{1}{V} \int_V p \frac{\partial}{\partial r_j} v_j dV 
-\frac{1}{V} \int_V \sigma'_{ij}\frac{\partial}{\partial r_j} v_i dV
-\frac{1}{V} \int_V \phi \frac{\partial}{\partial t}\rho dV
\label{eq:diss}\end{aligned}$$

##### Divergence equation [divergence-equation-1]

$$\begin{aligned}
\begin{split}
\frac{\partial}{\partial t}\theta-\theta^2-v_i\frac{\partial}{\partial r_i}\theta
+\frac{\partial^2}{\partial r_i \partial r_j}(v_i v_j) =&
\frac{1}{\rho^2} \left( \frac{\partial \rho}{\partial r_i} \right) 
 \left[ \frac{\partial p}{\partial r_i}-\frac{\partial}{\partial r_j}\sigma'_{ij} \right] \\
&-\frac{1}{\rho} \left[ \frac{\partial^2}{\partial r_i^2}p
-\frac{\partial^2}{\partial r_i \partial r_j}\sigma'_{ij} \right] 
-4\pi G \rho
\end{split}
\label{eq:div}\end{aligned}$$

##### Vorticity equation [vorticity-equation-1]

$$\begin{aligned}
\begin{split}
\frac{\partial}{\partial t}\omega_g
-\epsilon_{ghi}\frac{\partial}{\partial r_h} \epsilon_{ijm} v_j \omega_m =
&-\frac{1}{\rho^2} \left[ 
\epsilon_{ghi}  \left( \frac{\partial}{\partial r_h}\rho \right)   \left( \frac{\partial}{\partial r_i}p \right) 
-\epsilon_{ghi}  \left( \frac{\partial}{\partial r_h}\rho \right)   \left( \frac{\partial}{\partial r_j}\sigma'_{ij} \right)  \right] \\
&+\frac{1}{\rho}\epsilon_{ghi}\frac{\partial}{\partial r_h}\frac{\partial}{\partial r_j}\sigma'_{ij}
\end{split}
\label{eq:vort}\end{aligned}$$

### Newtonian compressible fluid

For a so called newtonian fluid it can be shown, that the stress tensor
$$\sigma'_{ij}$$ is of the form

$$\begin{aligned}
\sigma'_{ij}=2\eta S^*_{ij}+\zeta \delta_{ij}\frac{\partial v_k}{\partial r_k}
\label{eq:stress}\end{aligned}$$

with $$S^*_{ij}$$ being the symmetric tracefree part of the tensor
$$\frac{\partial v_i}{\partial x_j}$$

$$\begin{aligned}
S^*_{ij}= \left[ \frac{1}{2} \left( \frac{\partial}{\partial r_j}v_i+\frac{\partial}{\partial r_i}v_j \right) -\frac{1}{3}\delta_{ij}
\frac{\partial v_k}{\partial r_k} \right] \end{aligned}$$

The parameter $$\eta'$$ is called dynamic viscosity and $$\zeta$$ is the so called
second dynamic viscosity. The second term of equation <a href="#eq:stress" data-reference-type="eqref" data-reference="eq:stress">[eq:stress]</a> is
often considered as small and therefore neglected. This is true in case of a
monoatomic gases there it can be shown, that $$\zeta=0$$ (Landau/Lifschitz 10).
In case of a incompressible fluid with constant density the term can also be
neglected, because $$\frac{\partial v_k}{\partial r_k} = 0$$. Nevertheless for compressible fluids
(supersonic regime) $$\frac{\partial v_k}{\partial r_k}$$ can be very large (shocks) and the second
dynamic viscosity of a n-atomic gas can not be neglected. In this case the
second term of the stress tensor additionally contributes to the pressure,
which should be considered in the equation of state. This will alter the nature
of $$p$$ as a thermodynamic variable, which should only depend on the local values
of $$\rho$$ and $$e$$ and not on $$\frac{\partial v_k}{\partial r_k}$$. But since we have a stress tensor
we locally do not have a local thermodynamic equilibrium anyway, so one should
expect a change in the nature of the thermodynamic variables, which are
defined for local thermodynamic equilibrium.

#### Balance equations

In the following we will write the stress tensor for a newtonian compressible
fluid in the form

$$\begin{aligned}
\sigma'_{ij}= \eta \left( \frac{\partial}{\partial r_j}v_i+\frac{\partial}{\partial r_i}v_j \right)  
+  \left( \zeta-\frac{2}{3}\eta \right) \delta_{ij}\frac{\partial v_k}{\partial r_k} 
\label{eq:stress2}\end{aligned}$$

Inserting this explicitely into the momentum equation for a compressible fluid
<a href="#eq:mom" data-reference-type="eqref" data-reference="eq:mom">[eq:mom]</a> one gets for the term

$$\begin{aligned}
\begin{split}
\frac{\partial}{\partial r_j} \sigma'_{ij} &= 
\frac{\partial}{\partial r_j} \left[ \eta  \left( \frac{\partial v_i}{\partial r_j}+\frac{\partial v_j}{\partial r_i} \right)  \right] 
+ \left( \zeta-\frac{2}{3}\eta \right) \delta_{ij}\frac{\partial}{\partial r_j} \left( \frac{\partial v_k}{\partial r_k} \right) \\
&= \eta  \left[ \frac{\partial^2}{\partial r_j^2}v_i +\frac{\partial}{\partial r_i} \left( \frac{\partial v_j}{\partial r_j} \right)  \right] 
+ \left( \zeta-\frac{2}{3}\eta \right) \frac{\partial}{\partial r_i} \left( \frac{\partial v_k}{\partial r_k} \right) \\
&=\eta \frac{\partial^2}{\partial r_j^2}v_i
+ \left( \frac{\eta}{3}+\zeta \right) \frac{\partial}{\partial r_i} \left( \frac{\partial v_k}{\partial r_k} \right) .
\end{split}
\label{eq:divstress}\end{aligned}$$

Inserting the stress tensor into the energy equation for a compressible fluid
<a href="#eq:etot" data-reference-type="eqref" data-reference="eq:etot">[eq:etot]</a> one gets for

$$\begin{aligned}
\begin{split}
\frac{\partial}{\partial r_j}(v_i \sigma'_{ij}) 
&=\frac{\partial}{\partial r_j} \left[ \eta v_i  \left( \frac{\partial v_i}{\partial r_j}+\frac{\partial v_j}{\partial r_i} \right)  \right] 
+ \left( \zeta-\frac{2}{3}\eta \right) \frac{\partial}{\partial r_j} \left( v_j \frac{\partial v_k}{\partial r_k} \right) \\
&=\eta \frac{\partial^2}{\partial r_j^2} \left( \frac{1}{2}v_i^2 \right) +\eta \frac{\partial}{\partial r_j} \left( v_i\frac{\partial v_j}{\partial r_i} \right) 
+ \left( \zeta-\frac{2}{3}\eta \right) \frac{\partial}{\partial r_j} \left( v_j \frac{\partial v_k}{\partial r_k} \right) 
\end{split}\end{aligned}$$

In the end we get the following balance equations for a compressible,
selfgravitating, newtonian fluid

$$\begin{aligned}
\frac{\partial}{\partial t}\rho + \frac{\partial}{\partial r_j}(v_j \rho) =&\ 0 \\
\frac{\partial}{\partial t}(\rho v_i) + \frac{\partial}{\partial r_j}(v_j \rho v_i) =& -\frac{\partial}{\partial r_i}p + +\rho g_i
+\eta\frac{\partial^2}{\partial r_j^2}v_i
+ \left( \frac{\eta}{3}+\zeta \right) \frac{\partial}{\partial r_i} \left( \frac{\partial v_k}{\partial r_k} \right) \\
\begin{split}
\frac{\partial}{\partial t}(\rho e) + \frac{\partial}{\partial r_j}(v_j \rho e) =& -\frac{\partial}{\partial r_j}(v_j p) + v_i \rho g_i 
+\eta \frac{\partial^2}{\partial r_j^2} \left( \frac{1}{2}v_i^2 \right) + \eta \frac{\partial}{\partial r_j} \left( v_i\frac{\partial v_j}{\partial r_i} \right)  \\
&+ \left( \zeta-\frac{2}{3}\eta \right)  \frac{\partial}{\partial r_j} \left( v_j\frac{\partial r_k}{\partial v_k} \right) 
\end{split}\end{aligned}$$

#### Global dissipation of kinetic energy $$\mathcal{E}$$ [global-dissipation-of-kinetic-energy-mathcale-1]

Using the form <a href="#eq:stress2" data-reference-type="eqref" data-reference="eq:stress2">[eq:stress2]</a> we get for the term involving the stress
tensor in the equation <a href="#eq:diss" data-reference-type="eqref" data-reference="eq:diss">[eq:diss]</a> for the dissipation $$\mathcal{E}$$

$$\begin{aligned}
\begin{split}
\sigma'_{ij}\frac{\partial v_i}{\partial r_j} 
&= \eta \left( \frac{\partial}{\partial r_j}v_i+\frac{\partial}{\partial r_i}v_j \right) \frac{\partial v_i}{\partial r_j}
+  \left( \zeta-\frac{2}{3}\eta \right) \delta_{ij}\frac{\partial v_k}{\partial r_k}\frac{\partial v_i}{\partial r_j}\\
&=\frac{\eta}{2} \left( \frac{\partial}{\partial r_j}v_i+\frac{\partial}{\partial r_i}v_j \right) ^2
+ \left( \zeta-\frac{2}{3}\eta \right)  \left( \frac{\partial v_k}{\partial r_k} \right) ^2
\end{split}\end{aligned}$$

where we also made use of the relation <a href="#eq:uscontr" data-reference-type="eqref" data-reference="eq:uscontr">[eq:uscontr]</a> for the contraction of
a symmetric with an unsymmetric tensor. With this result the global dissipation
of kinetic energy for a newtonian compressible fluid is

$$\begin{aligned}
\begin{split}
\mathcal{E} =& 
\frac{1}{V} \int_V p \frac{\partial v_j}{\partial r_j} dV
-\frac{1}{V} \int_V \phi \frac{\partial}{\partial t}\rho dV
-\frac{1}{V} \int_V \frac{\eta}{2} \left( \frac{\partial}{\partial r_j}v_i+\frac{\partial}{\partial r_i}v_j \right) ^2 dV\\
&-\frac{1}{V} \int_V  \left( \zeta-\frac{2}{3}\eta \right)  \left( \frac{\partial v_k}{\partial r_k} \right) ^2 dV
\end{split}\end{aligned}$$

This equation should be compared to equation (79,1) from Landau and Lifschitz (1991)
which additionally includes heat conduction. Nevertheless Landau and Lifschitz (1991)
seem to forget the term due to the pressure in the equation for the
dissipation.

#### Divergence equation

Using the equation for the divergence of the stress tensor
for a newtonian compressible fluid <a href="#eq:divstress" data-reference-type="eqref" data-reference="eq:divstress">[eq:divstress]</a> we get for

$$\begin{aligned}
\begin{split}
\frac{\partial}{\partial r_i} \left( \frac{\partial}{\partial r_j}\sigma'_{ij} \right) &=
\eta\frac{\partial}{\partial r_i}\frac{\partial^2}{\partial r_j^2}v_i+ \left( \frac{\eta}{3}+\zeta \right) \frac{\partial^2}{\partial r_i^2}\theta\\
&=\eta\frac{\partial^2}{\partial r_j^2}\theta+ \left( \frac{\eta}{3}+\zeta \right) \frac{\partial^2}{\partial r_i^2}\theta\\
&= \left( \frac{4}{3}\eta+\zeta \right) \frac{\partial^2}{\partial r_i^2}\theta
\end{split}\end{aligned}$$

By inserting this and equation <a href="#eq:divstress" data-reference-type="eqref" data-reference="eq:divstress">[eq:divstress]</a> into the equation for the
divergence <a href="#eq:div" data-reference-type="eqref" data-reference="eq:div">[eq:div]</a> we get the divergence equation for a compressible
newtonian fluid

$$\begin{aligned}
\begin{split}
\frac{\partial}{\partial t}\theta-\theta^2-v_i\frac{\partial}{\partial r_i}\theta
+\frac{\partial^2}{\partial r_i \partial r_j}(v_i v_j) =&
\frac{1}{\rho^2} \left( \frac{\partial \rho}{\partial r_i} \right)  \cdot
 \left[ \frac{\partial p}{\partial r_i}-\eta\frac{\partial^2}{\partial r_j^2}v_i- \left( \frac{\eta}{3}+\zeta \right) 
\frac{\partial}{\partial r_i}\theta \right] \\
&-\frac{1}{\rho} \left[ \frac{\partial^2}{\partial r_i^2}p- \left( \frac{4}{3}\eta+\zeta \right) \frac{\partial^2}{\partial r_i^2}\theta \right]  
-4\pi G \rho
\end{split}\end{aligned}$$

or in vector notation

$$\begin{aligned}
\begin{split}
\frac{\partial}{\partial t}\theta-\theta^2-(\boldsymbol{v}\cdot \nabla)\theta 
+\nabla \left[ \nabla\cdot(v_i v_j) \right] =&
\frac{1}{\rho^2}\nabla \rho \cdot  \left[ \nabla p - \eta \Delta v
- \left( \frac{\eta}{3}+\zeta \right) \nabla \theta \right] \\
&-\frac{1}{\rho} \left[ \Delta p- \left( \frac{4}{3}\eta+\zeta \right) \Delta\theta \right] 
-4\pi G \rho
\end{split}\end{aligned}$$

#### Vorticity equation

If we plugin the equation for the divergence of the
stress tensor for a newtonian compressible fluid <a href="#eq:divstress" data-reference-type="eqref" data-reference="eq:divstress">[eq:divstress]</a> into
the vorticity equation <a href="#eq:vort" data-reference-type="eqref" data-reference="eq:vort">[eq:vort]</a> we get for a compressible newtonian
fluid with $$\theta=\frac{\partial v_k}{\partial r_k}$$

$$\begin{aligned}
\begin{split}
\frac{\partial}{\partial t}\omega_g
-\epsilon_{ghi}\frac{\partial}{\partial r_h} \epsilon_{ijm} v_j \omega_m =
&-\frac{1}{\rho^2}\epsilon_{ghi}\left[
 \left( \frac{\partial}{\partial r_h}\rho \right)   \left( \frac{\partial}{\partial r_i}p \right) 
-\eta  \left( \frac{\partial}{\partial r_h}\rho \right)   \left( \frac{\partial^2}{\partial r_j^2}v_i \right) \right.\\
&\left.- \left( \frac{\eta}{3}+\zeta \right)   \left( \frac{\partial}{\partial r_h}\rho \right)  
 \left( \frac{\partial}{\partial r_i}\theta \right)  \right]
+\frac{\eta}{\rho}\frac{\partial^2}{\partial r_j^2}\omega_g \\
&+\frac{1}{\rho} \left( \frac{\eta}{3}
+\zeta \right) \epsilon_{ghi}\frac{\partial}{\partial r_h}\frac{\partial}{\partial r_i} \left( \frac{\partial v_k}{\partial r_k} \right) 
\end{split}\end{aligned}$$

The last term on the right hand side vanishes, because it is the curl
of a gradient field. Therefore we are left with

$$\begin{aligned}
\begin{split}
\frac{\partial}{\partial t}\omega_g
-\epsilon_{ghi}\frac{\partial}{\partial r_h} \epsilon_{ijm} v_j \omega_m =
&-\frac{1}{\rho^2}\epsilon_{ghi}\left[
 \left( \frac{\partial}{\partial r_h}\rho \right)   \left( \frac{\partial}{\partial r_i}p \right) 
-\eta  \left( \frac{\partial}{\partial r_h}\rho \right)   \left( \frac{\partial^2}{\partial r_j^2}v_i \right) \right.\\
&\left.- \left( \frac{\eta}{3}+\zeta \right)   \left( \frac{\partial}{\partial r_h}\rho \right)  
 \left( \frac{\partial}{\partial r_i}\theta \right)  \right]
+\frac{\eta}{\rho}\frac{\partial^2}{\partial r_j^2}\omega_g
\end{split}\end{aligned}$$

or in vector notation with

$$\begin{aligned}
\begin{split}
\frac{\partial}{\partial t} \boldsymbol{\omega}-\nabla \times (\boldsymbol{v} \times \boldsymbol{\omega}) = 
&-\frac{1}{\rho^2}\left[
(\nabla \rho) \times(\nabla p)
- \eta (\nabla \rho) \times (\Delta \boldsymbol{v})\right.\\
&\left.-  \left( \zeta +\frac{\eta}{3} \right)  (\nabla \rho) \times (\nabla
\theta)\right]
+\frac{\eta}{\rho} \Delta \boldsymbol{\omega}
\end{split}\end{aligned}$$

#### Summary

##### Balance equations [balance-equations-3]

$$\begin{aligned}
\frac{\partial}{\partial t}\rho + \frac{\partial}{\partial r_j}(v_j \rho) =&\ 0 \label{eq:ncmass}\\
\frac{\partial}{\partial t}(\rho v_i) + \frac{\partial}{\partial r_j}(v_j \rho v_i) =& -\frac{\partial}{\partial r_i}p + +\rho g_i
+\eta\frac{\partial^2}{\partial r_j^2}v_i
+ \left( \frac{\eta}{3}+\zeta \right) \frac{\partial}{\partial r_i} \left( \frac{\partial v_k}{\partial r_k} \right)  \label{eq:ncmom}\\
\begin{split}
\frac{\partial}{\partial t}(\rho e) + \frac{\partial}{\partial r_j}(v_j \rho e) =& -\frac{\partial}{\partial r_j}(v_j p) + v_i \rho g_i 
+\eta \frac{\partial^2}{\partial r_j^2} \left( \frac{1}{2}v_i^2 \right) + \eta \frac{\partial}{\partial r_j} \left( v_i\frac{\partial v_j}{\partial r_i} \right)  \\
&+ \left( \zeta-\frac{2}{3}\eta \right)  \frac{\partial}{\partial r_j} \left( v_j\frac{\partial r_k}{\partial v_k} \right)  \label{eq:ncetot}
\end{split}\end{aligned}$$

with Newtonian gravity (Poisson Equation):

$$\begin{aligned}
\frac{\partial}{\partial r_j}g_j=4\pi G \rho\end{aligned}$$

and an equation of state dependent on the material of the fluid.
[^16]

##### Global dissipation of kinetic energy $$\mathcal{E}$$ [global-dissipation-of-kinetic-energy-mathcale-2]

$$\begin{aligned}
\begin{split}
\mathcal{E} =& 
\frac{1}{V} \int_V p \frac{\partial v_j}{\partial r_j} dV
-\frac{1}{V} \int_V \phi \frac{\partial}{\partial t}\rho dV
-\frac{1}{V} \int_V \frac{\eta}{2} \left( \frac{\partial}{\partial r_j}v_i+\frac{\partial}{\partial r_i}v_j \right) ^2 dV\\
&-\frac{1}{V} \int_V  \left( \zeta-\frac{2}{3}\eta \right)  \left( \frac{\partial v_k}{\partial r_k} \right) ^2 dV
\end{split}
\label{eq:ncdiss}\end{aligned}$$

##### Divergence equation [divergence-equation-3]

$$\begin{aligned}
\begin{split}
\frac{\partial}{\partial t}\theta-\theta^2-v_i\frac{\partial}{\partial r_i}\theta
+\frac{\partial^2}{\partial r_i \partial r_j}(v_i v_j) =&
\frac{1}{\rho^2} \left( \frac{\partial \rho}{\partial r_i} \right)  \cdot
 \left[ \frac{\partial p}{\partial r_i}-\eta\frac{\partial^2}{\partial r_j^2}v_i- \left( \frac{\eta}{3}+\zeta \right) 
\frac{\partial}{\partial r_i}\theta \right] \\
&-\frac{1}{\rho} \left[ \frac{\partial^2}{\partial r_i^2}p- \left( \frac{4}{3}\eta+\zeta \right) \frac{\partial^2}{\partial r_i^2}\theta \right]  
-4\pi G \rho
\end{split}\end{aligned}$$

##### Vorticity equation [vorticity-equation-3]

$$\begin{aligned}
\begin{split}
\frac{\partial}{\partial t}\omega_g
-\epsilon_{ghi}\frac{\partial}{\partial r_h} \epsilon_{ijm} v_j \omega_m =
&-\frac{1}{\rho^2}\epsilon_{ghi}\left[
 \left( \frac{\partial}{\partial r_h}\rho \right)   \left( \frac{\partial}{\partial r_i}p \right) 
-\eta  \left( \frac{\partial}{\partial r_h}\rho \right)   \left( \frac{\partial^2}{\partial r_j^2}v_i \right) \right.\\
&\left.- \left( \frac{\eta}{3}+\zeta \right)   \left( \frac{\partial}{\partial r_h}\rho \right)  
 \left( \frac{\partial}{\partial r_i}\theta \right)  \right]
+\frac{\eta}{\rho}\frac{\partial^2}{\partial r_j^2}\omega_g
\end{split}\end{aligned}$$

### General incompressible fluid

#### Balance equations

When we talk about an incompressible fluid we mean that the density of the fluid
is constant in time, i.e. $$\frac{\partial}{\partial t}\rho=0$$. In most cases it is also assumed that
the fluid is not stratified, that means the density is also spatially constant,
i.e. $$\frac{\partial}{\partial r_j}\rho=0$$. Therefore one could call an incompressible fluid also a
constant density fluid.

With constant density the continuity equation <a href="#eq:mass" data-reference-type="eqref" data-reference="eq:mass">[eq:mass]</a> becomes

$$\begin{aligned}
\frac{\partial \rho}{\partial t} + \frac{\partial}{\partial r_j}(v_j \rho) &= 0 \\
\Leftrightarrow \frac{\partial \rho}{\partial t}+ v_j \frac{\partial \rho}{\partial r_j} + \rho \frac{\partial v_j}{\partial r_j} &= 0\\
\Rightarrow \frac{\partial v_j}{\partial r_j} &= 0,\ \text{with $\rho\neq 0$} \label{eq:divzero}\end{aligned}$$

The momentum equation <a href="#eq:mom" data-reference-type="eqref" data-reference="eq:mom">[eq:mom]</a> and the energy equation
<a href="#eq:etot" data-reference-type="eqref" data-reference="eq:etot">[eq:etot]</a>
become

$$\begin{aligned}
\rho  \left[ \frac{\partial v_i}{\partial t} + v_j \frac{\partial v_i}{\partial r_j} + v_i \frac{\partial v_j}{\partial r_j} \right]   
=& -\frac{\partial}{\partial r_i}p + +\rho g_i+ \frac{\partial}{\partial r_j}\sigma'_{ij} \\
\rho  \left[ \frac{\partial e}{\partial t} + v_j \frac{\partial e}{\partial r_j} + e \frac{\partial v_j}{\partial r_j} \right]  
=& -\frac{\partial}{\partial r_j}(v_j p) + v_i \rho g_i +\frac{\partial}{\partial r_j}(v_i \sigma'_{ij})\end{aligned}$$

If we make use of the relation <a href="#eq:divzero" data-reference-type="eqref" data-reference="eq:divzero">[eq:divzero]</a> in the momentum and energy
equation we finally get as equations for an incompressible, selfgravitating,
newtonian fluid

$$\begin{aligned}
\frac{\partial v_j}{\partial r_j} =&\ 0\\
\frac{\partial v_i}{\partial t} + v_j \frac{\partial v_i}{\partial r_j} =& -\frac{1}{\rho}\frac{\partial}{\partial r_i}p + g_i
+\frac{1}{\rho}\frac{\partial}{\partial r_j}\sigma^*_{ij}\\
\frac{\partial e}{\partial t} + v_j \frac{\partial e}{\partial r_j} =& -\frac{1}{\rho} \frac{\partial}{\partial r_j}(v_j p) + v_i g_i
+\frac{1}{\rho}\frac{\partial}{\partial r_j}(v_i \sigma^*_{ij}).\end{aligned}$$

where $$\sigma^*_{ij}$$ is a divergence free stress tensor.

#### Divergence equation

In the incompressible case we can set $$\theta=0$$ and $$\frac{\partial}{\partial r_i}\rho=0$$ in
equation <a href="#eq:div" data-reference-type="eqref" data-reference="eq:div">[eq:div]</a> and so we are left with the following equation for an
general incompressible fluid

$$\begin{aligned}
\frac{\partial^2}{\partial r_i \partial r_j}(v_i v_j) = 
-\frac{1}{\rho}\frac{\partial^2}{\partial r_i^2}p 
+\frac{1}{\rho}\frac{\partial^2}{\partial r_i \partial r_j}\sigma^*_{ij} 
- 4\pi G \rho\end{aligned}$$

Solving for $$p$$ we get

$$\begin{aligned}
\frac{\partial^2}{\partial r_i^2}p= 
-\rho \frac{\partial^2}{\partial r_i \partial r_j}(v_i v_j)
+\frac{\partial^2}{\partial r_i \partial r_j}\sigma^*_{ij} 
-4\pi G \rho^2\end{aligned}$$

or in vector notation

$$\begin{aligned}
\Delta p = 
-\rho \nabla \left[ \nabla\cdot(v_i v_j) \right]  
+\nabla \left[ \nabla\cdot \sigma^*_{ij} \right] 
- 4\pi G \rho^2\end{aligned}$$

which can be interpreted as the equation of state for a general
incompressible fluid.[^17]

#### Vorticity equation

For an incompressible fluid $$\frac{\partial}{\partial r_h}\rho = 0$$ and inserting this into
the vorticity equation <a href="#eq:vort" data-reference-type="eqref" data-reference="eq:vort">[eq:vort]</a> we get

$$\begin{aligned}
\frac{\partial}{\partial t}\omega_g
&-\epsilon_{ghi}\frac{\partial}{\partial r_h} \epsilon_{ijm} v_j \omega_m =
\frac{1}{\rho}\epsilon_{ghi}\frac{\partial}{\partial r_h}\frac{\partial}{\partial r_j}\sigma^*_{ij}\end{aligned}$$

or in vector notation

$$\begin{aligned}
\frac{\partial}{\partial t} \boldsymbol{\omega}-\nabla \times (\boldsymbol{v} \times \boldsymbol{\omega}) = 
\frac{1}{\rho}  \left[ \nabla \times (\nabla \cdot \tilde{\sigma^*}) \right] \end{aligned}$$

#### Summary

##### Balance equations [balance-equations-5]

$$\begin{aligned}
\frac{\partial v_j}{\partial r_j} =&\ 0 \label{eq:icmass}\\
\frac{\partial v_i}{\partial t} + v_j \frac{\partial v_i}{\partial r_j} =& -\frac{1}{\rho}\frac{\partial}{\partial r_i}p + g_i
+\frac{1}{\rho}\frac{\partial}{\partial r_j}\sigma^*_{ij}\label{eq:icmom}\\
\frac{\partial e}{\partial t} + v_j \frac{\partial e}{\partial r_j} =& -\frac{1}{\rho} \frac{\partial}{\partial r_j}(v_j p) + v_i g_i
+\frac{1}{\rho}\frac{\partial}{\partial r_j}(v_i \sigma^*_{ij}).\label{eq:icetot}\end{aligned}$$

with Newtonian gravity (Poisson Equation):

$$\begin{aligned}
\frac{\partial}{\partial r_j}g_j=4\pi G \rho\end{aligned}$$

and as equation of state the "divergence equation".

##### Global dissipation of kinetic energy $$\mathcal{E}$$ [global-dissipation-of-kinetic-energy-mathcale-3]

$$\begin{aligned}
\mathcal{E} = 
-\frac{1}{V} \int_V \sigma^*_{ij}\frac{\partial}{\partial r_j} v_i dV
\label{eq:icdiss}\end{aligned}$$

##### Divergence equation (Equation of State) [divergence-equation-equation-of-state]

$$\begin{aligned}
\frac{\partial^2}{\partial r_i^2}p= 
-\rho \frac{\partial^2}{\partial r_i \partial r_j}(v_i v_j)
+\frac{\partial^2}{\partial r_i \partial r_j}\sigma^*_{ij} 
-4\pi G \rho^2
\label{eq:icdiv}\end{aligned}$$

##### Vorticity equation [vorticity-equation-5]

$$\begin{aligned}
\frac{\partial}{\partial t}\omega_g
&-\epsilon_{ghi}\frac{\partial}{\partial r_h} \epsilon_{ijm} v_j \omega_m =
\frac{1}{\rho}\epsilon_{ghi}\frac{\partial}{\partial r_h}\frac{\partial}{\partial r_j}\sigma^*_{ij}
\label{eq:icvort}\end{aligned}$$

### Newtonian incompressible fluid

For an incompressible newtonian fluid the terms proportional to the divergence
vanish in the stress tensor $$\sigma'_{ij}$$. This leads to the following
divergence free stress tensor

$$\begin{aligned}
\sigma^*_{ij}=\eta  \left( \frac{\partial v_i}{\partial r_j}+\frac{\partial v_j}{\partial r_i} \right) . 
\label{eq:nicstress}\end{aligned}$$

#### Balance equations

Setting the divergence to zero in equation <a href="#eq:divstress" data-reference-type="ref" data-reference="eq:divstress">[eq:divstress]</a> we get for
the divergence of the stress tensor for an incompressible fluid

$$\begin{aligned}
\frac{\partial}{\partial r_j} \sigma^*_{ij} =\eta \frac{\partial^2}{\partial r_j^2}v_i \label{eq:nicdivstress}\end{aligned}$$

and for

$$\begin{aligned}
\frac{\partial}{\partial r_j}(v_i \sigma^*_{ij})
=\eta \frac{\partial^2}{\partial r_j^2} \left( \frac{1}{2}v_i^2 \right) +\eta\frac{\partial}{\partial r_j} \left( v_i\frac{\partial v_j}{\partial r_i} \right) \end{aligned}$$

Inserting these results into <a href="#eq:icmass" data-reference-type="eqref" data-reference="eq:icmass">[eq:icmass]</a>-<a href="#eq:icetot" data-reference-type="eqref" data-reference="eq:icetot">[eq:icetot]</a> we get
as balance equations for an incompressible newtonian fluid

$$\begin{aligned}
\frac{\partial v_j}{\partial r_j} =&\ 0\\
\frac{\partial v_i}{\partial t} + v_j \frac{\partial v_i}{\partial r_j} =& -\frac{1}{\rho}\frac{\partial}{\partial r_i}p + g_i
+\nu\frac{\partial^2}{\partial r_j^2}v_i\\
\frac{\partial e}{\partial t} + v_j \frac{\partial e}{\partial r_j} =& -\frac{1}{\rho} \frac{\partial}{\partial r_j}(v_j p) + v_i g_i
+\nu \frac{\partial^2}{\partial r_j^2} \left( \frac{1}{2}v_i^2 \right) + \nu \frac{\partial}{\partial r_j} \left( v_i\frac{\partial v_j}{\partial r_i} \right) \end{aligned}$$

with the so called kinematic viscosity $$\nu=\frac{\eta}{\rho}$$
[^18]

#### Global dissipation of kinetic energy $$\mathcal{E}$$ [nicdiss]

Inserting the incompressible newtonian stress tensor <a href="#eq:nicstress" data-reference-type="eqref" data-reference="eq:nicstress">[eq:nicstress]</a>
in the equation for the dissipation of a general incompressible fluid
<a href="#eq:icdiss" data-reference-type="ref" data-reference="eq:icdiss">[eq:icdiss]</a> yields[^19]

$$\begin{aligned}
\mathcal{E} 
=-\frac{1}{V} \int_V  \eta \left( \frac{\partial}{\partial r_j}v_i+\frac{\partial}{\partial r_i}v_j \right) \frac{\partial v_i}{\partial r_j} dV
=-\frac{1}{V} \int_V \frac{\eta}{2} \left( \frac{\partial}{\partial r_j}v_i+\frac{\partial}{\partial r_i}v_j \right) ^2 dV\end{aligned}$$

where we used again relation <a href="#eq:uscontr" data-reference-type="ref" data-reference="eq:uscontr">[eq:uscontr]</a>.

We can express this result in terms of vorticity by making use of
equation <a href="#eq:rsrcontr" data-reference-type="eqref" data-reference="eq:rsrcontr">[eq:rsrcontr]</a> which gives us for

$$\begin{aligned}
\frac{\eta}{2} \left( \frac{\partial}{\partial r_j}v_i+\frac{\partial}{\partial r_i}v_j \right) ^2
&= 2 \eta S_{ij} S_{ij}
= 2 \eta  \left( R_{ij} R_{ij} + \frac{\partial v_i}{\partial r_j}\frac{\partial v_j}{\partial r_i} \right) \\
&= 2 \eta  \left( \frac{1}{2} \omega_k \omega_k 
+ \frac{\partial}{\partial r_j} \left( v_i \frac{\partial v_j}{\partial r_i} \right)  
- v_j \frac{\partial^2}{\partial r_i \partial r_j} v_i \right) \\
&= 2 \eta  \left( \frac{1}{2} \omega_k \omega_k 
+ \frac{\partial}{\partial r_j} \left( v_i \frac{\partial v_j}{\partial r_i} \right)  
- v_j \frac{\partial^2}{\partial r_j \partial r_i} v_i \right) \\ 
 &= \eta \omega_k \omega_k + 2 \eta \frac{\partial}{\partial r_j} \left( v_i \frac{\partial v_j}{\partial r_i} \right) \end{aligned}$$

So we can express the dissipation for an incompressible newtonian fluid as

$$\begin{aligned}
\mathcal{E} = 
-\frac{1}{V} \int_V \eta \omega_k \omega_k dV
-\frac{1}{V} \int_V 
2 \eta \frac{\partial}{\partial r_j} \left( v_i \frac{\partial v_j}{\partial r_i} \right)  dV \label{eq:nicdissfull}\end{aligned}$$

Using Gauss’ theorem to transform the second term we can express this like

$$\begin{aligned}
\mathcal{E} = 
-\frac{1}{V} \int_V \eta \omega_k \omega_k dV
-\frac{1}{V} \oint_A 
2 \eta v_i \frac{\partial v_j}{\partial r_i} dA\end{aligned}$$

So if $$v_i \frac{\partial v_j}{\partial r_i}=0$$ on the border of the volume $$V$$ the second term
will vanish
and the dissipation in a incompressible, newtonian fluid is
due to the first term only [^20]
[^21]

$$\begin{aligned}
\mathcal{E} = -\frac{\eta}{V} \int_V \omega^2 dV \label{eq:nicdissvort}\end{aligned}$$

From looking at the divergence equation (see below) we can derive another
relation, which tells us, when the second term in equation
<a href="#eq:nicdissfull" data-reference-type="eqref" data-reference="eq:nicdissfull">[eq:nicdissfull]</a> will vanish

$$\begin{aligned}
\frac{1}{V} \int_V 2\eta \frac{\partial}{\partial r_j} \left( v_i \frac{\partial v_j}{\partial r_i} \right)  dV 
&=\frac{1}{V} \int_V 2\eta \frac{\partial^2}{\partial r_i \partial r_j}(v_i v_j)dV\\
&= \frac{1}{V} \int_V \frac {2\eta}{\rho} \left( \frac{\partial^2}{\partial r_i^2}p  + 4\pi G \rho^2 \right)  dV
=  \frac{2 \nu}{V} \int_V  \left( \frac{\partial^2}{\partial r_i^2}p  + \rho \frac{\partial^2}{\partial r_i^2}\phi \right)  dV\\
&= \frac{2 \nu}{V} \oint_A \frac{\partial}{\partial r_i}p dA + \frac{2 \eta}{V} \oint_A \frac{\partial}{\partial r_i}\phi dV\end{aligned}$$

where we made use of the poisson equation for the gravitational potential and
Gauss’ theorem. So only if the pressure gradient and the gravitational force
balance at the surface of the fluid we can neglect the second term [^22].

We hope that from this discussion the assumptions behind equation
<a href="#eq:nicdissvort" data-reference-type="eqref" data-reference="eq:nicdissvort">[eq:nicdissvort]</a>
become clear compared to the rather obscure arguments by Frisch (1995).
Nevertheless it is still unknown weather our reasoning and the reasoning of
Frisch (1995) is
equivalent.

#### Divergence equation

From equation <a href="#eq:nicdivstress" data-reference-type="eqref" data-reference="eq:nicdivstress">[eq:nicdivstress]</a> we get for

$$\begin{aligned}
\frac{\partial^2}{\partial r_i \partial r_j}\sigma^*_{ij} 
= \eta \frac{\partial}{\partial r_i}  \left( \frac{\partial^2}{\partial r_j^2}v_i \right)  
= \eta \frac{\partial^2}{\partial r_j^2}  \left( \frac{\partial v_i}{\partial r_i} \right)  
= 0\end{aligned}$$

Using this result in equation <a href="#eq:icdiv" data-reference-type="eqref" data-reference="eq:icdiv">[eq:icdiv]</a> we get as equation of state for
an incompressible newtonian fluid

$$\begin{aligned}
\frac{\partial^2}{\partial r_i^2}p= 
-\rho \frac{\partial^2}{\partial r_i \partial r_j}(v_i v_j)
-4\pi G \rho^2\end{aligned}$$

or in vector notation

$$\begin{aligned}
\Delta p = 
-\rho \nabla \left[ \nabla\cdot(v_i v_j) \right] 
- 4\pi G \rho^2\end{aligned}$$

Actually the divergence equation for an incompressible newtonian fluid
has a very interesting form. It is show in the Appendix <a href="#diveq" data-reference-type="ref" data-reference="diveq">2.18.3</a> that the
divergence equation might be related to Bernoulli’s law and even to the
Einstein equation of general relativity.

#### Vorticity equation

Inserting equation <a href="#eq:nicdivstress" data-reference-type="eqref" data-reference="eq:nicdivstress">[eq:nicdivstress]</a> into <a href="#eq:icvort" data-reference-type="ref" data-reference="eq:icvort">[eq:icvort]</a>
we get the vorticity equation for an incompressible newtonian fluid

$$\begin{aligned}
\frac{\partial}{\partial t}\omega_g
-\epsilon_{ghi}\frac{\partial}{\partial r_h} \epsilon_{ijm} v_j \omega_m =
\frac{\eta}{\rho}\frac{\partial^2}{\partial r_j^2}\omega_g\end{aligned}$$

or in vector notation

$$\begin{aligned}
\frac{\partial}{\partial t} \boldsymbol{\omega}-\nabla \times (\boldsymbol{v} \times \boldsymbol{\omega}) = 
\frac{\eta}{\rho} \Delta \boldsymbol{\omega}.\end{aligned}$$

#### Summary

##### Balance equations [balance-equations-7]

$$\begin{aligned}
\frac{\partial v_j}{\partial r_j} =&\ 0\\
\frac{\partial v_i}{\partial t} + v_j \frac{\partial v_i}{\partial r_j} =& -\frac{1}{\rho}\frac{\partial}{\partial r_i}p + g_i
+\nu\frac{\partial^2}{\partial r_j^2}v_i\\
\frac{\partial e}{\partial t} + v_j \frac{\partial e}{\partial r_j} =& -\frac{1}{\rho} \frac{\partial}{\partial r_j}(v_j p) + v_i g_i
+\nu \frac{\partial^2}{\partial r_j^2} \left( \frac{1}{2}v_i^2 \right) + \nu \frac{\partial}{\partial r_j} \left( v_i\frac{\partial v_j}{\partial r_i} \right) \end{aligned}$$

with Newtonian gravity (Poisson Equation):

$$\begin{aligned}
\frac{\partial}{\partial r_j}g_j=4\pi G \rho\end{aligned}$$

and as equation of state the "divergence equation".

##### Global dissipation of kinetic energy $$\mathcal{E}$$ [^23] [global-dissipation-of-kinetic-energy-mathcale-4]

$$\begin{aligned}
\mathcal{E} = -\frac{\eta}{V} \int_V  \omega^2 dV
\label{eq:nicdiss}\end{aligned}$$

##### Divergence equation (Equation of State) [divergence-equation-equation-of-state-1]

$$\begin{aligned}
\frac{\partial^2}{\partial r_i^2}p= 
-\rho \frac{\partial^2}{\partial r_i \partial r_j}(v_i v_j)
-4\pi G \rho^2\end{aligned}$$

##### Vorticity equation [vorticity-equation-7]

$$\begin{aligned}
\frac{\partial}{\partial t}\omega_g
-\epsilon_{ghi}\frac{\partial}{\partial r_h} \epsilon_{ijm} v_j \omega_m =
\frac{\eta}{\rho}\frac{\partial^2}{\partial r_j^2}\omega_g\end{aligned}$$

### Fluid dynamics in comoving coordinates

#### Introduction

On large scales ($$>$$ 100Mpc) the distribution of matter
in the universe is isotropic (it looks the same in all directions)
and homogeneous (it is isotropic at each point). But only the space
is assumed to be isotropic and homogenous. The observed expansion of
the universe singles out a special direction in time.[^24]

The physical distance on large scales[^25]
between two points in such an expanding universe varies with time like

$$\begin{aligned}
r_i=a(t) x_i\end{aligned}$$

The factor $$a$$ is a dimensionless scale factor greater than zero, which must
be the same for each component of the distance vector because of the assumed isotropy.
The scale factor can only depend on the time $$t$$ and not on the position
$$x_i$$ because of the assumed homogenity of space.

The change of the distance with time in an expanding universe is then

$$\begin{aligned}
\dot{r}_i = \dot{a} x_i + a \dot{x}_i  \end{aligned}$$

The global velocity of a particle $$v_i = \dot{r}_i$$ which does not move
relative to the expanding space ($$\dot{x}_i = 0$$) is then

$$\begin{aligned}
\dot{v}_i = \dot{a} x_i = \frac{\dot{a}}{a}r_i = H(t) r_i \end{aligned}$$

where $$H$$ is the so called Hubble parameter. Is a particle moving relative to
the expanding space ($$\dot{x}_i \neq 0$$) then we measure the additional
local (also called proper) velocity $$u_i = a \dot{x}_i$$ of the particle. This local
velocity can, according to special relativity, be never greater than the speed of light
$$c$$. Nevertheless the global velocity (e.g. the measured escape velocities of galaxies at
great distances) can be greater than $$c$$ (Davis and Lineweaver 2004). Generally the physical velocity
of a particle is the the sum of global and local velocity

$$\begin{aligned}
v_i = \dot{a} x_i + u_i (x_i,t)\end{aligned}$$

##### Useful transformations

$$\begin{aligned}
r_i&=a(t) x_i\\
v_i&= a \dot{x}_i + \dot{a} x_i = u_i + \dot{a} x_i\\
\frac{\partial}{\partial r_i}&=\frac{1}{a}\frac{\partial}{\partial x_i}\\
\frac{\partial}{\partial r_i}v_i&=\frac{1}{a}\frac{\partial}{\partial x_i}u_i+3\frac{\dot{a}}{a}\label{eq:cotrans4}\\
\frac{\partial}{\partial r_i}v_j&=\frac{1}{a}\frac{\partial}{\partial x_i}u_j + \frac{\dot{a}}{a}\delta_{ij} \label{eq:cotrans5}\\
 \left( \frac{\partial A}{\partial t} \right) _r+v_j \frac{\partial A}{\partial r_j} &= 
 \left( \frac{\partial A}{\partial t} \right) _x+\frac{1}{a}u_j \frac{\partial A}{\partial
x_j}\\
A(r_i,v_i,t) &\neq A(x_i,u_i,t)\end{aligned}$$

The stress tensor for a newtonian fluid in comoving coordinates is[^26]

$$\begin{aligned}
\sigma'_{ij}=2\eta T^*_{ij}+\zeta\delta_{ij}\frac{1}{a} \left( \frac{\partial}{\partial x_k}u_k+n\dot{a} \right) \end{aligned}$$

with

$$\begin{aligned}
T^*_{ij}=\frac{1}{a} \left[ \frac{1}{2} \left( \frac{\partial}{\partial x_j}u_i+\frac{\partial}{\partial x_i}u_j \right) 
-\frac{1}{n}\delta_{ij}\frac{\partial}{\partial x_k}u_k \right] \end{aligned}$$

##### Transformed equations

$$\begin{aligned}
\frac{\partial}{\partial t}\rho + \frac{1}{a}\frac{\partial}{\partial x_j}(u_j \rho) =& -3\frac{\dot{a}}{a}\rho  \\
\frac{\partial}{\partial t}(\rho u_i) + \frac{1}{a}\frac{\partial}{\partial x_j}(u_j \rho u_i) =& 
-\frac{1}{a}\frac{\partial}{\partial x_i}p + \frac{1}{a}\frac{\partial}{\partial x_j}\sigma'_{ij} +\rho g^*_i -4\frac{\dot{a}}{a}\rho u_i
\label{eq:commom}
\\
\begin{split}
\frac{\partial}{\partial t}(\rho e) + \frac{1}{a}\frac{\partial}{\partial x_j}(u_j \rho e) =& 
-\frac{1}{a}\frac{\partial}{\partial x_j}(u_j p) + \frac{1}{a}\frac{\partial}{\partial x_j}(u_i \sigma'_{ij}) + \frac{1}{a}u_i \rho g^*_i \\ 
&- 3 \frac{\dot{a}}{a}(\rho e +\frac{1}{3}\rho u^2_i+p)\label{eq:cometot}
\end{split}\end{aligned}$$

with

-   Newtonian Gravity in comoving coordinates (Poisson Equation):

    $$\frac{1}{a}\frac{\partial}{\partial x_j}g^*_j=4\pi G$$
    with
    $$g_i^*=\frac{1}{a}\frac{\partial \varphi}{\partial x_i}$$ and
    $$\varphi=\phi+\frac{1}{2}a\ddot{a}x_i^2$$

The energy equation is the sum of the equation for the kinetic energy and the
internal energy

$$\begin{aligned}
\frac{\partial}{\partial t}(\rho e_k)+\frac{1}{a}\frac{\partial}{\partial x_j}(u_j \rho e_k) &= -\frac{1}{a}u_i \frac{\partial}{\partial x_i}p
+\frac{1}{a}u_i \frac{\partial}{\partial x_j}\sigma'_{ij}-\frac{1}{a}\rho u_i g^*_i-5\frac{\dot{a}}{a}\rho e_k\\
\frac{\partial}{\partial t}(\rho e_{int})+\frac{1}{a}\frac{\partial}{\partial x_j}(u_j \rho e_{int})&=
-\frac{1}{a}p \frac{\partial}{\partial x_j}u_j -\frac{1}{a}\sigma'_{ij}\frac{\partial}{\partial x_j}u_i- 3\frac{\dot{a}}{a}(\rho e_{int} +p)\end{aligned}$$

### Equations of State [eos]

#### Ideal gas

The ideal gas law (the thermic equation of state for an ideal gas)
is

$$\begin{aligned}
p V = N k_B T, \label{eq:igl}\end{aligned}$$

where $$N$$ is the number of molecules of the gas, $$k_B$$ is the Boltzmann
constant and $$p$$, $$V$$, $$T$$ are the state variables pressure, volume and
temperature respectivly. We can rewrite this equation also in the following
ways

$$\begin{aligned}
p V &= \frac{N}{N_A} N_A k_B  T \\ 
	 &= n R T \\
    &= \frac{m}{M} R T \\
\Rightarrow p &= \frac{m}{V} \frac{R}{M} T \\
              &= \rho R_S T\end{aligned}$$

where we used the number of moles $$n=\frac{N}{N_A}$$,
the molar mass of a gas molecule $$M = \frac{m}{n}$$ ($$m$$ is the
mass of the gas molecule), the universal gas constant $$R= N_A k_B$$ and the
specific gas constant $$R_S=\frac{R}{M}$$.

From statistical thermodynamics it follows that the
internal energy $$U$$ of an ideal gas is related to the temperature via the
so called caloric equation of state

$$\begin{aligned}
U = \frac{f}{2}N k_B T. \label{eq:caleos}\end{aligned}$$

$$f$$ specifies the number of degrees of freedom of a molecule (3 for a
one-atomic gas, 5 for a two-atomic gas and 6 for a nonlinear many-atomic
gas)[^27].
The adiabatic coefficient $$\gamma$$ is related to $$f$$ via

$$\begin{aligned}
\gamma = \frac{c_p}{c_v} = 1 + \frac{2}{f} \label{eq:gam}\end{aligned}$$

where $$\frac{c_p}{c_v}$$ is the ratio of the specific
heat capacity at constant pressure $$c_p$$ to the specific heat capacity at
constant volume $$c_v$$ of an ideal gas.[^28]

Inserting <a href="#eq:igl" data-reference-type="eqref" data-reference="eq:igl">[eq:igl]</a> and <a href="#eq:gam" data-reference-type="eqref" data-reference="eq:gam">[eq:gam]</a> into <a href="#eq:caleos" data-reference-type="eqref" data-reference="eq:caleos">[eq:caleos]</a> we get for
the pressure of an ideal gas

$$\begin{aligned}
p=(\gamma-1) \rho u \label{eq:igpress}\end{aligned}$$

where $$u$$ is the specific internal energy. The specific internal energy can be
computed from the total energy [^29] via $$u=e-\frac{1}{2}v^2$$ and so we get for the
pressure

$$\begin{aligned}
p= \left( \gamma-1 \right)  \rho  \left( e-\frac{1}{2}v^2 \right) \end{aligned}$$

#### Van-der-Waals-Gas

...to be completed

#### Relativistic Gas

...to be completed

### The speed of sound

The speed of sound is defined as the speed of propagation of small pertubations
in density and pressure in a fluid in a compressible fluid. For the derivation
of the sound speed it is also assumed that the fluid is homogeneous, i.e. the
average value of density $$\rho_0$$ and pressure $$p_0$$ are spatially (and
temporarily ?) constant. Then we can split the local value of density and
pressure in a spatially constant average value and a small spatially an
temporarily varying pertubation $$\rho'$$ and $$p'$$ respectively

$$\begin{aligned}
\rho &= \rho_0 + \rho'(x,t),\ \rho' \ll \rho_0 \\
p &= p_0 + p'(x,t),\ p' \ll p_0\end{aligned}$$

For an ideal fluid a temporal change in density or pressure has to adiabatic,
i.e. with constant entropy. The derivation of the equation of motion for a small
pertubation in density and pressure then leads us to a wave equation with a
propagation speed

$$\begin{aligned}
c=\sqrt{ \left( \frac{\partial p_0}{\partial \rho_0} \right) _S}\end{aligned}$$

which is called the adiabatic speed of sound.

Further on for an adiabatic change of state it applies

$$\begin{aligned}
p V^\gamma &= \text{const.} = C \ \ |\ \cdot\frac{1}{M^\gamma}\\
p  \left( \frac{V}{M} \right) ^\gamma &= \frac{C}{M^\gamma}\\
p  \left( \frac{1}{\rho} \right) ^\gamma &= C^*\\
\Longrightarrow 
p &= C^* \rho^\gamma \Longleftrightarrow \frac{p}{\rho} = C^* \rho^{\gamma-1}\end{aligned}$$

Using this we get

$$\begin{aligned}
\frac{\partial p_0}{\partial \rho_0} = \gamma C^* \rho_0^{\gamma-1} 
= \gamma \frac{p_0}{\rho_0}.\end{aligned}$$

With this result the adiabatic speed of sound for an ideal, compressible,
homogeneous fluid yields

$$\begin{aligned}
c=\sqrt{\gamma \frac{p_0}{\rho_0}}\end{aligned}$$

Using equation <a href="#eq:igpress" data-reference-type="eqref" data-reference="eq:igpress">[eq:igpress]</a> we can compute the sound speed for an ideal gas
directly from the internal energy

$$\begin{aligned}
c=\sqrt{\gamma (\gamma-1) u_0}\end{aligned}$$

#### Machnumber

The Machnumber of a flow is defined as

$$\begin{aligned}
\text{Ma}=\frac{v}{c}\end{aligned}$$

The Machnumber can also be interpreted as the ratio of the kinetic and internal
energy, because

$$\begin{aligned}
\text{Ma}^2=\frac{v^2}{\gamma(\gamma-1) u}=\frac{1}{\gamma(\gamma-1)}\frac{e_{kin}}{e_{int}}\end{aligned}$$

So, the Machnumber $$Ma = x \cdot \sqrt{e_{kin}/e_{int}}$$ where $$x$$ for some
values of $$\gamma$$ can be found in the following table:

<div class="center" markdown="1">

|   $$\gamma$$    | $$\gamma(\gamma-1)$$ | $$x =  \left[ \gamma(\gamma-1) \right] ^{-1/2}$$ |
|:---------------:|:--------------------:|:------------------------------------------------:|
| $$\frac{5}{3}$$ |   $$\frac{10}{9}$$   |       $$\frac{3}{\sqrt{10}}\approx 0.949$$       |
| $$\frac{7}{5}$$ |  $$\frac{14}{25}$$   |       $$\frac{5}{\sqrt{14}}\approx 1.336$$       |
| $$\frac{4}{3}$$ |   $$\frac{4}{9}$$    |              $$\frac{3}{2}$$ = 1.5               |

</div>

### Dimensional analysis

We want to write down the general equations of fluid dynamics in dimensionless
form. Therefore we introduce the following dimensionless quantities

$$\begin{aligned}
r_i^*=\frac{r_i}{l_0} &\Rightarrow
\frac{\partial}{\partial r_i}=\frac{\partial}{\partial r_i^*}\frac{\partial r_i^*}{\partial r_i}=\frac{1}{l_0}\frac{\partial}{\partial r_i^*}\\
v_i^*=\frac{v_i}{v_0}\\
t^*=\frac{t}{l_0} &\Rightarrow
\frac{\partial}{\partial t}=\frac{\partial}{\partial t^*}\frac{\partial t^*}{\partial t}=\frac{1}{t_0}\frac{\partial}{\partial t^*}\\
\rho^*=\frac{\rho}{\rho_0}\end{aligned}$$

Inserting these into the continuity equation <a href="#eq:mass" data-reference-type="eqref" data-reference="eq:mass">[eq:mass]</a> we get

$$\begin{aligned}
\frac{\rho_0}{t_0}\frac{\partial}{\partial t^*}\rho^* 
+ \frac{v_0 \rho_0}{l_0} \frac{\partial}{\partial r_j^*}(v_j^*\rho^*) &= 0 \ | 
\cdot \frac{l_0}{\rho_0 v_0}\\
\underbrace{\frac{l_0}{v_0 t_0}}_{Sr}\frac{\partial}{\partial t^*}\rho^* 
+ \frac{\partial}{\partial r_j^*}(v_j^*\rho^*) = 0\end{aligned}$$

This derivation shows, that solutions of the continuity equation
are similar, if the Strouhal number $$Sr = \frac{l_0}{v_0 t_0}$$ is the
same. Flows with a Strouhal number $$Sr=0$$, are so called stationary flows.
Nevertheless the Strouhal number is most often set to one, by assuming
$$v_o=\frac{l_0}{t_0}$$.
Using the additional dimensionless quantities
$$p^*=\frac{p}{p_0},\sigma_{ij}^*=\frac{\sigma_{ij}}{\sigma_0},
g^*=\frac{g}{g_0 }$$
in the momentum equation <a href="#eq:mom" data-reference-type="eqref" data-reference="eq:mom">[eq:mom]</a> yields

$$\begin{aligned}
\frac{\rho_0 v_0}{t_0}\frac{\partial}{\partial t^*}(\rho^* v_i^*) 
+ \frac{\rho_0 v_0^2}{l_0}\frac{\partial}{\partial r_j^*}(v_j^* \rho^* v_i^*) &= 
- \frac{p_0}{l_0}\frac{\partial}{\partial r_i^*}p^* 
+\frac{\sigma_0}{l_0}\frac{\partial}{\partial r_j^*}\sigma_{ij}^*
+\rho_0 g_0 \rho^* g_i^*\ | \cdot \frac{l_0}{\rho_0 v_0^2}\\
\underbrace{\frac{l_0}{v_0 t_0}}_{Sr}\frac{\partial}{\partial t^*}(\rho^* v_i^*) 
+ \frac{\partial}{\partial r_j^*}(v_j^* \rho^* v_i^*) &= 
-\underbrace{\frac{p_0}{\rho_0 v_0^2}}_{Ma^{-2}_{iso}}\frac{\partial}{\partial r_i^*}p^* 
+\underbrace{\frac{\sigma_0}{\rho_0 v_0^2}}_{Re^{-1}}\frac{\partial}{\partial r_j^*}\sigma_{ij}^*
+\underbrace{\frac{\rho_0 g_0 l_0}{\rho_0 v_0^2}}_{Fr^{-1}} \rho^* g_i^*\end{aligned}$$

The occuring dimesionless numbers are the isothermal Mach number $$Ma_{iso}$$,
which is related to the Euler number $$Eu$$ or the Ruark number $$Ru$$ like
$$Ma^2_{iso}=Eu=Ru^{-1}$$, the Froude number $$Fr$$, which is related to the
Richardson number $$Ri$$ like $$Fr=Ri^{-1}$$ and the Reynolds number $$Re$$.
All these numbers measure the importance of the term they are related to
compared to the nonlinear advection term $$\frac{\partial}{\partial r_j^*}(v_j^* \rho^* v_i^*)$$,
e.g. for high Mach numbers the pressure term $$\frac{\partial}{\partial r_i^*}p^*$$ becomes less and
less important compared to the advection term; for high Reynolds numbers the
stress term $$\frac{\partial}{\partial r_j^*}\sigma_{ij}^*$$ becomes less and less important compared
to the advection term and the equation shows more and more nonlinear behaviour.
For a newtonian fluid with

$$\begin{aligned}
\sigma_{ij} \simeq \eta \frac{\partial v_i}{\partial r_j} = 
\frac{\eta_0 v_0}{l_0} \eta^* \frac{\partial v_i^*}{\partial r_j^*},\end{aligned}$$

where we introduced the dimensionless quantity $$\eta^*=\frac{\eta}{\eta_0}$$,
we get $$\sigma_0= \frac{\eta_0 v_0}{l_0}$$ and therefore we can express the
Reynolds number[^30] like

$$\begin{aligned}
Re=\frac{l_0 \rho_0 v_0^2 }{\eta_0 v_0} = \frac{\rho_0 l_0 v_0 }{\eta_0}\end{aligned}$$

Playing the same game with the equation for the internal energy <a href="#eq:eint" data-reference-type="ref" data-reference="eq:eint">[eq:eint]</a>
using $$e_{int}^*=\frac{e_{int}}{u_0}$$ we get

$$\begin{aligned}
\underbrace{\frac{l_0}{v_0 t_0}}_{Sr} \frac{\partial}{\partial t^*}\rho^* e_{int}^* 
+ \frac{\partial}{\partial r_j^*} v_j^* \rho^* e_{int}^* &= 
-\underbrace{\frac{p_0}{\rho_0 u_0}}_{Ga_1} p^*\frac{\partial}{\partial r_j^*}v_j^*
+\underbrace{\frac{\sigma_0}{\rho_0 u_0}}_{Ga_2} \sigma_{ij}^* \frac{\partial}{\partial r_j^*}v_i^*\end{aligned}$$

The new dimensionles quantities that occur in the energy equation seem to have
no name in the literature, but we will call them "Gamma1" (Ga1) and "Gamma2"
(Ga2) for now, since they are related to the adiabatic coefficient.
This can be seen by replacing $$p_0 p^*$$ according to equation

$$\begin{aligned}
p_0 p^* = (\gamma-1) \rho_0 u_0 \rho^* e_{int}^*\end{aligned}$$

which is valid for an an ideal, nonisothermal ($$\gamma \neq 1$$) gas. Doing this
we get[^31]

$$\begin{aligned}
 Sr \cdot \frac{\partial}{\partial t^*}\rho^* e_{int}^* 
+ \frac{\partial}{\partial r_j^*} v_j^* \rho^* e_{int}^* &= 
-(\gamma-1) \rho^* e_{int}^* \frac{\partial}{\partial r_j^*}v_j^*
+Ga_2 \cdot \sigma_{ij}^* \frac{\partial}{\partial r_j^*}v_i^*\end{aligned}$$

For a selfgravitating fluid we even have on more dimensionless quantity, which
appears, when we write down the dimesionless form of the poisson equation of
gravity

$$\begin{aligned}
\underbrace{\frac{g_0}{4\pi G \rho_0 l_0}}_{C_G}\frac{\partial g_i^*}{\partial r_i^*} = \rho^*.\end{aligned}$$

But this quantity $$C_G$$ also seems to have no name in the literature
(e.g. Durst 2007).

### Color fields

There are two possibilities to implement a color field $$c$$ in a fluid code. One
can treat color like the density, i.e. the color variable obeys a conservation
law like the density

$$\begin{aligned}
\frac{\partial}{\partial t} c + \frac{\partial}{\partial r_j}(v_j c) &= 0.\end{aligned}$$

In this case $$c$$ will exactly behave like density if density and color have the
same initial value.

On the other hand one can treat it like a specific quantity obeying a
conservation law like

$$\begin{aligned}
\frac{\partial}{\partial t} \rho c + \frac{\partial}{\partial r_j}(v_j \rho c) &= 0.\end{aligned}$$

In this case we see, that if $$c$$ is spatially constant at a time $$t_0$$, i.e.
$$c(t_0)=c_0$$, $$\frac{\partial c(t_0)}{\partial r_j}=0$$, it will stay constant forever

$$\begin{aligned}
\frac{\partial}{\partial t} \rho c + \frac{\partial}{\partial r_j}(v_j \rho c) &= 0 \\
\Leftrightarrow \frac{\partial c}{\partial t} + v_j \frac{\partial c}{\partial r_j} &= 0 \\
\Leftrightarrow \frac{d}{d t} c &=0\\
\Rightarrow c &= const. = c_0\end{aligned}$$

### Fluid dynamics in one dimension

#### Balance equations in Eulerian form

In one dimension all vector and tensor quantities from the balance equations
<a href="#eq:mass" data-reference-type="eqref" data-reference="eq:mass">[eq:mass]</a>-<a href="#eq:etot" data-reference-type="eqref" data-reference="eq:etot">[eq:etot]</a> degenerate to scalar quantities, eg.

$$\begin{aligned}
v_i,v_j &\longrightarrow v_1=v\\
\sigma'_{ij} &\longrightarrow \sigma'_{11}=\sigma'\\
g_i &\longrightarrow g_1=g\end{aligned}$$

The balance equations of fluid dynamics in one dimensions are therefore written
like:

$$\begin{aligned}
\frac{\partial}{\partial t}\rho + \frac{\partial}{\partial r}(v \rho) &= 0 \label{eq:1dmass}\\
\frac{\partial}{\partial t}(\rho v) + \frac{\partial}{\partial r}(v \rho v) &= -\frac{\partial}{\partial r}p + \frac{\partial}{\partial r}\sigma' -\rho g
\label{eq:1dmom} \\
\frac{\partial}{\partial t}(\rho e) + \frac{\partial}{\partial r}(v \rho e) &= -\frac{\partial}{\partial r}(v p) + \frac{\partial}{\partial r}(v \sigma') - v \rho g
\label{eq:1detot}\end{aligned}$$

#### Balance equations in Lagrangian form

By using the Euler derivate we can write <a href="#eq:1dmass" data-reference-type="eqref" data-reference="eq:1dmass">[eq:1dmass]</a>-<a href="#eq:1detot" data-reference-type="eqref" data-reference="eq:1detot">[eq:1detot]</a>
like

$$\begin{aligned}
\frac{d}{d t}\rho + \rho \frac{\partial}{\partial r} v &= 0 \\
\frac{d}{d t}(\rho v) + \rho v \frac{\partial}{\partial r} v &= -\frac{\partial}{\partial r}p + \frac{\partial}{\partial r}\sigma' -\rho g  \\
\frac{d}{d t}(\rho e) + \rho e \frac{\partial}{\partial r} v &= -\frac{\partial}{\partial r}(v p) + \frac{\partial}{\partial r}(v \sigma') - v \rho g\end{aligned}$$

If we introduce the so called mass coordinate defined by

$$\begin{aligned}
\partial m = \rho\cdot\partial r \Rightarrow \partial r = \frac{\partial
m}{\rho}\end{aligned}$$

we can write <a href="#eq:1dmass" data-reference-type="eqref" data-reference="eq:1dmass">[eq:1dmass]</a> like

$$\begin{aligned}
&&\frac{d}{d t}\rho + \rho^2 \frac{\partial}{\partial m} v &= 0 \\
\Leftrightarrow&& -\frac{1}{\rho^2}\frac{d}{d t}\rho - \frac{\partial}{\partial m} v &= 0 \\
\Leftrightarrow&& \frac{d}{d t} \left( \frac{1}{\rho} \right)  &= \frac{\partial}{\partial m} v\end{aligned}$$

The momentum equation <a href="#eq:1dmom" data-reference-type="eqref" data-reference="eq:1dmom">[eq:1dmom]</a> is transformed like

$$\begin{aligned}
&& \frac{d}{d t}(\rho v) + \rho^2 v \frac{\partial}{\partial m} v &= -\rho \frac{\partial}{\partial m}p + \rho \frac{\partial}{\partial m}\sigma'
-\rho g\\
\Leftrightarrow&& \frac{1}{\rho} \left( \rho \frac{d}{d t}v + v\frac{d}{d t}\rho \right)  
+ \rho v \frac{\partial}{\partial m}v &= -\frac{\partial}{\partial m}p + \frac{\partial}{\partial m}\sigma' - g \\
\Leftrightarrow&& \frac{d}{d t}v + v  \left( \frac{1}{\rho}\frac{d}{d t}\rho+\rho \frac{\partial}{\partial m} v \right)  &=
-\frac{\partial}{\partial m}p + \frac{\partial}{\partial m}\sigma' - g \\
\Leftrightarrow&& \frac{d}{d t}v - \rho v
\underbrace{ \left( -\frac{1}{\rho^2}\frac{d}{d t}\rho-\frac{\partial}{\partial m} v \right) }_{0} &=
-\frac{\partial}{\partial m}p + \frac{\partial}{\partial m}\sigma' - g\\
\Leftrightarrow&& \frac{d}{d t}v &= -\frac{\partial}{\partial m}p + \frac{\partial}{\partial m}\sigma' - g\end{aligned}$$

With an analogous transformation the energy equation <a href="#eq:1detot" data-reference-type="eqref" data-reference="eq:1detot">[eq:1detot]</a> can be
written like

$$\begin{aligned}
\frac{d}{d t}e &= -\frac{\partial}{\partial m}(vp) + \frac{\partial}{\partial m}(v\sigma') - vg\end{aligned}$$

Summing up the three equations an retransforming them in terms of $$dr$$ we get
the balance equations in one-dimensional lagrangian form as they are often used
in numerical fluid dynamics:

$$\begin{aligned}
\frac{d}{d t} \left( \frac{1}{\rho} \right)  &= \frac{1}{\rho}\frac{\partial}{\partial r} v \\
\frac{d}{d t}v &= -\frac{1}{\rho}\frac{\partial}{\partial r}p + \frac{1}{\rho}\frac{\partial}{\partial r}\sigma' - g\\
\frac{d}{d t}e &= -\frac{1}{\rho}\frac{\partial}{\partial r}(vp) + \frac{1}{\rho}\frac{\partial}{\partial r}(v\sigma') - vg\end{aligned}$$

### Rate of strain tensor, rotation tensor [rotstraintensor]

The decomposition of the jacobian of the velocity field into a symmetric and
an antisymmetric part yields

$$\begin{aligned}
\frac{\partial v_i}{\partial r_j}&=
\underbrace{\frac{1}{2} \left( \frac{\partial v_i}{\partial r_j}+\frac{\partial v_j}{\partial r_i} \right) }_{S_{ij}=S_{ji}}
+\underbrace{\frac{1}{2} \left( \frac{\partial v_i}{\partial r_j}-\frac{\partial v_j}{\partial r_i} \right) }_{R_{ij}=-R_{ji}}\end{aligned}$$

The symmetric part $$S_{ij}$$ is called *rate of strain tensor* and the
antisymmetric part is called *rotation tensor*. $$R_{ij}$$ has only three
independent components and can be expressed in terms of a (pseudo-)
3-vector.[^32]
This vector is equivalent to the negative curl of the velocity field
$$\boldsymbol{\omega}=\nabla \times \boldsymbol{v}$$ as we can see by multiplying $$R_{jk}$$ with
$$\epsilon_{ijk}$$

$$\begin{aligned}
\frac{1}{2}\epsilon_{ijk}R_{jk}&=\frac{1}{2}\epsilon_{ijk} \left( \frac{\partial v_j}{\partial r_k}
-\frac{\partial  v_k}{\partial  r_j } \right) =
\frac{1}{2}\epsilon_{ijk}\frac{\partial v_j}{\partial r_k}-\frac{1}{2}\epsilon_{ijk}\frac{\partial v_k}{\partial r_j}
\\
&=
\frac{1}{2}\epsilon_{ijk}\frac{\partial v_j}{\partial r_k}-\frac{1}{2}\epsilon_{ikj}\frac{\partial v_j}{\partial r_k}
=
\frac{1}{2}\epsilon_{ijk}\frac{\partial v_j}{\partial r_k}+\frac{1}{2}\epsilon_{ijk}\frac{\partial v_j}{\partial r_k}
\\
&=
\epsilon_{ijk}\frac{\partial v_j}{\partial r_k}
=\epsilon_{ikj}\frac{\partial v_k}{\partial r_j}
=-\epsilon_{ijk}\frac{\partial v_k}{\partial r_j}\\
&=-\omega_i\end{aligned}$$

Further one can show that

$$\begin{aligned}
-\frac{1}{2}\epsilon_{ijk}\omega_k
&=-\frac{1}{2}\epsilon_{ijk}\epsilon_{klm}\frac{\partial v_m}{\partial r_l}
=-\frac{1}{2}\epsilon_{kij}\epsilon_{klm}\frac{\partial v_m}{\partial r_l}\\
&=-\frac{1}{2} \left( \delta_{il}\delta_{jm}-\delta_{im}\delta_{jl} \right) \frac{\partial v_m}{\partial r_l}
\\
&=-\frac{1}{2} \left( \frac{\partial v_j}{\partial r_i}-\frac{\partial v_i}{\partial r_j} \right) 
=\frac{1}{2} \left( \frac{\partial v_i}{\partial r_j}-\frac{\partial v_j}{\partial r_i} \right) \\
&=R_{ij}\end{aligned}$$

and

$$\begin{aligned}
R_{ij} R_{ij} 
&=-\frac{1}{2}\epsilon_{ijk}\omega_k \cdot -\frac{1}{2}\epsilon_{ijl}\omega_l\\
&=\frac{1}{4}\epsilon_{ijk}\epsilon_{ijl}\omega_k \omega_l
=\frac{1}{4} \cdot 2 \delta_{kl} \omega_k \omega_l\\
&=\frac{1}{2} \omega_k \omega_k
\label{eq:rrcontr}\end{aligned}$$

Additionally the following relation between the contraction of rate of strain
tensor and the contraction of the rotation tensor is useful

$$\begin{aligned}
\begin{split}
4 S_{ij}S_{ij} -  4 R_{ij}R_{ij} &=
 \left( \frac{\partial v_i}{\partial r_j}+\frac{\partial v_j}{\partial r_i} \right) ^2 -  \left( \frac{\partial v_i}{\partial r_j}-\frac{\partial v_j}{\partial r_i} \right) ^2\\ 
&= \left( \frac{\partial v_i}{\partial r_j} \right) ^2 +  \left( \frac{\partial v_j}{\partial r_i} \right) ^2 +
2\frac{\partial v_i}{\partial r_j}\frac{\partial v_j}{\partial r_i}
- \left( \frac{\partial v_i}{\partial r_j} \right) ^2 -  \left( \frac{\partial v_j}{\partial r_i} \right) ^2 +
2\frac{\partial v_i}{\partial r_j}\frac{\partial v_j}{\partial r_i}\\ 
&= 4 \frac{\partial v_i}{\partial r_j}\frac{\partial v_j}{\partial r_i}
\end{split}
\label{eq:rsrcontr}\end{aligned}$$

### Derivation of the stress tensor for a newtonian fluid

It is generally assumed, that friction between fluid elements is proportional
to the area of their surfaces. So in general the frictional or vicous force
on a fluid element can be expressed like

$$\begin{aligned}
F_{visc,i}= \oint_A \sigma'_{ij} n_j dA = \int_V \frac{\partial}{\partial r_j} \sigma'_{ij} dV.\end{aligned}$$

This force leads to an irreversible rise of temperature in the fluid or an irreversible
decrease of kinetic energy expressed by the equation for the dissipation

$$\begin{aligned}
\frac{\partial}{\partial t} E_{kin,visc} = - \int_V \sigma'_{ij} \frac{\partial}{\partial r_j} v_i dV\end{aligned}$$

For a motionless fluid ($$v_i=0$$) and for a fluid with constant velocity
($$\frac{\partial v_i}{\partial r_j} = 0$$) this integral is zero. But also a rotating observer
of a motionless fluid should not see a rise in the temperature of a fluid
[^33]
that means

$$\begin{aligned}
 \int_V \sigma'_{ij} \frac{\partial}{\partial r_j} v_i dV = 0 \label{eq:nodiss}\end{aligned}$$

A rotating observer of a motionless fluid sees a velocity field of the form

$$\begin{aligned}
v_i=\epsilon_{ijk}\omega_j r_k\end{aligned}$$

where $$\omega_j$$ is the angular velocity vector and $$r_k$$ is the position vector.
It can be shown, that for such a velocity field the jacobian is antisymmetric, that
means

$$\begin{aligned}
\frac{\partial v_i}{\partial r_j} = -\frac{\partial v_j}{\partial r_i}\end{aligned}$$

Using this and equation <a href="#eq:uascontr" data-reference-type="eqref" data-reference="eq:uascontr">[eq:uascontr]</a> in equation <a href="#eq:nodiss" data-reference-type="eqref" data-reference="eq:nodiss">[eq:nodiss]</a> we get

$$\begin{aligned}
\int_V \frac{1}{2} \left( \sigma'_{ij}-\sigma'_{ji} \right)  \frac{\partial}{\partial r_j} v_i dV = 0\end{aligned}$$

This relation can only be fulfilled if the stress tensor $$\sigma'_{ij}$$
is symmetric

$$\begin{aligned}
\sigma'_{ij} = \sigma'_{ji}\end{aligned}$$

For a newtonian fluid is it assumed that the stress tensor is proportional only to the
first derivatives of the velocity field. Together with the requirement of symmetry
the most general for of such a tensor is

$$\begin{aligned}
\sigma'_{ij} = a \left( \frac{\partial v_j}{\partial r_i}+\frac{\partial v_i}{\partial r_j} \right) +b \delta_{ij}\frac{\partial v_k}{\partial r_k}\end{aligned}$$

Usually the trace is split off the first term and added the second term so

$$\begin{aligned}
\sigma'_{ij} = a  \left( \frac{\partial v_j}{\partial r_i}+\frac{\partial v_i}{\partial r_j}
-\frac{2}{3}\delta_{ij}\frac{\partial v_k}{\partial r_k} \right) + \left( \frac{2 a}{3}+b \right) \delta_{ij}\frac{\partial v_k}{\partial r_k}\end{aligned}$$

Using the definitions $$2a = \eta$$ and $$\frac{2 a}{3}+b=\zeta$$ we get the form
most common in literature

$$\begin{aligned}
\sigma'_{ij} =  
\eta \left[ \frac{1}{2} \left( \frac{\partial v_i}{\partial r_j}+\frac{\partial v_j}{\partial r_i} \right) 
-\frac{1}{3}\delta_{ij}\frac{\partial v_k}{\partial r_k} \right] 
+\zeta \delta_{ij}\frac{\partial v_k}{\partial r_k}\end{aligned}$$

### Stress tensor in cartesian coordinates for 1d, 2d and 3d [stress1d2d3d]

For a so called newtonian fluid it can be shown, that the stress tensor
$$\sigma'_{ij}$$ in cartesian coordinates in $$n$$ dimensions is of the form

$$\begin{aligned}
\sigma'_{ij}=2\eta S^*_{ij}+\zeta\delta_{ij}\frac{\partial}{\partial r_k}v_k \end{aligned}$$

with $$S^*_{ij}$$ being the symmetric tracefree part of the tensor $$\frac{\partial}{\partial x_j}v_i$$

$$\begin{aligned}
S^*_{ij}=S_{ij}-\frac{1}{n}\delta_{ij}\frac{\partial}{\partial r_k}v_k\end{aligned}$$

and $$S_{ij}$$ being the so called rate-of-strain tensor

$$\begin{aligned}
\frac{1}{2} \left( \frac{\partial}{\partial r_j}v_i+\frac{\partial}{\partial r_i}v_j \right) \end{aligned}$$

In 1d cartesian coordinates the component of the
symmetric tracefree part of the stress tensor $$S^*_{ij}$$ is

$$\begin{aligned}
S_{xx}=\frac{1}{2} \left( \frac{\partial}{\partial r_x}v_x+\frac{\partial}{\partial r_x}v_x \right) -\frac{1}{1}\frac{\partial}{\partial r_x}v_x = 0.\end{aligned}$$

In 2d cartesian coordinates the components of $$S^*_{ij}$$ are

$$\begin{aligned}
S_{xx}&=\frac{1}{2} \left( \frac{\partial}{\partial r_x}v_x+\frac{\partial}{\partial r_x}v_x \right) 
		 -\frac{1}{2} \left( \frac{\partial}{\partial r_x}v_x+\frac{\partial}{\partial r_y}v_y \right) \\
		&=\frac{1}{2} \left( \frac{\partial}{\partial r_x}v_x-\frac{\partial}{\partial r_y}v_y \right) , \\
S_{xy}&=\frac{1}{2} \left( \frac{\partial}{\partial r_y}v_x+\frac{\partial}{\partial r_x}v_y \right) , \\
S_{yx}&=S_{xy},\\
S_{yy}&=\frac{1}{2} \left( \frac{\partial}{\partial r_y}v_y-\frac{\partial}{\partial r_x}v_x \right) .\end{aligned}$$

In 3d cartesian coordinates the components of $$S^*_{ij}$$ are

$$\begin{aligned}
S_{xx}&=\frac{1}{2} \left( \frac{\partial}{\partial r_x}v_x+\frac{\partial}{\partial r_x}v_x \right) 
		 -\frac{1}{3} \left( \frac{\partial}{\partial r_x}v_x+\frac{\partial}{\partial r_y}v_y+\frac{\partial}{\partial r_z}v_z \right) \\
		&=\frac{1}{3} \left( 2\frac{\partial}{\partial r_x}v_x-\frac{\partial}{\partial r_y}v_y-\frac{\partial}{\partial r_z}v_z \right) ,\\
S_{xy}&=\frac{1}{2} \left( \frac{\partial}{\partial r_y}v_x+\frac{\partial}{\partial r_x}v_y \right) ,\\
S_{xz}&=\frac{1}{2} \left( \frac{\partial}{\partial r_z}v_x+\frac{\partial}{\partial r_x}v_z \right) ,\\
S_{yx}&=S_{xy},\\
S_{yy}&=\frac{1}{3} \left( 2\frac{\partial}{\partial r_y}v_y-\frac{\partial}{\partial r_x}v_x-\frac{\partial}{\partial r_z}v_z \right) ,\\
S_{yz}&=\frac{1}{2} \left( \frac{\partial}{\partial r_z}v_y+\frac{\partial}{\partial r_y}v_z \right) ,\\
S_{zx}&=S_{xz},\\
S_{zy}&=S_{yz},\\
S_{zz}&=\frac{1}{3} \left( 2\frac{\partial}{\partial r_z}v_z-\frac{\partial}{\partial r_x}v_x-\frac{\partial}{\partial r_y}v_y \right) .\\\end{aligned}$$

### Stress tensor in comoving coordinates [costress]

With the help of transformation (<a href="#eq:cotrans4" data-reference-type="ref" data-reference="eq:cotrans4">[eq:cotrans4]</a>) and (<a href="#eq:cotrans5" data-reference-type="ref" data-reference="eq:cotrans5">[eq:cotrans5]</a>) we
can transform the stress tensor for a newtonian fluid in cartesian coordinates

$$\begin{aligned}
\sigma'_{ij}= 2\eta \left[ \frac{1}{2} \left( \frac{\partial}{\partial r_j}v_i+\frac{\partial}{\partial r_i}v_j \right) 
-\frac{1}{n}\delta_{ij}\frac{\partial}{\partial r_k}v_k \right] +\zeta\delta_{ij}\frac{\partial}{\partial r_k}v_k.\end{aligned}$$

into the stress tensor for a newtonian fluid in comoving coordinates

$$\begin{aligned}
\sigma'_{ij}=& 
2\eta \left[ \frac{1}{2} 
 \left( \frac{1}{a}\frac{\partial}{\partial x_j}u_i+\frac{\dot{a}}{a}\delta_{ij}+\frac{1}{a}\frac{\partial}{\partial x_i}u_j+\frac{\dot{a}}{a}\delta_{ji} \right)  
-\frac{1}{n}\delta_{ij} \left( \frac{1}{a}\frac{\partial}{\partial x_k}u_k+n\frac{\dot{a}}{a} \right)  \right] \\
&+\zeta\delta_{ij} \left( \frac{1}{a}\frac{\partial}{\partial x_k}u_k+n\frac{\dot{a}}{a} \right) \\
=&2\eta \left[ \frac{1}{2 a} 
 \left( \frac{\partial}{\partial x_j}u_i+\frac{\partial}{\partial x_i}u_j \right) +\frac{\dot{a}}{a}\delta_{ij}-\frac{n}{n}\frac{\dot{a}}{a}\delta_{ij}
-\frac{1}{n a}\delta_{ij}\frac{\partial}{\partial x_k}u_k \right] \\
&+\zeta\delta_{ij} \left( \frac{1}{a}\frac{\partial}{\partial x_k}u_k+n\frac{\dot{a}}{a} \right) \\
=&\frac{1}{a} \left\{ 2\eta \left[ \frac{1}{2} 
 \left( \frac{\partial}{\partial x_j}u_i+\frac{\partial}{\partial x_i}u_j \right) -\frac{1}{n}\delta_{ij}\frac{\partial}{\partial x_k}u_k \right] 
+\zeta\delta_{ij} \left( \frac{\partial}{\partial x_k}u_k+n\dot{a} \right)  \right\} .\end{aligned}$$

### A contradiction when computing the dissipation [contradiss]

The dissipation for an newtonian incompressible fluid is

$$\begin{aligned}
\mathcal{E}
=-\frac{1}{V} \int_V \frac{\eta}{2} \left( \frac{\partial v_i}{\partial r_j}+\frac{\partial v_j}{\partial r_i} \right) ^2 dV\end{aligned}$$

If we assume a flowfield with $$\frac{\partial v_x}{\partial r_x} \neq 0$$ (which causes
$$\frac{\partial v_y}{\partial r_y}+\frac{\partial v_z}{\partial r_z}=-\frac{\partial v_x}{\partial r_x}$$, because the divergence
of the velocity field must be zero) and vanishing diagonal components of
the jacobian
$$\frac{\partial v_y}{\partial r_x}=\frac{\partial v_x}{\partial r_y}=\frac{\partial v_z}{\partial r_y}=\frac{\partial v_y}{\partial r_z}=\frac{\partial v_x}{\partial r_z}=\frac{\partial v_z}{\partial r_x}=0$$
, we compute via the formula above a
dissipation of

$$\begin{aligned}
\mathcal{E}
=-\frac{2 \eta}{V} \int_V  \left( \frac{\partial v_x}{\partial r_x} \right) ^2 + \left( \frac{\partial v_y}{\partial r_y} \right) ^2 +
 \left( \frac{\partial v_z}{\partial r_z} \right) ^2 dV. \end{aligned}$$

The absolut value of the dissipation must be greater than zero, because
all the terms in the integral are quadratic and our assumption was
$$\frac{\partial v_x}{\partial r_x},\frac{\partial v_y}{\partial r_y},\frac{\partial v_z}{\partial r_z}  \neq 0$$.

But if we compute the dissipation according to equation <a href="#eq:nicdiss" data-reference-type="eqref" data-reference="eq:nicdiss">[eq:nicdiss]</a>
like

$$\begin{aligned}
\mathcal{E} = -\frac{\eta}{V} \int_V  \omega^2 dV\end{aligned}$$

we get for our flow field $$\mathcal{E} = 0$$, because the vorticity of
our velocity field is zero! This seems to be an obvious contradiction,
and either raises some doubts about the validity of the derivation of equation
<a href="#eq:nicdiss" data-reference-type="eqref" data-reference="eq:nicdiss">[eq:nicdiss]</a> for example in Frisch (1995) or implies that our
assumed flow field is unphysical.

### Structure functions

A structure function of order $$p$$ is defined as[^34]

$$\begin{aligned}
S_p(f(x))=\langle  \left[ f(x+x')-f(x') \right] ^p \rangle = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty} \left[ f(x+x')-f(x') \right] ^p
dx'\end{aligned}$$

The second order structure function is related to the spectrum
$$\left\lvert F(k)\right\rvert$$ of the function $$f$$ like

$$\begin{aligned}
\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty} \left[ f(x+x')-f(x') \right] ^2 dx' = \frac{2}{\sqrt{2\pi}}\int_{-\infty}^{\infty}(1-e^{ikx})
\left\lvert F(k)\right\rvert^2 dk \label{eq:structspec}\end{aligned}$$

which can be proved[^35] by expanding the second order structure function

$$\begin{aligned}
&S_2(f(x))=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty} \left[ f(x+x')-f(x') \right] ^2 dx'\\
&= \frac{1}{\sqrt{2\pi}} \left[ \int_{-\infty}^{\infty}\left\lvert f(x+x')\right\rvert^2 dx' 
- 2 \int_{-\infty}^{\infty}f^*(x')f(x+x')dx'
+ \int_{-\infty}^{\infty}\left\lvert f(x')\right\rvert^2 dx' \right] .\end{aligned}$$

Substituting $$x''=x+x'$$ in the first term we get

$$\begin{aligned}
S_2(f(x)) &= \frac{1}{\sqrt{2\pi}} \left[ \int_{-\infty}^{\infty}\left\lvert f(x'')\right\rvert^2 dx'' 
- 2\int_{-\infty}^{\infty}f^*(x')f(x+x')dx'
+\int_{-\infty}^{\infty}\left\lvert f(x')\right\rvert^2 dx' \right] \\
&= \frac{2}{\sqrt{2\pi}}  \left[ \int_{-\infty}^{\infty}\left\lvert f(x')\right\rvert^2 dx'-\int_{-\infty}^{\infty}f^*(x')f(x+x')dx' \right] .\end{aligned}$$

Using Parseval’s and the Wiener-Khichnin theorem we obtain the final result

$$\begin{aligned}
&S_2(f(x))=\frac{2}{\sqrt{2\pi}}  \left[ \int_{-\infty}^{\infty}\left\lvert F(k)\right\rvert^2 dk
-\int_{-\infty}^{\infty}\left\lvert F(k)\right\rvert^2 e^{ikx} dk \right]  \\
&= \frac{2}{\sqrt{2\pi}}\int_{-\infty}^{\infty}(1-e^{ikx})\left\lvert F(k)\right\rvert^2 dk. \end{aligned}$$

The structure functions used in the theory of Kolmogorov
are the so called longitudinal structure functions of the velocity, which are
defined like

$$\begin{aligned}
S_2(v_{\parallel}(l)) =  
\langle  \left(  \left[ \boldsymbol{v}(\boldsymbol{x}+\boldsymbol{l})-\boldsymbol{v}(\boldsymbol{x}) \right] \cdot
\frac{\boldsymbol{l}}{l} \right) ^p \rangle =
\langle  \left( v_{\parallel}(\boldsymbol{x}+\boldsymbol{l})-v_{\parallel}(\boldsymbol{x}) \right) ^p \rangle\end{aligned}$$

They are related to the longitudinal velocity
spectrum[^36] $$\left\lvert V_{\parallel}(k)\right\rvert^2$$ via equation
<a href="#eq:structspec" data-reference-type="eqref" data-reference="eq:structspec">[eq:structspec]</a>.
Sometimes also second order transvers structure functions are measured.
These are defined as

$$\begin{aligned}
S_2(v_{\perp}(l)) =  
\langle  \left( \frac{\left\lvert \left[ \boldsymbol{v}(\boldsymbol{x}+\boldsymbol{l})-\boldsymbol{v}(\boldsymbol{x}) \right] \times
\boldsymbol{l}\right\rvert}{l} \right) ^p \rangle.\end{aligned}$$

The behaviour of the second order transvers structure functions for
homogeneous turbulence is uniquely determined by the longitudinal structure
function (Pope 2000, 192, Eqs. (6.28)). They also show the
characteristic $$2/3$$-slope as predicted for the longitudinal structure functions
(Frisch 1995, 60).

In general structure functions of vectorial quantities like the velocity are
tensors, e.g. the general second order structure function of the velocity can
be defined as

$$\begin{aligned}
S_{ij}(\boldsymbol{x},\boldsymbol{l}) =  
\langle  \left[ v_i(\boldsymbol{x}+\boldsymbol{l})-v_i(\boldsymbol{x}) \right] 
 \left[ v_j(\boldsymbol{x}+\boldsymbol{l})-v_j(\boldsymbol{x}) \right]  \rangle\end{aligned}$$

But it can be shown, that for local isotropy only the longitudinal
structure function $$S_2(v_{\perp}(l))=S_{11}$$ and the transversal structure
$$S_2(v_{\perp}(l))=S_{22}=S_{33}$$ are unequal zero (Pope 2000). Since
the transvers structure function is determined by the longitudinal structure
function in case of local homogeneity, for homogeneous and isotropic
turbulence $$S_{ij}$$ is determined by the single scalar
function $$S_{11} = S_2(v_{\parallel}(l))$$ (Pope 2000).

The third order structure function used in Kolmogorov theory is
defined as

$$\begin{aligned}
S_{111}(\boldsymbol{x},\boldsymbol{l}) = \langle  \left[ v_1(\boldsymbol{x}+\boldsymbol{l})-v_1(\boldsymbol{x}) \right] ^3 \rangle\end{aligned}$$

which is often simply called $$S_3(v(l))$$. So the famous four-fifths law of
Kolmogorov is actually true only for one component of the third order
structure function tensor, but again for homogeneous and isotropic turbulence
the third order structure function tensor $$S_{ijk}$$ is uniquely determined by
the single scalar function $$S_{111}=S_3(v(l))$$.

### Some speculations about ...

#### Entropy [entro]

The second law of thermodynamics states that the total entropy of any isolated
thermodynamic system tends to increase over time, approaching a maximum value
(not an infinite value!). But the higher the entropy, the lower the free,
useable energy of a system. Therefore it is perhabs easier to formulate the
second law of thermodynamics the other way round (Feynman 1967): the free,
useable energy of any isolated thermodynamic system tends to decrease over time,
approaching a minimum value (zero?!). In this sense we can also understand
Penrose (Penrose 1989). He pointed out, that a closed self-gravitating
system will collapse to a black hole, which is the state with the maximum
entropy of the system. But this state is not a very unordered state
as one often imagines states with high entropy. But it is the state where no
free energy is left, the potential energy of the system is zero and so no
directed kinetic energy can be produced any more. Thats why the "ordered state"
of a black hole has the biggest entropy.

#### Newtonian gravity

Newtonan gravity is often expressed in form of the poisson equation

$$\begin{aligned}
\Delta \phi = 4\pi G\rho\end{aligned}$$

Nevertheless this is equivalent to
[^37]

$$\begin{aligned}
\nabla \cdot \boldsymbol{g} = - 4\pi G\rho \label{eq:maxgrav1} \\
\nabla \times (K\boldsymbol{g}) = \boldsymbol{0},\label{eq:maxgrav2}\end{aligned}$$

because $$\nabla \times (-\nabla \phi) = \boldsymbol{0}$$ and therefore we are able to
express the gravitational field as a gradient of a potential like
$$\boldsymbol{g}=-\nabla \phi$$. As we can see these equations are similar to Maxwells
equations for the electric field in the electrostatic case
($$\frac{\partial \boldsymbol{B}}{\partial t}=0$$)

$$\begin{aligned}
\nabla \cdot (\epsilon \boldsymbol{E}) = - \frac{1}{\epsilon_0} \rho \\
\nabla \times \boldsymbol{E} = \boldsymbol{0}\end{aligned}$$

and therefore we might call the equations <a href="#eq:maxgrav1" data-reference-type="eqref" data-reference="eq:maxgrav1">[eq:maxgrav1]</a> and
<a href="#eq:maxgrav2" data-reference-type="eqref" data-reference="eq:maxgrav2">[eq:maxgrav2]</a> Maxwells equations of gravity.

The following is just speculation:

One cannot derive the continuity equation from these equations

$$\begin{aligned}
\frac{\partial \rho}{\partial t} + \nabla \cdot  \left( \boldsymbol{v} \rho \right)  &= 0 \\
\Leftrightarrow 
-\frac{1}{4\pi G}\frac{\partial}{\partial t} \left( \nabla\cdot\boldsymbol{g} \right)  
+ \nabla \cdot  \left( \boldsymbol{v} \rho \right)  &= 0 \\
\Leftrightarrow
-\frac{1}{4\pi G}\nabla\cdot\dot{\boldsymbol{g}}
+\nabla \cdot  \left( \boldsymbol{v} \rho \right)  &= 0 \\
\Leftrightarrow 
\nabla \cdot  \left( \boldsymbol{v} \rho - \frac{1}{4\pi G} \dot{\boldsymbol{g}} \right)  &= 0\end{aligned}$$

But if we fixed equation <a href="#eq:maxgrav2" data-reference-type="eqref" data-reference="eq:maxgrav2">[eq:maxgrav2]</a> in analogy to Maxwell by writing
[^38]

$$\begin{aligned}
\nabla \times (K \boldsymbol{g}) = \boldsymbol{v} \rho - \frac{1}{4\pi G} \dot{\boldsymbol{g}},\end{aligned}$$

we could derive the continuity equation from our systems of equations like

$$\begin{aligned}
\nabla \cdot  \left( \nabla \times (K \boldsymbol{g}) \right)  &= \nabla \cdot  \left( \boldsymbol{v} \rho 
- \frac{1}{4\pi G} \dot{\boldsymbol{g}} \right) \\
\Leftrightarrow 
0 &= \nabla \cdot  \left( \boldsymbol{v} \rho \right)  
+ \frac{\partial}{\partial t} \left( -\frac{1}{4\pi G} \nabla \cdot \boldsymbol{g} \right)  \\
\Leftrightarrow
0 &= \nabla \cdot  \left( \boldsymbol{v} \rho \right)  + \frac{\partial}{\partial t} \rho\end{aligned}$$

So instead of <a href="#eq:maxgrav1" data-reference-type="eqref" data-reference="eq:maxgrav1">[eq:maxgrav1]</a> and <a href="#eq:maxgrav2" data-reference-type="eqref" data-reference="eq:maxgrav2">[eq:maxgrav2]</a> we could have

$$\begin{aligned}
\nabla \cdot \boldsymbol{g} = - 4\pi G\rho, \\
\nabla \times (K\boldsymbol{g}) = \boldsymbol{v} \rho - \frac{1}{4\pi G} \dot{\boldsymbol{g}}.\end{aligned}$$

In vacuum ($$\rho=0$$) we would have

$$\begin{aligned}
\nabla \cdot \boldsymbol{g} = 0, \\
\nabla \times (K\boldsymbol{g}) = - \frac{1}{4\pi G} \dot{\boldsymbol{g}}.\end{aligned}$$

From these vacuum equations we can try to derive a wave equation in analogy
to the wave equation for the electromagnetic field like

$$\begin{aligned}
\nabla \times  \left( \nabla \times (K\boldsymbol{g}) \right)  = 
- \frac{1}{4\pi G}  \left( \nabla \times \dot{\boldsymbol{g}} \right) = 
- \frac{1}{4\pi G} \frac{\partial}{\partial t}  \left( \nabla \times \boldsymbol{g} \right) =
+ \frac{1}{(4\pi G)^2 K} \frac{\partial^2}{\partial t^2} \boldsymbol{g}\end{aligned}$$

and

$$\begin{aligned}
\nabla \times  \left( \nabla \times (K \boldsymbol{g}) \right)  = 
K \nabla  \left( \nabla \cdot \boldsymbol{g} \right)  - K \Delta \boldsymbol{g}\end{aligned}$$

With $$\nabla \cdot \boldsymbol{g} = 0$$ it follows that

$$\begin{aligned}
\frac{\partial^2}{\partial t^2} \boldsymbol{g}+(4\pi G K)^2 \Delta \boldsymbol{g} = 0\end{aligned}$$

This could be interpreted as a wave equation with complex velocity
$$c=i \cdot 4\pi G K$$.

#### The divergence equation [diveq]

##### General fluid

We start with the momentum equation <a href="#eq:mom" data-reference-type="eqref" data-reference="eq:mom">[eq:mom]</a>

$$\begin{aligned}
\frac{\partial}{\partial t}(\rho v_i) + \frac{\partial}{\partial r_j}(v_j \rho v_i) &= -\frac{\partial}{\partial r_i}p + \frac{\partial}{\partial r_j}\sigma'_{ij}
-\rho \frac{\partial}{\partial r_i} \phi\end{aligned}$$

where we assumed $$g_i=-\frac{\partial}{\partial r_i} \phi$$. If we make the substitutions
$$\frac{\partial}{\partial r_i} p \rightarrow \frac{\partial}{\partial r_j} p \delta_{ij}$$ and
$$\frac{\partial}{\partial r_i} \phi \rightarrow \frac{\partial}{\partial r_j} \phi \delta_{ij}$$ we can write it in the
form

$$\begin{aligned}
\frac{\partial}{\partial t}(\rho v_i) + \frac{\partial}{\partial r_j}(v_j \rho v_i + p \delta_{ij} - \sigma'_{ij}) 
= -\rho \frac{\partial}{\partial r_j} \phi \delta_{ij}\end{aligned}$$

Taking the divergence of this equation we get

$$\begin{aligned}
\frac{\partial}{\partial t} \left[ \frac{\partial}{\partial r_i} (\rho v_i) \right]  
+ \frac{\partial^2}{\partial r_i \partial r_j}(v_j \rho v_i + p \delta_{ij} - \sigma'_{ij}) 
= -\frac{\partial}{\partial r_i} \left( \rho \frac{\partial}{\partial r_j} \phi \delta_{ij} \right) \end{aligned}$$

where we assumed that $$\frac{\partial}{\partial t}$$ and $$\frac{\partial}{\partial r_i}$$ commute. Using the continuity
equation <a href="#eq:mass" data-reference-type="ref" data-reference="eq:mass">[eq:mass]</a> we get a interesting form of the fluiddynamic
equations

$$\begin{aligned}
\frac{\partial^2}{\partial t^2}\rho 
- \frac{\partial^2}{\partial r_i \partial r_j}(v_j \rho v_i + p \delta_{ij} - \sigma'_{ij}) 
= +\frac{\partial}{\partial r_i} \left( \rho \frac{\partial}{\partial r_j} \phi \delta_{ij} \right) \label{eq:divbeauty}\end{aligned}$$

In case of no gravitation, the fluiddynamic equation can be written in a form
showing some similarity to a wave equation

$$\begin{aligned}
\frac{\partial^2}{\partial t^2}\rho 
- \frac{\partial^2}{\partial r_i \partial r_j}(v_j \rho v_i + p \delta_{ij} - \sigma'_{ij}) 
= 0\end{aligned}$$

But despite of its simple form, this equation hides an extreme complexity.

Neglecting the stress tensor $$\sigma'_{ij}$$ and solving for pressure this
equation is written like

$$\begin{aligned}
\frac{\partial^2}{\partial r_i^2} p = \frac{\partial^2}{\partial t^2}\rho- 
\frac{\partial^2}{\partial r_i \partial r_j}(\rho v_i v_j)\label{eq:instpres}\end{aligned}$$

and sometimes called equation for the instantaneous pressure.

##### Incompressible newtonian fluid

If we have a newtonian fluid
($$\sigma'_{ij}=\eta  \left( \frac{\partial v_i}{\partial r_j}+\frac{\partial v_j}{\partial r_i} \right)$$) with $$\rho=const$$ in
time and space (which leads to $$\frac{\partial v_i}{\partial r_i}=0$$ and also
$$\frac{\partial^2}{\partial r_i \partial r_j}\sigma'_{ij}=0$$)
equation <a href="#eq:divbeauty" data-reference-type="eqref" data-reference="eq:divbeauty">[eq:divbeauty]</a> simplifies to

$$\begin{aligned}
\frac{\partial^2}{\partial r_i \partial r_j}\phi = 
- \frac{1}{\rho} \frac{\partial^2}{\partial r_i \partial r_j}  \left( p \delta_{ij} + \rho v_i v_j \right) \end{aligned}$$

which can be intepreted as an equation for the gravitational potential of a
fluid with constant(!) density:

$$\begin{aligned}
\Delta \phi = - \frac{1}{\rho} \frac{\partial^2}{\partial r_i \partial r_j} 
 \left( p \delta_{ij} + \rho v_i v_j \right) \end{aligned}$$

So this equation seems to imply, that pressure and velocity stresses can be a
source of gravity not only in general relativity, but also in a newtonian
framework. We can exploit the relation to general relativity further by
introducing the Stress-Energy-Tensor $$T_{ij} = p \delta_{ij} + \rho v_i v_j$$

$$\begin{aligned}
\Delta \phi = - \frac{1}{\rho} \frac{\partial^2}{\partial r_i \partial r_j} T_{ij} \label{eq:gravdiv}\end{aligned}$$

This looks similar to the Einstein equation

$$\begin{aligned}
G_{\mu \nu} = \frac{8\pi G}{c^4} T_{\mu \nu}\end{aligned}$$

except that we have higher derivatives of the stress energy tensor on the
right hand side.

If we substitute again $$\phi \rightarrow \phi \delta_{ij}$$ we can write equation
<a href="#eq:gravdiv" data-reference-type="eqref" data-reference="eq:gravdiv">[eq:gravdiv]</a> like

$$\begin{aligned}
\frac{\partial^2}{\partial r_i \partial r_j} 
 \left( \frac{p}{\rho} \delta_{ij} + v_i v_j +  \phi \delta_{ij} \right) =0\end{aligned}$$

which looks like a tensor version of Bernoullis law:

$$\begin{aligned}
\frac{\partial}{\partial r_i} \left( \frac{p}{\rho} + \frac{1}{2} v^2 +  \phi \right)  = 0\end{aligned}$$

#### The limit $$\nu \longrightarrow 0$$ [the-limit-nu-longrightarrow-0]

At first we should mention that the limit $$\nu \longrightarrow 0$$ is not
equivalent to the limit $$Re \longrightarrow \infty$$. The definition of the
Reynoldsnumber is[^39]

$$\begin{aligned}
Re = \frac{L V}{\nu}\end{aligned}$$

where $$L$$ is a characteristic length and $$V$$ is a characteristic velocity
of the system (whatever that means). Therefore the limit
$$Re \longrightarrow \infty$$ can mean
$$\nu \longrightarrow 0$$, but also $$L \longrightarrow \infty$$ or
$$V \longrightarrow \infty$$. So if we would make an experiment with the same
fluid and the same setup, but would only increase the (characteristic) speed
of the fluid, we would measure effects for higher and higher Reynoldsnumbers.
But this has nothing to do with the limit $$\nu \longrightarrow 0$$, because
we not only have a higher Reynoldsnumber, but also have a higher Machnumber.
This would mean, that we would also measure more and more effects due to the
the increasing compressibility of our fluid. Therefore when doing an
experiment, that should give insights into the regime of vanishing viscosity
we cannot simply increase the Reynoldsnumber by increasing the
characteristic velocity. We have to take care that all the other
characteristic numbers (Mach-Number, Froude-Number ...) stay the same.
Otherwise we measure effects that have nothing to with the interesting limit
$$\nu \longrightarrow 0$$. Sadly the limit $$\nu \longrightarrow 0$$ is often mixed
up with $$Re \longrightarrow \infty$$ in the literature.

For understanding the limit $$\nu \longrightarrow 0$$ Feynman (1964)
investigates the vorticity equation for a newtonian incompressible fluid

$$\begin{aligned}
\frac{\partial}{\partial t}\omega_g
-\epsilon_{ghi}\frac{\partial}{\partial r_h} \epsilon_{ijm} v_j \omega_m =
\nu \frac{\partial^2}{\partial r_j^2}\omega_g\end{aligned}$$

First he mentions that for the case of very high viscosity

$$\begin{aligned}
\nu \frac{\partial^2}{\partial r_j^2}\omega_g \gg \frac{\partial}{\partial t}\omega_g
-\epsilon_{ghi}\frac{\partial}{\partial r_h} \epsilon_{ijm} v_j \omega_m.\end{aligned}$$

Therefore the left side of the vorticity equation can be neglected and
the problem describing a fluid with high viscosity can be simplified to solving
the so called Stokes equation[^40]

$$\begin{aligned}
\frac{\partial^2}{\partial r_j^2}\omega_g =  0_g\end{aligned}$$

But what happens for very low viscosity? Feynman (1964) says decreasing
the viscosity of a fluid leads to an increase of the velocity fluctuations and
so the increasing factor $$\frac{\partial^2}{\partial r_j^2}\omega_g$$ compensates the smallness of
the viscosity. The product of viscosity and $$\frac{\partial^2}{\partial r_j^2}\omega_g$$ doesn’t go to
the limit $$0_g$$, which we would expect from the equation of vorticity we get
for an ideal fluid

$$\begin{aligned}
\frac{\partial}{\partial t}\omega_g
-\epsilon_{ghi}\frac{\partial}{\partial r_h} \epsilon_{ijm} v_j \omega_m =0_g  \end{aligned}$$

So the equations for an ideal fluid do not(!) yield the right limit for
vanishing viscosity. Can we find other equations that do give the right limit
for vanishing viscosity?

One idea might be the following: Feynman (1964) said that the limit
of $$\nu \frac{\partial^2}{\partial r_j^2}\omega_g$$ is not $$0_g$$, but what is it then? The
easiest alternative would be a constant vector $$C_g \neq 0$$! This leads us to
the equations

$$\begin{aligned}
\frac{\partial}{\partial t}\omega_g
-\epsilon_{ghi}\frac{\partial}{\partial r_h} \epsilon_{ijm} v_j \omega_m = C_g 
\label{eq:lowvisvort} \end{aligned}$$

and(!)

$$\begin{aligned}
 \frac{\partial^2}{\partial r_j^2}\omega_g = \frac{1}{\nu} C_g
\label{eq:lowvisvort2} \end{aligned}$$

If we ignore the second term on the left hand side of equation
<a href="#eq:lowvisvort" data-reference-type="eqref" data-reference="eq:lowvisvort">[eq:lowvisvort]</a> for a moment, we see that $$C_g$$ stand for
a dissipation of vorticity independent of $$\nu$$. Something we might expect
for a fluid with low viscosity. The second equation <a href="#eq:lowvisvort2" data-reference-type="eqref" data-reference="eq:lowvisvort2">[eq:lowvisvort2]</a> is
more confusing, since it is a second order partial differential equation,
dependent on viscosity and shows no dependency on time
[^41].
Maybe we can intepret the existence of the two equations in the sense that
equation <a href="#eq:lowvisvort" data-reference-type="eqref" data-reference="eq:lowvisvort">[eq:lowvisvort]</a> gives the vorticity for $$\nu=0$$ and therefore
must be independent of $$\nu$$ and equation <a href="#eq:lowvisvort2" data-reference-type="eqref" data-reference="eq:lowvisvort2">[eq:lowvisvort2]</a> gives the
vorticity for a very small but not zero viscosity and therefore is still
dependent on the viscosity.

Nevertheless it is interesting that we can solve equation
<a href="#eq:lowvisvort2" data-reference-type="eqref" data-reference="eq:lowvisvort2">[eq:lowvisvort2]</a> if we know $$\boldsymbol{C}(\boldsymbol{x})$$, because it is a
vector-poisson equation well know from electrodynamics.
Written in vector notation equation <a href="#eq:lowvisvort2" data-reference-type="eqref" data-reference="eq:lowvisvort2">[eq:lowvisvort2]</a> is

$$\begin{aligned}
\Delta \boldsymbol{\omega} = \frac{1}{\nu} \boldsymbol{C}(\boldsymbol{x})\end{aligned}$$

The solution to this equation is

$$\begin{aligned}
\boldsymbol{\omega} = 
\frac{1}{4 \pi \nu} 
\int \frac{\boldsymbol{C}(\boldsymbol{x}')}{\left\lvert\boldsymbol{x}-\boldsymbol{x}'\right\rvert} dV'\end{aligned}$$

From this we can compute the velocity field because we know

$$\begin{aligned}
\nabla \cdot \boldsymbol{v} = 0,&& \nabla \times \boldsymbol{v} = \boldsymbol{\omega}\end{aligned}$$

So the velocity is a so called pure curl field, because the divergence
of the velocity is zero. Following Bronstein, p. 665 we make the ansatz

$$\begin{aligned}
\boldsymbol{v}= \nabla \times \boldsymbol{A},&& \nabla \cdot \boldsymbol{A} = 0\end{aligned}$$

That means

$$\begin{aligned}
\nabla \times (\nabla \times \boldsymbol{A}) &= \boldsymbol{\omega}\\
\Leftrightarrow \nabla(\nabla \cdot \boldsymbol{A}) - \Delta \boldsymbol{A} &= \boldsymbol{\omega}\\
\Rightarrow \Delta \boldsymbol{A} &= -\boldsymbol{\omega}\end{aligned}$$

So again this leads to a vector-poisson equation for our vectorfield $$\boldsymbol{A}$$.
The complete solution for equation <a href="#eq:lowvisvort2" data-reference-type="eqref" data-reference="eq:lowvisvort2">[eq:lowvisvort2]</a> is then

$$\begin{aligned}
\boldsymbol{v}= \nabla \times \boldsymbol{A} \end{aligned}$$

with

$$\begin{aligned}
\boldsymbol{A} &=\frac{1}{4 \pi} 
\int \frac{\boldsymbol{\omega}(\boldsymbol{x}')}{\left\lvert\boldsymbol{x}-\boldsymbol{x}'\right\rvert} dV'\\
&=\frac{1}{4 \pi} 
\int \frac{-\frac{1}{4 \pi \nu} 
\int \frac{\boldsymbol{C}(\boldsymbol{x}'')}{\left\lvert\boldsymbol{x'}-\boldsymbol{x}''\right\rvert} dV''}
{\left\lvert\boldsymbol{x}-\boldsymbol{x}'\right\rvert} dV'\\
&= -\frac{1}{16 \pi^2 \nu} \int \frac{1}{\left\lvert\boldsymbol{x}-\boldsymbol{x}'\right\rvert}
\int \frac{\boldsymbol{C}(\boldsymbol{x}'')}{\left\lvert\boldsymbol{x'}-\boldsymbol{x}''\right\rvert} dV'' dV'\end{aligned}$$

<div id="refs" class="references csl-bib-body hanging-indent" markdown="1">

<div id="ref-Davis2004" class="csl-entry" markdown="1">

Davis, T. M., and C. H. Lineweaver. 2004. “<span class="nocase">Expanding Confusion: Common Misconceptions of Cosmological Horizons and the Superluminal Expansion of the Universe</span>.” *Publications of the Astronomical Society of Australia* 21: 97–109. <https://doi.org/10.1071/AS03040>.

</div>

<div id="ref-Durst2007" class="csl-entry" markdown="1">

Durst, Franz. 2007. *<span class="nocase">Fluid Mechanics. An Introduction to the Theory of Fluid Flows</span>*. Springer, Berlin.

</div>

<div id="ref-Feynman1964" class="csl-entry" markdown="1">

Feynman, R. P. 1964. *<span class="nocase">Feynman lectures on physics. Volume 2: Mainly electromagnetism and matter</span>*. Reading, Ma.: Addison-Wesley, 1964, edited by Feynman, Richard P.; Leighton, Robert B.; Sands, Matthew. [http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=1964flp..book.....F&db\_ key=AST](http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=1964flp..book.....F&db_ key=AST).

</div>

<div id="ref-Feynman1967" class="csl-entry" markdown="1">

———. 1967. *<span class="nocase">The Character of Physical Law</span>*. The Character of Physical Law, The M.I.T. Press, Cambridge/Massachusetts, London 1967. [http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=1999pfto.book.....F&db\_ key=AST](http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=1999pfto.book.....F&db_ key=AST).

</div>

<div id="ref-Frisch1995" class="csl-entry" markdown="1">

Frisch, U. 1995. *<span class="nocase">Turbulence. The legacy of A.N. Kolmogorov</span>*. Cambridge: Cambridge University Press, \|c1995. [http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=1995tlnk.book.....F&db\_ key=AST](http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=1995tlnk.book.....F&db_ key=AST).

</div>

<div id="ref-Landau1991" class="csl-entry" markdown="1">

Landau, L. D., and E. M. Lifschitz. 1991. *Hydrodynamik*. Vol. 6. Lehrbuch Der Theoretischen Physik. Akademie Verlag, Berlin.

</div>

<div id="ref-Penrose1989" class="csl-entry" markdown="1">

Penrose, R. 1989. *<span class="nocase">The emperor’s new mind. Concerning computers, minds and laws of physics</span>*. Oxford: University Press, 1989. [http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=1989esnm.book.....P&db\_ key=AST](http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=1989esnm.book.....P&db_ key=AST).

</div>

<div id="ref-Pope2000" class="csl-entry" markdown="1">

Pope, S. B. 2000. *Turbulent Flows*. Cambridge University Press. [http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2000tufl.book.....P&db\_ key=PHY](http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2000tufl.book.....P&db_ key=PHY).

</div>

<div id="ref-Truelove1997" class="csl-entry" markdown="1">

Truelove, J. K., R. I. Klein, C. F. McKee, J. H. Holliman II, L. H. Howell, and J. A. Greenough. 1997. “<span class="nocase">The Jeans Condition: A New Constraint on Spatial Resolution in Simulations of Isothermal Self-gravitational Hydrodynamics</span>.” *ApJ* 489 (November): L179+. <https://doi.org/10.1086/316779>.

</div>

</div>

[^1]: Is it possible to fix up the Levi-Civita-Symbol so
    it becomes a real tensor?

[^2]: How does this look like in general
    curvilinear coordinates?

[^3]: Nature does not know about the inverse Fourier transform.
    If you have some optical device, which produces the Fourier transform
    of some image and you use it twice on your image you will get a mirrored image!

[^4]: Note that
    with our definition of the fourier transform we cannot define
    the autocorrelation function as $$h_{AC}(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}f(x') f^*(x+x') dx'$$
    because we could then not derive the Wiener-Khinchin theorem.

[^5]: Fluids are materials which behave like a liquid, so they can be
    deformed by shear stresses without limits. All gases and liquids are fluids.

[^6]: This is not a contradiction. Because the molecules of a fluid
    are treated as pointlike particles (which have no volume) one can have a bunch
    of point particles at one point.

[^7]: Also see Appendix
    <a href="#jacobi" data-reference-type="ref" data-reference="jacobi">5.3.1</a>.

[^8]: For a derivation see Appendix <a href="#jacdt" data-reference-type="ref" data-reference="jacdt">5.3.2</a>.

[^9]: Why is there no gravitational effect on the
    internal energy? Can self-gravity be understood similar to
    Van-der-Waals forces? This would imply that the pressure of an ideal gas has
    to be reduced by some factor $$p=\frac{NkT}{V}-\frac{a}{V^2}, a \sim G$$.

[^10]: It is also important to note, that even for a timedependent
    potential the Greenfunction $$\frac{1}{\left\lvert x_i-x_j\right\rvert}$$
    is not timedependent!

[^11]: Nevertheless Penrose (Penrose 1989) pointed out,
    that the entropy of a selfgravitating fluid should rise up to the point when
    all matter is collapsed to a black hole, see also Appendix <a href="#entro" data-reference-type="ref" data-reference="entro">2.18.1</a>.

[^12]: Equivalent to the criterion described in this paragraph might be the
    so called Truelove criterion (Truelove et al. 1997).

[^13]: This is true if
    the velocity field is continuously differentiable twice

[^14]: An equivalent equation would be
    an equation for the rotation tensor, because the rotation tensor is
    dual to the vorticity vector (also see appendix <a href="#rotstraintensor" data-reference-type="ref" data-reference="rotstraintensor">2.12</a>).
    The advantage of formulating an equation for the rotation tensor would be,
    that the rotation tensor can be consistently defined in other dimensions
    than three and also in curved space.

[^15]: See Appendix <a href="#eos" data-reference-type="ref" data-reference="eos">2.7</a>.

[^16]: See Appendix <a href="#eos" data-reference-type="ref" data-reference="eos">2.7</a>.

[^17]: Why can’t we use $$p=R_s \rho T$$ as equation of
    state for an incompressible fluid?

[^18]: Be aware that $$\nu$$ is a spatially independent quantity only
    for incompressible fluids! In the compressible case $$\rho=\rho(x,t)$$ and
    therefore also the kinematic viscosity $$\nu=\nu(x,t)$$. So do not use $$\nu$$
    when dealing with compressible fluid.

[^19]: The same result can be
    obtained by setting $$\frac{\partial v_j}{\partial r_j}=0$$ and $$\frac{\partial}{\partial t}\rho=0$$ in the equation for
    the dissipation of a compressible newtonian fluid <a href="#eq:ncdiss" data-reference-type="eqref" data-reference="eq:ncdiss">[eq:ncdiss]</a>.

[^20]: Have a look at the Appendix
    <a href="#contradiss" data-reference-type="ref" data-reference="contradiss">2.16</a> for an simple example of a flow field, where the second term
    doesn’t
    vanish!

[^21]: Very often $$\rho$$ is moved to the other side of the equation (which is
    possible, because we assume it is spatially constant) and so we get
    $$\epsilon = \frac{1}{V}\frac{\partial}{\partial t}\int_V v^2 dV = -\frac{\nu}{V} \int_V \omega^2 dV$$

[^22]: In
    case of no gravity the pressure gradient has to be zero at the surface.

[^23]: Have a look at section <a href="#nicdiss" data-reference-type="ref" data-reference="nicdiss">2.5.2</a> to understand the assumptions made when deriving this equation.

[^24]: The
    universe is not a maximally symmetric 4-dimensional manifold, but can be
    depicted as maximally symmetric 3-dimensional sheets spacelike sheets
    in 4-dimensional spacetime. The metric on such a manifold is the
    Robertson-Walker-metric.

[^25]: This is a very important
    point. If the space would also
    expand on small scales we couldn’t measure the expansion, because everything
    including our distance measurement device would expand. But on small scales
    the universe is not homogenous. On small scales the metric of the universe is
    not a Robertson-Walker metric, but more like a Schwarzschild metric, which
    is isotropic, but not homogenous.

[^26]: See
    Appendix <a href="#costress" data-reference-type="ref" data-reference="costress">2.15</a>.

[^27]: These values are for a three dimensional world! If we would be
    in a two-dimensional world we would have 2 for a
    one-atomic gas, 3 for a two-atomic gas and 3 for
    any asymmetric many-atomic gas.

[^28]: Is this relation dependent on
    the number of dimensions we live in?

[^29]: This depends
    on the definition of the total energy! If one includes the potential energy
    in the total energy one has to substract also the potential energy to get the
    internal energy.

[^30]: We neglected the second viscosity $$\zeta$$. In
    principle there exists a second Reynolds number $$Re_2=\frac{\rho_0 l_0 v_0
    }{\zeta_0}$$

[^31]: We cannot get rid of Ga2 in the same way, since therefore we
    would have to assume an equation relating $$\sigma_0 \sigma_{ij}^*$$ to the
    internal energy. But this would only be possible, if we would assume that the
    internal energy is a tensorial quantity, which is not the way how
    internal energy is defined normally.

[^32]: $$S_{ij}$$ has six independent components. Is it possible to
    find a representation in terms of two (pseudo-) 3-vectors?

[^33]: We do not consider here the a rigidly rotating fluid as it os often done
    in the literature, because a rigidly rotating fluid is unphysical.

[^34]: See
    Pope (2000).

[^35]: A sketch of this prove can also be found in
    Pope (2000, Appendix G).

[^36]: In the literature this is often called
    kinetic energy spectrum, but this is only true for
    inkompressible flows.

[^37]: K is a constant with units $$\unit{kg\ s\ m^{-2}}$$. The role of
    $$K$$ is similar to the role of $$\epsilon$$ in the electromagnetic theory.

[^38]: The units of the left hand side and the right hand side are the
    same, because of our choice of units for $$K$$.

[^39]: Just from looking at the units we could also write
    $$Re=\frac{\text{Area per time}}{nu}$$ or
    $$Re=\frac{\rho L V}{\eta} = \frac{\rho V^2 t}{\eta} 
    = \frac{\text{Action density}}{\eta}$$.

[^40]: $$0_g$$ is a component of the zero vector.

[^41]: If $$\boldsymbol{C}=\boldsymbol{C}(x,t)$$, then it would also depend on time
