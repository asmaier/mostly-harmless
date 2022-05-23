# Complex numbers

Complex numbers are not numbers. They cannot be ordered according to their size. This basic insight makes clear that trying to work with complex numbers like with usual “real numbers” must fail (e.g. division doesn’t work) and in general is also the reason for the big confusion around them.

But complex numbers are also no vectors (in the geometrical sense). The multiplication rule for complex numbers is completely different from the usual scalar product of geometrical vectors. Multiplying two complex numbers yields another complex number, whereas the usual scalar multiplication of geometrical vectors yields a scalar. So although a complex number can be represented as a set of two numbers, this set of two numbers should not(!) be visualized as geometrical vector.

Instead the modern view is that complex numbers are 2d matrices. They represent the group of (antisymmetric) 2d rotation matrices (https://en.wikipedia.org/wiki/Complex_number#Matrix_representation_of_complex_numbers). All the features of complex numbers follow naturally from this representation, e.g. multiplying two matrices yields another matrix. Unfortunately most textbooks don’t even mention the matrix representation of complex numbers, although this really makes clear what complex “numbers” are, how they can be extended (e.g. quaternions are 3d rotation matrices) and how they fit into the bigger picture which is https://en.wikipedia.org/wiki/Group_theory .

