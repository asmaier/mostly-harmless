# Triangle inequality
## Number inequality
We start with the obvious inequality that the absolute value of a number $\left|a\right|$ is always greater or equal the number $a$ itself
$$
|a| \geq a
$$
Using this we can prove that for numbers $a$ and $b$ we have
$$
|a| + |b|  \geq |a+b|
$$
because
$$
(|a| + |b|)^2  \geq (|a+b|)^2 \\
|a|^2  + 2|a||b| + |b|^2 \geq |a^2 + 2ab + b^2| \\
2|a||b| \geq 2|ab|
$$
The last line is always true if $|a| \geq a$, which completes our prove.
# Cauchy-Schwarz inequality
Because $\sin^2(\phi) + \cos^2(\phi) = 1$ we have
$$
|a|^2|b|^2 = |a|^2|b|^2 (\sin^2(\phi) + \cos^2(\phi))
$$
With the definition of the scalar product $|a\cdot b| = |a||b|\cos(\phi)$ and the cross product $|a\times b| = |a||b|\sin(\phi)$ of vectors $a$ and $b$ we can write this expression as
$$
|a|^2|b|^2 = |a\times b|^2 + |a\cdot b|^2
$$
This is called Lagrange's identity. Since all terms are squared and therefor positive we immediatelly can derive the inequalities
$$
|a|^2|b|^2 \geq |a\cdot b|^2 \\
|a|^2|b|^2 \geq |a\times b|^2
$$
The first of these equations is the Cauchy-Schwarz inequality. The second equations doesn't seem to have a name in the literature. 

The Cauchy-Schwarz inequality comes in many different forms. For example for $n$-dimensional vectors it can be written in cartesian coordinates like
$$
\sum (a_i)^2 \sum (b_i)^2 \geq \left(\sum a_i b_i \right)^2
$$
This can even be generalized to uncountable infinite dimensional vectors (also called continuous square integrable functions) like
$$
\left(\int \left|a(x)\right|^2 dx\right) \cdot \left(\int \left|b(x)\right|^2 dx\right) \geq \left|\int a(x) \cdot b(x)dx\right|^2
$$


```python
from IPython.core.display import HTML

def css_styling():
    styles = open("../styles/custom.html", "r").read()
    return HTML(styles)
css_styling()
```




<style>

    @import url('http://fonts.googleapis.com/css?family=Crimson+Text');
    @import url('http://fonts.googleapis.com/css?family=Source+Code+Pro');

    /* Change code font */
    .CodeMirror pre {
        font-family: 'Source Code Pro', Consolas, monocco, monospace;
    }

    div.input_area {
        width: 60em;
    }

    div.cell{
        width: 60em;
        margin-left: auto;
        margin-right:auto;
    }

    div.text_cell {
        width: 60em;
        margin-left: auto;
        margin-right: auto;
    }

    div.text_cell_render {
        text-align: justify;
        font-family: "Crimson Text";
        font-size: 18pt;
        line-height: 145%;
    }

    div.text_cell_render h1 {
        font-size: 30pt;
    }

    div.text_cell_render h2 {
        font-size: 24pt;
    }

    div.text_cell_render h3 {
        font-size: 20pt;
    }

    .prompt{
        display: None;
    }
</style>



