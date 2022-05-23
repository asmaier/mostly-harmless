# Central limit theorem
## Introduction
The central limit theorem is of great importance in statistics and the theory of probabilities. 


```python
%matplotlib inline
```


```python
import random
import matplotlib.pyplot as plt
import numpy as np 
from scipy import stats
import seaborn as sns
```


```python
def compute_sample_means(sample_size, sample_number):
    samples = [[random.random() for i in range(0,sample_size)] for j in range(0,sample_number)]
    sample_means = [sum(sample)/len(sample) for sample in samples]
    return np.asarray(sample_means)
```


```python
def plot_histogram(sample_size, sample_number):
    sample_means = compute_sample_means(sample_size,sample_number) 
    stat = stats.describe(sample_means)
    plot=sns.distplot(sample_means, kde=False, fit=stats.norm)
    plot.set(xlim=(0, 1), ylim=(0,10));
    plot.set_title("Sample size: " + str(sample_size) + ", Sample number: " + str(sample_number))
    plot.set_xlabel("Mean: " + str(stat.mean) + ", Variance: " + str(stat.variance) + "\n" 
                    "Skew: " + str(stat.skewness) + ", Kurtosis: " + str(stat.kurtosis))
```


```python
plt.figure()
plt.subplot(2, 3, 1)
plot_histogram(1,10000)
plt.subplot(2, 3, 2)
plot_histogram(2,10000)
plt.subplot(2, 3, 3)
plot_histogram(4,10000)
plt.subplot(2, 3, 4)
plot_histogram(8,10000)
plt.subplot(2, 3, 5)
plot_histogram(16,10000)
plt.subplot(2, 3, 6)
plot_histogram(32,10000)
plt.subplots_adjust(right=3,top=3)
plt.show()
```


    
![png](central-limit-theorem_files/central-limit-theorem_5_0.png)
    



```python
from ipywidgets import interact
```


```python
interact(plot_histogram, sample_size=(1,32), sample_number=(1,10000))
```




    <function __main__.plot_histogram>




    
![png](central-limit-theorem_files/central-limit-theorem_7_1.png)
    



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





```python

```
