# Mostly harmless...

- Physics
- [Mathematics](http://nbviewer.jupyter.org/github/asmaier/mostly-harmless/tree/master/math)
- [Statistics](http://nbviewer.jupyter.org/github/asmaier/mostly-harmless/tree/master/statistics)

Here I'm collecting some mostly harmless notes on physics, mathematics, statistics as Jupyter notebooks.
You can view the notebooks with nbviewer by clicking on the links above.

Notes converted from LaTeX: http://asmaier.github.io/mostly-harmless/

{% assign doclist = site.pages | sort: 'url'  %}
    <ul>
       {% for doc in doclist %}
            {% if doc.name contains '.md' or doc.name contains '.html' %}
                <li><a href="{{ site.baseurl }}{{ doc.url }}">{{ doc.url }}</a></li>
            {% endif %}
        {% endfor %}
    </ul>
