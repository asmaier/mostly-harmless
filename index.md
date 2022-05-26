[see https://ongclement.com/blog/github-pages-indexing-directory-copy]: #
{% assign doclist = site.pages | sort: 'url'  %}
<ul>
   {% for doc in doclist %}
        {% if doc.name contains '.md' or doc.name contains '.html' %}
            <li><a href="{{ site.baseurl }}{{ doc.url }}">{{ doc.url }}</a></li>
        {% endif %}
    {% endfor %}
</ul>

{% assign pageurl = page.url | replace: 'index.md', '' %}
<ul>
{% for file in site.static_files %}  
  {% if file.path contains pageurl %}
    {% if file.extname == '.md' %}
    <li><a href="{{ file.path }}">{{ file.path }}</a></li>
    {% endif %}
  {% endif %}
{% endfor %}
</ul>
