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
{% for file in site.pages %}  
   <li>{{ pageurl }}</li>
   <li><a href="{{ site.baseurl }}{{ file.url }}">{{ file.url }}</a></li>
{% endfor %}
</ul>

{% assign dirs = site.pages | map: 'dir' | uniq %}
<ul>
  {% for dir in dirs %}
    <li><a class="page-link" href="{{ dir | prepend: site.baseurl }}">{{ dir }}</a></li>
  {% endfor %}
</ul>
