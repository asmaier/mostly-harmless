{% assign pageurl = page.url | replace: 'index.md', '' %}
<ul>
{% for file in site.html_pages %}
   {% if file.url contains pageurl %}
   <li><a href="{{ site.baseurl }}{{ file.url }}">{{ file.url }}</a></li>
   {% endif %}
{% endfor %}
</ul>
