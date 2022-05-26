{% assign pageurl = page.url | replace: 'index.md', '' %}
<ul>
{% for file in site.static_files %}  
  {% if file.path contains pageurl %}
    {% if file.extname == '.html' %}
    <li><a href="{{ file.path }}">{{ file.path }}</a></li>
    {% endif %}
  {% endif %}
{% endfor %}
</ul>
