.. note::

   This page is a reference documentation.

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    {% if not objname in skipclassesmembers %}

        {% block attributes %}
        {% if attributes %}
        .. rubric:: {{ _('Properties') }}

        
        {% for item in attributes %}
        .. autoattribute:: {{ name }}.{{ item }}
        {%- endfor %}
        {% endif %}
        {% endblock %}
        

        {% block methods %}
        {% if methods %}
        .. rubric:: {{ _('Methods') }}

        .. autosummary::
            :toctree:
            :template: methods.rst
        {% for item in methods %}
                {% if not item in skipmethods %}
                    {{ name }}.{{ item }}
                {% endif %}
        {%- endfor %}
        {% endif %}
        {% endblock %}

        

    {% endif %}

.. include:: {{module}}.{{objname}}.examples