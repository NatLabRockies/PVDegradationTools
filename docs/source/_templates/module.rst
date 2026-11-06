.. Please when editing this file make sure to keep it matching the
   docs in ../configuration.rst:reference_to_examples

{{ fullname }}
{{ underline }}

.. automodule:: {{ fullname }}

   {% block function_overview %}
   {% if functions %}
   .. rubric:: Function Overview

   .. autosummary::
      :toctree:
      :nosignatures:
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: Classes
   {% for item in classes %}
   .. autoclass:: {{ item }}
      :members:
   {% endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: Exceptions

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}