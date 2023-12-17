====
estimate_complexity
====

Simple package for estimating complexity of functions (big O notation)

* Free software: GNU General Public License v3


Features
--------
Library for estimating the complexity of various functions.
Currently only supports big O with respect to one variable (cannot handle O(m*n) for example).
Details on how to use and what each argument means are inside estimate_complexity.py file in the main class called Complexity

Example:

.. code-block:: python

  from estimate_complexity.estimate_complexity import Complexity

  x = Complexity(sort, lambda n: ([range(n)], {}))
  print(x.get_complexity()) # should print 'nlog(n)'
  x.draw_plot_with_all() # draws plots of fitted functions

Credits
---------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

