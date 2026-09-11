The *pfas* API Reference
========================

This chapter documents the Python API for the *pfas* package.

Core modules
------------

The ``model`` module
~~~~~~~~~~~~~~~~~~~~

.. automodule:: pfas.model
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

The ``data_structure`` module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pfas.data_structure
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

The ``data_loader`` module
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pfas.data_loader
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

The ``solver_utils`` module
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pfas.solver_utils
   :members:
   :undoc-members:
   :exclude-members: Z, T, T_list, P, omega
   :show-inheritance:
   :member-order: bysource

The ``unit_converter`` module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pfas.unit_converter
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

The ``utils`` module
~~~~~~~~~~~~~~~~~~~~

.. automodule:: pfas.utils
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Components
----------

Components are the building blocks used to configure a PFAS model.

The ``component.awi`` module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pfas.component.awi
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

The ``component.kaw`` module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pfas.component.kaw
   :members:
   :undoc-members:
   :exclude-members: outputs
   :show-inheritance:
   :member-order: bysource

The ``component.kd`` module
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pfas.component.kd
   :members:
   :undoc-members:
   :exclude-members: outputs
   :show-inheritance:
   :member-order: bysource

The ``component.preprocessing`` module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pfas.component.preprocessing
   :members:
   :undoc-members:
   :exclude-members: outputs, Field, default_l_when_null
   :show-inheritance:
   :member-order: bysource

The ``component.retardation`` module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pfas.component.retardation
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

The ``component.solver`` module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pfas.component.solver
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

