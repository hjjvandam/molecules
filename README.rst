.. raw:: html

    <embed>
        <p align="center">
            <img width="300" src="https://github.com/yngtodd/molecules/blob/master/img/molecules.png">
        </p>
    </embed>

--------------------------

.. image:: https://badge.fury.io/py/molecules.png
    :target: http://badge.fury.io/py/molecules
    
.. highlight:: shell

=========
Molecules
=========

Machine learning for molecular dynamics.

Documentation
--------------

For references, tutorials, and examples check out our `documentation`_.

Installation
------------

.. code-block:: console

    git clone git://github.com/yngtodd/molecules
    python3 -m venv env
    source env/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install -e .

Then, install pre-commit hooks: this will auto-format and auto-lint _on commit_ to enforce consistent code style:

.. code-block:: console

    pre-commit install
    pre-commit autoupdate

.. _documentation: https://molecules.readthedocs.io/en/latest
