# -*- coding: utf-8 -*-
"""
Created on 2025/04/21 21:50:32
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com

This part of the code comes from https://github.com/cvxgrp/signal-decomposition
OSD: Optimization(-based) Signal Decomposition

Modeling language for finding signal decompositions
This software provides a modeling language for describing and solving signal decomposition problems.
This framework is described in detail in an acompanying monograph, and examples are available in the notebooks directory.
New users are encouraged to try out our no-code, no-math introduction to signal decomposition, available here:
http://signal-decomp-tutorial.org/

We formulate the problem of decomposing a signal into components as an optimization problem, where components are
described by their loss functions. Once the component class loss functions are chosen, we minimize the total loss subject
to replicating the given signal with the components. Our software provides a robust algorithm for carying out this
decomposition, which is guaranteed to find the globally optimal descomposition when the loss functions are all convex,
and is a good heuristic when they are not.
"""
