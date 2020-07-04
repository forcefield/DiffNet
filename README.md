# DiffNet 

DiffNet is a Python tool for finding optimal allocations of sampling
in computational or experimental measurements of the individual
quantities and their pairwise differences, so as to minimize the covariance
in the estimated quantities.

## Prerequisites

DiffNet depends on [CVXOPT](http://cvxopt.org) and [networkx](https://networkx.github.io/).  You can install these two libraries using
anaconda:

```
conda install -c conda-forge cvxopt
conda install -c anaconda networkx
```

## Civil matters

DiffNet is free open source software.  NO WARRANTY, Use AS IS.

Copyright (C) 2018-2020 Huafeng Xu

If you use DiffNet in a published work, please cite 

Huafeng Xu, Optimal measurement network of pairwise differences, J. Chem. Inf. Model. 59, 4720-4728, 2019, https://doi.org/10.1021/acs.jcim.9b00528.

## How to use

Some examples are provided in [examples.py](https://github.com/forcefield/DiffNet/blob/master/examples.py).

The following outlines an example application of the DiffNet: the
calculation of binding free energies of a set of molecules from
individual (a.k.a. absolute) and relative binding free energy
calculations.  (Underscored function names __func__ indicate
user-defined external functions.)

NOTE: For large networks (number of nodes greater than 200), diffnet does
not scale well in memory.  The users may want to replace A_optimize() with 
the sparse approximation sparse_A_optimal_network() in such cases.

### Binding free energy calculations

```
import numpy as np
from diffnet import A_optimize, update_A_optimal
from diffnet import round_to_integers
from diffnet import MLestimate, covariance

mols = __get_set_of_molecules__(...) # load up the molecules
nmols = len(mols)

# Optionally, experimental values for reference molecules may be 
# incorporated
if __experimental_values__:
    dgexp = np.array( [ None ]*nmols)     
    for i, dg in __experimental_values__:
        dgexp[i] = dg
else: 
    dgexp = None

nsofar = np.zeros( (nmols, nmols))
# Initialize s_{ij} with random numbers
sij = np.random.rand( nmols, nmols)
sij = 0.5*(sij + sij.T)  # symmetrize the fluctuation matrix
converged = False
while not converged:
    # Update the A-optimal difference network given the current estimate of sij
    # the numbers of samples so far, and the total number of samples for
    # the next iteration
    nij = A_optimize(sij, ndelta, nsofar)
    # nij = A_optimize( sij)  # call this if not using iterative optimization.
    # nij[nij < ncut] = 0  # Omit pairs with impractically small allocations.
    nij = round_to_integers( nij)
    for i in xrange(nmols):
        if nij[i,i] > 0:
            __individual_free_energy__( mols[i], nsamples=nij[i,i], ...)
        for j in xrange(i+1, nmols):
            if nij[i,j] == 0: continue        
            # Compute the relative free energy between i and j,
            # using nij[i,j] number of samples.
            __relative_free_energy__(mols[i], mols[j], nsamples=nij[i,j], 
                                     ...)
    nsofar += nij
    # Get ALL the past free energy results once the calculations are done
    fe_results = __get_free_energy_results__(...)
    dgij = np.zeros( (nmols, nmols))
    invsij2 = np.zeros( (nmols, nmols))
    # Loop over pairs (i, j) between which the free energy have been 
    # computed in the past.
    for i, j, dg, var in fe_results:
        # Update the s_{ij} estimate from the calculated variance.    
        # \sigma_e = s_e/\sqrt(n_e) => s_e = \sigma_e \sqrt(n_e)
        sij[i,j] = sij[j,i] = np.sqrt( var*nsofar[i,j]) 
        invsij2[i,j] = invsij2[j,i] = 1./var
        dgij[i,j] = dg
        if (i!=j): dgij[j,i] = -dg

    # Use Maximum-likelihood estimator to derive the individual free energies
    # from the pairwise differences and their variances.
    dgi, vec = MLestimate( dgij, invsij2, dgexp)
    covar = covariance( sij, nsofar)
    
    converged = __check_convergence__(...)

__report_free_energy__( dgi, covar, ...)

```

### Binding free energy calculations with experimental data for some reference molecules

Please refer to
[netbfe.py](https://github.com/forcefield/DiffNet/blob/master/netbfe.py)
and the Jupyter notebook
[netbfe.ipynb](https://github.com/forcefield/DiffNet/blob/master/netbfe.ipynb)
to see how to use DiffNet for binding free energy calculations when
experimental binding free energies are available for some reference
molecules.


