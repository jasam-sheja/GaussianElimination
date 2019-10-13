# Parallel Gaussian Elimination Using MPI

This project was developed as an assignment for the class of Parallel Programming at the graduate school of Information Science and Technology at Osaka University.

Solve a system of linear equations using the gaussian elimination method. The current implementation only supports solvable systems and assumes no pivoting is required. 
For better utilization of the parallel processes, cyclic scattering is used to distribute the matrix rows among these processes and their computations.
This distribution method is implemented as `MPI_CyclicScatter` with similar API to `MPI_Scatter`. Extra parameters `sendcyclesize` & `recvcyclesize` define the scatter size, wherein our implementation is size of one row. The counter part, `MPI_Scatter`, is also implemented.
![distribute methods](https://github.com/jasam-sheja/GaussianElimination/blob/master/docs/_static/img/distribution.png)

This project is released under MIT license. Check LICENSE.
