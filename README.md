# Circuit generator for Grover K-coloring

This repository contains the code for the generation of quantum circuits for the k-coloring problem.
Also, examples of the generated circuits are present in the qasm directory.

## Dependencies

To run this code, ensure that you have installed the following libraries: qiskit, qiskit aer, networkx, matplotlib, alive progress, scipy, numpy, pylatexenc

## Usage

The `main` function in `main.py` generates a circuit for a specific k-coloring problem.
The parameters are read from the `config.ini` configuration file and can be overriden by passing them as arguments to the funciton.
The `config.ini` file contains explanation for which parameters and what values are accepted. Example usage:

    from main import main
    # The following line generates and simulates locally a complete graph of three colors for k=3
    # using the minimum width approach
    r = main(k=3, system="simple")  
    # r contains information about the outcome of the generation/execution, most of which
    # is already printed by main
    
