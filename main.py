from math import sqrt, pi, factorial, ceil, log2, floor
from statistics import fmean

import configparser

import networkx as nx
import matplotlib.pyplot as plt

from qiskit_ibm_runtime import (QiskitRuntimeService, Options,
                                Sampler)
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister,
                    Aer, execute)
from qiskit.visualization import plot_histogram

from problem import decompose, diffusion, make_problem
from optimization import deduplicate

from simple import simple_system
from minimal import minimal_system
from balanced import balanced_system
from original import original_system


def make_circuit(graph, color_number, method, grover_iterations=-1):
    qubit_count, init, compose = method

    n = graph.order()
    k = color_number
    if grover_iterations == -1:
        grover_iterations = floor(
            pi / 4 * sqrt(2 ** (n * ceil(log2(k))) / factorial(k)))
    print("Grover iterations:", grover_iterations)
    qn = ceil(log2(k))  # qubits per node
    # inv_col = 2 ** qn - k

    def node_qubits(i):
        return list(range(qn*i, qn*(i+1)))

    problem = make_problem(graph, k)

    num_qubits = qubit_count(problem)

    qc = QuantumCircuit(QuantumRegister(num_qubits),
                        ClassicalRegister(n * qn))

    for i in range(qn*n):
        qc.h(i)

    qc.x(num_qubits - 1)
    qc.h(num_qubits - 1)

    init(qc, problem)

    for i_grv in range(grover_iterations):
        problem = make_problem(graph, k)  # new history
        qv = compose(qc, problem)
        qc.mcx(qv, num_qubits - 1)
        decompose(problem)
        diffusion(qc, problem)

    for i in range(qn*n):
        qc.measure(i, i)

    return qc


def cpu_color_graph(graph, k, node=0, coloring=[]):
    if node == graph.order():
        return [coloring]

    def admissible_colors():
        return set(range(k)) - set(
            [coloring[i] for i in graph.adj[node] if i < len(coloring)])

    valids = []
    for color in admissible_colors():
        valids += cpu_color_graph(graph, k, node+1, coloring + [color])
    return valids


def configure(args_kw):
    config_f = configparser.ConfigParser()
    config_f.read("config.ini")

    def parse(v):
        if v.lower() in ["yes", "true", "on"]:
            return True
        if v.lower() in ["no", "false", "off"]:
            return False
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return v

    conf = {}
    conf.update([(key, parse(value))
                 for key, value in config_f["graph"].items()])
    conf.update([(key, parse(value))
                 for key, value in config_f["options"].items()])
    conf.update([(key, value)
                 for key, value in args_kw.items()
                 if value is not None])

    if "figsize" not in conf:
        conf["figsize"] = (conf["histogram_size_w"], conf["histogram_size_h"])
    if "graph" not in conf:
        if conf["generate"] == "complete":
            conf["graph"] = nx.complete_graph(conf["k"])
            print("Generating complete graph of", conf["k"], "colors")
        elif conf["generate"] == "random":
            conf["graph"] = gen_graph(
                conf.get("nodes", conf["k"]),
                conf["k"])
            print("Generating random graph with", conf["nodes"],
                  "nodes with chromatic number:", conf["k"])

    systems = {"simple": simple_system,
               "minimal": minimal_system,
               "balanced": balanced_system,
               "original": original_system}

    conf["system"] = systems[conf["system"]]
    return conf


def run_circ(qc, conf):
    if conf["run"] == "local":  # local sim
        print("Simulating circuit locally")
        backend = Aer.get_backend('aer_simulator')
        job = execute(qc, backend, shots=conf["local_shots"])
        results = job.result()
        return results.get_counts()
    elif conf["run"] == "online" or conf["run"] == "quantum":
        service = QiskitRuntimeService()
        if conf["run"] == "online":
            print("Simulating circuit on online servers")
            backend = service.get_backend(conf["online_sim"])
            shots = conf["online_shots"]
        else:
            qp = conf["quantum_sim"]
            if not qp:
                qp = service.least_busy(simulator=False,
                                        min_num_qubits=qc.num_qubits)
                print("Running on quantum computer", qp.name)
            backend = service.get_backend(qp.name)
            shots = conf["quantum_shots"]
        options = Options()
        options.execution.shots = shots
        sampler = Sampler(backend=backend, options=options)
        job = sampler.run(qc)
        results = job.result()
        return results.quasi_dists[0].binary_probabilities()
    else:
        print("Not executing the circuit")
        return None


def plot_graph(graph):
    p = nx.draw(graph)
    plt.show()
    plt.close(p)


def plot_figures(measures, figsize):
    fig = plot_histogram(measures, figsize=figsize)
    fig.tight_layout()
    fig.savefig("measures_new.png")
    # plt.show()
    plt.close(fig)


def plot_circuit(qc):
    p = qc.draw(output="mpl")
    plt.show()
    plt.close(p)


graphs = []


def main(k=None, graph=None, run=None, grover_iterations=None, online_sim=None,
         quantum_sim=None, local_shots=None, online_shots=None,
         quantum_shots=None, print_circuit=None, figsize=None,
         generate=None, nodes=None, print_graph=None, system=None):
    kwargs = locals().copy()  # keyword arguments as a dict

    global conf, measures
    conf = configure(kwargs)

    graphs.append(conf["graph"])

    drawings = []

    if conf["print_graph"]:
        drawings.append(lambda: plot_graph(conf["graph"]))

    qc = make_circuit(conf["graph"], conf["k"],
                      conf["system"],
                      conf["grover_iterations"])

    print("depth:", qc.depth())
    print("width:", qc.num_qubits)
    # qc.draw(output="mpl")
    qc = deduplicate(qc)
    print("  After optimization:")
    print("depth:", qc.depth())
    print("width:", qc.num_qubits)
    print(" -> depth * width:", qc.depth() * qc.num_qubits)
    if print_circuit:
        drawings.append(lambda: plot_circuit(qc))

    measures = run_circ(qc, conf)

    if measures is None:
        while drawings:
            drawings.pop()()
        return

    plot_figures(measures, conf["figsize"])

    n = conf["graph"].order()
    qn = ceil(log2(conf["k"]))

    (all_colorings, correct_colors, measures_of_correct,
     measures_of_incorrect) = interpret_measures(conf, measures)

    print("Number of solutions(cpu):", correct_colors,
          "/", all_colorings)
    if correct_colors != 0:
        print("Optimal grover iterations number:",
              pi / 4 * sqrt(all_colorings / correct_colors))
        print("Random guess chance of being correct:",
              correct_colors / all_colorings)
        print("Chance of getting a correct result:",
              sum(measures_of_correct)/sum(measures.values()))
        print("Average \"probability\" of individual correct outcomes:",
              fmean(measures_of_correct)/sum(measures.values()))
        print("Average \"probability\" of individual incorrect outcomes:",
              fmean(measures_of_incorrect)/sum(measures.values()))

    while drawings:
        drawings.pop()()


def interpret_measures(conf, measures):
    qn = ceil(log2(conf["k"]))
    all_colorings = 2 ** (qn * len(conf["graph"]))
    correct_colors = cpu_color_graph(conf["graph"], conf["k"])

    def coloring(code):  # parse the measured colors into ints
        for i in reversed(list(range(int(len(code)/qn)))):
            yield int("".join(reversed(code[i*qn:(i+1)*qn])), 2)

    measures_of_correct = []
    measures_of_incorrect = []

    for code, count in measures.items():
        if list(coloring(code)) in correct_colors:
            measures_of_correct.append(count)
        else:
            measures_of_incorrect.append(count)

    for i in range(len(correct_colors) - len(measures_of_correct)):
        measures_of_correct.append(0)

    for i in range(2 ** (qn * len(conf["graph"]))
                   - len(correct_colors) - len(measures_of_incorrect)):
        measures_of_incorrect.append(0)

    return (all_colorings, len(correct_colors),
            measures_of_correct, measures_of_incorrect)


# generate graph with specific chromatic number
def gen_graph(nodes, colors, max_tries=10000):
    for i in range(max_tries):
        g = nx.gnp_random_graph(nodes, 0.5)
        if (
                len(cpu_color_graph(g, colors - 1)) == 0
                and len(cpu_color_graph(g, colors)) != 0
        ):
            return g


# generate graph with low number of grover iterations required
def cheat_graph(n, k, amt=20):
    graphs = [gen_graph(n, k) for i in range(amt)]
    return min(graphs,
               key=lambda g: floor(pi/4 * sqrt(2 ** (ceil(log2(k)) * len(g))
                                               / len(cpu_color_graph(g, k)))))
