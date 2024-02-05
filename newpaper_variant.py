from math import sqrt, pi, factorial, ceil, log2, log, floor
from statistics import fmean

import configparser

import networkx as nx
import matplotlib.pyplot as plt

from qiskit_ibm_runtime import (QiskitRuntimeService, Options,
                                Sampler)
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister,
                    Aer, execute)
from qiskit.visualization import plot_histogram

# c -- function that generates part of the circuit.
#      expects one argument: the target.

history = []


def put(qc, i, j, constr, offset):
    if len(constr) != 3 ** i:
        raise Exception("Put at level %d requires %d constraints" %
                        (i, 3 ** i))
    if i == 0:  # setting one constraint
        constr[0](offset + 3 * i + j)
        history.append(lambda c=constr[0], target=offset+3*i+j: c(target))
        return

    for h, start in zip(range(3), range(0, len(constr), 3**(i-1))):
        end = start + 3**(i-1)
        put(qc, i-1, h, constr[start:end], offset)

    qc.mcx([offset + 3 * (i-1) + h for h in range(3)], offset + 3 * i + j)
    history.append(lambda
                   sources=[offset + 3 * (i-1) + h for h in range(3)],
                   x=offset+3*i+j:
                   qc.mcx(sources, x))

    for h, start in reversed(list(
            zip(range(3), range(0, len(constr), 3**(i-1))))):
        end = start + 3**(i-1)
        put(qc, i-1, h, constr[start:end], offset)


def setup(qc, lines, constr, offset):
    if len(constr) == 0:
        return []
    i = floor(log(len(constr), 3))
    while i >= len(lines) or 0 not in lines[i]:
        i -= 1
    chunk = constr[:3**i]
    j = lines[i].index(0)
    put(qc, i, j, chunk, offset)
    lines[i][j] = 1
    return [offset + 3 * i + j] + setup(
        qc, lines, constr[3**i:], offset)


def values(n, q):  # number of values given q qubits
    g = floor(q / n)  # number of complete groups
    m = q - g * n  # qubits of last, incomplete group
    head = m * n ** g
    tail = (n ** (g + 1) - 1) / (n - 1) - 1
    return round(tail + head)


def qubits(n, v):  # number of qubits to represent v values
    q = n*ceil(log(2/n * v + 1, n))
    while values(n, q-1) >= v:
        q -= 1
    return q


def decompose(qc):
    while history:
        history.pop()()


def comparator(qc, a, b, f):
    for i, j in zip(a, b):
        qc.cx(i, j)
        qc.x(j)
    qc.mcx(b, f)
    for i, j in reversed(list(zip(a, b))):
        qc.x(j)
        qc.cx(i, j)
    qc.barrier()


def invalid_color(qc, color, a, dest):
    x_gates = [not bool(int(x)) for x in bin(color)[2:]]
    qnum = len(x_gates)
    for j in range(qnum):
        if x_gates[j]:
            qc.x(a[0] + j)
    qc.mcx(a, dest)
    for j in range(qnum):
        if x_gates[j]:
            qc.x(a[0] + j)


def diffusion2(qc, a_idx):
    for i in range(a_idx-1):
        qc.h(i)
        qc.x(i)
    qc.z(a_idx - 1)
    qc.mcx(list(range(a_idx-1)), a_idx - 1)
    qc.z(a_idx - 1)
    for i in range(a_idx-1):
        qc.x(i)
        qc.h(i)


def tern_compose(qc, graph, k):
    n = graph.order()
    qn = ceil(log2(k))
    offset = qn * n

    components = get_components(qc, graph, k)
    q = qubits(3, len(components))

    lines = [[0] * 3 for j in range(3, q+1, 3)]
    if q % 3 != 0:
        lines.append([0] * (q % 3))

    return setup(qc, lines, components, offset)


def tern_qubit_count(graph, k):
    n = graph.order()
    qn = ceil(log2(k))
    inv_col = 2 ** qn - k
    return n * qn + qubits(3, inv_col * n + len(graph.edges)) + 1


def tern_init(qc, graph, k):
    n = graph.order()
    qn = ceil(log2(k))
    qc.x(qn*n)
    qc.x(qn*n + 1)
    qc.x(qn*n + 2)


def simple_init(qc, graph, k):
    n = graph.order()
    qn = ceil(log2(k))
    inv_col = 2 ** qn - k

    for i in range(inv_col * n + len(graph.edges)):
        qc.x(qn * n + i)


def simple_qubit_count(graph, k):
    n = graph.order()
    qn = ceil(log2(k))
    inv_col = 2 ** qn - k
    return n * qn + inv_col * n + len(graph.edges) + 1


def simple_compose(qc, graph, k):
    n = graph.order()
    qn = ceil(log2(k))
    offset = qn * n
    components = get_components(qc, graph, k)
    for i in range(len(components)):
        components[i](offset + i)
        history.append(lambda c=components[i], x=offset+i: c(x))
    return [offset + i for i in range(len(components))]


def balanced_ancillas(graph, k):
    n = graph.order()
    qn = ceil(log2(k))
    inv_col = 2 ** qn - k
    conditions = inv_col * n + len(graph.edges)
    if conditions <= floor(n/2):
        return conditions, 0
    elif conditions <= n * floor(n/2) + floor(n/2):
        return floor(n/2), ceil((conditions - floor(n/2)) / floor(n/2))
    else:
        return ceil(conditions / (1 + n)), n


def balanced_qubit_count(graph, k):
    n = graph.order()
    qn = ceil(log2(k))
    return n * qn + sum(balanced_ancillas(graph, k)) + 1


def balanced_init(qc, graph, k):
    n = graph.order()
    qn = ceil(log2(k))
    anc1, anc2 = balanced_ancillas(graph, k)
    for i in range(anc1):
        qc.x(qn * n + i)


def balanced_compose(qc, graph, k):
    n = graph.order()
    qn = ceil(log2(k))
    return n + qn


def get_components(qc, graph, k):
    n = graph.order()
    qn = ceil(log2(k))

    def node_qubits(i):
        return list(range(qn*i, qn*(i+1)))

    components = []
    for color in range(k, 2 ** qn):
        for i in range(n):
            components.append(
                lambda x, nqs=node_qubits(i), color=color:
                invalid_color(qc, color, nqs, x))
    for i, j in graph.edges:
        components.append(
            lambda x, nqi=node_qubits(i), nqj=node_qubits(j):
            comparator(qc, nqi, nqj, x))
    return components


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

    num_qubits = qubit_count(graph, k)

    qc = QuantumCircuit(QuantumRegister(num_qubits),
                        ClassicalRegister(n * qn))

    for i in range(qn*n):
        qc.h(i)

    qc.x(num_qubits - 1)
    qc.h(num_qubits - 1)

    init(qc, graph, k)

    for i_grv in range(grover_iterations):
        qv = compose(qc, graph, k)
        qc.mcx(qv, num_qubits - 1)
        decompose(qc)
        diffusion2(qc, qn*n)

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

    if conf["system"] == "tern":
        conf["system"] = (tern_qubit_count, tern_init, tern_compose)
    elif conf["system"] == "simple":
        conf["system"] = (simple_qubit_count, simple_init, simple_compose)
    else:
        raise Exception("Unrecognized circuit generation system")

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

    print("depth:", qc.depth())
    print("width:", qc.num_qubits)

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


# gate == qiskit instruction
def gate_equal(gate1, gate2):
    return (gate1.operation.name == gate2.operation.name
            and gate1.qubits == gate2.qubits)


def gate_qubits(qc, gate):
    return list(map(lambda q: qc.find_bit(q).index, gate.qubits))


def gate_print(qc, gate):
    print(gate.operation.name, gate_qubits(qc, gate))


def gate_write_qubit(qc, gate):
    return max(gate_qubits(qc, gate))


def gate_read_qubits(qc, gate):
    qubits = gate_qubits(qc, gate)
    qubits.remove(max(qubits))
    return qubits


def make_lines(qc, num_qubits):
    return (qc, [[] for i in range(num_qubits)])


def line_gate(lines_q, q, i):
    qc, lines = lines_q
    return lines[q][i][1]


def line_index(lines_q, q, i):
    qc, lines = lines_q
    return lines[q][i][0]


def lines_add(lines_q, index, gate):
    qc, lines = lines_q
    for q in gate_qubits(qc, gate):
        lines[q].append((index, gate))


def lines_remove(lines_q, gate, gate_index):
    qc, lines = lines_q
    for q in gate_qubits(qc, gate):
        indexes = list(zip(*lines[q]))[0]
        i = indexes.index(gate_index)
        lines[q].pop(i)


def line_back_till(lines_q, q, circuit_index):
    qc, lines = lines_q
    for index, gate in reversed(lines[q]):
        if index <= circuit_index:
            break
        yield index, gate


def empty_line(lines_q, q):
    qc, lines = lines_q
    return len(lines[q]) == 0


# check for writes from i onwards on qubit q
def line_clean(lines_q, q, circuit_index):
    qc, lines = lines_q
    for index, gate in line_back_till(lines_q, q, circuit_index):
        if gate_write_qubit(qc, gate) == q:
            return False
    return True


def get_duplicate(lines_q, gate):
    qc, lines = lines_q
    if empty_line(lines_q, gate_write_qubit(qc, gate)):
        return None, None

    duplicate_index = line_index(lines_q, gate_write_qubit(qc, gate), -1)
    duplicate_gate = line_gate(lines_q, gate_write_qubit(qc, gate), -1)

    if (
            gate.operation.name != "h"
            and gate_equal(gate, duplicate_gate)
            and all(map(lambda q: line_clean(lines_q, q, duplicate_index),
                        gate_read_qubits(qc, gate)))):
        return duplicate_gate, duplicate_index
    return None, None


def deduplicate(qc):
    # previous gates as seen by each qubit
    lines = make_lines(qc, len(qc.qubits))

    for index, gate in enumerate(qc.data):
        duplicate_gate, duplicate_index = get_duplicate(lines, gate)
        if duplicate_gate is not None:
            gate.operation.label = "delenda"
            duplicate_gate.operation.label = "delenda"

            lines_remove(lines, duplicate_gate, duplicate_index)
        elif gate.operation.name != "barrier":
            lines_add(lines, index, gate)

    qc_opt = QuantumCircuit(QuantumRegister(qc.num_qubits),
                            ClassicalRegister(qc.num_clbits))

    for ins in qc.data:
        if ins.operation.label != "delenda":
            #  This is simpler than figuring out which arguments
            #  ins.operation.__class__() requires
            name = ins.operation.name
            qs = gate_qubits(qc, ins)
            if name == "h":
                qc_opt.h(qs[0])
            elif name == "z":
                qc_opt.z(qs[0])
            elif name == "x":
                qc_opt.x(qs[0])
            elif name == "cx":
                qc_opt.cx(qs[0], qs[1])
            elif name == "ccx":
                qc_opt.ccx(qs[0], qs[1], qs[2])
            elif name == "mcx_gray" or ins.operation.name == "mcx":
                qc_opt.mcx(qs[:-1], qs[-1])

    for i in range(qc.num_clbits):
        qc_opt.measure(i, i)

    return qc_opt


# generate graph with low number of grover iterations required
def cheat_graph(n, k, amt=20):
    graphs = [gen_graph(n, k) for i in range(amt)]
    return min(graphs,
               key=lambda g: floor(pi/4 * sqrt(2 ** (ceil(log2(k)) * len(g))
                                               / len(cpu_color_graph(g, k)))))


# main(k=3, run="no")
