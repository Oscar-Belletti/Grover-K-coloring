from math import floor, ceil, log
from problem import (n, qn, history_add, ancilla_index,
                     problem_comp_num,
                     # get_components_list,
                     get_greedy_components_list)


def put(qc, problem, i, j, components):
    if len(components) != 3 ** i:
        raise Exception("Put at level %d requires %d constraints" %
                        (i, 3 ** i))
    if i == 0:  # setting one constraint
        target = ancilla_index(problem) + 3 * i + j
        components[0](target)
        history_add(problem, lambda c=components[0], target=target: c(target))
        return

    for h, start in zip(range(3), range(0, len(components), 3**(i-1))):
        end = start + 3**(i-1)
        put(qc, problem, i-1, h, components[start:end])

    sources = [ancilla_index(problem) + 3 * (i-1) + h for h in range(3)]
    target = ancilla_index(problem) + 3 * i + j
    qc.mcx(sources, target)
    history_add(problem,
                lambda sources=sources, target=target: qc.mcx(sources, target))

    for h, start in reversed(list(
            zip(range(3), range(0, len(components), 3**(i-1))))):
        end = start + 3**(i-1)
        put(qc, problem, i-1, h, components[start:end])


def setup(qc, problem, lines, components):
    if len(components) == 0:
        return []
    i = floor(log(len(components), 3))
    while i >= len(lines) or 0 not in lines[i]:
        i -= 1
    chunk = components[:3**i]
    j = lines[i].index(0)
    put(qc, problem, i, j, chunk)
    lines[i][j] = 1
    return ([ancilla_index(problem) + 3 * i + j]
            + setup(qc, problem, lines, components[3**i:]))


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


def tern_compose(qc, problem):

    components = get_greedy_components_list(qc, problem)
    q = qubits(3, len(components))

    lines = [[0] * 3 for j in range(3, q+1, 3)]
    if q % 3 != 0:
        lines.append([0] * (q % 3))

    return setup(qc, problem, lines, components)


def tern_qubit_count(problem):
    return (n(problem) * qn(problem)
            + qubits(3, problem_comp_num(problem)) + 1)


def tern_init(qc, problem):
    qc.x(ancilla_index(problem))
    qc.x(ancilla_index(problem) + 1)
    qc.x(ancilla_index(problem) + 2)


tern_system = (tern_qubit_count, tern_init, tern_compose)
