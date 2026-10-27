# Mathematical Model and Worked Example

## 1. Mathematical model

### 1.1 Problem definition

We assign `n` independent computational tasks to `m` heterogeneous machines. Each task must be assigned to exactly one machine. Tasks have no precedence constraints or communication costs.

**Task parameters**

- `T_i`: task `i`
- `w_i > 0`: workload of task `i`

**Machine parameters**

- `M_j`: machine `j`
- `s_j > 0`: processing speed of machine `j`
- `P_j > 0`: active power of machine `j`

We assume fixed machine speeds and active power. Tasks assigned to the same machine are processed sequentially. Idle energy and machine startup/shutdown costs are excluded.

### 1.2 Decision variables and chromosome

A schedule is represented by the chromosome

$$
X = (x_1, x_2, \\ldots, x_n)
$$

where

$$
x_i \\in \\{1, 2, \\ldots, m\\}.
$$

The value `x_i = j` means task `i` is assigned to machine `j`. For example, `X = (1, 2)` assigns task 1 to machine 1 and task 2 to machine 2.

### 1.3 Objective 1: minimize makespan

The processing time of task `i` on machine `j` is

$$
p_{ij} = \\frac{w_i}{s_j}.
$$

The total processing time assigned to machine `j` is

$$
C_j(X) = \\sum_{i:x_i=j} \\frac{w_i}{s_j}.
$$

The makespan is the completion time of the busiest machine:

$$
f_1(X) = \\max_{1 \\leq j \\leq m} C_j(X).
$$

We minimize `f_1(X)`.

### 1.4 Objective 2: minimize modeled active energy

In this simplified model, the active energy used by machine `j` is its active power multiplied by its busy time:

$$
E_j(X) = P_j C_j(X).
$$

Total modeled active energy is

$$
f_2(X) = \\sum_{j=1}^{m} P_j C_j(X).
$$

Substituting the machine completion-time expression gives

$$
f_2(X) = \\sum_{i=1}^{n} \\frac{w_i P_{x_i}}{s_{x_i}}.
$$

Define energy cost per unit of workload on machine `j` as

$$
e_j = \\frac{P_j}{s_j}.
$$

Then the energy objective can also be written as

$$
f_2(X) = \\sum_{i=1}^{n} w_i e_{x_i}.
$$

This is **modeled active energy**, not measured electricity consumption. Idle energy, startup/shutdown energy, communication delays, and precedence constraints are outside the current model.

### 1.5 Analytical lower bounds

Let total workload be

$$
W = \\sum_{i=1}^{n} w_i,
\\qquad
e_{\\min} = \\min_j e_j.
$$

**Energy lower bound**

Every task has an energy cost per unit of workload of at least `e_min`, so

$$
LB_E = W e_{\\min}.
$$

This is the exact minimum energy. It is attained by assigning all tasks to a machine whose `e_j` equals `e_min`.

**Makespan lower bound**

A valid lower bound on makespan is

$$
LB_C =
\\max\\left(
\\frac{W}{\\sum_{j=1}^{m} s_j},
\\frac{\\max_i w_i}{\\max_j s_j}
\\right).
$$

The first term follows from total workload divided by total machine speed. The second follows because the largest task cannot run faster than it would on the fastest machine. This is a lower bound, not necessarily an attainable optimum.

### 1.6 Pareto optimality

A feasible schedule `X` dominates another feasible schedule `Y` if

$$
f_1(X) \\leq f_1(Y), \\qquad
f_2(X) \\leq f_2(Y)
$$

and at least one of these inequalities is strict.

A **Pareto-optimal schedule** is not dominated by any other feasible schedule. The **Pareto front** is the set of objective pairs `(f_1, f_2)` produced by Pareto-optimal schedules.

NSGA-II aims to approximate this front, providing schedules with different trade-offs between makespan and modeled active energy.

---

## 2. Worked example

We use two tasks and two machines. Units are workload units, workload units per second, watts, seconds, and joules.

### 2.1 Input data

**Tasks**

| Task | Workload |
|---|---:|
| `T_1` | 4 |
| `T_2` | 6 |

**Machines**

| Machine | Speed `s_j` | Active power `P_j` | Energy/work `e_j = P_j/s_j` |
|---|---:|---:|---:|
| `M_1` | 2 | 3 | 1.5 |
| `M_2` | 4 | 8 | 2 |

Total workload is

$$
W = 4 + 6 = 10.
$$

Energy cost per unit of workload:

$$
e_1 = \\frac{3}{2} = 1.5,
\\qquad
e_2 = \\frac{8}{4} = 2.
$$

There are `m^n = 2^2 = 4` possible assignments.

### 2.2 Assignment 1: `X = (1, 1)`

Both tasks go to machine 1.

$$
C_1 = \\frac{4}{2} + \\frac{6}{2} = 2 + 3 = 5,
\\qquad C_2 = 0.
$$

$$
f_1(1,1) = \\max(5,0) = 5 \\text{ s}.
$$

$$
f_2(1,1) = 3(5) + 8(0) = 15 \\text{ J}.
$$

### 2.3 Assignment 2: `X = (2, 2)`

Both tasks go to machine 2.

$$
C_1 = 0,
\\qquad
C_2 = \\frac{4}{4} + \\frac{6}{4} = 1 + 1.5 = 2.5.
$$

$$
f_1(2,2) = 2.5 \\text{ s}.
$$

$$
f_2(2,2) = 3(0) + 8(2.5) = 20 \\text{ J}.
$$

### 2.4 Assignment 3: `X = (1, 2)`

Task 1 goes to machine 1; task 2 goes to machine 2.

$$
C_1 = \\frac{4}{2} = 2,
\\qquad
C_2 = \\frac{6}{4} = 1.5.
$$

$$
f_1(1,2) = \\max(2,1.5) = 2 \\text{ s}.
$$

$$
f_2(1,2) = 3(2) + 8(1.5) = 6 + 12 = 18 \\text{ J}.
$$

### 2.5 Assignment 4: `X = (2, 1)`

Task 1 goes to machine 2; task 2 goes to machine 1.

$$
C_1 = \\frac{6}{2} = 3,
\\qquad
C_2 = \\frac{4}{4} = 1.
$$

$$
f_1(2,1) = \\max(3,1) = 3 \\text{ s}.
$$

$$
f_2(2,1) = 3(3) + 8(1) = 9 + 8 = 17 \\text{ J}.
$$

---

## 3. Consolidated results

| Chromosome | `C_1` (s) | `C_2` (s) | Makespan `f_1` (s) | Energy `f_2` (J) | Pareto-optimal? |
|---|---:|---:|---:|---:|---|
| `(1, 1)` | 5 | 0 | 5 | 15 | Yes |
| `(2, 2)` | 0 | 2.5 | 2.5 | 20 | No |
| `(1, 2)` | 2 | 1.5 | 2 | 18 | Yes |
| `(2, 1)` | 3 | 1 | 3 | 17 | Yes |

The non-dominated objective pairs are

$$
\\mathcal{F} = \\{(5,15), (2,18), (3,17)\\}.
$$

The order corresponds to chromosomes `(1,1)`, `(1,2)`, and `(2,1)`. The points are not sorted by makespan.

### 3.1 Dominance check

Schedule `(1,2)` has objectives `(2,18)`. Schedule `(2,2)` has objectives `(2.5,20)`.

Because

$$
2 < 2.5,
\\qquad
18 < 20,
$$

schedule `(1,2)` is better in both objectives and dominates `(2,2)`. Therefore, `(2,2)` is not on the Pareto front.

The other three schedules are non-dominated:
- `(1,1)` has the lowest energy but the highest makespan.
- `(1,2)` has the lowest makespan.
- `(2,1)` provides an intermediate trade-off.

### 3.2 Lower-bound check

Energy lower bound:

$$
LB_E = W e_{\\min} = 10(1.5) = 15 \\text{ J}.
$$

Schedule `(1,1)` reaches 15 J, confirming that this energy lower bound is exact.

Makespan lower bound:

$$
LB_C = \\max\\left(\\frac{10}{2+4},\\frac{6}{4}\\right)
= \\max(1.6667,1.5)
= 1.6667 \\text{ s}.
$$

The best makespan among the four schedules is 2 seconds. Since `2 >= 1.6667`, the lower bound is valid but is not attained in this example.

---

## 4. Re-validation checklist

- [x] All four possible assignments are enumerated.
- [x] Machine completion times and makespans are recalculated.
- [x] Energy is checked using both `sum(P_j * C_j)` and `sum(w_i * e_(x_i))`.
- [x] Pareto dominance is checked against all four assignments.
- [x] Exact energy lower bound is 15 J and is attained.
- [x] Makespan lower bound is 1.6667 s, valid but unattained.
- [x] Energy is described as modeled active energy, not measured electricity consumption.

**Modeling note:** A machine assigned no tasks contributes zero energy under `E_j = P_j * C_j`. This is consistent with the active-energy-only assumption. Including idle energy or machine-on time would require a different energy model.
