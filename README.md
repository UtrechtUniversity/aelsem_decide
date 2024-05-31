# aelsem_decide
Decide Algebraic Equivalence of Linear Structural Equation Models

## Functions
  - `decide_model_inclusion(G1, G2, p, use_nonzero_only=False, verbose=0)`:
    Check whether one graph's algebraic model is a subset of another's.

  - `decide_model_equivalence(G1, G2, p, use_nonzero_only=False, verbose=0)`:
    Check whether two graphs have the same algebraic model.


## Paper
The theoretical background, including proofs of correctness and efficiency,
are described in the following paper:

> Thijs van Ommen, Efficiently Deciding Algebraic Equivalence of
> Bow-Free Acyclic Path Diagrams, Proceedings of the 40th Annual
> Conference on Uncertainty in Artificial Intelligence (UAI-24), 2024.

If you use this software for a scientific publication, we would
appreciate it if you could cite our paper.


## Example usage
The following code demonstrates a simple application of these functions:

```
# Construct the graph from Verma and Pearl (1991) Figure 2(a).
G1 = DirectedMixedGraph(4)
G1.add_edge(0, 1, '-->')
G1.add_edge(1, 2, '-->')
G1.add_edge(2, 3, '-->')
G1.add_edge(1, 3, '<->')
print("G1:")
G1.print_graph()

# Construct the graph from Van Ommen and Mooij (2017) Figure 1.
G2 = DirectedMixedGraph(4)
G2.add_edge(0, 1, '<->')
G2.add_edge(0, 2, '<->')
G2.add_edge(1, 2, '-->')
G2.add_edge(2, 3, '-->')
G2.add_edge(1, 3, '<->')
print("G2:")
G2.print_graph()

# Does G1 also impose the algebraic constraint imposed by G2?
p = 2 ** 31 - 1
print("decide_model_inclusion(G1, G2, p) returns",
      decide_model_inclusion(G1, G2, p))
# Also check the reverse. Note that subsequent calls are much faster,
# because the first call needs to set up a finite field class.
print("decide_model_inclusion(G2, G1, p) returns",
      decide_model_inclusion(G2, G1, p))
```

This will result in the following output:

```
G1:
 O  -->  .   .
<--  O  --> <->
 .  <--  O  -->
 .  <-> <--  O
G2:
 O  <-> <->  .
<->  O  --> <->
<-> <--  O  -->
 .  <-> <--  O
decide_model_inclusion(G1, G2, p) returns True
decide_model_inclusion(G2, G1, p) returns False
```

(The above code is executed if you run the file using `python aelsem_decide.py`.)
