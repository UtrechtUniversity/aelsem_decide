import numpy as np
import galois

class DirectedMixedGraph:
    """A simple class to represent graphs with directed and bidirected edges.

    """

    def __init__(self, d):
        """Create an empty graph with d nodes.

        """

        self.mB = np.zeros((d,d), dtype=bool)
        self.mO = np.eye(d, dtype=bool)

    def add_edge(self, v, w, edge):
        """Add an edge to the graph.

        Parameters
        ----------
        v, w : int
            The indices of the nodes between which to add the edge.
        edge : str
            The type of edge(s) to add, in the format used by print_graph:
            -->, <--: directed edge
            <->: bidirected edge
            ==>, <==: bow
            =-=: 2-cycle
            ===: 2-cycle and bidirected edge.
        """

        if v == w:
            raise ValueError('An edge must connect two different nodes')
        error = False
        if edge == "<->":
            self.mO[v,w] = self.mO[w,v] = True
        else:
            if edge[1] == '=':
                self.mO[v,w] = self.mO[w,v] = True
            elif edge[1] != '-':
                error = True

            if edge[0] == '=' and edge[2] == '=':
                self.mB[v,w] = self.mB[w,v] = True
            else:
                if (edge[0] == '<') == (edge[2] == '>'):
                    error = True

                if edge[0] == '<':
                    self.mB[v,w] = True
                elif edge[0] != edge[1]:
                    error = True

                if edge[2] == '>':
                    self.mB[w,v] = True
                elif edge[2] != edge[1]:
                    error = True
        if error:
            raise ValueError('Unknown edge type: ' + edge)

    def num_nodes(self):
        return self.mB.shape[0]

    def skeleton(self):
        """Returns the graph's skeleton as adjacency matrix.

        """

        return np.logical_or(self.mO, np.logical_or(mB, mB.T))

    def transitive_reflexive_closure(self):
        """Returns the transitive reflexive closure of the directed part.

        """

        d = self.mB.shape[0]
        trans_refl_closure = np.logical_or(np.eye(d, dtype=bool), self.mB)
        prev_num = np.count_nonzero(trans_refl_closure)
        # O(log n) calls to numpy matrix multiplication probably faster than
        # O(n^3) loop in Python
        while True:
            #print(trans_refl_closure)
            trans_refl_closure = np.linalg.matrix_power(trans_refl_closure, 2)
            num = np.count_nonzero(trans_refl_closure)
            if num == prev_num:
                break
            prev_num = num
        return trans_refl_closure

    def print_graph(self, indent=0):
        """Print a plain-text visualization shaped like an adjacency matrix.

        """

        d = self.num_nodes()
        for i in range(d):
            print(' '*4*indent, end='')
            for j in range(d):
                if i == j:
                    print(" O ", end=' ')
                elif not self.mO[i,j]:
                    if self.mB[i,j] and self.mB[j,i]:
                        print("=-=", end=' ')
                    elif self.mB[i,j]:
                        print("<--", end=' ')
                    elif self.mB[j,i]:
                        print("-->", end=' ')
                    else:
                        print(" . ", end=' ')
                else:
                    if self.mB[i,j] and self.mB[j,i]:
                        print("===", end=' ')
                    elif self.mB[i,j]:
                        print("<==", end=' ')
                    elif self.mB[j,i]:
                        print("==>", end=' ')
                    else:
                        print("<->", end=' ')
            print()

def compute_all_minors_naive(M):
    """Computes all nxn minors of an nx(n+1) matrix.

    More precisely (relevant for the signs), this function computes the
    determinants that occur in Cramer's rule: the determinant of the left
    nxn matrix, and of the matrices obtained by replacing each of its
    columns by the rightmost column.

    This implementation is slow for large n: O(n^4), while some other
    versions of compute_all_minors_... run in O(n^3).
    """

    n = M.shape[0]
    GF = type(M)
    minors = GF.Zeros(n + 1)
    minors[-1] = np.linalg.det(M[:,0:n])
    for i in range(n):
        A_mod = M[:,0:n].copy()
        A_mod[:,i] = M[:,-1]
        minors[i] = np.linalg.det(A_mod)
    return minors

def compute_all_minors_double_sweep(M, with_python=True):
    """Computes all nxn minors of an nx(n+1) matrix.

    More precisely (relevant for the signs), this function computes the
    determinants that occur in Cramer's rule: the determinant of the left
    nxn matrix, and of the matrices obtained by replacing each of its
    columns by the rightmost column.

    This implementation is O(n^3) when used with with_python=True (the
    default).
    """

    n = M.shape[0]
    GF = type(M)
    minors = GF.Zeros(n + 1)
    cumulative_diagonal_product = GF.Ones(n + 1)

    # Do Gaussian elimination (LU decomposition with partial pivoting) to
    # get the matrix into upper-triangular form.
    # Keep only the upper-triangular part of the PLU decomposition: the
    # other parts have determinant +/- 1.
    P, _, M = M.plu_decompose()
    sign = np.linalg.det(P)

    for i in range(0, n):
        cumulative_diagonal_product[i + 1] = (cumulative_diagonal_product[i]
                                              * M[i,i])
    # The determinant of the square submatrix that leaves out the final column
    # is given by the product of the diagonal elements.
    minors[-1] = sign * cumulative_diagonal_product[-1]
    # The next submatrix (leaving out the next-to-final column) is given by a
    # similar product, only the final factor is now taken from the final
    # instead of the next-to-final column.
    minors[-2] = sign * cumulative_diagonal_product[-2] * M[-1,-1]

    # Further operations work on the left-right flipped matrix.
    M_flipped = np.fliplr(M)
    for k in range(2, n + 1):
        # We will sweep the bottom k rows to upper triangular form.
        # Note that before this operation:
        # - columns beyond the first k+1 have not yet been modified since the
        #   fliplr operation; in particular, in the rows we are sweeping (the
        #   bottom k), only the first k+1 columns can contain nonzeros: the
        #   rest is still zero due to the original PLU;
        # - below the main diagonal of the submatrix consisting of the bottom k
        #   rows, there is one nonzero diagonal, but below that everything is
        #   zero.  (The plu_decompose function doesn't know this, so it
        #   performs redundant row operations there. TODO OPTIMIZATION. The
        #   with_python implementation does us this, but is not necessarily
        #   faster.)
        if not with_python:
            P, _, U = M_flipped[(n-k):n, 0:(k+1)].plu_decompose()
            M_flipped[(n-k):n, 0:(k+1)] = U
            number_of_pair_transpositions = (P.view(np.ndarray)
                                             .diagonal(-1).sum())
        else:
            number_of_pair_transpositions = 0
            for j in range(0, k - 1):
                i = j + n - k
                # The pivot is [i, j]. The element immediately below it may be
                # nonzero; as mentioned above, elements further down are zero.
                if M_flipped[i+1, j] != 0:
                    # There is a nonzero below the pivot. (If not, do nothing.)
                    if M_flipped[i, j] != 0:
                        # The pivot is also nonzero: do a row operation.
                        factor = M_flipped[i+1, j] / M_flipped[i, j]
                        M_flipped[i+1, j] = 0
                        M_flipped[i+1, j+1:k+1] -= (factor
                                                    * M_flipped[i, j+1:k+1])
                    else:
                        # The pivot is zero: swap the two rows.
                        M_flipped[[i, i+1], :] = M_flipped[[i+1, i], :]
                        number_of_pair_transpositions += 1
        if number_of_pair_transpositions % 2:
            sign = -sign
        if k % 2:
            sign = -sign
        minors[n-k] = (sign * cumulative_diagonal_product[n-k]
                       * M_flipped.diagonal(k-n).prod())
    return minors

def decide_model_inclusion(G1, G2, p, use_nonzero_only=False, verbose=0):
    """Check whether one graph's algebraic model is a subset of another's.

    Parameters
    ----------
    G1 : DirectedMixedGraph
        Graph `G1`; must be acyclic.
    G2: DirectedMixedGraph
        Graph `G2`; must be bow-free acyclic with the same number of nodes.
        as `G1`.
    p : int
        A large prime.
    use_nonzero_only : bool, default False
        If True, sample only nonzero parameters.
    verbose : int, default 0
        Determines how much debugging output is printed.

    Returns
    -------
    bool
        False means `G1`'s model is definitely not a subset of `G2`'s.  There
        is a small probability that True is returned even if the correct answer
        is False. This probability becomes smaller when `p` is larger; see the
        paper for details.
    """

    # Sample random parameters for G1 and compute the resulting Sigma.
    d = G1.num_nodes()
    mB1 = G1.mB
    mO1 = G1.mO
    # Create a finite field object with characteristic p, degree 1
    # (i.e. order p).
    GF = galois.GF(p, 1)
    # TODO OPTIMIZATION: It may be more efficient to only sample random
    # numbers where needed, especially for sparse graphs.
    if use_nonzero_only:
        B1 = GF.Random((d, d), low = 1)
        O1 = GF.Random((d, d), low = 1)
    else:
        B1 = GF.Random((d, d))
        O1 = GF.Random((d, d))
    # Apply masks, using elementwise products. For O1, we only keep the
    # upper triangle.
    # Galois doesn't allow this for boolean arrays, so we temporarily view
    # the FieldArray's as ndarray's.
    B1 = B1.view(np.ndarray)
    O1 = O1.view(np.ndarray)
    B1 *= mB1
    O1 *= np.triu(mO1) # (This is only necessary with use_nonzero_only.)
    B1 = B1.view(GF)
    O1 = O1.view(GF)
    # Make O1 symmetric. Adding the transpose gives the desired distribution
    # provided p > 2: it is correct for off-diagonal and diagonal elements,
    # and with and without use_nonzero_only.
    O1 += O1.T
    if verbose >= 1:
        print("B1:")
        print(B1)
        print("O1:")
        print(O1)
    # Sigma = (I - B)^{-1} O (I - B)^{-T}
    I_minus_B1_inv = np.linalg.inv(GF.Identity(d) - B1)
    Sigma = I_minus_B1_inv @ O1 @ I_minus_B1_inv.T
    if verbose >= 1:
        print("Sigma:")
        print(Sigma)

    # Use the htc-identification algorithm to find the parameters for G2.
    # Use memoization to do only necessary computations, and to do them only
    # once. If there is a constraint between v and w, then we need the B
    # parameters of the edges into v and w. We will also need B parameters
    # of more nodes due to the recursive nature of the htc computation.
    mB2 = G2.mB
    mO2 = G2.mO
    skel = np.logical_or(mO2, np.logical_or(mB2, mB2.T))
    reachable = G2.transitive_reflexive_closure().T
    # TODO OPTIMIZATION: It might be faster to compute htr(v) inside
    # solve(v) using BFS.
    htr = (mO2 @ reachable).astype(bool)
    is_solved = np.zeros(d, dtype=bool)
    I_minus_B2 = GF.Identity(d)
    def solve(v):
        if is_solved[v]:
            return
        pa_v = np.where(mB2[v,:])[0]
        num_pa_v = len(pa_v)
        if num_pa_v == 0:
            is_solved[v] = True
            return
        for w in pa_v:
            if htr[v,w]:
                solve(w)
        if verbose >= 1:
            print(f"Solving {v}")
            if verbose >= 2:
                print("pa_v =", pa_v)

        M = GF.Zeros((num_pa_v, d))
        for i, pa in enumerate(pa_v):
            if htr[v,pa]:
                # Reverse half-trek reachability: a row from I - B2.
                M[i,:] = I_minus_B2[pa,:]
            else:
                # No reverse half-trek reachability: just a row from the
                # identity matrix.
                M[i,pa] = 1
        pa_v_plus_v = np.append(pa_v, v)
        Ab = M @ Sigma[:, pa_v_plus_v]
        # Use Cramer's rule (without the denominator) to compute a multiple
        # of the system's solution.
        # Use the identity from the PGM 2022 paper (the display below (3)).
        #minors = compute_all_minors_naive(Ab)
        minors = compute_all_minors_double_sweep(Ab)
        I_minus_B2[v,v] = minors[-1]
        for i in range(num_pa_v):
            pa = pa_v_plus_v[i]
            I_minus_B2[v,pa] = -minors[i]

        if verbose >= 2:
            print(f"I_minus_B2[{v},:] =")
            print(I_minus_B2[v,:])
        is_solved[v] = True
    for w in range(d):
        for v in range(w):
            if not skel[v,w]:
                if verbose >= 1:
                    print(f"There is a constraint between {v} and {w}")
                solve(v)
                solve(w)
    O2 = I_minus_B2 @ Sigma @ I_minus_B2.T
    if verbose >= 2:
        print("I_minus_B2 =")
        print(I_minus_B2)
        print("Sigma =")
        print(Sigma)
        print("O2 =")
        print(O2)
    for w in range(d):
        for v in range(w):
            if not skel[v,w]:
                if O2[v,w] != 0:
                    if verbose >= 1:
                        print(f"Constraint between {v} and {w} is violated: "
                              "definitely no inclusion")
                    return False
    if verbose >= 1:
        print("All constraints satisfied: evidence for model inclusion")
    return True

def decide_model_equivalence(G1, G2, p, use_nonzero_only=False, verbose=0):
    """Check whether two graphs have the same algebraic models.

    Parameters
    ----------
    G1, G2 : DirectedMixedGraph
        The graphs to be compared; must be bow-free acyclic with the same
        number of nodes.
    p : int
        A large prime.
    use_nonzero_only : bool, default False
        If True, sample only nonzero parameters.
    verbose : int, default 0
        Determines how much debugging output is printed.

    Returns
    -------
    bool
        False means the graphs are definitely not algebraically equivalent.
        There is a small probability that True is returned even if the correct
        answer is False. This probability becomes smaller when `p` is larger;
        see the paper for details.
    """

    if np.any(G1.skeleton() != G2.skeleton()):
        return False
    return decide_model_inclusion(G1, G2, p, use_nonzero_only, verbose)

def large_graph_from_lemma(d, s):
    """Returns graphs of the type that maximize the bound in Lemma 4.

    Parameters
    ----------
    d : int
        The number of nodes.
    s : int
        The node not adjacent to node d-1. Must be in {0, 1, ..., d-2}.
    """

    if s > d-2:
        raise ValueError("s too large")
    # s: 0, 1, ..., d-2
    G = DirectedMixedGraph(d)
    G.add_edge(0, 1, '-->')
    for i in range(2, d-1):
        G.add_edge(0, i, '<->')
        for j in range(1, i):
            G.add_edge(j, i, '-->')
    # bidir as early as possible
    bidir_placed = False
    for j in range(0, d-1):
        if j == s:
            pass # the nonadjacent pair
        elif not bidir_placed:
            G.add_edge(j, d-1, '<->')
            bidir_placed = True
        else:
            G.add_edge(j, d-1, '-->')
    return G

if __name__ == "__main__":
    """Demonstration"""

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
