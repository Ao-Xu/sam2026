# Mathematical interval certificates

Run `python certify_witness.py` and `python certify_radius.py` with Python and mpmath. The scripts use outward interval arithmetic. The first certifies a negative fixed-direction derivative of limiting covariance for a bounded nonunit-diagonal kernel. The second bounds cubic coefficient derivatives over whole radius intervals. Existing JSON files retain the certified enclosures.

The harmonic radius does not certify a numerical signal-strength cutoff: the full RKHS remainder constant and admissible weak-signal range are additionally required. The counterexample concerns limiting covariance. Full proofs and assumptions are in the manuscript appendix.
