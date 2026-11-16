"""C1 comparison package for the JSC revision of the AADMM paper.

Convention used everywhere in this package (same as sn-article.tex eq. (2)/(7)):
    min f(x) + g(y)  s.t.  A x + B y = b,
    L(x, y, lam) = f(x) + g(y) - <lam, A x + B y - b>.
g is mu-strongly convex.  In all test problems A = a*I for a scalar a.
"""
__version__ = "C1-1.2"
