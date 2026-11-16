"""R9 check: dual proximal gradient (AMA, c2.methods.run_dual_pg) == AADMM-beta prediction step (17a)-(17c)
iterated alone (lam <- lam~, z <- B y~; no extrapolation (18)-(19), no correction).
Runs 50 iterations on one a_range instance per problem class and prints the max relative deviation of the dual
iterates.  Run from any directory: python tests/check_dualpg_equiv.py (uses this C2_supplementary folder)."""
import sys, os, numpy as np
PKG = os.environ.get("C2PKG", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.argv = ["x"]; sys.path.insert(0, PKG); os.chdir(PKG)
import run_c2 as R
from c1.problems import make_problem
from c1.methods import _xstep

tasks = R.build_tasks()
want = ["P1_m100", "P2_n350", "P3_p500_mu1_", "P4_N64"]
for key in want:
    t = next(t for t in tasks if t["id"].startswith("a_range__" + key) and t["id"].endswith("__dual_pg__beta1.5"))
    prob = make_problem(t["spec"])
    a, b, B = prob.a, prob.b, prob.B
    gg = prob.make_gstar_grad()
    for beta in (1.0, 1.5, 1.9):
        # prediction-only AADMM
        lam = np.zeros(prob.m); z = B.mv(gg(B.rmv(lam)))   # compatible start: z = B y(lam^1)
        # dual PG
        xi = np.zeros(prob.m)
        dev = 0.0
        for k in range(50):
            xt = _xstep(prob, lam, z - b, beta)
            lt = lam - beta * (a * xt + z - b)
            lam, z = lt, B.mv(gg(B.rmv(lt)))
            gpsi = B.mv(gg(B.rmv(xi))); sarg = xi - beta * gpsi
            x = _xstep(prob, sarg, -b, beta)
            xi = sarg - beta * (a * x - b)
            dev = max(dev, np.linalg.norm(lam - xi) / max(1.0, np.linalg.norm(xi)))
        print(f"{t['id'].split('__')[1]:24s} beta={beta}: max rel. deviation over 50 its = {dev:.1e}")
