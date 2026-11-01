"""
Numerical analysis: FDA optimal mechanism when pi_1 in S_2
(both IC constraints potentially active).

Compares:
  Case A: pi_1 not in S_2, IC_1 slack  (closed-form, Theorem 2)
  Case B: pi_1 in S_2, IC_2 binding    (semi-closed-form, Proposition 1)
  Case C: Full numerical optimisation   (scipy, for validation)

Also: joint optimisation over experiments (the truly hard problem).
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'font.size': 11, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False
})

# ── helpers ──────────────────────────────────────────────────
def h_val(pi, p):
    x, y = pi
    return p * y + (1 - p) * (1 - x)

def dg(pi, p):
    x, y = pi
    return p * y - (1 - p) * (1 - x)

def db(pi, p):
    x, y = pi
    return p * (1 - y) - (1 - p) * x

def fda_per_type(pi, Ag, Ae, p):
    return (1 - p) + dg(pi, p) * Ag + db(pi, p) * Ae

def firm_eu(pi, Ag, Ae, p):
    hi = h_val(pi, p)
    return Ag * hi + Ae * (1 - hi)


# ── Parameters ───────────────────────────────────────────────
p  = 0.45
t  = 0.5
pi1 = (0.70, 0.95)   # big type: same specificity, higher sensitivity
pi2 = (0.70, 0.75)   # small type: same specificity, lower sensitivity

h1 = h_val(pi1, p)
h2 = h_val(pi2, p)
dg1, db1 = dg(pi1, p), db(pi1, p)
dg2, db2 = dg(pi2, p), db(pi2, p)

print("=" * 60)
print("MODEL PARAMETERS")
print("=" * 60)
print(f"  p = {p},  t = {t}")
print(f"  pi_1 = {pi1}  =>  h_1 = {h1:.4f},  Dg = {dg1:.4f},  Db = {db1:.4f}")
print(f"  pi_2 = {pi2}  =>  h_2 = {h2:.4f},  Dg = {dg2:.4f},  Db = {db2:.4f}")
print(f"  Dg+Db = {dg1+db1:.4f}  (= 2p-1 = {2*p-1:.4f})")
print(f"  h_1 > h_2?  {h1 > h2}  ({h1:.4f} vs {h2:.4f})")
print(f"  Both convincing?  Dg1={dg1:.4f}>0: {dg1>0},  Dg2={dg2:.4f}>0: {dg2>0}")


# ── Case A: pi_1 NOT in S_2, IC_1 slack (Theorem 2) ─────────
def solve_A(c):
    """Closed-form: Ag1=Ag2=1, Ae=max(0, alpha*)."""
    if c <= h2:
        Ae = 0.0
    else:
        alpha_s = (c - h2) / (1 - h2)
        if alpha_s > 1.0:
            Ae = 0.0  # infeasible
        else:
            # V_both at alpha_s
            V_both = ((1-p) + t*(dg1 + db1*alpha_s)
                      + (1-t)*(dg2 + db2*alpha_s))
            # V without participation-inducing
            if c <= h1:
                V_alt = t*((1-p) + dg1) + (1-t)*(1-p)  # type 1 only
            else:
                V_alt = 1 - p  # nobody
            Ae = alpha_s if V_both > V_alt else 0.0

    Ag1 = Ag2 = 1.0
    # payoffs
    eu1 = firm_eu(pi1, Ag1, Ae, p)
    eu2 = firm_eu(pi2, Ag2, Ae, p)
    V1 = fda_per_type(pi1, Ag1, Ae, p) if eu1 >= c - 1e-9 else (1 - p)
    V2 = fda_per_type(pi2, Ag2, Ae, p) if eu2 >= c - 1e-9 else (1 - p)
    V  = t * V1 + (1 - t) * V2
    return Ag1, Ag2, Ae, V


# ── Case B: pi_1 IN S_2, IC_2 binding (Proposition 1) ───────
def solve_B(c):
    """
    IC_2 binding: (Ag2-Ae)*h2 = (Ag1-Ae)*h1.
    Set Ag2=1 => Ag1 = Ae*(1-h2/h1) + h2/h1.
    V decreasing in Ae => Ae = max(0, alpha*).
    Firm EU = h2 + Ae*(1-h2)  =>  alpha* = (c-h2)/(1-h2).
    """
    if c <= h2:
        Ae = 0.0
    else:
        alpha_s = (c - h2) / (1 - h2)
        if alpha_s > 1.0:
            return 1.0, 1.0, 0.0, 1 - p  # infeasible
        # check if participation is worthwhile for FDA
        Ag1_cand = alpha_s * (1 - h2/h1) + h2/h1
        V_both = ((1 - p) + t*(dg1*Ag1_cand + db1*alpha_s)
                  + (1 - t)*(dg2*1.0 + db2*alpha_s))
        # Compare with best no-participation option
        if c <= h1:
            # type 1 participates alone at Ae=0
            V_alt = t * ((1-p) + dg1*1.0) + (1-t) * (1-p)
        else:
            # nobody participates at Ae=0
            V_alt = 1 - p
        if V_both > V_alt:
            Ae = alpha_s
        else:
            Ae = 0.0

    Ag2 = 1.0
    if Ae > 0 or c <= h2:
        Ag1 = Ae * (1 - h2/h1) + h2/h1
    else:
        Ag1 = 1.0

    eu1 = firm_eu(pi1, Ag1, Ae, p)
    eu2 = firm_eu(pi2, Ag2, Ae, p)
    V1 = fda_per_type(pi1, Ag1, Ae, p) if eu1 >= c - 1e-9 else (1 - p)
    V2 = fda_per_type(pi2, Ag2, Ae, p) if eu2 >= c - 1e-9 else (1 - p)
    V  = t * V1 + (1 - t) * V2
    return Ag1, Ag2, Ae, V


# ── Case C: Full numerical (scipy) ──────────────────────────
def solve_C(c, pi1_in_S2=True):
    """Numerical optimisation over (Ag1, Ag2, Ae)."""
    def neg_V(x):
        Ag1, Ag2, Ae = x
        eu1 = firm_eu(pi1, Ag1, Ae, p)
        eu2 = firm_eu(pi2, Ag2, Ae, p)
        V1 = fda_per_type(pi1, Ag1, Ae, p) if eu1 >= c - 1e-9 else (1-p)
        V2 = fda_per_type(pi2, Ag2, Ae, p) if eu2 >= c - 1e-9 else (1-p)
        return -(t*V1 + (1-t)*V2)

    cons = []
    # IC_1
    cons.append({'type': 'ineq',
                 'fun': lambda x: (x[0]-x[2])*h1 - (x[1]-x[2])*h2})
    if pi1_in_S2:
        # IC_2
        cons.append({'type': 'ineq',
                     'fun': lambda x: (x[1]-x[2])*h2 - (x[0]-x[2])*h1})
    # IR_1
    cons.append({'type': 'ineq',
                 'fun': lambda x: x[0]*h1 + x[2]*(1-h1) - c})
    # IR_2
    cons.append({'type': 'ineq',
                 'fun': lambda x: x[1]*h2 + x[2]*(1-h2) - c})

    bounds = [(0,1)]*3
    best, best_V = None, -np.inf
    for a1 in [0.3, 0.7, 1.0]:
        for a2 in [0.3, 0.7, 1.0]:
            for ae in [0.0, 0.3, 0.6]:
                res = minimize(neg_V, [a1,a2,ae], method='SLSQP',
                               bounds=bounds, constraints=cons,
                               options={'ftol':1e-12})
                if res.success and -res.fun > best_V:
                    best_V = -res.fun
                    best = res
    if best is not None:
        return (*best.x, best_V)
    return (np.nan,)*4


# ── Sweep over c ─────────────────────────────────────────────
c_vals = np.linspace(0.01, 0.95, 200)

rA = {k: [] for k in ['Ag1','Ag2','Ae','V']}
rB = {k: [] for k in ['Ag1','Ag2','Ae','V']}
rC = {k: [] for k in ['Ag1','Ag2','Ae','V']}

print("\nSweeping over c ...")
for c in c_vals:
    for solver, store in [(solve_A, rA), (solve_B, rB)]:
        a1, a2, ae, v = solver(c)
        store['Ag1'].append(a1); store['Ag2'].append(a2)
        store['Ae'].append(ae); store['V'].append(v)
    a1, a2, ae, v = solve_C(c, pi1_in_S2=True)
    rC['Ag1'].append(a1); rC['Ag2'].append(a2)
    rC['Ae'].append(ae); rC['V'].append(v)

for d in [rA, rB, rC]:
    for k in d:
        d[k] = np.array(d[k])


# ── c* threshold for Case A ─────────────────────────────────
D = abs(t*db1 + (1-t)*db2)
c_star_A = h2 + (1-t)*dg2*(1-h2) / D if D > 0 else np.inf
print(f"\n  c* (Case A) = {c_star_A:.4f}")

# c* threshold for Case B (same alpha*, different net gain)
coeff_Ae_B = t * ((2*p-1) - dg1*h2/h1) + (1-t)*db2
V0_B = t*((1-p) + dg1*h2/h1) + (1-t)*((1-p) + dg2)
V0_only1 = t*((1-p) + dg1) + (1-t)*(1-p)
# gain from participation at alpha*:
# V_both(alpha*) - V_only1(0) = [V0_B - V0_only1] + coeff_Ae_B * alpha*
# = [(1-t)*dg2 + t*dg1*(h2/h1-1)] + coeff_Ae_B * alpha*
jump_B = (1-t)*dg2 + t*dg1*(h2/h1 - 1)
if abs(coeff_Ae_B) > 1e-10:
    c_star_B = h2 + jump_B * (1-h2) / abs(coeff_Ae_B) if jump_B > 0 else h2
else:
    c_star_B = np.inf
print(f"  c* (Case B) = {c_star_B:.4f}")


# ── Figure 1: Approval probs vs c (3 panels) ────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
panels = [
    (rA, r'(a) $\pi_1 \notin S_2$ (Theorem 2)'),
    (rB, r'(b) $\pi_1 \in S_2$, IC$_2$ binding (Prop. 1)'),
    (rC, r'(c) $\pi_1 \in S_2$ (numerical validation)'),
]
for ax, (r, title) in zip(axes, panels):
    ax.plot(c_vals, r['Ag1'], 'b-',  lw=2, label=r'$A_g^1$')
    ax.plot(c_vals, r['Ag2'], 'r--', lw=2, label=r'$A_g^2$')
    ax.plot(c_vals, r['Ae'],  'k-.', lw=2, label=r'$A(\emptyset)$')
    ax.axvline(h2, color='gray', ls=':', alpha=0.7)
    ax.axvline(h1, color='gray', ls='--', alpha=0.5)
    ax.text(h2+0.01, 1.05, r'$h_2$', fontsize=9, color='gray')
    ax.text(h1+0.01, 1.05, r'$h_1$', fontsize=9, color='gray')
    ax.set_xlabel('Development cost $c$')
    ax.set_title(title)
    ax.legend(fontsize=9, loc='center left')
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel('Approval probability')
plt.tight_layout()
plt.savefig('fig1_approval_vs_cost.png', bbox_inches='tight')
plt.close()
print("Saved fig1_approval_vs_cost.png")


# ── Figure 2: FDA payoff comparison ──────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(c_vals, rA['V'], 'b-',  lw=2,
        label=r'Case A: $\pi_1\notin S_2$ (closed-form)')
ax.plot(c_vals, rB['V'], 'r--', lw=2,
        label=r'Case B: $\pi_1\in S_2$, IC$_2$ binding')
ax.plot(c_vals, rC['V'], 'g:',  lw=2.5,
        label=r'Case C: $\pi_1\in S_2$ (numerical)')
ax.axhline(1-p, color='gray', ls=':', alpha=0.5,
           label=r'Outside option $1-p$')
ax.axvline(h2, color='gray', ls=':', alpha=0.4)
ax.axvline(h1, color='gray', ls='--', alpha=0.4)
ax.set_xlabel('Development cost $c$')
ax.set_ylabel("FDA expected payoff $V$")
ax.set_title('FDA Payoff: Closed-Form vs. IC-Constrained vs. Numerical')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig2_fda_payoff.png', bbox_inches='tight')
plt.close()
print("Saved fig2_fda_payoff.png")


# ── Figure 3: Distortion of Ag1 ─────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(c_vals, rA['Ag1'] - rB['Ag1'], 'purple', lw=2,
        label=r'$A_g^1(\mathrm{A}) - A_g^1(\mathrm{B})$ (IC$_2$ distortion)')
ax.fill_between(c_vals, 0, rA['Ag1']-rB['Ag1'],
                alpha=0.15, color='purple')
ax.axvline(h2, color='gray', ls=':', alpha=0.7, label=r'$c=h_2$')
ax.set_xlabel('Development cost $c$')
ax.set_ylabel(r'Distortion in $A_g^1$')
ax.set_title(r'IC$_2$-induced distortion of big type approval')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig3_distortion.png', bbox_inches='tight')
plt.close()
print("Saved fig3_distortion.png")


# ── Figure 4: Welfare loss from IC_2 ────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
welfare_loss = rA['V'] - rB['V']
ax.plot(c_vals, welfare_loss, 'darkred', lw=2,
        label=r'$V_A - V_B$ (welfare loss from IC$_2$)')
ax.fill_between(c_vals, 0, welfare_loss, alpha=0.15, color='darkred')
ax.axvline(h2, color='gray', ls=':', alpha=0.7, label=r'$c=h_2$')
ax.set_xlabel('Development cost $c$')
ax.set_ylabel('Welfare loss')
ax.set_title(r'Welfare cost of IC$_2$ constraint ($\pi_1\in S_2$)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig4_welfare_loss.png', bbox_inches='tight')
plt.close()
print("Saved fig4_welfare_loss.png")


# ── Print key results ────────────────────────────────────────
print("\n" + "=" * 60)
print("KEY RESULTS")
print("=" * 60)
for c_ex in [0.30, 0.45, 0.55, 0.65, 0.80]:
    print(f"\n  c = {c_ex}:")
    a1,a2,ae,v = solve_A(c_ex)
    print(f"    A (closed):  Ag1={a1:.3f}  Ag2={a2:.3f}  Ae={ae:.3f}  V={v:.4f}")
    a1,a2,ae,v = solve_B(c_ex)
    print(f"    B (IC_2):    Ag1={a1:.3f}  Ag2={a2:.3f}  Ae={ae:.3f}  V={v:.4f}")
    a1,a2,ae,v = solve_C(c_ex)
    print(f"    C (num):     Ag1={a1:.3f}  Ag2={a2:.3f}  Ae={ae:.3f}  V={v:.4f}")

# analytical distortion at a sample c
c_sample = 0.55
alpha_s = (c_sample - h2) / (1 - h2)
Ag1_B = alpha_s * (1 - h2/h1) + h2/h1
print(f"\n  Analytical check at c={c_sample}:")
print(f"    alpha* = {alpha_s:.4f}")
print(f"    Ag1 (Case B) = Ae*(1-h2/h1) + h2/h1 = {Ag1_B:.4f}")
print(f"    Ag1 distortion = 1 - {Ag1_B:.4f} = {1-Ag1_B:.4f}")
print(f"    Welfare loss = t*Dg1*(1-Ag1) = {t*dg1*(1-Ag1_B):.4f}")

print("\nDone.")
