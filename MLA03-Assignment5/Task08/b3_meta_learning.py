"""B3 - Simplified meta-learning (MAML-style) for forest-fire adaptation."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Complete the Code (B3.2) ----------
def adapt_parameter(theta, target, alpha):
    return theta - alpha * (theta - target)

# ---------- B3.1 Environment configurations ----------
ENVS = {
    "earthquake": {"obstacles": "collapsed buildings, blocked paths", "hazard": "low", "need": "obstacle avoidance, victim ID",
                   "target": np.array([6.0, 2.0, 4.0, 1.0])},
    "flood":      {"obstacles": "waterlogged roads", "hazard": "medium", "need": "safe routes, water level",
                   "target": np.array([5.0, 3.0, 4.5, 2.0])},
    "forest_fire": {"obstacles": "fire zones", "hazard": "high, changing", "need": "hazard avoidance, emergency navigation",
                    "target": np.array([5.5, 2.5, 5.0, 3.0])},
}
ALPHA, BETA, TOL = 0.3, 0.1, 0.1


def inner(theta, y, alpha=ALPHA):
    return adapt_parameter(theta, y, alpha)            # theta' = theta - alpha*(theta - y)


def meta_train(task_targets, iters=500):
    """Outer loop: minimise sum_i L(theta_i'), L = 0.5||theta' - y||^2 (exact MAML gradient)."""
    theta = np.zeros(4)
    for _ in range(iters):
        grad = np.zeros(4)
        for y in task_targets:
            th_p = inner(theta, y)
            grad += (1 - ALPHA) * (th_p - y)           # dL/dtheta through the inner step
        theta -= BETA * grad / len(task_targets)
    return theta


def steps_to_adapt(theta0, y, tol=TOL, max_steps=100):
    th, errs = theta0.copy(), [np.linalg.norm(theta0 - y)]
    for k in range(1, max_steps + 1):
        th = inner(th, y); errs.append(np.linalg.norm(th - y))
        if errs[-1] < tol:
            return k, errs
    return max_steps, errs


if __name__ == "__main__":
    theta, target, alpha = 2.0, 5.0, 0.5
    print("Initial parameter:", theta)
    print("Adapted parameter:", adapt_parameter(theta, target, alpha))

    print("\nEnvironment configurations:")
    for k, v in ENVS.items():
        print(f"  {k}: {({a: b for a, b in v.items() if a != 'target'})}")

    rng = np.random.default_rng(0)
    base = (ENVS["earthquake"]["target"] + ENVS["flood"]["target"]) / 2
    meta_tasks = [base + rng.normal(0, 0.5, 4) for _ in range(20)]   # training tasks (earthquake/flood-like)
    meta_tasks += [ENVS["earthquake"]["target"], ENVS["flood"]["target"]]
    theta_meta = meta_train(meta_tasks)
    theta_scratch = np.zeros(4)
    theta_transfer = ENVS["earthquake"]["target"].copy()   # fully trained on one source env

    y_fire = ENVS["forest_fire"]["target"]
    methods = {"Scratch": theta_scratch, "Transfer (earthquake)": theta_transfer, "Meta-learning (MAML)": theta_meta}
    print("\nMethod | Initial error | Adaptation episodes | Final error")
    plt.figure(figsize=(7, 4))
    for name, th0 in methods.items():
        k, errs = steps_to_adapt(th0, y_fire)
        print(f"{name} | {errs[0]:.3f} | {k} | {errs[-1]:.3f}")
        plt.plot(errs[:15], marker="o", label=name)
    plt.axhline(TOL, color="k", ls="--", lw=.8)
    plt.xlabel("Adaptation step (episode) on forest fire"); plt.ylabel("Parameter error")
    plt.title("B3: Adaptation to forest-fire environment"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("b3_adaptation_graph.png", dpi=150)
