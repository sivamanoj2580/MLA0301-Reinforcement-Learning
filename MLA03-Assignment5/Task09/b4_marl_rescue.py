"""B4 - Cooperative multi-agent rescue: independent (selfish) vs cooperative (shared team reward)."""
import random, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Complete the Code (B4.2) ----------
def team_reward(agent_rewards, penalty):
    return sum(agent_rewards) - penalty

AGENTS = ["Drone", "GroundRobot", "MedicalRobot", "CommRobot"]
N_ZONES, MAX_ROUNDS = 4, 8
R_RESCUE, C_COLLISION, C_DELAY, C_RESOURCE = 10, 5, 1, 2   # lambda1=5, lambda3=1, lambda2=2 (per wasted unit)


def run_episode(Q, eps, alpha, cooperative, learn=True):
    pending = [True] * N_ZONES                    # unrescued victims
    rescued = collisions = wasted = rounds = 0
    total_team = 0
    for t in range(MAX_ROUNDS):
        if not any(pending):
            break
        rounds += 1
        state = tuple(pending)
        acts = []
        for q in Q:
            v = q.setdefault(state, np.zeros(N_ZONES))
            acts.append(random.randrange(N_ZONES) if random.random() < eps else int(random.choice(np.flatnonzero(v == v.max()))))
        counts = {z: acts.count(z) for z in set(acts)}
        ind_rewards = []
        round_rescued = round_coll = round_waste = 0
        for i, z in enumerate(acts):
            ind = R_RESCUE if pending[z] else 0       # selfish reward: +10 for choosing an unrescued zone
            ind_rewards.append(ind)
        for z, c in counts.items():
            if pending[z]:
                round_rescued += 1
            if c > 1:
                round_coll += c - 1; round_waste += c - 1
        for z in counts:
            pending[z] = False
        shared = team_reward([R_RESCUE] * round_rescued, C_COLLISION * round_coll + C_RESOURCE * round_waste + C_DELAY)
        rescued += round_rescued; collisions += round_coll; wasted += round_waste; total_team += shared
        if learn:
            rewards = [shared] * 4 if cooperative else ind_rewards
            nxt = tuple(pending)
            for i, q in enumerate(Q):
                nv = q.setdefault(nxt, np.zeros(N_ZONES))
                q[state][acts[i]] += alpha * (rewards[i] + 0.9 * nv.max() * (any(pending)) - q[state][acts[i]])
    return dict(rescued=rescued, time=rounds, collisions=collisions, wasted=wasted, team=total_team,
               comm_msgs=rounds * 4)


def train(cooperative, episodes=600, seed=0, alpha=0.2, eps=0.2):
    random.seed(seed); np.random.seed(seed)
    Q = [dict() for _ in AGENTS]
    hist = [run_episode(Q, eps, alpha, cooperative) for _ in range(episodes)]
    final = [run_episode(Q, 0.0, alpha, cooperative, learn=False) for _ in range(100)]
    return hist, final


if __name__ == "__main__":
    rewards = [10, 15, 20, 12]
    print("Team reward:", team_reward(rewards, 7))

    out, curves = {}, {}
    for name, coop in (("Independent (selfish reward)", False), ("Cooperative (team reward)", True)):
        hs, fs = [], []
        for sd in range(5):
            h, f = train(coop, seed=sd); hs.append(h); fs.append(f)
        flat = [x for f in fs for x in f]
        out[name] = {k: round(float(np.mean([x[k] for x in flat])), 2) for k in ("rescued", "time", "collisions", "wasted", "team")}
        out[name]["collision_rate"] = round(out[name]["collisions"] / (out[name]["time"] * 4), 3)
        curves[name] = np.array([[x["team"] for x in h] for h in hs]).mean(0)

    print("\nMetric (greedy evaluation, 5 seeds x 100 episodes)")
    hdr = ["rescued", "time", "collision_rate", "wasted", "team"]
    print("Method | Victims rescued | Completion time (rounds) | Collision rate | Wasted resources | Team reward")
    for name, d in out.items():
        print(name, "|", *[d[k] for k in hdr])
    json.dump(out, open("b4_results.json", "w"))

    S, A, N = 16, N_ZONES, 4
    print("\nComplexity: IQL table entries =", N * S * A, "| joint-action (centralised) table entries =", S * A ** N)

    plt.figure(figsize=(7, 4))
    for name, c in curves.items():
        plt.plot(np.convolve(c, np.ones(20) / 20, mode="valid"), label=name)
    plt.xlabel("Training episode"); plt.ylabel("Team reward (20-ep avg, 5 seeds)")
    plt.title("B4: Independent vs Cooperative learning"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("b4_team_reward_graph.png", dpi=150)
