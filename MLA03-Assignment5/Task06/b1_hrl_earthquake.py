"""B1 - Hierarchical RL vs Flat Q-learning, 5x5 earthquake grid."""
import random, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Complete the Code (B1.3) ----------
def select_subtask(state):
    robot_position, victim_found, rescued, at_base = state
    if rescued:
        return "RETURN_TO_BASE"
    if not victim_found:
        return "EXPLORE"
    if victim_found and not rescued:
        return "RESCUE"
    return None

# ---------- Environment (B1.1) ----------
SIZE = 5
START = (0, 0)
VICTIM = (3, 3)
OBSTACLES = {(1, 1), (2, 1), (3, 1)}
HAZARDS = {(1, 2), (2, 2)}
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # UP DOWN LEFT RIGHT
MAX_STEPS = 100
R_STEP, R_HAZARD, R_BLOCK, R_RESCUE, R_BASE = -1, -10, -2, 50, 100


class RescueEnv:
    def reset(self):
        self.pos, self.found, self.rescued, self.steps = START, False, False, 0
        self.hazard_entries = 0
        return self.pos

    def step(self, a):
        self.steps += 1
        dr, dc = ACTIONS[a]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        r = R_STEP
        if not (0 <= nr < SIZE and 0 <= nc < SIZE) or (nr, nc) in OBSTACLES:
            r += R_BLOCK
        else:
            self.pos = (nr, nc)
            if self.pos in HAZARDS:
                r += R_HAZARD
                self.hazard_entries += 1
        if abs(self.pos[0] - VICTIM[0]) + abs(self.pos[1] - VICTIM[1]) <= 1:
            self.found = True
        if self.found and self.pos == VICTIM and not self.rescued:
            self.rescued = True
            r += R_RESCUE
        success = self.rescued and self.pos == START
        done = success or self.steps >= MAX_STEPS
        if success:
            r += R_BASE
        return self.pos, r, done, success


def eps_greedy(v, eps):
    if random.random() < eps:
        return random.randrange(4)
    return int(random.choice(np.flatnonzero(v == v.max())))


def run_flat(seed, episodes=400, alpha=0.2, gamma=0.95, eps=0.2):
    random.seed(seed); np.random.seed(seed)
    env, Q, log = RescueEnv(), {}, []
    q = lambda s: Q.setdefault(s, np.zeros(4))
    for ep in range(episodes):
        env.reset(); tot = 0; done = False
        s = (env.pos, env.found, env.rescued)
        while not done:
            a = eps_greedy(q(s), eps)
            _, r, done, succ = env.step(a)
            s2 = (env.pos, env.found, env.rescued)
            target = r + (0 if done else gamma * q(s2).max())
            q(s)[a] += alpha * (target - q(s)[a]); s = s2; tot += r
        log.append((tot, succ, env.steps, env.hazard_entries))
    return Q, log


def eval_flat(Q, n=50):
    env, out = RescueEnv(), []
    for _ in range(n):
        env.reset(); tot = 0; done = False
        while not done:
            s = (env.pos, env.found, env.rescued)
            _, r, done, succ = env.step(int(np.argmax(Q.get(s, np.zeros(4))))); tot += r
        out.append((tot, succ, env.steps, env.hazard_entries))
    return out


SUBGOAL_BONUS = 20
def run_hrl(seed, episodes=400, alpha=0.2, gamma=0.95, eps=0.2):
    random.seed(seed); np.random.seed(seed)
    env = RescueEnv()
    Q = {t: {} for t in ("EXPLORE", "RESCUE", "RETURN_TO_BASE")}
    log = []
    q = lambda t, s: Q[t].setdefault(s, np.zeros(4))
    st = lambda: (env.pos, env.found, env.rescued, env.pos == START)
    for ep in range(episodes):
        env.reset(); tot = 0; done = False
        while not done:
            task = select_subtask(st()); s = env.pos
            a = eps_greedy(q(task, s), eps)
            _, r, done, succ = env.step(a)
            nxt = select_subtask(st())
            sub_done = done or nxt != task
            rr = r + (SUBGOAL_BONUS if nxt != task and not done else 0)
            target = rr + (0 if sub_done else gamma * q(task, env.pos).max())
            q(task, s)[a] += alpha * (target - q(task, s)[a]); tot += r
        log.append((tot, succ, env.steps, env.hazard_entries))
    return Q, log


def eval_hrl(Q, n=50):
    env, out = RescueEnv(), []
    for _ in range(n):
        env.reset(); tot = 0; done = False
        while not done:
            task = select_subtask((env.pos, env.found, env.rescued, env.pos == START))
            _, r, done, succ = env.step(int(np.argmax(Q[task].get(env.pos, np.zeros(4))))); tot += r
        out.append((tot, succ, env.steps, env.hazard_entries))
    return out


def episodes_to_converge(log, window=10, thr=100):
    """First episode where the moving-average training reward reaches thr."""
    rew = np.array([x[0] for x in log], float)
    for i in range(window, len(rew) + 1):
        if rew[i - window:i].mean() >= thr:
            return i
    return len(rew)


if __name__ == "__main__":
    states = [((0, 0), False, False, False), ((2, 2), True, False, False), ((3, 3), True, True, False)]
    for s_ in states:
        print("Selected subtask:", select_subtask(s_))

    res = {"Flat Q-learning": {"train": [], "eval": []}, "Hierarchical RL": {"train": [], "eval": []}}
    for sd in range(5):
        Qf, lf = run_flat(sd); res["Flat Q-learning"]["train"].append(lf); res["Flat Q-learning"]["eval"] += eval_flat(Qf)
        Qh, lh = run_hrl(sd); res["Hierarchical RL"]["train"].append(lh); res["Hierarchical RL"]["eval"] += eval_hrl(Qh)

    table = {}
    print("\nMethod | AvgReward | Success% | AvgSteps | HazardEntries | EpisodesToConverge | AvgTrainReward(first 50 eps)")
    for name, d in res.items():
        ev = np.array(d["eval"], float)
        conv = np.mean([episodes_to_converge(l) for l in d["train"]])
        table[name] = [round(ev[:, 0].mean(), 1), round(100 * ev[:, 1].mean(), 1),
                       round(ev[:, 2].mean(), 1), round(ev[:, 3].mean(), 2), round(conv, 1),
                       round(np.mean([[x[0] for x in l[:50]] for l in d["train"]]), 1)]
        print(name, "|", *table[name])
    json.dump(table, open("b1_results.json", "w"))

    plt.figure(figsize=(7, 4))
    for name, d in res.items():
        curves = np.array([[x[0] for x in l] for l in d["train"]]).mean(0)
        plt.plot(np.convolve(curves, np.ones(10) / 10, mode="valid"), label=name)
    plt.xlabel("Training episode"); plt.ylabel("Episode reward (10-ep avg, 5 seeds)")
    plt.title("B1: Flat RL vs Hierarchical RL"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("b1_reward_graph.png", dpi=150)
