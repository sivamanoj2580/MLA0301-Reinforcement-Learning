"""B5 - POMDP belief update, ethical decision rules and integrated evaluation."""
import json, os

# ---------- Complete the Code (B5.1) ----------
def update_belief(prior, likelihood):
    unnormalized = [p * l for p, l in zip(prior, likelihood)]
    eta = sum(unnormalized)
    return [round(u / eta, 2) for u in unnormalized]


def update_belief_exact(prior, likelihood):
    un = [p * l for p, l in zip(prior, likelihood)]
    eta = sum(un)
    return [u / eta for u in un]


# ---------- B5.2 Ethical decision framework ----------
def ethical_decision(case):
    """Rule-based gate: safety > fairness > privacy > transparency, with human supervision for uncertainty."""
    c = case["type"]
    if c == "scarce_supply":      # fairness: triage by urgency (medical need), tie -> human supervisor
        a, b = case["victims"]
        if a["severity"] == b["severity"]:
            return "ESCALATE to human supervisor (equal severity); log decision"
        return "SUPPLY to victim " + (a["id"] if a["severity"] > b["severity"] else b["id"]) + " (higher severity); log reason"
    if c == "risky_route":        # safety: reject routes above risk threshold
        return "TAKE safer route" if case["risk"] > case["threshold"] else "TAKE fastest route"
    if c == "privacy":            # privacy: keep only task-relevant data
        return "MASK/DISCARD personal data; store only victim location"
    if c == "uncertain_victim":   # uncertainty: act on belief, escalate if ambiguous
        b = case["belief"]
        if b >= 0.8:
            return "INITIATE rescue (belief >= 0.8) with operator notification"
        if b >= 0.4:
            return "REQUEST more observation / human confirmation"
        return "CONTINUE scanning"


if __name__ == "__main__":
    prior = [0.6, 0.4]
    likelihood = [0.8, 0.3]
    print("Updated belief:", update_belief(prior, likelihood))

    print("\nSequential belief update (repeated observations favouring s1):")
    b = prior
    for k in range(3):
        b = update_belief_exact(b, likelihood)
        print(f"  after observation {k + 1}: {[round(x, 3) for x in b]}")

    cases = [
        {"type": "scarce_supply", "victims": [{"id": "V1", "severity": 3}, {"id": "V2", "severity": 2}]},
        {"type": "risky_route", "risk": 0.7, "threshold": 0.3},
        {"type": "privacy"},
        {"type": "uncertain_victim", "belief": 0.8},
    ]
    print("\nEthical decisions:")
    for i, cs in enumerate(cases, 1):
        print(f"  Case {i}: {ethical_decision(cs)}")
    compliance = 100.0   # all 4 scenarios resolved by rule gates (safety, fairness, privacy, supervision)
    print("Ethical compliance (scenarios handled by a rule gate): ", compliance, "%")

    b1 = json.load(open("../Task06/b1_results.json"))
    b4 = json.load(open("../Task09/b4_results.json"))
    coop = b4["Cooperative (team reward)"]
    ind = b4["Independent (selfish reward)"]
    hrl = b1["Hierarchical RL"]
    print("\nIntegrated evaluation")
    print("Metric | Independent | Cooperative+HRL system")
    print("Rescue success (%) | -", "|", hrl[1])
    print("Avg response time (rounds) |", ind["time"], "|", coop["time"])
    print("Team reward |", ind["team"], "|", coop["team"])
    print("Collision rate |", ind["collision_rate"], "|", coop["collision_rate"])
    print("Ethical compliance (%) | n/a |", compliance)

    S, O = 25, 4
    print("\nMDP vs POMDP: MDP policy table over states =", S, "| POMDP belief vector size =", S, "(continuous simplex), observation model entries =", S * O)
