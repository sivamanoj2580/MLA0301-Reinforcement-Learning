"""B2 - MAXQ task hierarchy and value calculation for flood rescue."""

# ---------- Complete the Code (B2.2) ----------
def maxq_value(subtask_value, completion_value):
    return subtask_value + completion_value

# ---------- B2.1 Task hierarchy ----------
HIERARCHY = {
    "FLOOD_RESCUE": ["NAVIGATION", "VICTIM_DETECTION", "MEDICAL_ASSISTANCE", "RETURN_TO_BASE"],
    "NAVIGATION": ["MOVE", "AVOID_WATER"],
    "VICTIM_DETECTION": ["SCAN", "IDENTIFY"],
    "MEDICAL_ASSISTANCE": ["DELIVER_SUPPLY", "CONFIRM_RESCUE"],
    "RETURN_TO_BASE": [],
}
PRIMITIVE = {"MOVE", "AVOID_WATER", "SCAN", "IDENTIFY", "DELIVER_SUPPLY", "CONFIRM_RESCUE"}
PREREQ = {"MEDICAL_ASSISTANCE": ["VICTIM_DETECTION", "NAVIGATION"], "RETURN_TO_BASE": ["MEDICAL_ASSISTANCE"]}


def show(node="FLOOD_RESCUE", depth=0):
    kind = "primitive" if node in PRIMITIVE else "composite"
    print("  " * depth + f"- {node} ({kind})")
    for c in HIERARCHY.get(node, []):
        show(c, depth + 1)


def can_start(task, done):
    return all(p in done for p in PREREQ.get(task, []))


if __name__ == "__main__":
    print("Task hierarchy:")
    show()
    print("\nComposite tasks:", [t for t in HIERARCHY if t not in PRIMITIVE])
    print("Primitive actions:", sorted(PRIMITIVE))

    subtasks = {"Navigation": (5, 3), "Victim Detection": (8, 4), "Medical Assistance": (10, 6)}
    print("\nMAXQ values  Q(i,s,a) = V(a,s) + C(i,s,a)")
    q = {}
    for task, values in subtasks.items():
        q[task] = maxq_value(values[0], values[1])
        print(task, ":", q[task])
    print("Preferred subtask:", max(q, key=q.get))

    print("\nDependency check (MEDICAL_ASSISTANCE):")
    for done in ([], ["NAVIGATION"], ["NAVIGATION", "VICTIM_DETECTION"]):
        print(f"  completed={done} -> can start: {can_start('MEDICAL_ASSISTANCE', done)}")

    S, A, K = 1000, 6, 4   # illustrative sizes
    print("\nMemory (illustrative S=1000, A=6, K=4 subtasks):")
    print("  Flat Q-table entries   : S*A       =", S * A)
    print("  MAXQ V+C entries (<=)  : K*S*A     =", K * S * A, "(but with state abstraction per subtask it shrinks)")
