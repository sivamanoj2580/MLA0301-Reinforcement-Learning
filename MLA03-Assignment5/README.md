# MLA03 – Reinforcement Learning – Assignment 5 (CO5)

Name: U. Lakshmi Chenna Kesava Reddy
Register Number: 192425206
Python: 3.12 | Libraries: NumPy 2.x, Matplotlib 3.x (Graphviz only for diagrams)

## Install

&#x20;   pip install -r requirements.txt


## Folders

|Folder|Task|Run|
|-|-|-|
|Task01–Task05|Part A design answers (A1–A5): design\_answers.md, diagrams (.png/.dot)|–|
|Task06|B1 Hierarchical RL (earthquake)|`cd Task06 \&\& python b1\_hrl\_earthquake.py`|
|Task07|B2 MAXQ (flood)|`cd Task07 \&\& python b2\_maxq\_flood.py`|
|Task08|B3 Meta-learning (forest fire)|`cd Task08 \&\& python b3\_meta\_learning.py`|
|Task09|B4 Multi-agent RL|`cd Task09 \&\& python b4\_marl\_rescue.py`|
|Task10|B5 POMDP + ethics (run Task06 and Task09 first)|`cd Task10 \&\& python b5\_pomdp\_ethics.py`|

## Results summary

* B1: flat and hierarchical agents both learn the optimal 12-step hazard-free policy; HRL converges slightly faster (\~15 vs \~17 episodes).
* B2: Navigation 8, Victim Detection 12, Medical Assistance 16 (preferred).
* B3: adaptation episodes to forest fire – scratch 13, transfer 9, meta-learning 8.
* B4: cooperative team reward 39.0 vs -13.2 independent; collision rate 0 vs 0.643.
* B5: belief \[0.6, 0.4] -> \[0.8, 0.2]; all 4 ethical scenarios handled by rule gates.

Each output.txt is the captured execution output; commit one task at a time.

