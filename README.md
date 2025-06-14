﻿
---

# Multi-Robot Pathfinding in Dynamic Environments

## Overview

This project tackles the classic **multi-agent pathfinding** problem in a **dynamic, partially observable grid-world**. Multiple autonomous robots must reach their respective goals while avoiding collisions with:
- Static obstacles
- Moving (dynamic) agents with cyclic paths
- Each other (other robots)

Each robot plans its path independently and uses a **hybrid bidirectional heuristic search** strategy to compute optimal time-efficient paths under constraints like mandatory movement per timestep and uncertain peer behavior.

---

## Features

- **N x M Grid Representation**  
  - Free cells, static obstacles (`X`), dynamic obstacles (agents), and goal cells
- **Multiple Independent Robots**  
  - Each with unique start and goal positions
  - No prior knowledge of other robots' paths
- **Dynamic Agents**  
  - Predefined cyclic paths known in advance
  - Occupy different cells at different time steps
- **Bidirectional A\* Heuristic Search**  
  - Optimized for both distance and dynamic agent prediction
- **Collision Detection and Resolution**  
  - Random redirection + replanning on robot-robot collisions
- **Time-Aware Planning**  
  - Robots cannot wait in place; must move every timestep
  - Simulation advances step-by-step with detailed logging

---

## Directory Structure

```
.
├── main.py                     # Entry point for the simulation
├── Input_Files/
│   └── Data/
│       ├── Agent0.txt          # Dynamic agent paths
│       ├── Robots0.txt         # Robot start/goal pairs
│       ├── data0.txt           # Grid layout
│       └── ...                 # Other datasets
└── README.md                   # You're reading it!
```

---

## ⚙How to Run

1. **Install Python 3.x**

2. **Place your dataset files in the directory**:  
   `Input_Files/Data/AgentX.txt`, `RobotsX.txt`, `dataX.txt`  
   (`X` being 0–4)

3. **Run the simulation**
   ```bash
   python main.py
   ```

4. **Choose a dataset** when prompted

---

## Sample Input (Dataset Format)

- **Grid (5x5)**  
  `data0.txt`  
  ```
  5
  S1 . . . .
  . X . . .
  . . . X .
  . . . . .
  . . . . G1
  ```

- **Dynamic Agents**  
  `Agent0.txt`  
  ```
  Agent 1: [((1,1), (1,2), (1,3))] at times [1,2,3]
  Agent 2: [((3,3), (2,3), (1,3))] at times [1,2,3]
  ```

- **Robots**  
  `Robots0.txt`  
  ```
  Robot 1: Start (0, 0) End (4, 4)
  Robot 2: Start (4, 0) End (0, 4)
  ```

---

## Algorithms Used

- **Bidirectional Heuristic Search**  
  Combines forward and backward A\* search to reduce search space

- **Heuristic Function**  
  - Manhattan Distance + Penalty for predicted dynamic obstacle conflict

- **Collision Resolution**  
  - Robots with conflicting moves randomly select legal alternate directions and replan paths

---

## Output

For each robot:
- Start and Goal position
- Path taken
- Time to reach the goal
- Collision events and random reroutes (if any)

Example:
```
Robot 1:
  Start: (0, 0)
  Goal: (4, 4)
  Total Time to reach goal: 9
  Path taken: [(0,0), (0,1), (0,2), ... ,(4,4)]
```

---

## Scalability

- Designed to scale to:
  - Grid sizes up to 1000 x 1000
  - 100 dynamic agents
  - 10 independently navigating robots

---
