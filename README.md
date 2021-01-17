![tests](https://github.com/APLA-Toolbox/PythonPDDL/workflows/tests/badge.svg?branch=main)
![build](https://github.com/APLA-Toolbox/PythonPDDL/workflows/build/badge.svg?branch=main)
[![codecov](https://codecov.io/gh/APLA-Toolbox/PythonPDDL/branch/main/graph/badge.svg?token=63GHA9JUND)](https://codecov.io/gh/APLA-Toolbox/PythonPDDL)
[![CodeFactor](https://www.codefactor.io/repository/github/apla-toolbox/pythonpddl/badge)](https://www.codefactor.io/repository/github/apla-toolbox/pythonpddl)
[![Percentage of issues still open](http://isitmaintained.com/badge/open/APLA-Toolbox/PythonPDDL.svg)](http://isitmaintained.com/project/APLA-Toolbox/PythonPDDL "Percentage of issues still open")
[![GitHub license](https://img.shields.io/github/license/Apla-Toolbox/PythonPDDL.svg)](https://github.com/Apla-Toolbox/PythonPDDL/blob/master/LICENSE)
[![GitHub contributors](https://img.shields.io/github/contributors/Apla-Toolbox/PythonPDDL.svg)](https://GitHub.com/Apla-Toolbox/PythonPDDL/graphs/contributors/)

# PyMAPF
A Python toolbox for Multi-Agents Planning (Centralized and Decentralized)

## Planners
### Decentralized (distributed)
- Velocity Obstacles
- Nonlinear Model Predictive Control
- TO DO:
    - Add parallelism
    - Replanning RRT*
### Centralized
- TO DO:
    - CBS
    - ECBS
    - SIPP

## Dependencies

- Install Python (3.7.5 is the tested version)
- Install Pip: `sudo apt install python3-pip`
- Upgrade Pip: `python3 -m pip install --upgrade pip`

### Using the repository

- Clone the repo: `git clone https://github.com/apla-toolbox/pymapf`
- Cd into the repo `cd pymapf`
- Install requirements: `python3 -m pip install -r requirements.txt`

### Using the pip package (Not yet available)

- Install the package: `python3 -m pip install pymapf`

## Usage

### Scripts

Launch hub switch scripts using:
    - `python3 scripts/switch_positions_nmpc.py`
    - `python3 scripts/switch_positions_vel_obstacles.py` (broken)

More to come...

### Library

```python
from pymapf.decentralized import MultiAgentNMPC
from pymapf.decentralized.position import Position
import numpy as np

sim = MultiAgentNMPC()
sim.register_agent("r2d2", Position(0, 3), Position(10, 7))
sim.register_agent("bb8", Position(0, 7), Position(5, 10))
sim.register_agent("c3po", Position(10, 7), Position(5, 0))
sim.register_obstacle(2, np.pi/4, Position(0, 0))
sim.run_simulation()
sim.visualize("filename_test", 10, 10)
```

```python
from pymapf.decentralized.velocity_obstacle import MultiAgentVelocityObstacle
from pymapf.decentralized.position import Position

sim = MultiAgentVelocityObstacle(simulation_time=8.0)
sim.register_agent("r2d2", Position(0, 3), Position(10, 7))
sim.register_agent("bb8", Position(0, 7), Position(5, 10))
sim.register_agent("c3po", Position(10, 7), Position(5, 0))
sim.run_simulation()
sim.visualize("filename_test_2", 10, 10)
```
