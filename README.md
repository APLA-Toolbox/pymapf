<div align="center">
    
<img src="https://svgshare.com/i/TFJ.svg" alt="Logo" width="300">     
    
</div>

<div align="center">

# PyMAPF

✨ A Python toolbox for Multi-Agents Planning (Centralized and Decentralized) ✨

</div>

<div align="center">
    
![tests](https://github.com/APLA-Toolbox/pymapf/workflows/tests/badge.svg?branch=main)
![pip-package](https://github.com/APLA-Toolbox/pymapf/workflows/.github/workflows/pip-tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/APLA-Toolbox/pymapf/branch/main/graph/badge.svg?token=63GHA9JUND)](https://codecov.io/gh/APLA-Toolbox/pymapf)
[![CodeFactor](https://www.codefactor.io/repository/github/apla-toolbox/pymapf/badge)](https://www.codefactor.io/repository/github/apla-toolbox/pymapf)
[![Percentage of issues still open](http://isitmaintained.com/badge/open/APLA-Toolbox/pymapf.svg)](http://isitmaintained.com/project/APLA-Toolbox/pymapf "Percentage of issues still open")
![PipPerMonths](https://img.shields.io/pypi/dm/pymapf.svg)
[![Pip version fury.io](https://badge.fury.io/py/pymapf.svg)](https://pypi.python.org/pypi/pymapf/)
[![GitHub license](https://img.shields.io/github/license/Apla-Toolbox/pymapf.svg)](https://github.com/Apla-Toolbox/pymapf/blob/master/LICENSE)
[![GitHub contributors](https://img.shields.io/github/contributors/Apla-Toolbox/pymapf.svg)](https://GitHub.com/Apla-Toolbox/pymapf/graphs/contributors/)

</div>

<div align="center">
    
[Report Bug](https://github.com/APLA-Toolbox/pymapf/issues) · [Request Feature](https://github.com/APLA-Toolbox/pymapf/issues)

Loved the project? Please consider [donating](https://www.buymeacoffee.com/dq01aOE) to help it improve!

</div>

## Features 🌱

- ✨ Built to be expanded: easy to add new planners
- 🖥️ Supported on Ubuntu
- 🎌 Built with Python
- 🔎 Reactive Distributed Planners (Nonlinear Model Predictive Control, Velocity Obstacles)
- 🧭 Centralized Planners (Space-Time A*)
- 📊 Benchmark Tools (Incoming...)
- 🍻 Maintained (Incoming: Enhanced Conflict-Based Search, Local-Repair A*, Replanning RRT*...)

<div align="center">
    
<img src="https://user-images.githubusercontent.com/43545812/104828684-56bef700-586c-11eb-83d4-2763831d4155.gif" alt="Logo" width="300">     
    
</div>

## Dependencies 🖇️

- Install Python (3.7.5 is the tested version)
- Install Pip: `sudo apt install python3-pip`
- Upgrade Pip: `python3 -m pip install --upgrade pip`

### Using the repository 💾

- Clone the repo: `git clone https://github.com/apla-toolbox/pymapf`
- Cd into the repo `cd pymapf`
- Install requirements: `python3 -m pip install -r requirements.txt`

### Using the pip package 📦

- Install the package: `python3 -m pip install pymapf`

## Usage 📑

### Scripts 💨

Launch hub switch scripts using:
- `python3 scripts/switch_positions_nmpc.py`
- `python3 scripts/switch_positions_vel_obstacles.py` (broken)

More to come...

### Library 🗺️

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

## Cite 📰

If you use the project in your work, please consider citing it with:
```
@misc{https://doi.org/10.13140/rg.2.2.14030.28486,
  doi = {10.13140/RG.2.2.14030.28486},
  url = {http://rgdoi.net/10.13140/RG.2.2.14030.28486},
  author = {Erwin Lejeune and Sampreet Sarkar},
  language = {en},
  title = {Survey of the Multi-Agent Pathfinding Solutions},
  publisher = {Unpublished},
  year = {2021}
}
```

List of publications & preprints using `pymapf` (please open a pull request to add missing entries):

* [Survey of MAPF solutions](https://www.researchgate.net/publication/348716625_Survey_of_the_Multi-Agent_Pathfinding_Solutions) (January 2021)

## Contribute 🆘

Open an issue to state clearly the contribution you want to make. Upon aproval send in a PR with the Issue referenced. (Implement Issue #No / Fix Issue #No).

## Maintainers Ⓜ️

- Erwin Lejeune
- Sampreet Sarkar
