import matplotlib as mpl
import os
if "DISPLAY" not in os.environ:
    mpl.use("agg")
else:
    mpl.use("TkAgg")
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# SAME ENVIRONMENT (Spot Swapping)

# CBS vs NMPC: Total Runtime w.r.t. number of agents
# CBS vs NMPC: Total Runtime w.r.t. search space
# CBS vs Worst Agent NMPC: Total Runtime w.r.t. number of agents

agents = [2, 4, 8]
spaces = [100, 144, 256, 400, 2500]

nmpc_runtime_agents_scale = [
    3.8522963523864746,
    7.887270927429199,
    19.57982873916626,
]
nmpc_runtime_space_scale = [
    8.909427881240845, 
    10.181878805160522,
    7.111235618591309, 
    9.221412420272827, 
    7.429507255554199,
]

cbs_runtime_agents_scale = [
    0.11863327026367188,
    1.1745281219482422,
    48.70840406417847
]

cbs_runtime_space_scale = [
    0.4471933841705322,
    0.5206875801086426,
    0.6985411643981934,
    1.171295166015625,
    6.556893348693848,
]

nmpc_runtime_per_agents_agents_scale = [
    {'1': 2.1708545684814453, '2': 1.6804776191711426}, 
    {'1': 1.9971742630004883, '2': 2.4954023361206055, '3': 1.4243495464324951, '4': 1.9686675071716309}, 
    {'1': 2.137916088104248, '2': 1.9024684429168701, '3': 3.3005177974700928, '4': 2.718195676803589, '5': 1.6852214336395264, '6': 2.306485891342163, '7': 2.455350637435913, '8': 3.069617748260498}
]
nmpc_worst_agent_agents_scale = [2.1708545684814453, 2.4954023361206055, 3.3005177974700928]

nmpc_runtime_per_agents_space_scale = [
    {'1': 1.2272868156433105, '2': 1.5924506187438965, '3': 3.7263431549072266, '4': 2.3616256713867188}, 
    {'1': 1.192922592163086, '2': 3.030954360961914, '3': 3.6219232082366943, '4': 2.334315061569214}, 
    {'1': 1.1881194114685059, '2': 1.6314845085144043, '3': 1.9552299976348877, '4': 2.3347766399383545}, 
    {'1': 1.995544195175171, '2': 1.6433308124542236, '3': 2.023364782333374, '4': 3.557486057281494}, 
    {'1': 1.1526515483856201, '2': 1.796224594116211, '3': 2.172724723815918, '4': 2.306177854537964}
]
nmpc_worst_agent_space_scale = [3.7263431549072266, 3.6219232082366943, 2.3347766399383545, 3.557486057281494, 2.306177854537964]

def plot1():
    _, ax = plt.subplots()
    plt.xlabel("Total Runtime")
    plt.ylabel("Number of Agent(s)")
    ax.plot(agents, nmpc_runtime_agents_scale, "-o", label="NMPC")
    ax.plot(agents, cbs_runtime_agents_scale, "-o", label="CBS")
    plt.title("Complexity results for Agents performing Hub swaps")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show(block=False)

def plot2():
    _, ax = plt.subplots()
    plt.xlabel("Search Space (x*y)")
    plt.ylabel("Search Space Size")
    ax.plot(spaces, nmpc_runtime_space_scale, "-o", label="NMPC")
    ax.plot(spaces, cbs_runtime_space_scale, "-o", label="CBS")
    plt.title("Complexity results for Agents performing Hub swaps")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show(block=False)

def plot3():
    _, ax = plt.subplots()
    plt.xlabel("Total Runtime")
    plt.ylabel("Number of Agent(s)")
    ax.plot(agents, nmpc_worst_agent_agents_scale, "-o", label="NMPC (Worst Agent Perf)")
    ax.plot(agents, cbs_runtime_agents_scale, "-o", label="CBS")
    plt.title("Complexity results for Agents performing Hub swaps")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot1()
    plot2()
    plot3()
