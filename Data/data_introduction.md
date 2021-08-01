# Data Introduction

This data is used in paper "Anomalous Trajectory Detection and Classification Based on Difference and Intersection Set Distance".

[1] J. Wang et al., "Anomalous Trajectory Detection and Classification Based on Difference and Intersection Set Distance," in IEEE Transactions on Vehicular Technology, vol. 69, no. 3, pp. 2487-2500, March 2020, doi: 10.1109/TVT.2020.2967865.
https://ieeexplore.ieee.org/document/8963673

It contains six datasets. Each one contains diagrams and labels of trajectories sharing the same pair of origin and destination.

Trajectories are labeled in 5 categories.
0 - global detour (GD)
1 - local detour (LD)
2 - normal trajectory (NT)
3 - local shortcut (LS)
4 - global shortcut (GS)

Each subfolder './pic' contains trajectory diagrams. Each diagram is transformed from original grid sequence of trajectory, which can be found in X_Label.csv.
X_Label.csv contains the label and index of trajectories from the same pair of origin and destination.
X_TrajectoryGrid.csv contains the index and grid sequence of trajectories.
(X=1,2,3,4,5,6)
