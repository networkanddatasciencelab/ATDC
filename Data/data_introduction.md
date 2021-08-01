# Data Introduction

This data is used in paper "Anomalous Trajectory Detection and Classification Based on Difference and Intersection Set Distance".

[1] J. Wang et al., "Anomalous Trajectory Detection and Classification Based on Difference and Intersection Set Distance," in IEEE Transactions on Vehicular Technology, vol. 69, no. 3, pp. 2487-2500, March 2020, doi: 10.1109/TVT.2020.2967865.

It is divided into six sub-database. Each one contains diagrams of trajectories sharing the same pair of origin and destination.

Trajectories are labeled in 5 categories.
0 - global detour (GD)
1 - local detour (LD)
2 - normal trajectory (NT)
3 - local shortcut (LS)
4 - global shortcut (GS)

Each subfolder './pic' contains trajectory diagram. Each diagram is transformed from original grid sequence of trajectory, which can be found in X_Label.csv.
X_Label.csv contains label and index of trajectories.
X_TrajectoryGrid.csv contains index and trajectory grid sequence.
(X=1,2,3,4,5,6)
