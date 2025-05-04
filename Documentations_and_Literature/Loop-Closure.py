!pip install open3d
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Create Pose Graph
pose_graph = o3d.pipelines.registration.PoseGraph()

# Add 5 nodes (poses)
pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
for i in range(1, 5):
    T = np.eye(4)
    T[0, 3] = i * 1.0
    # The following line was changed to append a PoseGraphNode object
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(T))

# Create Pose Graph
pose_graph = o3d.pipelines.registration.PoseGraph()

# Add 5 nodes (poses)
pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
for i in range(1, 5):
    T = np.eye(4)
    T[0, 3] = i * 1.0
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(T))

# Add odometry edges
for i in range(4):
    T = np.eye(4)
    T[0, 3] = 1.0
    pose_graph.edges.append(
        o3d.pipelines.registration.PoseGraphEdge(i, i+1, T, uncertain=False)
    )

# Add loop closure edge (node 4 → node 0)
loop_T = np.eye(4)
loop_T[0, 3] = -4.0
pose_graph.edges.append(
    o3d.pipelines.registration.PoseGraphEdge(4, 0, loop_T, uncertain=True)
)

# Global optimization
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=1.0,
    edge_prune_threshold=0.25,
    reference_node=0
)
o3d.pipelines.registration.global_optimization(
    pose_graph,
    o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
    o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
    option
)

# Plot optimized poses
x_vals = [node.pose[0, 3] for node in pose_graph.nodes]
y_vals = [node.pose[1, 3] for node in pose_graph.nodes]

plt.plot(x_vals, y_vals, 'o-', label='Optimized Poses')
plt.title("Loop Closure Optimization Result")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()
