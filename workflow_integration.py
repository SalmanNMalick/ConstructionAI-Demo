# Additional: Integration with Construction Workflows (Bonus)
# Using NetworkX for workflow graphs, simulating project management tools.

import networkx as nx
import matplotlib.pyplot as plt

# Build a directed graph for construction workflow
G = nx.DiGraph()
G.add_edges_from([
    ("Planning", "Foundation"),
    ("Foundation", "Framing"),
    ("Framing", "Roofing"),
    ("Roofing", "Inspection")
])

# Add AI-driven attributes (e.g., agent assignments)
nx.set_node_attributes(G, {"Planning": {"agent": "PlannerAgent"}, "Inspection": {"agent": "ReasonerAgent"}})

# Visualize and analyze (e.g., critical path)
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
plt.show()

# Shortest path for planning
critical_path = nx.shortest_path(G, "Planning", "Inspection")
print("Critical Path:", critical_path)
