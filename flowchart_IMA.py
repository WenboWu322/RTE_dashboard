import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arrow

def add_box(ax, text, xy, boxstyle="round,pad=0.3", box_color="lightblue", text_color="black"):
    """Add a box with text to the plot."""
    ax.add_patch(FancyBboxPatch(
        xy,  # bottom-left corner of the box
        width=3.5, height=1.2,
        boxstyle=boxstyle,
        edgecolor="black",
        facecolor=box_color,
    ))
    ax.text(xy[0] + 1.75, xy[1] + 0.6, text, color=text_color,
            ha="center", va="center", fontsize=10)

def add_arrow(ax, start, end):
    """Add an arrow between two points."""
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(facecolor="black", arrowstyle="->"))

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis("off")  # Turn off the axis

# Add boxes for each step
add_box(ax, "Start: Initialize Parameters\n(window_size, global_task_count)", (3, 10))
add_box(ax, "Select Sliding Window\n(window_start, window_end)", (3, 8))
add_box(ax, "Filter Tasks in Current Window\n(DATE >= window_start & DATE < window_end)", (3, 6))
add_box(ax, "Preprocess Tasks\n- Convert to Relative Time\n- Extract First Tasks per System", (3, 4))
add_box(ax, "Cluster First Tasks\n- Assign to Centers\n- Update Global Task Counts", (3, 2))
add_box(ax, "Update Secondary Tasks\n- Adjust Times for Related Tasks\n- Update CENTER Column", (3, 0))

# Add decision diamond
ax.add_patch(FancyBboxPatch(
    (7.5, 6), width=3.5, height=1.2,
    boxstyle="round,pad=0.3", edgecolor="black",
    facecolor="lightgreen"))
ax.text(9.25, 6.6, "End of Timeline?", color="black",
        ha="center", va="center", fontsize=10)

# Add final output box
add_box(ax, "Finalize Remaining Tasks\nOutput Final Maintenance Plan", (7.5, 8), box_color="lightyellow")

# Add arrows between steps
add_arrow(ax, (4.75, 10), (4.75, 9.2))  # Start -> Select Sliding Window
add_arrow(ax, (4.75, 8), (4.75, 7.2))   # Select Sliding Window -> Filter Tasks
add_arrow(ax, (4.75, 6), (4.75, 5.2))   # Filter Tasks -> Preprocess Tasks
add_arrow(ax, (4.75, 4), (4.75, 3.2))   # Preprocess Tasks -> Cluster First Tasks
add_arrow(ax, (4.75, 2), (4.75, 1.2))   # Cluster First Tasks -> Update Secondary Tasks

# Add arrow to decision point
add_arrow(ax, (4.75, 6), (8.25, 6))     # Filter Tasks -> End of Timeline?

# Add loop-back arrow for "No"
ax.annotate("",
            xy=(4.75, 9.2),  # End of decision point (No)
            xytext=(3, 8),   # Back to Select Sliding Window
            arrowprops=dict(facecolor="red", arrowstyle="->"))

plt.show()