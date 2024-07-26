State diagrams are generated using a tool called Graphviz (https://graphviz.org/about/)
The "generate_dot.py" script generates a DOT file, which defines a graph 
Currently, FMPT's, sojourn times, and transition probabilities must be pasted into this file
This script can be run for each Markov chain, though a little work is necessary to include and exclude the appropriate nodes and transitions
You may need to modify the DOT file to make it look nicer
From there, the generate_all_graphs.sh script can be run to generate PNG images from all DOT files in the directory
Depending on the OS you're on, this may require modification - or you can run each command by hand

Of course, you may find a better approach to all of this, especially with the changes being requested