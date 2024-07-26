def calc_fmpt_width(i, j):
    fmpt = [[1.526, 4.572, 53.387, 1064.642, "inf", "inf"],
            [2.238, 3.073, 54.437, 1066.880, "inf", "inf"],
            [2.483, 2.598, 54.109, 1067.125, "inf", "inf"],
            [3.238, 1.000, 55.437, 1067.880, "inf", "inf"],
            ["inf", "inf", "inf", "inf", 1.000, "inf"],
            ["inf", "inf", "inf", "inf", "inf", 1.000]]

    base_pen_width = 0.4

    return min(8, base_pen_width * fmpt[i][j])


def calc_node_size(i):
    sojourn_times = [1, 3, 5, "inf", "inf", "inf"]

    base_size = 0.5

    return min(4, base_size * sojourn_times[i])


file_path = "reference_daily.dot"

transition_matrix = [[0.768, 0.208, 0.023, 0.001, 0.000, 0.000],
                    [0.448, 0.542, 0.010, 0.000, 0.000, 0.000],
                    [0.339, 0.643, 0.018, 0.000, 0.000, 0.000],
                    [0.000, 1.000, 0.000, 0.000, 0.000, 0.000],
                    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000]]

with open(file_path, 'w') as file:
    file.write('digraph {\n')

    file.write(
        f'G -> G [label="{transition_matrix[0][0]}", fontsize="90", penwidth="{calc_fmpt_width(0, 0)}"];\n')
    file.write(
        f'G -> M[label="{transition_matrix[0][1]}", fontsize="90", penwidth="{calc_fmpt_width(0, 1)}"];\n')
    file.write(
        f'G -> S [label="{transition_matrix[0][2]}", fontsize="90", penwidth="{calc_fmpt_width(0, 2)}"];\n')

    node_size = calc_node_size(0)
    file.write(f'G [width="{node_size}", height="{node_size}", fontsize="90", style="filled", fillcolor="lightgrey", label=< <B>G</B> >];\n')

    file.write(
        f'M -> G [label="{transition_matrix[1][0]}", fontsize="90", penwidth="{calc_fmpt_width(1, 0)}"];\n')
    file.write(
        f'M -> M [label="{transition_matrix[1][1]}", fontsize="90", penwidth="{calc_fmpt_width(1, 1)}"];\n')
    file.write(
        f'M -> S [label="{transition_matrix[1][2]}", fontsize="90", penwidth="{calc_fmpt_width(1, 2)}"];\n')
    
    node_size = calc_node_size(1)
    file.write(f'M [width="{node_size}", height="{node_size}", fontsize="90", style="filled", fillcolor="lightgrey", label=< <B>M</B> >];\n')

    file.write(
        f'S -> G [label="{transition_matrix[2][0]}", fontsize="90", penwidth="{calc_fmpt_width(2, 0)}"];\n')
    file.write(
        f'S -> M [label="{transition_matrix[2][1]}", fontsize="90", penwidth="{calc_fmpt_width(2, 1)}"];\n')
    file.write(
        f'S -> S [label="{transition_matrix[2][2]}", fontsize="90", penwidth="{calc_fmpt_width(2, 2)}"];\n')
    # file.write(
    #     f'S -> U [label="{transition_matrix[2][3]}", fontsize="90", penwidth="{calc_fmpt_width(2, 3)}"];\n')

    node_size = calc_node_size(2)
    file.write(f'S [width="{node_size}", height="{node_size}", fontsize="90", style="filled", fillcolor="lightgrey", label=< <B>S</B> >];\n')
    
    # file.write(
    #     f'U -> G [label="{transition_matrix[3][0]}", fontsize="90", penwidth="{calc_fmpt_width(3, 0)}"];\n')
    # file.write(
    #     f'U -> M [label="{transition_matrix[3][1]}", fontsize="90", penwidth="{calc_fmpt_width(3, 1)}"];\n')
    # file.write(
    #     f'U -> S [label="{transition_matrix[3][2]}", fontsize="90", penwidth="{calc_fmpt_width(3, 2)}"];\n')
    # file.write(
    #     f'U -> U [label="{transition_matrix[3][3]}", fontsize="90", penwidth="{calc_fmpt_width(3, 3)}"];\n')

    # node_size = calc_node_size(3)
    # file.write(f'U [width="{node_size}", height="{node_size}", fontsize="90", style="filled", fillcolor="lightgrey", label=< <B>U</B> >];\n')

    file.write('size="14,14"\n')
    file.write('ratio="2"')

    file.write("}")
