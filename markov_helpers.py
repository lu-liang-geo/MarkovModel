from dataclasses import dataclass
import copy
import numpy

@dataclass
class ClassicMarkovResult:
    transition_count_matrix: list
    transition_prob_matrix: list
    transition_sample_sizes: list


#Generate a list of transition count matrices
#Where rows with low sample size in any entry
#Are zeroed out in all entries to eliminate their effect
def generate_adjusted_matrix_list(results_list):
    # For each test, if any row for a season has low sample size, zero out that row in all seasonal matrices
    # This way, transitions with too low a sample size won't throw off test results

    indices_to_zero_out = []
    threshold = 150
    matrix_list = []

    for result in results_list:
        matrix_list.append(result.transition_count_matrix)

        for i in range(len(result.transition_sample_sizes)):
            if result.transition_sample_sizes[i] < threshold:
                indices_to_zero_out.append(i)

    matrix_size = len(result.transition_sample_sizes)
    zero_row = [0 for i in range(matrix_size)]

    for index in indices_to_zero_out:
        for j in range(len(matrix_list)):
            matrix_list[j][index] = zero_row

    return matrix_list


#Remove states with low sample size from a transition probability matrix
#This may be useful in determining steady state distributions
def remove_low_sample_size_states(result: ClassicMarkovResult, threshold=150):
    indices_to_remove = []
    for i in range(len(result.transition_sample_sizes)):
        if result.transition_sample_sizes[i] < threshold:
            indices_to_remove.append(i)
            
    indices_to_remove.reverse()
    
    count_matrix = copy.deepcopy(result.transition_count_matrix.tolist())
    for index in indices_to_remove:
        del count_matrix[index]

        for i in range(len(count_matrix)):
            del count_matrix[i][index]
            
    prob_matrix = []
    count_matrix = numpy.array(count_matrix)
    for transition_count_row in count_matrix:
        total_transitions = transition_count_row.sum()
        transition_probability_row = transition_count_row

        if total_transitions > 0:
            transition_probability_row = numpy.multiply(
                transition_probability_row, 1.0/total_transitions)

        #print(str(transition_probability_row) + " | " + str(total_transitions))
        prob_matrix.append(transition_probability_row.tolist())
        #transition_sample_sizes.append(total_transitions)
            
    return prob_matrix
            
    