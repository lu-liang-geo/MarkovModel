import unittest
import classic_markov
import helper_utils
import pandas


class TestMarkovChunking(unittest.TestCase):
    def setUp(self):
        self.good = "Good"
        self.moderate = "Moderate"
        self.unhealthy_sensitive_groups = "Unhealthy for Sensitive Groups"
        self.unhealthy = "Unhealthy"
        self.very_unhealty = "Very Unhealthy"
        self.hazardous = "Hazardous"

    def test_encoding(self):
        pm25_data = [0, 6.79, 11.5, 12.3, 27.6,
                     32.1, 35.5, 52, 75.3, 169.0021, 273.2]
        df = pandas.DataFrame(pm25_data, columns=["PM2.5 (ATM)"])
        df["pm25_state"] = df.apply(helper_utils.encode_categories, axis=1)

        expected_states = [self.good, self.good, self.good, self.moderate, self.moderate, self.moderate,
                           self.unhealthy_sensitive_groups, self.unhealthy_sensitive_groups,
                           self.unhealthy, self.very_unhealty, self.hazardous]

        self.assertEqual(df["pm25_state"].to_list(), expected_states)

    def assert_chunks_equal(self, raw_data, expected_chunks):
        df = pandas.DataFrame(
            raw_data, columns=["Timestamp", "PM2.5 (ATM)", "Location ID"])
        chunks = helper_utils.obtain_continuous_chunks_daily(df)
        self.assertEqual(chunks, expected_chunks)

    def test_chunk_splitting(self):
        # Cases:
        # 1. No missing data - done, 1234
        # 2. Missing 1 day partway through - done, 5678
        # 3. Missing 1 day at end - done, 1357
        # 4 Missing 1 day at beginning - done, 2468
        # 5 Alternating missing days - done, 1278
        # 6

        data_no_missing = [["2021-04-25 00:00:00-05:00", 15.2, 1234],
                           ["2021-04-26 00:00:00-08:00", 17.1, 1234],
                           ["2021-04-27 00:00:00-05:00", 16.4, 1234],
                           ["2021-04-28 00:00:00-06:00", 11.8, 1234],
                           ["2021-04-29 00:00:00-05:00", 11.6, 1234]]
        expected_chunks_no_missing = [
            [self.moderate, self.moderate, self.moderate, self.good, self.good]]
        self.assert_chunks_equal(data_no_missing, expected_chunks_no_missing)

        data_middle_missing = [["2021-04-25 00:00:00-05:00", 3.2, 5678],
                               ["2021-04-26 00:00:00-05:00", 5.8, 5678],
                               ["2021-04-28 00:00:00-05:00", 12.2, 5678],
                               ["2021-04-29 00:00:00-05:00", 11.4, 5678]]
        expected_chunks_middle_missing = [
            [self.good, self.good], [self.moderate, self.good]]
        self.assert_chunks_equal(data_middle_missing, expected_chunks_middle_missing)

        data_end_missing = [["2021-04-25 00:00:00-05:00", 1.3, 1357],
                            ["2021-04-26 00:00:00-05:00", 2.1, 1357],
                            ["2021-04-27 00:00:00-05:00", 5.5, 1357],
                            ["2021-04-28 00:00:00-05:00", 6.1, 1357]]
        expected_chunks_end_missing = [
            [self.good, self.good, self.good, self.good]]
        self.assert_chunks_equal(data_end_missing, expected_chunks_end_missing)

        data_beginning_missing = [["2021-04-26 00:00:00-05:00", 32.1, 2468],
                                  ["2021-04-27 00:00:00-05:00", 34.9, 2468],
                                  ["2021-04-28 00:00:00-05:00", 35.6, 2468],
                                  ["2021-04-29 00:00:00-05:00", 34.7, 2468]]
        expected_chunks_beginning_missing = [[self.moderate, self.moderate, self.unhealthy_sensitive_groups,
                                              self.moderate]]
        self.assert_chunks_equal(data_beginning_missing,
                           expected_chunks_beginning_missing)

        data_alternating_missing = [["2021-04-25 00:00:00-05:00", 60.1, 1278],
                                    ["2021-04-27 00:00:00-05:00", 42.3, 1278],
                                    ["2021-04-29 00:00:00-05:00", 34.1, 1278]]
        expected_chunks_alternating_missing = [[self.unhealthy], [self.unhealthy_sensitive_groups],
                                               [self.moderate]]
        self.assert_chunks_equal(data_alternating_missing,
                           expected_chunks_alternating_missing)

    def assert_matrices_equal(self, matrix1, matrix2):
        self.assertEqual(len(matrix1),
                         len(matrix2))

        for i in range(len(matrix1)):
            self.assertEqual(len(matrix1[i]), len(
                matrix2[i]))

            for j in range(len(matrix1[i])):
                self.assertAlmostEqual(
                    matrix1[i][j], matrix2[i][j], delta=0.01)

    def test_markov_chunks(self):
        # Cases:
        # 1. One contiguous chunk
        # 2. Two contiguous chunks
        # 3. Multiple chunks, different lengths
        
        #Must also cover chunks with only 1 entry

        one_contiguous_chunk = [[self.good, self.good, self.good, self.moderate, self.moderate,
                                        self.good, self.good, self.good, self.moderate, self.unhealthy_sensitive_groups,
                                        self.unhealthy_sensitive_groups, self.moderate, self.moderate,
                                        self.moderate, self.good, self.good, self.good, self.good]]
        transition_matrix = classic_markov.run_classic_markov_chunks(one_contiguous_chunk,
                                                                     "contiguous_one_sensor", False)
        expected_transition_matrix = [[0.778, 0.222, 0, 0, 0, 0],
                                      [0.333, 0.5, 0.167, 0, 0, 0],
                                      [0, 0.5, 0.5, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0]]
        self.assert_matrices_equal(
            transition_matrix, expected_transition_matrix)
        
        two_contiguous_chunks = [[self.good, self.good, self.good, self.moderate, self.moderate,
                                        self.good, self.good, self.good, self.moderate, self.unhealthy_sensitive_groups,
                                        self.unhealthy_sensitive_groups, self.moderate, self.moderate,
                                        self.moderate, self.good, self.good, self.good, self.good],
                                              [self.moderate, self.moderate, self.unhealthy_sensitive_groups,
                                        self.unhealthy_sensitive_groups, self.moderate, self.moderate,
                                        self.moderate, self.moderate, self.good, self.good, self.moderate]]
        transition_matrix = classic_markov.run_classic_markov_chunks(two_contiguous_chunks,
                                                                     "contiguous_multiple_sensor", False)
        expected_transition_matrix = [[0.727, 0.273, 0, 0, 0, 0],
                                      [0.25, 0.583, 0.167, 0, 0, 0],
                                      [0, 0.5, 0.5, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0]]
        self.assert_matrices_equal(
            transition_matrix, expected_transition_matrix)
        
        multiple_chunks = [[self.good, self.good, self.good, self.good, self.moderate, self.moderate],
                           [self.unhealthy_sensitive_groups, self.moderate, self.unhealthy_sensitive_groups,
                            self.unhealthy_sensitive_groups, self.unhealthy, self.unhealthy,
                            self.unhealthy_sensitive_groups, self.unhealthy_sensitive_groups, self.moderate],
                           [self.good, self.good, self.good], [self.good], [self.moderate, self.moderate],
                           [self.good, self.good, self.good, self.good, self.good, self.moderate, self.moderate,
                            self.good, self.good, self.moderate, self.moderate, self.unhealthy_sensitive_groups,
                            self.good]]
        transition_matrix = classic_markov.run_classic_markov_chunks(multiple_chunks,
                                                                     "multiple_chunks", False)
        expected_transition_matrix = [[0.769, 0.231, 0, 0, 0, 0],
                                      [0.143, 0.571, 0.286, 0, 0, 0],
                                      [0.167, 0.333, 0.333, 0.167, 0, 0],
                                      [0, 0, 0.5, 0.5, 0, 0],
                                      [0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0]]
        self.assert_matrices_equal(
            transition_matrix, expected_transition_matrix)


if __name__ == '__main__':
    unittest.main()
