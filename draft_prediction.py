import numpy as np


class RelevancePrediction:
    """
    This class represents the relevance-based prediction model.
    These functions implicitly show that the likelihood of an observation
    from a multivariate normal distribution is proportional to the exponential
    of the negative Mahalanobis distance, as the Mahalanobis distance is used to
    determine how "likely" or "similar" an observation is in the context of the given model.

    Relevant section: Section 2.1

    relevant text from paper:
    Relevance has three components:
    the similarity of a previously drafted player to the draft prospect (similarity function)
    the informativeness of the previously drafted player (informativeness i)
    and the informativeness of the draft prospect (informativeness j)
    """

    def __init__(self, predictive_variables, inverse_cov_matrix, average_values):
        """
        Initialize with the predictive variables, inverse covariance matrix, and average values.

        :param predictive_variables: Dictionary containing the predictive variables for all players.
                                     Format: {player_id: np.array([variables])}(xₜ)
        :param average_values: Average values of the predictive variables (x̄)
        :param inverse_cov_matrix: Inverse covariance matrix (Ω^-1)
        """
        self.predictive_variables = predictive_variables
        self.inverse_cov_matrix = inverse_cov_matrix
        self.average_values = average_values

    def similarity(self, x_i, x_j):
        """
        Calculates the negative Mahalanobis distance between two players' predictive variables,
        directly relating to the likelihood of one player's predictive variables given another's.

        :param x_i: Predictive variables for player i (np.array)
        :param x_j: Predictive variables for player j (np.array)
        :return: Similarity score
        sim will transpose the covariance matrix though you may have to apply
        an epsilon to the diagonal to prevent the matrix from being non invertable
        """
        diff = x_i - x_j
        sim = -0.5 * np.dot(np.dot(diff, self.inverse_cov_matrix), diff.T)
        """
        Relevant text for sim score:
        Notice that in the formula for similarity we multiply the Mahalanobis distance of a previously
        drafted player from the draft prospect by negative one half.
        The negative sign converts a measure of difference into a measure of similarity.
        We multiply by one half because the average squared distances between pairs of players is twice
        as large as the players average squared differences from the average of all players.

        the sim score will always be negative as it is a measure of similarity instead of a measure
        of difference. the closer to 0 (e.g. the less negative) the more similar  they are
        """
        return sim

    def informativeness(self, x):
        """
        Calculates the Mahalanobis distance of a player's predictive variables from the average,
        indicating how typical or atypical the player's characteristics are.

        :param x: Predictive variables for the player (np.array)
        :return: Informativeness score
        """
        diff = x - self.average_values
        info = np.dot(np.dot(diff, self.inverse_cov_matrix), diff.T)
        return info

    def relevance(self, x_i, x_j):
        """
        Calculate relevance of a previously drafted player to a draft prospect.
        
        relevant paper quote:
        All else being equal, previously drafted players who are like the draft prospect
        but different from the average of all previously drafted players are more relevant
        to a prediction than those who are not.

        :param x_i: Predictive variables for previously drafted player i (np.array)
        :param x_j: Predictive variables for draft prospect j (np.array)
        :return: Relevance score
        """
        sim = self.similarity(x_i, x_j)
        info_i = self.informativeness(x_i)
        info_j = self.informativeness(x_j)
        relevance = sim + 0.5 * (info_i + info_j)
        return relevance


class FitCalculation:
    """
    This class represents the fit calculation for the relevance-based prediction model.

    Relevant section: Section 2.2
    """

    def __init__(self, weights, outcomes):
        """
        Initialize with the weights and outcomes for players.

        :param weights: Dictionary of weights for players. Format: {player_id: float}
        in this context, the weights are the relevance scores by player
        you should pass the dict of player:relevance as the weight dict in here
        
        :param outcomes: Dictionary of outcomes for players. Format: {player_id: float}
        """
        self.weights = weights
        self.outcomes = outcomes

    def calculate_fit(self):
        """
        Calculate the fit for the relevance-based prediction.
        steps:
        1. Loop through each pair of players (i, j).
        2. For each pair (i, j), calculate the product of their weights
        3. For each pair (i, j), calculate the product of their outcomes
        4. Sum the products of weights and outcomes for all pairs to get the fit score

        :return: Fit score
        """
        fit = 0
        players = list(self.weights.keys())
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                weight_product = self.weights[players[i]] * self.weights[players[j]]
                outcome_product = self.outcomes[players[i]] * self.outcomes[players[j]]
                fit += weight_product * outcome_product
        return fit


class AdjustedFitCalculation(FitCalculation):
    """
    This class represents the calculation of adjusted fit, including asymmetry.
    
    Section 2.3: Codependence
    """

    def __init__(self, weights, outcomes, num_predictive_variables):
        """
        Initialize with weights, outcomes, and number of predictive variables.
        
        :param weights: Dictionary of weights for players. Format: {player_id: float}
        :param outcomes: Dictionary of outcomes for players. Format: {player_id: float}
        :param num_predictive_variables: Number of predictive variables (K)
        we super the weights/outcomes to pull from the parent FitCalculation class
        this ensures that we are not re-instantiating those variables and are, instead,
        passing them along to the helper classes/functions
        """
        super().__init__(weights, outcomes)
        self.num_predictive_variables = num_predictive_variables

    def calculate_asymmetry(self):
        """
        Calculate the asymmetry between retained and censored subsamples.
        
        :return: Asymmetry score
        see calculation 18 in section 2.3
        """
        retained_weights = [w for w in self.weights.values() if w > 0]
        censored_weights = [w for w in self.weights.values() if w <= 0]
        retained_fit = np.sum(retained_weights)
        censored_fit = np.sum(censored_weights)
        asymmetry = 0.5 * (retained_fit - censored_fit) ** 2
        return asymmetry

    def calculate_adjusted_fit(self):
        """
        Calculate the adjusted fit for the relevance-based prediction.
        
        :return: Adjusted fit score
        """
        fit = self.calculate_fit()
        asymmetry = self.calculate_asymmetry()
        adjusted_fit = self.num_predictive_variables * (fit + asymmetry)
        return adjusted_fit



class CompositePrediction:
    """
    This class represents the composite prediction calculation.

    Relevant section: Section 2.3
    """

    def __init__(self, predictions, adjusted_fits):
        """
        Initialize with the predictions and adjusted fits for each calibration.

        :param predictions: Dictionary of predictions for each calibration. Format: {calibration_id: prediction_value}
        :param adjusted_fits: Dictionary of adjusted fits for each calibration. Format: {calibration_id: adjusted_fit_value}
        """
        self.predictions = predictions
        self.adjusted_fits = adjusted_fits

    def calculate_reliability_weights(self):
        """
        Calculate the reliability weights for each calibration.

        :return: Dictionary of reliability weights. Format: {calibration_id: reliability_weight}
        """
        total_adjusted_fit = sum(self.adjusted_fits.values())
        reliability_weights = {calibration: fit / total_adjusted_fit for calibration, fit in self.adjusted_fits.items()}
        return reliability_weights

    def calculate_composite_prediction(self):
        """
        Calculate the composite prediction based on reliability weights.

        :return: Composite prediction value
        """
        reliability_weights = self.calculate_reliability_weights()
        composite_prediction = sum(
            reliability_weights[calibration] * self.predictions[calibration]
            for calibration in self.predictions
        )
        return composite_prediction
