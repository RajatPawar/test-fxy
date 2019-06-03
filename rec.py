import os
from sklearn.neighbors import NearestNeighbor
import pandas as pnd
import fuzzywuzzy as fuzzy

class Recommender:

    def __init__(self, products_set_path, ratings_set_path, min_product_interactions, string_match_threshold):
        # Product information dataset with relevant metadata
        self.products_set_path = products_set_path
        # Product user interaction dataset (Ratings, etc)
        self.ratings_set_path = ratings_set_path
        # Min number of user interactions product must have to count in model
        self.min_product_interactions = min_product_interactions
        # We use the K nearest neighbors model to classify
        self.model = NearestNeighbour()
        # Threshold to match given product name to our list of products (60+)
        self.string_match_threshold = string_match_threshold

    def set_model_parameters(self, k_value, algorithm, metric_to_use, parallel_jobs):
        # Make sure if we allow running parallel jobs, then we give it a scratchpad
        if parallel_jobs and (parallel_jobs > 1 or parallel_jobs == -1):
            os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp';
        self.model.set_model_params({
            # How many neigbors to gather to get a classification vote?
            'n_neighbors': k_value,
            # Which algo to use to iterate through points to get neighbors?
            'algorithm': algorithm,
            # Which metric (cosine, euclidean distance) to use to get distance?
            'metric': metric_to_use,
            # How many parallel jobs for this task? -1 = all processors
            'n_jobs': parallel_jobs
        })

    def prepare_clean_data(self):
        # Load product dataset into a dataframe
        # Use only product ID and the product name as features after specifying their datatype
        products_dataframe = pnd.read_csv(os.path.join(self.products_set_path), usecols=['productId', 'name'], dtype={'productId': 'int32', 'name': 'str'})
        # Load ratings dataset into a dataframe
        # Use only product ID, user ID and rating given as features for our model
        ratings_dataframe = pnd.read_csv(os.path.join(self.ratings_set_path), usecols=['productId', 'userId', 'rating'], dtype={'productId': 'int32', 'userId': 'int32', 'rating': 'float32'})

        # Steps to clean data -
        # Eliminate inactive users from dataset
        # Only take products with count(ratings) > min_product_interactions

        # The below DF helps us get products with more than > min_product_interactions number of ratings
        active_products_count_dataframe = pnd.DataFrame(ratings_dataframe.groupby('productId').size(), columns=['number_of_ratings'])
        active_products_dataframe = active_products_count_dataframe.query('number_of_ratings > @self.min_product_interactions')

        # got to finish this

    def string_match(self, prodname_index_map, product_name):
        # Uses the fuzzywuzzy lib to return closest match to product name in the map
        all_matches = []

        # For all products in the map, get all similar strings & append into our list
        for name, index in prodname_index_map.items():
            match_percent = fuzzy.ratio(name.tolower(), product_name)
            if match_percent > self.string_match_threshold:
                all_matches.append((name, index, match_percent))

        # Existential check
        if not all_matches:
            print("ERROR: No matches found for that product")
        else:
            print("Matches - we've got them!")
            # Sort according to similarity fuzzy ratio and present it
            all_matches = sorted(all_matches, key = lambda element : element[2], reverse = True)
            # Print movie titles of all similar products
            print('{0}'.format([row[0] for row in all_matches]))
            # Return the MOST similar product's index
            return all_matches[0][1]

    def infer(self, prod_user_matrix, prodname_index_map, root_product, num_of_recommendations):
        # Study the data, and use the values to understand mathematical relationships
        # between variables and create an equation that fits our given set well
        # Using this, we can try and predict outcome given certain observations
        self.model.fit(prod_user_matrix)

        # Root product is product from which we will derive num_of_recommendations

        # Get index of root_product in the prodname_index_map
        root_prod_index = self.string_match(prodname_index_map, root_product)

        # got to finish this

    def recommend(self, product, num_of_recommendations):

        # got to finish this

    def start_user_loop(self):
        # Take input of persona/domain interest from user
        while True:
            # Initially, go through the motions because it is a cold start
            # Add each interaction between the user and a product to our ratings dataset
            # Product dataset will already be populated when we create products
            # We call recommend() on last shown product each time and it is
            # guarenteed to get better

            # got to finish this

if __name__ == '__main__':
    recommender = Recommender(.., .., ..)
    recommender.set_model_parameters(.., ..)
    recommender.start_user_loop()
