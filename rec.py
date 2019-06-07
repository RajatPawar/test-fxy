# scikit for machine learning
from sklearn.neighbors import NearestNeighbor
# Pandas for data manipulation
import pandas as pnd
# String matching for user searches
import fuzzywuzzy as fuzzy

class Recommender:

    def __init__(self, products_set_path, ratings_set_path, min_product_interactions, min_user_ratings, string_match_threshold):
        # Product information dataset with relevant metadata
        self.products_set_path = products_set_path
        # Product user interaction dataset (Ratings, etc)
        self.ratings_set_path = ratings_set_path
        # Min number of user interactions product must have to count in model
        self.min_product_interactions = min_product_interactions
        # Min number of ratings user must give to count
        self.min_user_ratings = min_user_ratings
        # We use the K nearest neighbors model to classify
        self.model = NearestNeighbour()
        # Threshold to match given product name to our list of products (60+)
        self.string_match_threshold = string_match_threshold

    def set_model_parameters(self, k_value, algorithm, metric_to_use, parallel_jobs):
        # Make sure if we allow running parallel jobs, then we give it a scratchpad
        if parallel_jobs and (parallel_jobs > 1 or parallel_jobs == -1):
            os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp';
        self.model.set_params({
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
        products_dataframe = pnd.read_csv(self.products_set_path, usecols=['productId', 'name'], dtype={'productId': 'int32', 'name': 'str'})
        # Load ratings dataset into a dataframe
        # Use only product ID, user ID and rating given as features for our model
        ratings_dataframe = pnd.read_csv(self.ratings_set_path, usecols=['productId', 'userId', 'rating'], dtype={'productId': 'int32', 'userId': 'int32', 'rating': 'float32'})

        # Steps to clean data -
        # 1. Only take products with count(ratings) > min_product_interactions
        # 2. Eliminate inactive users from dataset

        # 1. The filtered_ratings below DF helps us retain products with more than > min_product_interactions number of ratings
        active_products_count_dataframe = pnd.DataFrame(ratings_dataframe.groupby('productId'), columns=['number_of_ratings'])
        active_products_dataframe_idx = list((active_products_count_dataframe.query('number_of_ratings > @self.min_product_interactions').index))
        # Get a list of active products and only keep their ratings in the ratings dataset - eliminate rest
        filtered_ratings = ratings_dataframe[ratings_dataframe.productId.isin(active_products_dataframe_idx)]

        # 2. Eliminate inactive users - they are very likely to rate sporadically causing recommendation skew
        active_users_count_dataframe = pnd.DataFrame(filtered_ratings.groupby('userId'), columns=['number_of_given_ratings'])
        active_users_count_idx = list(active_users_count_dataframe.query('number_of_given_ratings > @self.min_user_ratings').index)
        filtered_ratings = filtered_ratings[filtered_ratings.userId.isin(active_users_count_idx)]

        # Get it into a matrix from a normal array (We need them in a vector format for KNN to work)
        prod_user_matrix = filtered_ratings.pivot(index = 'productId', columns = 'userId', value = 'rating').fillna(0)

        # Keep reference of productId index in filtered matrix to its row in product_dataset
        # got to finish this
        # -----------------------------------

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
            # Print product titles of all similar products
            print('{0}'.format([row[0] for row in all_matches]))
            # Return the MOST similar product's index
            return all_matches[0][1]

    def infer(self, prod_user_matrix, prodname_index_map, root_product, num_of_recommendations = None):
        # Study the data, and use the values to understand mathematical relationships
        # between variables and create an equation that fits our given set well
        # Using this, we can try and predict outcome given certain observations
        self.model.fit(prod_user_matrix)

        # Root product is product from which we will derive num_of_recommendations

        # Get index of root_product in the prodname_index_map
        root_prod_index = self.string_match(prodname_index_map, root_product)

        # The actual magic happens here
        similarities, idxs = self.model.kneighbors(prod_user_matrix, n_neighbors = ((num_of_recommendations == None) ? self.model.get_params('k_value') : num_of_recommendations))

        # We have a list of products similar to our given product along with the indices
        # Use those indices to get details from prodname_index_map and display
        # got to finish this
        # -----------------------------------

    def recommend(self, product, num_of_recommendations):
        # Get prod_user_matrix & map from prepare_clean_data()
        # infer(prod_user_matrix, map, product, num_of_recommendations)
        # returns num_of_recommendations most relevant items based on product
        return
        # -----------------------------------

    def start_user_loop(self):
        # Take input of persona/domaininterest from user
        # Input needs to be taken just like Netflix interests when you sign up
        # to avoid the cold start problem. Use item similarities to show initial
        # products or popular ones or any content-based method.
        while True:
            # Add each interaction between the user and a product to our ratings dataset
            # Product info in product_dataset will already be populated when we create products
            # We call recommend() on last shown product each time and it is
            # guarenteed to get better
            return
        # -----------------------------------

    def search(self, product):
        # User search
        # Use string_match to return closest product to searched one
        # Use recommend() on match to get top results for searched product
        # -----------------------------------

if __name__ == '__main__':
    # Initialize model, pass parameters
    recommender = Recommender(.., .., ..)
    recommender.set_model_parameters(.., ..)
    # Start user engagement here onwards
    recommender.start_user_loop()
