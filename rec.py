# scikit for machine learning
from sklearn.neighbors import NearestNeighbors
# Compression sparse Matrix
from scipy.sparse import csr_matrix
# Pandas for data manipulation
import pandas as pnd
# String matching for user searches
from fuzzywuzzy import fuzz
from math import isnan
import os

class Recommender:

    # Datasets - static vars
    # ---------------------
    # Google - Folder struct
    folder_struct_google = os.path.join("datasets", "Google")
    # Apple - Folder struct
    folder_struct_apple = os.path.join("datasets", "Apple")
    # Sample - Folder struct
    folder_struct_sample = os.path.join("datasets", "Sample")
    # All our datasets
    datasets = [
        {
            'reviews': os.path.join(folder_struct_google, "reviews.csv"),
            'products': os.path.join(folder_struct_google,  "products.csv")
        },
        {
            'reviews': os.path.join(folder_struct_apple, "reviews.csv"),
            'products': os.path.join(folder_struct_apple, "products.csv"),
            'product_id': 'id',
            'product_name': 'track_name',
            'user_id': 'user',
            'ratings_id': 'rating'
        },
        {
            'reviews': os.path.join(folder_struct_sample, "reviews.csv"),
            'products': os.path.join(folder_struct_sample, "products.csv"),
            'product_id': 'productId',
            'product_name': 'name',
            'user_id': 'userId',
            'ratings_id': 'rating'
        }
    ]

    # Initialize
    def __init__(self, products_set_path, ratings_set_path, min_product_interactions, min_user_ratings, string_match_threshold, dataset):
        """
            # Returns - none
        """
        # Product information dataset with relevant metadata
        self.products_set_path = products_set_path
        # Product user interaction dataset (Ratings, etc)
        self.ratings_set_path = ratings_set_path
        # Min number of user interactions product must have to count in model
        self.min_product_interactions = min_product_interactions
        # Min number of ratings user must give to count
        self.min_user_ratings = min_user_ratings
        # We use the K nearest neighbors model to classify
        self.model = NearestNeighbors()
        # Threshold to match given product name to our list of products (60+)
        self.string_match_threshold = string_match_threshold
        # Which dataset?
        self.dataset = dataset


    def set_model_parameters(self, k_value, algorithm, metric_to_use, parallel_jobs):
        """
            # Returns - none
        """
        self.neighbors_to_consider = k_value
        # Make sure if we allow running parallel jobs, then we give it a scratchpad
        if parallel_jobs and (parallel_jobs > 1 or parallel_jobs == -1):
            os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp';
        self.model.set_params(**{
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
        """
            # Returns           - prod_user_matrix & prodname_index_map

            prod_user_matrix    - A filtered matrix of productId vs userId with ratings as values
            prodname_index_map  - Maintains map of product_name vs product index in main matrix for searches, etc
        """
        # Load product dataset into a dataframe
        # Use only product ID and the product name as features after specifying their datatype
        products_dataframe = pnd.read_csv(self.products_set_path, usecols=[Recommender.datasets[self.dataset]['product_id'], Recommender.datasets[self.dataset]['product_name']], dtype={Recommender.datasets[self.dataset]['product_id']: 'int32', Recommender.datasets[self.dataset]['product_name']: 'str'})
        products_dataframe.dropna(subset=[Recommender.datasets[self.dataset]['product_name']], inplace=True)

        # Load ratings dataset into a dataframe
        # Use only product ID, user ID and rating given as features for our model
        ratings_dataframe = pnd.read_csv(self.ratings_set_path, usecols=[Recommender.datasets[self.dataset]['product_id'], Recommender.datasets[self.dataset]['user_id'], Recommender.datasets[self.dataset]['ratings_id']], dtype={Recommender.datasets[self.dataset]['product_id']: 'int32', Recommender.datasets[self.dataset]['user_id']: 'int32', Recommender.datasets[self.dataset]['ratings_id']: 'float32'})

        # Steps to clean data -
        # 1. Only take products with count(ratings) > min_product_interactions
        # 2. Eliminate inactive users from dataset

        # 1. The filtered_ratings below DF helps us retain products with more than > min_product_interactions number of ratings
        active_products_count_dataframe = pnd.DataFrame(ratings_dataframe.groupby(Recommender.datasets[self.dataset]['product_id']).size(), columns=['number_of_ratings'])
        active_products_dataframe_idx = list((active_products_count_dataframe.query('number_of_ratings > @self.min_product_interactions').index))
        # Get a list of active products and only keep their ratings in the ratings dataset - eliminate rest
        filtered_ratings = ratings_dataframe[Recommender.datasets[self.dataset]['product_id']].isin(active_products_dataframe_idx).values

        # 2. Eliminate inactive users - they are very likely to rate sporadically causing recommendation skew
        active_users_count_dataframe = pnd.DataFrame(ratings_dataframe.groupby(Recommender.datasets[self.dataset]['user_id']).size(), columns=['number_of_given_ratings'])
        active_users_count_idx = list(active_users_count_dataframe.query('number_of_given_ratings > @self.min_user_ratings').index)
        filtered_users = ratings_dataframe[Recommender.datasets[self.dataset]['user_id']].isin(active_users_count_idx).values

        filtered_matrix = ratings_dataframe[filtered_ratings & filtered_users]

        # Get it into a matrix from a normal array (We need them in a vector format for KNN to work)
        prod_user_matrix = filtered_matrix.pivot(index = Recommender.datasets[self.dataset]['product_id'], columns = Recommender.datasets[self.dataset]['user_id'], values = Recommender.datasets[self.dataset]['ratings_id']).fillna(0)

        # generate hashmap from products to matrix index
        iter_list = list(products_dataframe.set_index(Recommender.datasets[self.dataset]['product_id']).loc[prod_user_matrix.index][Recommender.datasets[self.dataset]['product_name']])

        prodname_index_map = {
            product: i for i, product in
            enumerate(iter_list)
        }
        prodname_index_map = {k: prodname_index_map[k] for k in prodname_index_map if type(k) is str}
        print(prodname_index_map)

        csr_prod_user_matrix = csr_matrix(prod_user_matrix)
        print(prod_user_matrix)

        return csr_prod_user_matrix, prodname_index_map
        # Keep reference of productId index in filtered matrix to its row in product_dataset
        # Make the map here and return the matrix & map
        # got to finish this
        # -----------------------------------

    def string_match(self, prodname_index_map, product_name):
        """
            # Returns - Top X match(es) to given product_name in our map of product names
        """
        # Uses the fuzzywuzzy lib to return closest match to product name in the map
        all_matches = []

        # For all products in the map, get all similar strings & append into our list
        for name, index in prodname_index_map.items():
            match_percent = fuzz.ratio(name.lower(), product_name)
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
        """
            # Returns - (raw list of num_of_recommendations products similar to root_product)
        """
        # Study the data, and use the values to understand mathematical relationships
        # between variables and create an equation that fits our given set well
        # Using this, we can try and predict outcome given certain observations
        self.model.fit(prod_user_matrix)

        # Root product is product from which we will derive num_of_recommendations

        # Get index of root_product in the prodname_index_map
        root_prod_index = self.string_match(prodname_index_map, root_product)

        # The actual magic happens here
        similarities, idxs = self.model.kneighbors(prod_user_matrix[root_prod_index], n_neighbors = (num_of_recommendations if num_of_recommendations else self.neighbors_to_consider)) # Use self.model.get_params to get k_value
        print(similarities, idxs)

        raw_list = sorted(
            list(
                zip(
                    idxs.squeeze().tolist(),
                    similarities.squeeze().tolist()
                )
            ),
            key=lambda x: x[1]
        )[:0:-1]

        return raw_list
        # We have a list of products similar to our given product along with the indices
        # Use those indices to get details from prodname_index_map and display
        # got to finish this
        # -----------------------------------

    def recommend(self, product, num_of_recommendations):
        """
            # Returns - User facing list of num_of_recommendations products to show
        """
        # Get prod_user_matrix & map from prepare_clean_data()
        # infer(prod_user_matrix, map, product, num_of_recommendations)
        # returns num_of_recommendations most relevant items based on product
        return
        # -----------------------------------

    def start_user_loop(self):
        """
            # Returns - none
        """
        prod_user_matrix, prodname_index_map = self.prepare_clean_data()
        list_raw = self.infer(prod_user_matrix, prodname_index_map, "Twisty Arrow - Shoot the Circle Wheel")

        reverse_hashmap = {v:k for k,v in prodname_index_map.items()}
        print('Recommendations for input - {}:'.format("Twisty Arrow - Shoot the Circle Wheel"))

        for i, (idx, dist) in enumerate(list_raw):
            if idx in reverse_hashmap.keys():
                print('RECOMMENDATION: {1}, with distance of {2}'.format(i+1, reverse_hashmap[idx], dist))
        # Take input of persona/domaininterest from user
        # Input needs to be taken just like Netflix interests when you sign up
        # to avoid the cold start problem. Use item similarities to show initial
        # products or popular ones or any content-based method.
        while True:
            # Add each interaction between the user and a product to our ratings dataset
            # Product info in product_dataset will already be populated when we create products
            # Wait for user action (rating of a product) and then call recommend() again
            return
        # -----------------------------------

    def search(self, root_product):
        """
            # Returns - list of num_of_recommendations products relevant to root_product
        """
        # User search
        # Use string_match to return closest product to searched one
        # Use recommend() on match to get top results for searched product
        # -----------------------------------

if __name__ == "__main__":
    # Which dataset to use?
    dataset = int(input("Which dataset do we want to try?\n1. Google\n2. Apple\n3. Sample\nEnter choice - "))
    if 0 < dataset < 4:
        dataset = dataset - 1 # Adjust for array indices
        # Initialize model, pass parameters
        recommender = Recommender(Recommender.datasets[dataset]['products'], Recommender.datasets[dataset]['reviews'], 6, 0, 70, dataset)
        recommender.set_model_parameters(30, 'brute', 'cosine', -1)

        # Start user engagement here onwards
        recommender.start_user_loop()
    else:
        print("Invalid choice")
        exit()
