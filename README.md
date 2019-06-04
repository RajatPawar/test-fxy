###### On hold until we can review and discuss the next best strategy!
## Problem statement

The program that I'd like you to build is called ProductRoulette. It can be a command line tool. The user enters his email and persona (developer, salesperson, marketing, etc.), and the program immediately starts showing interesting B2B products that she may or may not like. Depending on the feedback, the program gets progressive smarter and gets the user's interest profile. Like Netflix/Tinder for Software Products.

## Related literature:
https://arxiv.org/pdf/1604.02071.pdf

## Thoughts

There are a few types of recommender systems that I’ve read about - 
### Popularity based systems
We recommend the most popular apps to a user, top X of a category, etc. But this leads to a vicious cycle where apps that have not been previously consumed will never be pushed as a recommendation and will remain largely unknown. Let’s avoid that because it’d make for a terrible recommender system.
### User-based collaborative filtering
Suggest an app to user A based on other users similar to user A liking the app. 
It eventually comes down to grouping preferences on users but faces a problem in the initial stages. How do we recommend apps when there are no user preferences to digest?
### Item-based collaborative filtering
We build app/product neighborhoods i.e. we group products according to similarity based on metadata & properties that they have and assume that if user A liked product X, then a product Y similar to product X may also be liked by user A.

### How and what?

Here, we can see that it makes sense to use Item-based collaborative filtering to deal with recommending B2B products to the user. User-based collaborative filtering seems to be difficult to scale because user behavior is always changing, meaning we always have to keep retraining our model. On the other hand, our Item-based collaborative filtering seems better because we can do it offline, and we can do it once for all the products based on their properties (which would not change) and there’d be no need to retrain frequently. There’s also a hybrid way (which I thought of in my initial document - point #4 in less formal terms!) where we use both of these to best rank products and suggest them to a user.

First of all, we need data to train our model with and begin the process. Let’s assume a dataset with the following properties - a product dataset with columns like product ID, title, tags & associated descriptions. Let’s assume another dataset - a ratings dataset with columns like user, product ID, and rating. There could be lots of problems here. Popular products could have a high amount of ratings while some products have none. Some products could be rated highly through a very low number of reviews. We have to remove these disparities initially. Maybe we could discard products with no ratings and simply consider products with more than X amount of ratings initially. 

We create a matrix out of the productId against the userId with the value as the rating. A very real-world problem is that there are generally a lot of N/A values in most data & here, our matrix that we could fill with 0. But then, we’d be wasting a LOT of space! We could turn this sparse matrix into something more efficient & compress it using the csr_matrix method from scipy. There are ways to do this, one being the Triplet array representation that seems to frequently pop up when googling how to efficiently represent a sparse matrix.

Also, since our matrix has a tremendous amount of dimensions, it might suffer from the curse of dimensionality - meaning that due to the huge amount of dimensions, a Euclidean distance approach to judging the similarity of products becomes meaningless because the data becomes sparse by virtue of being in so many dimensions. (All vectors kind of become equidistant from target vector in this case)

Turns out that there is another metric, cosine similarity, that we can use to handle these sort of cases. Depending on the inner product & hence the angle, it eliminates the above problem.

The classification that appeals to me after reading up is KNN (K-nearest neighbors). It basically chooses a fuzzy circle of K neighbors around a given target data point to classify our current target point. It is also known as lazy - it doesn’t require previous training. It makes no assumption about our dataset and builds all its inferences from what it is being trained with. K seems to be an important factor and we need to choose wisely! If we set K = 1, our error rate is going to be zero because 1 means that it itself would count as the closest neighbor. We can take a test data point and go from K = 1 to K = ∞, plot the graph of the value of K vs the error rate and see where the minimum lies. That would be our optimal value of K. The next steps would be to try and classify a test point. How? Get its distance from all other rows (data points) using the cosine metric we’ve chosen, sort and take the top X rows, check the most frequently occurring class in these rows and confidently say that our test point belongs to that class.
 

## Initial thoughts of the untrained mind

Here’s what I can think of from our currently limited information. 

### Authentication

The user needs to sign up for using this. We need to have profile verified through an email link. We need to also provide logging in through various vendors & also the ability to sign out & delete profile. 
For example, all the above can be done using Firebase as it already provides authentication flow through loads of channels. We only make the respective forms and use the methods detailed in the Firebase documentation like createUserWithEmailAndPassword(..), etc.

### Profile/interests form

This form is crucial to the program’s working. We allow the user to add/edit any details we may require, but most importantly allow the addition of interests or ‘tags’ to their profile. These tags help us match the B2B products to these users. For example, developer or finance both qualify because the former is an occupation while the latter narrows the interest domain down.
 
### The proximity of users to B2B products

Users and products in close ‘proximity’ could be shortlisted - our user could then decide to like or dislike a product. If a product is liked, we could consider it matched with the user. The ‘proximity’ could be a factor that could be calculated based on various factors - essentially being the core matching algorithm. The next point outlines a few ideas for the algorithm.
Based on a threshold that we choose after probably some piloting, we can decide to display products to each user depending on their proximity. If however, we just rank them all based on proximity, that’d mean the user can scroll to the very end of the page & they’d find the most irrelevant products to their role. Imagine a developer finding an audio-mixing product. We could avoid that by cutting off the listing at a certain proximity factor. 
 
### Getting smarter with each match

There are many ways where we can try to make our algorithm act ‘smarter’ (smarter is in quotes because it is quite debatable here). Assume the initial listing is based on simple tag matching; more the tags that match between a user and a product, the higher it is shown. Now onto how we could learn from the actions of the user.
Say our user likes a product. We could get a pair of a user-product match, and essentially add the product’s related tags to the user’s interest. And now, we just increased the granularity and detail of the user’s interest - making our displaying algorithm automatically better. That is because as each interest is added, more factors in each product are pitted against each other and are taken into consideration when listing them. Consider the below example. 
Example: User interests are (X, Y) & product tags are as follows:
ProdOne (Y, Z), ProdTwo(A, B), ProdThree(A,Y) ProdFour(Y), ProdFive(X,A)
We see that user’s interest Y matches with ProdOne, Three & Four. Also, ProdFive’s X matches with the user’s interest X. We assume the threshold as at least one tag on either side matching. So we show these products. Now the user decides to like ProdThree. Now we pull interests from the liked product and add them into the interests of the user as - (X, Y, A). Now the Prods again are as follows (we don't show ProdThree since already chosen) - 
ProdOne (Y, Z), ProdTwo(A, B), ProdFour(Y), ProdFive(X,A)
Now, our algorithm just became better - we know that the most relevant here is ProdFive because it matches X & A for the user. That’d be on top. Rest of them have one interest matching with the user. How do we rank all these ‘second-placed’ products between themselves? 
One approach could be where if the product has only one tag and it matches to a user - that’s a very good fit for the user. The rest, we could rank based on other users’ interactions with the products. More on that in the next point. 
If a certain product is bought more by other users whose interests are the same as our current user, then we could rank that higher. We could call that factor as the WhoElseBuysThis (WEBT) factor for the product. Essentially, we’ll have to maintain a transaction history of every user-product match. Now, when we are showing the product to our current user and deciding to rank it - here’s how we go. We get all the people’s interests who have bought said product. We try to count how many people’s interests match with our current user. Say we get a WEBT factor 5. Then we do this for another product and we get the result as a WEBT factor 10. Now that means that the latter product was bought by more people like our current user making it more relevant to our current user. So we could rank that higher amongst the two products.
 
 
### Products

We need to give the ability to add products to our catalog to product owners.
Things we’d require them to add are - 
Product name (mandatory)
Product cost (mandatory)
Product description (mandatory)
Product media (images/videos)
Product free-trial link and other deals (optional)


### The post-match lifecycle

After a user likes a product, what options do they have? Can they buy this product, or do we simply display information about it, or can they talk to someone from the product’s sales team.. These are few of the many things I can think of.

### Miscellaneous

A user may decide to clear their history of likes to reset the algorithm.


### DEVELOPMENT:

To understand better on how Tinder works (as per your parallel) because they’ve gone through all the things we are trying to accomplish here, I took a look at https://siftery.com/company/tinder. 
Turns out they use MongoDB for persisting data, and probably Node.js for APIs, etc. To hasten things up, I feel like going for Firebase to set up user authentication and using MongoDB when we have to mature the product - we can use Firebase for prototyping quickly.
We could later add Redis to cache stuff but let’s get the normal version working first.
