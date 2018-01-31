Adaptation of [this](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/reddit_tft) example to be a classifier to predict a positive or negative reddit score for a comment. Also some scripting to actually deploy it, get predictions, and load them back into BigQuery (google cloud ml samples seem to not cover this part).

Things i've done:

 - Changed `reddit.py` to be more generic `make_bq_sql.py`. This is just superficial. 
 - Some small changes to `requirements.txt` and `setup.py` to get working on google-cloud-ml.
 - Pulling in a smaller set of comments from Reddit (filtering by subbreddit and a LIMIT). Just so i can pull in smaller sets of data and iterate quicker as i play with it.
 - Changing score to be 0,1 where 1 is if score>0
 - Included in the pull from reddit a field called `id as example_id` to try use this as a key when getting predictions. 
 - Adding another `model_type` called `deep_classifier` which creates a `DNNClassifier` in very similar way model_type of `deep` creates a `DNNRegressor`.
 - Added some bash scripts (require manually setting veriables of where to pick stuff up from) to actually deploy the model, run a predction, and then load that prediction back into BigQuery.
