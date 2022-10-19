### Scripts for dataset creation
The scripts contained in this folder cover all the steps needed to create datasets of Reddit posts in the format required by the models.
Dataset creation steps and the corresponding scripts are the following:
- ```download.py```: downloads Reddit submissions from the Pushshift dump at https://files.pushshift.io/reddit/submissions/, removes empty posts, crossposts, posts flagged as over_17 or posts with missing essential info, and saves as tsv
- ```prefilter.py```: removes users with too few posts from tsvs
- ```split_authors.py```: splits files into single-user files
- ```remove_authors.py```: reads all user files, removes duplicate posts, and deletes the user file if there are less than 5 posts left, if remaining posts belong to less than 5 distinct subreddits.
- ```batch_authors.py```: groups the remaining files into larger files with 10000 users each. Also removes html tags from main text.
- ```make_db.py```: also stores the data as a SQLite3 database.

Subfolders contain workflows for specific types of datasets, e.g. ```triplet``` (based on json-stored data):
- ```make_triplets.py```: for each user, picks anchor, positive and negative example and saves as json
- ```make_triplet_dataset.py```: creates TF dataset from json files (includes tokenization step)
- ```make_triplet_author_index.py```: makes an index of which author is in which batch

and ```triplet_baselines```, which includes workflows to create datasets for the subreddit classification task.