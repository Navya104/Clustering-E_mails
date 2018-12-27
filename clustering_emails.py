import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
emails=pd.read_csv('split_emails.csv')
print(emails.shape)
def parse_raw_message(raw_message):
    lines = raw_message.split('\n')
    email = {}
    message = ''
    keys_to_extract = ['from', 'to']
    for line in lines:
        if ':' not in line:
            message += line.strip()
            email['body'] = message
        else:
            pairs = line.split(':')
            key = pairs[0].lower()
            val = pairs[1].strip()
            if key in keys_to_extract:
                email[key] = val
    return email

def parse_into_emails(messages):
    emails = [parse_raw_message(message) for message in messages]
    return {
        'body': map_to_list(emails, 'body'), 
        'to': map_to_list(emails, 'to'), 
        'from_': map_to_list(emails, 'from')
    }
def map_to_list(emails, key):
    results = []
    for email in emails:

        if key not in email:
            results.append('')
        else:
            results.append(email[key])
    return results
#creating dataframe with 3 columns to,from,body
email_df = pd.DataFrame(parse_into_emails(emails.message))
#print columns of the dataframe
print(email_df.columns)
print(email_df)
#drop emails with empty body
email_df.drop(email_df.query("body == '' | to == '' | from_ == ''").index, inplace=True)
#after dropping null emails
print(email_df.shape)
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
vect = TfidfVectorizer(stop_words='english', max_df=0.50, min_df=2)
X = vect.fit_transform(email_df.body)
from sklearn.decomposition import PCA
X_dense = X.todense()
coords = PCA(n_components=2).fit_transform(X_dense)
plt.scatter(coords[:, 0], coords[:, 1], c='m')
plt.show()
def top_tfidf_feats(row, features, top_n=20):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats, columns=['features', 'score'])
    return df

def top_feats_in_doc(X, features, row_id, top_n=25):
    row = np.squeeze(X[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)
features = vect.get_feature_names()
print(top_feats_in_doc(X,features,1,10))
def top_mean_feats(X, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    if grp_ids:
        D = X[grp_ids].toarray()
    else:
        D = X.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)
#top terms out of all mails
print(top_mean_feats(X, features, top_n=10))
from sklearn.cluster import KMeans, MiniBatchKMeans
n_clusters = 3
clf = KMeans(n_clusters=n_clusters, 
            max_iter=150, 
            init='k-means++', 
            n_init=1)
labels = clf.fit_predict(X)
label_colors = ["#2AB0E9", "#2BAF74", "#D7665E", "#CCCCCC", 
                "#D2CA0D", "#522A64", "#A3DB05", "#FC6514"]
colors = [label_colors[i] for i in labels]
from sklearn.decomposition import PCA
centroids = clf.cluster_centers_
centroid_coords = PCA(n_components=2).fit_transform(centroids)
plt.scatter(coords[:, 0],coords[:, 1],color=colors)
plt.show()
def top_feats_per_cluster(X, y, features, min_tfidf=0.1, top_n=25):
    dfs = []

    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label) 
        feats_df = top_mean_feats(X, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def plot_tfidf_classfeats_h(dfs):
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("cluster = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.score, align='center', color='#7530FF')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.features)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()
plot_tfidf_classfeats_h(top_feats_per_cluster(X, labels, features, 0.1, 25))
stopwords = ENGLISH_STOP_WORDS.union(['ect', 'hou', 'com', 'recipient'])
vec = TfidfVectorizer(analyzer='word', stop_words=stopwords, max_df=0.3, min_df=2)
vec_train = vec.fit_transform(email_df.body)
plot_tfidf_classfeats_h(top_feats_per_cluster(vec_train, labels, features, 0.1, 25))
from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(vec_train[0:1], vec_train).flatten()
query = "phillip"
vec_query = vec.transform([query])
cosine_sim = linear_kernel(vec_query, vec_train).flatten()
related_email_indices = cosine_sim.argsort()[:-10:-1]
print(related_email_indices)
first_email_index = related_email_indices[3]
print(email_df.body.as_matrix()[first_email_index])
