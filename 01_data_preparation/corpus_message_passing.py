import os

import pandas as pd
from collections import Counter
import networkx as nx
import random
import pickle
import numpy as np

if __name__ == '__main__':
    n_mp_steps = 1  # number of message passing steps
    n_nodes = 20  # number of communities
    seed = 2023  # random seed

    random.seed(seed)

    print('Reading tweets....')
    df = pd.read_csv('../data/df_election_tweets_processed.csv', lineterminator='\n')
    print(df.head())

    # Generating the edge list of the rewteet network
    print('Preparing the retweet network....')
    tweet2user = dict(zip(df['tweetid'].tolist(), df['userid'].tolist()))
    user2comm = dict(zip(df['userid'].tolist(), df['community'].tolist()))
    retweet_links = []
    for tweetid, tweet_type, rt_tweetid in zip(df['tweetid'].tolist(),
                                               df['tweet_type'].tolist(),
                                               df['rt_tweetid'].tolist()):
        # only consider retweets as the community interactions
        if tweet_type == 'retweeted_tweet_without_comment':
            userid1 = tweet2user[tweetid]
            if rt_tweetid in tweet2user:
                userid2 = tweet2user[rt_tweetid]
                comm1 = user2comm[userid1]
                comm2 = user2comm[userid2]
                retweet_links.append((comm1, comm2))
    print(len(retweet_links))

    # weight the edge list
    retweet_links_weighted = []
    count_edges = Counter(retweet_links)
    edges_retweet_user_weighted = []
    for e, w in count_edges.items():
        retweet_links_weighted.append(f"{e[0]} {e[1]} {w}")
    print(retweet_links_weighted[:n_nodes])

    # construct the networkx graph (directed)
    G_retweet = nx.parse_edgelist(retweet_links_weighted, nodetype=int, data=(("weight", float),),
                                  create_using=nx.DiGraph)
    G_retweet = G_retweet.subgraph(range(n_nodes))  # only consider the top 20 communities

    # preparing for message passing
    print('Preparing for message passing....')
    comm2tweets = {0: {}}   #  tweets for each community after each message passing step. 0 means the tweets before message passing
    comm2tweets_lib = {0: {}}   # liberal tweets for each community after each message passing step.
    comm2tweets_con = {0: {}}   # conservative tweets for each community after each message passing step.
    for node in range(n_nodes):
        df_comm = df.query(f"community=={node}")
        df_comm_lib = df_comm.query(f"political==0")
        df_comm_con = df_comm.query(f"political==1")
        comm_tweets = df_comm['processed_text'].tolist()
        comm_tweets_lib = df_comm_lib['processed_text'].tolist()
        comm_tweets_con = df_comm_con['processed_text'].tolist()
        comm2tweets[0][node] = comm_tweets
        comm2tweets_lib[0][node] = comm_tweets_lib
        comm2tweets_con[0][node] = comm_tweets_con
        print(f"{node}, {len(comm2tweets[0][node])} tweets, "
              f"{len(comm2tweets_lib[0][node])} tweets, {len(comm2tweets_con[0][node])} tweets")

    # message passing
    # dict of community and its lib/con fractions. {0: [0.9, 0.1]}
    comm2political = pickle.load(open('../data/comm2political.pkl', 'rb'))

    print('Running message passing....')
    for i in range(n_mp_steps):
        print(f'*************Step {i+1}*****************')
        comm2tweets[i+1] = {}
        comm2tweets_lib[i+1] = {}
        comm2tweets_con[i+1] = {}
        for node in range(n_nodes):
            cur_tweets = comm2tweets[i][node]
            N = len(cur_tweets)
            lib_frac, con_frac = comm2political[node]
            neighbors = list(G_retweet.successors(node))
            out_weights = [G_retweet[node][neighbor]['weight'] for neighbor in neighbors]
            total_out_weight = sum(out_weights)

            new_tweets = []
            for neighbor, weight in zip(neighbors, out_weights):
                n = int(N * weight / total_out_weight)
                neighbor_tweets_lib = comm2tweets_lib[i][neighbor]
                neighbor_tweets_con = comm2tweets_lib[i][neighbor]
                n_lib = int(N * weight / total_out_weight * lib_frac)
                n_con = int(N * weight / total_out_weight * con_frac)

                sampled_tweets_lib = random.choices(cur_tweets, k=n_lib)
                sampled_tweets_con = random.choices(cur_tweets, k=n_con)
                new_tweets.extend(sampled_tweets_lib)
                new_tweets.extend(sampled_tweets_con)

            comm2tweets[i+1][node] = new_tweets

        for node in range(n_nodes):
            print(f"{node}, {len(comm2tweets[i+1][node])}")

    # aggregate tweets after each step of message passing to create the holistic corpus
    comm2tweets_all = {}
    for node in range(n_nodes):
        comm2tweets_all[node] = []

    for i in range(n_mp_steps + 1):
        for node in range(n_nodes):
            comm2tweets_all[node].extend(comm2tweets[i][node])

    for node in G_retweet.nodes:
        print(f"{node}, {len(comm2tweets_all[node])}")

    os.makedirs('../data/community_texts', exist_ok=True)

    # save the corpus for the baseline model (without message passing)
    for node in range(n_nodes):
        comm_tweets = comm2tweets_all[node]
        file_path = f'../data/community_texts/tweets_community_mp_{node}.txt'
        with open(file_path, "w") as f:
            for item in comm_tweets:
                f.write(f"{item}\n")

    # save the corpus for our model (with message passing)
    for node in range(n_nodes):
        comm_tweets = comm2tweets[0][node]
        file_path = f'../data/community_texts/tweets_community_{node}.txt'
        with open(file_path, "w") as f:
            for item in comm_tweets:
                f.write(f"{item}\n")