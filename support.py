import random
import time

from torch.utils.data import Dataset
import sys
import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

def serialize_user(user_set):
    user_set = set(user_set)
    user_idx = 0
    user_serialize_dict = {}
    for user in user_set:
        user_serialize_dict[user] = user_idx
        user_idx += 1
    return user_serialize_dict

def serialize_item(item_set):
    item_set = set(item_set)
    item_idx = 0
    item_serialize_dict = {}
    for item in item_set:
        item_serialize_dict[item] = item_idx
        item_idx += 1
    return item_serialize_dict

def sample_negative_user(user_set, interaction_user_set):
    users = set(interaction_user_set)
    candidate_users = set(user_set) - set(users)
    return random.sample(list(candidate_users), 1)[0]

def build_user_item_interaction_dict(train_csv,
                                     user_item_interaction_dict_save='pkl/user_item_interaction_dict.pkl'):
    if os.path.exists(user_item_interaction_dict_save) is True:
        print('Load user_item_interaction_dict from cache')
        pkl_file = open(user_item_interaction_dict_save, 'rb')
        data = pickle.load(pkl_file)
        return data['user_item_interaction_dict']
    if os.path.exists("pkl") is False:
        os.makedirs("pkl")
    df = train_csv
    user_item_interaction_dict = {}
    for _, row in tqdm(df.iterrows()):
        movie = row['song_id']
        user = row['user_id']
        res = user_item_interaction_dict.get(user)
        if res is None:
            user_item_interaction_dict[user] = [movie]
        else:
            res.append(movie)
            user_item_interaction_dict[user] = res
    with open(user_item_interaction_dict_save, 'wb') as file:
        pickle.dump({'user_item_interaction_dict': user_item_interaction_dict}, file)
    return user_item_interaction_dict


# Create a new item-user interaction dictionary.
def build_item_user_interaction_dict(train_csv,
                                     item_user_interaction_dict_save='pkl/item_user_interaction_dict.pkl'):
    if os.path.exists(item_user_interaction_dict_save) is True:
        print('Load from cache', item_user_interaction_dict_save)
        pkl_file = open(item_user_interaction_dict_save, 'rb')
        data = pickle.load(pkl_file)
        return data['item_user_interaction_dict']
    if os.path.exists("pkl") is False:
        os.makedirs("pkl")
    df = train_csv
    item_user_interaction_dict = {}
    for _, row in tqdm(df.iterrows()):
        movie = row['song_id']
        user = row['user_id']
        res = item_user_interaction_dict.get(movie)
        if res is None:
            item_user_interaction_dict[movie] = [user]
        else:
            res.append(user)
            item_user_interaction_dict[movie] = res
    with open(item_user_interaction_dict_save, 'wb') as file:
        pickle.dump({'item_user_interaction_dict': item_user_interaction_dict}, file)
    return item_user_interaction_dict


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RatingDataset(torch.utils.data.Dataset):
    def __init__(self, train_csv, genres, category_num, user_num,  positive_number, negative_number):
        self.train_csv = train_csv
        # Read other content
        self.genres_dict = genres
        album_list = [v[0] for k, v in genres.items()]
        artist_list = [v[1] for k, v in genres.items()]

        self.album_serialize_dict = serialize_item(album_list)
        self.artist_serialize_dict = serialize_item(artist_list)

        # print(self.item_pn_df)
        self.user = self.train_csv["user_id"]
        # self.artist = self.train_csv["artist_id"]
        # self.album = self.train_csv["album_id"]
        self.item = self.train_csv["song_id"]
        self.item_set = set(self.item)
        self.rating = self.train_csv["rating"]
        self.neg_user = self.train_csv['neg_user_id']
        self.item_serialize_dict = serialize_item(self.item)
        # When returning the count, return the number of users in the full set and the number of items in the training set
        self.user_number = user_num
        self.item_number = len(self.item_set)
        self.positive_number = positive_number
        self.negative_number = negative_number
        self.category_num = category_num
        self.user_item_interaction_dict = build_user_item_interaction_dict(train_csv)
        self.item_user_interaction_dict = build_item_user_interaction_dict(train_csv)
        print("Number of Users:", self.user_number, "Number of users in file", len(set(self.user)))

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self, index):
        row = self.train_csv.iloc[index]
        user = row['user_id']
        item = row['song_id']
        neg_user = row['neg_user_id']
        # Process item genres
        genres = np.full((self.category_num, 1), -1)
        genres_index = self.genres_dict.get(item, [0,0,0,0,0])
        if len(genres_index) > 0:
            genres[genres_index[2:]] = 1

        albums = self.album_serialize_dict[genres_index[0]]
        artists = self.artist_serialize_dict[genres_index[1]]
        positive_items_ = self.user_item_interaction_dict.get(user, [])
        positive_items = list(np.random.choice(positive_items_, self.positive_number, replace=True))
        positive_items_list = [self.item_serialize_dict[item] for item in positive_items]
        # runtime sampling negative
        neg_item_set = list(self.item_set - set(positive_items_))
        # merge multi negative sample result
        negative_items_ = list(
            np.random.choice(neg_item_set, self.negative_number * (self.positive_number + 1), replace=True))
        negative_items_ = [self.item_serialize_dict[it] for it in negative_items_]
        negative_item_list = [negative_items_[self.negative_number * i:self.negative_number * (i + 1)] for i in
                              range(self.positive_number)]
        # self neg list 完成 序列化, self的抽样放在和collaborative items中一起抽样负例子，最后分割出来就行了
        self_neg_list = negative_items_[self.positive_number * self.negative_number:]
        return (torch.tensor(user), torch.tensor(self.item_serialize_dict[item]), torch.tensor(genres), torch.tensor(artists),
                torch.tensor(albums), torch.tensor(neg_user), torch.tensor(positive_items_list),
                torch.tensor(negative_item_list), torch.tensor(self_neg_list))

# Test data packaging
if __name__ == '__main__':
    print("support.py")

    genre_df = pd.read_csv("data/genre_hierarchy.csv")
    song_df = pd.read_csv("data/final_song_genre_hierarchy.csv")
    genre_num = len(genre_df["genre_id"].unique())
    song_feature_dict = {}
    for idx, row in song_df.iterrows():
        tmp_list = [row['artist_id'], row['album_id'], row['genre_id'], row['parent_genre_id'], row['gp_genre_id']]
        song_feature_dict[row['song_id']] = tmp_list
    train_csv = pd.read_csv("data/train_rating_with_neg_s2.csv") #, sep="\t", names=["user_id", "song_id", "rating"])
    print("ratings.length:", train_csv.__len__())
    users_length = len(train_csv["user_id"].unique())
    dataset = RatingDataset(train_csv, song_feature_dict, genre_num, users_length, 10, 20)
    user, item, genres, neg_user, positive_items_list, negative_item_list, self_neg_list = dataset.run_all()
    print(dataset.__len__())
    # dataIter = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    # for i_index in dataIter:
    #     start = time.time()
    #     u, i, g, p_list, n_list, self_n_list = i_index
    #     print("time spend:", time.time()-start)
        # i_index += 1
    # print(u, i, g, i_f, n_user)
    # print("positive_list, negative_list, self_negative_list")
    # print("genres.shape:", g.shape, "img_f.shape:", i_f.shape)
