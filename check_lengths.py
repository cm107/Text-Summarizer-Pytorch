import struct
from collections import Counter, OrderedDict
from tensorflow.core.example import example_pb2
import seaborn as sns
import matplotlib.pyplot as plt

def example_generator(path):
    reader = open(path, 'rb')
    while True:
        len_bytes = reader.read(8)
        if not len_bytes: break # finished reading this file
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        yield example_pb2.Example.FromString(example_str)

train_bin_path = '/home/clayton/workspace/study/Text-Summarizer-Pytorch/data/cnn-dailymail/finished/test.bin'
count = 0
article_len_list = []
abstract_len_list = []
for data in example_generator(train_bin_path):
    count += 1
    article = data.features.feature['article'].bytes_list.value[0].decode()
    abstract = data.features.feature['abstract'].bytes_list.value[0].decode()
    article_len = len(article.split(' '))
    abstract_len = len(abstract.split(' '))
    # print(f"Article {count}: {article}")
    # print(f'article_len: {article_len}')
    # print(f"Abstract {count}: {abstract}")
    # print(f'abstract_len: {abstract_len}')
    article_len_list.append(article_len)
    abstract_len_list.append(abstract_len)

    # if count >= 1000: break

article_len_count_dict = OrderedDict(sorted(Counter(article_len_list).items(), reverse=False))
abstract_len_count_dict = OrderedDict(sorted(Counter(abstract_len_list).items(), reverse=False))

def get_cumulative_proportions(len_count_dict: OrderedDict) -> OrderedDict:
    result = OrderedDict()
    cum_count = 0
    for len_val, count in len_count_dict.items():
        cum_count += count
        result[len_val] = cum_count
    for len_val, _ in result.items():
        result[len_val] = round(result[len_val] / cum_count, 3)
    return result

def plot_len_cum_prop(len_cum_prop_dict: OrderedDict, save_path: str):
    data = {
        'Number of Tokens': list(len_cum_prop_dict.keys()),
        'Cumulative Corpus Proportion': list(len_cum_prop_dict.values())
    }
    ax = sns.lineplot(x='Number of Tokens', y='Cumulative Corpus Proportion', data=data)
    plt.savefig(save_path)
    plt.clf()
    plt.close("all")

article_len_cum_prop_dict = get_cumulative_proportions(article_len_count_dict)
abstract_len_cum_prop_dict = get_cumulative_proportions(abstract_len_count_dict)

print(f"Data count: {count}")
print(f'article_len_cum_prop_dict: {article_len_cum_prop_dict}')
print(f'abstract_len_cum_prop_dict: {abstract_len_cum_prop_dict}')
plot_len_cum_prop(article_len_cum_prop_dict, 'article_len_prop.png') # 2500 -> 99.9%
plot_len_cum_prop(abstract_len_cum_prop_dict, 'abstract_len_prop.png') # 200 -> 99.9%