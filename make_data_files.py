import os
import shutil
import collections
import tqdm
from tensorflow.core.example import example_pb2
import struct
import random
import shutil

finished_path = "data/finished"
unfinished_path = "data/unfinished"
chunk_path = "data/chunked"

vocab_path = "data/vocab"
VOCAB_SIZE = 200000

CHUNK_SIZE = 15000 # num examples per chunk, for the chunked data
train_bin_path = os.path.join(finished_path, "train.bin")
valid_bin_path = os.path.join(finished_path, "valid.bin")

def make_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

def shuffle_text_data(unshuffled_art, unshuffled_abs, shuffled_art, shuffled_abs):
    article_itr = open(os.path.join(unfinished_path, unshuffled_art), "r")
    abstract_itr = open(os.path.join(unfinished_path, unshuffled_abs), "r")
    list_of_pairs = []
    for article in article_itr:
        article = article.strip()
        abstract = next(abstract_itr).strip()
        list_of_pairs.append((article, abstract))
    article_itr.close()
    abstract_itr.close()
    random.shuffle(list_of_pairs)
    article_itr = open(os.path.join(unfinished_path, shuffled_art), "w")
    abstract_itr = open(os.path.join(unfinished_path, shuffled_abs), "w")
    for pair in list_of_pairs:
        article_itr.write(pair[0]+"\n")
        abstract_itr.write(pair[1]+"\n")
    article_itr.close()
    abstract_itr.close()

def write_to_bin(article_path, abstract_path, out_file, vocab_counter = None):

    with open(out_file, 'wb') as writer:

        article_itr = open(article_path, 'r')
        abstract_itr = open(abstract_path, 'r')
        for article in tqdm.tqdm(article_itr):
            article = article.strip()
            abstract = next(abstract_itr).strip()

            tf_example = example_pb2.Example()
            tf_example.features.feature['article'].bytes_list.value.extend([article.encode()])
            tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))

            if vocab_counter is not None:
                art_tokens = article.split(' ')
                abs_tokens = abstract.split(' ')
                # abs_tokens = [t for t in abs_tokens if
                #               t not in [SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens]  # strip
                tokens = [t for t in tokens if t != ""]  # remove empty
                vocab_counter.update(tokens)

    if vocab_counter is not None:
        with open(vocab_path, 'w') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')


def creating_finished_data():
    make_folder(finished_path)

    vocab_counter = collections.Counter()

    write_to_bin(os.path.join(unfinished_path, "train.art.shuf.txt"), os.path.join(unfinished_path, "train.abs.shuf.txt"), train_bin_path, vocab_counter)
    write_to_bin(os.path.join(unfinished_path, "valid.art.shuf.txt"), os.path.join(unfinished_path, "valid.abs.shuf.txt"), valid_bin_path)


def chunk_file(set_name, chunks_dir, bin_file):
    make_folder(chunks_dir)
    reader = open(bin_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunks_dir, '%s_%04d.bin' % (set_name, chunk)) # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


if __name__ == "__main__":
    shuffle_text_data("train.article.txt", "train.title.txt", "train.art.shuf.txt", "train.abs.shuf.txt")
    shuffle_text_data("valid.article.filter.txt", "valid.title.filter.txt", "valid.art.shuf.txt", "valid.abs.shuf.txt")
    print("Completed shuffling train & valid text files")
    delete_folder(finished_path)
    creating_finished_data()        #create bin files
    print("Completed creating bin file for train & valid")
    delete_folder(chunk_path)
    chunk_file("train", os.path.join(chunk_path, "train"), train_bin_path)
    chunk_file("valid", os.path.join(chunk_path, "main_valid"), valid_bin_path)
    print("Completed chunking main bin files into smaller ones")
    #Performing rouge evaluation on 1.9 lakh sentences takes lot of time. So, create mini validation set & test set by borrowing 15k samples each from these 1.9 lakh sentences
    make_folder(os.path.join(chunk_path, "valid"))
    make_folder(os.path.join(chunk_path, "test"))
    bin_chunks = os.listdir(os.path.join(chunk_path, "main_valid"))
    bin_chunks.sort()
    samples = random.sample(set(bin_chunks[:-1]), 2)      #Exclude last bin file; contains only 9k sentences
    valid_chunk, test_chunk = samples[0], samples[1]
    shutil.copyfile(os.path.join(chunk_path, "main_valid", valid_chunk), os.path.join(chunk_path, "valid", "valid_00.bin"))
    shutil.copyfile(os.path.join(chunk_path, "main_valid", test_chunk), os.path.join(chunk_path, "test", "test_00.bin"))

    # delete_folder(finished)
    # delete_folder(os.path.join(chunk_path, "main_valid"))




