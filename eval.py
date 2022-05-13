import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from model import Model

from data_util import config, data
from data_util.batcher import Batcher
from data_util.data import Vocab
from train_util import *
from beam_search import *
from rouge import Rouge
import argparse

def get_cuda(tensor):
    if T.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

class Evaluate(object):
    def __init__(self, data_path, opt, batch_size = config.batch_size, eval_dump_save: str='eval.txt'):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(data_path, self.vocab, mode='eval',
                               batch_size=batch_size, single_pass=True)
        self.opt = opt
        time.sleep(5)
        self.eval_dump_save = eval_dump_save

    def setup_valid(self):
        self.model = Model()
        self.model = get_cuda(self.model)
        checkpoint = T.load(os.path.join(config.save_model_path, self.opt.load_model))
        self.model.load_state_dict(checkpoint["model_dict"])


    def print_original_predicted(self, decoded_sents, ref_sents, article_sents, loadfile):
        filename = "test_"+loadfile.split(".")[0]+".txt"
    
        with open(os.path.join("data",filename), "w") as f:
            for i in range(len(decoded_sents)):
                f.write("article: "+article_sents[i] + "\n")
                f.write("ref: " + ref_sents[i] + "\n")
                f.write("dec: " + decoded_sents[i] + "\n\n")

    def write_eval_dump(self, text: str, append: bool=False):
        f = open(self.eval_dump_save, 'a' if append else 'w')
        f.write(f'\n{text}' if append else text)
        f.close()

    def evaluate_batch(self, print_sents = False, first: bool=True, max_batches: int=None):

        self.setup_valid()
        batch = self.batcher.next_batch()
        start_id = self.vocab.word2id(data.START_DECODING)
        end_id = self.vocab.word2id(data.STOP_DECODING)
        unk_id = self.vocab.word2id(data.UNKNOWN_TOKEN)
        decoded_sents = []
        ref_sents = []
        article_sents = []
        rouge = Rouge()
        num_batches = self.batcher._batch_queue.qsize()
        from tqdm import tqdm
        max_batches = num_batches if max_batches is None else max_batches
        pbar = tqdm(total=min(num_batches, max_batches), unit='batches', leave=False)
        loop_count = 0
        while batch is not None and loop_count < max_batches:
            loop_count += 1
            enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, ct_e = get_enc_data(batch)
            with T.autograd.no_grad():
                enc_batch = self.model.embeds(enc_batch)
                enc_out, enc_hidden = self.model.encoder(enc_batch, enc_lens)

            #-----------------------Summarization----------------------------------------------------
            with T.autograd.no_grad():
                pred_ids = beam_search(enc_hidden, enc_out, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, self.model, start_id, end_id, unk_id)
            for i in range(len(pred_ids)):
                decoded_words = data.outputids2words(pred_ids[i], self.vocab, batch.art_oovs[i])
                if len(decoded_words) < 2:
                    decoded_words = "xxx"
                else:
                    decoded_words = " ".join(decoded_words)
                decoded_sents.append(decoded_words)
                abstract = batch.original_abstracts[i]
                article = batch.original_articles[i]
                ref_sents.append(abstract)
                article_sents.append(article)
            batch = self.batcher.next_batch()
            pbar.update()
        pbar.close()

        load_file = self.opt.load_model
        if print_sents:
            self.print_original_predicted(decoded_sents, ref_sents, article_sents, load_file)
        scores = rouge.get_scores(decoded_sents, ref_sents, avg = True)
        if self.opt.task == "test":
            text = f'{load_file} scores: {scores}'
            print(text)
            self.write_eval_dump(text=text, append=not first)
        else:
            rouge_l = scores["rouge-l"]["f"]
            rouge_l_text = "%.4f" % rouge_l
            text = f'{load_file} rouge_l: {rouge_l_text}'
            print(text)
            self.write_eval_dump(text=text, append=not first)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="validate", choices=["validate","test"])
    parser.add_argument("--start_from", type=str, default="0020000.tar")
    parser.add_argument("--load_model", type=str, default=None)
    opt = parser.parse_args()
    max_batches = 10 # Evaluation takes too long unless I reduce the number of batches like this.

    if opt.task == "validate":
        saved_models = os.listdir(config.save_model_path)
        saved_models.sort()
        file_idx = saved_models.index(opt.start_from)
        saved_models = saved_models[file_idx:]
        first = True
        for f in saved_models:
            opt.load_model = f
            eval_processor = Evaluate(config.valid_data_path, opt)
            eval_processor.evaluate_batch(first=first, max_batches=max_batches)
            first = False
    else:   #test
        eval_processor = Evaluate(config.test_data_path, opt, max_batches=max_batches)
        eval_processor.evaluate_batch(first=True)
