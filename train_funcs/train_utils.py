import torch
import torch.nn as nn
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from train_funcs import register
from torchvision import transforms
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
# import seaborn as sns
    
class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = '-'+alphabet  # for `-1` index

        self.dict = {}
        for i, char in enumerate(self.alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """

        length = []
        result = []
        decode_flag = True if type(text[0])==bytes else False

        for item in text:

            if decode_flag:
                item = item.decode('utf-8','strict')
            length.append(len(item))
            if len(item)<1:
                continue
            for char in item:
                index = self.dict[char]
                result.append(index)
        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def encode_char(self, char):

        return self.dict[char]
    
    def encode_list(self, text, K=7):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.
            K : the max length of texts

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        # print(text)
        length = []
        all_result = []
        decode_flag = True if type(text[0])==bytes else False

        for item in text:
            result = []
            if decode_flag:
                item = item.decode('utf-8','strict')
            # print(item)
            length.append(len(item))
            for i in range(K):
                # print(item)
                if i<len(item): 
                    char = item[i]
                    # print(char)
                    index = self.dict[char]
                    result.append(index)
                else:
                    result.append(0)
            all_result.append(result)
        return (torch.LongTensor(all_result))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i]])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts
    
    def decode_list(self, t):
        texts = []
        for i in range(t.shape[0]):
            t_item = t[i,:]
            char_list = []
            for i in range(t_item.shape[0]):
                if t_item[i] == 0:
                    pass
                    # char_list.append('-')
                else:
                    char_list.append(self.alphabet[t_item[i]])
                # print(char_list, self.alphabet[44])
            # print('char_list:  ' ,''.join(char_list))
            texts.append(''.join(char_list))
        # print('texts:  ', texts)
        return texts

    def decode_sa(self, text_index):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(text_index):
            text = ''.join([self.alphabet[i] for i in text_index[index, :]])
            texts.append(text.strip('-'))
        return texts

@register('GP_LPR_TRAIN')
def train_ocr(train_loader, model, opt, loss_fn, confusing_pairs, *args):
    config = args[0]
    converter = strLabelConverter(config['alphabet'])
    for p in model.parameters():
        p.requires_grad = True
    model.train()
    pbar = tqdm(train_loader, leave=False, desc='train')
    train_loss = []
    
    for i_batch, batch in enumerate(pbar):
        text = converter.encode_list(batch['text'], K=7).cuda()
        _, preds,_ = model(batch['img'].cuda())
        loss = 0        
        preds = torch.chunk(preds, preds.size(0), 0)
        for (i, item) in enumerate(preds):
            item = item.squeeze()
            gt = text[i,:]
            loss_item = loss_fn(item, gt)
            loss+= loss_item
        loss = loss/(i+1)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        train_loss.append(loss.detach().item())
        pbar.set_postfix({'loss': sum(train_loss)/len(train_loss)})
        
    return sum(train_loss)/len(train_loss)

@register('GP_LPR_VAL')
def validation_ocr(val_loader, model, loss_fn, confusing_pairs, *args):
    config = args[0]
    converter = strLabelConverter(config['alphabet'])
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    i = 0
    n_correct = 0
    pbar = tqdm(val_loader, leave=False, desc='val')
    val_loss = []
    total = 0
    for i_batch, batch in enumerate(pbar):
        text = converter.encode_list(batch['text'], K=7).cuda()
        _, preds,_ = model(batch['img'].cuda())
        preds_all = preds        
        
        loss = 0
        preds = torch.chunk(preds, preds.size(0), 0)    
        
        for (i, item) in enumerate(preds):
            item = item.squeeze()
            gt = text[i,:]
            loss += loss_fn(item, gt)
        loss = loss/(i+1)
        
        _, preds_all = preds_all.max(2)
        sim_preds = converter.decode_list(preds_all.data)
        text_label = batch['text']
        val_loss.append(loss.detach().item())
        
        for pred, target in zip(sim_preds, text_label):
            pred = pred.replace('-', '')
            if pred == target:
                n_correct += 1
            total += 1
        
        pbar.set_postfix({'loss': sum(val_loss)/len(val_loss)})  
    print()
    for raw_pred, pred, gt in zip(preds_all, sim_preds, text_label):
        raw_pred = raw_pred.data
        pred = pred.replace('-', '')
        print('raw_pred: %-20s, pred: %-8s, gt: %-8s, match: %s' % (raw_pred, pred, gt, pred==gt))    
    accuracy = (n_correct / float(total))
    
    print(f'accuracy: {accuracy*100:.2f}%')
    
    return 1-accuracy, None
