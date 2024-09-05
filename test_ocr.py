import os
import csv
import cv2
import yaml
import torch
import models
import datasets
import argparse
import numpy as np
import torch.nn as nn
# import tensorflow as tf
from collections import Counter
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from Levenshtein import distance
from train import make_dataloader
from matplotlib import pyplot as plt
import torchvision.transforms as T
import kornia as K
torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

def prepare_testing():
    # Create a data loader for the test dataset
    test_loader = make_dataloader(config['test_dataset'], tag='test')
    # If multiple GPUs are available, use DataParallel to parallelize the SR model
    # if n_gpus > 1:
    #     model_sr = nn.parallel.DataParallel(model_sr)
    sv_file = config['model_ocr']
    sv_file = torch.load(sv_file['load'])
    # Load the OCR model based on the configuration
    model_ocr = models.make(sv_file['model'], load_model=True).cuda()
    
    # Return the test data loader, the SR model, and the OCR model
    return test_loader, model_ocr

def build_character_accuracy_histogram(ground_truth, predictions, bar_width=0.6, space_between_bars=1.5, figure_size=(20, 12), title_postfix = 'brazilian'):
    """
    Build and plot a stylized histogram showing the percentage of correct predictions
    for each character, relative to the total number of occurrences of that character,
    and display the total number of occurrences in the labels.
    
    Parameters:
    ground_truth (list of str): The list of correct license plate strings.
    predictions (list of str): The list of license plate strings predicted by the OCR.
    bar_width (float): The width of each bar in the histogram.
    space_between_bars (float): The space between bars in the histogram.
    figure_size (tuple): The size of the figure (width, height).
    
    Returns:
    char_accuracy_percentage (dict): A dictionary with characters as keys and their correct prediction percentage as values.
    """
    
    total_characters_count = Counter()
    correct_characters_count = Counter()

    for pred, gt in zip(predictions, ground_truth):
        for p_char, g_char in zip(pred, gt):
            total_characters_count[g_char] += 1  # Count every occurrence of the ground truth character
            if p_char == g_char:
                correct_characters_count[g_char] += 1  # Count correct predictions

    characters = sorted(total_characters_count.keys())
    char_accuracy_percentage = {char: (correct_characters_count[char] / total_characters_count[char]) * 100 
                                for char in characters}

    # Calculate bar positions with extra space
    positions = np.arange(len(characters)) * (bar_width + space_between_bars)

    # Set the figure size
    plt.figure(figsize=figure_size)

    counts = [char_accuracy_percentage[char] for char in characters]
    plt.bar(positions, counts, width=bar_width, color='lightgreen')
    plt.xlabel('Character')
    plt.ylabel('Correct Prediction Percentage (%)')
    plt.title(f'OCR Character Prediction Accuracy Percentage - {title_postfix}' )

    # Add percentages above the bars
    for i, percentage in enumerate(counts):
        plt.text(positions[i], percentage + 0.5, f'{percentage:.1f}%', ha='center', va='bottom')

    # Set custom x-ticks with corresponding labels
    character_labels = [f'{char}\n{total_characters_count[char]}' for char in characters]
    plt.xticks(positions, character_labels)

    plt.show()
    
    return char_accuracy_percentage

def build_ocr_accuracy_histogram(ground_truth, predictions, title_postfix = 'brazilian'):
    """
    Build and plot a stylized histogram showing the number of license plates 
    with a specific number of correctly predicted characters, including totals.
    
    Parameters:
    ground_truth (list of str): The list of correct license plate strings.
    predictions (list of str): The list of license plate strings predicted by the OCR.
    
    Returns:
    histogram (Counter): A counter object representing the frequency of correct character counts.
    """
    
    def count_correct_characters(pred, gt):
        return sum(p == g for p, g in zip(pred, gt))

    histogram = Counter()

    for pred, gt in zip(predictions, ground_truth):
        correct_count = count_correct_characters(pred, gt)
        histogram[correct_count] += 1

    total_lps = len(predictions)
    histogram_list = [histogram[i] for i in range(8)]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(8), histogram_list, tick_label=range(8), color='skyblue')
    plt.xlabel('Number of Correct Characters')
    plt.ylabel('Number of License Plates')
    plt.title(f'OCR Prediction Accuracy Histogram - {title_postfix} (Total LPs: {total_lps})')

    for i, count in enumerate(histogram_list):
        plt.text(i, count + 0.2, str(round((count/total_lps)*100, 1))+'%', ha='center', va='bottom')

    plt.show()
    
    return histogram

def test(val_loader, model, save_path):
    total = 0
    name = []
    preds_all_r, sim_preds_r, text_label_r = [], [], []
    converter = strLabelConverter(config['alphabet'])
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    n_correct = 0
    pbar = tqdm(val_loader, leave=False, desc='testing')
    
    for i_batch, batch in enumerate(pbar):
        name.append(batch['name'])
        # text = converter.encode_list(batch['text'], K=7).cuda()
        _, preds,_ = model(batch['img'].cuda())
        preds_all = preds        

        preds = torch.chunk(preds, preds.size(0), 0)    
        
        _, preds_all = preds_all.max(2)
        sim_preds = converter.decode_list(preds_all.data)
        text_label = batch['text']
        
        for pred, target in zip(sim_preds, text_label):
            pred = pred.replace('-', '')
            if pred == target:
                n_correct += 1
            total += 1
        preds_all_r.extend(preds_all)
        sim_preds_r.extend(sim_preds)
        text_label_r.extend(text_label)
       
    for n, raw_pred, pred, gt in zip(name, preds_all, sim_preds, text_label):
        raw_pred = raw_pred.data
        pred = pred.replace('-', '')
        print('%-20s, pred: %-8s, gt: %-8s, match: %s' % (n, pred, gt, pred==gt))    
    accuracy = (n_correct / float(total))
    print(f'accuracy: {accuracy*100:.2f}%')
    
    threshold_counts = [0] * 8
    total_predictions = 0
    # Open the CSV file for writing
    with open(save_path / Path('results.csv'), mode='w+', newline='') as file:
        writer = csv.writer(file)
        
        # Loop through your data and calculate matching character counts
        for raw_pred, pred, gt in zip(preds_all_r, sim_preds_r, text_label_r):
            raw_pred = raw_pred.data
            pred = pred.replace('-', '')
            
            # Calculate the number of matching characters between prediction and ground truth
            matching_chars = len(gt) - distance(pred, gt)
            
            # Update counts for each threshold (if matching_chars is ≥ threshold)
            for i in range(8):  # From 0 to 7 characters
                if matching_chars >= i:
                    threshold_counts[i] += 1
            
            # Increment the total number of predictions processed
            total_predictions += 1
        
        # Calculate total accuracy for each threshold
        total_accuracy = [round((count / total_predictions) * 100 , 2) for count in threshold_counts]

        # Write the header with accuracy columns aligned to the rightmost part
        header = ['Image Name', 'Prediction', 'Ground Truth', 'Match'] + ['TOTAL ACCURACY'] + ['all'] + [f'>= {i}' for i in range(6, -1, -1)]
        writer.writerow(header)
        index = 0
        # Now write the data rows for each prediction
        for n, raw_pred, pred, gt in zip(name, preds_all, sim_preds, text_label):
            index+=1
            raw_pred = raw_pred.data
            pred = pred.replace('-', '')
            
            # Check if the prediction matches the ground truth
            exact_match = pred == gt
            if index == 1:
                writer.writerow([n, pred, gt, exact_match] + [""] + list(reversed(total_accuracy)))
            else:
                # Write the row for each prediction with total accuracy and thresholds on the right
                writer.writerow([n, pred, gt, exact_match] + [""])
            
def main(config_, save_path):
    global config
    config = config_    
         
    # Call the prepare_testing function to set up testing
    test_loader, model_ocr = prepare_testing()

    # Call the test function to perform the testing
    test(test_loader, model_ocr, save_path)
    

if __name__ == '__main__':            
    # Create an argument parser to parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--save', default=None)    
    parser.add_argument('--tag', default=None)

    # Parse the command line arguments
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    # Create a save_name based on the configuration file and tag
    save_name = args.save
    if save_name is not None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    
    # Create a save_path directory for saving the test results
    save_path = Path('./save') / Path(save_name) 
    save_path.mkdir(parents=True, exist_ok=True)    

    # Call the main function to start the testing process
    main(config, save_path)
