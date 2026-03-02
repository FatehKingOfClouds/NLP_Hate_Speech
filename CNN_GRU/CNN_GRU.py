import neptune
import numpy as np
import os
from Models.modelUtils import *
from Models.torchDataGen import *

from transformers import *

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,classification_report,accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder
import torch
import time
from Models.model import *
from collections import Counter
import collections
import io
import numpy as np
from bert_codes.feature_generation import combine_features,return_dataloader,return_cnngru_dataloader
from bert_codes.data_extractor import data_collector
from bert_codes.own_bert_models import *
from bert_codes.utils import *
import glob
from BERT_inference import *

# Function to select the appropirate model
def select_model(args,vector=None):
    text=args["path_files"]
    if(text=="birnn"):
        model=BiRNN(args)
    if(text == "birnnatt"):
        model=BiAtt_RNN(args,return_att=False)
    if(text == "birnnscrat"):
        model=BiAtt_RNN(args,return_att=True)
    if(text == "cnngru"):
        model=CNN_GRU(args,vector)
    if(text == "lstm_bad"):
        model=LSTM_bad(args)
    return model

# Function to load the MUSE embeddings
def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id

# Function to evaluate CNN-GRU model
def Eval_phase_cnngru(params, phase, model, data):
    """Evaluate the CNN-GRU model on train/val/test data"""
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model.eval()
    
    # Create dataloader for evaluation
    eval_dataloader = return_cnngru_dataloader(data, batch_size=params['batch_size'], is_train=False)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in eval_dataloader:
            # Get batch data
            if len(batch) >= 2:
                b_input_ids = batch[0].to(device)
                b_labels = batch[1].to(device)
            else:
                continue
            
            # Get model predictions
            outputs = model(b_input_ids)
            
            # Get predicted class
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Handle case where batch size=1 and squeeze removes batch dimension
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(b_labels.cpu().numpy())
    
    # Calculate metrics
    fscore = f1_score(all_labels, all_predictions, average='binary', zero_division=0)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    model.train()
    return fscore, accuracy


# Main training function

def cnn_gru_train_model(params):
    # Determine file paths based on training mode
    if(params['how_train']=='baseline'):
        train_path=os.path.join(params['files'], 'train', params['language']+'*_full.csv')
        val_path=os.path.join(params['files'], 'val', params['language']+'*_full.csv')
        test_path=os.path.join(params['files'], 'test', params['language']+'*_full.csv')
    else:
        # For 'all' mode, load all *_full.csv files (untranslated data)
        train_path=os.path.join(params['files'], 'train', '*_full.csv')
        val_path=os.path.join(params['files'], 'val', '*_full.csv')
        test_path=os.path.join(params['files'], 'test', '*_full.csv')
    
    # Load the datasets
    train_files=glob.glob(train_path)
    val_files=glob.glob(val_path)
    test_files=glob.glob(test_path)
    df_train=data_collector(train_files,params,True)
    df_val=data_collector(val_files,params,False)
    df_test=data_collector(test_files,params,False)
    
    # Encode the datasets
    lang_map={'Arabic':'ar','French':'fr','Portugese':'pt','Spanish':'es','English':'en','Indonesian':'id','Italian':'it','German':'de','Polish':'pl'}
    path='muse_embeddings/wiki.multi.'+lang_map[params['language']]+'.vec'
    vector,id2word,word2id=load_vec(path)
    train_data=encode_data(df_train,word2id)
    val_data=encode_data(df_val,word2id)
    test_data=encode_data(df_test,word2id)
    
    pad_vec=np.random.randn(1,300) 
    unk_vec=np.random.randn(1,300)
    merged_vec=np.append(vector, unk_vec, axis=0)
    merged_vec=np.append(merged_vec, pad_vec, axis=0)
    params['vocab_size']=merged_vec.shape[0]

    # Generate the dataloaders
    train_dataloader = return_cnngru_dataloader(train_data,batch_size=params['batch_size'],is_train=params['is_train'])
    validation_dataloader=return_cnngru_dataloader(val_data,batch_size=params['batch_size'],is_train=False)
    
    model=select_model(params,merged_vec)
    # Tell pytorch to run this model on the GPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"CNN-GRU model on: {device}")


    optimizer = AdamW(model.parameters(),
                  lr = params['learning_rate'], # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = params['epsilon'] # args.adam_epsilon  - default is 1e-8.
                )

    fix_the_random(seed_val = params['random_seed'])
    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    bert_model = params['path_files']
    language  = params['language']
    name_one=bert_model+"_"+language
    if(params['take_target']):
        name_one += '_target'
    if(params['logging']=='neptune'):
        neptune.create_experiment(name_one,params=params,send_hardware_metrics=False,run_monitoring_thread=False)
        neptune.append_tag(bert_model)
        neptune.append_tag(language)
        neptune.append_tag('storing_best')
        if(params['pair']):
            neptune.append_tag('sentences_pair')
        if(params['take_target']):
            neptune.append_tag('sentence_target')
    # For each epo=ch...
    best_val_fscore=0
    best_test_fscore=0

    for epoch_i in range(0, params['epochs']):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, params['epochs']))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in tqdm(enumerate(train_dataloader)):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(device)
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            outputs = model(b_input_ids,b_labels)
            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple.
            loss = outputs[0]

            if(params['logging']=='neptune'):
                neptune.log_metric('batch_loss',loss)
            else:
                if step % 40 == 0 and not step == 0:
                    print('batch_loss',loss)

            #Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            # Update the learning rate.
            #scheduler.step()
            
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        train_fscore,train_accuracy=Eval_phase_cnngru(params,'train',model,train_data)
        if(params['logging']=='neptune'):
            neptune.log_metric('avg_train_loss',avg_train_loss)
            neptune.log_metric('train_fscore',train_fscore)
            neptune.log_metric('train_accuracy',train_accuracy)
        else:
            print('avg_train_loss',avg_train_loss)
            print('train_fscore',train_fscore)
            print('train_accuracy',train_accuracy)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        val_fscore,val_accuracy=Eval_phase_cnngru(params,'val',model,val_data)      
        test_fscore,test_accuracy=Eval_phase_cnngru(params,'test',model,test_data)

        #Report the final accuracy for this validation run.
        if(params['logging']=='neptune'):   
            neptune.log_metric('val_fscore',val_fscore)
            neptune.log_metric('val_acc',val_accuracy)
            neptune.log_metric('test_fscore',test_fscore)
            neptune.log_metric('test_accuracy',test_accuracy)
        if(val_fscore > best_val_fscore):
            print(val_fscore,best_val_fscore)
            best_val_fscore=val_fscore
            best_test_fscore=test_fscore
            #save_model(model,tokenizer,params)
#   
    if(params['logging']=='neptune'):
        neptune.log_metric('best_val_fscore',best_val_fscore)
        neptune.log_metric('best_test_fscore',best_test_fscore)
        neptune.stop()
    else:
        print('best_test_fscore',best_test_fscore)
        print('best_val_fscore',best_val_fscore)
    del model
    torch.cuda.empty_cache()
    return 1
                    
                                                
                                                
                                                
                                                
# Explanation of all the params used below. 

# 'logging':where logging {'local','neptune'}
# 'language': language {'Arabic', 'English','German','Indonesian','Italian','Polish','Portugese','Spanish','French'}
# 'is_train': whether train dataset 
# 'is_model':is model 
# 'learning_rate':Adam parameter lr
# 'files': Path to the dataset folder ( containing the train, val and test subfolders)
# 'csv_file': The regex used by glob to load the datasets. {'*_full.csv','*_translated.csv'} for untranslated and translated datasets respectively
# 'samp_strategy': The way in which we sample the training data points. {'stratified'}
# 'epsilon': Adam parameter episilon
# 'path_files':bert path from where the bert model should be loaded,
# 'take_ratio': Whether the sample ratio is ratio of total points or absolute number of points needed.
# 'sample_ratio':ratio or the number of the training data points to take
# 'how_train':how the bert is trained possible option {'all','baseline','all_but_one'}
# 'epochs': number of epochs to train bert
# 'batch_size': batch size
# 'to_save': whether to save the model or not
# 'weights': weights for binary classifier
# 'what_bert': type of bert possible option {'normal','weighted'}
# 'save_only_bert': if only bert (without classifier) should be used 
# 'max_length': maximum length for input tokenization
# 'random_seed': seed value for reproducibility                                        
                                                
                                                
                                                
params={
    'logging':'local',
    'language':'English',
    'is_train':True,
    'is_model':True,
    'learning_rate':1e-4,
    'files':'../Dataset',
    'csv_file':'*_full.csv',
    'samp_strategy':'stratified',
    'epsilon':1e-8,
    'path_files':'cnngru',
    'take_ratio':True,
    'sample_ratio':100,
    'how_train':'all',
    'epochs':5,
    'batch_size':16,
    'to_save':True,
    'weights':[1.0,1.0],
    'what_bert':'normal',
    'save_only_bert':False,
    'max_length':128,
    'columns_to_consider':['directness','target','group'],
    'random_seed':42,
    'embed_size':300,
    'train_embed':True,
    'take_target':False,
    'pair':False
}
 
    
    
if __name__=='__main__':

	cnn_gru_train_model(params)
	print('============================')
	print(f'Model for Language {params["language"]} trained successfully!')
	print('============================' )
    
    #train_model(params)
