"""
Usage: python infer.py --bs=<batch_size> --intra=<num_intra_threads> --inter=<num_inter_threads>  
"""

import argparse
import time
import pickle
import tensorflow as tf
from input import DataInput, DataInputTest
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument("--bs",type=int)
parser.add_argument("--intra",type=int,default=0)
parser.add_argument("--inter",type=int,default=0)

args = parser.parse_args()



with open('dataset.pkl', 'rb') as f:
  train_set = pickle.load(f)
  test_set = pickle.load(f)
  cate_list = pickle.load(f)
  user_count, item_count, cate_count = pickle.load(f)

model = Model(user_count, item_count, cate_count, cate_list)

test_batch_size = args.bs

cpu_config = tf.ConfigProto()
cpu_config.intra_op_parallelism_threads = args.intra
cpu_config.inter_op_parallelism_threads = args.inter


t1=time.time()

with tf.Session(config=cpu_config) as sess:
    model.restore(sess,'save_path/ckpt')
    for k, uij in DataInputTest(test_set, test_batch_size):
        model.test(sess,uij)

t2=time.time()

f = open('inference_time.txt','a')
f.write(str(t2-t1)+"\n")
f.close()
