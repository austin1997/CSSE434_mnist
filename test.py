from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from tensorflowonspark import TFCluster
from tensorflowonspark import TFNode
#from com.yahoo.ml.tf import TFCluster, TFNode
from datetime import datetime


def main_fun(argv, ctx):
  import tensorflow as tf
  worker_num = ctx.worker_num
  job_name = ctx.job_name
  task_index = ctx.task_index
  
  cluster_spec, server = TFNode.start_cluster_server(ctx)
  '''  if job_name == "ps":
    time.sleep((worker_num + 1) * 5)
	
  if job_name == "ps":
    server.join()
  elif job_name == "worker":'''
  hello = tf.constant('Hello, TensorFlow!')
  sess = tf.Session()
  print(sess.run(hello))
  
 
 
if __name__ == '__main__':
  # tf.app.run()
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")
  
  sc = SparkContext(conf=SparkConf().setAppName("your_app_name"))
  num_executors = int(sc._conf.get("spark.executor.instances"))
  num_ps = 1
  tensorboard = True

  cluster = TFCluster.run(sc, main_fun, sys.argv, num_executors, num_ps, tensorboard, TFCluster.InputMode.TENSORFLOW)
  cluster.shutdown()