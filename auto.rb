#!/usr/bin/ruby
require 'optparse'

# parse some command line options
dataset = nil
epochs = nil
download = nil
num-executors = nil

OptionParser.new do |opts|
  opts.on("-d dataset", "--dataset dataset", "Target dataset") do |val|
    dataset = val
  end
  opts.on("-e epochs", "--epochs epochs", "Number of epochs") do |val|
    epochs = val
  end
  opts.on("-dl download", "--download download", "If to download the dataset") do |val|
    download = val
  end
  opts.on("-n num-executors", "--num-executors num-executors", "Number of executors") do |val|
	num-executors = val
  end
end.parse(ARGV)
raise "You must specify a target dataset (-i)" if dataset.nil?
raise "You must specify the number of epochs (-o)" if epochs.nil?
raise "You must specify if to download the dataset (-dl)" if download.nil?
raise "You must specify Number of executors (-n)" if num-executors.nil?

if dataset == 'mnist' 
	if download == 'true'
		`rm *.gz`
		`curl -O "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"`
		`curl -O "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"`
		`curl -O "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"`
		`curl -O "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"`
		`zip -r mnist.zip *.gz`
	end
	`hadoop fs -rmr mnist/csv`
	`spark-submit \
	--master yarn \
	--deploy-mode cluster \
	--queue default \
	--num-executors #{num-executors} \
	--archives mnist.zip#mnist \
	mnist_data_setup.py \
	--output mnist/csv \
	--format csv`
	
	`spark-submit \
	--master yarn \
	--deploy-mode cluster \
	--queue default \
	--num-executors #{num-executors} \
	--py-files mnist_dist2.py \
	--conf spark.dynamicAllocation.enabled=false \
	--conf spark.yarn.maxAppAttempts=1 \
	mnist_spark2.py \
	--images mnist/csv/train/images \
	--labels mnist/csv/train/labels \
	--mode train \
	--model mnist_model`
elsif dataset == 'cifar10'
	if download == 'true'
		`rm *.tar.gz`
		`curl -O "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"`
	end
	`hadoop fs -rmr cifar10/csv`
	`spark-submit \
	--master yarn \
	--deploy-mode cluster \
	--queue default \
	--num-executors #{num-executors} \
	--archives cifar-10-python.tar.gz#dataset \
	cifar10_data_setup.py \
	--output mnist/csv \
	--format csv`
	
	`spark-submit --master yarn \
	--deploy-mode cluster \
	--queue default \
	--num-executors #{num-executors} \
	--py-files cifar10.zip \
	--conf spark.dynamicAllocation.enabled=false \
	--conf spark.yarn.maxAppAttempts=1 \
	cifar10_spark2.py \
	--images cifar10/csv/train/images \
	--labels cifar10/csv/train/labels \
	--mode train \
	--model cifar10_model \
	--epochs 1200 \
	--steps 50000`
	
elsif dataset=='cifar100'
	if download == 'true'
		`rm *.tar.gz`
		`curl -O "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"`
	end
	`hadoop fs -rmr cifar100/csv`
	`hadoop fs -rmr mnist/csv`
	`spark-submit \
	--master yarn \
	--deploy-mode cluster \
	--queue default \
	--num-executors #{num-executors} \
	--archives cifar-100-python.tar.gz#dataset \
	cifar10_data_setup.py \
	--output mnist/csv \
	--format csv`
	
	`spark-submit --master yarn \
	--deploy-mode cluster \
	--queue default \
	--num-executors #{num-executors} \
	--py-files cifar100_spark2.py \
	--conf spark.dynamicAllocation.enabled=false \
	--conf spark.yarn.maxAppAttempts=1 \
	cifar100_spark2.py \
	--images cifar10/csv/train/images \
	--labels cifar10/csv/train/labels \
	--mode train \
	--model cifar10_model \
	--epochs 120 \
	--steps 10000`
end
		