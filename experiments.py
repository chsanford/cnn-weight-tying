import os
import numpy as np
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import models
from mappings import cnn2fc, cnn2lc
from utils import get_id, get_data, accuracy
import pickle
import sys
import gc
from math import log
from main import train



def run_all_experiments():

	print("Training CNN")
	train_cnn()

	epochs = param_dict['epochs']
	n_splits = param_dict['n_splits']
	splits = np.unique(np.logspace(0, np.log10(epochs), n_splits).astype(int))

	for i, split in enumerate(splits):
		print("Training weight-tied NN #{} at split {}".format(i, split))
		train_untied_nn(split)


def load_network(file):
	# file = <file name>.pyT

	if not file.endswith(".pyT"):
		raise ValueError('file name should end with .pyT')

	# model = file.split("_")[0]
	model_class = getattr(models, model)
	net = model_class(num_classes=num_classes).to(device)
	is_tied = True

	if "_lc_" in file:
		is_tied = False
		net = cnn2lc(net).to(device)

	state = torch.load(file, map_location=device)  # gives the state_dict and opt
	net.load_state_dict(state['weights'])

	return (net, is_tied)


def train_cnn():

	model_class = getattr(models, model)
	net = model_class(num_classes=num_classes).to(device)

	input_dict = param_dict.copy()
	input_dict['net'] = net
	input_dict['is_tied'] = True
	input_dict['n_saves'] = 100
	input_dict['model'] = model
	input_dict['save_dir'] = save_dir
	input_dict['lr'] = input_dict['cnn_lr']

	configure_and_train(**input_dict)


def train_untied_nn(split_number):
	convert_to = 'lc'
	load_model = '{}/{}_{}.pyT'.format(save_dir, model, split_number)
	untied_save_dir = '{}/{}_{}_{}'.format(save_dir, model, convert_to, split_number)

	state = torch.load(load_model, map_location=device) # gives the state_dict and opt
	split_model = load_model.split("/")[-1].split("_")[0] # this is by our saving convention
	model_class = getattr(models, split_model)
	net = model_class(num_classes=num_classes).to(device)
	net.load_state_dict(state['weights'])

	net = cnn2lc(net).to(device)
	split_model += '_lc_version'

	input_dict = param_dict.copy()
	input_dict['net'] = net
	input_dict['is_tied'] = False
	input_dict['n_saves'] = 20
	input_dict['model'] = split_model
	input_dict['save_dir'] = untied_save_dir
	input_dict['lr'] = input_dict['lc_lr']

	configure_and_train(**input_dict)


def configure_and_train(**input_dict):
	save_dir, model, net, lr, epochs, n_saves, is_tied = [input_dict.pop(key) for key in
	('save_dir', 'model', 'net', 'lr', 'epochs', 'n_saves', 'is_tied')]

	opt = optim.SGD(
		net.parameters(),
		lr=lr,
		momentum=mom,
		weight_decay=wd
		)
	p = pickle.dumps(net)
	size = sys.getsizeof(p)
	print('size of model in bytes : ', size)

	crit = nn.CrossEntropyLoss().to(device)

	print(net)

	if n_saves == epochs:
		checkpoints = np.arange(epochs)
	else:
		checkpoints = np.unique(np.logspace(0, np.log10(epochs), n_saves).astype(int))
		checkpoints = np.insert(checkpoints, 0, 0) # add the initial point
	print('checkpoints: {}'.format(checkpoints))

	# save the initial state and args, assumes folder exists
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	state = {'weights': net.state_dict(), 'optimizer': opt.state_dict()}
	model_path = save_dir + '/{}_0.pyT'.format(model)
	torch.save(state, model_path)
	# torch.save(args, save_dir + '/args.pyT')

	# training process
	training_history = {'tr_step_loss': [], 'tr_step_acc': []}
	evaluation_history = {'train_loss': [], 'test_loss': [],
						  'train_acc': [], 'test_acc': [],
						  'margin': [], 'norm': [],
						  'log_prod_norm': [], 'norm_margin_normalized': [],
						  'log_prod_norm_normalized': []}
	training_history['checkpoints'] = checkpoints
	evaluation_history['checkpoints'] = checkpoints

	time_mem_history = {'train': [], 'eval': [], 'total': '', 'mem': []}

	# initial performance
	te_epoch = evaluate(te_loader_eval, net, crit, device)
	print('Test loss and accuracy :', te_epoch)
	tr_epoch = evaluate(tr_loader_eval, net, crit, device)
	print('Train loss and accuracy :', tr_epoch)
	evaluation_history['train_loss'].append(tr_epoch[0])
	evaluation_history['train_acc'].append(tr_epoch[1])
	evaluation_history['test_loss'].append(te_epoch[0])
	evaluation_history['test_acc'].append(te_epoch[1])
	state = {'weights': net.state_dict(), 'optimizer': opt.state_dict()}
	torch.save(state, save_dir + '/{}_0.pyT'.format(model))

	for epoch in range(epochs):

		print('epoch {} begins'.format(epoch + 1))

		t = time.time()
		step_loss, step_acc = train(train_loader, net, crit, opt, device)
		training_history['tr_step_loss'] += step_loss
		training_history['tr_step_acc'] += step_acc
		time_mem_history['train'].append((epoch + 1, '{:3f}'.format(time.time() - t)))

		if (epoch + 1) in checkpoints:
			t = time.time()
			te_epoch = evaluate(te_loader_eval, net, crit, device)
			print('Test loss and accuracy :', te_epoch)
			tr_epoch = evaluate(tr_loader_eval, net, crit, device)
			print('Train loss and accuracy :', tr_epoch)

			# margin = np.average(compute_margins(net))
			margin = te_epoch[2]
			if is_tied:
				norm = weight_norm_tied(net, log_prod=False)
				log_prod_norm = weight_norm_tied(net, log_prod=True)
			else:
				norm = weight_norm(net, log_prod=False)
				log_prod_norm = weight_norm(net, log_prod=True)
			if margin > 0:
				norm_margin_normalized = norm / margin
				log_prod_norm_normalized = log_prod_norm - log(margin)
			else:
				norm_margin_normalized = -1
				log_prod_norm_normalized = -1


			evaluation_history['train_loss'].append(tr_epoch[0])
			evaluation_history['train_acc'].append(tr_epoch[1])
			evaluation_history['test_loss'].append(te_epoch[0])
			evaluation_history['test_acc'].append(te_epoch[1])
			evaluation_history['margin'].append(margin)
			evaluation_history['norm'].append(norm)
			evaluation_history['log_prod_norm'].append(log_prod_norm)
			evaluation_history['norm_margin_normalized'].append(norm_margin_normalized)
			evaluation_history['log_prod_norm_normalized'].append(log_prod_norm_normalized)

			time_mem_history['eval'].append((epoch + 1, '{:3f}'.format(time.time() - t)))

			if device == 'cuda':
				time_mem_history['mem'].append((torch.cuda.memory_allocated() / (1024**2),
											torch.cuda.memory_cached() / (1024**2)))

			state = {'weights': net.state_dict(), 'optimizer': opt.state_dict()}
			model_path = save_dir + '/{}_{}.pyT'.format(model, epoch + 1)
			torch.save(state, model_path)
			torch.save(training_history, save_dir + '/training_history.hist')
			torch.save(evaluation_history, save_dir + '/evaluation_history.hist')
			time_mem_history['total'] = '{:3f}'.format(time.time() - t_init)
			torch.save(time_mem_history, save_dir + '/time_mem_history.hist')


def weight_norm(net, log_prod=False):
	# Produces sum of weight norms squred or the product of layer norms
	params = net.named_parameters()
	norm = 0
	for (name, param) in params:
		if name.endswith("weight"):
			layer_norm = np.linalg.norm(param.cpu().data.numpy(), axis=None)
			if log_prod:
				norm += log(layer_norm)
			else:
				norm += layer_norm ** 2
	return norm


def weight_norm_tied(net, log_prod=False):
	# state = torch.load(model_path) # gives the state_dict and opt
	reference_net = cnn2lc(net)
	# reference_net.load_state_dict(state['weights'])

	params = net.named_parameters()
	reference_params = reference_net.named_parameters()
	
	norm = 0
	for (name, param), (_, reference_param) in zip(params, reference_params):
		if name.endswith("weight"):
			scale = reference_param.nelement() / param.nelement()
			layer_norm = np.linalg.norm(param.cpu().data.numpy(), axis=None)**2 * scale
			if log_prod:
				norm += log(layer_norm)
			else:
				norm += layer_norm
	return norm


def compute_margins(net, is_estimate=False):
	net.eval()
	all_margins = None
	with torch.no_grad(): # alt. just call backward to free memory
		total_size = 0
		total_loss = 0
		total_acc = 0
		for x, y in tr_loader_eval:
			# print(7)
			# loop over dataset
			x, y = x.to(device), y.to(device)
			out = net(x)
			# print(x.shape, y.shape, out.shape)
			y_pred = torch.argmax(out, 1)

			is_correct = y_pred == y
			correct_out = out[range(len(y)), y]
			top_incorrect = torch.where(is_correct, torch.kthvalue(out, 2, dim=1)[0], torch.max(out, 1)[0])
			margin = correct_out - top_incorrect

			if is_estimate:
				return margin.cpu().data.numpy()

			if all_margins is None:
				all_margins = margin.cpu().data.numpy()
			else:
				all_margins = np.concatenate((all_margins, margin.cpu().data.numpy()))

		return all_margins


def evaluate(eval_loader, net, crit, device):
	net.eval()
	all_margins = None
	with torch.no_grad(): # alt. just call backward to free memory
		total_size = 0
		total_loss = 0
		total_acc = 0
		for x, y in eval_loader:
			x, y = x.to(device), y.to(device)
			out = net(x)
			loss = crit(out, y).item()
			prec = accuracy(out, y)
			bs = x.size(0)

			y_pred = torch.argmax(out, 1)
			is_correct = y_pred == y
			correct_out = out[range(len(y)), y]
			top_incorrect = torch.where(is_correct, torch.kthvalue(out, 2, dim=1)[0], torch.max(out, 1)[0])
			margin = correct_out - top_incorrect

			if all_margins is None:
				all_margins = margin.cpu().data.numpy()
			else:
				all_margins = np.concatenate((all_margins, margin.cpu().data.numpy()))
			
			total_size += bs
			total_loss += loss * bs
			total_acc += prec * bs
			
	return [total_loss / total_size, total_acc / total_size, np.average(all_margins)]


def params_per_layer(net):
	# Helper function to determine number of trainable parameters within each layer
	from operator import mul
	from functools import reduce
	params = list(net.parameters())
	n = len(params)
	weights = params[0:n:2]
	biases = params[1:n:2]
	return list(map(lambda x: reduce(mul, x[0].size()) + int(x[1].size()[0]), zip(weights, biases)))


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', default= 100, type=int)
	parser.add_argument('--n_splits', default= 20, type = int)
	parser.add_argument('--cnn_lr', default= 0.1, type=float)
	parser.add_argument('--lc_lr', default= 0.01, type=float)
	parser.add_argument('--exec', default=0, type=int, choices=[0, 1])

	param_dict = vars(parser.parse_args())

	t_init = time.time()

	# Static global variables
	save_dir = './results/' + '-'.join(["{}={}".format(*item) for item in param_dict.items()])
	dataset = 'cifar10'
	model = 'skinnyprime'
	path = 'data'
	data_size = 0
	mom = 0
	wd = 0
	bs_train = 250
	bs_eval = 1000
	seed = 0

	use_cuda = torch.cuda.is_available()
	device = torch.device('cuda' if use_cuda else 'cpu')
	torch.manual_seed(seed)

	train_loader, tr_loader_eval, te_loader_eval, num_classes = get_data(dataset, path, bs_train, bs_eval, data_size)

	if param_dict['exec']:
		run_all_experiments()
