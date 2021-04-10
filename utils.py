import os

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
import warnings

sns.set(style='white')


# Flattens a list of dicts with torch Tensors
def flatten_list_dicts(list_dicts):
  return {k: torch.cat([d[k] for d in list_dicts], dim=0) for k in list_dicts[-1].keys()}


# Makes a lineplot with scalar x and statistics of vector y
def lineplot(x, y, filename, xaxis='Steps', yaxis='Returns'):
  y = np.array(y)
  y_mean, y_std = y.mean(axis=1), y.std(axis=1)
  sns.lineplot(x=x, y=y_mean, color='coral')
  plt.fill_between(x, y_mean - y_std, y_mean + y_std, color='coral', alpha=0.3)
  plt.xlim(left=0, right=x[-1])
  plt.ylim(bottom=0, top=500)  # Return limits for CartPole-v1
  plt.xlabel(xaxis)
  plt.ylabel(yaxis)
  plt.savefig(os.path.join('./results', filename + '.png'))
  plt.close()



class MetricSaver():
  '''This object stores relevant data from the training/test runs. Used in the Hydra sweeping and analyzizng the data later'''
  def __init__(self, load_file=None, algorithm=None, env=None):
    self.data = dict()
    self.data['algorithm'] = algorithm
    self.data['env'] = env
    if load_file:
      self.load_data(load_file)

  def __getitem(self, item):
      return self.data[item]

  def save_data(self, filename="./result_data.pth"):
    if os.path.exists(filename):
      warnings.warn(filename + " already exists.")
      i=1
      fp, fn = os.path.split(filename)
      fsp = fn.split('.')
      new_filename = fsp[0] + str(i) + '.' + fsp[-1]
      while os.path.exists(fp+'/'+new_filename):
        i+=1
        new_filename = fsp[0] + str(i) + '.' + fsp[-1]
      warnings.warn('saved file as ' + fp+'/'+ new_filename)
      torch.save(self.data, fp+'/'+new_filename)
      return
    torch.save(self.data, filename)
    return

  def load_data(self, filename=None):
    if filename:
      self.data = torch.load(filename)
    else:
      raise ValueError("No filename specified!")

  def add_train_step(self, step, loss):
    if 'training_step' not in self.data.keys():
      self.data['training_step'] = [step]
    else:
      self.data['training_step'].append(step)

    if 'training_loss' not in self.data.keys():
      self.data['training_loss'] = [loss]
    else:
      self.data['training_loss'].append(loss)


  def add_test_step(self, step, losses):
    if 'test_step' not in self.data.keys():
      self.data['test_step'] = [step]
    else:
      self.data['test_step'].append(step)

    y = np.array(losses)
    loss_mean, loss_std = y.mean(), y.std()
    if 'test_loss_mean' not in self.data.keys():
      self.data['test_loss_mean'] = [loss_mean]
    else:
      self.data['test_loss_mean'].append(loss_mean)
    if 'test_loss_std' not in self.data.keys():
      self.data['test_loss_std'] = [loss_std]
    else:
      self.data['test_loss_std'].append(loss_std)

  def store_model(self, model : torch.nn.Module):
    if "model" not in self.data.keys():
      self.data['model'] = model.state_dict()
  def load_model(self, model : torch.nn.Module):
    if 'model' in self.data.keys():
      model.load_state_dict(self.data['model'])
      return model

  def store_model_checkpoint(self, model : torch.nn.Module, step=None):
    if "model_checkpoint" not in self.data.keys():
      self.data['model_checkpoint'] = [model.state_dict()]
      self.data['model_checkpoint_step'] = [step]
    else:
      self.data['model_checkpoint'].append(model.state_dict())
      self.data['model_checkpoint_step'].append(step)
    return
  def load_model_checkpoint(self, model : torch.nn.Module, step=None):
    if 'model_checkpoint' not in self.data.keys():
      raise ValueError('No model_checkpoint in data')
    if step:
      index = self.data['model_checkpoint_step'].index(step)
      model.load_state_dict(self.data['model_checkpoint'][index])
      return model
    else:
      model.load_state_dict(self.data['model_checkpoint'][-1])
      return model