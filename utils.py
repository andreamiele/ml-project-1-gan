import numpy as np

def build_k_indices(y, k_fold, seed):
  num_row = y.shape[0]
  interval = int(num_row / k_fold)
  np.random.seed(seed)
  indices = np.random.permutation(num_row)
  k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
  return np.array(k_indices)
  
def cross_validation_one(y, x, k_indices, k, hp, method, loss):
  x_test = x[k_indices[k],:]
  y_test = y[k_indices[k]]
  train_indices = np.array([])
  i = 0
  for i in range(len(k_indices)):
    if i == k: continue
    train_indices = np.append(train_indices, k_indices[i])
      
  x_train = x[train_indices.astype(int)]
  y_train = y[train_indices.astype(int)]
  
  w = method(y_train, x_train, hp)
  loss_te = loss(y_test, x_test, w)
  return loss_te

def cross_validation(x, y , method, loss, k_fold, hyperparams):
  seed = 12
  degree = degree
  k_fold = k_fold
  k_indices = build_k_indices(y, k_fold, seed)
  
  loss_te = []
  for hp in hyperparams:
    loss_te_temp = []
    for k in range(k_fold):
      loss_te = cross_validation(y, x, k_indices, k, hp, method, loss)
      loss_te_temp.append(loss_te)
    loss_te.append(np.mean(loss_te_temp))
  best_hp = hyperparams[np.argmin(loss_te)]
  best_loss = loss_te[np.argmin(loss_te)]

  return best_hp, best_loss

def knn(x_train, y_train, x_test, k):
  y_test = np.ones(x_test.shape[0])
  for i,x in enumerate(x_test):
    distances = []
    for j,x_t in enumerate(x_train):
      distances.append((np.linalg.norm(x-x_t),j))
    distances = sorted(distances)
    count = 0
    for nn in distances[:k]:
      if y_train[nn[1]] == 1:
        count += 1
    if count < k/2:
      y_test[i] = -1
  return y_test