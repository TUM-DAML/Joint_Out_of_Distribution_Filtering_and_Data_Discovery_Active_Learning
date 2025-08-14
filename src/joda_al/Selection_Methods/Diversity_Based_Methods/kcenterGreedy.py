
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Returns points that minimizes the maximum distance of any point to a center.
Implements the k-Center-Greedy method in
Ozan Sener and Silvio Savarese.  A Geometric Approach to Active Learning for
Convolutional Neural Networks. https://arxiv.org/abs/1708.00489 2017
Distance metric defaults to l2 distance.  Features used to calculate distance
are either raw features or if a model has transform method then uses the output
of model.transform(X).
Can be extended to a robust k centers algorithm that ignores a certain number of
outlier datapoints.  Resulting centers are solution to multiple integer program.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance
from tqdm import tqdm
import abc
import numpy as np

class SamplingMethod(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def __init__(self, X, y, seed, **kwargs):
    self.X = X
    self.y = y
    self.seed = seed

  def flatten_X(self):
    shape = self.X.shape
    flat_X = self.X
    if len(shape) > 2:
      flat_X = np.reshape(self.X, (shape[0],np.product(shape[1:])))
    return flat_X


  @abc.abstractmethod
  def select_batch_(self):
    return

  def select_batch(self, **kwargs):
    return self.select_batch_(**kwargs)

  def select_batch_unc_(self, **kwargs):
      return self.select_batch_unc_(**kwargs)

  def to_dict(self):
    return None



class kCenterGreedy(SamplingMethod):

    def __init__(self, X,  metric='euclidean', weights=None):
        self.X = X
        # self.y = y
        self.flat_X = self.flatten_X()
        self.name = 'kcenter'
        self.features = self.flat_X
        self.metric = metric
        self.min_distances = None
        self.max_distances = None
        self.n_obs = self.X.shape[0]
        self.already_selected = []
        self.weights = weights
        self.vi=None

    def calculate_cov(self,X):
        try:
            VI = np.linalg.inv(np.cov(X.T)).T
        except:
            VI = np.linalg.pinv(np.cov(X.T)).T
        return VI

    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        """Update min distances given cluster centers.
        Args:
          cluster_centers: indices of cluster centers
          only_new: only calculate distance for newly selected points and update
            min_distances.
          rest_dist: whether to reset min_distances.
        """
        if self.vi is None and self.metric == "mahalanobis":
            self.vi=self.calculate_cov(self.features[cluster_centers])
        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [d for d in cluster_centers
                            if d not in self.already_selected]
        if len(cluster_centers)==0:
            if self.metric == 'mahalanobis':
                dist = pairwise_distances(self.features, None, metric=self.metric, VI=self.vi)  # ,n_jobs=4)
            else:
                dist = pairwise_distances(self.features, None, metric=self.metric)
            idx=np.argmin(dist[dist > 0])
            row, col = np.unravel_index(idx, dist.shape)
            cluster_centers+=[row]
        if cluster_centers:
            x = self.features[cluster_centers]
            # Update min_distances for all examples given new cluster center.
            if self.metric == 'mahalanobis':
                dist = pairwise_distances(self.features, x, metric=self.metric,VI=self.vi)#,n_jobs=4)
            else:
                dist = pairwise_distances(self.features, x, metric=self.metric)

        if self.min_distances is None:
            self.min_distances = np.min(dist, axis=1).reshape(-1,1)
        else:
            self.min_distances = np.minimum(self.min_distances, dist)

    def select_batch_(self, already_selected, N, break_on_overlap=False, **kwargs):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.
        Args:
          model: model with scikit-like API with decision_function implemented
          already_selected: index of datapoints already selected
          N: batch size
        Returns:
          indices of points selected to minimize distance to cluster centers
        """
        # Both empty
        if not (self.already_selected and already_selected):
            # Initialize centers with a randomly selected datapoint
            ind = np.random.choice(np.arange(self.n_obs))
            already_selected=np.concatenate([already_selected,np.array([ind])])
        try:
          # Assumes that the transform function takes in original data and not
          # flattened data.
          print('Getting transformed features...')
        #   self.features = model.transform(self.X)
          print('Calculating distances...')
          self.update_distances(already_selected, only_new=False, reset_dist=True)
        except:
          print('Using flat_X as features.')
          self.update_distances(already_selected, only_new=True, reset_dist=False)

        new_batch = []
        new_batch_distance = []
        if N > (self.features.shape[0]-already_selected.shape[0]):
            print('Requested Batch Size large than available data')
            N=(self.features.shape[0]-already_selected.shape[0])
        for _ in range(N):
            if self.already_selected is None: # [] is not None! so happens never
                # Initialize centers with a randomly selected datapoint
                ind = np.random.choice(np.arange(self.n_obs))
            else:
                ind = np.argmax(self.min_distances)
                # New examples should not be in already selected since those points
                # should have min_distance of zero to a cluster center.
                if ind in already_selected:
                    print(f"Dataoverlap at {_}")
                    if break_on_overlap:
                        break

            new_batch_distance.append(self.min_distances[ind])
            self.update_distances([ind], only_new=True, reset_dist=False)
            new_batch.append(ind)
        print('Maximum distance from cluster centers is %0.2f' % max(self.min_distances))

        self.already_selected = already_selected

        return new_batch_distance, new_batch

class kCenterGreedyOOD(kCenterGreedy):
  
    def select_batch_(self, already_selected, N, break_on_overlap=False, **kwargs):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.
        Args:
          model: model with scikit-like API with decision_function implemented
          already_selected: index of datapoints already selected
          N: batch size
        Returns:
          indices of points selected to minimize distance to cluster centers
        """

        try:
          # Assumes that the transform function takes in original data and not
          # flattened data.
          print('Getting transformed features...')
        #   self.features = model.transform(self.X)
          print('Calculating distances...')
          self.update_distances(already_selected, only_new=False, reset_dist=True)
        except:
          print('Using flat_X as features.')
          self.update_distances(already_selected, only_new=True, reset_dist=False)

        new_batch = []
        new_batch_distance = []
        if N > (self.features.shape[0]-already_selected.shape[0]):
            print('Requested Batch Size large than available data')
            N=(self.features.shape[0]-already_selected.shape[0])
        for _ in tqdm(range(N)):
            if self.already_selected == []:
                # Initialize centers with a randomly selected datapoint
                ind = np.random.choice(np.arange(self.n_obs))
                new_batch_distance.append(float(0))
                self.update_distances([ind], only_new=True, reset_dist=False)
                new_batch.append(ind)
                self.already_selected = [ind]
                continue
            else:
                ind = np.argmax(self.min_distances)
                # New examples should not be in already selected since those points
                # should have min_distance of zero to a cluster center.
                if ind in already_selected:
                    print(f"Dataoverlap at {_}")
                    if break_on_overlap:
                        break
               
            new_batch_distance.append(self.min_distances[ind])
            self.update_distances([ind], only_new=True, reset_dist=False)
            new_batch.append(ind)
        print('Maximum distance from cluster centers is %0.2f'
                % max(self.min_distances))


        self.already_selected = already_selected

        return new_batch_distance, new_batch