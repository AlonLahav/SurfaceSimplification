import time
import os
import copy
import heapq

from tqdm import tqdm
import numpy as np
import pylab as plt

import io_off_model
import mesh_calc

# Some flags / constants to define the simplification
ENABLE_NON_EDGE_CONTRACTION = False
CLOSE_DIST_TH = 0.1
SAME_V_TH_FOR_PREPROCESS = 0.001
PRINT_COST = False
SELF_CHECKING = True
np.set_printoptions(linewidth=200)

def calc_Q_for_vertex(mesh, v_idx):
  # Calculate K & Q according to eq. (2)
  Q = np.zeros((4, 4))
  for f_idx in np.where(mesh['vf_adjacency_matrix'][v_idx])[0]:
    plane_params = mesh['face_plane_parameters'][f_idx][:, None]
    Kp = plane_params * plane_params.T
    Q += Kp
  return Q

def calc_Q_for_each_vertex(mesh):
  # Prepare some mesh paramenters and run on all vertices to call Q calculation
  mesh_calc.calc_vf_adjacency_matrix(mesh)
  mesh_calc.calc_face_plane_parameters(mesh)
  mesh['all_v_in_same_plane'] = np.abs(np.diff(mesh['face_plane_parameters'], axis=0)).sum() == 0
  mesh['Qs'] = []
  for v_idx in range(mesh['n_vertices']):
    Q = calc_Q_for_vertex(mesh, v_idx)
    mesh['Qs'].append(Q)

def add_pair(mesh, v1, v2, edge_connection):
  # Add pair of indices to the heap, keys by the cost
  Q = mesh['Qs'][v1] + mesh['Qs'][v2]
  new_v1_ = calc_new_vertex_position(mesh, v1, v2, Q)
  if mesh['all_v_in_same_plane']:
    cost = np.linalg.norm(mesh['vertices'][v1] - mesh['vertices'][v2])
  else:
    new_v1 = np.vstack((new_v1_[:, None], np.ones((1, 1))))
    cost = np.dot(np.dot(new_v1.T, Q), new_v1)[0, 0]
  if PRINT_COST:
    print('For pair: ', v1, ',', v2, ' ; the cost is: ', cost)
  heapq.heappush(mesh['pair_heap'], (cost, v1, v2, edge_connection, new_v1_))

def select_vertex_pairs(mesh):
  mesh_calc.calc_v_adjacency_matrix(mesh)
  print('Calculating pairs cost and add to heap')
  tb = time.time()
  for v1 in tqdm(range(mesh['n_vertices'])):
    for v2 in range(v1 + 1, mesh['n_vertices']):
      edge_connection = mesh['v_adjacency_matrix'][v1, v2]
      vertices_are_very_close = ENABLE_NON_EDGE_CONTRACTION and np.linalg.norm(mesh['vertices'][v2] - mesh['vertices'][v1]) < CLOSE_DIST_TH
      if edge_connection or vertices_are_very_close:
        add_pair(mesh, v1, v2, edge_connection)
  print('time:', time.time() - tb)

def look_for_minimum_cost_on_connected_line():        # TODO
  return None

def calc_new_vertex_position(mesh, v1, v2, Q):
  # Calculating the new vetrex position, given 2 vertices (paragraph 4.):
  # 1. If A (to be defined below) can be inverted, use it
  # 2. If this matrix is not invertible, attempt to find the optimal vertex along the segment V1 and V2
  # 3. The new vertex will be at the midpoint
  A = Q.copy()
  A[3] = [0, 0, 0, 1]                                 # Defined by eq. (1)
  A_can_be_ineverted = np.linalg.matrix_rank(A) == 4  # TODO: bug fix!
  A_can_be_ineverted = False
  if A_can_be_ineverted:
    A_inv = np.linalg.inv(A)
    new_v1 = np.dot(A_inv, np.array([[0, 0, 0, 1]]).T)[:3]
    new_v1 = np.squeeze(new_v1)
  else:
    new_v1 = look_for_minimum_cost_on_connected_line()
    if new_v1 is None:
      new_v1 = (mesh['vertices'][v1] + mesh['vertices'][v2]) / 2

  return new_v1

def contract_best_pair(mesh):
  # Get the best pair of indices from heap, and contract them to a single vertex

  # get pair from heap
  if len(mesh['pair_heap']) == 0:
    return
  cost, v1, v2, is_edge, new_v1 = heapq.heappop(mesh['pair_heap'])

  # update v1 - position
  mesh['vertices'][v1] = new_v1

  # remove v2:
  mesh['vertices'][v2] = [-1, -1, -1]                 # "remove" vertex from mesh (will be finally removed at function: clean_mesh_from_removed_items)
  mesh['v_adjacency_matrix'][v1, v2] = mesh['v_adjacency_matrix'][v2, v1] = False
  if is_edge:
    all_v2_faces = np.where(mesh['vf_adjacency_matrix'][v2])[0]
    for f in all_v2_faces:
      if v1 in mesh['faces'][f]:                      # If the face contains v2 also share vertex with v1:
        mesh['faces'][f] = [-1, -1, -1]               #  "remove" face from mesh.
        mesh['vf_adjacency_matrix'][v2, v1] = False
      else:                                           # else:
        v2_idx = np.where(mesh['faces'][f] == v2)[0]  #  replace v2 with v1
        new_v1_nbrs = mesh['faces'][f][mesh['faces'][f] != v2]
        mesh['faces'][f, v2_idx] = v1
        mesh['vf_adjacency_matrix'][v1, f] = True
        mesh['v_adjacency_matrix'][v1, new_v1_nbrs] = True
  else:
    mesh['faces'][mesh['faces'] == v2] = v1
    idxs = np.where(np.sum(mesh['faces'] == v1, axis=1) > 1)[0]
    mesh['faces'][idxs, :] = -1

  if SELF_CHECKING:         # Check that all adjacent matrices coherent to faces list
    for f_idx, f in enumerate(mesh['faces']):
      if f[0] == -1:
        continue
      for v1_, v2_ in [(f[0], f[1]), (f[0], f[2]), (f[1], f[2])]:
        if mesh['v_adjacency_matrix'][v1_, v2_] == False:
          raise Exception('Bad v_adjacency_matrix')
      for v_ in f:
        if mesh['vf_adjacency_matrix'][v_, f_idx] == False:
          raise Exception('Bad vf_adjacency_matrix')

  # remove all v1, v2 pairs from heap (forbidden_vertices can be than removed)
  for pair in mesh['pair_heap'][:]:
    if pair[1] in [v1, v2] or pair[2] in [v1, v2]:
      mesh['pair_heap'].remove(pair)

  # Check if a face have 2 same vertex indecis
  if SELF_CHECKING:
    idxs = np.where(mesh['faces'][:, 0] != -1)[0]
    to_check = mesh['faces'][idxs]
    if np.any(np.diff(np.sort(to_check, axis=1), axis=1) == 0):
      raise Exception('Bug: face found with 2 idintical vertex indices!')

  # Update Q of vertex v1
  #update_planes_parameters_near_vertex()         --> TODO
  #calc_Q_for_vertex(mesh, v1)

  # add new pairs of the new vertex
  v2 = None
  for v2_ in range(mesh['n_vertices']):
    if v1 == v2:
      continue
    edge_connection = mesh['v_adjacency_matrix'][v1, v2_]
    vertices_are_very_close = ENABLE_NON_EDGE_CONTRACTION and np.linalg.norm(mesh['vertices'][v2] - mesh['vertices'][v1]) < CLOSE_DIST_TH
    #if edge_connection or vertices_are_very_close:     #  -->> BUG!
    #  add_pair(mesh, v1, v2_, edge_connection)

def clean_mesh_from_removed_items(mesh):
  # Clean up the mesh from faces
  # Note that unused vertices are still there!
  # To be fixed later (not so important).
  faces2delete = np.where(np.all(mesh['faces'] == -1, axis=1))[0]
  mesh['faces'] = np.delete(mesh['faces'], faces2delete, 0)

def mesh_preprocess(mesh):
  # Unite all "same" vertices - ones that are very close
  for v_idx, v in enumerate(mesh['vertices']):
    d = np.linalg.norm(mesh['vertices'] - v, axis=1)
    idxs0 = np.where(d < SAME_V_TH_FOR_PREPROCESS)[0][1:]
    for v_idx_to_update in idxs0:
      mesh['vertices'][v_idx_to_update] = [np.nan, np.nan, np.nan]
      idxs1 = np.where(mesh['faces'] == v_idx_to_update)
      mesh['faces'][idxs1] = v_idx

def simplify_mesh(mesh_orig, n_vertices_to_merge):
  mesh = copy.deepcopy(mesh_orig)
  mesh_preprocess(mesh)

  mesh['pair_heap'] = []

  # Calc Q matrix for eack vertex
  calc_Q_for_each_vertex(mesh)

  # Select pairs and add them to a heap
  select_vertex_pairs(mesh)

  # Take and contract pairs
  for _ in range(int(n_vertices_to_merge)):
    contract_best_pair(mesh)

  # Remove old unused faces
  clean_mesh_from_removed_items(mesh)

  return mesh

def get_mesh(idx=0):
  global CLOSE_DIST_TH

  if idx == -1:
    mesh = io_off_model.get_simple_mesh('for_mesh_simplification_1')
    mesh['name'] = 'simple_2d_mesh_1'
    n_vertices_to_merge = 1
  elif idx == -2:
    mesh = io_off_model.get_simple_mesh('for_mesh_simplification_2')
    mesh['name'] = 'simple_2d_mesh_2'
    n_vertices_to_merge = 1
    CLOSE_DIST_TH = 0.5
  else:
    mesh_fns = [['meshes/bottle_0320.off',    30],    # 50
                ['meshes/person_0067.off',    600],
                ['meshes/airplane_0359.off',  1000],
                ['meshes/person_0004.off',    1000],
                ['meshes/bunny2.off',         4000],
                ]
    n_vertices_to_merge = mesh_fns[idx][1]
    mesh = io_off_model.read_off(mesh_fns[idx][0], verbose=True)
    mesh['name'] = os.path.split(mesh_fns[idx][0])[-1]
  return mesh, n_vertices_to_merge

def run_one(mesh_id=0):
  mesh, n_vertices_to_merge = get_mesh(mesh_id)
  mesh_simplified = simplify_mesh(mesh, n_vertices_to_merge)
  if not os.path.isdir('output_meshes'):
    os.makedirs('output_meshes')
  fn = 'output_meshes/' + mesh['name'].split('.')[0] + '_simplified.off'
  io_off_model.write_off_mesh(fn, mesh_simplified)
  fn = 'output_meshes/' + mesh['name'].split('.')[0] + '.obj'
  io_off_model.write_mesh(fn, mesh)

def run_all():
  for mesh_id in [-2, -1, 0, 1, 2, 3, 4]:
    run_one(mesh_id)

if __name__ == '__main__':
  #run_all()
  run_one(-1)