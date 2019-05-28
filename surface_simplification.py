import os
import copy
import heapq

import numpy as np

import io_off_model
import mesh_calc

OPTIMAL_POS = False
CLOSE_DIST_TH = 0.01

def calc_Q_for_each_vertex(mesh):
  mesh_calc.calc_fv_adjacency_matrix(mesh)
  mesh_calc.calc_face_plane_parameters(mesh)
  mesh['Qs'] = []
  for v_idx in range(mesh['n_vertices']):
    Q = np.zeros((4, 4))
    for f_idx in np.where(mesh['fv_adjacency_matrix'][v_idx])[0]:   # v_adjacency_list -- >>
      plane_params = mesh['face_plane_parameters'][f_idx][:, None]
      Kp = plane_params * plane_params.T                      # -->>
      Q += Kp
    mesh['Qs'].append(Q)

def add_pair(mesh, v1, v2, edge_connection):
  Q = np.sum((mesh['Qs'][v1], mesh['Qs'][v2]))
  new_v1_ = calc_new_vertex_position(mesh, v1, v2)
  new_v1 = np.vstack((new_v1_[:, None], np.ones((1, 1))))
  cost = np.dot(np.dot(new_v1.T, Q), new_v1)[0, 0]
  heapq.heappush(mesh['pair_heap'], (cost, v1, v2, edge_connection, new_v1_))

def select_vertex_pairs(mesh):
  mesh_calc.calc_v_adjacency_matrix(mesh)
  for v1 in range(mesh['n_vertices']):
    for v2 in range(v1 + 1, mesh['n_vertices']):
      edge_connection = mesh['v_adjacency_matrix'][v1, v2]
      if np.linalg.norm(mesh['vertices'][v2] - mesh['vertices'][v1]) < CLOSE_DIST_TH or edge_connection:
        add_pair(mesh, v1, v2, edge_connection)

def calc_new_vertex_position(mesh, v1, v2):
  if OPTIMAL_POS:
    raise Exception('TODO')
  else:
    new_v1 = (mesh['vertices'][v1] + mesh['vertices'][v2]) / 2

  return new_v1

def contract_best_pair(mesh):
  # get pair from heap
  cost, v1, v2, is_edge, new_v1 = heapq.heappop(mesh['pair_heap'])

  # update v1 - position
  mesh['vertices'][v1] = new_v1

  # remove v2:
  mesh['vertices'][v2] = [0, 0, 0] # "remove" vertex from mesh
  if 0:#is_edge:
    all_v2_faces = np.where(mesh['fv_adjacency_matrix'][v2])[0]
    for f in all_v2_faces:
      if v1 in mesh['faces'][f]:
        mesh['faces'][f] = [-1, -1, -1] # "remove" face from mesh
      else:
        v2_idx = np.where(mesh['faces'][f] == v2)[0]
        mesh['faces'][f, v2_idx] = v1
  else:
    mesh['faces'][mesh['faces'] == v2] = v1
    idxs = np.where(np.sum(mesh['faces'] == v1, axis=1) > 1)[0]
    mesh['faces'][idxs, :] = -1
  #mesh['faces']  # - update relevant faces
  #mesh['egdes'] - connect its edges to v1 ?
  # remove all v1, v2 pairs from heap

  # add new pairs of the new vertex or update the cost of all pairs of v1 (?)

def clean_mesh_from_removed_items(mesh):
  faces2delete = np.where(np.all(mesh['faces'] == -1, axis=1))[0]
  mesh['faces'] = np.delete(mesh['faces'], faces2delete, 0)

def simplify_mesh(mesh_orig, simplification_ratio):
  mesh = copy.deepcopy(mesh_orig)
  mesh['pair_heap'] = []
  desired_number_of_vertices = mesh_orig['n_vertices'] * simplification_ratio

  # Calc Q matrix for eack vertex
  calc_Q_for_each_vertex(mesh)

  # Select pairs and add them to a heap
  select_vertex_pairs(mesh)

  #while mesh['n_vertices'] > desired_number_of_vertices:
  for _ in range(1):
    contract_best_pair(mesh)

  clean_mesh_from_removed_items(mesh)

  return mesh

def get_mesh(idx=0):
  mesh_fns = [r"C:\Users\alon\Downloads\ModelNet40\ModelNet40\car\train\car_0016.off",
              r"C:\Users\alon\Downloads\ModelNet40\ModelNet40\bottle\train\bottle_0320.off",
              r"C:\Users\alon\Downloads\ModelNet40\ModelNet40\airplane\train\airplane_0169.off",
              r"C:\Users\alon\Downloads\ModelNet40\ModelNet40\cone\train\cone_0088.off",
              r"C:\Users\alon\Downloads\ModelNet40\ModelNet40\person\train\person_0034.off",
              r"C:\Users\alon\Downloads\ModelNet40\ModelNet40\cup\train\cup_0019.off",
              r"C:\Users\alon\Downloads\ModelNet40\ModelNet40\bottle\train\bottle_0190.off",
              'hw2_data/sphere_s0.off',
              'hw2_data/phands.off'
                ]
  mesh = io_off_model.read_off(mesh_fns[idx], verbose=True)
  mesh['name'] = os.path.split(mesh_fns[idx])[-1]
  return mesh

if __name__ == '__main__':
  mesh = get_mesh(6)
  mesh_simplified = simplify_mesh(mesh, 0.5)
  fn = mesh['name'].split('.')[0] + '_simplified.obj'
  io_off_model.write_mesh(fn, mesh_simplified)