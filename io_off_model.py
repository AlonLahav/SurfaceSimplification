import numpy as np
import trimesh

def write_mesh(fn, mesh):
  tr_mesh = trimesh.Trimesh(vertices=mesh['vertices'], faces=mesh['faces'])
  tr_mesh.export(fn)

def write_off_mesh(fn, mesh):
  write_off(fn, mesh['vertices'], mesh['faces'])

def write_off(fn, points, polygons, colors=None):
  with open(fn, 'wt') as f:
    f.write('OFF\n')
    f.write(str(len(points)) + ' ' + str(len(polygons)) + ' 0\n')
    for p in points:
      if np.isnan(p[0]):
        p_ = [0, 0, 0]
      else:
        p_ = p
      f.write(str(p_[0]) + ' ' + str(p_[1]) + ' ' + str(p_[2]) + ' \n')
    polygons_ = []
    if colors is None:
      for p in polygons:
        polygons_.append([len(p)] + p)
    else:
      for p, c in zip(polygons, colors):
        polygons_.append([len(p)] + p + c)
    for p in polygons_:
      for v in p:
        f.write(str(v) + ' ')
      f.write('\n')

def read_off(fn, max_point=np.inf, verbose=False):
  if verbose:
    print('Reading', fn)

  def _read_line_ignore_comments(fp):
    while 1:
      l = fp.readline()
      if l.startswith('#') or l.strip() == '':
        continue
      return l

  if not fn.endswith('off'):
    return None, None
  points = []
  polygons = []
  with open(fn) as file:
    l = _read_line_ignore_comments(file)
    assert (l.strip() == 'OFF')
    n_points, n_polygons, n_edges = [int(s) for s in file.readline().split()]
    if n_points > max_point:
      return None, None

    for i in range(n_points):
      point = [float(s) for s in _read_line_ignore_comments(file).split()]
      points.append(point)

    for i in range(n_polygons):
      polygon = [int(s) for s in _read_line_ignore_comments(file).split()][1:]
      polygons.append(polygon)

  points = np.array(points).astype('float32')
  polygons = np.array(polygons).astype('int')

  mesh = {'vertices': points, 'faces': polygons, 'n_vertices': points.shape[0], 'n_faces': polygons.shape[0]}

  if verbose:
    print('Number of vertices: ', mesh['n_vertices'])
    print('Number of faces: ', mesh['n_faces'])

  return mesh

def get_simple_mesh(type):
  if type == 'one_triangle':
    points = np.array(([0, 0, 0], [1, 0, 0], [0, 1, .1]))
    polygons = np.array(([0, 1, 2],))
    mesh = {'vertices': points, 'faces': polygons, 'n_vertices': points.shape[0], 'n_faces': polygons.shape[0]}
  else:
    raise Exception('Unsupported mesh type')

  return mesh

if __name__ == '__main__':
  mesh = read_off('bunny.off')