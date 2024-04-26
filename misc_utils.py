from concurrent import futures
from tqdm import tqdm

def parallel_map(f, iterable, max_threads=None, show_pbar=False, desc="", **kwargs):
  """Parallel version of map()."""
  with futures.ThreadPoolExecutor(max_threads) as executor:
    if show_pbar:
      results = tqdm(
          executor.map(f, iterable, **kwargs), total=len(iterable), desc=desc)
    else:
      results = executor.map(f, iterable, **kwargs)
    return list(results)


def get_llffhold(config_f):
    llffhold = None
    with open(config_f, 'r') as f:
        for line in f:
            if line.startswith('llffhold'):
                llffhold = int(line.split('=')[1].strip())
                break
    return llffhold