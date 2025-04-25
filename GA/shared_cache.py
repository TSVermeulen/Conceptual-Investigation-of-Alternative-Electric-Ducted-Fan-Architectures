"""
shared_cache
============

"""

import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import pickle

class SharedCache:
    def __init__(self):
        # Create shared dictionary for metadata
        self.manager = mp.Manager()
        self.cache_keys = self.manager.dict()
        self.cache_info = self.manager.dict()
        self.lock = self.manager.Lock()
        
    def put(self, key, value):
        with self.lock:
            # Serialize metadata
            metadata = {}
            shared_arrays = {}
            
            # Process each array in the result
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    # Create shared memory for numpy array
                    shm = shared_memory.SharedMemory(create=True, size=v.nbytes)
                    # Create a numpy array backed by shared memory
                    shared_array = np.ndarray(v.shape, dtype=v.dtype, buffer=shm.buf)
                    # Copy the data
                    shared_array[:] = v[:]
                    # Store metadata about the array
                    metadata[k] = {'type': 'ndarray', 'shape': v.shape, 'dtype': str(v.dtype), 
                                 'shm_name': shm.name}
                    shared_arrays[k] = shm
                else:
                    # Store regular values directly
                    metadata[k] = {'type': 'value', 'value': v}
            
            # Store key, metadata and shared memory references
            pickled_key = pickle.dumps(key)
            self.cache_keys[pickled_key] = True
            self.cache_info[pickled_key] = (metadata, shared_arrays)
            
    def get(self, key):
        pickled_key = pickle.dumps(key)
        if pickled_key not in self.cache_keys:
            return None
            
        metadata, shared_arrays = self.cache_info[pickled_key]
        result = {}
        
        for k, meta in metadata.items():
            if meta['type'] == 'ndarray':
                # Get shape and dtype information
                shape = meta['shape']
                dtype = np.dtype(meta['dtype'])
                # Attach to existing shared memory
                shm = shared_memory.SharedMemory(name=meta['shm_name'])
                # Create array using the shared memory buffer
                array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                # Make a copy for the result
                result[k] = array.copy()
            else:
                # Return regular values
                result[k] = meta['value']
                
        return result