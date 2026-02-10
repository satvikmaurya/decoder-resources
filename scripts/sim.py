import os
import re
import numpy as np
from log_parser import LogParser
from collections import defaultdict, deque
from typing import Tuple, Dict, List
from tqdm import tqdm
import pickle
import ray
from dataclasses import dataclass
import heapq

import scipy.stats as stats

def calculate_lognormal_params(mean, percentile_value, percentile_prob):
    """
    Calculate parameters for a log-normal distribution based on a target mean and percentile.
    
    Parameters:
    -----------
    mean : float
        Target mean of the distribution
    percentile_value : float
        Target value at the specified percentile
    percentile_prob : float
        Probability level for the percentile (between 0 and 1)
        
    Returns:
    --------
    mu : float
        Location parameter (mean of the underlying normal distribution)
    sigma : float
        Scale parameter (std dev of the underlying normal distribution)
    """
    from scipy.optimize import fsolve
    
    z_score = stats.norm.ppf(percentile_prob)
    
    def equations(sigma):
        mu = np.log(mean) - (sigma**2)/2
        percentile_eq = np.exp(mu + sigma * z_score) - percentile_value
        return percentile_eq
    
    sigma = fsolve(equations, 0.5)[0]
    
    mu = np.log(mean) - (sigma**2)/2
    
    return mu, sigma

def calculate_loglog_normal_params(mean, percentile_value, percentile_prob):
    """
    Calculate parameters for a log-log-normal distribution based on target mean and percentile
    using a grid search approach.
    """
    mu_values = np.linspace(-3, 10, 300)
    sigma_values = np.linspace(0.01, 5, 300)
    
    best_error = float('inf')
    best_params = None
    
    for mu in mu_values:
        for sigma in sigma_values:
            try:
                normal_samples = np.random.normal(mu, sigma, 50000)
                loglog_normal_samples = np.exp(np.exp(normal_samples))
                
                if np.any(np.isinf(loglog_normal_samples)) or np.any(np.isnan(loglog_normal_samples)):
                    continue
                
                sample_mean = np.mean(loglog_normal_samples)
                sample_percentile = np.percentile(loglog_normal_samples, percentile_prob * 100)
                
                mean_error = abs((sample_mean - mean) / mean)
                percentile_error = abs((sample_percentile - percentile_value) / percentile_value)
                total_error = mean_error + percentile_error
                
                if total_error < best_error:
                    best_error = total_error
                    best_params = (mu, sigma)
            except:  # noqa: E722
                continue
    
    if best_params is None:
        raise ValueError("Could not find suitable parameters")
    
    return best_params

@dataclass
class DecoderTask:
    priority: int = 0
    num_decoders_temporal: int = 0
    num_decoders_spatial: int = 0
    slice: int = 0
    qubit: Tuple[int, int] = None
    is_spatial: bool = False
    dependent_qubits: List[Tuple[int, int]] = None
    
    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority > other.priority
        # Temporal tasks have higher priority than spatial tasks
        if not self.is_spatial and other.is_spatial:
            return False
        if self.is_spatial and not other.is_spatial:
            return True
        return self.slice < other.slice

class Sim:
    def __init__(self, benchmark:str, directory:str, **kwargs) -> None:
        self.benchmark = benchmark
        self.directory = directory
        self.lli = f'{benchmark}_edpc.lli'
        self.log = f'{benchmark}_edpc.log'
        self.err = f'{benchmark}_edpc.err'
        self.parser = LogParser()
        self.parser.parse_layout_from_err(os.path.join(self.directory, self.err))
        self.parser.parse_log_file(os.path.join(self.directory, self.log))
        self.mapping = self.parser.create_qubit_mapping_for_lli(os.path.join(self.directory, self.lli))
        self.unique_qubits = set(self.mapping.values())
        self.magic_state_idxs = [list(self.unique_qubits).index(coord) 
                                for coord in set(self.parser.magic_state_locations.values())]
        self.core_qubits_idxs = [list(self.unique_qubits).index(coord) 
                                for coord in set(self.parser.core_locations.values())]
        self.num_benchmark_qubits = len(self.unique_qubits)
        self.alg_qubits = [coord for coord in set(self.parser.core_locations.values())] + [coord for coord in set(self.parser.magic_state_locations.values())]
        self.window_size = 3
        self.mfd_tracker = None
        self.fixed_decoder_latency = kwargs.get('fixed_decoder_latency', True)
        np.random.seed(42)
        self.idx = 0
        # self.sample_latencies(10000)
        with open('decoder_latencies.pkl', 'rb') as file:
            self.decoder_latencies = pickle.load(file)
        pass
    
    def sample_latencies(self, num:int) -> None:
        td = 0.5
        tgen = 1000 # ns
        worst_case_decoder_latency = 5000 # ns
        cycles = num
        mu, sigma = calculate_lognormal_params(td * tgen, worst_case_decoder_latency, 0.99)
        # Create the distribution
        lognorm_dist = stats.lognorm(s=sigma, scale=np.exp(mu))
        dlat = np.mean(np.array([lognorm_dist.rvs(size=cycles) for _ in range(100)]), axis=0)
        self.decoder_latencies = dlat
        return
    
    def decoder_latency(self) -> float:
        if self.fixed_decoder_latency:
            return 0.5
        self.idx = (self.idx + 1) % len(self.decoder_latencies)
        lat = self.decoder_latencies[self.idx] / 1000
        return lat

    def get_operands_for_mbm(self, operation:str) -> Tuple[Tuple[int, int], str, Tuple[int, int], str]:
        """Extracts all MultiBodyMeasure operands from a MultiBodyMeasure operation"""
        pattern = re.compile(r'MultiBodyMeasure\s+(\d+):([XYZ]),(\d+):([XYZ])')
        match = pattern.match(operation)
        if match:
            q1 = int(match.group(1))
            b1 = str(match.group(2))
            q2 = int(match.group(3))
            b2 = str(match.group(4))
            # q1, q2 are from the "virtual" qubit set, need to map them to the physical patches
            return tuple((self.mapping[q1], b1, self.mapping[q2], b2))
        exit(-1, f"Failed to parse MultiBodyMeasure operation: {operation} in benchmark {self.benchmark}")
        
    def get_operands_for_request(self, operation:str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Extracts the operands from a RequestMagicState or RequestYState operation"""
        pattern = re.compile(r'Request(Magic|Y)State (\d+) (\d+).*')
        match = pattern.match(operation)
        if match:
            q1 = int(match.group(2))
            q2 = int(match.group(3))
            if q1 not in self.mapping or q2 not in self.mapping:
                return (None, None)
            return (self.mapping[q1], self.mapping[q2])
        exit(-1, f"Failed to parse RequestMagicState or RequestYState operation: {operation} in benchmark {self.benchmark}")
        
    def get_operands_for_measure(self, operation:str) -> Tuple[Tuple[int, int], str]:
        """Extracts the operands from a MeasureSinglePatch operation"""
        pattern = re.compile(r'MeasureSinglePatch (\d+) ([XYZ])')
        match = pattern.match(operation)
        if match:
            q1 = int(match.group(1))
            b1 = str(match.group(2))
            return (self.mapping[q1], b1)
        exit(-1, f"Failed to parse MeasureSinglePatch operation: {operation} in benchmark {self.benchmark}")
    
    def build_distance_histogram(self) -> Tuple[Tuple[Tuple[int, int], str, Tuple[int, int], str], int]:
        """Parses the .lli file to build the distance histogram by taking routing conflicts into account"""
        with open(os.path.join(self.directory, self.log), 'r') as file:
            content = file.read()
        histogram = defaultdict(int)
        slice_path_lengths = defaultdict(list)
        all_distances = []
        lines = content.splitlines()
        slice_pattern = re.compile(r"slice:\s*(\d+)")
        path_length_pattern = re.compile(r"Path length:\s*(\d+)")
        multibody_pattern =  re.compile(r'MultiBodyMeasure:\s+Patch\s+(\d+):([XYZ])\s+at\s+\(\d+,\d+\)\s+\+\s+Patch\s+(\d+):([XYZ])\s+at\s+\(\d+,\d+\)')
        current_slice = 0
        for line in tqdm(lines, desc=f"Processing {self.benchmark} LLI", unit="lines"):
            slice_match = slice_pattern.match(line)
            if slice_match:
                current_slice = int(slice_match.group(1))
                continue
                # Record last seen MultiBodyMeasure line
            if multibody_pattern.match(line):
                previous_multibody_line = line
                continue
            path_match = path_length_pattern.match(line)
            if path_match and current_slice is not None:
                multibody_match = multibody_pattern.match(previous_multibody_line)
                path_length = int(path_match.group(1))
                histogram[path_length] += 1
                q1 = int(multibody_match.group(1))
                b1 = str(multibody_match.group(2))
                q2 = int(multibody_match.group(3))
                b2 = str(multibody_match.group(4))
                op = (self.mapping[q1], b1, self.mapping[q2], b2)
                slice_path_lengths[current_slice].append((op, path_length))
        all_distances = list(slice_path_lengths.values())
        with open(os.path.join(self.directory, f'{self.benchmark}_distance_trace.pkl'), 'wb') as file:
            pickle.dump(all_distances, file)
        return histogram
    
    def _hash(self, op: Tuple[Tuple[int, int], str, Tuple[int, int], str]) -> str:
        """Generates a hash for the operand"""
        return f"{op[0]}:{op[1]}-{op[2]}:{op[3]}"
    
    def _create_distance_map(self) -> None:
        self.dist_map = {}
        with open(os.path.join(self.directory, f'{self.benchmark}_distance_trace.pkl'), 'rb') as file:
            trace = pickle.load(file)
        for sublist in trace:
            for entry in sublist:
                op, path_length = entry
                self.dist_map[self._hash(op)] = path_length
        return
    
    def critical_decode_histogram(self) -> Dict[int, int]:
        """Creates a histogram of the number of critical decodes per slice"""
        with open(os.path.join(self.directory, self.lli), 'r') as file:
            content = file.read()
        histogram = defaultdict(int)
        tstates = deque()
        for line in tqdm(content.splitlines()):
            operations = line.split(';')
            tops = 0
            for op in operations:
                if 'RequestMagicState' in op:
                    operands = self.get_operands_for_request(op)
                    if operands[0] is None and operands[1] is None:
                        # This means we are at the point where the compiler timed out
                        continue
                    assert operands[1] in self.unique_qubits, f"Qubit {operands[1]} not found in unique qubits for benchmark {self.benchmark}"
                    index = list(self.unique_qubits).index(operands[1])
                    tstates.append(index)
                if 'MultiBodyMeasure' in op:
                    # Extract the operands and their qubit indices
                    operands = self.get_operands_for_mbm(op)
                    index = list(self.unique_qubits).index(operands[0])
                    if index in tstates:
                        tstates.remove(index)
                        tops += 1
            histogram[tops] += 1
            pass
        with open(os.path.join(self.directory, f'{self.benchmark}_critical_decode_histogram.pkl'), 'wb') as file:
            pickle.dump(histogram, file)
        return histogram
    
    def test(self):
        self._create_distance_map()
        with open(os.path.join(self.directory, self.lli), 'r') as file:
            content = file.read()
        for slice, line in enumerate(content.splitlines()):
            if not line.strip():
                continue
            operations = line.split(';')
            for op in operations:
                if 'MultiBodyMeasure' in op:
                    operands = self.get_operands_for_mbm(op)
                    distance = self.dist_map.get(self._hash(operands), None)
                    assert distance is not None, f"Distance not found for operands {operands} in slice {slice} in {self.benchmark}"
                    operand = ((operands[2], operands[0]))
                    assert operand[0] in self.unique_qubits, f"Operand0 {operand[0]} not found in unique qubits for benchmark {self.benchmark}"
                    assert operand[1] in self.unique_qubits, f"Operand1 {operand[1]} not found in unique qubits for benchmark {self.benchmark}"
        return
    
    def _mfd_tracker_init(self) -> None:
        # Scan through the LLI file to find most frequently decoded qubits after every slice
        non_clifford_operands = []
        with open(os.path.join(self.directory, self.lli), 'r') as file:
            content = file.read()
        total_slices = len(content.splitlines())
        mfd_tracker = np.zeros((len(self.alg_qubits), total_slices), dtype=np.int8)
        for slice, line in enumerate(content.splitlines()):
            if not line.strip():
                continue
            operations = line.split(';')
            for op in operations:
                if 'RequestMagicState' in op:
                    operands = self.get_operands_for_request(op)
                    if operands[0] is None and operands[1] is None:
                        # This means we are at the point where the compiler timed out
                        continue
                    assert operands[1] in self.unique_qubits, f"Qubit {operands[1]} not found in unique qubits for benchmark {self.benchmark}"
                    non_clifford_operands.insert(0, operands)
                if 'MultiBodyMeasure' in op:
                    operands = self.get_operands_for_mbm(op)
                    # core qubit, magic state storage
                    operand = ((operands[2], operands[0]))
                    if operand in non_clifford_operands:
                        non_clifford_operands.remove(operand)
                        mfd_tracker[self.alg_qubits.index(operand[0]), slice] += 1
                        mfd_tracker[self.alg_qubits.index(operand[1]), slice] += 1
                        pass
                    pass
                pass
            pass
        self.mfd_tracker = mfd_tracker
        return
    
    def schedule(self, num_decoders:int=100, policy:str='MLS', swd:bool=False) -> Dict[str, any]:
        # Iterate through the LLI file and schedule decoding
        self._create_distance_map()
        if policy == 'MFD' and self.mfd_tracker is None:
            self._mfd_tracker_init()
        with open(os.path.join(self.directory, self.lli), 'r') as file:
            content = file.read()
        
        # total_slices = len(content.splitlines())
        # self.sample_latencies(total_slices)
        
        non_clifford_operands = []
        deferred_tasks = []
        decoder_usage = []
        qubit_decode_history = {q:[] for q in self.unique_qubits}
        qubit_temporal_backlog = {q:0 for q in self.unique_qubits}
        qubit_backlog_history = []
        total_backlog = 0
        
        rr_queue = deque(self.alg_qubits)
        mfd_tracker = self.mfd_tracker
            
        for slice, line in enumerate(content.splitlines()):
            if not line.strip():
                continue
            
            operations = line.split(';')
            decoder_pool = list(range(num_decoders))
            
            for op in operations:
                if 'RequestMagicState' in op:
                    operands = self.get_operands_for_request(op)
                    if operands[0] is None and operands[1] is None:
                        # This means we are at the point where the compiler timed out
                        continue
                    assert operands[1] in self.unique_qubits, f"Qubit {operands[1]} not found in unique qubits for benchmark {self.benchmark}"
                    non_clifford_operands.insert(0, operands)
                    pass
            
            slice_queue = []
            
            for task in deferred_tasks:
                task.slice = slice
                heapq.heappush(slice_queue, task)
            deferred_tasks.clear()
            
            decoded_qubits = set()
            
            for op in operations:
                if 'MultiBodyMeasure' in op:
                    operands = self.get_operands_for_mbm(op)
                    distance = self.dist_map.get(self._hash(operands), None)
                    assert distance is not None, f"Distance not found for operands {operands} in slice {slice} in {self.benchmark}"
                    operand = ((operands[2], operands[0]))
                    priority = 1
                    if operand in non_clifford_operands:
                        # This MBM is a non-Clifford operation: critical decode: introduces spatial requirements
                        non_clifford_operands.remove(operand)
                        priority = 1000
                        pass
                    else:
                        # Some Clifford MBM (Y-state injection) -> introduces spatial requirements
                        priority = 5
                        pass
                    
                    if not swd:
                        spatial_decoders = max(1, distance // self.window_size)
                        spatial_task = DecoderTask(priority=priority,
                                                num_decoders_temporal=0,
                                                num_decoders_spatial=spatial_decoders,
                                                slice=slice,
                                                is_spatial=True,
                                                dependent_qubits=list(operand))
                        heapq.heappush(slice_queue, spatial_task)
                    for q in operand:
                        if qubit_temporal_backlog[q] > 0:
                            temporal_decoders = max(1, qubit_temporal_backlog[q] // self.window_size)
                            task = DecoderTask(priority=priority,
                                               num_decoders_temporal=temporal_decoders,
                                               num_decoders_spatial=0,
                                               slice=slice,
                                               qubit=q,
                                               is_spatial=False)
                            heapq.heappush(slice_queue, task)
                            pass
                        pass
                    pass
                pass
            while slice_queue:
                task = heapq.heappop(slice_queue)
                decoders_needed = int(task.num_decoders_spatial) + int(task.num_decoders_temporal)
                if decoders_needed <= len(decoder_pool):
                    decoder_pool = decoder_pool[decoders_needed:]
                    if task.is_spatial:
                        for q in task.dependent_qubits:
                            qubit_decode_history[q].append((slice, 'spatial', task.num_decoders_spatial))
                            qubit_temporal_backlog[q] -= 1 # a spatial decode also consumes the latest slice of syndromes
                            qubit_temporal_backlog[q] = max(0, qubit_temporal_backlog[q]) + 2 * self.decoder_latency()
                    else:
                        qubit_temporal_backlog[task.qubit] = 2 * self.decoder_latency() # account for decoder latency
                        decoded_qubits.add(task.qubit)
                        qubit_decode_history[task.qubit].append((slice, 'temporal', int(task.num_decoders_temporal)))
                elif not task.is_spatial and int(task.num_decoders_temporal) > 0:
                    # Consume as many undecoded slices as possible
                    decoders_needed = min(len(decoder_pool), int(task.num_decoders_temporal))
                    decoder_pool = decoder_pool[decoders_needed:]
                    qubit_temporal_backlog[task.qubit] = max(0, qubit_temporal_backlog[task.qubit] - decoders_needed) + 2 * self.decoder_latency()
                    decoded_qubits.add(task.qubit)
                else:
                    task.priority += 1
                    deferred_tasks.append(task)
                pass
            # Final decoding scheduling
            if len(decoder_pool) > 0:
                if policy == 'MLS':
                    # Find the qubit with the longest backlog that hasn't been decoded yet
                    sorted_qubits = sorted(qubit_temporal_backlog.items(), key=lambda x: x[1], reverse=True)
                    sorted_qubits = [q for q, _ in sorted_qubits if qubit_temporal_backlog[q] > 0 and q not in decoded_qubits]
                    for qubit in sorted_qubits:
                        if len(decoder_pool) == 0:
                            break
                        decoders_needed = min(len(decoder_pool), max(1, int(qubit_temporal_backlog[qubit] // self.window_size)))
                        decoder_pool = decoder_pool[decoders_needed:]
                        qubit_temporal_backlog[qubit] = max(0, qubit_temporal_backlog[qubit] - decoders_needed) + 2 * self.decoder_latency()
                        decoded_qubits.add(qubit)
                        qubit_decode_history[qubit].append((slice, 'temporal', decoders_needed))
                        pass
                    pass
                elif policy == 'RR':
                    for _ in self.alg_qubits:
                        qubit = rr_queue.popleft()
                        if len(decoder_pool) == 0:
                            rr_queue.appendleft(qubit)
                            break
                        if qubit in decoded_qubits:
                            rr_queue.append(qubit)
                            continue
                        decoders_needed = min(len(decoder_pool), max(1, int(qubit_temporal_backlog[qubit] // self.window_size)))
                        decoder_pool = decoder_pool[decoders_needed:]
                        qubit_temporal_backlog[qubit] = max(0, qubit_temporal_backlog[qubit] - decoders_needed) + 2 * self.decoder_latency()
                        decoded_qubits.add(qubit)
                        qubit_decode_history[qubit].append((slice, 'temporal', decoders_needed))
                        rr_queue.append(qubit)
                    pass
                elif policy == 'MFD':
                    timeline = mfd_tracker[:, slice:]
                    crit_decodes = np.sum(timeline, axis=1)
                    qubits = np.argsort(crit_decodes)[::-1]
                    qubits = [self.alg_qubits[q] for q in qubits if self.alg_qubits[q] not in decoded_qubits]
                    for qubit in qubits:
                        if len(decoder_pool) == 0:
                            break
                        decoders_needed = min(len(decoder_pool), max(1, int(qubit_temporal_backlog[qubit] // self.window_size)))
                        decoder_pool = decoder_pool[decoders_needed:]
                        qubit_temporal_backlog[qubit] = max(0, qubit_temporal_backlog[qubit] - decoders_needed) + 2 * self.decoder_latency()
                        decoded_qubits.add(qubit)
                        qubit_decode_history[qubit].append((slice, 'temporal', decoders_needed))
                        pass
                    pass
                pass
            for qubit in self.alg_qubits:
                if qubit not in decoded_qubits:
                    qubit_temporal_backlog[qubit] += 2 * self.decoder_latency() # Because each qubit not decoded cannot be decoded until the decoders are available again
            
            decoder_usage.append(num_decoders - len(decoder_pool))
            qubit_backlog_history += [qubit_temporal_backlog]
            total_backlog += max(qubit_temporal_backlog.values())
            pass
        
        # There could be some deferred decodes left
        extra_slices = 0
        while deferred_tasks:
            slice = len(content.splitlines()) + extra_slices
            if slice > 3 * len(content.splitlines()):
                break # Avoid infinite loops in case there are too few decoders
            
            decoder_pool = list(range(num_decoders))
            decoded_qubits = set()
            slice_queue = []
            for task in deferred_tasks:
                task.slice = slice
                heapq.heappush(slice_queue, task)
                pass
            
            deferred_tasks.clear()
            
            while slice_queue:
                task = heapq.heappop(slice_queue)
                decoders_needed = int(task.num_decoders_spatial) + int(task.num_decoders_temporal)
                if decoders_needed <= len(decoder_pool):
                    decoder_pool = decoder_pool[decoders_needed:]
                    if task.is_spatial:
                        for q in task.dependent_qubits:
                            qubit_decode_history[q].append((slice, 'spatial', int(task.num_decoders_spatial)))
                            qubit_temporal_backlog[q] -= 1 # a spatial decode also consumes the latest slice of syndromes
                            qubit_temporal_backlog[q] = max(0, qubit_temporal_backlog[q]) + 2 * self.decoder_latency()
                    else:
                        qubit_temporal_backlog[task.qubit] = 2 * self.decoder_latency() # account for decoder latency
                        decoded_qubits.add(task.qubit)
                        qubit_decode_history[task.qubit].append((slice, 'temporal', int(task.num_decoders_temporal)))
                elif not task.is_spatial and task.num_decoders_temporal > 0:
                    # Consume as many undecoded slices as possible
                    decoders_needed = min(len(decoder_pool), int(task.num_decoders_temporal))
                    decoder_pool = decoder_pool[decoders_needed:]
                    qubit_temporal_backlog[task.qubit] = max(0, qubit_temporal_backlog[task.qubit] - decoders_needed) + 2 * self.decoder_latency()
                    decoded_qubits.add(task.qubit)
                else:
                    task.priority += 1
                    deferred_tasks.append(task)
                pass
            # Final decoding scheduling
            if len(decoder_pool) > 0:
                if policy == 'MLS':
                    # Find the qubit with the longest backlog that hasn't been decoded yet
                    sorted_qubits = sorted(qubit_temporal_backlog.items(), key=lambda x: x[1], reverse=True)
                    sorted_qubits = [q for q, _ in sorted_qubits if qubit_temporal_backlog[q] > 0 and q not in decoded_qubits]
                    for qubit in sorted_qubits:
                        if len(decoder_pool) == 0:
                            break
                        decoders_needed = min(len(decoder_pool), max(1, int(qubit_temporal_backlog[qubit] // self.window_size)))
                        decoder_pool = decoder_pool[decoders_needed:]
                        qubit_temporal_backlog[qubit] = max(0, qubit_temporal_backlog[qubit] - decoders_needed) + 2 * self.decoder_latency()
                        decoded_qubits.add(qubit)
                        qubit_decode_history[qubit].append((slice, 'temporal', decoders_needed))
                        pass
                    pass
                elif policy == 'RR':
                    for _ in self.alg_qubits:
                        qubit = rr_queue.popleft()
                        if len(decoder_pool) == 0:
                            rr_queue.appendleft(qubit)
                            break
                        if qubit in decoded_qubits:
                            rr_queue.append(qubit)
                            continue
                        decoders_needed = min(len(decoder_pool), max(1, int(qubit_temporal_backlog[qubit] // self.window_size)))
                        decoder_pool = decoder_pool[decoders_needed:]
                        qubit_temporal_backlog[qubit] = max(0, qubit_temporal_backlog[qubit] - decoders_needed) + 2 * self.decoder_latency()
                        decoded_qubits.add(qubit)
                        qubit_decode_history[qubit].append((slice, 'temporal', decoders_needed))
                        rr_queue.append(qubit)
                    pass
                elif policy == 'MFD':
                    timeline = mfd_tracker # Sort according to the entire timeline now that all the slices have been exhausted
                    crit_decodes = np.sum(timeline, axis=1)
                    qubits = np.argsort(crit_decodes)[::-1]
                    qubits = [self.alg_qubits[q] for q in qubits if self.alg_qubits[q] not in decoded_qubits]
                    for qubit in qubits:
                        if len(decoder_pool) == 0:
                            break
                        decoders_needed = min(len(decoder_pool), max(1, int(qubit_temporal_backlog[qubit] // self.window_size)))
                        decoder_pool = decoder_pool[decoders_needed:]
                        qubit_temporal_backlog[qubit] = max(0, qubit_temporal_backlog[qubit] - decoders_needed) + 2 * self.decoder_latency()
                        decoded_qubits.add(qubit)
                        qubit_decode_history[qubit].append((slice, 'temporal', decoders_needed))
                        pass
                    pass
                pass
            for qubit in self.alg_qubits:
                if qubit not in decoded_qubits:
                    qubit_temporal_backlog[qubit] += 2 * self.decoder_latency()
            
            decoder_usage.append(num_decoders - len(decoder_pool))
            qubit_backlog_history += [qubit_temporal_backlog]
            total_backlog += max(qubit_temporal_backlog.values())
            extra_slices += 1
            pass
        results = {
            'decoder_usage': decoder_usage,
            'extra_slices': extra_slices,
            'qubit_decode_history': qubit_decode_history,
            'qubit_temporal_backlog': qubit_temporal_backlog,
            'qubit_backlog_history': qubit_backlog_history,
            'total_backlog': total_backlog,
            'longest_backlog': max(qubit_temporal_backlog.values())
        }
        return results
    
    def decoder_demand(self) -> Dict[int, int]:
        timeline = self.create_timeline()
        # Count the number of decodes required per slice
        demand = defaultdict(int)
        for col in range(timeline.shape[1]):
            col_data = timeline[:, col]
            # Count the number of positive entries in the column
            num_decodes = np.sum(np.abs(col_data) >= 0)
            demand[col] = num_decodes
        with open(os.path.join(self.directory, f'{self.benchmark}_decoder_demand.pkl'), 'wb') as file:
            pickle.dump(demand, file)
        return demand

    # Deprecated
    def create_timeline(self) -> np.ndarray:
        """Creates a timeline of the operations in the LLI file"""
        # Scheduling insight -- limited parallelism per slice, and number of long distance interactions are limited
        self._create_distance_map()
        with open(os.path.join(self.directory, self.lli), 'r') as file:
            content = file.read()
        # Create a timeline for the IR - every column is a slice, and every row is a qubit in the system
        # Prioritize T state measurements, then Y states if possible
        # Example: RequestMagicState 168 10;MultiBodyMeasure 10:Z,168:Z;MeasureSinglePatch 168 X
        timeline = []
        ystates = deque()
        tstates = deque()
        prev_line_empty = False
        for slice, line in enumerate(content.splitlines()):
            if line.strip() == "":
                if prev_line_empty:
                    break 
                prev_line_empty = True
            else:
                prev_line_empty = False
            operations = line.split(';')
            ystate_measures = []
            tstates_map = {'qubits':[], 'distances':{}} # T states in this slice
            other_ops = {'qubits':[], 'distances':{}} # Other operations in this slice
            for op in operations: # Qubit order convention hardcoded right now
                if 'RequestMagicState' in op:
                    operands = self.get_operands_for_request(op)
                    if operands[0] is None and operands[1] is None:
                        # This means we are at the point where the compiler timed out
                        continue
                    assert operands[1] in self.unique_qubits, f"Qubit {operands[1]} not found in unique qubits for benchmark {self.benchmark}"
                    index = list(self.unique_qubits).index(operands[1])
                    tstates.append(index)
                    pass
                if 'RequestYState' in op:
                    operands = self.get_operands_for_request(op)
                    if operands[0] is None and operands[1] is None:
                        # This means we are at the point where the compiler timed out
                        continue
                    assert operands[0] in self.unique_qubits, f"Qubit {operands[0]} not found in unique qubits for benchmark {self.benchmark}"
                    index = list(self.unique_qubits).index(operands[0])
                    ystates.append(index)
                if 'MultiBodyMeasure' in op:
                    # Extract the operands and their qubit indices
                    operands = self.get_operands_for_mbm(op)
                    distance = self.dist_map.get(self._hash(operands), None)
                    assert distance is not None, f"Distance not found for operands {operands} in slice {slice} in {self.benchmark}"
                    index = list(self.unique_qubits).index(operands[0])
                    if index in tstates: # Could be a Y state MBM
                        tstates_map['distances'][index] = distance + 1 # +1 since we consider the distance between neighboring patches to be 0 but that will still need one decoder in space
                        tstates_map['qubits'].append(index)
                        tstates.remove(index)
                    else:
                        other_ops['distances'][index] = distance + 1 # This could be an ancilla patch that was initialized
                        other_ops['qubits'].append(index)
                    # print(f"Slice {slice}, Operation {op}, Operands {operands}, Distance {distance}")
                if 'MeasureSinglePatch' in op:
                    operands = self.get_operands_for_measure(op)
                    assert operands[0] in self.unique_qubits, f"Qubit {operands[0]} not found in unique qubits for benchmark {self.benchmark}"
                    index = list(self.unique_qubits).index(operands[0])
                    if index in ystates:
                        ystate_measures.append(index)
                        ystates.remove(index)
                    pass
            timeline.append((tstates_map['qubits'], tstates_map['distances'], other_ops['qubits'], other_ops['distances'])) # which qubits in this slice were involved in a non-clifford gate, which were measured after a Ystate op, and what the distances were
            pass
        
        # Now construct a matrix repr of the timeline
        _timeline = np.zeros((len(self.unique_qubits), len(timeline)), dtype=np.int16)
        magic_state_set = set(self.magic_state_idxs)
        core_qubits_set = set(self.core_qubits_idxs)
        always_decode_indices = magic_state_set.union(core_qubits_set)
        for col, (qubits, distances, other_qubits, other_distances) in enumerate(timeline):
            qubits_set = set(qubits)
            other_qubits_set = set(other_qubits)
            for idx in always_decode_indices:
                if idx not in qubits_set and idx not in other_qubits_set:
                    _timeline[idx, col] = -1
            for row in qubits:
                _timeline[row, col] = distances[row]
            for row in other_qubits:
                _timeline[row, col] = -1 * other_distances[row]
                
        return _timeline
    
    # Deprecated
    def schedule_decoding_sw(self, num_decoders:int=100, policy:str='MLS') -> Dict[str, any]:
        """Schedule decoding for SWD based on the timeline"""
        timeline = self.create_timeline()
        decoder_latency = 0.15
        per_qubit_backlog = np.zeros((timeline.shape[0], timeline.shape[1] + 1), dtype=np.uint32)
        rr_queue = deque(range(timeline.shape[0]))
        num_crit_decodes = (timeline > 0).astype(np.int32)
        num_crit_decodes = np.flip(np.cumsum(np.flip(num_crit_decodes, axis=1), axis=1), axis=1)

        for col_idx in range(timeline.shape[1]):
            decoder_pool = list(range(num_decoders))
            decoded_qubits = set()
            col = timeline[:, col_idx]
            critical_decodes = np.where(col > 0)[0]
            for qubit in critical_decodes:
                decoders_needed = 1
                if decoders_needed > len(decoder_pool):
                    per_qubit_backlog[qubit][col_idx + 1] = per_qubit_backlog[qubit][col_idx] + 1 + col[qubit]
                else:
                    decoder_pool = decoder_pool[decoders_needed:]
                    per_qubit_backlog[qubit][col_idx + 1] = int(np.ceil((per_qubit_backlog[qubit][col_idx] + 1) * decoder_latency))
                    decoded_qubits.add(qubit)
            if len(decoder_pool) > 0:
                if policy == 'MLS':
                    sorted_qubits = np.argsort(per_qubit_backlog[:, col_idx])[::-1]
                    for qubit in sorted_qubits:
                        if qubit in decoded_qubits:
                            continue
                        if timeline[qubit, col_idx] == 0:
                            # This qubit is not involved in any operation in this slice, skip it
                            continue
                        decoders_needed = 1
                        if decoders_needed > len(decoder_pool):
                            per_qubit_backlog[qubit][col_idx + 1] = per_qubit_backlog[qubit][col_idx] + 1 + col[qubit]
                        else:
                            decoder_pool = decoder_pool[decoders_needed:]
                            per_qubit_backlog[qubit][col_idx + 1] = int(np.ceil((per_qubit_backlog[qubit][col_idx] + 1) * decoder_latency))
                            decoded_qubits.add(qubit)
                elif policy == 'RR':
                    for _ in range(len(rr_queue)):
                        qubit = rr_queue.popleft()
                        if qubit in decoded_qubits:
                            rr_queue.append(qubit)
                            continue
                        if timeline[qubit, col_idx] == 0:
                            # This qubit is not involved in any operation in this slice, skip it
                            rr_queue.append(qubit)
                            continue
                        decoders_needed = 1
                        if decoders_needed > len(decoder_pool):
                            per_qubit_backlog[qubit][col_idx + 1] = per_qubit_backlog[qubit][col_idx] + 1 + col[qubit]
                        else:
                            decoder_pool = decoder_pool[decoders_needed:]
                            per_qubit_backlog[qubit][col_idx + 1] = int(np.ceil((per_qubit_backlog[qubit][col_idx] + 1) * decoder_latency))
                            decoded_qubits.add(qubit)
                        rr_queue.append(qubit)
                elif policy == 'MFD':
                    # Sort qubits based on the number of non-clifford ops from this point on
                    if col_idx + 1 >= timeline.shape[1]:
                        num_decodes = np.zeros(timeline.shape[0], dtype=np.int32)
                    else:
                        num_decodes = num_crit_decodes[:, col_idx + 1]
                    sorted_qubits = np.argsort(num_decodes)[::-1]
                    for qubit in sorted_qubits:
                        if qubit in decoded_qubits:
                            continue
                        if timeline[qubit, col_idx] == 0:
                            # This qubit is not involved in any operation in this slice, skip it
                            continue
                        decoders_needed = 1
                        if decoders_needed > len(decoder_pool):
                            per_qubit_backlog[qubit][col_idx + 1] = per_qubit_backlog[qubit][col_idx] + 1 + col[qubit]
                        else:
                            decoder_pool = decoder_pool[decoders_needed:]
                            per_qubit_backlog[qubit][col_idx + 1] = int(np.ceil((per_qubit_backlog[qubit][col_idx] + 1) * decoder_latency))
                            decoded_qubits.add(qubit)
                    pass
            for qubit in range(timeline.shape[0]):
                if qubit not in decoded_qubits:
                    if timeline[qubit, col_idx] == 0:
                        # This qubit is not involved in any operation in this slice, skip it
                        continue
                    per_qubit_backlog[qubit][col_idx + 1] = per_qubit_backlog[qubit][col_idx] + 1
            pass
        metadata = {
            'backlog': per_qubit_backlog,
            'longest_backlog': np.max(per_qubit_backlog[:, -1].flatten()),
        }
        return metadata
    
    # Deprecated
    def schedule_decoding_pw(self, num_decoders:int=100, policy:str='MLS') -> Dict[str, any]:
        """Schedule decoding based on the timeline"""
        unutilized_decoders = {}
        spatial_decoders_required = lambda x: x // 3  # noqa: E731
        timeline = self.create_timeline()
        num_crit_decodes = (timeline > 0).astype(np.int32)
        num_crit_decodes = np.flip(np.cumsum(np.flip(num_crit_decodes, axis=1), axis=1), axis=1)
        per_qubit_schedule = {qubit:[] for qubit in range(timeline.shape[0])}
        per_qubit_backlog = np.zeros((timeline.shape[0], timeline.shape[1] + 1), dtype=np.uint32)
        # Map a decoder from the decoder pool to each qubit in a slice until the decoders run out. 
        # If a slice has a positive integer, it means that the those many patches need to be decoded that slice
        rr_queue = deque(range(timeline.shape[0]))
        for col_idx in range(timeline.shape[1]):
            decoder_pool = list(range(num_decoders))
            unutilized_decoders[col_idx] = 0
            decoded_qubits = set()
            col = timeline[:, col_idx]
            # Find critical decodes
            critical_decodes = np.where(col > 0)[0]
            for qubit in critical_decodes:
                # spatial requirement
                decoders_needed = max(1, spatial_decoders_required(col[qubit]))
                # temporal requirement
                decoders_needed += (per_qubit_backlog[qubit][col_idx] + 1) // 3 # Assuming one window is 3d^3
                if decoders_needed > len(decoder_pool):
                    # We don't have enough decoders for this qubit
                    # This incurs a program slowdown
                    if (per_qubit_backlog[qubit][col_idx] + 1) // 3 < len(decoder_pool):
                        # We can schedule some decodes, but not all
                        decoders_needed = (per_qubit_backlog[qubit][col_idx] + 1) // 3
                        per_qubit_schedule[qubit].append((col_idx, decoder_pool[:decoders_needed]))
                        decoder_pool = decoder_pool[decoders_needed:]
                        per_qubit_backlog[qubit][col_idx + 1] = 2
                        decoded_qubits.add(qubit)
                    else:
                        per_qubit_backlog[qubit][col_idx + 1] = per_qubit_backlog[qubit][col_idx] + 1 + col[qubit]
                # Property: Only the qubit annotated with the distance could have undecoded rounds, the routing qubits are only active for one slice
                else:
                    # schedule decoding for this qubit, remove decoders from the pool, reduce its undecoded rounds
                    per_qubit_schedule[qubit].append((col_idx, decoder_pool[:decoders_needed]))
                    decoder_pool = decoder_pool[decoders_needed:]
                    per_qubit_backlog[qubit][col_idx + 1] = 2
                    decoded_qubits.add(qubit)
            if len(decoder_pool) > 0:
                if policy == 'MLS':
                    # If there are decoders left, schedule decoding for other qubits
                    # Priority 1: longest undecoded qubits
                    # Sort qubits by their backlog in descending order
                    sorted_qubits = np.argsort(per_qubit_backlog[:, col_idx])[::-1]
                    sorted_qubits = [q for q in sorted_qubits if per_qubit_backlog[q][col_idx] > 0]
                    for qubit in sorted_qubits:
                        if len(decoder_pool) == 0:
                            break
                        if qubit in decoded_qubits:
                            continue
                        if timeline[qubit, col_idx] == 0:
                            # This qubit is not involved in any operation in this slice, skip it
                            continue
                        # temporal requirement
                        decoders_needed = max(1, per_qubit_backlog[qubit][col_idx] // 3)
                        # spatial requirement 
                        decoders_needed += spatial_decoders_required(-1 * col[qubit]) # Negative distance indicates a non-critical MBM
                        if decoders_needed <= len(decoder_pool):
                            per_qubit_schedule[qubit].append((col_idx, decoder_pool[:decoders_needed]))
                            decoder_pool = decoder_pool[decoders_needed:]
                            per_qubit_backlog[qubit][col_idx + 1] = 2
                            decoded_qubits.add(qubit)
                        else:
                            # Chip away at the backlog for this qubit
                            if max(1, per_qubit_backlog[qubit][col_idx] // 3) < len(decoder_pool):
                                # We can schedule some decodes, but not all
                                decoders_needed = max(1, per_qubit_backlog[qubit][col_idx] // 3)
                                per_qubit_schedule[qubit].append((col_idx, decoder_pool[:decoders_needed]))
                                decoder_pool = decoder_pool[decoders_needed:]
                                per_qubit_backlog[qubit][col_idx + 1] = 2
                                decoded_qubits.add(qubit)
                            elif spatial_decoders_required(-1 * col[qubit]) > 0:
                                # Spatial requirement is not met, so we cannot schedule this qubit
                                per_qubit_backlog[qubit][col_idx + 1] = per_qubit_backlog[qubit][col_idx] + 1 + -1 * col[qubit]
                            else:
                                decoders_needed = min(len(decoder_pool), 1 + per_qubit_backlog[qubit][col_idx] // 3)
                                per_qubit_schedule[qubit].append((col_idx, decoder_pool[:decoders_needed]))
                                decoder_pool = decoder_pool[decoders_needed:]
                                per_qubit_backlog[qubit][col_idx + 1] = per_qubit_backlog[qubit][col_idx] - decoders_needed + 1
                                decoded_qubits.add(qubit)
                elif policy == 'MFD':
                    # Most frequently (critically) decoded
                    if col_idx + 1 >= timeline.shape[1]:
                        num_decodes = np.zeros(timeline.shape[0], dtype=np.int32)
                    else:
                        num_decodes = num_crit_decodes[:, col_idx + 1]
                    sorted_qubits = np.argsort(num_decodes)[::-1]
                    for qubit in sorted_qubits:
                        if len(decoder_pool) == 0:
                            break
                        if qubit in decoded_qubits:
                            continue
                        if timeline[qubit, col_idx] == 0:
                            # This qubit is not involved in any operation in this slice, skip it
                            continue
                        # temporal requirement
                        decoders_needed = max(1, per_qubit_backlog[qubit][col_idx] // 3)
                        # spatial requirement 
                        decoders_needed += spatial_decoders_required(-1 * col[qubit]) # Negative distance indicates a non-critical MBM
                        if decoders_needed <= len(decoder_pool):
                            per_qubit_schedule[qubit].append((col_idx, decoder_pool[:decoders_needed]))
                            decoder_pool = decoder_pool[decoders_needed:]
                            per_qubit_backlog[qubit][col_idx + 1] = 2
                            decoded_qubits.add(qubit)
                        else:
                            # Chip away at the backlog for this qubit
                            if max(1, per_qubit_backlog[qubit][col_idx] // 3) < len(decoder_pool):
                                # We can schedule some decodes, but not all
                                decoders_needed = max(1, per_qubit_backlog[qubit][col_idx] // 3)
                                per_qubit_schedule[qubit].append((col_idx, decoder_pool[:decoders_needed]))
                                decoder_pool = decoder_pool[decoders_needed:]
                                per_qubit_backlog[qubit][col_idx + 1] = 2
                                decoded_qubits.add(qubit)
                            elif spatial_decoders_required(-1 * col[qubit]) > 0:
                                # Spatial requirement is not met, so we cannot schedule this qubit
                                per_qubit_backlog[qubit][col_idx + 1] = per_qubit_backlog[qubit][col_idx] + 1 + -1 * col[qubit]
                            else:
                                decoders_needed = min(len(decoder_pool), 1 + per_qubit_backlog[qubit][col_idx] // 3)
                                per_qubit_schedule[qubit].append((col_idx, decoder_pool[:decoders_needed]))
                                decoder_pool = decoder_pool[decoders_needed:]
                                per_qubit_backlog[qubit][col_idx + 1] = per_qubit_backlog[qubit][col_idx] - decoders_needed + 1
                                decoded_qubits.add(qubit)
                elif policy == 'RR':
                    # Round robin
                    # maintain decoding frames, qubits decoded in the previous slice are not considered
                    for _ in range(len(rr_queue)):
                        if len(decoder_pool) == 0:
                            break
                        qubit = rr_queue.popleft()
                        if qubit in decoded_qubits:
                            rr_queue.append(qubit)
                            continue
                        if timeline[qubit, col_idx] == 0:
                            # This qubit is not involved in any operation in this slice, skip it
                            rr_queue.append(qubit)
                            continue
                        # temporal requirement
                        decoders_needed = max(1, per_qubit_backlog[qubit][col_idx] // 3)
                        # spatial requirement 
                        decoders_needed += spatial_decoders_required(-1 * col[qubit]) # Negative distance indicates a non-critical MBM
                        if decoders_needed <= len(decoder_pool):
                            per_qubit_schedule[qubit].append((col_idx, decoder_pool[:decoders_needed]))
                            decoder_pool = decoder_pool[decoders_needed:]
                            per_qubit_backlog[qubit][col_idx + 1] = 2
                            decoded_qubits.add(qubit)
                        else:
                            # Chip away at the backlog for this qubit
                            if max(1, per_qubit_backlog[qubit][col_idx] // 3) < len(decoder_pool):
                                # We can schedule some decodes, but not all
                                decoders_needed = max(1, per_qubit_backlog[qubit][col_idx] // 3)
                                per_qubit_schedule[qubit].append((col_idx, decoder_pool[:decoders_needed]))
                                decoder_pool = decoder_pool[decoders_needed:]
                                per_qubit_backlog[qubit][col_idx + 1] = 2
                                decoded_qubits.add(qubit)
                            elif spatial_decoders_required(-1 * col[qubit]) > 0:
                                # Spatial requirement is not met, so we cannot schedule this qubit
                                per_qubit_backlog[qubit][col_idx + 1] = per_qubit_backlog[qubit][col_idx] + 1 + -1 * col[qubit]
                            else:
                                decoders_needed = min(len(decoder_pool), 1 + per_qubit_backlog[qubit][col_idx] // 3)
                                per_qubit_schedule[qubit].append((col_idx, decoder_pool[:decoders_needed]))
                                decoder_pool = decoder_pool[decoders_needed:]
                                per_qubit_backlog[qubit][col_idx + 1] = per_qubit_backlog[qubit][col_idx] - decoders_needed + 1
                                decoded_qubits.add(qubit)
                        rr_queue.append(qubit)  # Reinsert the qubit at the end of the queue
                if len(decoder_pool) > 0:
                    unutilized_decoders[col_idx] = len(decoder_pool)
                pass
            # All decoders have been scheduled for this slice, update backlog for the remaining qubits
            for qubit in range(timeline.shape[0]):
                if qubit not in decoded_qubits:
                    if timeline[qubit, col_idx] == 0:
                        # This qubit is not involved in any operation in this slice, skip it
                        per_qubit_backlog[qubit][col_idx + 1] = per_qubit_backlog[qubit][col_idx]
                        continue
                    per_qubit_backlog[qubit][col_idx + 1] = per_qubit_backlog[qubit][col_idx] + 1
            pass
        metadata = {
            'qubit_schedule': per_qubit_schedule,
            'backlog': per_qubit_backlog,
            'unutilized_decoders': unutilized_decoders,
            'longest_backlog': np.max(per_qubit_backlog[:, -1].flatten()),
        }
        return metadata
    
    def optimize(self, policy:str='MLS', swd:bool=False) -> int:
        """Optimize to find the smallest number of decoders that yields the smallest longest backlog"""
        if policy == 'MFD':
            self._mfd_tracker_init()
        min_decoders = 1
        max_decoders = self.num_benchmark_qubits
        best_n_decoders = max_decoders
        best_backlog = 2 if self.fixed_decoder_latency else 10 
        last_backlog = None
        while min_decoders <= max_decoders:
            mid = (min_decoders + max_decoders) // 2
            metadata = self.schedule(num_decoders=mid, policy=policy, swd=swd)
            backlog = metadata['longest_backlog']
            last_backlog = backlog
            if backlog <= best_backlog:
                best_n_decoders = mid
                max_decoders = mid - 1
                if self.fixed_decoder_latency:
                    best_backlog = backlog
            else:
                min_decoders = mid + 1
        metadata = self.schedule(num_decoders=best_n_decoders, policy=policy)
        last_backlog = metadata['longest_backlog']
        best_backlog = last_backlog
        print(f"Best configuration for {self.benchmark} with policy {policy}: {best_n_decoders} decoders with longest backlog of {last_backlog}, extra slices {metadata['extra_slices']}")
        suffix = '_swd' if swd else ''
        suffix_2 = '' if self.fixed_decoder_latency else '_variable_'
        with open(os.path.join(self.directory, f'{self.benchmark}_opt_decoder{suffix_2}{policy}{suffix}.pkl'), 'wb') as file:
            pickle.dump(best_n_decoders, file)
        return best_n_decoders
    
    def get_decoder_sweep_for_best_decoder(self, num_decoders:int, policy:str='MLS') -> Dict[int, int]:
        """Get the decoder sweep for the best number of decoders"""
        nprocs = os.cpu_count() // 2
        if policy == 'MFD':
            self._mfd_tracker_init()
        if self.benchmark == 'bbqram-12':
            return {num_decoders: 0}  # No need to sweep for this benchmark, it has a fixed number of decoders
        decoders_to_try = np.linspace(0, 1, num=nprocs)
        decoders_to_try = decoders_to_try ** 3
        decoders_to_try = 1 - decoders_to_try
        decoders_to_try = num_decoders // 2 + decoders_to_try * (num_decoders // 2 - 1)
        decoders_to_try = np.unique(np.round(decoders_to_try).astype(int))
        if num_decoders not in decoders_to_try:
            decoders_to_try[-1] = num_decoders
        
        @ray.remote
        def schedule_with_decoders(benchmark, directory, num_decoders, policy):
            sim = Sim(benchmark, directory)
            meta = sim.schedule(num_decoders=num_decoders, policy=policy)
            return num_decoders, meta['longest_backlog']

        futures = [schedule_with_decoders.remote(self.benchmark, self.directory, n_decoders, policy) 
                   for n_decoders in decoders_to_try]
        
        backlogs = {}
        for n_decoders, longest_backlog in ray.get(futures):
            backlogs[n_decoders] = longest_backlog
            
        with open(os.path.join(self.directory, f'{self.benchmark}_backlogs_sweep_{policy}.pkl'), 'wb') as file:
            pickle.dump(backlogs, file)
        return backlogs
    
    # Deprecated
    def get_decoder_sweep_for_best_decoder_sw(self, num_decoders:int, policy:str='MLS') -> Dict[int, int]:
        """[Deprecated] Get the decoder sweep for the best number of decoders"""
        nprocs = os.cpu_count() // 2
        if policy == 'MFD':
            self._mfd_tracker_init()
        if self.benchmark == 'bbqram-12':
            return {num_decoders: 0}  # No need to sweep for this benchmark, it has a fixed number of decoders
        decoders_to_try = np.linspace(0, 1, num=nprocs)
        decoders_to_try = decoders_to_try ** 3
        decoders_to_try = 1 - decoders_to_try
        decoders_to_try = 1 + decoders_to_try * (num_decoders - 1)
        decoders_to_try = np.unique(np.round(decoders_to_try).astype(int))
        if num_decoders not in decoders_to_try:
            decoders_to_try[-1] = num_decoders
        
        @ray.remote
        def schedule_with_decoders(benchmark, directory, num_decoders, policy):
            sim = Sim(benchmark, directory)
            meta = sim.schedule_decoding_sw(num_decoders=num_decoders, policy=policy)
            return num_decoders, meta['longest_backlog']

        futures = [schedule_with_decoders.remote(self.benchmark, self.directory, n_decoders, policy) 
                   for n_decoders in decoders_to_try]
        
        backlogs = {}
        for n_decoders, longest_backlog in ray.get(futures):
            backlogs[n_decoders] = longest_backlog
            
        with open(os.path.join(self.directory, f'{self.benchmark}_backlogs_sweep_sw_{policy}.pkl'), 'wb') as file:
            pickle.dump(backlogs, file)
        return backlogs
    
    def baseline_slowdown(self, policy:str='MLS') -> List:
        with open('./active_patches.pkl', 'rb') as file:
            demands = pickle.load(file)
        from natsort import natsorted
        files = natsorted([f.split('.')[0] for f in os.listdir('../benchmarks/new_benchmarks') if f.endswith('.qasm')])
        baseline2 = {benchmark: int(np.mean(list(demands[benchmark]))) for benchmark in files}
        # baseline2 = {benchmark: len(self.alg_qubits) for benchmark in files}
        metadata = self.schedule(num_decoders=baseline2[self.benchmark], policy=policy)
        print(f'Completed baseline-2 for {self.benchmark}, {policy}')
        backlog = metadata['longest_backlog']
        backlog = metadata['total_backlog']
        with open(os.path.join(args.directory, f'{args.benchmark}_opt_decoder{args.policy}.pkl'), 'rb') as file:
            num_decoders = pickle.load(file)
        metadata = self.schedule(num_decoders=num_decoders, policy=policy)
        backlog2 = metadata['total_backlog']
        print(f"Baseline-2 for {self.benchmark} with policy {policy}: {baseline2[self.benchmark]} decoders with longest backlog of {backlog}, opt: {backlog2}, extra slices {metadata['extra_slices']}")
        with open(os.path.join(self.directory, f'{self.benchmark}_backlogs_baseline2_{policy}.pkl'), 'wb') as file:
            pickle.dump([backlog, backlog2], file)
        return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Capacity planning and scheduling for QEC decoders.", add_help=True)
    parser.add_argument('--benchmark', type=str, required=True, help='Benchmark name')
    parser.add_argument('--directory', type=str, required=True, help='Directory containing the LLI files')
    parser.add_argument('--policy', type=str, default='MLS', choices=['MLS', 'RR', 'MFD'], help='Decoder scheduling policy to use')
    parser.add_argument('--histogram', action='store_true', help='Build distance histogram for the benchmark')
    parser.add_argument('--variable-decoder-latency', action='store_true', help='Assume a variable decoder latency for the optimization')
    parser.add_argument('--optimize_pw', action='store_true', help='Run optimization to find the best number of decoders (PWD)')
    parser.add_argument('--optimize_sw', action='store_true', help='[Deprecated] Run optimization to find the best number of decoders (SWD)')
    parser.add_argument('--sweep_sw', action='store_true', help='[Deprecated] Run a decoder sweep to find backlogs for different number of decoders (SWD)')
    parser.add_argument('--sweep_pw', action='store_true', help='Run a decoder sweep to find backlogs for different number of decoders (PWD)')
    parser.add_argument('--num_decoders', type=int, default=None, help='Number of decoders to use for the sweep')
    parser.add_argument('--get_trace', action='store_true', help='Get the trace of the benchmark')
    parser.add_argument('--baseline2', action='store_true', help='Get the baseline-2 slowdown')
    parser.add_argument('--test', action='store_true', help='Testing')
    args = parser.parse_args()

    sim = Sim(args.benchmark, args.directory, fixed_decoder_latency=not args.variable_decoder_latency)
    if args.test:
        print(sim.num_benchmark_qubits, len(sim.alg_qubits))
        sim.test()
    if args.histogram:
        _ = sim.build_distance_histogram()
    if args.baseline2:
        _ = sim.baseline_slowdown(args.policy)
    if args.optimize_pw:
        _ = sim.optimize(args.policy)
    if args.optimize_sw:
        _ = sim.optimize(args.policy, swd=True)
    if args.sweep_sw:
        with open(os.path.join(args.directory, f'{args.benchmark}_opt_decoder_sw_{args.policy}.pkl'), 'rb') as file:
            num_decoders = pickle.load(file)
        _ = sim.get_decoder_sweep_for_best_decoder_sw(num_decoders, args.policy)
    if args.sweep_pw:
        with open(os.path.join(args.directory, f'{args.benchmark}_opt_decoder{args.policy}.pkl'), 'rb') as file:
            num_decoders = pickle.load(file)
        _ = sim.get_decoder_sweep_for_best_decoder(num_decoders, args.policy)
    if args.get_trace:
        with open(os.path.join(args.directory, f'{args.benchmark}_opt_decoder{args.policy}.pkl'), 'rb') as file:
            num_decoders = pickle.load(file)
        temp = sim.schedule_decoding_pw(num_decoders=num_decoders, policy=args.policy)
        with open(f'./{args.benchmark}_scheduling_{args.policy}.pkl', 'wb') as file:
            pickle.dump(temp, file)
    if args.num_decoders:
        temp = sim.schedule(num_decoders=args.num_decoders, policy=args.policy)
        print(temp['longest_backlog'])
