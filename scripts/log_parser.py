# Construct a parser that goes through every log file for all workloads and extracts the mapping of qubits to coords
import re
import os
from typing import Dict, Tuple, List, Set
import rustworkx as rx

class LayoutGrid:
    def __init__(self):
        self.grid = []
        self.rows = 0
        self.cols = 0
        self.routing_nodes = set()
        
    def parse_from_err_file(self, err_file_path: str):
        """Parse layout grid from .err file"""
        with open(err_file_path, 'r') as f:
            content = f.read()
        
        grid_lines = []
        for line in content.split('\n'):
            if '[LSC OUTPUT] ::::::' in line and any(x in line for x in ['Mr', 'Qr', 'rr']):
                grid_content = line.split('::::::')[-1].strip()
                if grid_content and not grid_content.startswith('The '):
                    grid_lines.append(grid_content)
                            
        if not grid_lines:
            raise ValueError("Could not find grid layout in .err file")
        
        self.rows = len(grid_lines)
        self.cols = len(grid_lines[0]) if grid_lines else 0
        
        self.grid = []
        for row_idx, line in enumerate(grid_lines):
            row = []
            for col_idx, char in enumerate(line):
                row.append(char)
                if char == 'r':
                    self.routing_nodes.add((row_idx, col_idx))
            self.grid.append(row)

    def to_1d(self, coord: Tuple[int, int]) -> int:
        """Convert 2D coordinates to 1D index"""
        return coord[0] * self.cols + coord[1]
    
    def from_1d(self, index: int) -> Tuple[int, int]:
        """Convert 1D index to 2D coordinates"""
        return (index // self.cols, index % self.cols)
    
    def vertical_neighbors(self, index : int) -> List[int]:
        neighbors = []
        down = index + self.cols
        up = index - self.cols
        if index // self.cols != 0:
            neighbors.append(up)
        if index // self.cols != self.rows - 1:
            neighbors.append(down)
        return neighbors

    def horizontal_neighbors(self, index : int) -> List[int]:
        neighbors = []
        left = index - 1
        right = index + 1
        if index % self.cols != 0:
            neighbors.append(left)
        if index % self.cols != self.cols - 1:
            neighbors.append(right)
        return neighbors

    def route_pair(self, start : Tuple[int, int], end : Tuple[int, int], op1 : str, op2 : str, blocked : Set[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], Set[Tuple[int, int]]]:
        """
        Route a single pair of coordinates 
        """
        layout_graph = rx.generators.grid_graph(rows=self.rows, cols=self.cols)
        blocked1d = [self.to_1d(coord) for coord in blocked]
        layout_graph.remove_nodes_from(blocked1d)
        (row_start, col_start) = start
        (row_end, col_end) = end
        # special cases of immediate neighbors
        if abs(col_start - col_end) == 1 and row_start == row_end and op1 == 'X' and op2 == 'X':
            return ([start, end], blocked)
        elif abs(row_start - row_end) == 1 and col_start == col_end and op1 == 'Z' and op2 == 'Z':
            return ([start, end], blocked)
        # general case
        else: 
            shortest_path_len = float('inf')
            shortest_pair = None

            starts = self.vertical_neighbors(self.to_1d(start)) if op1 == 'Z' else self.horizontal_neighbors(self.to_1d(start))
            ends = self.vertical_neighbors(self.to_1d(end)) if op2 == 'Z' else self.horizontal_neighbors(self.to_1d(end))
            pairs = [(s,t ) for s in starts for t in ends if layout_graph.has_node(s) and layout_graph.has_node(t)]
            for s,t in pairs:
                print(self.from_1d(s), self.from_1d(t))
                const_1 = lambda _ : 1
                dist_dict = (rx.dijkstra_shortest_path_lengths(layout_graph, edge_cost_fn=const_1, node=s, ))
                if t in dist_dict.keys():
                    dist = dist_dict[t]
                    print(f"{dist=}")
                elif s == t:
                    dist = 0
                else:   
                    dist = float('inf')
                if dist < shortest_path_len:
                    shortest_path_len = dist
                    shortest_pair = s,t
            if shortest_pair != None:
                s,t = shortest_pair 
                if s == t:
                    path = [s]
                else:
                    path = list(rx.dijkstra_shortest_paths(layout_graph, source=s, target = t)[t])
                route = [self.from_1d(v) for v in path]
                route = [start] + route + [end]
                print(f"Found route from {start} to {end} with ops {op1}, {op2}: {route}")
                for v in path:
                    blocked.add(self.from_1d(v))
            else:
                blocked_between = [b for b in blocked if b[0] >= min(start[0], end[0]) and b[0] <= max(start[0], end[0]) and b[1] >= min(start[1], end[1]) and b[1] <= max(start[1], end[1])]
                print(f"Blocked nodes: {sorted(blocked)}")
                print("s,t pairs:", [(self.from_1d(p[0]), self.from_1d(p[1])) for p in pairs])
                route = []
                raise Exception(f"Could not find route from {start} to {end} with ops {op1}, {op2}")

        return route, blocked

class LogParser:
    def __init__(self):
        self.qubit_locations = {} 
        self.magic_state_locations = {} 
        self.core_locations = {}  
        self.multibody_measures = [] 
        self.layout_grid = None
        
    def parse_layout_from_err(self, err_file_path: str):
        """Parse the layout grid from .err file"""
        self.layout_grid = LayoutGrid()
        self.layout_grid.parse_from_err_file(err_file_path)
        
    def parse_log_file(self, log_file_path: str):
        """Parse a single log file and extract qubit mappings and MultiBodyMeasure info"""
        with open(log_file_path, 'r') as f:
            content = f.read()
        
        core_qubit_pattern = r'Core qubit mapping: Qubit ID (\d+) -> Physical location \((\d+),(\d+)\)'
        for match in re.finditer(core_qubit_pattern, content):
            qubit_id = int(match.group(1))
            row = int(match.group(2))
            col = int(match.group(3))
            self.qubit_locations[qubit_id] = (row, col)
            self.core_locations[qubit_id] = (row, col)
        
        magic_state_pattern = r'Magic state allocated: Patch ID (\d+) -> Physical location \((\d+),(\d+)\)'
        for match in re.finditer(magic_state_pattern, content):
            patch_id = int(match.group(1))
            row = int(match.group(2))
            col = int(match.group(3))
            self.magic_state_locations[patch_id] = (row, col)
            self.qubit_locations[patch_id] = (row, col) 
        
        ancilla_pattern = r'Ancilla allocated: Patch ID (\d+) -> Physical location \((\d+),(\d+)\)'
        for match in re.finditer(ancilla_pattern, content):
            patch_id = int(match.group(1))
            row = int(match.group(2))
            col = int(match.group(3))
            self.qubit_locations[patch_id] = (row, col)
        
        y_state_patterns = [
            r'Y state allocated \(pre-distilled\): Patch ID (\d+) -> Physical location \((\d+),(\d+)\)',
            r'Y state prepared: Patch ID (\d+) -> Physical location \((\d+),(\d+)\)'
        ]
        for pattern in y_state_patterns:
            for match in re.finditer(pattern, content):
                patch_id = int(match.group(1))
                row = int(match.group(2))
                col = int(match.group(3))
                self.qubit_locations[patch_id] = (row, col)
        
        multibody_pattern = r'MultiBodyMeasure: Patch (\d+):([XYZ]) at \((\d+),(\d+)\)(?:\s+\[MAGIC STATE\])?\s+\+\s+Patch (\d+):([XYZ]) at \((\d+),(\d+)\)(?:\s+\[MAGIC STATE\])?'
        for match in re.finditer(multibody_pattern, content):
            patch1_id = int(match.group(1))
            patch1_op = match.group(2)
            patch1_row = int(match.group(3))
            patch1_col = int(match.group(4))
            patch2_id = int(match.group(5))
            patch2_op = match.group(6)
            patch2_row = int(match.group(7))
            patch2_col = int(match.group(8))
            
            full_match = match.group(0)
            patch1_is_magic = "[MAGIC STATE]" in full_match.split("+")[0]
            patch2_is_magic = "[MAGIC STATE]" in full_match.split("+")[1]
            
            # Calculate Manhattan distance
            manhattan_dist = abs(patch1_row - patch2_row) + abs(patch1_col - patch2_col)
            
            multibody_info = {
                'patch1_id': patch1_id,
                'patch1_op': patch1_op,
                'patch1_coords': (patch1_row, patch1_col),
                'patch1_is_magic': patch1_is_magic,
                'patch2_id': patch2_id,
                'patch2_op': patch2_op,
                'patch2_coords': (patch2_row, patch2_col),
                'patch2_is_magic': patch2_is_magic,
                'manhattan_distance': manhattan_dist
            }
            self.multibody_measures.append(multibody_info)
            
            self.qubit_locations[patch1_id] = (patch1_row, patch1_col)
            self.qubit_locations[patch2_id] = (patch2_row, patch2_col)
    
    def get_qubits_used_in_lli(self, lli_file_path: str) -> Set[int]:
        """Extract all qubit IDs used in MultiBodyMeasure instructions from LLI file"""
        qubits_in_lli = set()
        
        with open(lli_file_path, 'r') as f:
            content = f.read()
        
        multibody_lli_pattern = r'MultiBodyMeasure\s+([0-9:,XYZ\s]+)'
        for match in re.finditer(multibody_lli_pattern, content):
            qubit_ops = match.group(1)
            qubit_pattern = r'(\d+):[XYZ]'
            for qubit_match in re.finditer(qubit_pattern, qubit_ops):
                qubit_id = int(qubit_match.group(1))
                qubits_in_lli.add(qubit_id)
        
        return qubits_in_lli
    
    def create_qubit_mapping_for_lli(self, lli_file_path: str) -> Dict[int, Tuple[int, int]]:
        """Create a mapping of qubit IDs to coordinates for all qubits used in the LLI file"""
        qubits_needed = self.get_qubits_used_in_lli(lli_file_path)
        
        qubit_mapping = {}
        missing_qubits = []
        
        if not qubits_needed:
            # No MultiBodyMeasure instructions found in LLI file
            for qubit_id, coord in self.qubit_locations.items():
                qubit_mapping[qubit_id] = coord
            return qubit_mapping
        
        for qubit_id in qubits_needed:
            if qubit_id in self.qubit_locations:
                qubit_mapping[qubit_id] = self.qubit_locations[qubit_id]
            else:
                missing_qubits.append(qubit_id)
                
        for qubit_id, coord in self.qubit_locations.items(): # All remaining mappings
            if qubit_id not in qubit_mapping:
                qubit_mapping[qubit_id] = coord
        
        if missing_qubits:
            print(f"Warning: Could not find coordinates for qubits: {missing_qubits}")
        
        return qubit_mapping
    
    def get_multibody_measure_summary(self) -> List[Dict]:
        """Get summary of all MultiBodyMeasure instructions with distances"""
        return self.multibody_measures
    
    def manhattan_distance(self, coord1: Tuple[int, int], coord2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two coordinates"""
        return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])
    

    def get_distance(self, coord_pairs: List[Tuple[Tuple[int, int], str, Tuple[int, int], str]]) -> List[int]:
        """
        Calculate routing distances for multiple coordinate pairs with conflict avoidance
        
        Args:
            coord_pairs: List of (start_coord, start_op, end_coord, end_op) tuples
            where start_op and end_op are 'X' or 'Z'
        Returns:
            List of distances corresponding to each pair
        """
        if self.layout_grid is None:
            raise ValueError("Layout grid not loaded. Call parse_layout_from_err() first.")
    
        blocked = set(self.core_locations.values()).union(set(self.magic_state_locations.values()))

        routes = []
        for coord in coord_pairs:
            route, blocked = self.layout_grid.route_pair(
                start=coord[0],
                end=coord[2],
                op1=coord[1],
                op2=coord[3],
                blocked=blocked
            )
            routes.append(route)
        return [len(r) for r in routes]
def parse_all_logs(log_directory: str, lli_file_path: str, err_file_path: str) -> Tuple[Dict[int, Tuple[int, int]], 'LogParser']:
    """Parse all log files and layout, create qubit mapping for LLI file"""
    parser = LogParser()
    
    print(f"Parsing layout from {err_file_path}...")
    parser.parse_layout_from_err(err_file_path)
    
    for filename in os.listdir(log_directory):
        if filename.endswith('.log'):
            log_path = os.path.join(log_directory, filename)
            print(f"Parsing {log_path}...")
            parser.parse_log_file(log_path)
    
    qubit_mapping = parser.create_qubit_mapping_for_lli(lli_file_path)
    
    print(f"\nFound coordinates for {len(qubit_mapping)} qubits")
    print(f"Parsed {len(parser.multibody_measures)} MultiBodyMeasure instructions")
    print(f"Magic state locations: {len(parser.magic_state_locations)}")
    print(f"Layout grid: {parser.layout_grid.rows}x{parser.layout_grid.cols} with {len(parser.layout_grid.routing_nodes)} routing nodes")
    return qubit_mapping, parser

def extract_from_multibody_measure(measure: Dict) -> Tuple[Tuple[int, int], str, Tuple[int, int], str]:
    """Extract relevant information from a MultiBodyMeasure instruction"""
    start = measure['patch1_coords']
    end = measure['patch2_coords']
    start_op = measure['patch1_op']
    end_op = measure['patch2_op']
    return start, start_op, end, end_op

