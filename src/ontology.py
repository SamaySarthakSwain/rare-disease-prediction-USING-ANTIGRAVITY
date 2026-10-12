"""
HPO Ontology Parser and Graph Traversal Utilities
Extracts DAG structure, ancestor closures, and Information Content (IC) calculation.
"""

import os
from collections import defaultdict, deque

class HPOntology:
    def __init__(self, obo_path=None):
        self.terms = {}           # term_id -> {"name": str, "def": str, "is_obsolete": bool, ...}
        self.parents = defaultdict(set)   # child_id -> set(parent_ids)
        self.children = defaultdict(set)  # parent_id -> set(child_ids)
        self.alt_ids = {}         # alt_id -> primary_id
        self.replaced_by = {}     # obsolete_id -> replacement_id
        self.names = {}           # id -> name
        self.definitions = {}     # id -> def
        self.ancestors_cache = {} # id -> set(ancestor_ids) including self
        self.root = "HP:0000118"  # Phenotypic abnormality root
        
        if obo_path:
            self.load_obo(obo_path)
            
    def load_obo(self, obo_path):
        in_term = False
        current_id = None
        current_name = None
        current_def = None
        current_obsolete = False
        current_parents = []
        
        with open(obo_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == "[Term]":
                    if in_term and current_id:
                        self._add_term(current_id, current_name, current_def, current_obsolete, current_parents)
                    in_term = True
                    current_id = None
                    current_name = None
                    current_def = None
                    current_obsolete = False
                    current_parents = []
                    continue
                elif line.startswith("[") and line.endswith("]"):
                    if in_term and current_id:
                        self._add_term(current_id, current_name, current_def, current_obsolete, current_parents)
                    in_term = False
                    continue
                    
                if in_term:
                    if line.startswith("id: "):
                        current_id = line[4:].strip()
                    elif line.startswith("name: "):
                        current_name = line[6:].strip()
                    elif line.startswith("def: "):
                        current_def = line[5:].strip()
                    elif line.startswith("is_obsolete: true"):
                        current_obsolete = True
                    elif line.startswith("alt_id: "):
                        alt = line[8:].strip()
                        self.alt_ids[alt] = current_id
                    elif line.startswith("replaced_by: "):
                        self.replaced_by[current_id] = line[13:].strip()
                    elif line.startswith("is_a: "):
                        p = line[6:].split("!")[0].strip()
                        current_parents.append(p)
                        
            if in_term and current_id:
                self._add_term(current_id, current_name, current_def, current_obsolete, current_parents)
                
    def _add_term(self, tid, name, definition, is_obsolete, parents):
        self.terms[tid] = {
            "name": name,
            "definition": definition,
            "is_obsolete": is_obsolete
        }
        if not is_obsolete:
            self.names[tid] = name
            if definition:
                self.definitions[tid] = definition
            for p in parents:
                self.parents[tid].add(p)
                self.children[p].add(tid)
                
    def resolve_term(self, term_id):
        """Resolves alt_ids and replacements to active primary HPO term ID."""
        if not term_id:
            return None
        term_id = term_id.strip()
        if term_id in self.terms and not self.terms[term_id]["is_obsolete"]:
            return term_id
        if term_id in self.alt_ids:
            primary = self.alt_ids[term_id]
            if primary in self.terms and not self.terms[primary]["is_obsolete"]:
                return primary
        if term_id in self.replaced_by:
            repl = self.replaced_by[term_id]
            return self.resolve_term(repl)
        return None
        
    def get_ancestors(self, term_id):
        """Returns all ancestors of term_id, including term_id itself."""
        canonical = self.resolve_term(term_id)
        if not canonical:
            return set()
        if canonical in self.ancestors_cache:
            return self.ancestors_cache[canonical]
            
        ancestors = set()
        queue = deque([canonical])
        while queue:
            curr = queue.popleft()
            if curr in ancestors:
                continue
            ancestors.add(curr)
            for p in self.parents.get(curr, []):
                resolved_p = self.resolve_term(p)
                if resolved_p and resolved_p not in ancestors:
                    queue.append(resolved_p)
                    
        self.ancestors_cache[canonical] = ancestors
        return ancestors

    def get_descendants(self, term_id):
        """Returns all descendants of term_id, including term_id itself."""
        canonical = self.resolve_term(term_id)
        if not canonical:
            return set()
        descendants = set()
        queue = deque([canonical])
        while queue:
            curr = queue.popleft()
            if curr in descendants:
                continue
            descendants.add(curr)
            for c in self.children.get(curr, []):
                resolved_c = self.resolve_term(c)
                if resolved_c and resolved_c not in descendants:
                    queue.append(resolved_c)
        return descendants
        
    def is_descendant_of(self, child_term, parent_term):
        parent_canonical = self.resolve_term(parent_term)
        return parent_canonical in self.get_ancestors(child_term)

    def term_depth(self, term_id):
        """Minimum distance from root (HP:0000118) to term."""
        canonical = self.resolve_term(term_id)
        if not canonical:
            return -1
        if canonical == self.root:
            return 0
        queue = deque([(self.root, 0)])
        visited = set()
        while queue:
            curr, depth = queue.popleft()
            if curr == canonical:
                return depth
            visited.add(curr)
            for ch in self.children.get(curr, []):
                if ch not in visited:
                    queue.append((ch, depth + 1))
        return -1
