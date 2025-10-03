import numpy as np
import re
import os
import time
import heapq
import pathlib
import json
from collections import defaultdict

# --- 1. High-Performance Tokenizer ---
class SimpleTokenizer:
    """A simple, regex-based tokenizer that maps words to integer IDs."""
    def __init__(self, stop_words=None):
        self.pattern = re.compile(r'\b\w+\b')
        self.stop_words = stop_words if stop_words else set()
        self.vocab = {}
        self.inv_vocab = {}

    def set_vocab(self, vocab):
        """Sets the vocabulary for the tokenizer."""
        self.vocab = vocab
        self.inv_vocab = {i: word for word, i in self.vocab.items()}

    def tokenize(self, text):
        """Tokenizes a string of text into a list of integer IDs."""
        return [self.vocab[token] for token in self.pattern.findall(text.lower()) if token in self.vocab]

# --- 2. The Core Engine: Sharded, Memory-Mapped Index ---
class ShardedIndex:
    """Builds a sharded, memory-mapped inverted index for fast search."""
    def __init__(self, tokenizer, num_shards=8):
        self.tokenizer = tokenizer
        self.num_shards = num_shards

    def build(self, doc_ids, docs, output_dir="nexus_kb_index_scaled", vocab_size=100000, min_idf_threshold=1.5):
        """Builds and saves the index to disk."""
        print(f"Building index for {len(docs)} documents...")
        start_time = time.time()
        
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # --- Step 1: Vocabulary Building ---
        print("Step 1: Building initial vocabulary...")
        word_counts = defaultdict(int)
        for doc in docs:
            for token in self.tokenizer.pattern.findall(doc.lower()):
                if token not in self.tokenizer.stop_words:
                    word_counts[token] += 1
        
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        initial_vocab = {word: i for i, (word, _) in enumerate(sorted_words[:vocab_size])}
        
        # --- Step 2: Precomputing IDF and Pruning Low-Energy Terms ---
        print("Step 2: Precomputing IDF and pruning low-energy terms...")
        df = np.zeros(len(initial_vocab), dtype=np.int32)
        
        temp_tokenizer = SimpleTokenizer()
        temp_tokenizer.set_vocab(initial_vocab)

        for doc in docs:
            tokens = list(set(temp_tokenizer.tokenize(doc)))
            for token_id in tokens:
                df[token_id] += 1
        
        N = len(docs)
        # Add-1 smoothing for IDF calculation to prevent division by zero
        idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
        
        # ** ENERGY-BASED PRUNING **
        high_energy_term_indices = np.where(idf >= min_idf_threshold)[0]
        
        # Create the final vocabulary by filtering and re-indexing.
        inv_initial_vocab = {i: word for word, i in initial_vocab.items()}
        high_energy_words = [inv_initial_vocab[i] for i in high_energy_term_indices]
        final_vocab = {word: new_id for new_id, word in enumerate(high_energy_words)}
        
        self.tokenizer.set_vocab(final_vocab)
        final_idf = idf[high_energy_term_indices]

        print(f"   - Pruned vocab from {len(initial_vocab)} to {len(final_vocab)} terms ({(len(final_vocab)/max(1, len(initial_vocab))):.2%} kept).")

        # --- Step 3: Final Index Build ---
        print("Step 3: Building final index with high-energy vocabulary...")
        doc_lengths = np.array([len(self.tokenizer.tokenize(doc)) for doc in docs], dtype=np.int32)
        avg_doc_length = np.mean(doc_lengths) if doc_lengths.size > 0 else 0

        np.savez_compressed(
            os.path.join(output_dir, "metadata.npz"),
            doc_ids=np.array(doc_ids),
            doc_lengths=doc_lengths,
            avg_doc_length=np.array([avg_doc_length]),
            idf=final_idf
        )
        with open(os.path.join(output_dir, "vocab.json"), "w") as f:
            json.dump(self.tokenizer.vocab, f)

        # Build and save sharded term frequencies (TF)
        for shard_id in range(self.num_shards):
            shard_tfs = []
            shard_term_ids = []
            for term_id in range(shard_id, len(self.tokenizer.vocab), self.num_shards):
                # ** TF QUANTIZATION **
                term_tfs = np.zeros(len(docs), dtype=np.uint8) 
                for doc_idx, doc in enumerate(docs):
                    tokens_in_doc = self.tokenizer.tokenize(doc)
                    count = tokens_in_doc.count(term_id)
                    term_tfs[doc_idx] = min(255, count)
                
                if np.any(term_tfs > 0):
                    shard_tfs.append(term_tfs)
                    shard_term_ids.append(term_id)

            if shard_term_ids:
                np.savez_compressed(
                    os.path.join(output_dir, f"shard_{shard_id}.npz"),
                    term_ids=np.array(shard_term_ids, dtype=np.int32),
                    tfs=np.array(shard_tfs, dtype=np.uint8)
                )
                print(f"  - Shard {shard_id} built with {len(shard_term_ids)} terms.")

        end_time = time.time()
        print(f"Index built in {end_time - start_time:.2f} seconds.")
        print(f"Index saved to: {output_dir}")


# --- 3. The Searcher: Zero-Copy Access and Vectorized Scoring ---
class Searcher:
    """Loads a Nexus KB index and performs fast, vectorized searches."""
    def __init__(self, index_dir="nexus_kb_index_scaled"):
        self.index_dir = index_dir
        self.tokenizer = SimpleTokenizer()
        self._load()

    def _load(self):
        print("Loading memory-mapped index...")
        start_time = time.time()
        
        with open(os.path.join(self.index_dir, "vocab.json"), "r") as f:
            self.tokenizer.set_vocab(json.load(f))

        meta = np.load(os.path.join(self.index_dir, "metadata.npz"))
        self.doc_ids = meta["doc_ids"]
        self.doc_lengths = meta["doc_lengths"]
        self.avg_doc_length = meta["avg_doc_length"][0]
        self.idf = meta["idf"]

        self.shards = []
        num_shards = 0
        while os.path.exists(os.path.join(self.index_dir, f"shard_{num_shards}.npz")):
            shard_path = os.path.join(self.index_dir, f"shard_{num_shards}.npz")
            # Use mmap_mode for zero-copy access
            data = np.load(shard_path, mmap_mode='r')
            term_map = {term_id: i for i, term_id in enumerate(data['term_ids'])}
            self.shards.append({'map': term_map, 'tfs': data['tfs']})
            num_shards += 1
        
        self.num_shards = num_shards
        end_time = time.time()
        print(f"Index loaded in {end_time - start_time:.2f} seconds.")

    def search(self, query, k=5):
        """Performs a vectorized BM25 search."""
        k1, b = 1.2, 0.75
        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens: return []

        total_scores = np.zeros(len(self.doc_ids), dtype=np.float32)
        len_norm = k1 * (1 - b + b * self.doc_lengths / self.avg_doc_length)

        for token_id in set(query_tokens):
            shard_id = token_id % self.num_shards
            if shard_id < len(self.shards):
                shard = self.shards[shard_id]
                if token_id in shard['map']:
                    tf_idx = shard['map'][token_id]
                    tfs = shard['tfs'][tf_idx].astype(np.float32)
                    
                    numerator = tfs * (k1 + 1)
                    denominator = tfs + len_norm
                    
                    scores = self.idf[token_id] * (numerator / denominator)
                    total_scores += scores
        
        top_k_indices = heapq.nlargest(k, range(len(total_scores)), total_scores.take)
        return [(self.doc_ids[i], total_scores[i]) for i in top_k_indices if total_scores[i] > 0]

# --- Main Demonstration ---
if __name__ == "__main__":
    corpus = {
        "doc1": "The Andromeda Galaxy is a barred spiral galaxy and is the nearest major galaxy to the Milky Way.",
        "doc2": "NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices.",
        "doc3": "The Python language is dynamically-typed and garbage-collected. It supports multiple programming paradigms.",
        "doc4": "The Milky Way is the galaxy that includes our Solar System.",
        "doc5": "A memory-mapped file is a segment of virtual memory that has been assigned a direct byte-for-byte correlation with some portion of a file or file-like resource."
    }
    doc_ids, docs = list(corpus.keys()), list(corpus.values())

    # 1. Build the scaled index with energy-based pruning
    tokenizer = SimpleTokenizer()
    indexer = ShardedIndex(tokenizer, num_shards=2)
    # Use a low IDF threshold for this tiny corpus to show it's working
    indexer.build(doc_ids, docs, output_dir="my_nexus_kb_index_scaled", min_idf_threshold=0.5)
    
    print("\n" + "="*50 + "\n")
    
    # 2. Load the index and perform searches
    searcher = Searcher(index_dir="my_nexus_kb_index_scaled")
    
    queries = ["python programming", "galaxy", "memory map"]
    for q in queries:
        start_time = time.time()
        results = searcher.search(q, k=3)
        end_time = time.time()
        
        print(f"Query: '{q}' (took {(end_time - start_time) * 1000:.2f} ms)")
        for doc_id, score in results:
            print(f"  - {doc_id} (Score: {score:.2f})")
        print("-" * 20)


