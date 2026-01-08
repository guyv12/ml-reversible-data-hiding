import heapq
from collections import Counter

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

    def __gt__(self, other):
        return self.freq > other.freq

def build_huffman_tree(data):
    counts = Counter(data) # dictionary
    heap = [HuffmanNode(char, freq) for char, freq in counts.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left_child = heapq.heappop(heap)
        right_child = heapq.heappop(heap)
        merged = HuffmanNode(None, left_child.freq + right_child.freq)
        merged.left = left_child
        merged.right = right_child
        heapq.heappush(heap, merged)

    return heap[0]

def get_huffman_codes(node: HuffmanNode, code="", huffman_codes=None):
    if huffman_codes is None:
        huffman_codes = {}
    if node:
        if node.char is not None:
            huffman_codes[node.char] = code
        get_huffman_codes(node.left, code + "0", huffman_codes)
        get_huffman_codes(node.right, code + "1", huffman_codes)

    return huffman_codes