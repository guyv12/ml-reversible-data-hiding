import math
import heapq
import torch
from collections import Counter

class HuffmanNode:
    def __init__(self, char: str | None, freq: int):
        self.char = char
        self.freq = freq
        self.left: HuffmanNode | None = None
        self.right: HuffmanNode | None = None

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

def get_huffman_codes(node: HuffmanNode | None, code="", huffman_codes=None):
    if huffman_codes is None:
        huffman_codes = {}
    if node:
        if node.char is not None:
            huffman_codes[node.char] = code
        get_huffman_codes(node.left, code + "0", huffman_codes)
        get_huffman_codes(node.right, code + "1", huffman_codes)

    return huffman_codes

def delta_encode(ref_pixels: torch.Tensor):
    ref_pixels = ref_pixels.to(torch.int16)

    deltas = torch.diff(ref_pixels)
    deltas += 255 # offset
    encoded_deltas = torch.cat((ref_pixels[0].unsqueeze(0), deltas))

    return encoded_deltas

def huffman_codebook_to_bits(huffman_codes, b_sym):
    codebook = ""

    L_max = max(len(code) for code in huffman_codes.values())
    b_code = math.ceil(math.log2(L_max))

    for symbol, code in huffman_codes.items():
        # symbol value
        codebook += format(int(symbol), f'0{b_sym}b')

        # code length
        codebook += format(len(code), f'0{b_code}b')

        # huffman code itself
        codebook += code

    return codebook
