class Node:
    def __init__(self, freq, symbol, left=None, right=None):
        self.freq = freq  # Frequency of the symbol
        self.symbol = symbol  # The character symbol
        self.left = left  # Left child
        self.right = right  # Right child
        self.huff = ''  # Huffman code for the symbol

def printNodes(node, val=''):
    newVal = val + str(node.huff)
    if node.left:
        printNodes(node.left, newVal)
    if node.right:
        printNodes(node.right, newVal)
    if not node.left and not node.right:  # Leaf node
        print(f"{node.symbol} -> {newVal}")

# Characters for Huffman Tree
chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
# Frequency of characters
freq = [4, 7, 12, 14, 17, 43, 54]
# List containing unused nodes
nodes = []

# Converting characters and frequencies into Huffman Tree nodes
for x in range(len(chars)):
    nodes.append(Node(freq[x], chars[x]))

# Build the Huffman Tree
while len(nodes) > 1:
    # Sort all the nodes in ascending order based on their frequency
    nodes = sorted(nodes, key=lambda x: x.freq)
    # Pick 2 smallest nodes
    left = nodes[0]
    right = nodes[1]
    # Assign directional value to these nodes
    left.huff = 0
    right.huff = 1
    # Combine the 2 smallest nodes to create a new node as their parent
    newNode = Node(left.freq + right.freq, left.symbol + right.symbol, left, right)
    # Remove the 2 nodes and add their parent as a new node among others
    nodes.remove(left)
    nodes.remove(right)
    nodes.append(newNode)

# Huffman Tree is ready! Print the codes
print("Huffman Codes:")
printNodes(nodes[0])
