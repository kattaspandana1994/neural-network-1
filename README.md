import tensorflow as tf
import numpy as np

# Set random seed for reproducibility
tf.random.set_seed(42)

### 1. Create a random tensor of shape (4, 6)
print("Task 1: Creating random tensor")
random_tensor = tf.random.normal([4, 6])
print("\nOriginal Tensor:")
print(random_tensor)

### 2. Find rank and shape
print("\nTask 2: Finding rank and shape")
print(f"Rank of tensor: {tf.rank(random_tensor).numpy()}")
print(f"Shape of tensor: {random_tensor.shape}")

### 3. Reshape and transpose
print("\nTask 3: Reshaping and transposing")
# Reshape to (2, 3, 4)
reshaped_tensor = tf.reshape(random_tensor, [2, 3, 4])
print("\nAfter reshaping to (2, 3, 4):")
print(reshaped_tensor)
print(f"New shape: {reshaped_tensor.shape}")

# Transpose to (3, 2, 4)
transposed_tensor = tf.transpose(reshaped_tensor, perm=[1, 0, 2])
print("\nAfter transposing to (3, 2, 4):")
print(transposed_tensor)
print(f"Final shape: {transposed_tensor.shape}")

### 4. Broadcasting
print("\nTask 4: Broadcasting")
# Create smaller tensor (1, 4)
small_tensor = tf.constant([[1.0, 2.0, 3.0, 4.0]])
print("\nSmall tensor (1, 4):")
print(small_tensor)

# Broadcast and add
broadcasted_result = small_tensor + random_tensor[:, :4]
print("\nResult after broadcasting and addition:")
print(broadcasted_result)

### 5. Broadcasting Explanation
print("\nTask 5: Broadcasting Explanation")
print("""
Broadcasting in TensorFlow follows these rules:
1. Arrays with fewer dimensions are padded with ones on their left side.
2. Arrays with fewer elements in a dimension are repeated to match the larger array.
3. If the shapes are incompatible, broadcasting fails.

In this example:
- The small tensor (1, 4) is broadcast to match (4, 4)
- The 1 in dimension 0 is repeated 4 times
- This allows element-wise addition with the larger tensor
""")# neural-network-1
ASSIGNMENT HW 1
