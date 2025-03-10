import matplotlib.pyplot as plt
import numpy as np

# Example data
x = np.linspace(0, 2*np.pi, 300)
y_sin = np.sin(x)
y_cos = np.cos(x)
y_tan = np.tan(x / 2)

# Make a figure (2 rows x 4 columns = 8 subplot "slots")
fig = plt.figure(figsize=(10, 6))

# Top-left spans subplots 1 and 2 (first row, columns 1–2)
top_left = fig.add_subplot(2, 4, (1, 2))
top_left.plot(x, y_sin, color='blue', label='Sine')
top_left.set_title("Top Left (slots 1–2)")
top_left.legend()

# Top-right spans subplots 3 and 4 (first row, columns 3–4)
top_right = fig.add_subplot(2, 4, (3, 4))
top_right.plot(x, y_cos, color='green', label='Cosine')
top_right.set_title("Top Right (slots 3–4)")
top_right.legend()

# Bottom-middle spans subplots 6 and 7 (second row, columns 2–3)
bottom_middle = fig.add_subplot(2, 4, (6, 7))
bottom_middle.plot(x, y_tan, color='red', label='Tangent')
bottom_middle.set_title("Bottom Middle (slots 6–7)")
bottom_middle.legend()

plt.tight_layout()
plt.savefig("big_subplots.png", dpi=300)
plt.show()
