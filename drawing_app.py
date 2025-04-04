import tkinter as tk
from tkinter import Canvas, messagebox
import numpy as np
import cv2
from PIL import Image, ImageDraw
from mnist_model import predict_digit  # Import model function

class DrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition")
        
        self.canvas = Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)
        
        self.predict_button = tk.Button(root, text='Predict', command=self.predict)
        self.predict_button.pack()
        
        self.clear_button = tk.Button(root, text='Clear', command=self.clear_canvas)
        self.clear_button.pack()
        
        self.image = Image.new("L", (280, 280), 255)  # Grayscale image
        self.draw_handle = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x, y, x+8, y+8, fill='black', width=5)
        self.draw_handle.ellipse([x, y, x+8, y+8], fill='black')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)  # Reset image
        self.draw_handle = ImageDraw.Draw(self.image)

    def predict(self):
        # Resize the image to 28x28
        img = self.image.resize((28, 28))
        img = np.array(img)
        img = cv2.bitwise_not(img)  # Invert colors (black → white, white → black)
        prediction = predict_digit(img)
        
        # Show the result
        print(f'Predicted digit: {prediction}')
        messagebox.showinfo("Prediction Result", f'Predicted Digit: {prediction}')

# Start the Tkinter app
if __name__ == "__main__":
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()
