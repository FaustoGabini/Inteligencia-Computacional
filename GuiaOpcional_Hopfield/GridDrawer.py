import tkinter as tk
import numpy as np

class GridDrawer: 
    def __init__(self, master, filas=7, cols=5, size=50): 
        self.filas = filas
        self.cols = cols
        self.size = size
        self.canvas = tk.Canvas(master, width=cols*size, height=filas*size, bg="white", highlightthickness=0)
        self.canvas.pack(pady=10)  # 👈 agregamos espacio arriba/abajo

        
        # Matriz que guarda el estado 
        self.grid = np.full((filas, cols), -1, dtype=int)
        self.rects = {}

        # Dibujamos la cuadricula
        for i in range(filas): 
            for j in range(cols): 
                x1, y1 = j*size, i*size
                x2, y2 = x1+size, y1+size
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="gray")
                self.rects[(i, j)] = rect
                self.canvas.tag_bind(rect, "<Button-1>", lambda e, r=i, c=j: self.toggle(r, c))

    def toggle(self, fila, col): 
        if self.grid[fila, col] == -1:
            self.grid[fila, col] = 1
            self.canvas.itemconfig(self.rects[(fila, col)], fill="black")
        else:
            self.grid[fila, col] = -1
            self.canvas.itemconfig(self.rects[(fila, col)], fill="white")

    def get_pattern(self):
        return self.grid.flatten()

    def clear(self):
        """Resetea la grilla a todo blanco (-1)"""
        self.grid[:, :] = -1
        for rect in self.rects.values():
            self.canvas.itemconfig(rect, fill="white")


