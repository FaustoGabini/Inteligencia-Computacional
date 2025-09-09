import tkinter as tk
from GridDrawer import GridDrawer
from digitos_patrones import digitos
from Hopfield import Hopfield
import numpy as np

# ------------------ MAIN ------------------
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Hopfield")
    root.configure(bg="#f0f0f0")  # color de fondo de la ventana
    root.geometry("300x500")
    drawer = GridDrawer(root)
    
    patron = None
    def obtener_patron(): 
        patron = drawer.get_pattern()
        print(patron)


    # Botón para mostrar patrón
    boton1 = tk.Button(root, text="Obtener patrón", command=obtener_patron)
    boton1.pack(pady=5)

    # Botón para limpiar
    boton2 = tk.Button(root, text="Limpiar", command=drawer.clear)
    boton2.pack(pady=5)

    root.mainloop()