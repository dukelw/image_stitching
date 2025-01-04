import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


class ImageCropTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Crop Tool")

        # Initialize variables
        self.image = None
        self.photo = None
        self.canvas = None
        self.rect = None
        self.start_x = None
        self.start_y = None

        # Frame to hold the buttons and input fields
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(pady=10)

        # Button to load image
        self.load_btn = tk.Button(
            self.control_frame, text="Load Image", command=self.load_image
        )
        self.load_btn.pack(side=tk.LEFT, padx=10)

        # Label to show image dimensions
        self.size_label = tk.Label(self.control_frame, text="Image size: N/A")
        self.size_label.pack(side=tk.LEFT, padx=10)

        # Entry for width and height
        self.width_label = tk.Label(self.control_frame, text="Width:")
        self.width_label.pack(side=tk.LEFT, padx=5)
        self.width_entry = tk.Entry(self.control_frame)
        self.width_entry.pack(side=tk.LEFT, padx=5)

        self.height_label = tk.Label(self.control_frame, text="Height:")
        self.height_label.pack(side=tk.LEFT, padx=5)
        self.height_entry = tk.Entry(self.control_frame)
        self.height_entry.pack(side=tk.LEFT, padx=5)

        # Button to crop image
        self.crop_btn = tk.Button(
            self.control_frame, text="Crop", command=self.crop_image
        )
        self.crop_btn.pack(side=tk.LEFT, padx=10)

        # Canvas to display image
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(pady=10)
        self.canvas = tk.Canvas(self.canvas_frame)
        self.canvas.pack()

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def load_image(self):
        """Load an image and display it on the canvas."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        if file_path:
            self.image = Image.open(file_path)
            width, height = self.image.size

            # Update size label with image dimensions
            self.size_label.config(text=f"Image size: {width}x{height}")

            # Resize image to fit within canvas size (800x600)
            self.canvas.config(width=width, height=height)

            # Convert to PhotoImage for Tkinter
            self.photo = ImageTk.PhotoImage(self.image)

            # Display image on canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.canvas.config(
                scrollregion=self.canvas.bbox(tk.ALL)
            )  # Update scroll region

    def on_press(self, event):
        """Store the starting coordinates when the user presses the mouse button."""
        self.start_x = event.x
        self.start_y = event.y

        if self.rect:
            self.canvas.delete(self.rect)  # Remove the previous rectangle

        # Create a new rectangle
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, outline="red"
        )

    def on_drag(self, event):
        """Update the size of the rectangle as the user drags the mouse."""
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        """Store the final coordinates when the user releases the mouse button."""
        self.end_x = event.x
        self.end_y = event.y

    def crop_image(self):
        """Crop the selected region and display the cropped image."""
        if not self.start_x or not self.start_y:
            messagebox.showerror("Error", "Please select a region to crop.")
            return

        width = self.width_entry.get()
        height = self.height_entry.get()

        try:
            width = int(width)
            height = int(height)
        except ValueError:
            messagebox.showerror(
                "Error", "Please enter valid numbers for width and height."
            )
            return

        # Ensure the selected region is within the bounds of the original image
        crop_box = (
            min(self.start_x, self.end_x),
            min(self.start_y, self.end_y),
            min(self.start_x, self.end_x) + width,
            min(self.start_y, self.end_y) + height,
        )

        cropped_img = self.image.crop(crop_box)
        return cropped_img



