import time
import customtkinter as ctk
from PIL import Image, ImageTk

import torch
from diffusers import StableDiffusionPipeline

# --------------------------------------------------
# 1) LOAD MODEL ONCE — correctly and predictably
# --------------------------------------------------

device = "cpu"   # you do NOT have CUDA, stop pretending

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
).to(device)

# --------------------------------------------------
# 2) APP SETUP — consistent and readable
# --------------------------------------------------

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("550x650")
app.title("Stable Bud")

# Main container (better structure than random x,y)
frame = ctk.CTkFrame(app)
frame.pack(fill="both", expand=True, padx=10, pady=10)

# --------------------------------------------------
# 3) UI ELEMENTS — structured, not messy
# --------------------------------------------------

prompt = ctk.CTkEntry(
    master=frame,
    height=40,
    font=("Arial", 18),
    fg_color="white",
    text_color="black",
    placeholder_text="Describe your image..."
)
prompt.pack(fill="x", pady=(0, 10))

lmain = ctk.CTkLabel(
    master=frame,
    text="Generated image will appear here",
    width=512,
    height=512
)
lmain.pack(pady=10)

# --------------------------------------------------
# 4) GENERATION FUNCTION — safe + professional
# --------------------------------------------------

def generate():
    text = prompt.get().strip()
    if not text:
        print("Error: Empty prompt")
        return

    try:
        print("Generating image... please wait.")

        image = pipe(
            text,
            guidance_scale=8.5
        ).images[0]

        # Save with unique name (your old code overwrote files)
        filename = f"image_{int(time.time())}.png"
        image.save(filename)

        img = ImageTk.PhotoImage(image)
        lmain.configure(image=img, text="")
        lmain.image = img   # keep reference

        print(f"Saved: {filename}")

    except Exception as e:
        print("Generation failed:", e)

# --------------------------------------------------
# 5) BUTTON — last, after function exists
# --------------------------------------------------

trigger = ctk.CTkButton(
    master=frame,
    text="Generate",
    command=generate,
    height=40,
    width=140
)
trigger.pack(pady=10)

app.mainloop()
