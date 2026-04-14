import tkinter as tk
from tkinter import scrolledtext
import threading

from rag import ask


BG = "#0a0e19"
TEXT = "#e2e8f0"

window = tk.Tk()
window.title("RAG Bot")
window.geometry("750x550")
window.configure(bg=BG)

chat = scrolledtext.ScrolledText(window, bg=BG, fg=TEXT, wrap=tk.WORD)
chat.pack(fill=tk.BOTH, expand=True)
chat.config(state=tk.DISABLED)

entry = tk.Entry(window, font=("Arial", 13))
entry.pack(fill=tk.X, padx=10, pady=8)


def add(sender, msg):
    chat.config(state=tk.NORMAL)
    chat.insert(tk.END, f"\n{sender}:\n{msg}\n")
    chat.config(state=tk.DISABLED)
    chat.see(tk.END)


def send(event=None):
    q = entry.get()
    if not q:
        return

    entry.delete(0, tk.END)
    add("Ты", q)

    chat.config(state=tk.NORMAL)
    idx = chat.index(tk.END)
    chat.insert(tk.END, "\nБот: думает...\n")
    chat.config(state=tk.DISABLED)

    def run():
        ans = ask(q)

        def update():
            chat.config(state=tk.NORMAL)
            chat.delete(idx, tk.END)
            chat.config(state=tk.DISABLED)
            add("Бот", ans)

        window.after(0, update)

    threading.Thread(target=run, daemon=True).start()


entry.bind("<Return>", send)

if __name__ == "__main__":
    window.mainloop()
