#!/usr/bin/env python3
"""
Full-featured GUI chat application for Ollama (local and cloud).

Uses tkinter (built-in) for the UI, integrates with the shared ollama_config
module, and supports streaming responses, web search, model selection,
conversation history, and save/clear functionality.

Run:
    python ui_chat_search.py
"""

import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ollama
from ollama_config import get_client, get_model

# ---------------------------------------------------------------------------
# Chat engine (non-UI logic)
# ---------------------------------------------------------------------------


class ChatEngine:
    """Handles all Ollama API interactions outside of the UI thread."""

    def __init__(self):
        self.client = get_client()
        self.messages: list[dict] = []
        self.system_prompt = ""

    def reset_conversation(self):
        """Clear message history and re-apply system prompt if set."""
        self.messages = []
        if self.system_prompt.strip():
            self.messages.append({"role": "system", "content": self.system_prompt.strip()})

    def set_system(self, prompt: str):
        """Replace the system prompt and rebuild message list."""
        self.system_prompt = prompt.strip()
        if self.messages and self.messages[0]["role"] == "system":
            if self.system_prompt:
                self.messages[0] = {"role": "system", "content": self.system_prompt}
            else:
                self.messages.pop(0)
        elif self.system_prompt:
            self.messages.insert(0, {"role": "system", "content": self.system_prompt})

    def add_user_message(self, text: str):
        self.messages.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str):
        self.messages.append({"role": "assistant", "content": text})

    def search_web(self, query: str, max_results: int = 3) -> list[str]:
        """Run a web search and return result texts."""
        results: list[str] = []
        try:
            response = ollama.web_search(query, max_results=max_results)
            for answer in response["results"]:
                results.append(answer.content)
        except Exception as exc:
            results.append(f"[Web search error: {exc}]")
        return results

    def chat_stream(self, model: str, temperature: float, callback):
        """Stream a chat response, calling `callback(token)` for each piece."""
        full = ""
        try:
            for chunk in self.client.chat(
                model,
                messages=self.messages,
                stream=True,
                options={"temperature": temperature},
            ):
                token = chunk.get("message", {}).get("content", "")
                if token:
                    full += token
                    callback(token)
        except Exception as exc:
            callback(f"\n[Error: {exc}]")
        return full


# ---------------------------------------------------------------------------
# GUI Application
# ---------------------------------------------------------------------------


class ChatApp:
    """Main tkinter application window."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Ollama Chat & Search")
        self.root.geometry("900x700")
        self.root.minsize(600, 400)

        self.engine = ChatEngine()
        self._streaming = False

        self._setup_styles()
        self._build_ui()
        self._setup_tags()

        self.engine.reset_conversation()

    # --- Style & Tags -------------------------------------------------------

    def _setup_styles(self):
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Send.TButton", font=("Helvetica", 11, "bold"))
        style.configure("Status.TLabel", foreground="#555555")

    def _setup_tags(self):
        d = self.chat_display
        d.tag_configure("user", foreground="#1a6e9e",
                        font=("Helvetica", 11, "bold"),
                        lmargin1=20, lmargin2=20, spacing3=6)
        d.tag_configure("assistant", foreground="#2d2d2d",
                        font=("Helvetica", 11),
                        lmargin1=20, lmargin2=20, spacing3=6)
        d.tag_configure("system_msg", foreground="#888888",
                        font=("Helvetica", 10, "italic"),
                        lmargin1=20, lmargin2=20, spacing3=4)
        d.tag_configure("error", foreground="#cc3333",
                        font=("Helvetica", 10, "italic"),
                        lmargin1=20, lmargin2=20)
        d.tag_configure("search_header", foreground="#8b4513",
                        font=("Helvetica", 10, "bold"),
                        lmargin1=20, lmargin2=20, spacing3=4)
        d.tag_configure("search_result", foreground="#555555",
                        font=("Helvetica", 10, "italic"),
                        lmargin1=30, lmargin2=30)

    # --- Build UI -----------------------------------------------------------

    def _build_ui(self):
        # -- Top settings bar --
        settings_frame = ttk.Frame(self.root, padding=(8, 4))
        settings_frame.pack(fill=tk.X)

        # Model
        ttk.Label(settings_frame, text="Model:").pack(side=tk.LEFT, padx=(0, 2))
        self.model_var = tk.StringVar(value=get_model())
        self.model_entry = ttk.Entry(settings_frame, textvariable=self.model_var, width=22)
        self.model_entry.pack(side=tk.LEFT, padx=(0, 10))

        # Temperature
        ttk.Label(settings_frame, text="Temp:").pack(side=tk.LEFT, padx=(0, 2))
        self.temp_var = tk.DoubleVar(value=0.7)
        temp_scale = ttk.Scale(
            settings_frame, from_=0.0, to=2.0, variable=self.temp_var,
            orient=tk.HORIZONTAL, length=100, command=self._on_temp_change,
        )
        temp_scale.pack(side=tk.LEFT, padx=(0, 2))
        self.temp_label = ttk.Label(settings_frame, text="0.7", width=4)
        self.temp_label.pack(side=tk.LEFT, padx=(0, 10))

        # Web search toggle
        self.search_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings_frame, text="Web Search",
                        variable=self.search_var).pack(side=tk.LEFT, padx=(0, 10))

        # System prompt button
        ttk.Button(settings_frame, text="System Prompt",
                   command=self._open_system_prompt).pack(side=tk.LEFT, padx=(0, 10))

        # Cloud indicator
        self.cloud_label = ttk.Label(settings_frame, text="")
        self.cloud_label.pack(side=tk.LEFT, padx=(0, 10))
        self._update_cloud_indicator()

        # -- Chat display --
        display_frame = ttk.Frame(self.root, padding=(8, 4))
        display_frame.pack(fill=tk.BOTH, expand=True)

        self.chat_display = tk.Text(
            display_frame, wrap=tk.WORD, state=tk.DISABLED,
            borderwidth=0, padx=4, pady=4, font=("Helvetica", 11),
        )
        self.chat_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        chat_scroll = ttk.Scrollbar(display_frame, command=self.chat_display.yview)
        chat_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_display.configure(yscrollcommand=chat_scroll.set)

        # -- Input area --
        input_frame = ttk.Frame(self.root, padding=(8, 4))
        input_frame.pack(fill=tk.X)

        self.input_text = tk.Text(
            input_frame, height=3, wrap=tk.WORD, font=("Helvetica", 11),
            borderwidth=1, relief=tk.SOLID,
        )
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.input_text.bind("<Return>", self._on_enter_key)
        self.input_text.bind("<Shift-Return>", self._on_shift_enter)

        button_panel = ttk.Frame(input_frame)
        button_panel.pack(side=tk.RIGHT, fill=tk.Y)

        self.send_btn = ttk.Button(button_panel, text="Send", style="Send.TButton",
                                   command=self._send_message)
        self.send_btn.pack(fill=tk.X, pady=(0, 2))

        ttk.Button(button_panel, text="Clear Chat", command=self._clear_chat).pack(
            fill=tk.X, pady=(0, 2))

        ttk.Button(button_panel, text="Save Chat", command=self._save_chat).pack(
            fill=tk.X)

        # -- Status bar --
        self.status_var = tk.StringVar(value="Ready.")
        status_bar = ttk.Label(
            self.root, textvariable=self.status_var, style="Status.TLabel",
            relief=tk.SUNKEN, padding=(4, 1),
        )
        status_bar.pack(fill=tk.X)

        # Focus on input on start
        self.input_text.focus_set()

    # --- Display helpers ----------------------------------------------------

    def _write_line(self, text: str, tag: str):
        """Insert a labeled, styled line into the chat display."""
        labels = {"user": "You", "assistant": "Assistant",
                  "search_header": "Web Search", "system_msg": "System"}
        label = labels.get(tag, tag)
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{label}: {text}\n", tag)
        self.chat_display.configure(state=tk.DISABLED)
        self._scroll_to_bottom()

    def _append_token(self, token: str):
        """Append a streaming token to the assistant's response."""
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, token, "assistant")
        self.chat_display.configure(state=tk.DISABLED)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        self.chat_display.yview_moveto(1.0)

    def _update_cloud_indicator(self):
        if os.environ.get("CLOUD"):
            self.cloud_label.configure(text="[cloud]", foreground="#cc6600")
        else:
            self.cloud_label.configure(text="[local]", foreground="#339933")

    def _on_temp_change(self, *_):
        self.temp_label.configure(text=f"{self.temp_var.get():.1f}")

    # --- Keyboard handlers --------------------------------------------------

    def _on_enter_key(self, event):
        """Enter sends the message; Shift-Enter inserts a newline."""
        self._send_message()
        return "break"

    def _on_shift_enter(self, event):
        """Allow Shift-Enter to insert a literal newline."""
        self.input_text.insert(tk.INSERT, "\n")
        return "break"

    # --- System prompt dialog -----------------------------------------------

    def _open_system_prompt(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("System Prompt")
        dialog.geometry("550x350")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Set a system prompt to guide the assistant's behavior:",
                  padding=(8, 4)).pack(anchor=tk.W)

        text = tk.Text(dialog, wrap=tk.WORD, font=("Helvetica", 11),
                       borderwidth=1, relief=tk.SOLID)
        text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))
        if self.engine.system_prompt:
            text.insert("1.0", self.engine.system_prompt)

        btn_frame = ttk.Frame(dialog, padding=(8, 4))
        btn_frame.pack(fill=tk.X)

        def apply():
            self.engine.set_system(text.get("1.0", "end-1c"))
            self.chat_display.configure(state=tk.NORMAL)
            if self.engine.system_prompt:
                self.chat_display.insert(tk.END, "System prompt updated.\n\n", "system_msg")
            else:
                self.chat_display.insert(tk.END, "System prompt cleared.\n\n", "system_msg")
            self.chat_display.configure(state=tk.DISABLED)
            dialog.destroy()

        ttk.Button(btn_frame, text="Apply", command=apply).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)

    # --- Send / stream ------------------------------------------------------

    def _send_message(self):
        if self._streaming:
            return

        text = self.input_text.get("1.0", "end-1c").strip()
        if not text:
            return

        self.input_text.delete("1.0", tk.END)
        self._write_line(text, "user")
        self.engine.add_user_message(text)

        self._streaming = True
        self.send_btn.configure(state=tk.DISABLED)
        self.status_var.set("Thinking...")

        if self.search_var.get():
            self._run_web_search_then_chat(text)
        else:
            self._start_chat_stream()

    def _run_web_search_then_chat(self, user_query: str):
        """Run web search in background thread, then stream the chat response."""

        def task():
            self.status_var.set("Searching web...")
            search_results = self.engine.search_web(user_query, max_results=3)

            # Display results on UI thread
            self.root.after(0, lambda: self._display_search_results(search_results))

            # Augment the last user message with search results as context
            if search_results:
                context = (
                    "The following are web search results for the user's query. "
                    "Use them to provide a well-informed answer.\n\n"
                    + "\n---\n".join(search_results)
                )
                last_msg = self.engine.messages[-1]
                last_msg["content"] = (
                    f"{last_msg['content']}\n\n[Web search results]:\n{context}"
                )

            self.root.after(0, self._start_chat_stream)

        threading.Thread(target=task, daemon=True).start()

    def _display_search_results(self, results: list[str]):
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "Web Search Results:\n", "search_header")
        for i, r in enumerate(results, 1):
            summary = r[:300] + ("..." if len(r) > 300 else "")
            self.chat_display.insert(tk.END, f"  [{i}] {summary}\n", "search_result")
        self.chat_display.insert(tk.END, "\n", "search_result")
        self.chat_display.configure(state=tk.DISABLED)
        self._scroll_to_bottom()

    def _start_chat_stream(self):
        """Begin streaming the model response."""
        self.status_var.set("Generating response...")

        # Write the "Assistant:" label
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "Assistant: ", "assistant")
        self.chat_display.configure(state=tk.DISABLED)
        self._scroll_to_bottom()

        model = self.model_var.get().strip() or get_model()
        temperature = self.temp_var.get()

        def on_token(token: str):
            self.root.after(0, lambda t=token: self._handle_token(t))

        def stream_thread():
            full = self.engine.chat_stream(model, temperature, on_token)
            self.root.after(0, lambda f=full: self._on_stream_done(f))

        threading.Thread(target=stream_thread, daemon=True).start()

    def _handle_token(self, token: str):
        self._append_token(token)

    def _on_stream_done(self, full_response: str):
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "\n\n", "assistant")
        self.chat_display.configure(state=tk.DISABLED)
        self._scroll_to_bottom()

        self.engine.add_assistant_message(full_response)

        self._streaming = False
        self.send_btn.configure(state=tk.NORMAL)
        self.status_var.set("Ready.")
        self.input_text.focus_set()

    # --- Clear / Save -------------------------------------------------------

    def _clear_chat(self):
        if self._streaming:
            return
        self.engine.reset_conversation()
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.configure(state=tk.DISABLED)
        self.status_var.set("Chat cleared.")

    def _save_chat(self):
        content = self.chat_display.get("1.0", "end-1c")
        if not content.strip():
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("Markdown", "*.md"), ("All files", "*.*")],
            initialfile="chat_log.txt",
        )
        if file_path:
            Path(file_path).write_text(content)
            self.status_var.set(f"Saved to {file_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    root = tk.Tk()
    ChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
