import gi
import sys
import time
from pathlib import Path

import pandas as pd

from docling.document_converter import DocumentConverter

# You need to install GTK 4 for Python on your system
gi.require_version("Gtk", "4.0")
from gi.repository import Gio, Gtk, Gdk, GLib

ALLOWED_EXTENSIONS = ["pdf"]


class DragDropWindow(Gtk.ApplicationWindow):
    """Main application window."""

    doc_converter = None
    file_dialog = None
    target_folder = None
    cur_file = None

    def __init__(self, app):
        Gtk.ApplicationWindow.__init__(self, application=app)
        self.file_dialog = Gtk.FileDialog()

        # Set up window
        self.set_title("Docling")
        self.set_default_size(500, 360)

        # Main container
        self.box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.set_child(self.box)

        # Add the logo
        self.logo = Gtk.Picture.new_for_filename("./docs/assets/logo.png")
        self.box.append(self.logo)

        # Label to indicate drop area
        self.label = Gtk.Label(
            label="Drop files (PDF, ..) here to start processing",
            halign=Gtk.Align.CENTER,
        )
        self.box.append(self.label)

        # Add some padding at the bottom
        self.box.append(Gtk.Label(label="\n\n"))

        # Enable drag-and-drop
        file_drop_target = Gtk.DropTarget.new(Gio.File, Gdk.DragAction.COPY)
        self.add_controller(file_drop_target)
        file_drop_target.connect("drop", self.on_file_drop)

        # Drop Hover Effect
        file_drop_target.connect(
            "enter", lambda _target, _x, _y: self.box.add_css_class("overlay-drag-area")
        )
        file_drop_target.connect(
            "leave", lambda _target: self.box.remove_css_class("overlay-drag-area")
        )

        # Initialize converter
        self.doc_converter = DocumentConverter()

    def on_file_drop(self, target, value, _x, _y):
        """Handle user interaction."""
        print(f"Feeling the drop on {target} {_x} {_y}")
        self.box.remove_css_class("overlay-drag-area")
        if not isinstance(value, Gio.File):
            return False
        if self.file_dialog is None:
            return

        file_info = value.query_info("standard::content-type", 0, None)
        content_type = file_info.get_content_type()
        if content_type.startswith("application/pdf"):
            file_name = value.get_basename()
            file_paths = []

            # Convert URI to local path with proper unescaping
            for ext in ALLOWED_EXTENSIONS:
                if file_name.lower().endswith("." + ext):
                    file_paths.append(value.get_path())

            # File handling logic
            for path in file_paths:
                self.label.set_text(f"Processing file: {file_name}")
                self.cur_file = path
                self.file_dialog.select_folder(
                    parent=self,
                    cancellable=None,
                    callback=self.on_select_folder,
                )

        else:
            self.label.set_text("No valid files dropped")

    def on_select_folder(self, file_dialog, result):
        """Call back from Save window."""
        file = file_dialog.select_folder_finish(result)
        self.target_folder = file.get_path()
        self.convert_document(self.cur_file, self.target_folder)

    def convert_document(self, input_doc_path, output_dir):
        """Starts the Docling processing."""
        start_time = time.time()
        if self.doc_converter is None:
            return

        conv_res = self.doc_converter.convert(input_doc_path)
        doc_filename = conv_res.input.file.stem

        # Export tables
        for table_ix, table in enumerate(conv_res.document.tables):
            table_df: pd.DataFrame = table.export_to_dataframe()

            # Save the table as csv
            element_csv_filename = (
                output_dir + "/" + f"{doc_filename}-table-{table_ix + 1}.csv"
            )
            print(f"Saving CSV table to {element_csv_filename}")
            table_df.to_csv(element_csv_filename)

            # Save the table as html
            element_html_filename = (
                output_dir
                + "/"
                + f"{doc_filename}-table-{table_ix + 1}.element_html_filenameml"
            )
            print(f"Saving HTML table to {element_html_filename}")
            with open(element_html_filename, "w") as fp:
                fp.write(table.export_to_html(doc=conv_res.document))

        end_time = time.time() - start_time
        self.label.set_text(f"Document converted in {end_time:.2f} sec.")


class Application(Gtk.Application):
    def __init__(self):
        Gtk.Application.__init__(self, application_id="ch.datalets.DoclingApp")
        self.connect("activate", self.on_activate)

    def on_activate(self, app):
        self.win = DragDropWindow(app)
        self.win.present()


if __name__ == "__main__":
    app = Application()
    app.run(sys.argv)
