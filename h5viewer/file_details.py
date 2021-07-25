import h5py as h5

from PyQt5.QtWidgets import QAction, QHBoxLayout, QPushButton, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt
from os import path

class Node():

  def __init__(self, key):

    self._key = key
    self._type = None
    self._metadata = None
    self._children = []

class FileDetails(QWidget):

  def __init__(self, file_name):
    
    super().__init__()

    self._detail_layout = QVBoxLayout()
    self._detail_layout.setAlignment(Qt.AlignTop)
    
    self._file_layout = QHBoxLayout()

    self._full_file_name = file_name
    self._short_file_name = path.basename(file_name)
    self._file_label = QPushButton(self._short_file_name)
    self._file_label.setFlat(True)
    self._file_label.connect(self._print)
    self._expand_button = QPushButton("+")
    self._expand_button.clicked.connect(self._show_details)
    self._expand_button.setCheckable(True)
    self._file_layout.addWidget(self._file_label)
    self._file_layout.addWidget(self._expand_button)
    
    self._detail_layout.addLayout(self._file_layout)
    self.setLayout(self._detail_layout)

    self._details = Node("/")
    self._data_read = False
  
    self._details_widget = QWidget()
    self._details_widget_layout = QVBoxLayout()
    self._details_widget.setLayout(self._details_widget_layout)

    self._detail_layout.addWidget(self._details_widget)

  def _print(self):

    print("Hello")

  def _show_details(self, checked):

    if checked:

      if not self._data_read:

        self._read_data()

        self._show_group(self._details)

      self._details_widget.setVisible(True)

    else:
      self._details_widget.setVisible(False)

  def _show_group(self, group, indent=0):

    print(group._key)
    group_label = QLabel(indent * "    " + group._key)
    self._details_widget_layout.addWidget(group_label)

    if group._metadata != None:

      meta_label = QLabel(indent * "    " + "Metadata:")
      self._details_widget_layout.addWidget(meta_label)

      for key, value in group._metadata.items():

        meta_label = QLabel(indent * "    " + "* " + key + ": " + str(value))
        self._details_widget_layout.addWidget(meta_label)

    for child in group._children:
      # Use em dash
      self._show_group(child, indent + 1)

  def _read_data(self):

    h5file = h5.File(self._full_file_name, 'r')
    self._read_key(h5file, '/', self._details)
    self._data_read = True

    print(self._details._children[0]._key)

  def _read_key(self, h5file, parent_key, parent_node):

    if type(h5file[parent_key]) != h5._hl.dataset.Dataset:
      for key in h5file[parent_key].keys():
        current_key = parent_key + key + '/'

        child_node = Node(current_key)

        if type(h5file[current_key] == h5._hl.dataset.Dataset):
          child_node.type = "data"
        else:
          child_node.type = "group"
        
        if len(h5file[current_key].attrs) != 0:
          metadata = {}
          for attr, value in h5file[current_key].attrs.items():

            metadata[attr] = value

          child_node._metadata = metadata
          print(child_node._metadata)

        parent_node._children.append(child_node)

        self._read_key(h5file, current_key, child_node)