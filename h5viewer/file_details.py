from PyQt5.QtGui import QIcon, QPixmap
import h5py as h5

from PyQt5.QtWidgets import QAction, QHBoxLayout, QPushButton, QWidget, QLabel, QVBoxLayout, QSpacerItem
from PyQt5.QtCore import Qt
from os import path

class PropertyLabel(QWidget):

  def __init__(self, indent, text, icon=None, subgroup=False):

    super().__init__()

    label_layout = QHBoxLayout()

    if indent != 0:

      if subgroup:
        label_layout.insertSpacing(0, (indent - 1) * 24)
      else:
        label_layout.insertSpacing(0, (indent) * 24)
        

      if subgroup:

        label_subgroup = QLabel()
        label_subgroup.setPixmap(QPixmap("h5viewer/icons/subgroup_24dp.svg").scaledToWidth(24))
        label_layout.addWidget(label_subgroup)

    if icon:

      label_icon = QLabel()
      label_icon.setPixmap(QPixmap(path.join("h5viewer/icons/", icon)).scaledToWidth(24))
      label_layout.addWidget(label_icon)

    label_text = QLabel(text)
    label_text.setIndent(6)
    label_layout.addWidget(label_text)

    label_layout.setContentsMargins(0, 0, 0, 0)
    label_layout.setSpacing(0)
    label_layout.setAlignment(Qt.AlignLeft)
    self.setLayout(label_layout)

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
    self._file_layout.setSpacing(5)
    self._file_layout.setContentsMargins(5, 10, 5, 0)
    self._file_layout.setAlignment(Qt.AlignLeft)

    self._full_file_name = file_name
    self._short_file_name = path.basename(file_name)
    self._file_label = QPushButton(self._short_file_name)
    self._file_label.setFlat(True)
    self._expand_button = QPushButton()
    self._expand_icon = QIcon("h5viewer/icons/add_black_24dp.svg")
    self._expand_button.setIcon(self._expand_icon)
    self._expand_button.setFixedWidth(24)
    self._expand_button.setFixedHeight(24)
    self._expand_button.setFlat(False)
    self._expand_button.clicked.connect(self._show_details)
    self._expand_button.setCheckable(True)
    self._file_layout.addWidget(self._expand_button)
    self._file_layout.addWidget(self._file_label)
    
    self._detail_layout.addLayout(self._file_layout)
    self.setLayout(self._detail_layout)

    self._details = Node("/")
    self._data_read = False
  
    self._details_widget = QWidget()
    self._details_widget_layout = QVBoxLayout()
    self._details_widget_layout.setContentsMargins(5, 10, 5, 0)
    self._details_widget_layout.setSpacing(0)
    self._details_widget.setLayout(self._details_widget_layout)

    self._detail_layout.addWidget(self._details_widget)
    self._details_widget.setVisible(False)

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

    if group._type == "data":
      data_label = PropertyLabel(indent, group._key, "data_24dp.svg")

      self._details_widget_layout.addWidget(data_label)
    else:
      group_label = PropertyLabel(indent, group._key, "group_24dp.svg", subgroup=True)
      self._details_widget_layout.addWidget(group_label)

    if group._metadata != None:

      meta_label = PropertyLabel(indent, "Metadata:")
      self._details_widget_layout.addWidget(meta_label)

      for key, value in group._metadata.items():

        meta_label = PropertyLabel(indent, key + ": " + str(value), "property_24dp.svg")
        self._details_widget_layout.addWidget(meta_label)

    for child in group._children:
      self._show_group(child, indent + 1)

  def _read_data(self):

    h5file = h5.File(self._full_file_name, 'r')
    self._read_key(h5file, '/', self._details)
    self._data_read = True

  def _read_key(self, h5file, parent_key, parent_node):

    if type(h5file[parent_key]) != h5._hl.dataset.Dataset:
      for key in h5file[parent_key].keys():
        current_key = parent_key + key + '/'

        child_node = Node(current_key)

        if type(h5file[current_key]) == h5._hl.dataset.Dataset:
          child_node._key = child_node._key[:-1]
          child_node._type = "data"
        else:
          child_node._type = "group"
        
        if len(h5file[current_key].attrs) != 0:
          metadata = {}
          for attr, value in h5file[current_key].attrs.items():

            metadata[attr] = value

          child_node._metadata = metadata

        parent_node._children.append(child_node)

        self._read_key(h5file, current_key, child_node)