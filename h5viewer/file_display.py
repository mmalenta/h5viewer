from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

from typing import List

from file_details import FileDetails

class Legend(QWidget):

  def __init__(self):

    super().__init__()

    legend_layout = QVBoxLayout()
    legend_label = QLabel("Legend:")
    legend_layout.addWidget(legend_label)

    items = {
      "Group": "h5viewer/icons/subgroup_24dp.svg",
      "Data": "h5viewer/icons/data_24dp.svg",
      "Metadata": "h5viewer/icons/property_24dp.svg"
    }

    for label, icon in items.items():
      row_legend = QHBoxLayout()
      row_legend.setAlignment(Qt.AlignLeft)
      legend_icon = QLabel()
      legend_icon.setPixmap(QPixmap(icon).scaledToWidth(24))
      row_legend.addWidget(legend_icon)
      legend_label = QLabel(label)
      row_legend.addWidget(legend_label)
      legend_layout.addLayout(row_legend)

    self.setLayout(legend_layout)

class FileDisplay(QWidget):

  def __init__(self):

    super().__init__()

    self._display_layout = QVBoxLayout()
    self._legend = Legend()
    self._display_layout.addWidget(self._legend)
    self._display_layout.setAlignment(Qt.AlignTop)
    self.setLayout(self._display_layout)
    
  
  def add_files(self, files : List, reset : bool = False) -> None:

    for file in files:

      self._display_layout.addWidget(FileDetails(file))
        
    self._display_layout.setSpacing(0)
    self.setLayout(self._display_layout)

