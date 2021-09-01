from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtWidgets import QToolBox, QScrollArea
from PyQt5.QtCore import Qt

from typing import List

from file_details import FileDetails

class FileDisplay(QWidget):

  def __init__(self):

    super().__init__()

    self._display_layout = QVBoxLayout()
    self._display_layout.setAlignment(Qt.AlignTop)
    self.setLayout(self._display_layout)
    
  
  def add_files(self, files : List, reset : bool = False) -> None:

    for file in files:

      self._display_layout.addWidget(FileDetails(file))
        
    self._display_layout.setSpacing(0)
    self.setLayout(self._display_layout)

