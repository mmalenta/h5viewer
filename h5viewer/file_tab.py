from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QTabWidget

class FileTab(QWidget):

  def __init__(self):

    super().__init__()

    tab_layout = QHBoxLayout()
    self._tab_area = QTabWidget()
    tab_layout.addWidget(self._tab_area)
    self.setLayout(tab_layout)
    