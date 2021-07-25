from glob import glob
from os import path

from PyQt5.QtWidgets import QMainWindow, QScrollArea, QToolBar, QStatusBar, QWidget
from PyQt5.QtWidgets import QAction, QApplication, QFileDialog, QHBoxLayout, QScrollArea
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize, Qt

from file_display import FileDisplay
from file_tab import FileTab

class Viewer(QMainWindow):

  def __init__(self):

    super().__init__()

    screen = QApplication.instance().primaryScreen()

    # Set basic window properties
    self.setMinimumSize(QSize(1024, 768))
    self.setGeometry(0, 0, screen.size().width(), screen.size().height())
    self.setWindowTitle("HDF5 file viewer")    

    self._toolbar = QToolBar("Toolbar")
    self._toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
    self._toolbar.setIconSize(QSize(32,32))
    self._toolbar.setMovable(False)

    open_dir_action = QAction(QIcon("h5viewer/icons/folder_open_black_24dp.svg"), "Open Directory", self)
    open_dir_action.setStatusTip("Open new directory")
    open_dir_action.setShortcut("Ctrl+Shift+o")
    open_dir_action.triggered.connect(self._click_open_dir)
    self._toolbar.addAction(open_dir_action)

    open_file_action = QAction(QIcon("h5viewer/icons/article_black_24dp.svg"), "Open File", self)
    open_file_action.setStatusTip("Open new file")
    open_file_action.setShortcut("Ctrl+o")
    open_file_action.triggered.connect(self._click_open_file)
    self._toolbar.addAction(open_file_action)

    save_action = QAction(QIcon("h5viewer/icons/save_black_24dp.svg"), "Save file", self)
    save_action.setStatusTip("Save changes")
    save_action.setShortcut("Ctrl+s")
    save_action.triggered.connect(self._click)
    self._toolbar.addAction(save_action)

    self._toolbar.addSeparator()

    tab_action = QAction(QIcon("h5viewer/icons/tab_black_24dp.svg"), "New tab", self)
    tab_action.setStatusTip("Open new tab")
    tab_action.setShortcut("Ctrl+t")
    tab_action.triggered.connect(self._click)
    self._toolbar.addAction(tab_action)

    self.addToolBar(self._toolbar)

    main_layout = QHBoxLayout()
    self._file_display = FileDisplay()
    self._file_tab = FileTab()
    main_layout.addWidget(self._file_tab, stretch=4)
    #main_layout.addWidget(self._file_display, stretch=1)

    self._scroll = QScrollArea()
    self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
    self._scroll.setWidgetResizable(True)
    main_layout.addWidget(self._scroll, stretch = 1)
    self._scroll.setWidget(self._file_display)

    main_widget = QWidget()
    main_widget.setLayout(main_layout)

    self.setCentralWidget(main_widget)

    self.setStatusBar(QStatusBar(self))

    self.show()

  def _click(self):

    pass

  def _click_open_dir(self) -> None:

    open_dir_dialog = QFileDialog(self)
    open_dir_dialog.setWindowTitle("Open Directory")
    open_dir_dialog.setDirectory("/home/mateusz/Desktop/MeerTRAP/test_data/viewer_hdf5")
    open_dir_dialog.setFileMode(QFileDialog.Directory)
    if open_dir_dialog.exec():

      directory = open_dir_dialog.selectedFiles()
      files = glob(path.join(directory[0], "*.hdf5"))
      if files:
        print(files[:50])
        self._file_display.add_files(files[:30], True)

  def _click_open_file(self) -> None:

    open_file_dialog = QFileDialog(self)
    open_file_dialog.setWindowTitle("Open File")
    open_file_dialog.setFileMode(QFileDialog.ExistingFile)
    open_file_dialog.setNameFilter("HDF5 archives (*.hdf *.h5 *.hdf5)")
    if open_file_dialog.exec():
      files = open_file_dialog.selectedFiles()
      
      self._file_display.add_files(files)