from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtPrintSupport import QPrinter
from PySide6.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMainWindow, QMenu, QApplication, QFileDialog


class RasterImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.printer = QPrinter()
        self.scaleFactor = 0.0

        self.image_label = QLabel()
        self.image_label.setContentsMargins(0, 0, 0, 0)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_label.setScaledContents(True)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.scroll_area = QScrollArea()
        self.scroll_area.setContentsMargins(0, 0, 0, 0)
        self.scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setVisible(False)

        self.setCentralWidget(self.scroll_area)

        # Actions
        self.zoom_in_act = QAction("Zoom &In (25%)", self, shortcut="Ctrl++", enabled=False, triggered=self.zoomIn)
        self.zoom_out_act = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-", enabled=False, triggered=self.zoomOut)
        self.normal_size_act = QAction("&Normal Size", self, shortcut="Ctrl+N", enabled=False,
                                       triggered=self.normalSize)
        self.fit_to_window_act = QAction("&Fit to Window", self, enabled=False, checkable=True, shortcut="Ctrl+F",
                                         triggered=self.fitToWindow)
        self.save_act = QAction("&Save", self, shortcut="Ctrl+S", enabled=False, triggered=self.save_image)

        # Menu
        self.viewMenu = QMenu("&View", self)
        self.viewMenu.addAction(self.zoom_in_act)
        self.viewMenu.addAction(self.zoom_out_act)
        self.viewMenu.addAction(self.normal_size_act)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fit_to_window_act)

        self.menuBar().addAction(self.save_act)
        self.menuBar().addMenu(self.viewMenu)

        self.setWindowTitle("Image Viewer")
        self.resize(522, 522 + self.viewMenu.height())

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.image_label.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fit_to_window_act.isChecked()
        self.scroll_area.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()

    def updateActions(self):
        self.zoom_in_act.setEnabled(not self.fit_to_window_act.isChecked())
        self.zoom_out_act.setEnabled(not self.fit_to_window_act.isChecked())
        self.normal_size_act.setEnabled(not self.fit_to_window_act.isChecked())
        self.save_act.setEnabled(not self.fit_to_window_act.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.image_label.resize(self.image_label.pixmap().size() * self.scaleFactor)

        self.adjustScrollBar(self.scroll_area.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scroll_area.verticalScrollBar(), factor)

        self.zoom_in_act.setEnabled(self.scaleFactor < 10.0)
        self.zoom_out_act.setEnabled(self.scaleFactor > 0.1)

    @staticmethod
    def adjustScrollBar(scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                               + ((factor - 1) * scrollBar.pageStep() / 2)))

    def set_image(self, pix):
        self.image_label.setPixmap(pix)
        self.scaleFactor = 1.0
        self.scroll_area.setVisible(True)
        self.fit_to_window_act.setEnabled(True)
        self.updateActions()

        if not self.fit_to_window_act.isChecked():
            self.image_label.adjustSize()

        new_width, new_height = self.calculate_size_from_pixmap(pix)

        screen_size = QApplication.primaryScreen().size()
        max_initial_width = screen_size.width() * 0.9
        max_initial_height = screen_size.height() * 0.9
        if max_initial_width < new_width or max_initial_height < new_height:
            new_width = int(max_initial_width)
            new_height = int(max_initial_height)

        self.resize(new_width, new_height)

    def calculate_size_from_pixmap(self, pixmap):
        width = pixmap.width()
        height = pixmap.height()

        new_width = width + 12
        new_height = height + 12 + self.viewMenu.height()

        return new_width, new_height

    def save_image(self):
        dialog = QFileDialog(self)
        dialog.setViewMode(QFileDialog.ViewMode.Detail)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        dialog.setNameFilter("All files (*.*);;BMP (*.bmp);;GIF (*.gif);;JPEG ("
                             "*.jpeg);;JPG (*.jpg);;PBM (*.pbm);;PGM (*.pgm);;PNG (*.png);;PPM (*.ppm);"
                             ";XBM (*.xbm);;XPM (*.xpm)")

        if dialog.exec():
            file_path = dialog.selectedFiles()[0]
            pix = self.image_label.pixmap()
            pix.save(file_path, quality=100)
        else:
            return
