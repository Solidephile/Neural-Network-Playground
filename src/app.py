import sys
import numpy as np
import pickle
import gzip
import webbrowser

import core

from PyQt5.QtGui import QStandardItemModel, QIcon
from PyQt5.QtWidgets import (
    QWidget,
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QHeaderView,
    QAbstractItemView,
    QTableWidgetItem,
    QFileDialog,
)
from PyQt5.QtCore import QThread, Qt, pyqtSignal

from qfluentwidgets.common import toggleTheme
from qfluentwidgets.components.widgets import (
    SubtitleLabel,
    BodyLabel,
    LargeTitleLabel,
    LineEdit,
    PrimaryPushButton,
    PushButton,
    TableWidget,
    TableView,
    ComboBox,
    SpinBox,
    DoubleSpinBox,
    ProgressBar,
    InfoBar,
    HorizontalSeparator,
    HeaderCardWidget,
)
from qfluentwidgets.components.navigation import NavigationItemPosition
from qfluentwidgets.window import FluentWindow
from qfluentwidgets import FluentIcon as FIF


# Dataset thread (deprecated, may implement multithread reading later)
# class DatasetThread(QThread):
#     value_change = pyqtSignal(int)
#     finished = pyqtSignal()

#     def __init__(self, parent):
#         super(DatasetThread, self).__init__()
#         self.parent = parent

#     def run(self):
#         self.parent.X, self.parent.Y, self.parent.X_test, self.parent.Y_test = dataset.create_data_mnist(
#             "mnist_images", self.value_change
#         )
#         self.parent.X, self.parent.Y, self.parent.X_test, self.parent.Y_test = dataset.preprocess_dataset(
#             self.parent.X, self.parent.Y, self.parent.X_test, self.parent.Y_test
#         )
#         self.finished.emit()


# Trainging thread
class TrainingThread(QThread):
    value_change = pyqtSignal(int)
    progress_data = pyqtSignal(float, float, float, float, float)
    validation_data = pyqtSignal(float, float)
    finished = pyqtSignal()
    stopped = pyqtSignal()

    def __init__(self, parent, epoch, batch_size):
        super(TrainingThread, self).__init__()
        self.parent = parent
        self.is_running = True
        self.epoch = epoch
        self.batch_size = batch_size

    def run(self):
        self.parent.model.train(
            self.parent.X,
            self.parent.Y,
            validation_data=(self.parent.X_test, self.parent.Y_test),
            epochs=self.epoch,
            batch_size=self.batch_size,
            print_every=100,
            progress_callback=self.value_change,
            progress_data_callback=self.progress_data,
            stop_callback=self.check_running,
            validation_callback=self.validation_data,
        )
        if self.is_running:
            self.finished.emit()

    def check_running(self):
        return self.is_running

    def stop(self):
        self.value_change.emit(0)
        self.is_running = False
        self.stopped.emit()


# Writing board
class Board(QWidget):
    def __init__(self, pixel_size):
        super().__init__()
        self.pixel_size = pixel_size
        self.init_UI()
        self.mouse_status = 0

    def init_UI(self):
        self.layout = QGridLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.pixels = []

        for i in range(28):
            self.pixels.append([])
            for j in range(28):
                pixel = Pixel(self.pixel_size)
                self.pixels[i].append(pixel)
                self.layout.addWidget(pixel, j, i)

        self.setLayout(self.layout)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_status = 1
        elif event.button() == Qt.RightButton:
            self.mouse_status = -1
        self.paint(event)

    def mouseReleaseEvent(self, event):
        self.mouse_status = 0

    def mouseMoveEvent(self, event):
        self.paint(event)

    def paint(self, event):
        x_pos = event.pos().x()
        y_pos = event.pos().y()
        pos_1 = [x_pos, y_pos]
        for x in range(28):
            for y in range(28):
                pos_2 = [
                    x * self.pixel_size + self.pixel_size / 2,
                    y * self.pixel_size + self.pixel_size / 2,
                ]
                if np.linalg.norm(np.array(pos_1) - np.array(pos_2)) < 1.5 * self.pixel_size:
                    if self.mouse_status == 1:
                        self.pixels[x][y].painted()
                    elif self.mouse_status == -1:
                        self.pixels[x][y].erased()

    def clear(self):
        for y in range(28):
            for x in range(28):
                self.pixels[x][y].erased()

    def get_data(self):
        result = np.array([])
        for y in range(28):
            for x in range(28):
                result = np.append(result, self.pixels[x][y].status)

        result = result.astype(np.float32)

        return result


# A single pixel in writing board
class Pixel(QLabel):
    def __init__(self, size):
        super().__init__()
        self.setFixedSize(size, size)
        self.status = -1
        self.setStyleSheet("QLabel{background:black;border-radius:0;}")

    def painted(self):
        self.setStyleSheet("""QLabel{background:white;border-radius:0;}""")
        self.status = 1

    def erased(self):
        self.setStyleSheet("""QLabel{background:black;border-radius:0;}""")
        self.status = -1


# Subinterface
class SubInterface(QWidget):
    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)

        self.setObjectName(text.replace(" ", "-"))


# Predict Interface
class PredictInterface(SubInterface):
    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent, text=text)
        self.parent = parent

        # Init layouts
        self.main_layout = QHBoxLayout()
        self.button_layout = QHBoxLayout()
        self.load_model_layout = QHBoxLayout()
        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        # Init paint board
        self.board = Board(16)

        # Init Labels on the left
        self.predict_result = SubtitleLabel("Predict result:")

        self.load_model_label = BodyLabel("Loaded Model:")

        # Init model path lineedit
        self.model_path = LineEdit(self)
        self.model_path.setPlaceholderText("None")

        # Init buttons
        self.predict_button = PrimaryPushButton("Predict", self)
        self.predict_button.clicked.connect(self.predict)

        self.clear_button = PushButton("Clear", self)
        self.clear_button.clicked.connect(self.board.clear)

        self.browse_button = PushButton("Browse", self)
        self.browse_button.clicked.connect(self.parent.load_model)

        # Init confidence table
        self.confidence_table = TableWidget(self)
        self.confidence_table.setRowCount(10)
        self.confidence_table.setColumnCount(2)
        self.confidence_table.verticalHeader().hide()
        self.confidence_table.setHorizontalHeaderLabels(["Numbers", "Confidences"])
        self.confidence_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.confidence_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        for i in range(10):
            item = QTableWidgetItem(str(i))
            self.confidence_table.setItem(i, 0, item)

        # Set layouts
        self.button_layout.addWidget(self.predict_button)
        self.button_layout.addWidget(self.clear_button)

        self.load_model_layout.addWidget(self.load_model_label)
        self.load_model_layout.addWidget(self.model_path)
        self.load_model_layout.addWidget(self.browse_button)

        self.left_layout.addWidget(self.board)
        self.left_layout.addWidget(self.predict_result)
        self.left_layout.addLayout(self.load_model_layout)
        self.left_layout.addLayout(self.button_layout)

        self.right_layout.addWidget(self.confidence_table)

        self.main_layout.addLayout(self.left_layout)
        self.main_layout.addLayout(self.right_layout)

        # Set stretch ratio
        self.left_layout.setStretch(0, 0)
        self.main_layout.setStretch(0, 0)
        self.main_layout.setStretch(1, 2)

        self.setLayout(self.main_layout)

    def predict(self):
        if self.parent.model is None:
            self.parent.update_model()

        data = self.board.get_data()

        # Predict on the image
        confidences = self.parent.model.predict(data)

        # Get prediction instead of confidence levels
        predictions = self.parent.model.output_layer_activation.predictions(confidences)

        self.predict_result.setText(f"Predict result: {predictions[0]}")

        # Update confidences table
        for i in range(10):
            item = QTableWidgetItem(np.format_float_scientific(confidences[0][i], precision=2, unique=False))
            self.confidence_table.setItem(i, 1, item)


# Train Interface
class TrainInterface(SubInterface):
    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent, text=text)
        self.parent = parent

        # Init params
        self.neuron_count_widgets = []
        self.activation_widgets = []
        self.activations = ["ReLU", "Sigmoid", "Linear"]
        self.optimizers = ["SGD", "AdaGrad", "RMSprop", "Adam"]
        self.learning_rate_values = ["1", "0.1", "0.01", "1e-3", "1e-4", "1e-5"]
        self.decay_values = ["0", "0.1", "0.01", "1e-3", "1e-4", "1e-5", "1e-6", "1e-7"]

        # Init layouts
        self.main_layout = QVBoxLayout()
        self.operation_layout = QHBoxLayout()
        self.model_params_layout = QGridLayout()
        self.train_params_layout = QGridLayout()
        self.progress_bar_layout = QVBoxLayout()
        self.train_info_layout = QHBoxLayout()
        self.validation_info_layout = QHBoxLayout()
        self.train_info_card = TrainInfoCard(self)

        # Train button
        self.train_button = PrimaryPushButton("Train", icon=FIF.PLAY_SOLID)
        self.train_button.clicked.connect(self.train)

        # Operation buttons
        self.load_model_button = PushButton("Load Model", icon=FIF.FOLDER)
        self.load_model_button.clicked.connect(parent.load_model)

        self.save_model_button = PushButton("Save Model", icon=FIF.SAVE)
        self.save_model_button.clicked.connect(parent.save_model)

        self.add_layer_button = PushButton("Add Layer", icon=FIF.ADD)
        self.add_layer_button.clicked.connect(lambda: self.add_layer(64, "ReLU"))

        self.remove_layer_button = PushButton("Remove Layer", icon=FIF.REMOVE)
        self.remove_layer_button.clicked.connect(self.remove_layer)

        self.load_dataset_button = PushButton("Load dataset", icon=FIF.BOOK_SHELF)
        self.load_dataset_button.clicked.connect(self.load_dataset)

        # Model param widgets
        self.optimizer_label = BodyLabel("Optimizer")
        self.learning_rate_label = BodyLabel("Learning Rate")
        self.decay_label = BodyLabel("Decay")
        self.momentum_label = BodyLabel("Momentum")

        self.optimizer_combobox = ComboBox(self)
        self.optimizer_combobox.addItems(self.optimizers)
        self.optimizer_combobox.setCurrentText("Adam")
        self.optimizer_combobox.currentTextChanged.connect(self.parent.state_toggle)

        self.learning_rate_combobox = ComboBox(self)
        self.learning_rate_combobox.addItems(self.learning_rate_values)
        self.learning_rate_combobox.setCurrentText("1e-3")
        self.learning_rate_combobox.currentTextChanged.connect(self.parent.state_toggle)

        self.decay_combobox = ComboBox(self)
        self.decay_combobox.addItems(self.decay_values)
        self.decay_combobox.setCurrentText("0")
        self.decay_combobox.currentTextChanged.connect(self.parent.state_toggle)

        self.momentum_spinbox = DoubleSpinBox(self)
        self.momentum_spinbox.setRange(0, 1)
        self.momentum_spinbox.setValue(0.9)
        self.momentum_spinbox.setSingleStep(0.1)
        self.momentum_spinbox.valueChanged.connect(self.parent.state_toggle)

        # Train param widgets
        self.epoch_label = BodyLabel("Epoch")
        self.batch_size_label = BodyLabel("Batch Size")

        self.epoch_spinbox = SpinBox(self)
        self.epoch_spinbox.setRange(1, 100)
        self.epoch_spinbox.setValue(10)

        self.batch_size_spinbox = SpinBox(self)
        self.batch_size_spinbox.setRange(1, 1024)
        self.batch_size_spinbox.setValue(128)

        # Layer info
        self.layer_info_view = TableView()
        self.layer_data = QStandardItemModel()
        self.layer_data.setHorizontalHeaderLabels(["Neuron Count", "Activation Function"])
        self.layer_info_view.setModel(self.layer_data)
        self.layer_info_view.horizontalHeader().setStretchLastSection(True)
        self.layer_info_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.layer_info_view.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        self.add_layer(64, "ReLU")
        self.add_layer(64, "ReLU")

        # Progress bar
        self.progress_bar = ProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignRight)
        self.progress_bar_label = BodyLabel("          ")

        # Train info widgets
        self.accuracy_label = BodyLabel("Accuracy:")
        self.loss_label = BodyLabel("Loss:")
        self.current_learning_rate_label = BodyLabel("Current Learning Rate:")

        # Validation info widgets
        self.validation_accuracy_label = BodyLabel("Validation Accuracy:")
        self.validation_loss_label = BodyLabel("Validation Loss:")

        # Add widgets
        self.operation_layout.addWidget(self.load_dataset_button)
        self.operation_layout.addWidget(self.load_model_button)
        self.operation_layout.addWidget(self.save_model_button)
        self.operation_layout.addWidget(self.add_layer_button)
        self.operation_layout.addWidget(self.remove_layer_button)

        self.model_params_layout.addWidget(self.optimizer_label, 0, 0)
        self.model_params_layout.addWidget(self.optimizer_combobox, 1, 0)
        self.model_params_layout.addWidget(self.learning_rate_label, 0, 1)
        self.model_params_layout.addWidget(self.learning_rate_combobox, 1, 1)
        self.model_params_layout.addWidget(self.decay_label, 0, 2)
        self.model_params_layout.addWidget(self.decay_combobox, 1, 2)
        self.model_params_layout.addWidget(self.momentum_label, 0, 3)
        self.model_params_layout.addWidget(self.momentum_spinbox, 1, 3)

        self.train_params_layout.addWidget(self.epoch_label, 0, 0)
        self.train_params_layout.addWidget(self.epoch_spinbox, 1, 0)
        self.train_params_layout.addWidget(self.batch_size_label, 0, 1)
        self.train_params_layout.addWidget(self.batch_size_spinbox, 1, 1)

        self.progress_bar_layout.addWidget(self.progress_bar_label)
        self.progress_bar_layout.addWidget(self.progress_bar)
        self.progress_bar_layout.addWidget(HorizontalSeparator(self))

        self.train_info_layout.addWidget(self.accuracy_label)
        self.train_info_layout.addWidget(self.loss_label)
        self.train_info_layout.addWidget(self.current_learning_rate_label)

        self.validation_info_layout.addWidget(self.validation_accuracy_label)
        self.validation_info_layout.addWidget(self.validation_loss_label)
        self.validation_info_layout.addWidget(BodyLabel(self))

        self.train_info_card.vBoxLayout.addLayout(self.train_info_layout)
        self.train_info_card.vBoxLayout.addLayout(self.validation_info_layout)

        # Layout settings
        self.model_params_layout.setSpacing(10)
        self.progress_bar_layout.setSpacing(10)

        # Add Layouts
        self.main_layout.addLayout(self.operation_layout)
        self.main_layout.addWidget(self.train_button)
        self.main_layout.addLayout(self.model_params_layout)
        self.main_layout.addLayout(self.train_params_layout)
        self.main_layout.addWidget(self.layer_info_view)
        self.main_layout.addLayout(self.progress_bar_layout)
        self.main_layout.addWidget(self.train_info_card)

        self.setLayout(self.main_layout)

    def add_layer(self, neurons, activation):
        self.add_layer_button.setDisabled(False)
        self.remove_layer_button.setDisabled(False)

        index = self.layer_data.rowCount()

        self.layer_data.insertRow(index)
        self.layer_data.setHeaderData(index, Qt.Orientation.Vertical, f"Hidden Layer{index + 1}")

        w = SpinBox(self)
        w.setRange(3, 1024)
        w.setValue(neurons)
        w.valueChanged.connect(self.parent.state_toggle)
        self.neuron_count_widgets.append(w)
        self.layer_info_view.setIndexWidget(self.layer_data.index(index, 0), w)

        w = ComboBox(self)
        w.addItems(self.activations)
        w.setCurrentText(activation)
        w.currentTextChanged.connect(self.parent.state_toggle)
        self.activation_widgets.append(w)
        self.layer_info_view.setIndexWidget(self.layer_data.index(index, 1), w)

        if index + 1 >= 10:
            self.add_layer_button.setDisabled(True)
            return

    def remove_layer(self):
        self.add_layer_button.setDisabled(False)
        self.remove_layer_button.setDisabled(False)

        index = self.layer_data.rowCount()
        self.layer_data.removeRow(index - 1)
        self.neuron_count_widgets.pop()
        self.activation_widgets.pop()

        if index - 1 == 1:
            self.remove_layer_button.setDisabled(True)
            return

    def load_dataset(self):
        if not hasattr(self.parent, "X"):
            self.train_button.setDisabled(True)
            self.load_dataset_button.setDisabled(True)

            file = gzip.GzipFile("mnist_data/train_images.gz", "rb")
            self.parent.X = pickle.load(file)
            file.close()
            file = gzip.GzipFile("mnist_data/test_images.gz", "rb")
            self.parent.X_test = pickle.load(file)
            file.close()
            file = gzip.GzipFile("mnist_data/train_labels.gz", "rb")
            self.parent.Y = pickle.load(file)
            file.close()
            file = gzip.GzipFile("mnist_data/test_labels.gz", "rb")
            self.parent.Y_test = pickle.load(file)
            file.close()

            w = InfoBar.success(
                title="Success",
                content="The mnist dataset have been loaded.",
                isClosable=False,
                duration=2000,
                parent=self.parent,
            )
            w.show()
            self.train_button.setDisabled(False)
            self.load_dataset_button.setDisabled(False)

            # deprecated method
            # self.dataset_thread = DatasetThread(self.parent)
            # self.dataset_thread.value_change.connect(self.progress_bar.setValue)
            # self.dataset_thread.finished.connect(self.dataset_loading_finished)
            # self.dataset_thread.start()

        else:
            w = InfoBar.warning(
                title="Notice",
                content="The mnist dataset have been loaded.",
                isClosable=False,
                duration=2000,
                parent=self.parent,
            )
            w.show()

    def train(self):
        if hasattr(self.parent, "X"):
            self.parent.is_toggled = False
            self.parent.update_model()

            self.progress_bar.setValue(0)
            self.progress_bar.setRange(0, self.epoch_spinbox.value())
            self.progress_bar_label.setText("Training......      Epoch: 0    ")

            self.load_model_button.setDisabled(True)
            self.save_model_button.setDisabled(True)

            self.training_thread = TrainingThread(
                self.parent, self.epoch_spinbox.value(), self.batch_size_spinbox.value()
            )
            self.training_thread.value_change.connect(self.update_training_progress)
            self.training_thread.progress_data.connect(self.update_training_info)
            self.training_thread.finished.connect(self.training_finished)
            self.training_thread.stopped.connect(self.training_stopped)
            self.training_thread.validation_data.connect(self.update_validation_info)

            self.train_button.setText("Stop Training")
            self.train_button.setIcon(FIF.CLOSE)
            self.train_button.clicked.disconnect()
            self.train_button.clicked.connect(self.training_thread.stop)

            self.training_thread.start()
        else:
            w = InfoBar.error(
                title="Error",
                content="You haven't loaded mnist dataset!",
                isClosable=False,
                duration=2000,
                parent=self.parent,
            )
            w.show()

    def update_training_progress(self, value):
        self.progress_bar.setValue(value)
        self.progress_bar_label.setText(f"Training......      Epoch: {value}    ")

    def update_training_info(self, accuracy, loss, data_loss, reg_loss, current_learning_rate):
        self.accuracy_label.setText(f"Accuracy: {accuracy:.3f}")
        self.loss_label.setText(f"Loss: {loss:.3f}")
        self.current_learning_rate_label.setText(
            f"Current Learning Rate: {np.format_float_scientific(current_learning_rate, precision=2, unique=False)}"
        )

    def update_validation_info(self, accuracy, loss):
        self.validation_accuracy_label.setText(f"Validation Accuracy: {accuracy:.3f}")
        self.validation_loss_label.setText(f"Validation Loss: {loss:.3f}")

    def training_stopped(self):
        self.training_finished()
        self.progress_bar_label.setText("Training stopped.    ")

    def training_finished(self):
        self.load_model_button.setDisabled(False)
        self.save_model_button.setDisabled(False)
        self.train_button.setText("Train")
        self.train_button.setIcon(FIF.PLAY_SOLID)
        self.train_button.clicked.disconnect()
        self.train_button.clicked.connect(self.train)
        self.progress_bar_label.setText("Training finished.    ")


# About Interface
class AboutInterface(SubInterface):
    def __init__(self, name, parent=None):
        super().__init__(name, parent)
        self.parent = parent
        self.setContentsMargins(5, 5, 5, 5)
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.title = LargeTitleLabel("About")

        self.content = BodyLabel(
            """
This is an application for building and training a handwritten digit recognition neural network. You can conveniently tweak varuious parameters to build and train your own neural network. The trained model can be saved and loaded for later use.

The neural network is a simple Artificial Neural Network (ANN) based on book 'Neural Networks from Scratch in Python' by Harrison Kinsley & Daniel Kukieła.

The application interface is built using PyQT5 and QFluentWidgets, a fluent design component library based on PyQT.

The application is only for educational purposes and should not be used for any malicious purposes.

The source code of the application is available on GitHub.

            """
        )
        self.content.setIndent(-1)
        self.content.setWordWrap(True)
        self.content.setAlignment(Qt.AlignTop)

        self.hyperlink_button = PrimaryPushButton(text="GitHub", icon=FIF.GITHUB, parent=self)
        self.hyperlink_button.clicked.connect(lambda: webbrowser.open("https://github.com/Solidephile/Neural-Network-Playground"))

        self.main_layout.addWidget(self.title)
        self.main_layout.addWidget(self.content)
        self.main_layout.addWidget(self.hyperlink_button)
        self.main_layout.setStretch(0, 0)
        self.main_layout.setStretch(1, 1)


# Train info card
class TrainInfoCard(HeaderCardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Training Info")
        self.vBoxLayout = QVBoxLayout()
        self.viewLayout.addLayout(self.vBoxLayout)
        self.viewLayout.setContentsMargins(10, 5, 5, 5)
        self.headerLayout.setContentsMargins(10, 5, 5, 5)


# Main Window
class MainWindow(FluentWindow):
    def __init__(self):
        super().__init__()

        # Init model variables
        self.model = None
        self.loaded_model_path = None
        self.neuron_counts = []
        self.activations = []

        # Init state flags
        self.is_toggled = False

        # Init
        self.init_window()
        self.init_ui()

    def init_window(self):
        # Init main window
        self.resize(720, 540)
        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)
        self.setMinimumWidth(720)
        self.setWindowIcon(QIcon("resources/icon.ico"))
        self.setWindowTitle("Neural Network Playground - by Solidephile")

    def init_ui(self):
        # Init predict interface
        self.predict_interface = PredictInterface("predict", self)
        self.addSubInterface(self.predict_interface, icon=FIF.ROBOT, text="PREDICT")

        # Init train interface
        self.train_interface = TrainInterface("train", self)
        self.addSubInterface(self.train_interface, icon=FIF.SPEED_HIGH, text="TRAIN")

        # Init about interface
        self.about_interface = AboutInterface("about", self)
        self.addSubInterface(self.about_interface, icon=FIF.INFO, text="ABOUT", position=NavigationItemPosition.BOTTOM)

        # Add toggle theme button
        self.navigationInterface.addItem(
            "theme_toggle",
            FIF.CONSTRACT,
            "TOGGLE THEME",
            onClick=self.change_theme,
            position=NavigationItemPosition.BOTTOM,
        )

    def change_theme(self):
        toggleTheme()

    def state_toggle(self):
        if self.is_toggled is False:
            self.is_toggled = True

    def update_model(self):
        hidden_layer_count = self.train_interface.layer_data.rowCount()
        self.neuron_counts = []
        self.activations = []
        for i in range(hidden_layer_count):
            self.neuron_counts.append(self.train_interface.neuron_count_widgets[i].value())
            self.activations.append(self.train_interface.activation_widgets[i].currentText())

        def add_activation(self, activation):
            if activation == "ReLU":
                self.model.add(core.Activation_ReLU())
            elif activation == "Sigmoid":
                self.model.add(core.Activation_Sigmoid())
            elif activation == "Linear":
                self.model.add(core.Activation_Linear())

        self.model = core.Model()
        self.model.add(core.Layer_Dense(784, self.neuron_counts[0]))

        if hidden_layer_count == 1:
            add_activation(self, self.activations[0])
            self.model.add(core.Layer_Dense(self.neuron_counts[0], 10))
        else:
            for i in range(hidden_layer_count - 1):
                add_activation(self, self.activations[i])
                self.model.add(core.Layer_Dense(self.neuron_counts[i], self.neuron_counts[i + 1]))
            add_activation(self, self.activations[hidden_layer_count - 1])
            self.model.add(core.Layer_Dense(self.neuron_counts[hidden_layer_count - 1], 10))

        self.model.add(core.Activation_Softmax())

        # Set loss, optimizer and accuracy objects
        def set_optimizer(self, optimizer, learning_rate, decay, momentum):
            if optimizer == "SGD":
                self.model.optimizer = core.Optimizer_SGD(learning_rate=learning_rate, decay=decay, momentum=momentum)
            elif optimizer == "AdaGrad":
                self.model.optimizer = core.Optimizer_Adagrad(learning_rate=learning_rate, decay=decay)
            elif optimizer == "RMSProp":
                self.model.optimizer = core.Optimizer_RMSprop(learning_rate=learning_rate, decay=decay)
            elif optimizer == "Adam":
                self.model.optimizer = core.Optimizer_Adam(learning_rate=learning_rate, decay=decay, beta_1=momentum)

        set_optimizer(
            self,
            self.train_interface.optimizer_combobox.currentText(),
            learning_rate=float(self.train_interface.learning_rate_combobox.currentText()),
            decay=float(self.train_interface.decay_combobox.currentText()),
            momentum=self.train_interface.momentum_spinbox.value(),
        )
        self.model.set(
            loss=core.Loss_CategoricalCrossentropy(),
            accuracy=core.Accuracy_Categorical(),
        )

        # Finalize the model
        self.model.finalize()

    def update_train_interface(self):
        for i in range(self.train_interface.layer_data.rowCount()):
            self.train_interface.remove_layer()

        def activation_to_string(layer):
            if isinstance(layer, core.Activation_ReLU):
                return "ReLU"
            elif isinstance(layer, core.Activation_Sigmoid):
                return "Sigmoid"
            elif isinstance(layer, core.Activation_Linear):
                return "Linear"

        def optimizer_to_string(optimizer):
            if isinstance(optimizer, core.Optimizer_SGD):
                return "SGD"
            elif isinstance(optimizer, core.Optimizer_Adagrad):
                return "AdaGrad"
            elif isinstance(optimizer, core.Optimizer_RMSprop):
                return "RMSProp"
            elif isinstance(optimizer, core.Optimizer_Adam):
                return "Adam"

        def learning_rate_to_string(learning_rate):
            if learning_rate >= 1:
                return "1"
            elif learning_rate >= 0.1:
                return "0.1"
            elif learning_rate >= 0.01:
                return "0.01"
            elif learning_rate >= 1e-3:
                return "1e-3"
            elif learning_rate >= 1e-4:
                return "1e-4"
            else:
                return "1e-5"

        def decay_to_string(decay):
            if decay >= 0.1:
                return "0.1"
            elif decay >= 0.01:
                return "0.01"
            elif decay >= 1e-3:
                return "1e-3"
            elif decay >= 1e-4:
                return "1e-4"
            elif decay >= 1e-5:
                return "1e-5"
            elif decay >= 1e-6:
                return "1e-6"
            elif decay >= 1e-7:
                return "1e-7"
            else:
                return "0"

        for i in range(len(self.model.trainable_layers) - 1):
            layer = self.model.layers[2 * i + 1]
            self.train_interface.add_layer(self.model.trainable_layers[i].n_neurons, activation_to_string(layer))

        self.train_interface.optimizer_combobox.setCurrentText(optimizer_to_string(self.model.optimizer))
        self.train_interface.learning_rate_combobox.setCurrentText(
            learning_rate_to_string(self.model.optimizer.learning_rate)
        )
        self.train_interface.decay_combobox.setCurrentText(decay_to_string(self.model.optimizer.decay))
        if hasattr(self.model.optimizer, "momentum"):
            self.train_interface.momentum_spinbox.setValue(self.model.optimizer.momentum)
        elif hasattr(self.model.optimizer, "beta_1"):
            self.train_interface.momentum_spinbox.setValue(self.model.optimizer.beta_1)
        else:
            self.train_interface.momentum_spinbox.setValue(0)

    def save_model(self):
        if self.model is None or self.is_toggled is True:
            self.update_model()

        path, _ = QFileDialog.getSaveFileName(self, "Save Model", "models", "Model File(*.model)")

        if path == "":
            return

        self.model.save(path)
        self.is_toggled = False
        w = InfoBar.success(title="Success", content="Model saved.", isClosable=False, duration=2000, parent=self)
        w.show()

    def load_model(self):
        try:
            path, _ = QFileDialog.getOpenFileName(self, "Select Model", "models", "Model File(*.model)")
            if path == "":
                return
        except pickle.PickleError:
            w = InfoBar.error(title="Error", content="Invalid model!", isClosable=False, duration=2000, parent=self)
            w.show()
            return

        self.is_toggled = False
        self.model = core.Model.load(path)
        self.loaded_model_path = path
        self.predict_interface.model_path.setText(path)
        self.update_train_interface()
        w = InfoBar.success(title="Success", content="Model loaded.", isClosable=False, duration=2000, parent=self)
        w.show()


if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)

    window = MainWindow()

    window.show()

    sys.exit(app.exec_())
