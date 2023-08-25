import os
import cv2
import numpy as np
import tensorflow as tf 
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout,
    QGridLayout, QWidget, QFileDialog, QTabWidget, QHBoxLayout, QScrollArea, QSplashScreen, QMenu, QMenuBar
)
import time
from PyQt6 import QtGui
from PyQt6.QtGui import QPainter, QPixmap, QImage
from PyQt6.QtGui import QPixmap, QColor, QPalette, QFont
from PyQt6.QtCore import Qt
import shutil
from PyQt6.QtWidgets import QMessageBox
import datetime
from PyQt6.QtGui import QPainter
from PyQt6.QtGui import QImage
from PIL import Image, ImageDraw, ImageFont
from PIL import Image as PILImage

class HumanDetector:
    def __init__(self, model_path='yolov3.weights', cfg_path='yolov3.cfg'):
        self.net = cv2.dnn.readNet(model_path, cfg_path)

    def identify_humans(self, image):
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_names = self.net.getUnconnectedOutLayersNames()
        outputs = self.net.forward(layer_names)

        human_detected = False
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if class_id == 0 and confidence > 0.5:  # Class ID 0 corresponds to "person" in COCO
                    human_detected = True
                    break
        
        return human_detected

class ImageViewerApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Smart Select")
        self.setWindowIcon(QtGui.QIcon("logo.png"))
        self.setGeometry(300, 100, 1000, 600)  # Adjust window size and position

        # Set the custom color palette (black, green, grey)
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(0, 0, 0))           # Black
        palette.setColor(QPalette.ColorRole.Base, QColor(24, 195, 124))        # Dark Green
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(128, 128, 128)) # Grey
        self.setPalette(palette)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        self.tab_widget = QTabWidget()

                # Add menu bar
        self.menu_bar = QMenuBar()
        self.setMenuBar(self.menu_bar)

        file_menu = QMenu("&File", self.menu_bar)
        self.menu_bar.addMenu(file_menu)

        about_menu = QMenu("&About", self.menu_bar)
        self.menu_bar.addMenu(about_menu)

        # Create tabs for different detection modes
        self.image_loading_tab = QWidget()
        self.blur_tab = QWidget()
        self.overexposure_tab = QWidget()
        self.people_detection_tab = QWidget()

        self.tab_widget.addTab(self.image_loading_tab, "Load Images")
        self.tab_widget.addTab(self.blur_tab, "Blur Detection")
        self.tab_widget.addTab(self.overexposure_tab, "Overexposure Detection")
        self.tab_widget.addTab(self.people_detection_tab, "People Detection")

        self.layout.addWidget(self.tab_widget)
        self.central_widget.setLayout(self.layout)

        self.sharp_overlay = QPixmap("sharp.png")
        self.blurred_overlay = QPixmap("blurred.png")

        # Initialize the tabs
        self.init_image_loading_tab()
        self.init_detection_tab("Blur Detection", self.detect_blur, self.blur_tab)
        self.init_detection_tab("Overexposure Detection", self.detect_overexposure, self.overexposure_tab)
        self.init_detection_tab("People Detection", self.detect_people, self.people_detection_tab)

        # Set the default active tab to "Load Images"
        self.tab_widget.setCurrentIndex(0)

        # Set the active tab indicator color to neon green
        self.tab_widget.setStyleSheet("QTabBar::tab:selected { background-color: #18c37c; }")

        self.tab_widget.currentChanged.connect(self.tab_changed)

        self.loaded_image_paths = []  # List to store loaded image paths
        self.is_blurred = []
        self.status_labels = []  # Initialize status_labels as a class attribute
        self.new_images_loaded = False  # Flag to track whether new images have been loaded
        self.classified_images = []  # List to store classified images

        self.blurred_paths = []  # List to store paths of blurred images
        self.sharp_paths = []    # List to store paths of sharp images
        self.overexposed_paths = []  # List to store paths of overexposed images
        self.normal_paths = []   # List to store paths of normal images
        self.with_people_paths = []      # List to store paths of images with people detected
        self.without_people_paths = []   # List to store paths of images without people detected

    def save_images_by_status(self, output_folder, status_folder_1, status_folder_2):
        blurred_folder = os.path.join(output_folder, status_folder_1)
        sharp_folder = os.path.join(output_folder, status_folder_2)

        classified_images = []  # List to store classified images

        num_images = min(len(self.loaded_images), len(self.loaded_image_paths))

        for i in range(num_images):
            image_path = self.loaded_image_paths[i]
            image_name = os.path.basename(image_path)

            status_labels = self.status_labels_container.findChildren(QLabel)
            status_label = status_labels[i]

            status_text = status_label.text()

            if status_text.lower() == status_folder_1:
                output_path = os.path.join(blurred_folder, image_name)
            else:
                output_path = os.path.join(sharp_folder, image_name)

            classified_images.append((image_path, output_path))  # Store the image path and output path

        self.classified_images = classified_images  # Store the classified images in the instance variable

        QMessageBox.information(self, "Images Classified", "Images classified.")

    def save_classified_images_to_folders(self):
        if not self.new_images_loaded:
            QMessageBox.information(self, "No Images to Save", "No classified images to save.")
            return

        output_folder = os.path.join(os.getcwd(), "ClassifiedImages")
        os.makedirs(output_folder, exist_ok=True)

        active_tab_index = self.tab_widget.currentIndex()

        if active_tab_index == 1:  # Blur Detection tab
            status_folder_1 = "blurred"
            status_folder_2 = "sharp"
            paths_list_1 = self.blurred_paths
            paths_list_2 = self.sharp_paths
        elif active_tab_index == 2:  # Overexposure Detection tab
            status_folder_1 = "overexposed"
            status_folder_2 = "normal"
            paths_list_1 = self.overexposed_paths
            paths_list_2 = self.normal_paths
        elif active_tab_index == 3:  # People Detection tab
            status_folder_1 = "with_people"
            status_folder_2 = "without_people"
            paths_list_1 = self.with_people_paths
            paths_list_2 = self.without_people_paths
        else:
            QMessageBox.warning(self, "Invalid Tab", "Saving images is only supported in the Blur Detection, Overexposure Detection, and People Detection tabs.")
            return

        for image_path in paths_list_1:
            image_name = os.path.basename(image_path)
            output_path = os.path.join(output_folder, status_folder_1, image_name)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copyfile(image_path, output_path)

        for image_path in paths_list_2:
            image_name = os.path.basename(image_path)
            output_path = os.path.join(output_folder, status_folder_2, image_name)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copyfile(image_path, output_path)

        # Append classified image paths to the existing text file
        with open("classified_image_paths.txt", "a") as file:
            file.write(f"{status_folder_1.capitalize()} Images:\n")
            for path in paths_list_1:
                file.write(path + "\n")

            file.write(f"\n{status_folder_2.capitalize()} Images:\n")
            for path in paths_list_2:
                file.write(path + "\n")

        QMessageBox.information(self, "Images Saved", f"Images saved to {output_folder} folder.\nClassified image paths updated in 'classified_image_paths.txt'.")


    def calculate_blur_score(self, image_path):
        image = cv2.imread(image_path)
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(grayscale_image, cv2.CV_64F).var()
        return blur_score


    def process_own_image(self, image_path):
        image = cv2.imread(image_path)
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image = cv2.resize(processed_image, (128, 128))  # Resize the image to (128, 128)
        processed_image = np.expand_dims(processed_image, axis=0)
        return processed_image

    def add_outline(self, label):
        # Add a dashed outline to the label using a border
        label.setStyleSheet("border: 1px dashed #808080; background-color: #333333;")  # Grey border, Dark Grey background

    def init_image_loading_tab(self):
        self.image_loading_layout = QVBoxLayout(self.image_loading_tab)

        # Create scroll area for image placeholders
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_area.setWidget(self.scroll_widget)
        self.image_loading_layout.addWidget(self.scroll_area)

        # Create the grid layout for image placeholders
        self.image_grid_layout = QGridLayout(self.scroll_widget)

        self.image_labels = []  # List to hold image labels

        for row in range(3):
            for col in range(3):
                label = QLabel(self)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setMinimumSize(180, 120)  # Set minimum size for equal-sized placeholders
                self.add_outline(label)  # Add dashed outline
                self.image_grid_layout.addWidget(label, row, col, alignment=Qt.AlignmentFlag.AlignCenter)

        # Create "Load Images" and "Clear Images" buttons
        button_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Images", self)
        self.load_button.clicked.connect(self.load_images)
        self.load_button.setFixedWidth(100)  # Set button width
        self.load_button.setFixedHeight(30)  # Set button height

        self.load_button.setStyleSheet(
    """
    QPushButton {
        background-color: #18C37C;
        color: white;
        border-radius: 5px;
    }

    QPushButton:hover {
        background-color: #1CAC78;
    }
    """
)
        button_layout.addWidget(self.load_button)
        self.clear_button = QPushButton("Clear Images", self)
        self.clear_button.clicked.connect(self.clear_images)
        self.clear_button.setFixedWidth(100)  # Set button width
        self.clear_button.setFixedHeight(30)  # Set button height
        self.clear_button.setStyleSheet(
    """
    QPushButton {
        background-color: #BB2525;
        color: white;
        border-radius: 5px;
    }

    QPushButton:hover {
        background-color: #AA1E1E;
    }
    """
)
        button_layout.addWidget(self.clear_button)

        # Add label to display the number of loaded images
        self.num_loaded_label = QLabel("Image Count: 0", self)

        self.image_loading_layout.addLayout(button_layout)
        self.image_loading_layout.addWidget(self.num_loaded_label)

        self.image_loading_tab.setLayout(self.image_loading_layout)

        self.loaded_images = []

    def init_detection_tab(self, tab_name, detection_function, tab):
        tab_layout = QVBoxLayout(tab)

        # Create scroll area for image placeholders
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_area.setWidget(scroll_widget)
        tab_layout.addWidget(scroll_area)

        # Create grid layout for image placeholders
        placeholder_layout = QGridLayout(scroll_widget)

        image_labels = []  # List to hold image labels

        for row in range(30):
            for col in range(3):
                label = QLabel(self)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setFixedSize(180, 120)  # Set size for image placeholders
                self.add_outline(label)  # Add dashed outline
                image_labels.append(label)
                placeholder_layout.addWidget(label, row, col, alignment=Qt.AlignmentFlag.AlignCenter)

        # Create a container for the status labels
        self.status_labels_container = QWidget(tab)
        status_labels_layout = QVBoxLayout(self.status_labels_container)
        status_labels_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Add the status labels container to the tab layout
        tab_layout.addWidget(self.status_labels_container)

        save_button = QPushButton("Save Images", self)
        save_button.setFixedHeight(30)  # Set button height

        save_button.setStyleSheet(
    """
    QPushButton {
        background-color: #18C37C;
        color: white;
        border-radius: 5px;
    }

    QPushButton:hover {
        background-color: #1CAC78;
    }
    """
)
        
        save_button.clicked.connect(self.save_classified_images_to_folders)  # Connect the button to the new method
        tab_layout.addWidget(save_button)

        tab.setLayout(tab_layout)

    def load_images(self):
        if os.path.exists("classified_image_paths.txt"):
            os.remove("classified_image_paths.txt")
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Open Images", "", "Image Files (*.png *.jpg *.jpeg *.gif *.bmp *.3fr *.ari *.arw *.bay *.braw *.crw *.cr2 *.cr3 *.cap *.data *.dcs *.dcr *.dng *.drf *.eip *.erf *.fff *.gpr *.iiq *.k25 *.kdc *.mdc *.mef *.mos *.mrw *.nef *.nrw *.obm *.orf *.pef *.ptx *.pxn *.r3d *.raf *.raw *.rwl *.rw2 *.rwz *.sr2 *.srf *.srw *.tif *.x3f)")

        if not file_paths:
            return

        for file_path in file_paths:
            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaled(180, 120, Qt.AspectRatioMode.KeepAspectRatio)
            self.loaded_images.append(pixmap)

            # Store the file path in the loaded_image_paths list
            self.loaded_image_paths.append(file_path)

            # Create a new status label for the loaded image
            status_label = QLabel(self)
            self.status_labels.append(status_label)

        self.update_scroll_layout()

        # Update the label displaying the number of loaded images
        self.num_loaded_label.setText(f"Image Count: {len(self.loaded_images)}")
        
        # Set the new_images_loaded flag to True
        self.new_images_loaded = True

    def clear_images(self):
        self.loaded_images = []
        self.loaded_image_paths = []

        # Clear the status labels list
        self.status_labels = []

        # Hide the status labels in the status_labels_container
        for status_label in self.status_labels_container.findChildren(QLabel):
            status_label.setVisible(False)

        # Recreate the central widget to reset the UI
        self.central_widget.deleteLater()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.tab_widget = QTabWidget()

        # Create tabs for different detection modes
        self.image_loading_tab = QWidget()
        self.blur_tab = QWidget()
        self.overexposure_tab = QWidget()
        self.people_detection_tab = QWidget()

        self.tab_widget.addTab(self.image_loading_tab, "Load Images")
        self.tab_widget.addTab(self.blur_tab, "Blur Detection")
        self.tab_widget.addTab(self.overexposure_tab, "Overexposure Detection")
        self.tab_widget.addTab(self.people_detection_tab, "People Detection")

        self.layout.addWidget(self.tab_widget)
        self.central_widget.setLayout(self.layout)

        # Initialize the tabs
        self.init_image_loading_tab()
        self.init_detection_tab("Blur Detection", self.detect_blur, self.blur_tab)
        self.init_detection_tab("Overexposure Detection", self.detect_overexposure, self.overexposure_tab)
        self.init_detection_tab("People Detection", self.detect_people, self.people_detection_tab)

        # Reset the appearance of the labels in the image loading tab
        for label in self.image_labels:
            self.add_outline(label)

        # Set the default active tab to "Load Images"
        self.tab_widget.setCurrentIndex(0)

        # Set the active tab indicator color to neon green
        self.tab_widget.setStyleSheet("QTabBar::tab:selected { background-color: #18c37c; }")

        self.tab_widget.currentChanged.connect(self.tab_changed)


    def update_scroll_layout(self):
        for i, pixmap in enumerate(self.loaded_images):
            row = i // 3
            col = i % 3
            label = QLabel(self)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(180, 120)  # Set minimum size for equal-sized placeholders
            label.setPixmap(pixmap)
            self.add_outline(label)
            self.image_labels.append(label)
            self.image_grid_layout.addWidget(label, row, col, alignment=Qt.AlignmentFlag.AlignCenter)

        # Update the label displaying the number of loaded images
        self.num_loaded_label.setText(f"Image Count {len(self.loaded_images)}")

    def detect_blur(self):
        # Clear the status_labels list at the beginning of the function
        self.status_labels = []

        # Populate the placeholders with loaded images
        image_labels = [label for label in self.blur_tab.findChildren(QLabel)]

        for i in range(min(len(self.loaded_images), len(image_labels))):
            image_path = self.loaded_image_paths[i]
            image = cv2.imread(image_path)
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(grayscale_image, cv2.CV_64F).var()

            label = image_labels[i]
            pixmap = self.loaded_images[i]
            pixmap = pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio)

            threshold = 100

            if blur_score < threshold:
                status_text = "Blurry"
                status_color = "red"
                overlay_path = "blurred1.png"
                self.blurred_paths.append(image_path)  # Append path to blurred list
            else:
                status_text = "Not Blurry"
                status_color = "green"
                overlay_path = "sharp1.png"
                self.sharp_paths.append(image_path)  # Append path to sharp list

            # Load the overlay image as a QPixmap
            overlay_pixmap = QPixmap(overlay_path)

            # Calculate the position to center the overlay within the placeholder
            overlay_x = (pixmap.width() - overlay_pixmap.width()) // 2
            overlay_y = (pixmap.height() - overlay_pixmap.height()) // 2

            # Create a new pixmap for blending
            blended_pixmap = QPixmap(pixmap.size())
            blended_pixmap.fill(Qt.GlobalColor.transparent)  # Fill with transparent color
            painter = QPainter(blended_pixmap)
            painter.drawPixmap(0, 0, pixmap)
            painter.drawPixmap(overlay_x, overlay_y, overlay_pixmap)  # Draw overlay centered
            painter.end()

            # Create a layout for the placeholder that includes the blended pixmap and status label
            placeholder_layout = QVBoxLayout(label)
            pixmap_label = QLabel(self)
            pixmap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pixmap_label.setPixmap(blended_pixmap)
            pixmap_label.setStyleSheet(f"border: 2px solid {status_color};")
            placeholder_layout.addWidget(pixmap_label)  # Add the pixmap label

            # Create a status label and add it below the pixmap label
            status_label = QLabel(self)
            status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            status_label.setText(f"<font color='{status_color}'>{status_text}</font>")
            placeholder_layout.addWidget(status_label)  # Add the status label

            # Add stretch to the layout to position the status label below the pixmap label
            placeholder_layout.addStretch(1)

            self.status_labels.append(status_label)  # Append the status label to the list

            # Add the status labels to the status_labels_container
            for status_label in self.status_labels:
                self.status_labels_container.layout().addWidget(status_label)

    def detect_overexposure(self):
        # Populate the placeholders with loaded images
        image_labels = [label for label in self.overexposure_tab.findChildren(QLabel)]

        for i in range(min(len(self.loaded_images), len(image_labels))):
            image_path = self.loaded_image_paths[i]
            image = cv2.imread(image_path)
            
            # Calculate the average brightness of the image
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            average_brightness = cv2.mean(gray_image)[0]
            
            label = image_labels[i]
            pixmap = self.loaded_images[i]
            pixmap = pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio)

            threshold = 120  # Adjust the threshold based on your preference

            if average_brightness > threshold:
                status_text = "Overexposed"
                status_color = "red"
                overlay_path = "blurred1.png"
                self.overexposed_paths.append(image_path)  # Append path to overexposed list
            else:
                status_text = "Not Overexposed"
                status_color = "green"
                overlay_path = "sharp1.png"
                self.normal_paths.append(image_path)  # Append path to normal list

            # Load the overlay image as a QPixmap
            overlay_pixmap = QPixmap(overlay_path)

            overlay_x = (pixmap.width() - overlay_pixmap.width()) // 2
            overlay_y = (pixmap.height() - overlay_pixmap.height()) // 2

            blended_pixmap = QPixmap(pixmap.size())
            blended_pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(blended_pixmap)
            painter.drawPixmap(0, 0, pixmap)
            painter.drawPixmap(overlay_x, overlay_y, overlay_pixmap)
            painter.end()

            # Set the blended pixmap as the pixmap for the image label
            label.setPixmap(blended_pixmap)
            label.setStyleSheet(f"border: 2px solid {status_color};")


    # Inside the ImageViewerApp class
    def detect_people(self):
        human_detector = HumanDetector()

        # Clear status labels from the previous run
        for status_label in self.status_labels_container.findChildren(QLabel):
            status_label.setVisible(False)

        # Populate the placeholders with loaded images
        image_labels = [label for label in self.people_detection_tab.findChildren(QLabel)]

        for i in range(min(len(self.loaded_images), len(image_labels))):
            image_path = self.loaded_image_paths[i]
            image = cv2.imread(image_path)

            human_detected = human_detector.identify_humans(image)

            label = image_labels[i]
            pixmap = self.loaded_images[i]
            pixmap = pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio)

            overlay_path = "sharp1.png" if human_detected else "blurred1.png"
            status_color = "green" if human_detected else "red"

            overlay_pixmap = QPixmap(overlay_path)

            overlay_x = (pixmap.width() - overlay_pixmap.width()) // 2
            overlay_y = (pixmap.height() - overlay_pixmap.height()) // 2

            blended_pixmap = QPixmap(pixmap.size())
            blended_pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(blended_pixmap)
            painter.drawPixmap(0, 0, pixmap)
            painter.drawPixmap(overlay_x, overlay_y, overlay_pixmap)
            painter.end()

            # Set the blended pixmap as the pixmap for the image label
            label.setPixmap(blended_pixmap)
            label.setStyleSheet(f"border: 2px solid {status_color};")

            if human_detected:
                subfolder = "with_people"
                self.with_people_paths.append(image_path)  # Append path to with_people list
            else:
                subfolder = "without_people"
                self.without_people_paths.append(image_path)  # Append path to without_people list

        # Set the new_images_loaded flag to True
        self.new_images_loaded = True

    def convert_cv_to_pixmap(self, cv_image):
        height, width, channel = cv_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        return pixmap


    def tab_changed(self, index):
        # Clear labels and status labels from the previous tab
        if index == 1:  # Blur Detection tab
            self.clear_labels(self.blur_tab)
            self.detect_blur()  # Call the detect_blur function to re-display results
        elif index == 2:  # Overexposure Detection tab
            self.clear_labels(self.overexposure_tab)
            self.detect_overexposure()  # Call the detect_overexposure function to re-display results
        elif index == 3:  # People Detection tab
            self.clear_labels(self.people_detection_tab)
            self.detect_people()  # Call the detect_people function to re-display results

    def clear_labels(self, tab):
        image_labels = tab.findChildren(QLabel)
        for label in image_labels:
            label.clear()

if __name__ == "__main__":
    app = QApplication([])

    # Create a splash screen
    splash_pix = QPixmap("splash.png")
    splash = QSplashScreen(splash_pix)
    splash.show()
    time.sleep(5)

    # Create an instance of your ImageViewerApp
    image_viewer = ImageViewerApp()

    # Process events to show the splash screen
    app.processEvents()

    # Show the main window
    image_viewer.show()

    # Hide the splash screen
    splash.finish(image_viewer)

        # Create a QTimer to close the splash screen after 10 seconds


    # Start the event loop
    app.exec()