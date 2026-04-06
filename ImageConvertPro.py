import os
import sys
import threading
import io
import json
import subprocess
import warnings
from pathlib import Path
from PIL import Image, ImageTk, ImageEnhance, ImageOps, PngImagePlugin
import rawpy
import pillow_heif

# GPU support imports (optional)
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
    try:
        # Check for CUDA support
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            CUDA_AVAILABLE = True
        else:
            CUDA_AVAILABLE = False
    except:
        CUDA_AVAILABLE = False
except ImportError:
    cv2 = None
    OPENCV_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    warnings.filterwarnings(
        "ignore",
        message="CUDA path could not be detected.*",
        category=UserWarning,
    )
    import cupy as cp
    try:
        CUPY_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        CUPY_AVAILABLE = False
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

try:
    import torch
    import torchvision.transforms as T
    PYTORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None
    PYTORCH_AVAILABLE = False

try:
    import winsound
except ImportError:
    winsound = None

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QListWidgetItem, QSlider, QSpinBox, QComboBox,
    QFileDialog, QMessageBox, QFrame, QScrollArea, QCheckBox, QInputDialog,
    QProgressBar, QSplitter, QTabWidget, QGroupBox, QGridLayout, QMenuBar,
    QMenu, QDialog, QTextEdit, QGraphicsDropShadowEffect, QGraphicsOpacityEffect
)
from PySide6.QtCore import Qt, QTimer, QSize, QPropertyAnimation, QRect, Signal, QThread, QEasingCurve, QEvent, QUrl
from PySide6.QtGui import QPixmap, QColor, QIcon, QFont, QImage, QPainter, QAction, QActionGroup

pillow_heif.register_heif_opener()

SUPPORTED = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".ico", ".icns", ".heic", ".heif", ".cr2", ".nef", ".arw", ".dng", ".raw")
EXPORT_FORMATS = {
    "JPG": "jpg", "JPEG": "jpeg", "PNG": "png", "WEBP": "webp", "BMP": "bmp",
    "TIFF": "tiff", "TIFF-LZW": "tiff", "GIF": "gif", "ICO": "ico", "ICNS": "icns"
}

ADJUSTMENT_SPECS = [
    ("exposure", "Exposure"),
    ("contrast", "Contrast"),
    ("brightness", "Brightness"),
    ("saturation", "Saturation"),
    ("sharpness", "Sharpness"),
    ("highlights", "Highlights"),
    ("shadows", "Shadows"),
    ("whites", "Whites"),
    ("blacks", "Blacks"),
    ("clarity", "Clarity"),
    ("vibrance", "Vibrance"),
    ("temperature", "Temperature"),
    ("tint", "Tint"),
    ("hue", "Hue"),
    ("fade", "Fade"),
    ("dehaze", "Dehaze"),
    ("gamma", "Gamma"),
    ("vignette", "Vignette"),
]

DEFAULT_ADJUSTMENT_IDS = [param_id for param_id, _ in ADJUSTMENT_SPECS]

PREVIEW_RESOLUTION_PRESETS = {
    "Fast": 900,
    "Balanced": 1400,
    "High": 2200,
    "Ultra": 3200,
}

class ProcessWorker(QThread):
    """Worker thread for batch processing images"""
    progress_updated = Signal(int)
    status_updated = Signal(str, str)  # message, color
    finished = Signal()

    def __init__(self, app, file_list, formats=None, options=None):
        super().__init__()
        self.app = app
        self.file_list = file_list
        self.formats = formats or ["JPG"]
        self.options = options or {}
        self.stop_requested = False
        self.was_cancelled = False

    def request_stop(self):
        """Ask the worker to stop after the current safe point."""
        self.stop_requested = True

    def run(self):
        total_operations = len(self.file_list) * len(self.formats)
        operation_count = 0

        for file_path in self.file_list:
            if self.stop_requested:
                self.was_cancelled = True
                break

            prepared = None
            try:
                prepared = self.app.prepare_image_for_export(file_path, raise_on_error=True)
            except Exception as e:
                print(f"Error preparing {file_path}: {e}")

            for fmt in self.formats:
                if self.stop_requested:
                    self.was_cancelled = True
                    break

                operation_count += 1
                try:
                    self.status_updated.emit(f"Processing: {os.path.basename(file_path)} -> {fmt} ({operation_count}/{total_operations})", "blue")
                    if prepared is None:
                        raise RuntimeError("Failed to prepare source image for export")
                    self.app.save_processed_image(
                        prepared["image"].copy(),
                        prepared["name"],
                        prepared["metadata"],
                        format_name=fmt,
                        options=self.options,
                    )
                    self.status_updated.emit(f"OK {os.path.basename(file_path)} ({fmt}) - {operation_count}/{total_operations}", "green")
                except Exception as e:
                    print(f"Error processing {file_path} to {fmt}: {e}")
                    self.status_updated.emit(f"X {os.path.basename(file_path)} ({fmt}) - {str(e)}", "red")
                finally:
                    self.progress_updated.emit(operation_count)

            if self.stop_requested:
                break

        if self.was_cancelled:
            self.status_updated.emit("Processing stopped by user", "orange")
        else:
            self.status_updated.emit(f"Batch processing completed ({len(self.file_list)} files x {len(self.formats)} formats)", "green")
        self.finished.emit()


class AnimeSlider(QSlider):
    """Custom slider with animation support"""
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.animation = QPropertyAnimation(self, b"value")
        
    def animate_to(self, value, duration=250):
        """Animate slider to a value"""
        self.animation.stop()
        self.animation.setDuration(duration)
        self.animation.setStartValue(self.value())
        self.animation.setEndValue(value)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.animation.start()

    def get_output_path(input_path):
        base_dir = os.path.dirname(input_path)
        output_dir = os.path.join(base_dir, "output")

        # create folder if not exists
        os.makedirs(output_dir, exist_ok=True)

        return output_dir


class ModernSliderWidget(QFrame):
    """Modern slider widget with label and value display"""
    value_changed = Signal(float)
    
    def __init__(self, label_text, min_val=0, max_val=100, default=50, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(5, 3, 5, 3)
        self.layout.setSpacing(8)
        
        # Label
        self.label = QLabel(label_text)
        self.label.setMinimumWidth(110)
        self.label.setFont(QFont("Segoe UI", 10))
        self.layout.addWidget(self.label)
        
        # Slider
        self.slider = AnimeSlider(Qt.Horizontal)
        self.slider.setMinimum(int(min_val))
        self.slider.setMaximum(int(max_val))
        self.slider.setValue(int(default))
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background-color: #e0e0e0;
                height: 4px;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background-color: #2196F3;
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
                border: 1px solid #1976D2;
            }
            QSlider::handle:horizontal:hover {
                background-color: #1E88E5;
            }
        """)
        self.layout.addWidget(self.slider, 1)
        
        self.value_spin = QSpinBox()
        self.value_spin.setRange(int(min_val), int(max_val))
        self.value_spin.setValue(int(default))
        self.value_spin.setSuffix("%")
        self.value_spin.setButtonSymbols(QSpinBox.UpDownArrows)
        self.value_spin.setKeyboardTracking(False)
        self.value_spin.setMinimumWidth(76)
        self.value_spin.setAlignment(Qt.AlignRight)
        self.layout.addWidget(self.value_spin)

        self.slider.valueChanged.connect(self.on_slider_value_changed)
        self.value_spin.valueChanged.connect(self.on_spin_value_changed)
        self.refresh_style("Dark")
    
    def on_slider_value_changed(self, value):
        self.value_spin.blockSignals(True)
        self.value_spin.setValue(int(value))
        self.value_spin.blockSignals(False)
        self.value_changed.emit(float(value))

    def on_spin_value_changed(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(int(value))
        self.slider.blockSignals(False)
        self.value_changed.emit(float(value))

    def refresh_style(self, resolved_theme):
        """Update slider and number styles for the active theme."""
        if resolved_theme == "Light":
            groove = "#cfd8e3"
            handle = "#2563eb"
            handle_hover = "#1d4ed8"
            label_color = "#4b5563"
            spin_bg = "#ffffff"
            spin_border = "#cfd8e3"
        else:
            groove = "#475569"
            handle = "#60a5fa"
            handle_hover = "#93c5fd"
            label_color = "#d9e0ea"
            spin_bg = "#151b23"
            spin_border = "#334155"

        self.label.setStyleSheet(f"color: {label_color};")
        self.value_spin.setStyleSheet(
            f"background-color: {spin_bg}; color: {label_color}; border: 1px solid {spin_border}; "
            "border-radius: 8px; padding: 4px 8px; font-weight: 600;"
        )
        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background-color: {groove};
                height: 4px;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background-color: {handle};
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
                border: 1px solid {handle_hover};
            }}
            QSlider::handle:horizontal:hover {{
                background-color: {handle_hover};
            }}
        """)
    
    def set_value(self, value, animate=True):
        if animate:
            self.slider.animate_to(int(value), duration=150)
        else:
            self.slider.setValue(int(value))
        self.value_spin.setValue(int(value))
    
    def get_value(self):
        return self.value_spin.value()


class ImageCanvas(QLabel):
    """Custom canvas for image display with zoom and pan"""
    files_dropped = Signal(list)  # Signal to emit when files are dropped
    view_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("background-color: #1a1a1a; border: none;")
        self.setAcceptDrops(True)

        self.original_pixmap = None
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        
        # Mouse hover controls
        self.is_hovering = False
        self.setMouseTracking(True)

        self.setCursor(Qt.ArrowCursor)

    def can_pan(self, scaled_width=None, scaled_height=None, canvas_width=None, canvas_height=None):
        """Return True when the scaled image is larger than the viewport in either direction."""
        if not self.original_pixmap:
            return False

        if scaled_width is None:
            scaled_width = int(self.original_pixmap.width() * self.zoom)
        if scaled_height is None:
            scaled_height = int(self.original_pixmap.height() * self.zoom)
        if canvas_width is None:
            canvas_width = self.width()
        if canvas_height is None:
            canvas_height = self.height()

        return scaled_width > canvas_width or scaled_height > canvas_height

    def get_zoom_percentage(self):
        """Return the current zoom percentage for the UI."""
        return int(round(self.zoom * 100))

    def enterEvent(self, event):
        """Handle mouse enter - show navigation cursor and highlight"""
        self.is_hovering = True
        if self.original_pixmap and self.can_pan():
            self.setCursor(Qt.OpenHandCursor)
            # Show parent status if available
            if hasattr(self.parent(), 'log_status'):
                self.parent().log_status("Click and drag to pan • Mouse wheel to zoom", "blue")
        else:
            self.setCursor(Qt.CrossCursor)
            if hasattr(self.parent(), 'log_status'):
                self.parent().log_status("Mouse wheel to zoom", "blue")
        # Add subtle highlight
        self.setStyleSheet("background-color: #2a2a2a; border: 1px solid #555;")
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Handle mouse leave - reset cursor and styling"""
        self.is_hovering = False
        self.setCursor(Qt.ArrowCursor)
        # Reset styling
        self.setStyleSheet("background-color: #1a1a1a; border: none;")
        # Clear status if available
        if hasattr(self.parent(), 'log_status'):
            self.parent().log_status("Ready", "black")
        super().leaveEvent(event)

    def set_image(self, pixmap):
        """Set the image to display"""
        self.original_pixmap = pixmap
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.display_image()

    def update_image(self, pixmap):
        """Update image display while preserving zoom and pan state"""
        self.original_pixmap = pixmap
        self.display_image()

    def display_image(self):
        """Display the current image with zoom and pan"""
        if not self.original_pixmap:
            self.setPixmap(QPixmap())
            self.view_changed.emit()
            return

        canvas_width = self.width()
        canvas_height = self.height()

        if canvas_width < 1 or canvas_height < 1:
            return

        # Get original image dimensions
        img_width = self.original_pixmap.width()
        img_height = self.original_pixmap.height()

        # Calculate scaled dimensions
        scaled_width = int(img_width * self.zoom)
        scaled_height = int(img_height * self.zoom)

        if scaled_width <= 0 or scaled_height <= 0:
            return

        self.constrain_pan(scaled_width, scaled_height, canvas_width, canvas_height)

        # Create scaled pixmap
        scaled_pixmap = self.original_pixmap.scaled(scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        canvas_pixmap = QPixmap(canvas_width, canvas_height)
        canvas_pixmap.fill(QColor("#1a1a1a"))

        if scaled_width > canvas_width:
            draw_x = -int(self.pan_x)
        else:
            draw_x = (canvas_width - scaled_width) // 2

        if scaled_height > canvas_height:
            draw_y = -int(self.pan_y)
        else:
            draw_y = (canvas_height - scaled_height) // 2

        painter = QPainter(canvas_pixmap)
        painter.drawPixmap(draw_x, draw_y, scaled_pixmap)
        painter.end()

        self.setPixmap(canvas_pixmap)

        self.view_changed.emit()

    def constrain_pan(self, scaled_width=None, scaled_height=None, canvas_width=None, canvas_height=None):
        """Clamp pan offsets so the viewport stays inside the scaled image."""
        if not self.original_pixmap:
            self.pan_x = 0
            self.pan_y = 0
            return

        if scaled_width is None:
            scaled_width = int(self.original_pixmap.width() * self.zoom)
        if scaled_height is None:
            scaled_height = int(self.original_pixmap.height() * self.zoom)
        if canvas_width is None:
            canvas_width = self.width()
        if canvas_height is None:
            canvas_height = self.height()

        max_pan_x = max(0, scaled_width - canvas_width)
        max_pan_y = max(0, scaled_height - canvas_height)
        self.pan_x = max(0, min(max_pan_x, self.pan_x))
        self.pan_y = max(0, min(max_pan_y, self.pan_y))

    def get_view_state(self):
        """Return the current image and viewport geometry for the mini-map."""
        if not self.original_pixmap:
            return None

        img_width = self.original_pixmap.width()
        img_height = self.original_pixmap.height()
        if img_width <= 0 or img_height <= 0:
            return None

        viewport_width = min(img_width, self.width() / max(self.zoom, 0.0001))
        viewport_height = min(img_height, self.height() / max(self.zoom, 0.0001))
        origin_x = max(0.0, min(img_width - viewport_width, self.pan_x / max(self.zoom, 0.0001)))
        origin_y = max(0.0, min(img_height - viewport_height, self.pan_y / max(self.zoom, 0.0001)))

        return {
            "pixmap": self.original_pixmap,
            "image_width": img_width,
            "image_height": img_height,
            "origin_x": origin_x,
            "origin_y": origin_y,
            "viewport_width": viewport_width,
            "viewport_height": viewport_height,
        }

    def set_view_center_from_ratio(self, ratio_x, ratio_y):
        """Center the viewport around a normalized image coordinate."""
        state = self.get_view_state()
        if not state:
            return

        ratio_x = max(0.0, min(1.0, ratio_x))
        ratio_y = max(0.0, min(1.0, ratio_y))
        center_x = state["image_width"] * ratio_x
        center_y = state["image_height"] * ratio_y

        self.pan_x = (center_x - state["viewport_width"] / 2) * self.zoom
        self.pan_y = (center_y - state["viewport_height"] / 2) * self.zoom
        self.constrain_pan()
        self.display_image()

    def wheelEvent(self, event):
        """Handle mouse wheel zoom"""
        if not self.original_pixmap:
            return
        zoom_factor = 1.2 if event.angleDelta().y() > 0 else 0.8
        self.apply_zoom_factor(zoom_factor, event.position().toPoint())

    def apply_zoom_factor(self, zoom_factor, anchor_point=None):
        """Zoom toward an anchor point and preserve the visible region."""
        if not self.original_pixmap:
            return

        old_zoom = self.zoom
        new_zoom = max(0.05, min(8.0, self.zoom * zoom_factor))
        if abs(new_zoom - old_zoom) < 1e-6:
            return

        canvas_width = max(1, self.width())
        canvas_height = max(1, self.height())
        old_scaled_width = max(1.0, self.original_pixmap.width() * old_zoom)
        old_scaled_height = max(1.0, self.original_pixmap.height() * old_zoom)

        if anchor_point is None:
            anchor_x = canvas_width / 2
            anchor_y = canvas_height / 2
        else:
            anchor_x = max(0.0, min(float(anchor_point.x()), float(canvas_width)))
            anchor_y = max(0.0, min(float(anchor_point.y()), float(canvas_height)))

        if old_scaled_width > canvas_width:
            image_ratio_x = (self.pan_x + anchor_x) / old_scaled_width
        else:
            draw_x = (canvas_width - old_scaled_width) / 2
            image_ratio_x = (anchor_x - draw_x) / old_scaled_width

        if old_scaled_height > canvas_height:
            image_ratio_y = (self.pan_y + anchor_y) / old_scaled_height
        else:
            draw_y = (canvas_height - old_scaled_height) / 2
            image_ratio_y = (anchor_y - draw_y) / old_scaled_height

        image_ratio_x = max(0.0, min(1.0, image_ratio_x))
        image_ratio_y = max(0.0, min(1.0, image_ratio_y))

        self.zoom = new_zoom
        new_scaled_width = self.original_pixmap.width() * self.zoom
        new_scaled_height = self.original_pixmap.height() * self.zoom

        if new_scaled_width > canvas_width:
            self.pan_x = image_ratio_x * new_scaled_width - anchor_x
        else:
            self.pan_x = 0

        if new_scaled_height > canvas_height:
            self.pan_y = image_ratio_y * new_scaled_height - anchor_y
        else:
            self.pan_y = 0

        self.constrain_pan(int(new_scaled_width), int(new_scaled_height), canvas_width, canvas_height)
        self.display_image()

    def mousePressEvent(self, event):
        """Start pan on mouse press"""
        if event.button() == Qt.LeftButton and self.original_pixmap and self.can_pan():
            self.is_panning = True
            self.pan_start_x = event.position().toPoint().x() + self.pan_x
            self.pan_start_y = event.position().toPoint().y() + self.pan_y
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Pan image during mouse move"""
        if self.is_panning and self.original_pixmap and self.can_pan():
            # Calculate new pan position
            new_pan_x = self.pan_start_x - event.position().toPoint().x()
            new_pan_y = self.pan_start_y - event.position().toPoint().y()

            # Constrain pan to image bounds
            img_width = self.original_pixmap.width() * self.zoom
            img_height = self.original_pixmap.height() * self.zoom
            canvas_width = self.width()
            canvas_height = self.height()

            max_pan_x = max(0, img_width - canvas_width)
            max_pan_y = max(0, img_height - canvas_height)

            self.pan_x = max(0, min(max_pan_x, new_pan_x))
            self.pan_y = max(0, min(max_pan_y, new_pan_y))

            self.display_image()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """End pan on mouse release"""
        if self.is_panning:
            self.is_panning = False
            if self.is_hovering and self.can_pan():
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.CrossCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        """Redraw on resize"""
        super().resizeEvent(event)
        self.display_image()

    def dragEnterEvent(self, event):
        """Handle drag enter event"""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """Handle drag move event"""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Handle drop event for files and folders"""
        if not event.mimeData().hasUrls():
            return

        paths = []

        # Extract all dropped items
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if os.path.isdir(file_path):
                # It's a folder - recursively add all supported image files
                for root, _, files in os.walk(file_path):
                    for file in files:
                        if file.lower().endswith(SUPPORTED):
                            paths.append(os.path.join(root, file))
            else:
                # It's a file - check if it's supported
                if file_path.lower().endswith(SUPPORTED):
                    paths.append(file_path)

        # Emit signal with collected paths
        if paths:
            self.files_dropped.emit(sorted(paths))

        event.accept()

    def fit_image(self):
        """Fit image to window"""
        if not self.original_pixmap:
            return

        canvas_width = self.width()
        canvas_height = self.height()
        img_width = self.original_pixmap.width()
        img_height = self.original_pixmap.height()

        if canvas_width > 0 and canvas_height > 0 and img_width > 0 and img_height > 0:
            # Calculate fit zoom
            zoom_x = canvas_width / img_width
            zoom_y = canvas_height / img_height
            self.zoom = min(zoom_x, zoom_y, 1.0)  # Don't zoom in beyond 100%

        self.pan_x = 0
        self.pan_y = 0
        self.display_image()

    def zoom_in(self):
        """Zoom in on the image"""
        if self.original_pixmap:
            self.apply_zoom_factor(1.2)

    def zoom_out(self):
        """Zoom out from the image"""
        if self.original_pixmap:
            self.zoom = max(self.zoom / 1.2, 0.05)
            # Reset pan if zoomed out
            if not self.can_pan(int(self.original_pixmap.width() * self.zoom), int(self.original_pixmap.height() * self.zoom)):
                self.pan_x = 0
                self.pan_y = 0
            self.display_image()


class ViewfinderWidget(QFrame):
    """Mini-map that shows the current viewport while zoomed."""

    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.setFixedSize(170, 170)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 0.78); border: 1px solid #444; border-radius: 6px;")
        self.setCursor(Qt.CrossCursor)
        self.canvas.view_changed.connect(self.update)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        state = self.canvas.get_view_state()
        if not state:
            painter.setPen(QColor("#666"))
            painter.drawText(self.rect(), Qt.AlignCenter, "View")
            return

        pixmap = state["pixmap"]
        thumb = pixmap.scaled(self.width() - 12, self.height() - 12, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        thumb_x = (self.width() - thumb.width()) // 2
        thumb_y = (self.height() - thumb.height()) // 2
        thumb_rect = QRect(thumb_x, thumb_y, thumb.width(), thumb.height())

        painter.drawPixmap(thumb_rect, thumb)
        painter.fillRect(thumb_rect, QColor(0, 0, 0, 45))

        image_width = max(1, state["image_width"])
        image_height = max(1, state["image_height"])
        viewport_rect = QRect(
            int(thumb_rect.x() + (state["origin_x"] / image_width) * thumb_rect.width()),
            int(thumb_rect.y() + (state["origin_y"] / image_height) * thumb_rect.height()),
            max(6, int((state["viewport_width"] / image_width) * thumb_rect.width())),
            max(6, int((state["viewport_height"] / image_height) * thumb_rect.height())),
        )

        painter.fillRect(viewport_rect, QColor(0, 170, 255, 55))
        painter.setPen(QColor("#4FC3F7"))
        painter.drawRect(viewport_rect)

    def mousePressEvent(self, event):
        self.center_canvas(event.position().toPoint())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.center_canvas(event.position().toPoint())

    def center_canvas(self, point):
        state = self.canvas.get_view_state()
        if not state:
            return

        pixmap = state["pixmap"]
        thumb = pixmap.scaled(self.width() - 12, self.height() - 12, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        thumb_x = (self.width() - thumb.width()) // 2
        thumb_y = (self.height() - thumb.height()) // 2
        if thumb.width() <= 0 or thumb.height() <= 0:
            return

        ratio_x = (point.x() - thumb_x) / thumb.width()
        ratio_y = (point.y() - thumb_y) / thumb.height()
        self.canvas.set_view_center_from_ratio(ratio_x, ratio_y)


class DragDropListWidget(QListWidget):
    """Custom QListWidget with drag and drop support"""
    files_dropped = Signal(list)  # Signal to emit when files are dropped
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.app = None  # Will be set by parent
    
    def dragEnterEvent(self, event):
        """Handle drag enter event"""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
    
    def dragMoveEvent(self, event):
        """Handle drag move event"""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
    
    def dropEvent(self, event):
        """Handle drop event for files and folders"""
        if not event.mimeData().hasUrls():
            return
        
        paths = []
        
        # Extract all dropped items
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if os.path.isdir(file_path):
                # It's a folder - recursively add all supported image files
                for root, _, files in os.walk(file_path):
                    for file in files:
                        if file.lower().endswith(SUPPORTED):
                            paths.append(os.path.join(root, file))
            else:
                # It's a file - check if it's supported
                if file_path.lower().endswith(SUPPORTED):
                    paths.append(file_path)
        
        # Emit signal with collected paths
        if paths:
            self.files_dropped.emit(sorted(paths))
        
        event.accept()


class ImageConverterProPySide(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Converter Pro - Modern Edition")
        self.setGeometry(100, 100, 1600, 900)
        self.minsize = (1200, 700)
        self.setMinimumSize(*self.minsize)
        self.theme_mode = "Dark"
        self.theme_actions = {}
        self.custom_sound_path = None
        self.setStyleSheet(self.get_stylesheet())
        
        # State variables
        self.files = []
        self.index = 0
        self.current = None
        self.current_original = None
        self.current_preview = None
        self.output = None
        self.output_is_custom = False
        self.show_original_preview = False
        self.processing = False
        self.worker = None
        self.preview_job = None
        self.thumbnail_cache = {}
        self.preview_source_cache = {}
        self.image_info_cache = {}
        
        # Store edits per image (non-destructive)
        self.image_edits = {}
        self.sync_all = False
        self.preview_high_res = False
        self.preview_resolution_mode = "Balanced"
        self.preview_resolution_actions = {}
        self.hover_animations_enabled = True
        self.click_animations_enabled = True
        self.status_animations_enabled = True
        self.sound_enabled = False
        self.sound_choice = "System Default"
        self.button_shadow_animations = {}
        self.button_click_animations = {}
        self.status_animation = None
        self.watermark_repo_url = "https://github.com/Dev8Github/WaterMarkPro"
        
        # GPU support
        self.system_gpu_names = self.detect_system_gpus()
        self.gpu_backend_error = None
        self.gpu_available = self.detect_gpu()
        self.use_gpu = self.gpu_available  # Auto-enable if available
        
        # Presets
        self.preset_folder = Path.home() / 'Documents' / 'ImageConverterPro'
        self.presets = {}
        
        # Export formats (for multi-format support)
        self.export_formats = ["JPG"]

        # Developer debug log
        self.dev_log_enabled = False
        self.dev_messages = []
        self.dev_log_dialog = None
        
        # Build UI
        self.build_ui()
        self.setup_ui_effects()
        self.apply_theme()
        self.load_presets()
        self.setup_shortcuts()
    
    def detect_gpu(self):
        """Detect available GPU acceleration"""
        gpu_info = []
        
        if CUPY_AVAILABLE and self.is_cupy_runtime_usable():
            gpu_info.append("CuPy")
        
        if gpu_info:
            print(f"GPU acceleration available: {', '.join(gpu_info)}")
            return True
        else:
            if self.system_gpu_names:
                backend_detail = self.format_gpu_backend_error(self.gpu_backend_error)
                if backend_detail:
                    detail = f"CuPy is installed but unavailable ({backend_detail})"
                else:
                    detail = "no supported GPU backend is installed"

                if OPENCV_AVAILABLE:
                    print(f"Detected GPU hardware: {', '.join(self.system_gpu_names)}; {detail}, using CPU (OpenCV available for CPU-side processing)")
                else:
                    print(f"Detected GPU hardware: {', '.join(self.system_gpu_names)}; {detail}, using CPU")
            elif OPENCV_AVAILABLE:
                print("No GPU acceleration available, using CPU (OpenCV available for CPU-side processing)")
            else:
                print("No GPU acceleration available, using CPU")
            return False

    def is_cupy_runtime_usable(self):
        """Verify that CuPy can execute kernels, not just detect a GPU."""
        if not CUPY_AVAILABLE or cp is None:
            return False

        try:
            probe = cp.asarray([1], dtype=cp.float32)
            _ = cp.asnumpy(probe * 2)
            self.gpu_backend_error = None
            return True
        except Exception as exc:
            self.gpu_backend_error = str(exc)
            print(f"CuPy runtime probe failed, falling back to CPU: {exc}")
            return False

    def format_gpu_backend_error(self, error):
        """Convert backend errors into short UI-friendly explanations."""
        if not error:
            return ""

        error_text = str(error)
        lowered = error_text.lower()
        if "nvrtc" in lowered:
            return "missing CUDA runtime compiler (nvrtc.dll)"
        return error_text

    def disable_gpu_acceleration(self, reason, update_status=True):
        """Disable GPU usage after a runtime failure and switch safely to CPU."""
        reason_text = self.format_gpu_backend_error(reason)
        self.gpu_backend_error = str(reason)
        self.gpu_available = False
        self.use_gpu = False
        if hasattr(self, "_gpu_logged"):
            delattr(self, "_gpu_logged")

        message = f"GPU acceleration disabled, using CPU: {reason_text}"
        print(message)
        self.log_debug(message)

        if hasattr(self, "gpu_checkbox"):
            self.gpu_checkbox.blockSignals(True)
            self.gpu_checkbox.setChecked(False)
            self.gpu_checkbox.setEnabled(False)
            self.gpu_checkbox.setToolTip(message)
            self.gpu_checkbox.blockSignals(False)

        if update_status and hasattr(self, "status_log"):
            self.log_status("GPU unavailable, switched to CPU", "orange")

    def detect_system_gpus(self):
        """Detect installed GPU hardware even if acceleration libraries are unavailable."""
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join '|'",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            names = [name.strip() for name in result.stdout.strip().split("|") if name.strip()]
            return names
        except Exception:
            return []

    def get_runtime_base_dirs(self):
        """Return likely application directories for scripts and frozen builds."""
        dirs = []
        if getattr(sys, "frozen", False):
            dirs.append(Path(sys.executable).resolve().parent)
        dirs.append(Path(__file__).resolve().parent)
        dirs.append(Path.cwd().resolve())

        unique_dirs = []
        seen = set()
        for directory in dirs:
            key = str(directory).lower()
            if key not in seen:
                seen.add(key)
                unique_dirs.append(directory)
        return unique_dirs

    def get_python_launch_command(self):
        """Return a usable Python command for launching companion scripts."""
        if getattr(sys, "frozen", False):
            return [sys.executable]
        return [sys.executable or "python"]

    def find_companion_executable(self, exe_name):
        """Locate a companion executable stored beside this app."""
        for directory in self.get_runtime_base_dirs():
            candidate = directory / exe_name
            if candidate.exists():
                return candidate
        return None

    def open_watermarkpro(self):
        """Launch WaterMarkPro when it exists beside this app."""
        companion = self.find_companion_executable("WaterMarkPro.exe")
        if companion:
            try:
                subprocess.Popen([str(companion)], cwd=str(companion.parent))
                self.log_status("Opening WaterMarkPro", "blue")
                return
            except Exception as exc:
                QMessageBox.critical(self, "WaterMarkPro", f"Failed to launch WaterMarkPro:\n{exc}")
                return

        self.show_companion_app_missing_dialog(
            title="WaterMarkPro Not Found",
            message="WaterMarkPro.exe was not found beside ImageConvertPro.",
            url=self.watermark_repo_url,
        )

    def show_companion_app_missing_dialog(self, title, message, url):
        """Show a small dialog with a clickable link when a companion app is missing."""
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setModal(True)
        dialog.resize(460, 170)

        layout = QVBoxLayout(dialog)
        info = QLabel(message)
        info.setWordWrap(True)
        layout.addWidget(info)

        link = QLabel(f'<a href="{url}">{url}</a>')
        link.setOpenExternalLinks(True)
        link.setTextInteractionFlags(Qt.TextBrowserInteraction)
        layout.addWidget(link)

        buttons = QHBoxLayout()
        buttons.addStretch()
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

        dialog.exec()

    def launch_nicegui_edition(self):
        """Launch the separate NiceGUI edition when available."""
        script_path = None
        for directory in self.get_runtime_base_dirs():
            candidate = directory / "ImageConvertPro_nicegui.py"
            if candidate.exists():
                script_path = candidate
                break

        if script_path is None:
            QMessageBox.information(
                self,
                "NiceGUI Edition",
                "ImageConvertPro_nicegui.py was not found beside this app.\n\nA separate NiceGUI edition is required because NiceGUI and PySide6 use different UI runtimes.",
            )
            return

        try:
            subprocess.Popen(self.get_python_launch_command() + [str(script_path)], cwd=str(script_path.parent))
            self.log_status("Launching NiceGUI edition", "blue")
        except Exception as exc:
            QMessageBox.critical(self, "NiceGUI Edition", f"Failed to launch NiceGUI edition:\n{exc}")

    def setup_ui_effects(self):
        """Register interactive UI effects without changing program behavior."""
        for button in self.findChildren(QPushButton):
            button.setCursor(Qt.PointingHandCursor)
            if not button.property("ui_effects_registered"):
                button.installEventFilter(self)
                button.clicked.connect(lambda checked=False, b=button: self.on_button_activated(b))
                button.setProperty("ui_effects_registered", True)

        if hasattr(self, "status_log") and not self.status_log.graphicsEffect():
            effect = QGraphicsOpacityEffect(self.status_log)
            effect.setOpacity(1.0)
            self.status_log.setGraphicsEffect(effect)

    def eventFilter(self, watched, event):
        """Animate buttons on hover and click when UI effects are enabled."""
        if isinstance(watched, QPushButton):
            if event.type() == QEvent.Enter and self.hover_animations_enabled:
                self.animate_button_shadow(watched, 18.0)
            elif event.type() == QEvent.Leave:
                self.animate_button_shadow(watched, 0.0)
            elif event.type() == QEvent.MouseButtonPress and self.click_animations_enabled:
                self.animate_button_shadow(watched, 28.0)
        return super().eventFilter(watched, event)

    def ensure_button_shadow(self, button):
        """Attach a reusable shadow effect to a button."""
        effect = button.graphicsEffect()
        if not isinstance(effect, QGraphicsDropShadowEffect):
            effect = QGraphicsDropShadowEffect(button)
            effect.setBlurRadius(0.0)
            effect.setOffset(0, 0)
            effect.setColor(QColor(55, 194, 255, 150))
            button.setGraphicsEffect(effect)
        return effect

    def animate_button_shadow(self, button, target_blur):
        """Animate button shadow blur for hover/click feedback."""
        effect = self.ensure_button_shadow(button)
        animation = QPropertyAnimation(effect, b"blurRadius", button)
        animation.setDuration(140)
        animation.setStartValue(effect.blurRadius())
        animation.setEndValue(target_blur)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        self.button_shadow_animations[button] = animation
        animation.start()

    def on_button_activated(self, button):
        """Play optional button effects when a button is clicked."""
        if self.sound_enabled:
            self.play_ui_sound()

        if self.click_animations_enabled:
            self.animate_button_shadow(button, 30.0)
            QTimer.singleShot(110, lambda b=button: self.animate_button_shadow(b, 18.0 if b.underMouse() and self.hover_animations_enabled else 0.0))

    def play_ui_sound(self):
        """Play the selected UI sound effect."""
        if winsound is not None:
            if self.sound_choice == "Custom WAV" and self.custom_sound_path and os.path.exists(self.custom_sound_path):
                winsound.PlaySound(self.custom_sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                return
            sound_map = {
                "System Default": -1,
                "Asterisk": winsound.MB_ICONASTERISK,
                "Exclamation": winsound.MB_ICONEXCLAMATION,
                "Question": winsound.MB_ICONQUESTION,
                "Ok": winsound.MB_OK,
            }
            winsound.MessageBeep(sound_map.get(self.sound_choice, -1))
        else:
            QApplication.beep()

    def animate_status_log(self):
        """Pulse the status label for feedback during UI actions and processing."""
        if not self.status_animations_enabled or not hasattr(self, "status_log"):
            return

        effect = self.status_log.graphicsEffect()
        if not isinstance(effect, QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect(self.status_log)
            effect.setOpacity(1.0)
            self.status_log.setGraphicsEffect(effect)

        animation = QPropertyAnimation(effect, b"opacity", self.status_log)
        animation.setDuration(180)
        animation.setStartValue(0.68)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        self.status_animation = animation
        animation.start()

    def set_sound_enabled(self, checked):
        """Toggle UI sound effects."""
        self.sound_enabled = bool(checked)
        self.log_status(f"Sound effects {'enabled' if self.sound_enabled else 'disabled'}", "blue")

    def set_hover_animations_enabled(self, checked):
        """Toggle hover animations."""
        self.hover_animations_enabled = bool(checked)
        if not checked:
            for button in self.findChildren(QPushButton):
                self.animate_button_shadow(button, 0.0)

    def set_click_animations_enabled(self, checked):
        """Toggle click pop animations."""
        self.click_animations_enabled = bool(checked)

    def set_status_animations_enabled(self, checked):
        """Toggle status/loading pulse animations."""
        self.status_animations_enabled = bool(checked)

    def set_sound_choice(self, sound_name):
        """Update the active button sound."""
        if sound_name == "Custom WAV" and not self.custom_sound_path:
            self.browse_custom_sound_file()
            return
        self.sound_choice = sound_name
        if hasattr(self, "sound_actions") and sound_name in self.sound_actions:
            self.sound_actions[sound_name].setChecked(True)
        if self.sound_enabled:
            self.play_ui_sound()

    def browse_custom_sound_file(self):
        """Let the user choose a custom WAV file for UI sounds."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select WAV File",
            "",
            "WAV files (*.wav)",
        )
        if not file_path:
            return

        self.custom_sound_path = file_path
        self.sound_choice = "Custom WAV"
        if hasattr(self, "sound_actions") and "Custom WAV" in self.sound_actions:
            self.sound_actions["Custom WAV"].setEnabled(True)
            self.sound_actions["Custom WAV"].setChecked(True)
        self.log_status(f"Custom sound loaded: {os.path.basename(file_path)}", "green")
        if self.sound_enabled:
            self.play_ui_sound()

    def set_theme_mode(self, theme_name):
        """Switch between Dark, Light, and System Default themes."""
        self.theme_mode = theme_name
        self.apply_theme()
        self.log_status(f"Theme: {theme_name}", "blue")

    def get_resolved_theme(self):
        """Return the active concrete theme after resolving system default."""
        if self.theme_mode == "System Default":
            return "Dark" if self.is_system_dark() else "Light"
        return self.theme_mode

    def is_system_dark(self):
        """Infer whether the current system/app palette is dark."""
        try:
            return QApplication.palette().window().color().lightness() < 128
        except Exception:
            return True

    def apply_theme(self):
        """Apply the selected application theme and sync menu actions."""
        self.setStyleSheet(self.get_stylesheet())
        if self.theme_actions and self.theme_mode in self.theme_actions:
            self.theme_actions[self.theme_mode].setChecked(True)
        resolved = self.get_resolved_theme()
        if resolved == "Light":
            panel_bg = "#ffffff"
            panel_border = "#cfd8e3"
            muted_text = "#5d6b7d"
            status_bg = "#f3f7fd"
        else:
            panel_bg = "#151b23"
            panel_border = "#293241"
            muted_text = "#9aa6b2"
            status_bg = "#101720"

        if hasattr(self, "info_label"):
            self.info_label.setStyleSheet(f"color: #4CAF50; padding: 8px; background-color: {panel_bg}; border: 1px solid {panel_border}; border-radius: 8px;")
        if hasattr(self, "zoom_label"):
            self.zoom_label.setStyleSheet(f"color: inherit; background-color: {panel_bg}; padding: 6px 10px; border: 1px solid {panel_border}; border-radius: 6px;")
        if hasattr(self, "img_info"):
            self.img_info.setStyleSheet(f"color: {muted_text};")
        if hasattr(self, "status_log"):
            self.status_log.setStyleSheet(f"color: inherit; background-color: {status_bg}; padding: 6px 10px; border-radius: 8px; border: 1px solid {panel_border};")
        if hasattr(self, "sliders"):
            for slider_widget in self.sliders.values():
                slider_widget.refresh_style(resolved)
    
    def get_stylesheet(self):
        resolved_theme = self.get_resolved_theme()

        if resolved_theme == "Light":
            colors = {
                "window": "#eef3f8",
                "surface": "#ffffff",
                "surface_alt": "#f7f9fc",
                "surface_hover": "#edf3ff",
                "border": "#cfd8e3",
                "accent": "#2563eb",
                "text": "#475569",
                "muted": "#5d6b7d",
                "menu_bar": "#ffffff",
                "menu_bg": "#ffffff",
                "menu_selected": "#e8f0ff",
                "scroll": "#bcc8d6",
                "progress_text": "#475569",
            }
        else:
            colors = {
                "window": "#0d1117",
                "surface": "#151b23",
                "surface_alt": "#0f141a",
                "surface_hover": "#1d2633",
                "border": "#293241",
                "accent": "#4c6fff",
                "text": "#e5e7eb",
                "muted": "#9aa6b2",
                "menu_bar": "#11161d",
                "menu_bg": "#11161d",
                "menu_selected": "#1d2633",
                "scroll": "#334155",
                "progress_text": "#f5f7fb",
            }

        return f"""
            QMainWindow {{
                background-color: {colors['window']};
                border: none;
            }}

            QWidget {{
                color: {colors['text']};
                font-family: 'Segoe UI', sans-serif;
            }}

            QMenuBar {{
                background-color: {colors['menu_bar']};
                color: {colors['text']};
                border-bottom: 1px solid {colors['border']};
                padding: 4px;
            }}

            QMenuBar::item {{
                background: transparent;
                padding: 8px 14px;
                border-radius: 8px;
                margin: 2px;
            }}

            QMenuBar::item:selected {{
                background-color: {colors['surface_hover']};
            }}

            QMenu {{
                background-color: {colors['menu_bg']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
                padding: 6px;
            }}

            QMenu::item {{
                padding: 8px 28px 8px 12px;
                border-radius: 6px;
                margin: 1px 0;
            }}

            QMenu::item:selected {{
                background-color: {colors['menu_selected']};
            }}

            QMenu::separator {{
                height: 1px;
                background: {colors['border']};
                margin: 6px 8px;
            }}

            QLabel {{
                color: {colors['text']};
            }}

            QPushButton {{
                background-color: {colors['surface']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
                padding: 9px 16px;
                border-radius: 8px;
                font-size: 11px;
                font-weight: 600;
            }}

            QPushButton:hover {{
                background-color: {colors['surface_hover']};
                border: 1px solid {colors['accent']};
            }}

            QPushButton:pressed {{
                background-color: {colors['surface_alt']};
            }}

            QComboBox, QSpinBox {{
                background-color: {colors['surface']};
                border: 1px solid {colors['border']};
                border-radius: 8px;
                padding: 6px 10px;
                color: {colors['text']};
                font-size: 11px;
            }}

            QComboBox:hover, QSpinBox:hover {{
                border: 1px solid {colors['accent']};
            }}

            QComboBox::drop-down {{
                border: none;
                padding-right: 8px;
            }}

            QComboBox QAbstractItemView {{
                background-color: {colors['menu_bg']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
                selection-background-color: {colors['menu_selected']};
            }}

            QListWidget {{
                background-color: {colors['surface_alt']};
                border: 1px solid {colors['border']};
                border-radius: 10px;
                color: {colors['text']};
                font-size: 11px;
                selection-background-color: {colors['menu_selected']};
            }}

            QListWidget::item:hover {{
                background-color: {colors['surface_hover']};
            }}

            QScrollArea {{
                border: none;
                background-color: transparent;
            }}

            QScrollBar:vertical {{
                background-color: {colors['surface_alt']};
                width: 8px;
                border: none;
            }}

            QScrollBar::handle:vertical {{
                background-color: {colors['scroll']};
                border-radius: 4px;
            }}

            QScrollBar::handle:vertical:hover {{
                background-color: {colors['accent']};
            }}

            QProgressBar {{
                background-color: {colors['surface']};
                border: 1px solid {colors['border']};
                border-radius: 8px;
                color: {colors['progress_text']};
                text-align: center;
                height: 20px;
            }}

            QProgressBar::chunk {{
                background-color: {colors['accent']};
                border-radius: 6px;
            }}

            QGroupBox {{
                border: 1px solid {colors['border']};
                border-radius: 10px;
                margin-top: 8px;
                padding-top: 8px;
                font-weight: 600;
                color: {colors['text']};
                font-size: 11px;
            }}

            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                left: 8px;
            }}

            QCheckBox {{
                color: {colors['text']};
                font-size: 11px;
                spacing: 6px;
            }}

            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border-radius: 3px;
            }}

            QCheckBox::indicator:unchecked {{
                background-color: {colors['surface']};
                border: 1px solid {colors['scroll']};
            }}

            QCheckBox::indicator:unchecked:hover {{
                border: 1px solid {colors['accent']};
            }}

            QCheckBox::indicator:checked {{
                background-color: {colors['accent']};
                border: 1px solid {colors['accent']};
            }}

            QSlider::groove:horizontal {{
                background-color: {colors['border']};
                height: 4px;
                border-radius: 2px;
            }}

            QSlider::handle:horizontal {{
                background-color: {colors['text']};
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
                border: 1px solid {colors['border']};
            }}

            QSlider::handle:horizontal:hover {{
                background-color: {colors['accent']};
            }}
        """
    
    def build_ui(self):
        """Build minimalist UI"""
        self.build_menu_bar()
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Splitter for panels
        splitter = QSplitter(Qt.Horizontal)

        # LEFT PANEL (Files) - Minimalist
        left_panel = self.build_left_panel()
        splitter.addWidget(left_panel)

        # MIDDLE PANEL (Viewer) - Clean
        middle_panel = self.build_middle_panel()
        splitter.addWidget(middle_panel)

        # RIGHT PANEL (Controls) - Compact
        right_panel = self.build_right_panel()
        splitter.addWidget(right_panel)

        # Set sizes
        splitter.setSizes([180, 800, 280])
        splitter.setCollapsible(0, True)
        splitter.setCollapsible(1, False)
        splitter.setCollapsible(2, True)

        self.create_dev_log_dialog()
        main_layout.addWidget(splitter)
    
    def build_menu_bar(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        action_open = QAction("Open Files...", self)
        action_open.setShortcut("Ctrl+O")
        action_open.triggered.connect(self.add_files)
        file_menu.addAction(action_open)

        action_folder = QAction("Import Folder...", self)
        action_folder.setShortcut("Ctrl+F")
        action_folder.triggered.connect(self.add_folder)
        file_menu.addAction(action_folder)

        file_menu.addSeparator()

        action_shortcuts = QAction("Shortcuts", self)
        action_shortcuts.setShortcut("F1")
        action_shortcuts.triggered.connect(self.show_shortcuts_help)
        file_menu.addAction(action_shortcuts)

        action_close = QAction("Close", self)
        action_close.setShortcut("Ctrl+Q")
        action_close.triggered.connect(self.close)
        file_menu.addAction(action_close)

        window_menu = menu_bar.addMenu("&Windows")
        preview_menu = window_menu.addMenu("Preview Resolution")
        preview_group = QActionGroup(self)
        preview_group.setExclusive(True)

        for mode in PREVIEW_RESOLUTION_PRESETS:
            action = QAction(mode, self)
            action.setCheckable(True)
            action.setData(mode)
            action.triggered.connect(self.on_preview_resolution_action)
            preview_group.addAction(action)
            preview_menu.addAction(action)
            self.preview_resolution_actions[mode] = action

        window_menu.addSeparator()
        self.action_hover_animations = QAction("Hover Animations", self)
        self.action_hover_animations.setCheckable(True)
        self.action_hover_animations.setChecked(self.hover_animations_enabled)
        self.action_hover_animations.triggered.connect(self.set_hover_animations_enabled)
        window_menu.addAction(self.action_hover_animations)

        self.action_click_animations = QAction("Click Pop Animations", self)
        self.action_click_animations.setCheckable(True)
        self.action_click_animations.setChecked(self.click_animations_enabled)
        self.action_click_animations.triggered.connect(self.set_click_animations_enabled)
        window_menu.addAction(self.action_click_animations)

        self.action_status_animations = QAction("Status Pulse Animation", self)
        self.action_status_animations.setCheckable(True)
        self.action_status_animations.setChecked(self.status_animations_enabled)
        self.action_status_animations.triggered.connect(self.set_status_animations_enabled)
        window_menu.addAction(self.action_status_animations)

        window_menu.addSeparator()

        self.action_sound_enabled = QAction("Enable Sound Effects", self)
        self.action_sound_enabled.setCheckable(True)
        self.action_sound_enabled.setChecked(self.sound_enabled)
        self.action_sound_enabled.triggered.connect(self.set_sound_enabled)
        window_menu.addAction(self.action_sound_enabled)

        sound_menu = window_menu.addMenu("Button Sound")
        sound_group = QActionGroup(self)
        sound_group.setExclusive(True)
        self.sound_actions = {}
        for sound_name in ("System Default", "Asterisk", "Exclamation", "Question", "Ok", "Custom WAV"):
            action = QAction(sound_name, self)
            action.setCheckable(True)
            action.setData(sound_name)
            action.triggered.connect(lambda checked=False, name=sound_name: self.set_sound_choice(name))
            if sound_name == "Custom WAV" and not self.custom_sound_path:
                action.setEnabled(False)
            sound_group.addAction(action)
            sound_menu.addAction(action)
            self.sound_actions[sound_name] = action
        self.sound_actions[self.sound_choice].setChecked(True)
        action_browse_sound = QAction("Browse Custom WAV...", self)
        action_browse_sound.triggered.connect(self.browse_custom_sound_file)
        window_menu.addAction(action_browse_sound)

        window_menu.addSeparator()
        theme_menu = window_menu.addMenu("Theme")
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        for theme_name in ("Light", "Dark", "System Default"):
            action = QAction(theme_name, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked=False, name=theme_name: self.set_theme_mode(name))
            theme_group.addAction(action)
            theme_menu.addAction(action)
            self.theme_actions[theme_name] = action
        theme_menu.addSeparator()
        action_launch_nicegui = QAction("Launch NiceGUI Edition", self)
        action_launch_nicegui.triggered.connect(self.launch_nicegui_edition)
        theme_menu.addAction(action_launch_nicegui)

        action_open_watermark = QAction("&WaterMarkPro", self)
        action_open_watermark.triggered.connect(self.open_watermarkpro)
        menu_bar.addAction(action_open_watermark)

        help_menu = menu_bar.addMenu("&Help")
        action_toggle_dev = QAction("Toggle Developer Log", self)
        action_toggle_dev.setShortcut("Ctrl+Shift+D")
        action_toggle_dev.setCheckable(True)
        action_toggle_dev.triggered.connect(self.toggle_dev_log)
        help_menu.addAction(action_toggle_dev)

        action_program_about = QAction("About Program", self)
        action_program_about.triggered.connect(self.show_program_about)
        help_menu.addAction(action_program_about)

        action_about_app = QAction("About", self)
        action_about_app.triggered.connect(self.show_about)
        help_menu.addAction(action_about_app)

        self.sync_preview_resolution_ui()

    def create_dev_log_dialog(self):
        if self.dev_log_dialog:
            return

        self.dev_log_dialog = QDialog(self)
        self.dev_log_dialog.setWindowTitle("Developer Debug Log")
        self.dev_log_dialog.resize(720, 450)

        layout = QVBoxLayout(self.dev_log_dialog)
        self.dev_log_text = QTextEdit()
        self.dev_log_text.setReadOnly(True)
        self.dev_log_text.setStyleSheet("background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI';")
        layout.addWidget(self.dev_log_text)

        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.dev_log_dialog.hide)
        buttons_layout.addWidget(btn_close)
        layout.addLayout(buttons_layout)

        for message in self.dev_messages:
            self.dev_log_text.append(message)

    def toggle_dev_log(self, checked=False):
        self.dev_log_enabled = bool(checked)
        if self.dev_log_enabled:
            self.create_dev_log_dialog()
            self.dev_log_dialog.show()
            self.dev_log_dialog.raise_()
            self.dev_log_dialog.activateWindow()
            self.log_debug("Developer log enabled")
        else:
            if self.dev_log_dialog:
                self.dev_log_dialog.hide()
            self.log_debug("Developer log disabled")

    def show_program_about(self):
        """Show a fuller program overview with shortcuts and feature summary."""
        about_text = """
        <h3>Image Converter Pro</h3>
        <p>Image Converter Pro is an image conversion and adjustment tool built with PySide6.</p>
        <p>You can convert images between common formats, preview edits live, target file size, preserve metadata where supported, and apply non-destructive image adjustments before export.</p>

        <h4>Main Features</h4>
        <p>
        • Multi-format export<br>
        • Preview-based image adjustments<br>
        • Output size targeting<br>
        • EXIF-aware export where the format supports it<br>
        • Batch processing with per-image settings
        </p>

        <h4>Shortcuts</h4>
        <p>
        <b>Left / Right or A / D:</b> Previous / Next image<br>
        <b>Home / End:</b> First / Last image<br>
        <b>+ / =:</b> Zoom in<br>
        <b>-:</b> Zoom out<br>
        <b>F or 0:</b> Fit image<br>
        <b>1:</b> Actual size (100%)<br>
        <b>R or Ctrl+R:</b> Reset adjustments<br>
        <b>Ctrl+S:</b> Save preset<br>
        <b>Ctrl+O:</b> Open files<br>
        <b>Delete:</b> Delete selected files<br>
        <b>Ctrl+E:</b> Export selected images<br>
        <b>Ctrl+Shift+E:</b> Export all images<br>
        <b>Ctrl+A:</b> Select all files<br>
        <b>Ctrl+Shift+A:</b> Deselect all files<br>
        <b>F1:</b> Show shortcuts
        </p>
        """

        dialog = QMessageBox(self)
        dialog.setWindowTitle("About Program")
        dialog.setTextFormat(Qt.RichText)
        dialog.setText(about_text)
        dialog.setStandardButtons(QMessageBox.Ok)
        dialog.exec()

    def show_about(self):
        QMessageBox.information(
            self,
            "About Image Converter Pro",
            "Image Converter Pro - Modern Edition\n\nImage conversion, preview adjustments, and export optimization in one workspace."
        )

    def log_debug(self, message):
        self.dev_messages.append(message)
        print(message)
        if self.dev_log_enabled and self.dev_log_dialog:
            self.dev_log_text.append(message)

    def build_left_panel(self):
        """Left panel: minimalist file management"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Simple title
        title = QLabel("Files")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout.addWidget(title)

        # File buttons - compact
        btn_add_files = QPushButton("Add Files")
        btn_add_files.clicked.connect(self.add_files)
        layout.addWidget(btn_add_files)

        btn_add_folder = QPushButton("Add Folder")
        btn_add_folder.clicked.connect(self.add_folder)
        layout.addWidget(btn_add_folder)

        # File list - clean
        self.file_list = DragDropListWidget()
        self.file_list.setMinimumHeight(200)
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)  # OS-style multi-select
        self.file_list.setSelectionBehavior(QListWidget.SelectRows)
        self.file_list.setIconSize(QSize(48, 48))
        self.file_list.setSpacing(4)
        self.file_list.itemClicked.connect(self.on_file_clicked)
        self.file_list.itemSelectionChanged.connect(self.on_file_selected)
        self.file_list.files_dropped.connect(self.on_files_dropped)
        layout.addWidget(self.file_list, 1)

        # Action buttons - horizontal
        action_layout = QHBoxLayout()
        action_layout.setSpacing(4)

        btn_delete = QPushButton("Delete")
        btn_delete.clicked.connect(self.delete_selected_files)
        action_layout.addWidget(btn_delete)

        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self.clear_files)
        action_layout.addWidget(btn_clear)

        layout.addLayout(action_layout)

        self.info_label = QLabel("0 files")
        self.info_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.info_label.setStyleSheet("color: #4CAF50; padding: 8px; background-color: #1a1a1a; border: 1px solid #333; border-radius: 4px; text-align: center;")
        layout.addWidget(self.info_label)

        # Help button
        btn_help = QPushButton("?")
        btn_help.setMaximumWidth(30)
        btn_help.setToolTip("Keyboard Shortcuts (F1)")
        btn_help.clicked.connect(self.show_shortcuts_help)
        layout.addWidget(btn_help)

        return panel
    
    def build_middle_panel(self):
        """Middle panel: minimalist image viewer"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.canvas = ImageCanvas()
        self.canvas.files_dropped.connect(self.on_files_dropped)
        self.canvas.view_changed.connect(self.update_zoom_label)
        layout.addWidget(self.canvas, 1)

        viewfinder_layout = QHBoxLayout()
        self.zoom_label = QLabel("100%")
        self.zoom_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.zoom_label.setStyleSheet("color: #e0e0e0; background-color: #1a1a1a; padding: 6px 10px; border: 1px solid #333; border-radius: 6px;")
        viewfinder_layout.addWidget(self.zoom_label, 0, Qt.AlignLeft)
        viewfinder_layout.addStretch()
        self.viewfinder = ViewfinderWidget(self.canvas)
        viewfinder_layout.addWidget(self.viewfinder)
        layout.addLayout(viewfinder_layout)

        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(4)

        btn_prev = QPushButton("Prev")
        btn_prev.clicked.connect(self.prev_image)
        nav_layout.addWidget(btn_prev)

        btn_next = QPushButton("Next")
        btn_next.clicked.connect(self.next_image)
        nav_layout.addWidget(btn_next)

        btn_zoom_in = QPushButton("Zoom +")
        btn_zoom_in.clicked.connect(self.canvas.zoom_in)
        nav_layout.addWidget(btn_zoom_in)

        btn_zoom_out = QPushButton("Zoom -")
        btn_zoom_out.clicked.connect(self.canvas.zoom_out)
        nav_layout.addWidget(btn_zoom_out)

        btn_fit = QPushButton("Fit")
        btn_fit.clicked.connect(self.canvas.fit_image)
        nav_layout.addWidget(btn_fit)

        self.before_after_btn = QPushButton("Before")
        self.before_after_btn.setCheckable(True)
        self.before_after_btn.toggled.connect(self.on_before_after_toggled)
        nav_layout.addWidget(self.before_after_btn)

        layout.addLayout(nav_layout)

        self.status_log = QLabel("Ready")
        self.status_log.setFont(QFont("Segoe UI", 9))
        self.status_log.setStyleSheet("color: #0D47A1; background-color: #f0f0f0; padding: 4px; border-radius: 2px;")
        layout.addWidget(self.status_log)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setVisible(False)
        progress_layout = QHBoxLayout()
        progress_layout.setSpacing(6)
        progress_layout.addWidget(self.progress, 1)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setVisible(False)
        self.stop_btn.clicked.connect(self.stop_processing)
        progress_layout.addWidget(self.stop_btn)
        layout.addLayout(progress_layout)

        self.img_info = QLabel("No image selected")
        self.img_info.setFont(QFont("Segoe UI", 8))
        self.img_info.setStyleSheet("color: #666;")
        layout.addWidget(self.img_info)

        return panel
    
    def build_right_panel(self):
        """Right panel: minimalist controls"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header
        header_layout = QHBoxLayout()
        header_layout.setSpacing(6)
        title = QLabel("Adjustments")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        header_layout.addWidget(title)
        header_layout.addStretch()

        btn_auto = QPushButton("Auto")
        btn_auto.clicked.connect(self.auto_adjust_current_image)
        header_layout.addWidget(btn_auto)

        btn_reset = QPushButton("Reset All")
        btn_reset.clicked.connect(self.reset_sliders)
        header_layout.addWidget(btn_reset)
        layout.addLayout(header_layout)

        # Scrollable adjustments
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_content = QWidget()
        adj_layout = QVBoxLayout(scroll_content)
        # Sliders
        self.sliders = {}
        for param_id, label in ADJUSTMENT_SPECS:
            slider_widget = ModernSliderWidget(label, 0, 100, 50)
            slider_widget.value_changed.connect(self.on_slider_changed)
            self.sliders[param_id] = slider_widget
            adj_layout.addWidget(slider_widget)

        # Options
        self.sync_checkbox = QCheckBox("Sync All Images")
        adj_layout.addWidget(self.sync_checkbox)

        self.preview_checkbox = QCheckBox("High Res Preview")
        self.preview_checkbox.setChecked(self.preview_high_res)
        self.preview_checkbox.stateChanged.connect(self.on_preview_quality_changed)
        adj_layout.addWidget(self.preview_checkbox)

        # GPU toggle (only show if GPU is available)
        if self.gpu_available:
            self.gpu_checkbox = QCheckBox("Use GPU Acceleration")
            self.gpu_checkbox.setChecked(self.use_gpu)
            self.gpu_checkbox.stateChanged.connect(self.on_gpu_toggle_changed)
            adj_layout.addWidget(self.gpu_checkbox)
        elif self.system_gpu_names:
            backend_detail = self.format_gpu_backend_error(self.gpu_backend_error)
            if backend_detail:
                gpu_text = f"GPU detected: {', '.join(self.system_gpu_names)}\nGPU acceleration unavailable: {backend_detail}."
            else:
                gpu_text = f"GPU detected: {', '.join(self.system_gpu_names)}\nInstall a supported CUDA/CuPy backend to enable GPU acceleration."
            gpu_notice = QLabel(gpu_text)
            gpu_notice.setWordWrap(True)
            gpu_notice.setStyleSheet("color: #bbbbbb; font-size: 10px;")
            adj_layout.addWidget(gpu_notice)

        adj_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        # Presets section
        preset_layout = QHBoxLayout()
        preset_layout.setSpacing(4)

        self.preset_combo = QComboBox()
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        preset_layout.addWidget(self.preset_combo)

        btn_save_preset = QPushButton("Save")
        btn_save_preset.clicked.connect(self.save_preset)
        preset_layout.addWidget(btn_save_preset)

        btn_load_preset = QPushButton("Load")
        btn_load_preset.clicked.connect(self.apply_preset)
        preset_layout.addWidget(btn_load_preset)

        btn_delete_preset = QPushButton("Delete")
        btn_delete_preset.clicked.connect(self.delete_preset)
        preset_layout.addWidget(btn_delete_preset)

        layout.addLayout(preset_layout)

        # Export section
        export_title = QLabel("Export")
        export_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        layout.addWidget(export_title)

        format_label = QLabel("Formats:")
        layout.addWidget(format_label)
        
        self.format_checkboxes = {}
        format_grid = QGridLayout()
        format_grid.setHorizontalSpacing(12)
        format_grid.setVerticalSpacing(6)
        default_formats = {"JPG", "PNG", "WEBP"}

        for index, fmt in enumerate(EXPORT_FORMATS.keys()):
            checkbox = QCheckBox(fmt)
            checkbox.setChecked(fmt in default_formats)
            checkbox.stateChanged.connect(self.on_export_formats_changed)
            self.format_checkboxes[fmt] = checkbox
            format_grid.addWidget(checkbox, index // 2, index % 2)

        layout.addLayout(format_grid)
        
        # Format add/remove buttons
        format_btn_layout = QHBoxLayout()
        format_btn_layout.setSpacing(4)
        
        btn_add_format = QPushButton("➕ Add")
        btn_add_format.setToolTip("Add export format")
        btn_add_format.clicked.connect(self.add_export_format)
        btn_add_format.hide()
        format_btn_layout.addWidget(btn_add_format)
        
        btn_remove_format = QPushButton("➖ Remove")
        btn_remove_format.setToolTip("Remove selected format")
        btn_remove_format.clicked.connect(self.remove_export_format)
        btn_remove_format.hide()
        format_btn_layout.addWidget(btn_remove_format)
        
        layout.addLayout(format_btn_layout)
        
        # Format selector combo for adding
        add_format_label = QLabel("Select & Add Format:")
        add_format_label.setFont(QFont("Segoe UI", 9))
        add_format_label.hide()
        layout.addWidget(add_format_label)
        
        format_select_layout = QHBoxLayout()
        format_select_layout.setSpacing(4)
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(list(EXPORT_FORMATS.keys()))
        self.format_combo.setCurrentText("PNG")
        self.format_combo.hide()
        format_select_layout.addWidget(self.format_combo)
        
        # Quick add button next to combo
        btn_quick_add = QPushButton("✓")
        btn_quick_add.setMaximumWidth(40)
        btn_quick_add.setToolTip("Quick add format")
        btn_quick_add.clicked.connect(self.add_export_format)
        btn_quick_add.hide()
        format_select_layout.addWidget(btn_quick_add)
        
        layout.addLayout(format_select_layout)
        
        self.export_formats = self.get_selected_export_formats()

        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Quality:"))
        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setMinimum(10)
        self.quality_slider.setMaximum(100)
        self.quality_slider.setValue(85)
        self.quality_label = QLabel("85%")
        self.quality_slider.valueChanged.connect(lambda v: self.quality_label.setText(f"{v}%"))
        quality_layout.addWidget(self.quality_slider)
        quality_layout.addWidget(self.quality_label)
        layout.addLayout(quality_layout)

        # Target size
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Target Size:"))
        self.target_size_combo = QComboBox()
        self.target_size_combo.addItems(["No Limit", "100 KB", "500 KB", "1 MB", "5 MB", "Custom"])
        self.target_size_combo.currentTextChanged.connect(self.on_target_size_changed)
        target_layout.addWidget(self.target_size_combo)
        layout.addLayout(target_layout)

        # Custom size (hidden by default)
        self.custom_size_layout = QHBoxLayout()
        self.custom_size_layout.addWidget(QLabel("Size (MB):"))
        self.custom_size_spinbox = QSpinBox()
        self.custom_size_spinbox.setMinimum(1)
        self.custom_size_spinbox.setMaximum(100)
        self.custom_size_spinbox.setValue(1)
        self.custom_size_layout.addWidget(self.custom_size_spinbox)
        self.custom_size_layout.addStretch()
        self.custom_size_frame = QWidget()
        self.custom_size_frame.setLayout(self.custom_size_layout)
        self.custom_size_frame.setVisible(False)
        layout.addWidget(self.custom_size_frame)

        # Output folder
        btn_output = QPushButton("Output Folder")
        btn_output.setToolTip("Select custom output folder (default: 'Output' in source folder)")
        btn_output.clicked.connect(self.pick_output_folder)
        layout.addWidget(btn_output)

        # Process buttons
        process_layout = QHBoxLayout()
        process_layout.setSpacing(4)

        btn_selected = QPushButton("Process Selected")
        btn_selected.clicked.connect(self.process_selected)
        process_layout.addWidget(btn_selected)

        btn_all = QPushButton("Process All")
        btn_all.clicked.connect(self.process_all)
        process_layout.addWidget(btn_all)

        layout.addLayout(process_layout)

        return panel
    
    def on_slider_changed(self):
        """Handle slider value change"""
        self.schedule_preview_update()

    def update_zoom_label(self):
        """Refresh the zoom percentage readout."""
        if hasattr(self, "zoom_label") and self.zoom_label:
            self.zoom_label.setText(f"{self.canvas.get_zoom_percentage()}%")
    
    def on_preview_quality_changed(self):
        """Handle preview quality toggle"""
        mode = "High" if self.preview_checkbox.isChecked() else "Balanced"
        self.set_preview_resolution(mode, from_checkbox=True)

    def on_preview_resolution_action(self):
        """Apply the preview resolution chosen from the Windows menu."""
        action = self.sender()
        if action and action.isChecked():
            self.set_preview_resolution(action.data())

    def sync_preview_resolution_ui(self):
        """Sync the menu and checkbox controls to the current preview mode."""
        self.preview_high_res = self.preview_resolution_mode in ("High", "Ultra")

        if hasattr(self, "preview_checkbox"):
            self.preview_checkbox.blockSignals(True)
            self.preview_checkbox.setChecked(self.preview_high_res)
            self.preview_checkbox.blockSignals(False)

        action = self.preview_resolution_actions.get(self.preview_resolution_mode)
        if action:
            action.setChecked(True)

    def set_preview_resolution(self, mode, from_checkbox=False):
        """Set the preview resolution preset for the viewer."""
        if mode not in PREVIEW_RESOLUTION_PRESETS:
            return

        if self.preview_resolution_mode == mode and not from_checkbox:
            self.sync_preview_resolution_ui()
            return

        self.preview_resolution_mode = mode
        self.sync_preview_resolution_ui()
        self.log_status(f"Preview resolution: {mode}", "blue")
        if self.current_original:
            self.current_preview = self.get_preview_source(self.files[self.index], mode=mode)
            self.update_preview()

    def on_before_after_toggled(self, checked):
        """Toggle between original and adjusted preview."""
        self.show_original_preview = checked
        if hasattr(self, "before_after_btn"):
            self.before_after_btn.setText("After" if checked else "Before")
        self.update_preview()
    
    def on_gpu_toggle_changed(self):
        """Handle GPU toggle"""
        self.use_gpu = self.gpu_checkbox.isChecked()
        if self.use_gpu and not self.gpu_available:
            self.gpu_checkbox.blockSignals(True)
            self.gpu_checkbox.setChecked(False)
            self.gpu_checkbox.blockSignals(False)
            self.use_gpu = False
            self.log_status("GPU acceleration is not available in this environment", "orange")
            return
        gpu_status = "enabled" if self.use_gpu else "disabled"
        self.log_status(f"GPU acceleration {gpu_status}", "blue")
        # Update preview to use new setting
        self.update_preview()
    
    def on_target_size_changed(self, text):
        """Handle target size selection change"""
        self.custom_size_frame.setVisible(text == "Custom")

    def get_selected_export_formats(self):
        """Return the export formats selected in the checkbox grid."""
        if not hasattr(self, "format_checkboxes"):
            return ["JPG"]
        return [fmt for fmt, checkbox in self.format_checkboxes.items() if checkbox.isChecked()]

    def on_export_formats_changed(self, *_):
        """Sync checkbox state to the export format list."""
        self.export_formats = self.get_selected_export_formats()
        if self.export_formats:
            self.log_debug(f"Export formats: {', '.join(self.export_formats)}")
        else:
            self.log_status("Select at least one export format", "orange")
    
    def add_export_format(self):
        """Legacy helper retained for compatibility with older UI actions."""
        self.on_export_formats_changed()
    
    def remove_export_format(self):
        """Legacy helper retained for compatibility with older UI actions."""
        self.on_export_formats_changed()
    
    def on_file_selected(self):
        """Handle file selection changes and update selection status."""
        items = self.file_list.selectedItems()
        if not items:
            self.log_status("No files selected", "black")
            self.update_file_info()
            return

        if len(items) > 1:
            self.log_status(f"{len(items)} files selected", "blue")
        else:
            self.log_status("File selected", "blue")

        self.update_file_info()

    def update_file_info(self):
        """Update the file count and selection info label."""
        total = len(self.files)
        selected = len(self.file_list.selectedItems())

        if total == 0:
            self.info_label.setText("0 files")
        elif selected == 0:
            self.info_label.setText(f"{total} files")
        elif selected == total:
            self.info_label.setText(f"📋 All {total} files selected")
        else:
            self.info_label.setText(f"📁 {total} files | {selected} selected")

    def on_file_clicked(self, item):
        """Handle click events for the file list."""
        if not item:
            return

        self.index = self.file_list.row(item)
        self.load_current()

        if self.file_list.selectedItems():
            count = len(self.file_list.selectedItems())
            if count > 1:
                self.log_status(f"{count} files selected", "blue")
            else:
                self.log_status("File selected", "blue")
    
    def on_files_dropped(self, paths):
        """Handle files dropped via drag and drop"""
        self.log_debug(f"Files dropped: {len(paths)} items")
        self.add(paths)
        if self.files:
            self.index = len(self.files) - len(paths)  # Select first dropped file
            self.file_list.setCurrentRow(self.index)
            self.load_current()
    
    def on_preset_changed(self, text):
        """Handle preset selection"""
        if text and text in self.presets:
            self.apply_preset()

    def resolve_default_output_folder(self, paths=None, import_root=None):
        """Build the default Output folder beside the imported images."""
        if import_root:
            base_dir = import_root
        elif paths:
            parent_dirs = [os.path.dirname(os.path.abspath(path)) for path in paths]
            unique_dirs = list(dict.fromkeys(parent_dirs))
            if len(unique_dirs) == 1:
                base_dir = unique_dirs[0]
            else:
                try:
                    base_dir = os.path.commonpath(unique_dirs)
                except ValueError:
                    base_dir = unique_dirs[0]
        elif self.files:
            base_dir = os.path.dirname(os.path.abspath(self.files[0]))
        else:
            return None

        return os.path.join(base_dir, "Output")

    def update_default_output_folder(self, paths=None, import_root=None):
        """Keep the default output folder aligned with the imported location."""
        if self.output_is_custom:
            return

        output_dir = self.resolve_default_output_folder(paths=paths, import_root=import_root)
        if output_dir:
            self.output = output_dir
            os.makedirs(self.output, exist_ok=True)
    
    # ========== FILE OPERATIONS ==========
    def add_files(self):
        """Add files dialog"""
        self.log_debug("Opening file dialog to add files")
        filter_str = "All Images (" + " ".join(f"*{x}" for x in SUPPORTED) + ")"
        files, _ = QFileDialog.getOpenFileNames(
            self, "Open Images", "", filter_str
        )
        if files:
            self.add(list(files))
    
    def add_folder(self):
        """Add folder dialog"""
        self.log_debug("Opening folder dialog to import files")
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        
        paths = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(SUPPORTED):
                    paths.append(os.path.join(root, file))
        
        self.add(sorted(paths), import_root=folder)

    
    def add(self, paths, import_root=None):
        """Add files to list"""
        added = 0
        for path in paths:
            if path not in self.files:
                self.files.append(path)
                item = self.build_file_list_item(path)
                self.file_list.addItem(item)
                added += 1
        
        self.log_debug(f"Added {added} file(s) to project")
        if added:
            self.update_default_output_folder(paths=paths, import_root=import_root)
        
        self.update_file_info()

    def build_file_list_item(self, path):
        """Create a list item with a thumbnail preview and file info."""
        item = QListWidgetItem(os.path.basename(path))
        item.setToolTip(self.build_file_tooltip(path))
        item.setSizeHint(QSize(0, 60))

        thumbnail = self.get_thumbnail_icon(path)
        if thumbnail is not None:
            item.setIcon(thumbnail)

        return item

    def build_file_tooltip(self, path):
        """Build a tooltip describing the imported file."""
        info = self.get_image_info(path)
        if info is None:
            return os.path.basename(path)
        return f"{os.path.basename(path)}\n{info['width']}x{info['height']}"

    def get_thumbnail_icon(self, path):
        """Generate a high-quality thumbnail icon for the file list."""
        if path in self.thumbnail_cache:
            return self.thumbnail_cache[path]

        try:
            img = self.open_image(path, raise_on_error=True)
            thumbnail = img.copy()
            if hasattr(thumbnail, "draft"):
                try:
                    thumbnail.draft("RGBA", (48, 48))
                except Exception:
                    pass
            thumbnail.thumbnail((48, 48), Image.Resampling.BILINEAR)
            pixmap = QPixmap.fromImage(self.pil_to_qimage(thumbnail))
            icon = QIcon(pixmap)
            self.thumbnail_cache[path] = icon
            return icon
        except Exception:
            return None

    def get_image_info(self, path):
        """Return cached image dimensions and file size."""
        if path in self.image_info_cache:
            return self.image_info_cache[path]

        try:
            img = Image.open(path)
            info = {
                "width": img.width,
                "height": img.height,
                "size_kb": os.path.getsize(path) / 1024,
            }
            self.image_info_cache[path] = info
            return info
        except Exception:
            return None
    
    def clear_files(self):
        """Clear all files"""
        self.files.clear()
        self.file_list.clear()
        self.thumbnail_cache.clear()
        self.preview_source_cache.clear()
        self.image_info_cache.clear()
        self.current = None
        self.current_original = None
        self.output = None
        self.output_is_custom = False
        self.canvas.set_image(None)
        self.update_file_info()
    
    def delete_selected_files(self):
        """Delete selected files from the list"""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select files to delete")
            return
        
        # Get indices of selected items
        indices = sorted([self.file_list.row(item) for item in selected_items], reverse=True)
        
        # Remove files in reverse order to maintain correct indices
        for idx in indices:
            if 0 <= idx < len(self.files):
                filepath = self.files[idx]
                self.files.pop(idx)
                self.file_list.takeItem(idx)
                # Also remove any stored edits for this file
                self.image_edits.pop(filepath, None)
                self.thumbnail_cache.pop(filepath, None)
                self.preview_source_cache.pop(filepath, None)
                self.image_info_cache.pop(filepath, None)
        
        # Update info label
        deleted_count = len(selected_items)
        self.update_file_info()
        
        # If we deleted the current image, load the next one
        if self.index >= len(self.files) and self.files:
            self.index = len(self.files) - 1
            self.file_list.setCurrentRow(self.index)
            self.load_current()
        elif not self.files:
            self.current = None
            self.current_original = None
            self.canvas.set_image(None)
        
        self.log_status(f"Deleted {deleted_count} file(s)", "orange")
    
    # ========== IMAGE OPERATIONS ==========
    def open_image(self, path, raise_on_error=False):
        """Open image from any supported format"""
        try:
            if path.lower().endswith((".cr2", ".nef", ".arw", ".dng", ".raw")):
                with rawpy.imread(path) as raw:
                    return Image.fromarray(raw.postprocess())
            img = Image.open(path)
            img.load()
            return ImageOps.exif_transpose(img)
        except Exception as e:
            if raise_on_error:
                raise
            QMessageBox.critical(self, "Error", f"Failed to open {os.path.basename(path)}: {str(e)}")
            return None
    
    def load_current(self):
        """Load current image"""
        if self.index >= len(self.files):
            return
        
        current_path = self.files[self.index]
        self.log_debug(f"Loading current image: {current_path}")
        self.current_original = self.open_image(current_path)
        if self.current_original:
            if hasattr(self, "before_after_btn"):
                self.before_after_btn.blockSignals(True)
                self.before_after_btn.setChecked(False)
                self.before_after_btn.setText("Before")
                self.before_after_btn.blockSignals(False)
            self.show_original_preview = False
            self.current_preview = self.get_preview_source(current_path, mode=self.preview_resolution_mode)
            
            # Restore saved edits
            current_file = current_path
            if current_file in self.image_edits:
                edits = self.image_edits[current_file]
                for param_id, slider_widget in self.sliders.items():
                    slider_widget.set_value(edits.get(param_id, 50), animate=False)
                self.log_status("Edits restored", "green")
            else:
                self.reset_sliders()
            
            self.update_preview()
            QTimer.singleShot(0, self.canvas.fit_image)
            info = self.get_image_info(current_path)
            if info:
                self.img_info.setText(f"{os.path.basename(current_path)} | {info['width']}x{info['height']} | {info['size_kb']:.1f} KB")
    
    def schedule_preview_update(self):
        """Schedule preview update with debounce"""
        if not self.current:
            return
        
        # Build current edit parameters
        current_edits = self.get_current_edit_params()
        
        # Store edits for current image
        self.image_edits[self.files[self.index]] = current_edits
        
        # If sync_all enabled
        if self.sync_checkbox.isChecked():
            for filepath in self.files:
                self.image_edits[filepath] = current_edits.copy()
            self.log_status("📋 Edits synced to all images", "blue")
        
        # Schedule update
        if self.preview_job:
            self.killTimer(self.preview_job)
        self.preview_job = self.startTimer(80)
    
    def timerEvent(self, event):
        """Handle timer event for preview update"""
        if event.timerId() == self.preview_job:
            self.killTimer(self.preview_job)
            self.preview_job = None
            self.update_preview()

    def get_preview_source(self, path, high_res=False, mode=None):
        """Return a cached preview source for the viewer."""
        cache_entry = self.preview_source_cache.setdefault(path, {})
        mode = mode or ("High" if high_res else self.preview_resolution_mode)
        cache_key = mode.lower()
        if cache_key in cache_entry:
            return cache_entry[cache_key].copy()

        source = self.current_original.copy() if self.current_original else self.open_image(path, raise_on_error=True)
        max_preview_size = PREVIEW_RESOLUTION_PRESETS.get(mode, PREVIEW_RESOLUTION_PRESETS["Balanced"])
        if max(source.size) > max_preview_size:
            scale = max_preview_size / max(source.size)
            new_size = (int(source.size[0] * scale), int(source.size[1] * scale))
            source = source.resize(new_size, Image.Resampling.LANCZOS)

        cache_entry[cache_key] = source.copy()
        return source

    def update_preview(self):
        """Update canvas with current edits"""
        if not self.current_original:
            return
        
        # Choose source image
        source = self.current_preview.copy() if self.current_preview else self.get_preview_source(self.files[self.index], mode=self.preview_resolution_mode)
        
        if self.show_original_preview:
            self.current = source
        else:
            current_edits = self.image_edits.get(self.files[self.index], None)
            self.current = self.apply_edits(source, current_edits)
        
        # Convert to pixmap
        qimage = self.pil_to_qimage(self.current)
        pixmap = QPixmap.fromImage(qimage)
        # Use update_image to preserve zoom state instead of set_image
        self.canvas.update_image(pixmap)
    
    @staticmethod
    def pil_to_qimage(pil_img):
        """Convert PIL image to QImage"""
        if pil_img.mode == "RGB":
            data = pil_img.tobytes("raw", "RGB")
            qimage = QImage(data, pil_img.width, pil_img.height, pil_img.width * 3, QImage.Format_RGB888)
        elif pil_img.mode == "RGBA":
            data = pil_img.tobytes("raw", "RGBA")
            qimage = QImage(data, pil_img.width, pil_img.height, pil_img.width * 4, QImage.Format_RGBA8888)
        else:
            converted = pil_img.convert("RGBA") if pil_img.mode in ("LA", "P") else pil_img.convert("RGB")
            if converted.mode == "RGBA":
                data = converted.tobytes("raw", "RGBA")
                qimage = QImage(data, converted.width, converted.height, converted.width * 4, QImage.Format_RGBA8888)
            else:
                data = converted.tobytes("raw", "RGB")
                qimage = QImage(data, converted.width, converted.height, converted.width * 3, QImage.Format_RGB888)

        return qimage.copy()

    def apply_edits(self, img, params=None):
        """Apply editing adjustments with GPU acceleration when available"""
        if params is None:
            params = self.get_current_edit_params()
        
        if self.are_params_default(params):
            return img
        
        alpha_channel = None
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            alpha_channel = rgba.getchannel("A")
            img = rgba.convert("RGB")
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Use GPU acceleration if available and enabled
        if self.use_gpu and self.gpu_available:
            # Log GPU usage (only occasionally to avoid spam)
            if not hasattr(self, '_gpu_logged'):
                self.log_status("Using GPU acceleration", "green")
                self._gpu_logged = True
            try:
                edited = self.apply_edits_gpu(img, params)
            except Exception as exc:
                self.disable_gpu_acceleration(exc)
                edited = self.apply_edits_cpu(img, params)
        else:
            edited = self.apply_edits_cpu(img, params)

        if alpha_channel is not None:
            edited = edited.convert("RGBA")
            edited.putalpha(alpha_channel)

        return edited
    
    def apply_edits_gpu(self, img, params):
        """Apply edits using GPU acceleration"""
        # Convert PIL to numpy array
        img_array = np.array(img)
        
        def norm_val(val):
            return (val - 50) / 50
        
        # Apply exposure/brightness
        exposure = (params["exposure"] - 50) / 25
        brightness = (params["brightness"] - 50) / 25
        total_brightness = exposure * 0.3 + brightness * 0.2
        
        if total_brightness != 0:
            if CUPY_AVAILABLE:
                # Use CuPy for GPU acceleration
                img_gpu = cp.asarray(img_array)
                img_gpu = cp.clip(img_gpu * (1 + total_brightness), 0, 255).astype(cp.uint8)
                img_array = cp.asnumpy(img_gpu)
            else:
                # Use OpenCV or numpy
                img_array = self.apply_brightness_opencv(img_array, total_brightness)
        
        # Apply contrast
        contrast = (params["contrast"] - 50) / 25
        if contrast != 0:
            img_array = self.apply_contrast_opencv(img_array, contrast)
        
        # Apply saturation
        saturation = (params["saturation"] - 50) / 25
        if saturation != 0:
            img_array = self.apply_saturation_gpu(img_array, saturation)
        
        # Apply temperature and tint
        temperature = norm_val(params["temperature"])
        tint = norm_val(params["tint"])
        if temperature != 0 or tint != 0:
            img_array = self.apply_color_temperature_gpu(img_array, temperature, tint)
        
        # Apply tonal adjustments (highlights, shadows, whites, blacks)
        highlights_factor = norm_val(params["highlights"])
        shadows_factor = norm_val(params["shadows"])
        whites_factor = norm_val(params["whites"])
        blacks_factor = norm_val(params["blacks"])
        
        if any([highlights_factor, shadows_factor, whites_factor, blacks_factor]):
            img_array = self.apply_tonal_adjustments_gpu(img_array, highlights_factor, shadows_factor, whites_factor, blacks_factor)

        dehaze_factor = norm_val(params["dehaze"])
        if dehaze_factor != 0:
            img_array = self.apply_dehaze_gpu(img_array, dehaze_factor)
        
        # Apply vibrance
        vibrance_factor = norm_val(params["vibrance"])
        if vibrance_factor != 0:
            img_array = self.apply_vibrance_gpu(img_array, vibrance_factor)
        
        # Apply hue shift
        hue_factor = norm_val(params["hue"])
        if hue_factor != 0:
            img_array = self.apply_hue_shift_gpu(img_array, hue_factor)
        
        # Apply clarity
        clarity_factor = norm_val(params["clarity"])
        if clarity_factor != 0:
            img_array = self.apply_clarity_gpu(img_array, clarity_factor)
        
        # Apply sharpness
        sharpness = (params["sharpness"] - 50) / 25
        if sharpness != 0:
            img_array = self.apply_sharpness_gpu(img_array, sharpness)
        
        # Apply fade
        fade_factor = norm_val(params["fade"])
        if fade_factor != 0:
            img_array = self.apply_fade_gpu(img_array, fade_factor)

        gamma_factor = norm_val(params["gamma"])
        if gamma_factor != 0:
            img_array = self.apply_gamma_gpu(img_array, gamma_factor)

        vignette_factor = norm_val(params["vignette"])
        if vignette_factor != 0:
            img_array = self.apply_vignette_gpu(img_array, vignette_factor)
        
        # Convert back to PIL
        return Image.fromarray(img_array)
    
    def apply_edits_cpu(self, img, params=None):
        """Apply edits using CPU (original PIL-based implementation)"""
        if params is None:
            params = self.get_current_edit_params()
        
        if self.are_params_default(params):
            return img
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        def norm_val(val):
            return (val - 50) / 50
        
        # Apply exposure
        exposure = (params["exposure"] - 50) / 25
        if exposure != 0:
            img = ImageEnhance.Brightness(img).enhance(1 + exposure * 0.3)
        
        # Apply contrast
        contrast = (params["contrast"] - 50) / 25
        if contrast != 0:
            img = ImageEnhance.Contrast(img).enhance(1 + contrast * 0.4)
        
        # Apply highlights and shadows
        highlights_factor = norm_val(params["highlights"])
        shadows_factor = norm_val(params["shadows"])
        whites_factor = norm_val(params["whites"])
        blacks_factor = norm_val(params["blacks"])
        
        if any([highlights_factor, shadows_factor, whites_factor, blacks_factor]):
            img = self.apply_tonal_adjustments(img, highlights_factor, shadows_factor, whites_factor, blacks_factor)

        dehaze_factor = norm_val(params["dehaze"])
        if dehaze_factor != 0:
            img = self.apply_dehaze(img, dehaze_factor)
        
        # Apply temperature
        temperature = norm_val(params["temperature"])
        if temperature != 0:
            img = self.apply_temperature(img, temperature)
        
        # Apply tint
        tint = norm_val(params["tint"])
        if tint != 0:
            img = self.apply_tint(img, tint)
        
        # Apply saturation
        saturation = (params["saturation"] - 50) / 25
        if saturation != 0:
            img = ImageEnhance.Color(img).enhance(1 + saturation * 0.5)
        
        # Apply vibrance
        vibrance_factor = norm_val(params["vibrance"])
        if vibrance_factor != 0:
            img = ImageEnhance.Color(img).enhance(1 + vibrance_factor * 0.3)
        
        # Apply hue
        hue_factor = norm_val(params["hue"])
        if hue_factor != 0:
            img = self.apply_hue_shift(img, hue_factor)
        
        # Apply clarity
        clarity_factor = norm_val(params["clarity"])
        if clarity_factor != 0:
            img = self.apply_clarity(img, clarity_factor)
        
        # Apply sharpness
        sharpness = (params["sharpness"] - 50) / 25
        if sharpness != 0:
            img = ImageEnhance.Sharpness(img).enhance(1 + sharpness * 0.3)
        
        # Apply brightness
        brightness = (params["brightness"] - 50) / 25
        if brightness != 0:
            img = ImageEnhance.Brightness(img).enhance(1 + brightness * 0.2)
        
        # Apply fade
        fade_factor = norm_val(params["fade"])
        if fade_factor != 0:
            img = self.apply_fade(img, fade_factor)

        gamma_factor = norm_val(params["gamma"])
        if gamma_factor != 0:
            img = self.apply_gamma(img, gamma_factor)

        vignette_factor = norm_val(params["vignette"])
        if vignette_factor != 0:
            img = self.apply_vignette(img, vignette_factor)
        
        return img
    
    def apply_tonal_adjustments(self, img, highlights, shadows, whites, blacks):
        """Apply highlights, shadows, whites, and blacks adjustments"""
        pixels = img.load()
        width, height = img.size
        
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y][:3]
                brightness = (r + g + b) / 3 / 255

                highlight_weight = max(0.0, (brightness - 0.5) / 0.5)
                shadow_weight = max(0.0, (0.5 - brightness) / 0.5)
                white_weight = max(0.0, (brightness - 0.75) / 0.25)
                black_weight = max(0.0, (0.25 - brightness) / 0.25)

                adjust = int(
                    highlights * 18 * highlight_weight +
                    shadows * 18 * shadow_weight +
                    whites * 24 * white_weight +
                    blacks * 24 * black_weight
                )

                r = max(0, min(255, r + adjust))
                g = max(0, min(255, g + adjust))
                b = max(0, min(255, b + adjust))
                
                pixels[x, y] = (r, g, b)
        
        return img
    
    def apply_temperature(self, img, factor):
        """Apply color temperature adjustment"""
        pixels = img.load()
        width, height = img.size
        
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y][:3]
                
                if factor > 0:
                    r = int(min(255, r * (1 + factor * 0.2)))
                    b = int(max(0, b * (1 - factor * 0.15)))
                else:
                    r = int(max(0, r * (1 + factor * 0.15)))
                    b = int(min(255, b * (1 - factor * 0.2)))
                
                pixels[x, y] = (r, g, b)
        
        return img
    
    def apply_tint(self, img, factor):
        """Apply color tint"""
        pixels = img.load()
        width, height = img.size
        
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y][:3]
                
                if factor > 0:
                    g = int(min(255, g * (1 + factor * 0.25)))
                else:
                    r = int(min(255, r * (1 - factor * 0.2)))
                    b = int(min(255, b * (1 - factor * 0.2)))
                
                pixels[x, y] = (r, g, b)
        
        return img
    
    def apply_vibrance(self, img, factor):
        """Enhance color saturation"""
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(1 + factor * 0.3)
    
    def apply_hue_shift(self, img, factor):
        """Apply hue shift"""
        pixels = img.load()
        width, height = img.size
        
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y][:3]
                
                if factor > 0:
                    r, g, b = g, b, r
                else:
                    r, g, b = b, r, g
                
                pixels[x, y] = (r, g, b)
        
        return img
    
    def apply_clarity(self, img, factor):
        """Enhance local contrast and clarity"""
        from PIL import ImageFilter
        
        if factor > 0:
            img_detail = img.filter(ImageFilter.DETAIL)
            enhancer = ImageEnhance.Brightness(img_detail)
            img = enhancer.enhance(1 + factor * 0.3)
        
        return img
    
    def apply_fade(self, img, factor):
        """Apply fade effect"""
        if factor > 0:
            white = Image.new('RGB', img.size, (255, 255, 255))
            img = Image.blend(img, white, factor * 0.3)
        
        return img

    def apply_gamma(self, img, factor):
        """Adjust image gamma for midtone control."""
        img_array = np.array(img)
        return Image.fromarray(self.apply_gamma_array(img_array, factor))

    def apply_dehaze(self, img, factor):
        """Apply a lightweight dehaze effect."""
        img_array = np.array(img)
        return Image.fromarray(self.apply_dehaze_array(img_array, factor))

    def apply_vignette(self, img, factor):
        """Darken or lift the edges to shape attention."""
        img_array = np.array(img)
        return Image.fromarray(self.apply_vignette_array(img_array, factor))

    def apply_gamma_array(self, img_array, factor, xp=np):
        """Apply gamma correction on an RGB array."""
        img_float = xp.asarray(img_array).astype(xp.float32) / 255.0
        gamma_value = max(0.35, min(2.2, 1.0 - factor * 0.85))
        adjusted = xp.power(xp.clip(img_float, 0.0, 1.0), gamma_value)
        return xp.clip(adjusted * 255.0, 0, 255).astype(xp.uint8)

    def apply_dehaze_array(self, img_array, factor, xp=np):
        """Apply a fast haze reduction approximation on an RGB array."""
        img_float = xp.asarray(img_array).astype(xp.float32) / 255.0
        if factor >= 0:
            dark_channel = xp.min(img_float, axis=2, keepdims=True)
            strength = factor * 0.45
            adjusted = (img_float - dark_channel * strength) / xp.clip(1.0 - dark_channel * strength, 0.35, 1.0)
        else:
            haze = abs(factor) * 0.28
            adjusted = img_float + (1.0 - img_float) * haze
        return xp.clip(adjusted * 255.0, 0, 255).astype(xp.uint8)

    def apply_vignette_array(self, img_array, factor, xp=np):
        """Apply a radial vignette on an RGB array."""
        img_float = xp.asarray(img_array).astype(xp.float32) / 255.0
        height, width = img_float.shape[:2]
        if height <= 0 or width <= 0:
            return img_array

        yy = xp.linspace(-1.0, 1.0, height, dtype=xp.float32).reshape(height, 1)
        xx = xp.linspace(-1.0, 1.0, width, dtype=xp.float32).reshape(1, width)
        distance = xp.sqrt(xx * xx + yy * yy) / 1.41421356
        distance = xp.clip(distance, 0.0, 1.0)
        edge_weight = xp.power(distance, 1.6)

        if factor >= 0:
            mask = 1.0 - edge_weight * factor * 0.55
            adjusted = img_float * mask[:, :, xp.newaxis]
        else:
            lift = edge_weight * abs(factor) * 0.25
            adjusted = img_float + (1.0 - img_float) * lift[:, :, xp.newaxis]

        return xp.clip(adjusted * 255.0, 0, 255).astype(xp.uint8)
    
    def apply_brightness_opencv(self, img_array, factor):
        """Apply brightness adjustment using OpenCV"""
        if OPENCV_AVAILABLE:
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            # Apply brightness
            img_bgr = cv2.convertScaleAbs(img_bgr, alpha=(1 + factor), beta=0)
            # Convert back to RGB
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            # Fallback to numpy
            return np.clip(img_array * (1 + factor), 0, 255).astype(np.uint8)
    
    def apply_contrast_opencv(self, img_array, factor):
        """Apply contrast adjustment using OpenCV"""
        if OPENCV_AVAILABLE:
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            # Apply contrast (mean-based)
            mean_val = cv2.mean(img_bgr)[:3]  # Get mean of BGR channels
            img_bgr = cv2.convertScaleAbs(img_bgr, alpha=(1 + factor * 0.4), beta=0)
            # Convert back to RGB
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            # Fallback to numpy
            mean_val = np.mean(img_array, axis=(0, 1), keepdims=True)
            return np.clip(mean_val + (img_array - mean_val) * (1 + factor * 0.4), 0, 255).astype(np.uint8)
    
    def apply_saturation_gpu(self, img_array, saturation):
        """Apply saturation adjustment using GPU"""
        # Convert to HSV for saturation adjustment
        if CUPY_AVAILABLE:
            img_gpu = cp.asarray(img_array).astype(cp.float32) / 255.0
            
            # Simple saturation adjustment (approximation)
            # Convert RGB to approximate saturation adjustment
            max_rgb = cp.max(img_gpu, axis=2, keepdims=True)
            min_rgb = cp.min(img_gpu, axis=2, keepdims=True)
            saturation_factor = (max_rgb - min_rgb) / (max_rgb + 1e-6)
            
            # Adjust saturation
            gray = cp.mean(img_gpu, axis=2, keepdims=True)
            img_gpu = gray + (img_gpu - gray) * (1 + saturation * 0.5)
            img_gpu = cp.clip(img_gpu, 0, 1)
            
            img_array = (img_gpu * 255).astype(cp.uint8)
            return cp.asnumpy(img_array)
        else:
            # Fallback to numpy
            img_float = img_array.astype(np.float32) / 255.0
            max_rgb = np.max(img_float, axis=2, keepdims=True)
            min_rgb = np.min(img_float, axis=2, keepdims=True)
            saturation_factor = (max_rgb - min_rgb) / (max_rgb + 1e-6)
            
            gray = np.mean(img_float, axis=2, keepdims=True)
            img_float = gray + (img_float - gray) * (1 + saturation * 0.5)
            img_float = np.clip(img_float, 0, 1)
            
            return (img_float * 255).astype(np.uint8)
    
    def apply_color_temperature_gpu(self, img_array, temperature, tint):
        """Apply color temperature and tint adjustments using GPU"""
        if CUPY_AVAILABLE:
            img_gpu = cp.asarray(img_array).astype(cp.float32)
            
            # Temperature adjustment
            if temperature > 0:
                img_gpu[:, :, 0] = cp.clip(img_gpu[:, :, 0] * (1 + temperature * 0.2), 0, 255)  # Red
                img_gpu[:, :, 2] = cp.clip(img_gpu[:, :, 2] * (1 - temperature * 0.15), 0, 255)  # Blue
            elif temperature < 0:
                img_gpu[:, :, 0] = cp.clip(img_gpu[:, :, 0] * (1 + temperature * 0.15), 0, 255)  # Red
                img_gpu[:, :, 2] = cp.clip(img_gpu[:, :, 2] * (1 - temperature * 0.2), 0, 255)  # Blue
            
            # Tint adjustment
            if tint > 0:
                img_gpu[:, :, 1] = cp.clip(img_gpu[:, :, 1] * (1 + tint * 0.25), 0, 255)  # Green
            elif tint < 0:
                img_gpu[:, :, 0] = cp.clip(img_gpu[:, :, 0] * (1 - tint * 0.2), 0, 255)  # Red
                img_gpu[:, :, 2] = cp.clip(img_gpu[:, :, 2] * (1 - tint * 0.2), 0, 255)  # Blue
            
            return cp.asnumpy(img_gpu.astype(cp.uint8))
        else:
            # Fallback to numpy
            if temperature > 0:
                img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (1 + temperature * 0.2), 0, 255).astype(np.uint8)  # Red
                img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (1 - temperature * 0.15), 0, 255).astype(np.uint8)  # Blue
            elif temperature < 0:
                img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (1 + temperature * 0.15), 0, 255).astype(np.uint8)  # Red
                img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (1 - temperature * 0.2), 0, 255).astype(np.uint8)  # Blue
            
            if tint > 0:
                img_array[:, :, 1] = np.clip(img_array[:, :, 1] * (1 + tint * 0.25), 0, 255).astype(np.uint8)  # Green
            elif tint < 0:
                img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (1 - tint * 0.2), 0, 255).astype(np.uint8)  # Red
                img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (1 - tint * 0.2), 0, 255).astype(np.uint8)  # Blue
            
            return img_array
    
    def apply_tonal_adjustments_gpu(self, img_array, highlights, shadows, whites, blacks):
        """Apply tonal adjustments using GPU"""
        if CUPY_AVAILABLE:
            img_gpu = cp.asarray(img_array).astype(cp.float32)
            
            # Calculate brightness for each pixel
            brightness = cp.mean(img_gpu, axis=2) / 255.0
            
            # Apply adjustments
            highlight_weight = cp.clip((brightness - 0.5) / 0.5, 0.0, 1.0)
            shadow_weight = cp.clip((0.5 - brightness) / 0.5, 0.0, 1.0)
            white_weight = cp.clip((brightness - 0.75) / 0.25, 0.0, 1.0)
            black_weight = cp.clip((0.25 - brightness) / 0.25, 0.0, 1.0)

            scalar_adjustment = (
                highlights * 18 * highlight_weight +
                shadows * 18 * shadow_weight +
                whites * 24 * white_weight +
                blacks * 24 * black_weight
            )
            adjustment = cp.repeat(scalar_adjustment[:, :, cp.newaxis], 3, axis=2)
            
            img_gpu = cp.clip(img_gpu + adjustment, 0, 255)
            
            return cp.asnumpy(img_gpu.astype(cp.uint8))
        else:
            # Fallback to numpy
            brightness = np.mean(img_array, axis=2) / 255.0

            highlight_weight = np.clip((brightness - 0.5) / 0.5, 0.0, 1.0)
            shadow_weight = np.clip((0.5 - brightness) / 0.5, 0.0, 1.0)
            white_weight = np.clip((brightness - 0.75) / 0.25, 0.0, 1.0)
            black_weight = np.clip((0.25 - brightness) / 0.25, 0.0, 1.0)

            scalar_adjustment = (
                highlights * 18 * highlight_weight +
                shadows * 18 * shadow_weight +
                whites * 24 * white_weight +
                blacks * 24 * black_weight
            )
            adjustment = np.repeat(scalar_adjustment[:, :, np.newaxis], 3, axis=2)
            
            return np.clip(img_array + adjustment, 0, 255).astype(np.uint8)
    
    def apply_vibrance_gpu(self, img_array, factor):
        """Apply vibrance adjustment using GPU"""
        return self.apply_saturation_gpu(img_array, factor * 0.3)
    
    def apply_hue_shift_gpu(self, img_array, factor):
        """Apply hue shift using GPU (simplified)"""
        if CUPY_AVAILABLE:
            img_gpu = cp.asarray(img_array)
            
            if factor > 0:
                # Rotate RGB channels: R->G->B->R
                img_gpu = cp.stack([img_gpu[:, :, 1], img_gpu[:, :, 2], img_gpu[:, :, 0]], axis=2)
            elif factor < 0:
                # Rotate RGB channels: R->B->G->R
                img_gpu = cp.stack([img_gpu[:, :, 2], img_gpu[:, :, 0], img_gpu[:, :, 1]], axis=2)
            
            return cp.asnumpy(img_gpu)
        else:
            if factor > 0:
                return np.stack([img_array[:, :, 1], img_array[:, :, 2], img_array[:, :, 0]], axis=2)
            elif factor < 0:
                return np.stack([img_array[:, :, 2], img_array[:, :, 0], img_array[:, :, 1]], axis=2)
            else:
                return img_array
    
    def apply_clarity_gpu(self, img_array, factor):
        """Apply clarity enhancement using GPU"""
        if factor <= 0:
            return img_array
            
        if CUPY_AVAILABLE:
            img_gpu = cp.asarray(img_array).astype(cp.float32)
            
            # Simple unsharp mask for clarity
            kernel = cp.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]], dtype=cp.float32) / 9.0
            
            # Apply convolution for each channel
            sharpened = cp.zeros_like(img_gpu)
            for c in range(3):
                sharpened[:, :, c] = cp.clip(cp.convolve2d(img_gpu[:, :, c], kernel, mode='same', boundary='wrap'), 0, 255)
            
            # Blend original with sharpened
            img_gpu = img_gpu + (sharpened - img_gpu) * (factor * 0.5)
            img_gpu = cp.clip(img_gpu, 0, 255)
            
            return cp.asnumpy(img_gpu.astype(cp.uint8))
        else:
            # Fallback to simple contrast enhancement
            return np.clip(img_array * (1 + factor * 0.1), 0, 255).astype(np.uint8)
    
    def apply_sharpness_gpu(self, img_array, sharpness):
        """Apply sharpness using GPU"""
        if sharpness <= 0:
            return img_array
            
        if CUPY_AVAILABLE:
            img_gpu = cp.asarray(img_array).astype(cp.float32)
            
            # Simple sharpening kernel
            kernel = cp.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]], dtype=cp.float32)
            
            # Apply convolution for each channel
            sharpened = cp.zeros_like(img_gpu)
            for c in range(3):
                sharpened[:, :, c] = cp.convolve2d(img_gpu[:, :, c], kernel, mode='same', boundary='wrap')
            
            # Blend with original
            img_gpu = img_gpu + (sharpened - img_gpu) * sharpness
            img_gpu = cp.clip(img_gpu, 0, 255)
            
            return cp.asnumpy(img_gpu.astype(cp.uint8))
        else:
            # Fallback to simple operation
            return np.clip(img_array * (1 + sharpness * 0.1), 0, 255).astype(np.uint8)
    
    def apply_fade_gpu(self, img_array, factor):
        """Apply fade effect using GPU"""
        if factor <= 0:
            return img_array
            
        if CUPY_AVAILABLE:
            img_gpu = cp.asarray(img_array).astype(cp.float32)
            white = cp.full_like(img_gpu, 255.0)
            
            # Blend with white
            img_gpu = img_gpu + (white - img_gpu) * (factor * 0.3)
            img_gpu = cp.clip(img_gpu, 0, 255)
            
            return cp.asnumpy(img_gpu.astype(cp.uint8))
        else:
            white = np.full_like(img_array, 255, dtype=np.float32)
            return np.clip(img_array + (white - img_array) * (factor * 0.3), 0, 255).astype(np.uint8)

    def apply_gamma_gpu(self, img_array, factor):
        """Apply gamma adjustment using GPU when available."""
        if CUPY_AVAILABLE:
            return cp.asnumpy(self.apply_gamma_array(img_array, factor, xp=cp))
        return self.apply_gamma_array(img_array, factor)

    def apply_dehaze_gpu(self, img_array, factor):
        """Apply dehaze using GPU when available."""
        if CUPY_AVAILABLE:
            return cp.asnumpy(self.apply_dehaze_array(img_array, factor, xp=cp))
        return self.apply_dehaze_array(img_array, factor)

    def apply_vignette_gpu(self, img_array, factor):
        """Apply vignette using GPU when available."""
        if CUPY_AVAILABLE:
            return cp.asnumpy(self.apply_vignette_array(img_array, factor, xp=cp))
        return self.apply_vignette_array(img_array, factor)
    
    # ========== NAVIGATION ==========
    def prev_image(self):
        """Previous image"""
        if self.index > 0:
            self.index -= 1
            self.file_list.setCurrentRow(self.index)
            self.load_current()
    
    def next_image(self):
        """Next image"""
        if self.index < len(self.files) - 1:
            self.index += 1
            self.file_list.setCurrentRow(self.index)
            self.load_current()
    
    # ========== SLIDERS & EXPORT ==========
    def reset_sliders(self):
        """Reset all sliders to default"""
        for slider_widget in self.sliders.values():
            slider_widget.set_value(50, animate=True)
        self.schedule_preview_update()

    def auto_adjust_current_image(self):
        """Estimate a neutral correction for the current image."""
        if not self.current_original:
            self.log_status("Load an image before using Auto", "orange")
            return

        img = self.current_original.convert("RGB")
        img_array = np.array(img, dtype=np.float32)
        luminance = 0.2126 * img_array[:, :, 0] + 0.7152 * img_array[:, :, 1] + 0.0722 * img_array[:, :, 2]

        mean_luma = float(np.mean(luminance))
        std_luma = float(np.std(luminance))
        low_clip = float(np.percentile(luminance, 5))
        high_clip = float(np.percentile(luminance, 95))
        sat_strength = float(np.mean(np.std(img_array, axis=2)))

        def clamp_slider(value):
            return max(0, min(100, int(round(value))))

        params = {param_id: 50 for param_id in DEFAULT_ADJUSTMENT_IDS}

        exposure_push = (118.0 - mean_luma) / 4.8
        contrast_push = (58.0 - std_luma) / 2.8
        highlight_pull = (high_clip - 228.0) / 2.0
        shadow_lift = (52.0 - low_clip) / 1.8

        params["exposure"] = clamp_slider(50 + exposure_push)
        params["brightness"] = clamp_slider(50 + exposure_push * 0.65)
        params["contrast"] = clamp_slider(50 + contrast_push)
        params["highlights"] = clamp_slider(50 - max(0.0, highlight_pull))
        params["shadows"] = clamp_slider(50 + max(0.0, shadow_lift))
        params["whites"] = clamp_slider(50 - max(0.0, (high_clip - 240.0) / 2.4) + max(0.0, (190.0 - high_clip) / 4.0))
        params["blacks"] = clamp_slider(50 + max(0.0, (28.0 - low_clip) / 1.6) - max(0.0, (low_clip - 45.0) / 3.5))
        params["clarity"] = clamp_slider(50 + max(0.0, (48.0 - std_luma) / 4.0))
        params["dehaze"] = clamp_slider(50 + max(0.0, (42.0 - std_luma) / 5.0))
        params["gamma"] = clamp_slider(50 + (126.0 - mean_luma) / 7.0)
        params["vibrance"] = clamp_slider(50 + max(0.0, (42.0 - sat_strength) / 1.8))
        params["saturation"] = clamp_slider(50 + max(0.0, (36.0 - sat_strength) / 2.5))

        self.set_edit_params(params)
        self.schedule_preview_update()
        self.log_status("Auto adjustments applied", "green")

    def log_status(self, msg, color="black"):
        """Update status log"""
        color_map = {
            "black": "#333333", 
            "green": "#1B5E20", 
            "blue": "#0D47A1", 
            "red": "#B71C1C", 
            "orange": "#E65100"
        }
        self.status_log.setText(msg)
        text_color = color_map.get(color, "#333333")
        if self.get_resolved_theme() == "Light":
            background = "#f0f4fa"
            border = "#d6e0ef"
        else:
            background = "#101720"
            border = "#293241"
        self.status_log.setStyleSheet(f"color: {text_color}; font-weight: bold; padding: 6px 10px; background-color: {background}; border-radius: 8px; border: 1px solid {border};")
        self.animate_status_log()
    
    def are_params_default(self, params):
        """Check if parameters are all default"""
        if not params:
            return True
        return all(params.get(k, 50) == 50 for k in DEFAULT_ADJUSTMENT_IDS)
    
    def get_current_edit_params(self):
        """Get current parameters from sliders"""
        return {param_id: slider_widget.get_value() for param_id, slider_widget in self.sliders.items()}
    
    def set_edit_params(self, params):
        """Set slider values from parameters"""
        for param_id, slider_widget in self.sliders.items():
            slider_widget.set_value(params.get(param_id, 50), animate=False)
    
    # ========== PRESETS ==========
    def ensure_preset_folder(self):
        """Ensure preset folder exists"""
        self.preset_folder.mkdir(parents=True, exist_ok=True)
    
    def load_presets(self):
        """Load presets from folder"""
        self.ensure_preset_folder()
        self.presets = {}
        for preset_file in sorted(self.preset_folder.glob("*.icv")):
            try:
                with open(preset_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.presets[preset_file.stem] = data
            except Exception as e:
                print(f"Failed to load preset {preset_file}: {e}")
        self.update_preset_combobox()
    
    def update_preset_combobox(self):
        """Update preset combobox"""
        names = sorted(self.presets.keys())
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItems(names)
        self.preset_combo.blockSignals(False)
    
    def apply_preset(self):
        """Apply selected preset"""
        name = self.preset_combo.currentText()
        if name and name in self.presets:
            self.set_edit_params(self.presets[name])
            self.schedule_preview_update()
            self.log_status(f"Loaded preset: {name}", "green")
    
    def save_preset(self):
        """Save current settings as preset"""
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not name.strip():
            return
        
        self.ensure_preset_folder()
        saved_name = name.strip()
        file_path = self.preset_folder / f"{saved_name}.icv"
        params = self.get_current_edit_params()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)
        self.presets[saved_name] = params
        self.update_preset_combobox()
        self.preset_combo.setCurrentText(saved_name)
        self.log_status(f"Saved preset: {saved_name}", "green")
    
    def delete_preset(self):
        """Delete selected preset"""
        name = self.preset_combo.currentText()
        if not name or name not in self.presets:
            return
        
        file_path = self.preset_folder / f"{name}.icv"
        if file_path.exists():
            file_path.unlink()
        self.presets.pop(name, None)
        self.update_preset_combobox()
        self.log_status(f"Deleted preset: {name}", "orange")
    
    # ========== PROCESSING ==========
    def pick_output_folder(self):
        """Pick output folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output = folder
            self.output_is_custom = True
            os.makedirs(self.output, exist_ok=True)
            QMessageBox.information(self, "Output Folder", f"Output set to:\n{folder}")
    
    def get_target_size_mb(self):
        """Get target size in MB"""
        size_map = {"No Limit": 0, "100 KB": 0.1, "500 KB": 0.5, "1 MB": 1, "5 MB": 5}
        if self.target_size_combo.currentText() in size_map:
            return size_map[self.target_size_combo.currentText()]
        elif self.target_size_combo.currentText() == "Custom":
            return self.custom_size_spinbox.value()
        return 0
    
    def process_selected(self):
        """Process selected files"""
        items = self.file_list.selectedItems()
        if not items:
            QMessageBox.warning(self, "Warning", "Please select one or more images to process")
            return
        file_list = [self.files[self.file_list.row(item)] for item in items]
        self.process_files(file_list)
    
    def process_all(self):
        """Process all files"""
        if not self.files:
            QMessageBox.warning(self, "Warning", "No files to process")
            return
        if not self.output_is_custom:
            self.update_default_output_folder(paths=self.files)
        elif self.output:
            os.makedirs(self.output, exist_ok=True)
        self.process_files(self.files)
    
    def process_files(self, file_list):
        """Process file list for all export formats"""
        if self.processing:
            self.log_status("Processing is already running", "orange")
            return
        self.export_formats = self.get_selected_export_formats()
        if not self.export_formats:
            QMessageBox.warning(self, "Warning", "Select at least one export format")
            return
        self.log_debug(f"Starting batch process for {len(file_list)} file(s) in {len(self.export_formats)} format(s)")
        if not self.output_is_custom:
            self.update_default_output_folder(paths=file_list)
        elif self.output:
            os.makedirs(self.output, exist_ok=True)
        
        self.processing = True
        self.progress.setVisible(True)
        self.stop_btn.setVisible(True)
        self.stop_btn.setEnabled(True)
        # Total progress = files × formats
        total_operations = len(file_list) * len(self.export_formats)
        self.progress.setMaximum(total_operations)
        self.progress.setValue(0)
        self.log_status(f"Preparing {len(file_list)} image(s) for export...", "blue")
        
        options = {
            'quality': self.quality_slider.value(),
            'target_size_mb': self.get_target_size_mb()
        }

        self.worker = ProcessWorker(self, file_list, self.export_formats, options=options)
        self.worker.progress_updated.connect(lambda v: self.progress.setValue(v))
        self.worker.status_updated.connect(lambda msg, color: self.log_status(msg, color))
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.start()

    def stop_processing(self):
        """Request the current export worker to stop."""
        if not self.processing or not hasattr(self, "worker") or self.worker is None:
            self.log_status("No export is running", "orange")
            return

        self.worker.request_stop()
        self.stop_btn.setEnabled(False)
        self.log_status("Stopping after current export step...", "orange")
    
    def on_processing_finished(self):
        """Handle processing finished"""
        self.processing = False
        self.progress.setVisible(False)
        self.stop_btn.setVisible(False)
        self.stop_btn.setEnabled(True)
        if hasattr(self, "worker") and getattr(self.worker, "was_cancelled", False):
            self.log_status("Processing stopped", "orange")
        else:
            self.log_status("Processing finished", "green")
        self.worker = None
    
    def get_format_capabilities(self, fmt):
        """Get format capabilities for size optimization."""
        fmt_lower = fmt.lower()
        
        # Lossy formats that support quality parameters
        lossy_formats = {'jpg', 'jpeg', 'webp', 'heic'}
        
        # Lossless formats that don't support quality
        lossless_formats = {'png', 'gif', 'bmp', 'tiff', 'ico', 'icns'}
        
        is_lossy = fmt_lower in lossy_formats
        is_lossless = fmt_lower in lossless_formats
        
        caps = {
            'format': fmt_lower,
            'output_extension': fmt_lower,
            'is_lossy': is_lossy,
            'is_lossless': is_lossless,
            'supports_quality': is_lossy,
            'pil_format': 'JPEG' if fmt_lower in ('jpg', 'jpeg') else 
                          'WEBP' if fmt_lower == 'webp' else fmt.upper(),
            'save_kwargs': {},
            'output_suffix': ''
        }

        if fmt_lower == 'tiff-lzw':
            caps['pil_format'] = 'TIFF'
            caps['output_extension'] = 'tiff'
            caps['is_lossless'] = True
            caps['save_kwargs'] = {'compression': 'tiff_lzw'}
            caps['output_suffix'] = '_lzw'

        return caps

    def prepare_image_for_format(self, img, fmt):
        """Normalize image mode for formats with strict mode requirements."""
        fmt_lower = fmt.lower()

        # JPEG cannot store alpha; flatten transparent pixels to white.
        if fmt_lower in ('jpg', 'jpeg'):
            has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
            if has_alpha:
                rgba = img.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                img = Image.alpha_composite(background, rgba).convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")

        if fmt_lower in ('ico', 'icns'):
            target_size = self.get_icon_output_size(img.size, fmt_lower)
            rgba = img.convert("RGBA")
            contained = ImageOps.contain(rgba, (target_size, target_size), Image.Resampling.LANCZOS)
            if contained.size != (target_size, target_size):
                icon_canvas = Image.new("RGBA", (target_size, target_size), (255, 255, 255, 0))
                offset = (
                    (target_size - contained.width) // 2,
                    (target_size - contained.height) // 2,
                )
                icon_canvas.paste(contained, offset, contained)
                img = icon_canvas
            else:
                img = contained

        return img

    def get_icon_output_size(self, size, fmt_lower):
        """Choose the best standard icon size supported by the destination format."""
        max_dim = max(size)
        supported_sizes = [16, 24, 32, 48, 64, 128, 256] if fmt_lower == 'ico' else [16, 32, 64, 128, 256, 512, 1024]
        return max(s for s in supported_sizes if s <= max(16, max_dim))

    def extract_metadata(self, img):
        """Collect metadata that can be preserved during export."""
        info = dict(getattr(img, "info", {}) or {})
        metadata = {}

        dpi = info.get("dpi")
        if isinstance(dpi, tuple) and len(dpi) == 2:
            metadata["dpi"] = dpi

        for key in ("icc_profile", "xmp"):
            value = info.get(key)
            if value:
                metadata[key] = value

        exif = info.get("exif")
        if not exif:
            try:
                exif_data = img.getexif()
                if exif_data:
                    exif = exif_data.tobytes()
            except Exception:
                exif = None
        if exif:
            metadata["exif"] = exif

        pnginfo = self.build_pnginfo(info)
        if pnginfo is not None:
            metadata["pnginfo"] = pnginfo

        return metadata

    def extract_metadata_from_path(self, file_path):
        """Collect metadata from the original source file before any transforms."""
        try:
            if file_path.lower().endswith((".cr2", ".nef", ".arw", ".dng", ".raw")):
                img = self.open_image(file_path, raise_on_error=True)
                return self.extract_metadata(img)

            img = Image.open(file_path)
            metadata = {}
            if img.info.get("dpi"):
                metadata["dpi"] = img.info["dpi"]
            for key in ("icc_profile", "xmp"):
                value = img.info.get(key)
                if value:
                    metadata[key] = value
            if img.info.get("exif"):
                metadata["exif"] = img.info["exif"]
            img.load()
            extracted = self.extract_metadata(img)
            metadata.update(extracted)
            return metadata
        except Exception:
            return {}

    def build_pnginfo(self, info):
        """Preserve simple PNG text metadata when available."""
        reserved_keys = {
            "dpi", "gamma", "transparency", "aspect", "duration", "loop", "timestamp",
            "icc_profile", "exif", "xmp", "interlace", "compression"
        }
        pnginfo = PngImagePlugin.PngInfo()
        has_text = False

        for key, value in info.items():
            if key in reserved_keys:
                continue
            if isinstance(value, str):
                pnginfo.add_text(key, value)
                has_text = True

        return pnginfo if has_text else None

    def build_save_kwargs(self, img, caps, metadata=None, quality=None, optimize=False):
        """Build format-aware save options, including metadata when supported."""
        fmt_lower = caps['format']
        metadata = metadata or {}
        save_kwargs = dict(caps.get('save_kwargs', {}))

        if optimize and fmt_lower not in ('ico', 'icns'):
            save_kwargs['optimize'] = True
        if quality is not None and caps['supports_quality']:
            save_kwargs['quality'] = quality

        if fmt_lower == 'ico':
            icon_size = self.get_icon_output_size(img.size, fmt_lower)
            save_kwargs['sizes'] = [(icon_size, icon_size)]

        if fmt_lower in ('jpg', 'jpeg', 'png', 'webp', 'bmp', 'gif', 'tiff', 'tiff-lzw'):
            if metadata.get('dpi'):
                save_kwargs['dpi'] = metadata['dpi']
        if fmt_lower in ('jpg', 'jpeg', 'png', 'webp', 'tiff', 'tiff-lzw'):
            if metadata.get('icc_profile'):
                save_kwargs['icc_profile'] = metadata['icc_profile']
            if metadata.get('exif'):
                save_kwargs['exif'] = metadata['exif']
        if fmt_lower == 'png' and metadata.get('pnginfo'):
            save_kwargs['pnginfo'] = metadata['pnginfo']
        if fmt_lower == 'webp':
            if metadata.get('xmp'):
                save_kwargs['xmp'] = metadata['xmp']
            if img.mode == 'RGBA':
                save_kwargs['exact'] = True

        return save_kwargs

    def save_image(self, img, output_path, caps, metadata=None, quality=None, optimize=False):
        """Save an image with format-aware options and metadata preservation."""
        save_kwargs = self.build_save_kwargs(
            img,
            caps,
            metadata=metadata,
            quality=quality,
            optimize=optimize,
        )
        img.save(output_path, format=caps['pil_format'], **save_kwargs)

    def encode_image(self, img, caps, metadata=None, quality=None, optimize=False):
        """Encode an image to bytes with the same save settings used for files."""
        buffer = io.BytesIO()
        save_kwargs = self.build_save_kwargs(
            img,
            caps,
            metadata=metadata,
            quality=quality,
            optimize=optimize,
        )
        img.save(buffer, format=caps['pil_format'], **save_kwargs)
        return buffer.getvalue()

    def build_output_path(self, name, caps):
        """Build a unique output path for the selected export format."""
        suffix = caps.get('output_suffix', '')
        return os.path.join(self.output, f"{name}{suffix}.{caps['output_extension']}")
    
    def optimize_lossless_format(self, img, fmt, target_size_mb, output_path, metadata=None):
        """Optimize lossless format by downscaling to reach target size.
        
        Strategy: Iteratively reduce image dimensions until target size is reached.
        """
        caps = self.get_format_capabilities(fmt)
        
        # First, try to save at full resolution
        encoded = self.encode_image(img, caps, metadata=metadata, optimize=True)
        current_size = len(encoded) / (1024 * 1024)
        
        self.log_debug(f"Lossless format {fmt}: full resolution size {current_size:.2f}MB (target: {target_size_mb}MB)")
        
        # If already within target, save and return
        if current_size <= target_size_mb:
            with open(output_path, 'wb') as f:
                f.write(encoded)
            self.log_debug(f"✓ Saved {fmt} at full resolution, size within target")
            return
        
        # Otherwise, apply downscaling to reduce file size
        self.log_debug(f"Full resolution exceeds target. Starting downscale optimization...")
        
        original_width, original_height = img.size
        min_scale = max(4 / max(original_width, 1), 4 / max(original_height, 1), 0.01)
        scale_factor = (target_size_mb / current_size) ** 0.5
        scale_factor = max(min_scale, min(0.95, scale_factor))

        measured = {}

        def measure_scale(scale):
            scale = max(min_scale, min(1.0, scale))
            scale_key = round(scale, 5)
            if scale_key in measured:
                return measured[scale_key]

            new_width = max(4, int(original_width * scale))
            new_height = max(4, int(original_height * scale))
            scaled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            encoded_bytes = self.encode_image(scaled_img, caps, metadata=metadata, optimize=True)
            size_mb = len(encoded_bytes) / (1024 * 1024)
            measured[scale_key] = (scaled_img, encoded_bytes, size_mb)
            self.log_debug(f"  Scale {scale:.2f} ({new_width}x{new_height}): {size_mb:.2f}MB")
            return measured[scale_key]

        best_img = img
        best_size = current_size
        best_scale = 1.0
        best_encoded = encoded

        too_large_scale = 1.0
        working_scale = None

        for scale in sorted(set([0.95, 0.75, 0.5, 0.33, scale_factor, scale_factor * 0.85, scale_factor * 0.7, min_scale]), reverse=True):
            scaled_img, encoded_bytes, size_mb = measure_scale(scale)
            if size_mb <= target_size_mb:
                working_scale = scale
                best_img = scaled_img
                best_size = size_mb
                best_scale = scale
                best_encoded = encoded_bytes
                break
            too_large_scale = min(too_large_scale, scale)

        if working_scale is None:
            scaled_img, encoded_bytes, size_mb = measure_scale(min_scale)
            best_img = scaled_img
            best_size = size_mb
            best_scale = min_scale
            best_encoded = encoded_bytes
        else:
            low = working_scale
            high = min(1.0, max(working_scale, too_large_scale if too_large_scale > working_scale else 1.0))

            for _ in range(6):
                if high - low < 0.01:
                    break
                test_scale = (low + high) / 2
                scaled_img, encoded_bytes, size_mb = measure_scale(test_scale)
                if size_mb <= target_size_mb:
                    low = test_scale
                    best_img = scaled_img
                    best_size = size_mb
                    best_scale = test_scale
                    best_encoded = encoded_bytes
                else:
                    high = test_scale
        
        # Save the best result
        with open(output_path, 'wb') as f:
            f.write(best_encoded)
        
        # Log result
        if best_size <= target_size_mb:
            self.log_debug(f"✓ Optimized to {best_size:.2f}MB at scale {best_scale:.2f} ({int(original_width*best_scale)}x{int(original_height*best_scale)})")
        else:
            self.log_debug(f"⚠ Best achieved: {best_size:.2f}MB at scale {best_scale:.2f} (target was {target_size_mb}MB)")
            self.log_debug(f"  Recommendation: Use WebP or JPEG format for better compression")
    
    def prepare_image_for_export(self, file_path, raise_on_error=False):
        """Load the original image once, preserve source metadata, and apply edits once."""
        img = self.open_image(file_path, raise_on_error=raise_on_error)
        if not img:
            return None

        metadata = self.extract_metadata_from_path(file_path)
        params = self.image_edits.get(file_path)
        edited_img = self.apply_edits(img, params)

        return {
            "name": os.path.splitext(os.path.basename(file_path))[0],
            "image": edited_img,
            "metadata": metadata,
        }

    def save_processed_image(self, img, name, metadata, format_name=None, options=None):
        """Save an already-adjusted image to the selected export format."""
        options = options or {}

        if format_name:
            normalized_fmt = format_name.lower()
        else:
            selected_formats = self.get_selected_export_formats()
            selected_name = selected_formats[0] if selected_formats else "JPG"
            normalized_fmt = selected_name.lower()

        caps = self.get_format_capabilities(normalized_fmt)
        img = self.prepare_image_for_format(img, normalized_fmt)
        quality = options.get('quality', self.quality_slider.value())
        target_size_mb = options.get('target_size_mb', self.get_target_size_mb())
        output_path = self.build_output_path(name, caps)

        if target_size_mb > 0:
            self.log_debug(f"Target size: {target_size_mb}MB, format: {normalized_fmt}")

            if caps['is_lossy']:
                optimal_quality = self.find_optimal_quality(img, normalized_fmt, target_size_mb, quality, metadata=metadata)
                self.save_image(img, output_path, caps, metadata=metadata, quality=optimal_quality, optimize=True)
                final_size = os.path.getsize(output_path) / (1024 * 1024)
                self.log_debug(f"Saved with quality {optimal_quality}, final size: {final_size:.2f}MB")
            elif caps['is_lossless']:
                self.optimize_lossless_format(img, normalized_fmt, target_size_mb, output_path, metadata=metadata)
            else:
                self.save_image(img, output_path, caps, metadata=metadata)
                final_size = os.path.getsize(output_path) / (1024 * 1024)
                self.log_debug(f"Saved {normalized_fmt} as-is, final size: {final_size:.2f}MB")
        else:
            if caps['supports_quality']:
                self.save_image(img, output_path, caps, metadata=metadata, quality=quality, optimize=True)
                self.log_debug(f"Saved with quality {quality}, no size limit")
            else:
                self.save_image(img, output_path, caps, metadata=metadata)
                self.log_debug(f"Saved {normalized_fmt} (lossless, no quality setting)")

    def process_image(self, file_path, format_name=None, options=None, raise_on_error=False):
        """Process a single image with format-aware size optimization."""
        self.log_debug(f"Processing image: {file_path}")
        prepared = self.prepare_image_for_export(file_path, raise_on_error=raise_on_error)
        if not prepared:
            return
        self.save_processed_image(
            prepared["image"],
            prepared["name"],
            prepared["metadata"],
            format_name=format_name,
            options=options,
        )
    
    def find_optimal_quality(self, img, fmt, target_size_mb, max_quality=95, metadata=None):
        """Find optimal quality using a faster size-target heuristic."""
        self.log_debug(f"Starting optimized quality search (target: {target_size_mb}MB, format: {fmt})")

        # Define quality ranges based on format
        if fmt.lower() == 'webp':
            min_quality = 1
            max_quality = min(max_quality, 100)
        elif fmt.lower() == 'jpeg' or fmt.lower() == 'jpg':
            min_quality = 1
            max_quality = min(max_quality, 100)
        else:
            min_quality = 10
            max_quality = min(max_quality, 95)
        caps = self.get_format_capabilities(fmt)
        measured_sizes = {}

        best_quality = max_quality
        best_diff = float("inf")
        best_under_quality = None
        best_under_gap = float("inf")

        def measure_size(q):
            q = int(max(min_quality, min(max_quality, q)))
            if q not in measured_sizes:
                encoded = self.encode_image(img, caps, metadata=metadata, quality=q, optimize=True)
                measured_sizes[q] = len(encoded) / (1024 * 1024)
            return measured_sizes[q]

        def consider_quality(q, size_mb):
            nonlocal best_quality, best_diff, best_under_quality, best_under_gap

            diff = abs(size_mb - target_size_mb)
            if diff < best_diff or (diff == best_diff and size_mb <= target_size_mb):
                best_quality = q
                best_diff = diff

            if size_mb <= target_size_mb:
                under_gap = target_size_mb - size_mb
                if under_gap < best_under_gap or (under_gap == best_under_gap and (best_under_quality is None or q > best_under_quality)):
                    best_under_quality = q
                    best_under_gap = under_gap

        tolerance = max(0.005, target_size_mb * 0.01)

        # Try the current maximum quality first
        current_size = measure_size(max_quality)
        current_diff = abs(current_size - target_size_mb)
        self.log_debug(f"Initial size at quality {max_quality}: {current_size:.2f}MB, diff {current_diff:.2f}MB")
        consider_quality(max_quality, current_size)
        if current_size <= target_size_mb or current_diff <= tolerance:
            return max_quality

        # Use a smart starting guess near the middle of the valid quality range
        start_quality = min(max_quality, 80)
        start_size = measure_size(start_quality)
        start_diff = abs(start_size - target_size_mb)
        self.log_debug(f"Start guess quality {start_quality}: size {start_size:.2f}MB, diff {start_diff:.2f}MB")
        consider_quality(start_quality, start_size)

        if start_diff <= tolerance:
            return best_quality

        # Estimate a better quality using size ratio
        if start_size > 0:
            estimated_quality = int(start_quality * target_size_mb / start_size)
        else:
            estimated_quality = start_quality

        estimated_quality = max(min_quality, min(max_quality, estimated_quality))
        if estimated_quality == start_quality:
            estimated_quality = max(min_quality, min(max_quality, start_quality - 10 if start_size > target_size_mb else start_quality + 10))

        est_size = measure_size(estimated_quality)
        est_diff = abs(est_size - target_size_mb)
        self.log_debug(f"Estimate quality {estimated_quality}: size {est_size:.2f}MB, diff {est_diff:.2f}MB")
        consider_quality(estimated_quality, est_size)

        if best_diff <= tolerance:
            return best_quality

        # Narrow search range around the best values
        low = max(min_quality, min(start_quality, estimated_quality) - 12)
        high = min(max_quality, max(start_quality, estimated_quality) + 12)
        left, right = low, high

        iterations = 0
        max_iterations = 5

        while left <= right and iterations < max_iterations:
            iterations += 1
            mid = (left + right) // 2
            mid_size = measure_size(mid)
            mid_diff = abs(mid_size - target_size_mb)
            self.log_debug(f"Refine {iterations}: quality {mid}, size {mid_size:.2f}MB, diff {mid_diff:.2f}MB")
            consider_quality(mid, mid_size)

            if best_diff <= tolerance:
                break

            if mid_size <= target_size_mb:
                left = mid + 1
            else:
                right = mid - 1

        final_quality = best_under_quality if best_under_quality is not None else best_quality

        # Walk quality upward from the best-under result to get as close as possible without exceeding target.
        if best_under_quality is not None:
            probe_quality = best_under_quality + 1
            while probe_quality <= max_quality:
                probe_size = measure_size(probe_quality)
                if probe_size <= target_size_mb:
                    consider_quality(probe_quality, probe_size)
                    final_quality = probe_quality
                    probe_quality += 1
                    continue
                break

        final_size = measure_size(final_quality)
        while final_size > target_size_mb and final_quality > min_quality:
            final_quality -= 1
            final_size = measure_size(final_quality)

        self.log_debug(f"Final result: quality {final_quality}, size {final_size:.2f}MB (target: {target_size_mb}MB)")
        return final_quality
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Handled via keyPressEvent
        pass
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        key = event.key()
        modifiers = event.modifiers()
        
        # Navigation
        if key == Qt.Key_Left or key == Qt.Key_A:
            self.prev_image()
        elif key == Qt.Key_Right or key == Qt.Key_D:
            self.next_image()
        elif key == Qt.Key_Home:
            self.index = 0
            self.file_list.setCurrentRow(self.index)
            self.load_current()
        elif key == Qt.Key_End:
            self.index = len(self.files) - 1
            self.file_list.setCurrentRow(self.index)
            self.load_current()
        
        # Zoom controls
        elif key == Qt.Key_Plus or key == Qt.Key_Equal or (key == Qt.Key_Z and modifiers & Qt.ControlModifier):
            self.canvas.zoom_in()
        elif key == Qt.Key_Minus or (key == Qt.Key_Z and modifiers & Qt.ControlModifier and modifiers & Qt.ShiftModifier):
            self.canvas.zoom_out()
        elif key == Qt.Key_F or key == Qt.Key_0:
            self.canvas.fit_image()
        elif key == Qt.Key_1:
            self.canvas.zoom = 1.0
            self.canvas.pan_x = 0
            self.canvas.pan_y = 0
            self.canvas.display_image()
        
        # Edit controls
        elif key == Qt.Key_R or (key == Qt.Key_R and modifiers & Qt.ControlModifier):
            self.reset_sliders()
        elif key == Qt.Key_S and modifiers & Qt.ControlModifier:
            self.save_preset()
        elif key == Qt.Key_O and modifiers & Qt.ControlModifier:
            self.add_files()
        elif key == Qt.Key_Delete:
            self.delete_selected_files()
        
        # Processing
        elif key == Qt.Key_E and modifiers & Qt.ControlModifier:
            self.process_selected()
        elif key == Qt.Key_E and modifiers & Qt.ControlModifier and modifiers & Qt.ShiftModifier:
            self.process_all()
        
        # Selection controls
        elif key == Qt.Key_A and modifiers & Qt.ControlModifier:
            if modifiers & Qt.ShiftModifier:
                # Ctrl+Shift+A: Deselect all
                self.file_list.clearSelection()
                self.log_status("All files deselected", "blue")
            else:
                # Ctrl+A: Select all
                self.file_list.selectAll()
                self.log_status(f"All {len(self.files)} files selected", "blue")
        
        # View controls
        elif key == Qt.Key_F11:
            if self.isMaximized():
                self.showNormal()
            else:
                self.showMaximized()
        elif key == Qt.Key_F5:
            self.update_preview()
        elif key == Qt.Key_F1:
            self.show_shortcuts_help()
        
        else:
            super().keyPressEvent(event)
    
    def show_shortcuts_help(self):
        """Show keyboard shortcuts help dialog"""
        shortcuts_text = """
        <h3>Keyboard Shortcuts</h3>
        
        <h4>Navigation</h4>
        <b>Left / Right or A / D:</b> Previous / Next image<br>
        <b>Home / End:</b> First / Last image<br>
        
        <h4>Zoom & View</h4>
        <b>+ / = or Ctrl+Z:</b> Zoom in<br>
        <b>- or Ctrl+Shift+Z:</b> Zoom out<br>
        <b>F or 0:</b> Fit to screen<br>
        <b>1:</b> Actual size (100%)<br>
        <b>F11:</b> Toggle fullscreen<br>
        
        <h4>Mouse Controls</h4>
        <b>Mouse Wheel:</b> Zoom in / out<br>
        <b>Click & Drag:</b> Pan when the preview is larger than the viewer<br>
        
        <h4>Editing</h4>
        <b>R or Ctrl+R:</b> Reset all adjustments<br>
        <b>Ctrl+S:</b> Save preset<br>
        <b>Ctrl+O:</b> Open files<br>
        <b>Delete:</b> Delete selected files<br>
        
        <h4>Processing</h4>
        <b>Ctrl+E:</b> Export selected images<br>
        <b>Ctrl+Shift+E:</b> Export all images<br>
        <b>F5:</b> Refresh preview<br>
        
        <h4>Selection</h4>
        <b>Ctrl+A:</b> Select all files<br>
        <b>Ctrl+Shift+A:</b> Deselect all files<br>
        
        <h4>Help & Tools</h4>
        <b>F1:</b> Show this help dialog<br>
        <b>Windows menu:</b> Preview resolution, sound effects, and UI effect toggles<br>
        <b>WaterMarkPro menu:</b> Launch the companion WaterMarkPro app<br>
        """
        
        msg = QMessageBox(self)
        msg.setWindowTitle("Keyboard Shortcuts")
        msg.setTextFormat(Qt.RichText)
        msg.setText(shortcuts_text)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()


if __name__ == "__main__":
    app = QApplication([])
    window = ImageConverterProPySide()
    window.show()
    app.exec()
